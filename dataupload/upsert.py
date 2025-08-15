import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
import glob
from datetime import datetime
from dataupload.bm25_manager import BM25Manager
from dataupload.bm25_config import BM25Config
from dataupload.vector_builder import build_vector_data, build_vector_via_db, batch_upsert_vectors

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataupload.data_vector_v3 import SemanticDocumentChunker, save_chunks_to_text, save_chunks_to_json

from dataupload.knowledge_graph_manager import KnowledgeGraphManager
from storage.storage_factory import StorageFactory
from conn.conn_factory import ConnectionFactory
from embedding.em_factory import EmbeddingFactory

from config import OPENSEARCH_INDEX


# 导入config中的日志配置函数
from config import setup_logging

class UpsertManager:
    """
    数据入库流程管理器 (Refactored)
    
    职责：
    1. 调用指定版本的切片器进行文档切片。
    2. 调用 knowledge_graph_manager 将实体和关系写入Neo4j。
    3. 调用 embedding_model 进行向量化。
    4. 将向量化后的数据上传到 Pinecone。
    5. 统一调度、配置和日志记录。
    """
    def __init__(self,embedding_model: str = "qwen", environment: str = "production", chunker_version: str = "v3", log_file_path: str = None):
        """
        初始化入库管理器
        
        Args:
            environment: 运行环境 ("production", "test", "development")
            chunker_version: 切片器版本 ("v2", "v3")
            log_file_path: 日志文件路径，如果为None则不保存到文件
        """
        self.environment = environment
        self.chunker_version = chunker_version
        self.log_file_path = log_file_path
        self.chunker = None
        self.kg_manager = None
        self.pinecone_index = None
        self.embedding_model = EmbeddingFactory.create_embedding_model(embedding_model)
        self.db_conn = None
        self.bm25_manager = None  # 新增：BM25Manager实例
        self.bm25_config = BM25Config()  # 新增：BM25Config实例

        # 配置日志
        if log_file_path:
            setup_logging(log_file_path)
        
        logging.info(f"🚀 UpsertManager 初始化，环境: {self.environment}, 切片器版本: {self.chunker_version}")

    def _initialize_conn_and_kg(self, enable_kg: bool,db_type:str,faction:str):
        '''
        初始化chunker的实例,知识图谱和数据库连接
        Args:
            enable_kg: 是否启用知识图谱写入
            db_type: 数据库类型
        '''
        if self.chunker is None:
            if self.chunker_version == 'v3':
                self.chunker = SemanticDocumentChunker(faction_name=faction)
                logging.info("✅ SemanticDocumentChunker (v3) 初始化完成")
            else:
                raise ValueError(f"不支持的切片器版本: {self.chunker_version}")
        if enable_kg and self.kg_manager is None:
            self.kg_manager = KnowledgeGraphManager(environment=self.environment)
            logging.info("✅ KnowledgeGraphManager 初始化完成")
        try:
            self.db_conn = ConnectionFactory.create_conn(connection_type=db_type)
        except Exception as e:
            logging.error(f"❌ 初始化数据库连接失败: {e}")
            raise e
            

    def _get_latest_userdict_path(self):
        """获取dict文件夹中最新的用户词典文件"""
        dict_dir = "dict"
        if not os.path.exists(dict_dir):
            return None
        
        # 查找所有用户词典文件（.txt格式）
        dict_pattern = os.path.join(dict_dir, "*.txt")
        dict_files = glob.glob(dict_pattern)
        
        if not dict_files:
            return None
        
        # 按文件名中的时间戳排序，取最新的
        dict_files.sort(reverse=True)
        return dict_files[0]
    
    def _get_latest_vocab_path(self):
        """获取dict文件夹中最新的BM25词汇表文件"""
        dict_dir = "dict"
        if not os.path.exists(dict_dir):
            return None
        
        # 查找所有bm25_vocab_*.json文件
        vocab_pattern = os.path.join(dict_dir, "bm25_vocab_*.json")
        vocab_files = glob.glob(vocab_pattern)
        
        if not vocab_files:
            return None
        
        # 按文件名中的时间戳排序，取最新的
        vocab_files.sort(reverse=True)
        return vocab_files[0]

    def _initialize_bm25_manager(self, userdict_path: str):
        """初始化BM25Manager，支持模型加载/保存"""
        if self.bm25_manager is None:
            # 使用BM25Config的参数实例化BM25Manager
            self.bm25_manager = BM25Manager(
                k1=self.bm25_config.k1,
                b=self.bm25_config.b,
                min_freq=self.bm25_config.min_freq,
                max_vocab_size=self.bm25_config.max_vocab_size,
                user_dict_path=userdict_path
            )
            logging.info("✅ BM25Manager 初始化完成")
        
        # 获取最新的BM25词汇表文件路径
        vocab_path = self.bm25_manager.get_latest_freq_dict_file()
        
        # 使用BM25Config生成模型路径
        model_path = self.bm25_config.get_model_path()
        
        # 检查模型文件是否存在
        if os.path.exists(model_path) and vocab_path and os.path.exists(vocab_path):
            # 模型文件存在，尝试加载
            try:
                self.bm25_manager.load_model(model_path, vocab_path)
                logging.info(f"✅ BM25模型已从文件加载: {model_path}, {vocab_path}")
            except Exception as e:
                logging.warning(f"⚠️ 加载BM25模型失败，将重新训练: {e}")
                self.bm25_manager.is_fitted = False
        else:
            logging.info("📝 BM25模型文件不存在，将进行训练")
            # 确保状态为未训练
            self.bm25_manager.is_fitted = False

    def _save_bm25_model(self):
        """保存BM25模型到文件"""
        if self.bm25_manager and self.bm25_manager.is_fitted:
            # 使用BM25Config生成模型路径
            model_path = self.bm25_config.get_model_path()
            # 确保models目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 获取最新的BM25词汇表路径，如果没有则生成新的
            vocab_path = self._get_latest_vocab_path()
            if not vocab_path:
                # 如果没有现有词汇表，生成新的
                vocab_path = self.bm25_config.get_vocab_path()
            
            # 确保dict目录存在
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            
            try:
                self.bm25_manager.save_model(model_path, vocab_path)
                logging.info(f"✅ BM25模型已保存到文件: {model_path}, {vocab_path}")
            except Exception as e:
                logging.warning(f"⚠️ 保存BM25模型失败: {e}")

    
    def process_and_upsert_document(self, file_path: str, enable_kg: bool = True, extra_metadata: dict = None, storage_type: str = 'local',db_type: str = 'opensearch',faction:str = None):
        """
        处理单个文档并将其入库
        
        Args:
            file_path: 文档文件路径
            enable_kg: 是否启用知识图谱写入
            extra_metadata: 额外的元数据（如faction），优先级最高
        """
        if not os.path.exists(file_path):
            logging.error(f"❌ 文件不存在: {file_path}")
            return

        logging.info(f"📄 开始处理文档: {file_path}")
        self._initialize_conn_and_kg(enable_kg, db_type,faction)

        try:
            # 1. 读取和切片文档
            logging.info("Step 1: 正在进行文档切片...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 组装extra_metadata
            base_metadata = {'source_file': os.path.basename(file_path)}
            if extra_metadata:
                base_metadata.update(extra_metadata)

            chunks = self.chunker.chunk_text(content, base_metadata)
            logging.info(f"✅ 文档切片完成，生成 {len(chunks)} 个切片。")
            
            # 保存切片到本地文件，用于观察切片效果
            try:
                save_chunks_to_json(chunks, "dataupload/chunks")
                logging.info("✅ 切片已保存到 dataupload/chunks 目录")
            except Exception as e:
                logging.warning(f"⚠️ 保存切片文件时出现警告: {e}")

            # === 修改：只使用现有词典，不重新生成 ===
            # 获取dict文件夹下最新的用户词典文件
            userdict_path = self._get_latest_userdict_path()
            if userdict_path:
                logging.info(f"✅ 使用现有词典: {userdict_path}")
            else:
                logging.warning("⚠️ 未找到词典文件，将使用默认分词")
                userdict_path = None

            # === 修改：初始化BM25Manager并支持模型加载/保存 ===
            self._initialize_bm25_manager(userdict_path)

            # 生成storage对象的目的是为了支持BM25模型的增量更新（incremental_update），
            # 这样可以将历史切片持久化存储，避免每次都全量重建模型。
            # 当前实现为本地存储，后续如需支持云端存储（如S3、MongoDB等），只需扩展StorageFactory即可无缝切换。
            storage = StorageFactory.create_storage(storage_type)

            # 如果模型未训练，则进行训练
            if not self.bm25_manager.is_fitted:
                self.bm25_manager.fit([chunk['text'] for chunk in chunks])
                logging.info("✅ BM25Manager已训练完成")
                self._save_bm25_model()
            else:
                # 模型已存在，进行增量更新
                self.bm25_manager.incremental_update([chunk['text'] for chunk in chunks], storage=storage)
                logging.info("✅ BM25Manager已进行增量更新")
                self._save_bm25_model()

            # 2. 知识图谱写入
            if enable_kg and self.kg_manager:
                logging.info("Step 2: 正在写入知识图谱...")
                total_entities = 0
                total_relationships = 0
                for i, chunk in enumerate(chunks):
                    entities = self.kg_manager.extract_entities(chunk['text'], chunk.get('section_heading'), chunk.get('hierarchy'))
                    relationships = self.kg_manager.extract_relationships(chunk['text'], entities, chunk.get('content_type'))
                    
                    if entities or relationships:
                        self.kg_manager.update_knowledge_graph(entities, relationships, chunk)
                        total_entities += len(entities)
                        total_relationships += len(relationships)
                        logging.debug(f"  - Chunk {i+1}: 写入 {len(entities)} 实体, {len(relationships)} 关系")
                
                logging.info(f"✅ 知识图谱写入完成，共计写入 {total_entities} 个实体, {total_relationships} 个关系。")

            # 3. 向量上传
            if self.db_conn and self.embedding_model:
                logging.info("Step 3: 正在进行向量化并上传到 Pinecone...")
                
                # 构建待上传的数据和上传记录
                vectors_to_upsert = []
                upload_records = []
                chunk_id_mapping = {}
                
                for i, chunk in enumerate(chunks):
                    # 使用faction作为ID前缀，格式：faction_number
                    faction = base_metadata.get('faction', 'unknown')
                    chunk_id = f"{faction}_{i}"
                    chunk_id_mapping[i] = chunk_id
                    try:
                        embedding = self.embedding_model.get_embedding(chunk['text'])
                        logging.info(f"[embedding] {chunk_id}的embedding向量生成完毕.")
                        # 用build_vector_via_db根据数据库类型构造vector
                        vector_data = build_vector_via_db(
                            chunk=chunk,
                            embedding=embedding,
                            bm25_manager=self.bm25_manager,  # 使用实例变量
                            chunk_id=chunk_id,
                            base_metadata=base_metadata,
                            db_str=db_type  # 传递数据库类型
                        )
                        logging.info("[vector_Data] vector_data创建完毕")
                        vectors_to_upsert.append(vector_data)
                        # 先标记为pending，等批量上传完成后再更新状态
                        # 从vector_data中提取需要的元数据字段
                        chunk_metadata = {
                            'chunk_id': chunk_id,
                            'chunk_type': vector_data.get('chunk_type', ''),
                            'content_type': vector_data.get('content_type', ''),
                            'faction': vector_data.get('faction', ''),
                            'section_heading': vector_data.get('section_heading', ''),
                            'source_file': vector_data.get('source_file', ''),
                            'chunk_index': vector_data.get('chunk_index', 0)
                        }
                        upload_records.append({
                            'chunk_id': chunk_id,
                            'status': 'pending',
                            'metadata': chunk_metadata
                        })
                        
                        # 记录稀疏向量生成情况
                        if 'sparse' in vector_data and vector_data['sparse']:
                            logging.info(f"[SPARSE_STATS] {chunk_id} - 稀疏向量生成成功")
                        else:
                            logging.warning(f"[SPARSE_STATS] {chunk_id} - 稀疏向量生成失败或为空")
                    except Exception as e:
                        logging.error(f"[ERROR] Chunk {i} embedding/upload failed: {e}")
                        upload_records.append({
                            'chunk_id': chunk_id,
                            'status': 'fail',
                            'error': str(e)
                        })

                # 执行批量上传
                try:
                    batch_results = batch_upsert_vectors(self.db_conn, vectors_to_upsert, batch_size=100, index_name=OPENSEARCH_INDEX)
                    logging.info(f"[DEBUG] batch_results: {batch_results}")
                    logging.info(f"[DEBUG] batch_results类型: {type(batch_results)}")
                    logging.info(f"[DEBUG] batch_results长度: {len(batch_results)}")
                    
                    # 根据批次结果更新每个chunk的状态
                    batch_size = 100
                    for batch_idx, batch_success in enumerate(batch_results):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(upload_records))
                        
                        logging.info(f"[DEBUG] 批次 {batch_idx}: batch_success={batch_success}, start_idx={start_idx}, end_idx={end_idx}")
                        
                        if batch_success:
                            # 批次成功，更新所有pending的记录为success
                            for record_idx in range(start_idx, end_idx):
                                if upload_records[record_idx]['status'] == 'pending':
                                    upload_records[record_idx]['status'] = 'success'
                            logging.info(f"[DEBUG] 批次 {batch_idx} 成功: 更新了 {end_idx - start_idx} 个记录为success")
                        else:
                            # 批次失败，更新所有pending的记录为fail
                            for record_idx in range(start_idx, end_idx):
                                if upload_records[record_idx]['status'] == 'pending':
                                    upload_records[record_idx]['status'] = 'fail'
                                    upload_records[record_idx]['error'] = f"批次{batch_idx+1}上传失败"
                            logging.info(f"[DEBUG] 批次 {batch_idx} 失败: 更新了 {end_idx - start_idx} 个记录为fail")
                    
                    logging.info("✅ 批量上传完成，状态已更新")
                except Exception as e:
                    logging.error(f"❌ 批量上传失败: {e}")
                    # 批量上传失败，更新所有pending的记录为fail
                    for record in upload_records:
                        if record['status'] == 'pending':
                            record['status'] = 'fail'
                            record['error'] = f"批量上传失败: {str(e)}"
                
                
                upload_log_file = f"dataupload/upload_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs(os.path.dirname(upload_log_file), exist_ok=True)
                
                # 统计成功和失败的数量
                successful_uploads = sum(1 for record in upload_records if record['status'] == 'success')
                failed_uploads = sum(1 for record in upload_records if record['status'] == 'fail')
                # 统计稀疏向量生成情况
                sparse_vectors_count = sum(1 for vector in vectors_to_upsert if 'sparse' in vector and vector['sparse'])
                dense_only_count = len(vectors_to_upsert) - sparse_vectors_count
                
                logging.info(f"✅ Pinecone 向量上传完成:")
                logging.info(f"  - 总切片数: {len(chunks)}")
                logging.info(f"  - 成功上传: {successful_uploads}")
                logging.info(f"  - 上传失败: {failed_uploads}")
                logging.info(f"  - 包含稀疏向量: {sparse_vectors_count}")
                logging.info(f"  - 仅稠密向量: {dense_only_count}")
                logging.info(f"  - 上传记录已保存到: {upload_log_file}")
                
                # 如果稀疏向量生成率过低，给出警告
                if sparse_vectors_count == 0:
                    logging.error(f"❌ 警告: 所有 {len(vectors_to_upsert)} 个向量都没有生成稀疏向量！")
                elif sparse_vectors_count < len(vectors_to_upsert) * 0.5:
                    logging.warning(f"⚠️ 稀疏向量生成率较低: {sparse_vectors_count}/{len(vectors_to_upsert)} ({sparse_vectors_count/len(vectors_to_upsert)*100:.1f}%)")
                
                # 检查特定切片是否上传成功
                if failed_uploads > 0:
                    logging.warning("⚠️ 部分切片上传失败，请检查上传记录文件")
                    for record in upload_records:
                        if record['status'] == 'fail':
                            logging.warning(f"  - 切片 {record['chunk_id']} 上传失败: {record.get('error', '未知错误')}")

            logging.info(f"🎉 文档 {file_path} 处理完成！")

        except Exception as e:
            logging.error(f"❌ 处理文档 {file_path} 时发生错误: {e}", exc_info=True)
        
        finally:
            if self.kg_manager:
                self.kg_manager.close()
                logging.info("Neo4j 连接已关闭")

def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="文档入库流程管理器")
    parser.add_argument("file_path", type=str, help="要处理的文档路径")
    parser.add_argument("--env", type=str, default="production", choices=["production", "test", "development"], help="运行环境")
    parser.add_argument("-cv", "--chunker-version", type=str, default="v3", choices=["v2", "v3"], help="选择使用的切片器版本 (v2/v3)")
    parser.add_argument("--no-kg", action="store_true", help="禁用知识图谱写入")
    parser.add_argument("--db", type=str, default="opensearch", choices=["pinecone", "opensearch"], help="选择使用的数据库")
    parser.add_argument("--faction", type=str, help="指定faction（可选，优先级最高）")
    parser.add_argument("--storage", type=str, default="local", choices=["local", "cloud"], help="存储类型")
    parser.add_argument("--embedding_model", type=str, default="qwen", choices=["qwen", "openai"], help="选择使用的嵌入模型")
    args = parser.parse_args()
    
    # 生成日志文件路径
    input_filename = os.path.splitext(os.path.basename(args.file_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{input_filename}_{timestamp}.log"
    log_file_path = os.path.join("dataupload", "log", log_filename)

    
    manager = UpsertManager(embedding_model=args.embedding_model,environment=args.env, chunker_version=args.chunker_version, log_file_path=log_file_path)
    # 组装extra_metadata，优先使用命令行参数
    extra_metadata = {'source_file': os.path.basename(args.file_path)}
    if args.faction:
        extra_metadata['faction'] = args.faction
    manager.process_and_upsert_document(
        file_path=args.file_path,
        enable_kg=not args.no_kg,
        extra_metadata=extra_metadata,
        storage_type=args.storage,
        db_type=args.db,
        faction=args.faction
        )


if __name__ == "__main__":
    main()
