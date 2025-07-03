import os
import sys
import argparse
import logging
from typing import List, Dict, Any
import subprocess
from DATAUPLOD.bm25_manager import BM25Manager

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_vector_v3 import SemanticDocumentChunker, save_chunks_to_text, save_chunks_to_json
from data_vector_v2 import DocumentChunker
from knowledge_graph_manager import KnowledgeGraphManager
from config import get_pinecone_index, get_embedding_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, environment: str = "production", chunker_version: str = "v3"):
        """
        初始化入库管理器
        
        Args:
            environment: 运行环境 ("production", "test", "development")
            chunker_version: 切片器版本 ("v2", "v3")
        """
        self.environment = environment
        self.chunker_version = chunker_version
        self.chunker = None
        self.kg_manager = None
        self.pinecone_index = None
        self.embedding_model = None

        logging.info(f"🚀 UpsertManager 初始化，环境: {self.environment}, 切片器版本: {self.chunker_version}")

    def _initialize_services(self, enable_kg: bool, enable_pinecone: bool):
        """延迟初始化所需服务"""
        if self.chunker is None:
            if self.chunker_version == 'v3':
                self.chunker = SemanticDocumentChunker()
                logging.info("✅ SemanticDocumentChunker (v3) 初始化完成")
            elif self.chunker_version == 'v2':
                self.chunker = DocumentChunker()
                logging.info("✅ DocumentChunker (v2) 初始化完成")
            else:
                raise ValueError(f"不支持的切片器版本: {self.chunker_version}")
            
        if enable_kg and self.kg_manager is None:
            self.kg_manager = KnowledgeGraphManager(environment=self.environment)
            logging.info("✅ KnowledgeGraphManager 初始化完成")
            
        if enable_pinecone and self.pinecone_index is None:
            self.pinecone_index = get_pinecone_index()
            self.embedding_model = get_embedding_model()
            logging.info("✅ Pinecone Index 和 Embedding Model 初始化完成")

    def process_and_upsert_document(self, file_path: str, enable_kg: bool = True, enable_pinecone: bool = True, extra_metadata: dict = None):
        """
        处理单个文档并将其入库
        
        Args:
            file_path: 文档文件路径
            enable_kg: 是否启用知识图谱写入
            enable_pinecone: 是否启用Pinecone向量上传
            extra_metadata: 额外的元数据（如faction），优先级最高
        """
        if not os.path.exists(file_path):
            logging.error(f"❌ 文件不存在: {file_path}")
            return

        logging.info(f"📄 开始处理文档: {file_path}")
        self._initialize_services(enable_kg, enable_pinecone)

        try:
            # 1. 读取和切片文档
            logging.info("Step 1: 正在进行文档切片...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 组装extra_metadata
            base_metadata = {'source_file': os.path.basename(file_path)}
            if extra_metadata:
                base_metadata.update(extra_metadata)
            # 尝试从文件名推断faction（如未指定）
            if 'faction' not in base_metadata:
                filename = os.path.basename(file_path)
                if 'aeldari' in filename.lower():
                    base_metadata['faction'] = '阿苏焉尼'
                elif 'corerule' in filename.lower():
                    base_metadata['faction'] = '核心规则'
                elif 'corefaq' in filename.lower():
                    base_metadata['faction'] = '核心FAQ'
            chunks = self.chunker.chunk_text(content, base_metadata)
            logging.info(f"✅ 文档切片完成，生成 {len(chunks)} 个切片。")
            
            # 保存切片到本地文件，用于观察切片效果
            try:
                save_chunks_to_json(chunks, "DATAUPLOD/chunks")
                logging.info("✅ 切片已保存到 DATAUPLOD/chunks 目录")
            except Exception as e:
                logging.warning(f"⚠️ 保存切片文件时出现警告: {e}")

            # === 新增：生成/刷新全库保护性词典 ===
            userdict_path = 'all_docs_dict.txt'
            # 假设所有文档路径可通过某种方式获得，这里仅用当前file_path做演示
            # 实际工程中建议传入所有文档路径
            subprocess.run(f"python3 DATAUPLOD/gen_userdict.py --f {file_path}", shell=True, check=True)
            logging.info(f"✅ 保护性词典已生成/刷新: {userdict_path}")

            # === 新增：初始化BM25Manager并fit所有chunk文本 ===
            bm25 = BM25Manager(user_dict_path=userdict_path)
            bm25.fit([chunk['text'] for chunk in chunks])
            logging.info(f"✅ BM25Manager已初始化并fit所有chunk文本")

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

            # 3. Pinecone 向量上传
            if enable_pinecone and self.pinecone_index and self.embedding_model:
                logging.info("Step 3: 正在进行向量化并上传到 Pinecone...")
                
                # 构建待上传的数据和上传记录
                vectors_to_upsert = []
                upload_records = []
                chunk_id_mapping = {}
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{os.path.basename(file_path)}-{i}"
                    chunk_id_mapping[i] = chunk_id
                    try:
                        embedding = self.embedding_model.embed_query(chunk['text'])
                        # 新增：生成sparse embedding
                        sparse_embedding = bm25.get_sparse_vector(chunk['text'])
                        pinecone_metadata = {
                            'text': chunk['text'],
                            'chunk_type': chunk.get('chunk_type', ''),
                            'content_type': chunk.get('content_type', ''),
                            'faction': chunk.get('faction', base_metadata.get('faction', '未知')),
                            'section_heading': chunk.get('section_heading', ''),
                        }
                        # 加入多级标题字段
                        for j in range(1, 7):
                            pinecone_metadata[f'h{j}'] = chunk.get(f'h{j}', '')
                        pinecone_metadata['chunk_id'] = str(i)
                        pinecone_metadata['source_file'] = chunk.get('source_file', '')
                        vector_data = {
                            'id': chunk_id,
                            'values': embedding,
                            'sparse_values': sparse_embedding,
                            'metadata': pinecone_metadata
                        }
                        vectors_to_upsert.append(vector_data)
                        upload_records.append({
                            'chunk_id': chunk_id,
                            'status': 'success',
                            'metadata': pinecone_metadata
                        })
                    except Exception as e:
                        print(f"[ERROR] Chunk {i} embedding/upload failed: {e}")
                        upload_records.append({
                            'chunk_id': chunk_id,
                            'status': 'fail',
                            'error': str(e)
                        })

                # 分批上传
    batch_size = 100
                successful_uploads = 0
                failed_uploads = 0
                
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    batch_start = i
                    batch_end = min(i + batch_size, len(vectors_to_upsert))
                    
                    try:
                        self.pinecone_index.upsert(vectors=batch)
                        successful_uploads += len(batch)
                        
                        # 更新批次中所有切片的状态
                        for j in range(batch_start, batch_end):
                            if j < len(upload_records):
                                upload_records[j]['status'] = 'uploaded'
                        
                        logging.info(f"  - 上传批次 {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}: {len(batch)} 个切片")
                        
                    except Exception as e:
                        failed_uploads += len(batch)
                        logging.error(f"❌ 批次 {i//batch_size + 1} 上传失败: {e}")
                        
                        # 更新批次中所有切片的状态
                        for j in range(batch_start, batch_end):
                            if j < len(upload_records):
                                upload_records[j]['status'] = 'failed'
                                upload_records[j]['error'] = str(e)
                
                # 保存上传记录到文件
                import json
                from datetime import datetime
                
                upload_log_file = f"DATAUPLOD/upload_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs(os.path.dirname(upload_log_file), exist_ok=True)
                
                upload_summary = {
                    'file_path': file_path,
                    'upload_time': datetime.now().isoformat(),
                    'total_chunks': len(chunks),
                    'successful_uploads': successful_uploads,
                    'failed_uploads': failed_uploads,
                    'upload_records': upload_records
                }
                
                with open(upload_log_file, 'w', encoding='utf-8') as f:
                    json.dump(upload_summary, f, ensure_ascii=False, indent=2)
                
                logging.info(f"✅ Pinecone 向量上传完成:")
                logging.info(f"  - 总切片数: {len(chunks)}")
                logging.info(f"  - 成功上传: {successful_uploads}")
                logging.info(f"  - 上传失败: {failed_uploads}")
                logging.info(f"  - 上传记录已保存到: {upload_log_file}")
                
                # 检查特定切片是否上传成功
                if failed_uploads > 0:
                    logging.warning("⚠️ 部分切片上传失败，请检查上传记录文件")
                    for record in upload_records:
                        if record['status'] == 'failed':
                            logging.warning(f"  - 切片 {record['chunk_id']} 上传失败: {record['error']}")

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
    parser.add_argument("--no-pinecone", action="store_true", help="禁用Pinecone向量上传")
    parser.add_argument("--faction", type=str, help="指定faction（可选，优先级最高）")
    args = parser.parse_args()
    
    manager = UpsertManager(environment=args.env, chunker_version=args.chunker_version)
    # 组装extra_metadata，优先使用命令行参数
    extra_metadata = {'source_file': os.path.basename(args.file_path)}
    if args.faction:
        extra_metadata['faction'] = args.faction
    manager.process_and_upsert_document(
        file_path=args.file_path,
        enable_kg=not args.no_kg,
        enable_pinecone=not args.no_pinecone,
        extra_metadata=extra_metadata
    )

if __name__ == "__main__":
    main()
