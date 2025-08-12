import logging
import os
from datetime import datetime
from opensearchpy import OpenSearch
from http import HTTPStatus
from .coninterface import ConnectionInterface
from config import OPENSEARCH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, EMBEDDING_MODEL_QWEN, QWEN_DIMENSION
from opensearchpy import OpenSearch


# Opensearch数据库的处理类,负责数据转化,数据上传,数据搜索
class OpenSearchConnection(ConnectionInterface):
    def __init__(self, index: str = None, uri: str = None, username: str = None, password: str = None):
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
        # 从config.py获取配置，如果没有传入参数的话
        self.index = index or OPENSEARCH_INDEX
        self.uri = uri or OPENSEARCH_URI
        self.username = username or OPENSEARCH_USERNAME
        self.password = password or OPENSEARCH_PASSWORD
        
        # 记录初始化信息
        self.logger.info(f"OpenSearch connection initialized - Index: {self.index}, URI: {self.uri}")
        
        # 初始化OpenSearch客户端
        try:
            self.client = OpenSearch(
                hosts=[self.uri],
                http_auth=(self.username, self.password),
                use_ssl=True,
                verify_certs=False
            )
            
            # 初始化成功后立即进行心跳测试
            self.logger.info("OpenSearch客户端初始化成功，开始心跳测试...")
            heartbeat_result = self.heartbeat()
            
            if not heartbeat_result:
                raise Exception("OpenSearch心跳测试失败，连接不可用")
            
            self.logger.info("OpenSearch连接验证成功")
            
        except ImportError:
            error_msg = "opensearchpy 未安装，连接功能将不可用"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"OpenSearch客户端初始化失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        


    #从数据库中搜索数据,这个search只负责召回检索记录,不负责答案生成
    def dense_search(self, query: str, filter: dict, embedding_model=None) -> list:
        """
        执行 dense search
        
        Args:
            query: 查询文本
            filter: 过滤条件字典
            embedding_model: 嵌入模型实例，用于生成查询向量
        
        Returns:
            list: 搜索结果列表，每个结果包含id、score、text等字段
        """
        try:
            # 1. 参数验证
            if not embedding_model:
                error_msg = "embedding_model实例为空，无法生成查询向量"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not query or not query.strip():
                error_msg = "查询文本不能为空"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 2. 确保连接可用
            self._ensure_connection()
            
            # 3. 生成查询向量
            query_embedding = embedding_model.get_embedding(query)
            if not query_embedding or len(query_embedding) != 2048:
                error_msg = f"查询向量生成失败或维度不正确，期望2048维，实际{len(query_embedding) if query_embedding else 0}维"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"查询向量生成成功，维度: {len(query_embedding)}")
            
            # 4. 构建搜索查询
            search_body = {
                "size": 10,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": 20  # 搜索20个最近邻，从中选择top 10
                        }
                    }
                }
            }
            
            # 5. 添加过滤条件
            if filter:
                search_body["query"] = {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": 20
                                    }
                                }
                            }
                        ],
                        "filter": self._build_filter_query(filter)
                    }
                }
            
            # 6. 执行搜索
            self.logger.info(f"执行dense search，查询: {query[:50]}...")
            response = self.client.search(
                index=self.index,
                body=search_body
            )
            
            # 7. 处理结果
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            for hit in hits:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'text': hit['_source']['text'],
                    'faction': hit['_source'].get('faction', ''),
                    'content_type': hit['_source'].get('content_type', ''),
                    'section_heading': hit['_source'].get('section_heading', ''),
                    'h1': hit['_source'].get('h1', ''),
                    'h2': hit['_source'].get('h2', ''),
                    'h3': hit['_source'].get('h3', ''),
                    'source_file': hit['_source'].get('source_file', ''),
                    'chunk_index': hit['_source'].get('chunk_index', 0),
                    'created_at': hit['_source'].get('created_at', '')
                }
                results.append(result)
            
            self.logger.info(f"Dense search 完成，返回 {len(results)} 个结果")
            return results
            
        except ValueError as e:
            # 重新抛出 ValueError，保持异常类型
            raise
        except Exception as e:
            error_msg = f"Dense search 失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    def hybrid_search(self, query: str, filter: dict) -> list:
        pass
    
    def heartbeat(self) -> bool:
        """
        测试与OpenSearch的连接状态
        
        Returns:
            bool: True表示连接正常，False表示连接失败
        """
        try:
            if not self.client:
                self.logger.warning("OpenSearch客户端未初始化")
                return False
            
            # 使用ping方法测试连接
            response = self.client.ping()
            if response:
                self.logger.info("OpenSearch连接正常")
                return True
            else:
                self.logger.warning("OpenSearch ping失败")
                return False
                
        except Exception as e:
            self.logger.error(f"OpenSearch心跳检测失败: {str(e)}")
            return False

    def adaptive_data(self, chunk: dict, chunk_id: str, embedding_model, bm25_manager) -> dict:
        """
        将单个chunk数据转换成适合OpenSearch的格式
        
        Args:
            chunk: 单个chunk数据，包含text和metadata
            chunk_id: 文档ID
            embedding_model: 用于生成dense向量的模型实例
            bm25_manager: 用于生成sparse向量的管理器实例
            
        Returns:
            dict: 转换后的OpenSearch文档格式
            
        Raises:
            ValueError: 当embedding_model或bm25_manager为空时
            Exception: 当向量生成或数据处理失败时
        """
        # 1. 参数验证和日志记录
        if not embedding_model:
            error_msg = "embedding_model实例为空，无法生成dense向量"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not bm25_manager:
            error_msg = "bm25_manager实例为空，无法生成sparse向量"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"开始处理chunk {chunk_id}，使用embedding_model: {type(embedding_model).__name__}")
        
        try:
            # 2. 生成dense向量
            # QWEN使用TextEmbedding.call()类方法调用
            resp = embedding_model.call(
                model=EMBEDDING_MODEL_QWEN,
                input=chunk['text'],
                text_type="document",
                dimension=QWEN_DIMENSION
            )
            
            if resp.status_code != HTTPStatus.OK:
                error_msg = f"chunk {chunk_id} dense向量生成失败: {resp.code}, {resp.message}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
            dense_vector = resp.output['embeddings'][0]['embedding']
            self.logger.info(f"chunk {chunk_id} dense向量生成成功，维度: {len(dense_vector)}")
            
            # 3. 生成sparse向量
            sparse_vector = bm25_manager.get_sparse_vector(chunk['text'])
            
            # 转换稀疏向量格式：从 {indices: [...], values: [...]} 到 {index: value, ...}
            # 这是OpenSearch rank_features类型需要的格式
            if sparse_vector and 'indices' in sparse_vector and 'values' in sparse_vector:
                converted_sparse = {}
                for i, index in enumerate(sparse_vector['indices']):
                    converted_sparse[str(index)] = sparse_vector['values'][i]
                sparse_vector = converted_sparse
                self.logger.info(f"chunk {chunk_id} sparse向量转换完成，特征数量: {len(sparse_vector)}")
            else:
                self.logger.warning(f"chunk {chunk_id} sparse向量格式异常或为空")
                sparse_vector = {}
            
            # 4. 构造OpenSearch文档
            document = {
                '_id': chunk_id,
                'text': chunk['text'],
                'embedding': dense_vector,
                'sparse': sparse_vector,
                'chunk_type': chunk.get('chunk_type', 'default'),
                'content_type': chunk.get('content_type', ''),
                'faction': chunk.get('faction', ''),
                'section_heading': chunk.get('section_heading', ''),
                # 标题层级字段
                'h1': chunk.get('h1', ''),
                'h2': chunk.get('h2', ''),
                'h3': chunk.get('h3', ''),
                'h4': chunk.get('h4', ''),
                'h5': chunk.get('h5', ''),
                'h6': chunk.get('h6', ''),
                # 其他元数据
                'source_file': chunk.get('source_file', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'created_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"chunk {chunk_id} 转换完成，文档结构: {list(document.keys())}")
            return document
            
        except Exception as e:
            error_msg = f"chunk {chunk_id} 转换失败: {str(e)}"
            self.logger.error(error_msg)
            # 直接抛出异常，让上层处理
            raise Exception(error_msg)

    def upload_data(self, index_name: str, document: dict, doc_id: str = None) -> dict:
        """上传单个文档到指定的OpenSearch索引"""
        try:
            # 准备上传参数
            upload_params = {
                'index': index_name,
                'body': document
            }
            
            # 如果提供了doc_id，则添加到参数中
            if doc_id:
                upload_params['id'] = doc_id
            
            # 执行上传
            self.logger.info(f"正在上传文档到索引 {index_name}")
            response = self.client.index(**upload_params)
            
            # 记录上传结果
            self.logger.info(f"文档上传成功: {response.get('result', 'unknown')}")
            return response
            
        except Exception as e:
            error_msg = f"文档上传失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def bulk_upload_data(self, index_name: str, documents: list, 
                         batch_size: int, 
                         doc_ids: list = None) -> dict:
        """批量上传多个文档到指定的OpenSearch索引"""
        try:
            # 验证输入参数
            if not documents:
                self.logger.warning("文档列表为空，跳过批量上传")
                return {'took': 0, 'errors': False, 'items': []}
            
            # 准备bulk API格式的数据
            bulk_data = []
            for i, doc in enumerate(documents):
                # 操作行
                operation = {
                    'index': {
                        '_index': index_name
                    }
                }
                
                # 如果提供了doc_ids，使用自定义ID
                if doc_ids and i < len(doc_ids):
                    operation['index']['_id'] = doc_ids[i]
                
                bulk_data.append(operation)
                bulk_data.append(doc)
            
            # 执行批量上传
            self.logger.info(f"正在批量上传 {len(documents)} 个文档到索引 {index_name}")
            response = self.client.bulk(body=bulk_data)
            
            # 记录上传结果
            if response.get('errors', False):
                self.logger.warning(f"批量上传完成，但有部分文档失败: {response}")
            else:
                self.logger.info(f"批量上传成功完成: {len(documents)} 个文档")
            
            return response
            
        except Exception as e:
            error_msg = f"批量上传失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def bulk_adaptive_data(self, chunks: list) -> list:
        """批量转换chunk数据（保持接口兼容性）"""
        try:
            if not chunks:
                self.logger.warning("chunks列表为空，跳过批量转换")
                return []
            
            converted_docs = []
            for chunk in chunks:
                try:
                    # 生成chunk_id
                    chunk_id = chunk.get('chunk_id', f"chunk_{len(converted_docs)}")
                    
                    # 使用现有的adaptive_data方法转换单个chunk
                    # 注意：这里需要embedding_model和bm25_manager，但接口中没有提供
                    # 为了保持兼容性，我们返回一个基本的转换结果
                    converted_doc = {
                        '_id': chunk_id,
                        'text': chunk.get('text', ''),
                        'chunk_type': chunk.get('chunk_type', 'default'),
                        'content_type': chunk.get('content_type', ''),
                        'faction': chunk.get('faction', ''),
                        'section_heading': chunk.get('section_heading', ''),
                        'h1': chunk.get('h1', ''),
                        'h2': chunk.get('h2', ''),
                        'h3': chunk.get('h3', ''),
                        'h4': chunk.get('h4', ''),
                        'h5': chunk.get('h5', ''),
                        'h6': chunk.get('h6', ''),
                        'source_file': chunk.get('source_file', ''),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    converted_docs.append(converted_doc)
                    
                except Exception as e:
                    self.logger.error(f"转换chunk失败: {str(e)}")
                    # 继续处理其他chunk
                    continue
            
            self.logger.info(f"批量转换完成: {len(converted_docs)}/{len(chunks)} 个chunk成功转换")
            return converted_docs
            
        except Exception as e:
            error_msg = f"批量转换失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_index_count(self, index_name: str) -> int:
        """获取指定 index 下的数据条目数量"""
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 使用 count API 获取文档数量
            response = self.client.count(index=index_name)
            count = response.get('count', 0)
            
            self.logger.info(f"Index {index_name} 中共有 {count} 个文档")
            return count
            
        except Exception as e:
            error_msg = f"获取 index {index_name} 文档数量失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def del_data_by_index(self, index_name: str) -> bool:
        """删除指定 index 下的所有数据"""
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 使用 delete_by_query API 删除所有文档
            # 使用 match_all 查询匹配所有文档
            response = self.client.delete_by_query(
                index=index_name,
                body={
                    "query": {
                        "match_all": {}
                    }
                }
            )
            
            # 获取删除的文档数量
            deleted_count = response.get('deleted', 0)
            total_count = response.get('total', 0)
            
            self.logger.info(f"成功删除 index {index_name} 中的 {deleted_count}/{total_count} 个文档")
            
            # 返回是否成功删除（即使删除0个文档也算成功）
            return True
            
        except Exception as e:
            error_msg = f"删除 index {index_name} 数据失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _build_filter_query(self, filter: dict) -> list:
        """
        构建过滤查询条件
        
        Args:
            filter: 过滤条件字典
        
        Returns:
            list: 过滤查询列表
        """
        filter_queries = []
        
        if 'faction' in filter:
            filter_queries.append({
                "term": {"faction": filter['faction']}
            })
        
        if 'content_type' in filter:
            filter_queries.append({
                "term": {"content_type": filter['content_type']}
            })
        
        if 'source_file' in filter:
            filter_queries.append({
                "term": {"source_file": filter['source_file']}
            })
        
        if 'h1' in filter:
            filter_queries.append({
                "term": {"h1": filter['h1']}
            })
        
        if 'h2' in filter:
            filter_queries.append({
                "term": {"h2": filter['h2']}
            })
        
        if 'h3' in filter:
            filter_queries.append({
                "term": {"h3": filter['h3']}
            })
        
        return filter_queries

    def _ensure_connection(self):
        """
        确保OpenSearch连接可用
        
        Raises:
            Exception: 当连接不可用时
        """
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 执行心跳测试
            if not self.heartbeat():
                raise Exception("OpenSearch连接不可用")
                
        except Exception as e:
            error_msg = f"连接检查失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)