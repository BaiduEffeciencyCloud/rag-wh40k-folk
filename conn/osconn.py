import logging
import os
from datetime import datetime
from http import HTTPStatus
from .coninterface import ConnectionInterface
from config import OPENSERACH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD, OPENSERACH_INDEX, EMBEDDING_MODEL_QWEN, QWEN_DIMENSION


# Opensearch数据库的处理类,负责数据转化,数据上传,数据搜索
class OpenSearchConnection(ConnectionInterface):
    def __init__(self, index: str = None, uri: str = None, username: str = None, password: str = None):
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
        # 从config.py获取配置，如果没有传入参数的话
        self.index = index or OPENSERACH_INDEX
        self.uri = uri or OPENSERACH_URI
        self.username = username or OPENSEARCH_USERNAME
        self.password = password or OPENSEARCH_PASSWORD
        
        # 记录初始化信息
        self.logger.info(f"OpenSearch connection initialized - Index: {self.index}, URI: {self.uri}")
        
        # 初始化OpenSearch客户端（暂时不创建，等后续实现）
        self.client = None

    #从数据库中搜索数据,这个search只负责召回检索记录,不负责答案生成
    def search(self, query: str, filter: dict) -> list:
        pass

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
            self.logger.info(f"chunk {chunk_id} sparse向量生成成功，特征数量: {len(sparse_vector)}")
            
            # 4. 构造OpenSearch文档
            document = {
                '_id': chunk_id,
                'text': chunk['text'],
                'embedding': dense_vector,
                'sparse': sparse_vector,
                'chunk_type': chunk.get('chunk_type', 'default'),
                'content_type': chunk.get('content_type', 'text'),
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

    #将数据上传到opensearch
    def upload_data(self, data: dict) -> dict:
        pass
    
    def bulk_adaptive_data(self, chunks: list) -> list:
        pass
    
    def bulk_upload_data(self, data: list) -> list:
        pass