'''
openai embedding
'''

from typing import List
from langchain_openai import OpenAIEmbeddings
from .eminterface import EmbeddingInterface
from config import OPENAI_DIMENSION, EMBEDDING_CONNECT_TIMEOUT, EMBEDDING_READ_TIMEOUT
from utils.retry_utils import embedding_retry_decorator

class OpenEmbedding(EmbeddingInterface):
    """OpenAI嵌入模型实现"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", dimensions: int = OPENAI_DIMENSION):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        
        # 使用配置的超时时间初始化OpenAIEmbeddings
        self._embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            dimensions=dimensions,
            # 设置HTTP客户端的超时配置
            request_timeout=EMBEDDING_READ_TIMEOUT
        )
    
    @embedding_retry_decorator()  # 使用默认配置
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量表示"""
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"OpenAI embedding failed: {str(e)}")
    
    @embedding_retry_decorator()  # 使用默认配置
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量表示"""
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            raise Exception(f"OpenAI batch embedding failed: {str(e)}")
    
    def get_dimensions(self) -> int:
        """获取向量维度"""
        return self.dimensions
