from typing import List
import dashscope
from dashscope.embeddings import TextEmbedding
from .eminterface import EmbeddingInterface
from config import QWEN_DIMENSION, CONNECT_TIMEOUT, READ_TIMEOUT
from utils.retry_utils import embedding_retry_decorator

class QwenEmbedding(EmbeddingInterface):
    """Qwen嵌入模型实现"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v4",dimensions:int=QWEN_DIMENSION):
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
        self.dimensions = dimensions
        self._embeddings = TextEmbedding
        
        # 设置dashscope的全局超时配置
        # 注意：dashscope的TextEmbedding.call()方法本身不支持超时参数
        # 但我们可以通过装饰器来处理重试逻辑
    
    @embedding_retry_decorator()  # 使用默认配置
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量表示"""
        response = self._embeddings.call(
            model=self.model,
            input=text,
            dimension=self.dimensions
        )
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"Qwen embedding failed: {response.message}")
    
    @embedding_retry_decorator()  # 使用默认配置
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量表示"""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return embeddings
    
    def get_dimensions(self) -> int:
        """获取向量维度"""
        # Qwen模型通常返回1536维向量
        return self.dimensions
