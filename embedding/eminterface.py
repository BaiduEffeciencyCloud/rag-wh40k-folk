from abc import ABC, abstractmethod
from typing import List, Union

class EmbeddingInterface(ABC):
    """嵌入模型接口抽象类"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量表示"""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """获取向量维度"""
        pass
    