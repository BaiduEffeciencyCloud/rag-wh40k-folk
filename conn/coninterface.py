from abc import ABC, abstractmethod
from typing import List, Dict

class ConnectionInterface(ABC):
    @abstractmethod
    def dense_search(self, query: str,filter:dict,embedding_model:None,top_k:int=20) -> list:
        pass
    @abstractmethod
    def hybrid_search(self, query: str,filter:dict,top_k:int=20) ->list:
        pass
    @abstractmethod
    def adaptive_data(self, chunk: dict, chunk_id: str, bm25_manager, pre_generated_embedding: list = None) -> dict:
        """
        将单个chunk数据转换成适合当前数据库的格式
        
        Args:
            chunk: 单个chunk数据，包含text和metadata
            chunk_id: 文档ID
            bm25_manager: 用于生成sparse向量的管理器实例
            pre_generated_embedding: 预生成的dense向量，如果提供则直接使用
            
        Returns:
            dict: 转换后的数据库文档格式
        """
        pass

    @abstractmethod
    def upload_data(self, data: dict) -> dict:
        pass

    @abstractmethod
    def bulk_upload_data(self, data: list) -> list:
        pass
    
    @abstractmethod
    def bulk_adaptive_data(self, chunks: list) -> list:
        pass
    
    @abstractmethod
    def get_index_count(self, index_name: str) -> int:
        """获取指定 index 下的数据条目数量"""
        pass
    
    @abstractmethod
    def del_data_by_index(self, index_name: str) -> bool:
        """删除指定 index 下的所有数据"""
        pass

    @abstractmethod
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5, model: str = None, **kwargs) -> List[Dict]:
        """重排序搜索结果"""
        pass
    
