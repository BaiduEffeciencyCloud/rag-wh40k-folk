from abc import ABC, abstractmethod

class ConnectionInterface(ABC):
    @abstractmethod
    def dense_search(self, query: str,filter:dict,embedding_model:None) -> list:
        pass
    @abstractmethod
    def hybrid_search(self, query: str,filter:dict) ->list:
        pass
    @abstractmethod
    def adaptive_data(self, chunks: list) -> dict:
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
    
