from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union


class SearchEngineInterface(ABC):
    """搜索引擎接口"""
    
    @abstractmethod
    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        执行搜索
        Args:
            query: 查询（可以是字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """获取搜索引擎类型"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """获取搜索引擎能力信息"""
        pass

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        对候选文档进行重排序（rerank）
        Args:
            query: 查询文本
            documents: 候选文档列表（每个为字符串）
            top_k: 返回前k个重排序结果
            **kwargs: 其他参数（如模型名等）
        Returns:
            List[Dict]: 每个结果包含text、score等
        """
        pass
    
