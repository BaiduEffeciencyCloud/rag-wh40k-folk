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

    
