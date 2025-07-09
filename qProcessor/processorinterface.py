from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

class QueryProcessorInterface(ABC):
    """查询处理器接口"""
    
    @abstractmethod
    def process(self, query: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        处理查询
        Args:
            query: 用户输入的查询
        Returns:
            单查询：返回单个字典
            多查询：返回字典列表
        """
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """获取处理器类型"""
        pass
    
    @abstractmethod
    def is_multi_query(self) -> bool:
        """是否生成多个查询"""
        pass