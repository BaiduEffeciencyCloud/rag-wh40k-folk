from .processorinterface import QueryProcessorInterface
from typing import Dict, Any

class StraightforwardProcessor(QueryProcessorInterface):
    """直接传递原生query的处理器"""
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        直接返回原始查询，不做任何处理
        Args:
            query: 用户输入的查询
        Returns:
            包含原始查询信息的字典
        """
        return {
            "text": query,
            "entities": [],
            "relations": [],
            "processed": True,
            "type": "straightforward",
            "original_query": query
        }
    
    def get_type(self) -> str:
        return "straight"
    
    def is_multi_query(self) -> bool:
        return False



