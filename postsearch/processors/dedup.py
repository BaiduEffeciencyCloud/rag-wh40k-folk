from typing import List, Dict, Any
from ..interfaces import PostSearchInterface

class DedupProcessor(PostSearchInterface):
    """去重处理器 - 去除重复或高度相似的检索结果"""
    
    def process(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        去除重复或高度相似的检索结果
        Args:
            results: 检索结果列表
            **kwargs: 其他参数（如similarity_threshold、query_info等）
        Returns:
            去重后的结果列表
        """
        # TODO: 实现去重逻辑
        # 目前简单返回原始结果，后续可实现基于相似度的去重
        return results

    def get_name(self) -> str:
        return 'dedup'
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取配置模式
        Returns:
            配置模式字典
        """
        return {
            'similarity_threshold': {
                'type': 'float',
                'default': 0.8,
                'description': '相似度阈值，超过此值认为是重复'
            },
            'dedup_method': {
                'type': 'string',
                'default': 'semantic',
                'description': '去重方法：semantic（语义相似度）或exact（精确匹配）'
            }
        }
