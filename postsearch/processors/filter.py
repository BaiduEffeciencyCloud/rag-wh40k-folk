from typing import List, Dict, Any
from ..interfaces import PostSearchInterface

class FilterProcessor(PostSearchInterface):
    """基础Filter处理器：simple模式下直接返回原始结果。"""
    
    def process(self, results: List[Dict[str, Any]], mode: str = 'simple', **kwargs) -> List[Dict[str, Any]]:
        """
        simple模式下直接返回原始结果。
        未来可扩展max_length等过滤逻辑。
        """
        if mode == 'simple':
            return results
        # 预留：其他模式下可添加过滤逻辑
        return results

    def get_name(self) -> str:
        return 'filter'
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取配置模式
        Returns:
            配置模式字典
        """
        return {
            'mode': {
                'type': 'string',
                'default': 'simple',
                'description': '过滤模式：simple为直接返回，其他模式可扩展'
            },
            'max_length': {
                'type': 'integer',
                'default': None,
                'description': '最大长度限制（token数）'
            }
        }
