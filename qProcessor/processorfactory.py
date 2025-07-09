from typing import Union, List, Dict, Any
from .processorinterface import QueryProcessorInterface
from .straightforward import StraightforwardProcessor
from .expander import ExpanderQueryProcessor
from .cot import COTProcessor

class QueryProcessorFactory:
    """查询处理器工厂类"""
    
    @staticmethod
    def create_processor(processor_type: str) -> QueryProcessorInterface:
        """
        根据处理器类型创建对应的处理器实例
        Args:
            processor_type: 处理器类型
        Returns:
            对应的处理器实例
        """
        processors = {
            "straightforward": StraightforwardProcessor(),
            "origin": StraightforwardProcessor(),  # origin等同于straightforward
            "expander": ExpanderQueryProcessor(),
            "cot": COTProcessor()
        }
        
        processor = processors.get(processor_type)
        if processor is None:
            # 默认返回straightforward处理器
            print(f"警告：未知的处理器类型 '{processor_type}'，使用默认处理器")
            return StraightforwardProcessor()
        
        return processor
    
    @staticmethod
    def process_query(processor_type: str, query: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        统一处理查询的静态方法
        Args:
            processor_type: 处理器类型
            query: 用户查询
        Returns:
            处理结果（单查询或多查询）
        """
        processor = QueryProcessorFactory.create_processor(processor_type)
        return processor.process(query)
    
    @staticmethod
    def get_available_processors() -> List[str]:
        """获取所有可用的处理器类型"""
        return ["straightforward", "origin", "expander", "cot"]
    
    @staticmethod
    def validate_processor_type(processor_type: str) -> bool:
        """
        验证处理器类型是否有效
        Args:
            processor_type: 处理器类型
        Returns:
            是否有效
        """
        return processor_type in QueryProcessorFactory.get_available_processors()
    
    @staticmethod
    def get_processor_info(processor_type: str) -> Dict[str, Any]:
        """
        获取处理器信息
        Args:
            processor_type: 处理器类型
        Returns:
            处理器信息字典
        """
        processor = QueryProcessorFactory.create_processor(processor_type)
        return {
            "type": processor.get_type(),
            "is_multi_query": processor.is_multi_query(),
            "description": QueryProcessorFactory._get_processor_description(processor_type)
        }
    
    @staticmethod
    def _get_processor_description(processor_type: str) -> str:
        """获取处理器描述"""
        descriptions = {
            "straightforward": "直接传递原始查询，不做任何处理",
            "origin": "直接传递原始查询，不做任何处理（与straightforward相同）",
            "expander": "查询扩写处理器，生成多个相关查询",
            "cot": "思维链处理器，将复杂查询分解为多个子查询"
        }
        return descriptions.get(processor_type, "未知处理器类型")


