from typing import Dict, Any, List
import logging
from .interfaces import PostSearchInterface
from .processors.filter import FilterProcessor
from .processors.rerank import RerankProcessor
from .processors.mmr import MMRProcessor
from .processors.dedup import DedupProcessor

logger = logging.getLogger(__name__)

class PostSearchFactory:
    """Post-Search 工厂类"""
    
    def __init__(self):
        # 注册所有可用的处理器
        self._processors = {
            'simple': FilterProcessor,
            'rerank': RerankProcessor,
            'mmr': MMRProcessor,
            'dedup': DedupProcessor
        }
    @staticmethod
    def create_processor(processor_name: str) -> PostSearchInterface:
        """
        创建指定的处理器实例
        Args:
            processor_name: 处理器名称（如'simple', 'rerank', 'mmr', 'dedup'）
        Returns:
            处理器实例
        Raises:
            ValueError: 当处理器名称不存在时
        """
        processors = {
            'simple': FilterProcessor,
            'rerank': RerankProcessor,
            'mmr': MMRProcessor,
            'dedup': DedupProcessor
        }
        
        if processor_name not in processors:
            available_processors = list(processors.keys())
            raise ValueError(f"未知的处理器名称: {processor_name}。可用处理器: {available_processors}")
        
        processor_class = processors[processor_name]
        try:
            processor = processor_class()
            logger.info(f"成功创建处理器: {processor_name}")
            return processor
        except Exception as e:
            logger.error(f"创建处理器 {processor_name} 失败: {str(e)}")
            raise
    
    def create_processors(self, processor_names: List[str]) -> List[PostSearchInterface]:
        """
        批量创建多个处理器实例
        Args:
            processor_names: 处理器名称列表
        Returns:
            处理器实例列表
        """
        processors = []
        for name in processor_names:
            processor = self.create_processor(name)
            processors.append(processor)
        return processors
    
    def get_available_processors(self) -> List[str]:
        """
        获取所有可用的处理器名称
        Returns:
            可用处理器名称列表
        """
        return list(self._processors.keys())
    
    def register_processor(self, name: str, processor_class: type):
        """
        注册新的处理器类
        Args:
            name: 处理器名称
            processor_class: 处理器类（必须继承PostSearchInterface）
        """
        if not issubclass(processor_class, PostSearchInterface):
            raise ValueError(f"处理器类 {processor_class.__name__} 必须继承 PostSearchInterface")
        
        self._processors[name] = processor_class
        logger.info(f"注册新处理器: {name} -> {processor_class.__name__}")
    
    def validate_strategy(self, strategy: List[str]) -> bool:
        """
        验证处理策略是否有效
        Args:
            strategy: 处理器名称列表
        Returns:
            策略是否有效
        """
        for processor_name in strategy:
            if processor_name not in self._processors:
                logger.warning(f"策略中包含未知处理器: {processor_name}")
                return False
        return True
