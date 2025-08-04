"""
特征提取器工厂类
根据配置创建不同类型的特征提取器
"""

from typing import Dict, Any, List
from intent_training.feature.base import BaseFeatureExtractor
from intent_training.feature.feature_engineering import IntentFeatureExtractor
from intent_training.feature.ad_feature import AdvancedFeature
from intent_training.feature.pa_feature import EnhancedFeatureExtractor
import logging

logger = logging.getLogger(__name__)


class FeatureExtractorFactory:
    """特征提取器工厂类"""
    
    @staticmethod
    def create_feature_extractor(config: Dict[str, Any]) -> BaseFeatureExtractor:
        """
        根据配置创建特征提取器
        
        Args:
            config: 配置字典
            
        Returns:
            特征提取器实例
        """
        feature_type = config.get('feature_type', 'basic')
        
        if feature_type == 'pa':
            logger.info("创建Position-Aware特征提取器")
            return EnhancedFeatureExtractor(config)
        elif feature_type == 'advanced':
            logger.info("创建高级特征提取器")
            return AdvancedFeature(config)
        else:
            logger.info("创建基础特征提取器")
            return IntentFeatureExtractor(config)
    
    @staticmethod
    def get_available_types() -> List[str]:
        """
        获取可用的特征提取器类型
        
        Returns:
            特征提取器类型列表
        """
        return ['basic', 'advanced', 'pa']
    
    @staticmethod
    def is_enhanced_feature_extractor(config: Dict[str, Any]) -> bool:
        """
        判断是否使用增强特征提取器
        
        Args:
            config: 配置字典
            
        Returns:
            是否使用增强特征
        """
        feature_type = config.get('feature_type', 'basic')
        return feature_type == 'advanced' or feature_type == 'pa' 