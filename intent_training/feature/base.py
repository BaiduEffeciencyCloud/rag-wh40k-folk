"""
特征提取器基础接口
定义统一的特征提取器接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseFeatureExtractor(ABC):
    """特征提取器基础接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """初始化特征提取器"""
        pass
    
    @abstractmethod
    def fit(self, queries: List[str]):
        """训练特征提取器"""
        pass
    
    @abstractmethod
    def transform(self, queries: List[str]) -> np.ndarray:
        """提取特征"""
        pass
    
    @abstractmethod
    def fit_transform(self, queries: List[str]) -> np.ndarray:
        """训练并提取特征"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存特征提取器"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载特征提取器"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        pass 