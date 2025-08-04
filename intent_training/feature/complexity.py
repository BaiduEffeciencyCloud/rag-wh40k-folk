"""
复杂度计算器模块
提供统一的复杂度计算功能，支持简单和高级两种计算方式
"""

import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class ComplexityCalculator:
    """复杂度计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化复杂度计算器
        
        Args:
            config: 配置字典，包含复杂度计算参数
        """
        self.config = config
        self._init_complexity_config()
    
    def _init_complexity_config(self):
        """初始化复杂度配置"""
        # 从配置中获取复杂度计算参数
        self.complexity_config = self.config.get('advanced_feature', {}).get('complexity', {})
        self.max_sentence_length = self.complexity_config.get('max_sentence_length', 20.0)
        self.long_word_threshold = self.complexity_config.get('long_word_threshold', 4)
        self.very_long_word_threshold = self.complexity_config.get('very_long_word_threshold', 6)
        self.punctuation_threshold = self.complexity_config.get('punctuation_threshold', 10.0)
        self.weight_length = self.complexity_config.get('weight_length', 0.25)
        self.weight_unique = self.complexity_config.get('weight_unique', 0.25)
        self.weight_long_word = self.complexity_config.get('weight_long_word', 0.25)
        self.weight_punctuation = self.complexity_config.get('weight_punctuation', 0.25)
        
        # 从配置中获取标点符号
        punctuation_config = self.config.get('advanced_feature', {}).get('punctuation', {})
        self.all_punctuation = punctuation_config.get('all_punctuation', '，。！？；：""''（）【】')
    
    def calculate(self, query: str) -> float:
        """
        计算查询复杂度
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度分数 (0.0-1.0)
        """
        if not query.strip():
            return 0.0
        
        # 默认使用高级复杂度计算
        return self.calculate_advanced(query)
    
    def calculate_simple(self, query: str) -> float:
        """
        简单复杂度计算（adaptive.py 的逻辑）
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度分数 (0.0-1.0)
        """
        if not query.strip():
            return 0.0
        
        # 基于查询长度和词汇复杂度计算
        length_factor = min(len(query) / 50.0, 1.0)  # 长度因子
        word_count = len(query.split())
        word_factor = min(word_count / 10.0, 1.0)  # 词汇因子
        
        return (length_factor + word_factor) / 2.0
    
    def calculate_advanced(self, query: str) -> float:
        """
        高级复杂度计算（ad_feature.py 的逻辑）
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度分数 (0.0-1.0)
        """
        if not query.strip():
            return 0.0
        
        words = query.split()
        if not words:
            return 0.0
        
        # 1. 句子长度复杂度
        length_complexity = self._calculate_length_complexity(words)
        
        # 2. 词汇多样性复杂度
        unique_complexity = self._calculate_unique_complexity(words)
        
        # 3. 长词复杂度
        long_word_complexity = self._calculate_long_word_complexity(words)
        
        # 4. 标点符号复杂度
        punctuation_complexity = self._calculate_punctuation_complexity(query)
        
        # 综合复杂度
        complexity = (self.weight_length * length_complexity + 
                     self.weight_unique * unique_complexity + 
                     self.weight_long_word * long_word_complexity + 
                     self.weight_punctuation * punctuation_complexity)
        
        return complexity
    
    def _calculate_length_complexity(self, words: List[str]) -> float:
        """计算长度复杂度"""
        return min(len(words) / self.max_sentence_length, 1.0)
    
    def _calculate_unique_complexity(self, words: List[str]) -> float:
        """计算词汇多样性复杂度"""
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _calculate_long_word_complexity(self, words: List[str]) -> float:
        """计算长词复杂度"""
        if not words:
            return 0.0
        long_word_count = len([w for w in words if len(w) > self.long_word_threshold])
        return long_word_count / len(words)
    
    def _calculate_punctuation_complexity(self, query: str) -> float:
        """计算标点符号复杂度"""
        punctuation_count = len([c for c in query if c in self.all_punctuation])
        return min(punctuation_count / self.punctuation_threshold, 1.0)
    
    def get_complexity_level(self, complexity: float) -> str:
        """
        根据复杂度分数获取复杂度等级
        
        Args:
            complexity: 复杂度分数
            
        Returns:
            复杂度等级: "simple", "medium", "complex"
        """
        if complexity < 0.3:
            return "simple"
        elif complexity < 0.7:
            return "medium"
        else:
            return "complex"


class ComplexityFeatureExtractor:
    """句子复杂度特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化句子复杂度特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.complexity_calculator = ComplexityCalculator(config)
    
    def extract_complexity_features(self, texts: List[str]) -> np.ndarray:
        """
        提取句子复杂度特征
        
        Args:
            texts: 文本列表
            
        Returns:
            句子复杂度特征矩阵
        """
        if not texts:
            return np.array([]).reshape(0, 1)
        
        features = []
        for text in texts:
            complexity = self.complexity_calculator.calculate(text)
            features.append([complexity])
        
        return np.array(features)
    
    def extract_single_complexity_feature(self, text: str) -> np.ndarray:
        """
        提取单个文本的句子复杂度特征
        
        Args:
            text: 单个文本
            
        Returns:
            句子复杂度特征向量
        """
        complexity = self.complexity_calculator.calculate(text)
        return np.array([complexity]) 