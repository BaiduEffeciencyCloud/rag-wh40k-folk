"""
CRF特征提取器
为序列标注提供丰富的特征，包括字符级、词汇级、上下文特征
"""

import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class CRFFeatureExtractor:
    """CRF特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CRF特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.feature_config = config.get('crf_features', {})
        
        # 特征开关
        self.use_char_features = self.feature_config.get('use_char_features', True)
        self.use_word_features = self.feature_config.get('use_word_features', True)
        self.use_context_features = self.feature_config.get('use_context_features', True)
        self.use_position_features = self.feature_config.get('use_position_features', True)
        
        # 窗口大小
        self.window_size = self.feature_config.get('window_size', 2)
        
        # 预定义的特征模式
        self.patterns = {
            'number': r'\d+',
            'chinese': r'[\u4e00-\u9fff]+',
            'english': r'[a-zA-Z]+',
            'punctuation': r'[^\w\s]',
            'whitespace': r'\s+'
        }
    
    def extract_features(self, tokens: List[str], position: int) -> Dict[str, Any]:
        """
        为指定位置的token提取特征
        
        Args:
            tokens: token序列
            position: 当前token位置
            
        Returns:
            特征字典
        """
        features = {}
        
        # 当前token特征
        current_token = tokens[position] if position < len(tokens) else ''
        features.update(self._extract_token_features(current_token, position, len(tokens)))
        
        # 上下文特征
        if self.use_context_features:
            features.update(self._extract_context_features(tokens, position))
        
        # 位置特征
        if self.use_position_features:
            features.update(self._extract_position_features(position, len(tokens)))
        
        return features
    
    def _extract_token_features(self, token: str, position: int, total_length: int) -> Dict[str, Any]:
        """
        提取单个token的特征
        
        Args:
            token: 当前token
            position: token位置
            total_length: 总长度
            
        Returns:
            token特征字典
        """
        features = {}
        
        # 基础特征
        features['token'] = token
        features['token_lower'] = token.lower()
        features['token_length'] = len(token)
        
        # 字符级特征
        if self.use_char_features:
            features.update(self._extract_char_features(token))
        
        # 词汇级特征
        if self.use_word_features:
            features.update(self._extract_word_features(token))
        
        # 位置特征
        features['is_first'] = position == 0
        features['is_last'] = position == total_length - 1
        features['position'] = position
        features['position_ratio'] = position / max(total_length, 1)
        
        return features
    
    def _extract_char_features(self, token: str) -> Dict[str, Any]:
        """
        提取字符级特征
        
        Args:
            token: token字符串
            
        Returns:
            字符特征字典
        """
        features = {}
        
        if not token:
            return features
        
        # 首字符特征
        features['first_char'] = token[0]
        features['first_char_lower'] = token[0].lower()
        
        # 尾字符特征
        features['last_char'] = token[-1]
        features['last_char_lower'] = token[-1].lower()
        
        # 字符类型特征
        features['is_digit'] = token.isdigit()
        features['is_alpha'] = token.isalpha()
        features['is_upper'] = token.isupper()
        features['is_lower'] = token.islower()
        features['is_title'] = token.istitle()
        
        # 模式匹配特征
        for pattern_name, pattern in self.patterns.items():
            features[f'is_{pattern_name}'] = bool(re.match(pattern, token))
        
        # 特殊字符特征
        features['has_punctuation'] = bool(re.search(r'[^\w\s]', token))
        features['has_number'] = bool(re.search(r'\d', token))
        features['has_chinese'] = bool(re.search(r'[\u4e00-\u9fff]', token))
        features['has_english'] = bool(re.search(r'[a-zA-Z]', token))
        
        return features
    
    def _extract_word_features(self, token: str) -> Dict[str, Any]:
        """
        提取词汇级特征
        
        Args:
            token: token字符串
            
        Returns:
            词汇特征字典
        """
        features = {}
        
        if not token:
            return features
        
        # 词形特征
        features['word_shape'] = self._get_word_shape(token)
        features['word_suffix'] = token[-3:] if len(token) >= 3 else token
        features['word_prefix'] = token[:3] if len(token) >= 3 else token
        
        # 长度特征
        features['length_1'] = len(token) == 1
        features['length_2'] = len(token) == 2
        features['length_3'] = len(token) == 3
        features['length_4'] = len(token) == 4
        features['length_5+'] = len(token) >= 5
        
        # 常见词汇特征（可以扩展）
        common_words = {'的', '是', '在', '有', '和', '与', '或', '查询', '比较', '分析'}
        features['is_common_word'] = token in common_words
        
        return features
    
    def _get_word_shape(self, token: str) -> str:
        """
        获取词形模式
        
        Args:
            token: token字符串
            
        Returns:
            词形模式字符串
        """
        shape = []
        for char in token:
            if char.isupper():
                shape.append('X')
            elif char.islower():
                shape.append('x')
            elif char.isdigit():
                shape.append('d')
            else:
                shape.append(char)
        return ''.join(shape)
    
    def _extract_context_features(self, tokens: List[str], position: int) -> Dict[str, Any]:
        """
        提取上下文特征
        
        Args:
            tokens: token序列
            position: 当前位置
            
        Returns:
            上下文特征字典
        """
        features = {}
        
        # 前一个token特征
        if position > 0:
            prev_token = tokens[position - 1]
            features['prev_token'] = prev_token
            features['prev_token_lower'] = prev_token.lower()
            features['prev_token_length'] = len(prev_token)
        else:
            features['prev_token'] = '<START>'
            features['prev_token_lower'] = '<start>'
            features['prev_token_length'] = 0
        
        # 后一个token特征
        if position < len(tokens) - 1:
            next_token = tokens[position + 1]
            features['next_token'] = next_token
            features['next_token_lower'] = next_token.lower()
            features['next_token_length'] = len(next_token)
        else:
            features['next_token'] = '<END>'
            features['next_token_lower'] = '<end>'
            features['next_token_length'] = 0
        
        # 窗口特征
        for i in range(1, self.window_size + 1):
            # 前i个token
            if position - i >= 0:
                features[f'prev_{i}_token'] = tokens[position - i]
            else:
                features[f'prev_{i}_token'] = '<START>'
            
            # 后i个token
            if position + i < len(tokens):
                features[f'next_{i}_token'] = tokens[position + i]
            else:
                features[f'next_{i}_token'] = '<END>'
        
        return features
    
    def _extract_position_features(self, position: int, total_length: int) -> Dict[str, Any]:
        """
        提取位置特征
        
        Args:
            position: 当前位置
            total_length: 总长度
            
        Returns:
            位置特征字典
        """
        features = {}
        
        features['position'] = position
        features['position_ratio'] = position / max(total_length, 1)
        features['is_first'] = position == 0
        features['is_last'] = position == total_length - 1
        features['is_second'] = position == 1
        features['is_second_last'] = position == total_length - 2
        
        # 位置区间特征
        if total_length > 0:
            ratio = position / total_length
            features['position_quarter'] = int(ratio * 4)
            features['position_half'] = int(ratio * 2)
        
        return features
    
    def extract_sequence_features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        为整个序列提取特征
        
        Args:
            tokens: token序列
            
        Returns:
            特征序列
        """
        sequence_features = []
        
        for i in range(len(tokens)):
            features = self.extract_features(tokens, i)
            sequence_features.append(features)
        
        return sequence_features
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名称
        
        Returns:
            特征名称列表
        """
        # 这里返回所有可能的特征名称，用于调试和验证
        feature_names = [
            'token', 'token_lower', 'token_length',
            'first_char', 'first_char_lower', 'last_char', 'last_char_lower',
            'is_digit', 'is_alpha', 'is_upper', 'is_lower', 'is_title',
            'is_number', 'is_chinese', 'is_english', 'is_punctuation',
            'has_punctuation', 'has_number', 'has_chinese', 'has_english',
            'word_shape', 'word_suffix', 'word_prefix',
            'length_1', 'length_2', 'length_3', 'length_4', 'length_5+',
            'is_common_word',
            'prev_token', 'prev_token_lower', 'prev_token_length',
            'next_token', 'next_token_lower', 'next_token_length',
            'position', 'position_ratio', 'is_first', 'is_last',
            'is_second', 'is_second_last', 'position_quarter', 'position_half'
        ]
        
        # 添加窗口特征
        for i in range(1, self.window_size + 1):
            feature_names.extend([f'prev_{i}_token', f'next_{i}_token'])
        
        return feature_names 