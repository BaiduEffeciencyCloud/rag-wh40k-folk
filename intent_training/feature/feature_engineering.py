import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class IntentFeatureExtractor:
    """意图识别特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征提取器
        Args:
            config: 配置字典
        """
        self.config = config
        self.tfidf_vectorizer = None
        self.is_fitted = False
        self.analyzer = self.config.get('analyzer', 'char')
        
    def fit(self, X: List[str]):
        """
        训练特征提取器
        Args:
            X: 查询文本列表
        """
        logger.info("开始训练意图特征提取器")
        max_features = self.config.get('max_features', 1000)
        ngram_range = self.config.get('ngram_range', (1, 2))
        # 确保ngram_range是元组类型
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        min_df = self.config.get('min_df', 1)
        analyzer = self.config.get('analyzer', self.analyzer)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            analyzer=analyzer
        )
        self.tfidf_vectorizer.fit(X)
        self.is_fitted = True
        logger.info("意图特征提取器训练完成")
        
    def transform(self, X: List[str]) -> np.ndarray:
        """
        提取特征
        
        Args:
            X: 查询文本列表
            
        Returns:
            特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练，请先调用fit方法")
        
        # 实现特征提取
        features = self.tfidf_vectorizer.transform(X)
        return features.toarray()
        
    def fit_transform(self, X: List[str]) -> np.ndarray:
        """
        训练并提取特征
        
        Args:
            X: 查询文本列表
            
        Returns:
            特征矩阵
        """
        self.fit(X)
        return self.transform(X)

class SlotFeatureExtractor:
    """槽位填充特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.is_fitted = False
        self.word_vocab = {}
        self.label_vocab = {}
        self.feature_dim = 0
        
        # 处理ngram_range参数（如果存在）
        if 'ngram_range' in self.config:
            ngram_range = self.config['ngram_range']
            # 确保ngram_range是元组类型
            if isinstance(ngram_range, list):
                self.config['ngram_range'] = tuple(ngram_range)
        
    def fit(self, sentences: List[List[str]], labels: List[List[str]]):
        """
        训练特征提取器
        
        Args:
            sentences: 句子列表
            labels: 标签列表
        """
        logger.info("开始训练槽位特征提取器")
        
        # 验证输入数据
        if not sentences or len(sentences) == 0:
            raise ValueError("句子列表不能为空")
        
        if not labels or len(labels) == 0:
            raise ValueError("标签列表不能为空")
        
        if len(sentences) != len(labels):
            raise ValueError(f"句子数量({len(sentences)})与标签数量({len(labels)})不一致")
        
        # 构建词汇表
        self._build_vocabulary(sentences)
        
        # 构建标签映射
        self._build_label_mapping(labels)
        
        # 计算特征维度
        self._calculate_feature_dim()
        
        self.is_fitted = True
        logger.info("槽位特征提取器训练完成")
        
    def transform(self, sentences: List[List[str]]) -> List[np.ndarray]:
        """
        提取特征
        
        Args:
            sentences: 句子列表
            
        Returns:
            特征列表，每个句子返回一个特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练，请先调用fit方法")
        
        features_list = []
        
        for sentence in sentences:
            sentence_features = []
            for i in range(len(sentence)):
                # 提取当前词的特征
                word_features = self._extract_word_features(sentence, i)
                
                # 提取位置特征
                position_features = self._extract_position_features(i, len(sentence))
                
                # 提取上下文特征
                context_features = self._extract_context_features(sentence, i)
                
                # 合并所有特征
                combined_features = np.concatenate([
                    word_features, 
                    position_features, 
                    context_features
                ])
                
                sentence_features.append(combined_features)
            
            # 转换为numpy数组
            sentence_features_array = np.array(sentence_features)
            features_list.append(sentence_features_array)
        
        return features_list
        
    def fit_transform(self, sentences: List[List[str]], labels: List[List[str]]) -> List[np.ndarray]:
        """
        训练并提取特征
        
        Args:
            sentences: 句子列表
            labels: 标签列表
            
        Returns:
            特征列表
        """
        self.fit(sentences, labels)
        return self.transform(sentences)
    
    def _build_vocabulary(self, sentences: List[List[str]]):
        """
        构建词汇表
        
        Args:
            sentences: 句子列表
        """
        word_freq = {}
        
        # 统计词频
        for sentence in sentences:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序，取前max_features个词
        max_features = self.config.get('max_features', 1000)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表映射
        self.word_vocab = {'<UNK>': 0}  # 未知词
        for i, (word, freq) in enumerate(sorted_words[:max_features-1]):
            self.word_vocab[word] = i + 1
    
    def _build_label_mapping(self, labels: List[List[str]]):
        """
        构建标签映射
        
        Args:
            labels: 标签列表
        """
        label_set = set()
        
        # 收集所有标签
        for label_seq in labels:
            for label in label_seq:
                label_set.add(label)
        
        # 构建标签映射
        self.label_vocab = {}
        for i, label in enumerate(sorted(label_set)):
            self.label_vocab[label] = i
    
    def _extract_word_features(self, sentence: List[str], position: int) -> np.ndarray:
        """
        提取词特征
        
        Args:
            sentence: 句子
            position: 词的位置
            
        Returns:
            词特征向量
        """
        # 当前词
        current_word = sentence[position] if position < len(sentence) else '<PAD>'
        current_word_id = self.word_vocab.get(current_word, 0)  # 0是<UNK>的ID
        
        # 创建one-hot特征向量
        word_features = np.zeros(len(self.word_vocab))
        word_features[current_word_id] = 1.0
        
        return word_features
    
    def _extract_position_features(self, position: int, sentence_length: int) -> np.ndarray:
        """
        提取位置特征
        
        Args:
            position: 词的位置
            sentence_length: 句子长度
            
        Returns:
            位置特征向量
        """
        # 相对位置特征
        relative_position = position / max(sentence_length, 1)
        
        # 绝对位置特征（one-hot编码）
        position_features = np.zeros(10)  # 固定10维位置特征
        
        # 将位置映射到10个区间
        position_bin = min(int(relative_position * 10), 9)
        position_features[position_bin] = 1.0
        
        return position_features
    
    def _calculate_feature_dim(self):
        """
        计算特征维度
        """
        # 词特征维度
        word_feature_dim = len(self.word_vocab)
        
        # 位置特征维度
        position_feature_dim = 10  # 固定位置特征维度
        
        # 上下文特征维度
        window_size = self.config.get('window_size', 2)
        context_feature_dim = window_size * 2 * len(self.word_vocab)  # 前后各window_size个词
        
        # 总特征维度
        self.feature_dim = word_feature_dim + position_feature_dim + context_feature_dim
    
    def _extract_context_features(self, sentence: List[str], position: int) -> np.ndarray:
        """
        提取上下文特征
        
        Args:
            sentence: 句子
            position: 词的位置
            
        Returns:
            上下文特征向量
        """
        window_size = self.config.get('window_size', 2)
        context_features = []
        
        # 提取前后window_size个词的特征
        for offset in range(-window_size, window_size + 1):
            if offset == 0:  # 跳过当前词
                continue
                
            context_pos = position + offset
            
            if 0 <= context_pos < len(sentence):
                context_word = sentence[context_pos]
                context_word_id = self.word_vocab.get(context_word, 0)
            else:
                context_word_id = 0  # <UNK>的ID
            
            # 为每个上下文位置创建one-hot特征
            word_features = np.zeros(len(self.word_vocab))
            word_features[context_word_id] = 1.0
            context_features.append(word_features)
        
        # 合并所有上下文特征
        if context_features:
            return np.concatenate(context_features)
        else:
            return np.zeros(window_size * 2 * len(self.word_vocab)) 