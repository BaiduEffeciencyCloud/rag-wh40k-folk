"""
高级特征提取器
实现多特征融合的意图识别特征提取
"""

import numpy as np
import json
import os
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class AdvancedFeature:
    """高级特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化高级特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.tfidf_vectorizer = None
        
        # 从配置中获取高级特征提取器配置
        self.advanced_config = config.get('advanced_feature', {})
        
        self.wh40k_keywords = self._load_wh40k_keywords()
        self.intent_keywords = self._get_intent_keywords()
        self.is_fitted = False
        
    def fit(self, queries: List[str]):
        """
        训练特征提取器
        
        Args:
            queries: 查询文本列表
        """
        if not queries:
            raise ValueError("查询列表不能为空")
            
        logger.info("开始训练高级特征提取器")
        
        # 从配置中获取TF-IDF参数
        tfidf_config = self.advanced_config.get('tfidf', {})
        max_features = tfidf_config.get('max_features', 1000)
        ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        min_df = tfidf_config.get('min_df', 1)
        analyzer = tfidf_config.get('analyzer', 'char')
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            analyzer=analyzer
        )
        
        self.tfidf_vectorizer.fit(queries)
        self.is_fitted = True
        
        logger.info("高级特征提取器训练完成")
        
    def transform(self, queries: List[str]) -> np.ndarray:
        """
        提取特征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练，请先调用fit方法")
            
        # 提取各类特征
        char_features = self._extract_char_features(queries)
        semantic_features = self._extract_semantic_features(queries)
        statistical_features = self._extract_statistical_features(queries)
        
        # 融合特征
        features = np.concatenate([
            char_features,
            semantic_features,
            statistical_features
        ], axis=1)
        
        return features
        
    def fit_transform(self, queries: List[str]) -> np.ndarray:
        """
        训练并提取特征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            特征矩阵
        """
        self.fit(queries)
        return self.transform(queries)
        
    def _extract_char_features(self, queries: List[str]) -> np.ndarray:
        """
        提取字符级TF-IDF特征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            字符级特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练")
            
        features = self.tfidf_vectorizer.transform(queries)
        return features.toarray()
        
    def _extract_semantic_features(self, queries: List[str]) -> np.ndarray:
        """
        提取语义特征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            语义特征矩阵
        """
        features = []
        
        # 从配置中获取标点符号配置
        punctuation_config = self.advanced_config.get('punctuation', {})
        question_exclamation = punctuation_config.get('question_exclamation', '？？!！')
        quotes = punctuation_config.get('quotes', '""''')
        brackets = punctuation_config.get('brackets', '（）【】')
        others = punctuation_config.get('others', '：；，。')
        
        for query in queries:
            query_lower = query.lower()
            
            # 1. WH40K关键词密度
            wh40k_match_count = sum(1 for kw in self.wh40k_keywords if kw.lower() in query_lower)
            keyword_density = wh40k_match_count / max(len(query.split()), 1)
            
            # 2. 意图关键词匹配分数
            intent_scores = {}
            for intent, keywords in self.intent_keywords.items():
                intent_scores[intent] = sum(1 for kw in keywords if kw in query_lower)
            
            # 3. 查询复杂度
            complexity = self._calculate_complexity(query)
            
            # 4. 句子结构特征
            structure_features = [
                len([c for c in query if c in question_exclamation]),  # 问号感叹号数
                len([c for c in query if c in quotes]),    # 引号数
                len([c for c in query if c in brackets]),  # 括号数
                len([c for c in query if c in others]),  # 其他标点数
            ]
            
            # 5. 长度特征
            length_features = [
                len(query),                    # 总字符数
                len(query.split()),            # 词数
                len(set(query.split())),       # 唯一词数
                np.mean([len(w) for w in query.split()]) if query.split() else 0,  # 平均词长
            ]
            
            semantic_feature = [
                keyword_density, 
                complexity
            ] + list(intent_scores.values()) + structure_features + length_features
            
            features.append(semantic_feature)
        
        return np.array(features)
        
    def _extract_statistical_features(self, queries: List[str]) -> np.ndarray:
        """
        提取统计特征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            统计特征矩阵
        """
        features = []
        
        # 从配置中获取统计特征配置
        statistical_config = self.advanced_config.get('statistical', {})
        long_word_threshold = statistical_config.get('long_word_threshold', 4)
        very_long_word_threshold = statistical_config.get('very_long_word_threshold', 6)
        
        # 从配置中获取标点符号配置
        punctuation_config = self.advanced_config.get('punctuation', {})
        all_punctuation = punctuation_config.get('all_punctuation', '，。！？；：""''（）【】')
        
        for query in queries:
            # 1. 字符统计
            char_stats = [
                len(query),                                    # 总字符数
                len([c for c in query if c.isalpha()]),       # 字母数
                len([c for c in query if c.isdigit()]),       # 数字数
                len([c for c in query if c.isspace()]),       # 空格数
                len([c for c in query if c in all_punctuation]),  # 标点符号数
            ]
            
            # 2. 词汇统计
            words = query.split()
            word_stats = [
                len(words),                                    # 词数
                len(set(words)),                              # 唯一词数
                len([w for w in words if len(w) > long_word_threshold]),        # 长词数
                len([w for w in words if len(w) > very_long_word_threshold]),        # 超长词数
                np.mean([len(w) for w in words]) if words else 0,  # 平均词长
                np.std([len(w) for w in words]) if len(words) > 1 else 0,  # 词长标准差
            ]
            
            # 3. 重复度统计
            repetition_stats = [
                len(words) - len(set(words)) if words else 0,  # 重复词数
                (len(words) - len(set(words))) / max(len(words), 1),  # 重复率
            ]
            
            # 4. 数字统计
            numbers = [c for c in query if c.isdigit()]
            number_stats = [
                len(numbers),                                  # 数字字符数
                len(set(numbers)),                            # 唯一数字数
            ]
            
            statistical_feature = char_stats + word_stats + repetition_stats + number_stats
            features.append(statistical_feature)
        
        return np.array(features)
        
    def _load_wh40k_keywords(self) -> List[str]:
        """
        加载WH40K关键词
        
        Returns:
            WH40K关键词列表
        """
        try:
            keywords = []
            # 从配置中获取词汇目录
            vocab_dir = self.advanced_config.get('vocab_dir', "dict/wh40k_vocabulary")
            
            if not os.path.exists(vocab_dir):
                # 如果目录不存在，返回基本关键词
                return self.advanced_config.get('basic_keywords', ['战锤', '40k', 'wh40k'])
            
            # 加载所有JSON文件
            for filename in os.listdir(vocab_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(vocab_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 提取所有词汇（包括同义词）
                            for term, info in data.items():
                                keywords.append(term)
                                if 'synonyms' in info:
                                    keywords.extend(info['synonyms'])
                    except Exception as e:
                        logger.warning(f"加载词汇文件 {filepath} 失败: {e}")
            
            # 添加基本关键词
            basic_keywords = self.advanced_config.get('basic_keywords', ['战锤', '40k', 'wh40k'])
            keywords.extend(basic_keywords)
            
            return keywords
        except Exception as e:
            logger.error(f"加载WH40K词汇失败: {e}")
            return self.advanced_config.get('basic_keywords', ['战锤', '40k', 'wh40k'])
        
    def _get_intent_keywords(self) -> Dict[str, List[str]]:
        """
        获取意图关键词
        
        Returns:
            意图关键词字典
        """
        # 从配置中获取意图关键词
        intent_keywords_config = self.advanced_config.get('intent_keywords', {})
        
        # 如果配置中没有，返回默认值
        if not intent_keywords_config:
            return {
                'compare': [
                    '哪个', '相比', '比较', '更', '最高', '最低', '哪个更高', '哪个更远',
                    '哪个更好', '哪个更强', '哪个更快', '哪个更准', '哪个更远'
                ],
                'list': [
                    '有哪些', '多少个', '列举', '哪些', '多少种', '什么类型',
                    '什么种类', '什么单位', '什么武器', '什么技能'
                ],
                'query': [
                    '什么是', '如何', '解释', '说明', '定义', '是什么',
                    '什么意思', '怎么用', '怎么触发', '怎么计算','是啥'
                ],
                'rule': [
                    '什么情况下', '如何触发', '规则', '何时', '条件', '当...时',
                    '什么时候', '什么条件', '什么要求', '什么限制'
                ]
            }
        
        return intent_keywords_config
        
    def _calculate_complexity(self, query: str) -> float:
        """
        计算查询复杂度
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度分数
        """
        words = query.split()
        if not words:
            return 0.0
        
        # 从配置中获取复杂度计算参数
        complexity_config = self.advanced_config.get('complexity', {})
        max_sentence_length = complexity_config.get('max_sentence_length', 20.0)
        long_word_threshold = complexity_config.get('long_word_threshold', 4)
        very_long_word_threshold = complexity_config.get('very_long_word_threshold', 6)
        punctuation_threshold = complexity_config.get('punctuation_threshold', 10.0)
        weight_length = complexity_config.get('weight_length', 0.25)
        weight_unique = complexity_config.get('weight_unique', 0.25)
        weight_long_word = complexity_config.get('weight_long_word', 0.25)
        weight_punctuation = complexity_config.get('weight_punctuation', 0.25)
        
        # 从配置中获取标点符号
        punctuation_config = self.advanced_config.get('punctuation', {})
        all_punctuation = punctuation_config.get('all_punctuation', '，。！？；：""''（）【】')
        
        # 1. 句子长度复杂度
        length_complexity = min(len(words) / max_sentence_length, 1.0)
        
        # 2. 词汇多样性复杂度
        unique_ratio = len(set(words)) / len(words)
        
        # 3. 长词复杂度
        long_word_ratio = len([w for w in words if len(w) > long_word_threshold]) / len(words)
        
        # 4. 标点符号复杂度
        punctuation_count = len([c for c in query if c in all_punctuation])
        punctuation_complexity = min(punctuation_count / punctuation_threshold, 1.0)
        
        # 综合复杂度
        complexity = (weight_length * length_complexity + 
                     weight_unique * unique_ratio + 
                     weight_long_word * long_word_ratio + 
                     weight_punctuation * punctuation_complexity)
        
        return complexity 