#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced特征提取器
"""

import os
import sys
import warnings

# 抑制torch相关的FutureWarning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import json
import logging
import numpy as np
import joblib
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseFeatureExtractor
from .complexity import ComplexityCalculator

logger = logging.getLogger(__name__)


class AdvancedFeature(BaseFeatureExtractor):
    """高级特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化高级特征提取器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.config = config  # 添加config属性以保持向后兼容
        self.advanced_config = config.get('advanced_feature', {})
        self.tfidf_vectorizer = None
        self.wh40k_keywords = self._load_wh40k_keywords()
        self.intent_keywords = self._get_intent_keywords()
        self.is_fitted = False
        
        # 初始化复杂度计算器
        self.complexity_calculator = ComplexityCalculator(config)
        
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
            complexity = self.complexity_calculator.calculate(query)
            
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
        
    def save(self, path: str):
        """保存特征提取器"""
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存所有必要的组件
        save_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'wh40k_keywords': self.wh40k_keywords,
            'intent_keywords': self.intent_keywords,
            'advanced_config': self.advanced_config
        }
        
        joblib.dump(save_data, path)
        logger.info(f"高级特征提取器已保存到: {path}")
    
    def load(self, path: str):
        """加载特征提取器"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征提取器文件不存在: {path}")
        
        # 加载所有组件
        save_data = joblib.load(path)
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        self.wh40k_keywords = save_data['wh40k_keywords']
        self.intent_keywords = save_data['intent_keywords']
        self.advanced_config = save_data['advanced_config']
        self.is_fitted = True
        
        logger.info(f"高级特征提取器已从 {path} 加载")
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练")
        
        # 使用实际的测试数据来计算特征维度
        test_query = ["测试查询"]
        features = self.transform(test_query)
        return features.shape[1]
    
    # ========== 新增句向量支持方法骨架 ==========
    
    def get_sentence_embedding(self, sentences: List[str]) -> np.ndarray:
        """获取批量句向量特征（仅训练时使用）"""
        if not hasattr(self, 'sentence_model'):
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('shibing624/text2vec-base-chinese')
                logger.info("句向量模型加载成功")
            except ImportError:
                logger.warning("sentence-transformers未安装，使用零向量代替")
                return np.zeros((len(sentences), 768))
            except Exception as e:
                logger.error(f"句向量模型加载失败: {e}")
                return np.zeros((len(sentences), 768))
        
        # 处理空列表
        if len(sentences) == 0:
            return np.zeros((0, 768))
        
        try:
            logger.info(f"开始提取批量句向量特征，样本数: {len(sentences)}")
            embeddings = self.sentence_model.encode(sentences, batch_size=64, show_progress_bar=True)
            logger.info(f"批量句向量提取成功，维度: {embeddings.shape}")
            return embeddings  # 返回(N, 768)矩阵
        except Exception as e:
            logger.error(f"批量句向量提取失败: {e}")
            return np.zeros((len(sentences), 768))
    
    def get_enhanced_intent_features(self, sentence: str) -> np.ndarray:
        """双路特征融合"""
        try:
            logger.info(f"开始ad_feature句向量融合，文本: {sentence[:30]}...")
            
            # 1. 原有特征
            original_features = self._extract_semantic_features([sentence])
            logger.info(f"原有特征维度: {original_features.shape}")
            
            # 2. 句向量特征
            sentence_embedding = self.get_sentence_embedding(sentence)
            logger.info(f"句向量维度: {sentence_embedding.shape}")
            
            # 3. 根据配置选择融合方法
            fusion_method = self.advanced_config.get('enhanced', {}).get('fusion_method', 'concatenate')
            
            if fusion_method == 'weighted':
                # 加权融合：给句向量更高权重
                sentence_weight = 2.0  # 句向量权重
                original_weight = 1.0  # 原有特征权重
                
                # 加权融合
                weighted_original = original_features * original_weight
                weighted_sentence = sentence_embedding.reshape(1, -1) * sentence_weight
                logger.info(f"加权融合: original={weighted_original.shape}, sentence={weighted_sentence.shape}")
                
                # 拼接加权特征
                combined_features = np.concatenate([weighted_original, weighted_sentence], axis=1)
                logger.info(f"ad_feature句向量融合成功，最终维度: {combined_features.shape}")
            else:
                # 默认拼接融合
                combined_features = np.concatenate([original_features, sentence_embedding.reshape(1, -1)], axis=1)
                logger.info(f"ad_feature句向量融合成功，最终维度: {combined_features.shape}")
            
            return combined_features
        except Exception as e:
            logger.error(f"ad_feature特征融合失败: {e}")
            # 返回原有特征作为兜底
            return self._extract_semantic_features([sentence])
    

    
    def save_enhanced_feature_extractor(self, path: str):
        """保存增强的特征提取器"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存所有必要的组件
            save_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'wh40k_keywords': self.wh40k_keywords,
                'intent_keywords': self.intent_keywords,
                'advanced_config': self.advanced_config
            }
            
            # 添加可选组件
            if hasattr(self, 'intent_classifier'):
                save_data['intent_classifier'] = self.intent_classifier
            
            if hasattr(self, 'enhanced_feature_extractor'):
                save_data['enhanced_feature_extractor'] = self.enhanced_feature_extractor
            
            joblib.dump(save_data, path)
            logger.info(f"增强特征提取器已保存到: {path}")
            
        except Exception as e:
            logger.error(f"增强特征提取器保存失败: {e}")
            raise
    
    def load_enhanced_feature_extractor(self, path: str):
        """加载增强的特征提取器"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"增强特征提取器文件不存在: {path}")
            
            # 加载所有组件
            save_data = joblib.load(path)
            self.tfidf_vectorizer = save_data['tfidf_vectorizer']
            self.wh40k_keywords = save_data['wh40k_keywords']
            self.intent_keywords = save_data['intent_keywords']
            self.advanced_config = save_data['advanced_config']
            
            # 加载可选组件
            if 'intent_classifier' in save_data:
                self.intent_classifier = save_data['intent_classifier']
            
            if 'enhanced_feature_extractor' in save_data:
                self.enhanced_feature_extractor = save_data['enhanced_feature_extractor']
            
            self.is_fitted = True
            
            logger.info(f"增强特征提取器已从 {path} 加载")
            
        except Exception as e:
            logger.error(f"增强特征提取器加载失败: {e}")
            raise
    
    def _calculate_complexity(self, query: str) -> float:
        """
        计算查询复杂度（向后兼容方法）
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度分数
        """
        return self.complexity_calculator.calculate(query)
    
 