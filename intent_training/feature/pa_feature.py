#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置感知特征提取器 (Position Aware Feature Extractor)
实现位置权重匹配、上下文感知匹配、分层关键词体系等优化功能
核心逻辑：通过位置权重和上下文感知来区分Query/List/Compare意图
"""

import logging
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import jieba
from .base import BaseFeatureExtractor
from .complexity import ComplexityCalculator

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor(BaseFeatureExtractor):
    """增强特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强特征提取器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.config = config
        self.tfidf_vectorizer = None
        self.primary_keywords = self._init_primary_keywords()
        self.secondary_keywords = self._init_secondary_keywords()
        self.negative_keywords = self._init_negative_keywords()
        
        # 初始化复杂度计算器
        self.complexity_calculator = ComplexityCalculator(config)
        
        # 从配置文件读取evilgood_feature配置
        self._init_evilgood_feature_from_config()
        
    def _init_primary_keywords(self) -> Dict[str, List[str]]:
        """初始化主要关键词 - 从config.yaml读取intent_keywords"""
        try:
            # 从config.yaml读取intent_keywords
            intent_keywords = self.config.get('advanced_feature', {}).get('intent_keywords', {})
            
            # 验证配置是否存在
            if not intent_keywords:
                raise ValueError("配置中缺少intent_keywords")
            
            # 验证配置格式是否正确
            if not isinstance(intent_keywords, dict):
                raise ValueError("intent_keywords必须是字典格式")
            
            # 验证必需的意图键是否存在
            required_intents = ['query', 'list', 'compare', 'rule']
            for intent in required_intents:
                if intent not in intent_keywords:
                    raise ValueError(f"配置中缺少必需的意图键: {intent}")
                if not isinstance(intent_keywords[intent], list):
                    raise ValueError(f"意图键 {intent} 的值必须是列表格式")
                if len(intent_keywords[intent]) == 0:
                    raise ValueError(f"意图键 {intent} 的关键词列表不能为空")
            
            logger.info(f"成功从config.yaml读取intent_keywords: {list(intent_keywords.keys())}")
            return intent_keywords
            
        except Exception as e:
            logger.error(f"读取intent_keywords失败: {e}")
            raise
        
    def _init_secondary_keywords(self) -> Dict[str, List[str]]:
        """初始化次要关键词（移除辅助词汇）"""
        return {
            'Query': ['查询', '了解', '知道'],  # 移除辅助词汇
            'List': ['查看', '浏览', '展示', '显示', '列出', '提供', '给出'],  # 移除辅助词汇
            'Compare': ['对比', '差异', '区别', '不同', '相似', '相同'],
            'Rule': ['规定', '约束', '限制', '允许', '禁止', '可以', '不能']
        }
        
    def _init_negative_keywords(self) -> Dict[str, List[str]]:
        """初始化否定关键词"""
        return {
            'Query': ['不是', '没有', '不包含', '不包含'],
            'List': ['不显示', '隐藏', '排除', '除了'],
            'Compare': ['不比', '不如', '不高于', '不低于'],
            'Rule': ['不要求', '不限制', '不必须', '不需要']
        }
        
    def _init_malicious_patterns(self) -> List[str]:
        """初始化恶意词汇模式"""
        return [
            '我不想知道', '不要告诉我', '别说出来', '不想知道',
            '别说了', '闭嘴', '不要回答', '别回答',
            '我不想听', '不要解释', '别解释', '不想了解',
            '不要说明', '别说明', '不想查询', '不要查询'
        ]
        
    def _init_helper_patterns(self) -> List[str]:
        """初始化辅助词汇模式"""
        return [
            '我想知道', '告诉我', '回答我', '请回答',
            '帮我查一下', '能否说明', '请解释', '帮我看看',
            '能否告诉我', '请告诉我', '帮我查询', '能否查询'
        ]
        
    def _contains_malicious_patterns(self, text: str) -> bool:
        """检测是否包含恶意词汇"""
        malicious_patterns = getattr(self, 'malicious_patterns', None)
        if malicious_patterns is None:
            malicious_patterns = self._init_malicious_patterns()
        return any(pattern in text for pattern in malicious_patterns)
        
    def _contains_helper_patterns(self, text: str) -> bool:
        """检测是否包含辅助词汇"""
        helper_patterns = getattr(self, 'helper_patterns', None)
        if helper_patterns is None:
            helper_patterns = self._init_helper_patterns()
        return any(pattern in text for pattern in helper_patterns)
        
    def _get_malicious_weight(self, text: str) -> float:
        """计算恶意词汇权重（兼容旧版本）"""
        return self._get_malicious_adjustment_coefficient(text)
        
    def _get_helper_context_weight(self, text: str, keyword: str, pos: int) -> float:
        """计算辅助词汇上下文权重（兼容旧版本）"""
        return self._get_helper_adjustment_coefficient(text, keyword, pos)
        
    def position_weighted_matching(self, query: str, keywords: List[str]) -> float:
        """位置权重匹配"""
        matches = []
        for keyword in keywords:
            if keyword in query:
                pos = query.find(keyword)
                
                # 检查是否被辅助词汇包围
                helper_weight = self._get_helper_context_weight(query, keyword, pos)
                
                # 如果被辅助词汇包围，降低权重
                if helper_weight < 1.0:
                    weight = helper_weight
                else:
                    # 正常位置权重逻辑
                    if pos < len(query) * 0.2:
                        weight = 0.8
                    elif pos > len(query) * 0.8:
                        weight = 0.9
                    else:
                        weight = 1.2
                
                matches.append(weight)
        return sum(matches) if matches else 0.0
        
    def context_aware_matching(self, query: str, keywords: List[str]) -> float:
        """上下文感知匹配"""
        matches = []
        for keyword in keywords:
            if keyword in query:
                # 检查关键词前后的词
                pos = query.find(keyword)
                before = query[max(0, pos-4):pos]
                after = query[pos+len(keyword):pos+len(keyword)+4]
                
                # 根据上下文调整权重
                context_weight = 1.0
                if any(word in before for word in ['哪个', '什么', '如何', '为什么']):
                    context_weight *= 1.3  # 疑问词增强权重
                if any(word in after for word in ['的', '值', '属性', '能力', '技能','武器','关键字','技能']):
                    context_weight *= 1.2  # 属性词增强权重
                matches.append(context_weight)
        return sum(matches) if matches else 0.0
        
    def fuzzy_keyword_matching(self, query: str, keywords: List[str]) -> float:
        """模糊匹配"""
        matches = []
        for keyword in keywords:
            # 计算相似度
            max_similarity = 0.0
            for i in range(len(query) - len(keyword) + 1):
                similarity = SequenceMatcher(None, query[i:i+len(keyword)], keyword).ratio()
                max_similarity = max(max_similarity, similarity)
            if max_similarity > 0.8:  # 相似度阈值
                matches.append(max_similarity)
        return sum(matches) if matches else 0.0
        
    def extract_structural_features(self, text: str) -> np.ndarray:
        """提取结构特征"""
        features = []
        
        # 恶意词汇检测 - 使用配置驱动的调整系数
        malicious_coefficient = self._get_malicious_adjustment_coefficient(text)
        
        # 疑问词特征
        question_words = ['什么', '如何', '为什么', '哪个', '怎么', '哪些']
        question_count = sum(1 for word in question_words if word in text)
        # 如果恶意词汇存在，降低疑问词权重
        if malicious_coefficient < 1.0:
            question_count *= malicious_coefficient
        features.append(question_count)
        
        # 数量词特征
        quantity_words = ['哪些', '多少个', '多少种', '几个', '多少']
        quantity_count = sum(1 for word in quantity_words if word in text)
        # 如果恶意词汇存在，降低数量词权重
        if malicious_coefficient < 1.0:
            quantity_count *= malicious_coefficient
        features.append(quantity_count)
        
        # 比较词特征
        comparison_words = ['哪个', '相比', '比较', '更', '最高', '最低', '最远', '最近']
        comparison_count = sum(1 for word in comparison_words if word in text)
        # 如果恶意词汇存在，降低比较词权重
        if malicious_coefficient < 1.0:
            comparison_count *= malicious_coefficient
        features.append(comparison_count)
        
        # 句子长度特征
        sentence_length = len(text)
        # 如果恶意词汇存在，降低句子长度权重
        if malicious_coefficient < 1.0:
            sentence_length *= malicious_coefficient
        features.append(sentence_length)
        
        # 词数特征
        word_count = len(list(jieba.cut(text)))
        # 如果恶意词汇存在，降低词数权重
        if malicious_coefficient < 1.0:
            word_count *= malicious_coefficient
        features.append(word_count)
        
        # 标点符号特征
        punctuation_count = sum(1 for char in text if char in '，。！？；：""''（）【】')
        # 如果恶意词汇存在，降低标点符号权重
        if malicious_coefficient < 1.0:
            punctuation_count *= malicious_coefficient
        features.append(punctuation_count)
        
        return np.array(features)
        
    def extract_interaction_features(self, char_features: np.ndarray, 
                                   semantic_features: np.ndarray, 
                                   statistical_features: np.ndarray) -> np.ndarray:
        """提取特征交互"""
        # 获取最小维度进行交互
        min_dim = min(char_features.shape[1], semantic_features.shape[1], statistical_features.shape[1])
        
        # 截取到相同维度进行交互
        char_subset = char_features[:, :min_dim]
        semantic_subset = semantic_features[:, :min_dim]
        statistical_subset = statistical_features[:, :min_dim]
        
        # 特征交互
        char_semantic_interaction = char_subset * semantic_subset  # 字符-语义交互
        char_statistical_interaction = char_subset * statistical_subset  # 字符-统计交互
        semantic_statistical_interaction = semantic_subset * statistical_subset  # 语义-统计交互
        
        # 拼接交互特征
        interaction_features = np.concatenate([
            char_semantic_interaction,
            char_statistical_interaction,
            semantic_statistical_interaction
        ], axis=1)
        
        return interaction_features
        
    def hierarchical_fusion(self, char_features: np.ndarray, 
                           semantic_features: np.ndarray, 
                           statistical_features: np.ndarray,
                           complexity_features: np.ndarray = None,
                           original_texts: List[str] = None) -> np.ndarray:
        """分层特征融合"""
        # 检查配置决定是否使用句向量
        enhanced_config = self.config.get('advanced_feature', {}).get('enhanced', {})
        use_sentence_embedding = enhanced_config.get('use_sentence_embedding', False)
        
        if use_sentence_embedding and original_texts:
            # 批量句向量融合路径
            try:
                # 获取批量句向量
                sentence_embeddings = self.get_sentence_embedding(original_texts)
                
                # 调用批量句向量融合方法 - 传递复杂度特征
                return self._sentence_embedding_fusion(
                    char_features, semantic_features, statistical_features, sentence_embeddings, complexity_features, original_texts
                )
            except Exception as e:
                logger.warning(f"批量句向量融合失败，回退到原有逻辑: {e}")
                # 回退到原有逻辑 - 传递复杂度特征
                return self._original_hierarchical_fusion(
                    char_features, semantic_features, statistical_features, complexity_features, original_texts
                )
        else:
            # 原有融合路径 - 保持现有逻辑不变
            return self._original_hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features, original_texts
            )
    
    def _original_hierarchical_fusion(self, char_features: np.ndarray, 
                                     semantic_features: np.ndarray, 
                                     statistical_features: np.ndarray,
                                     complexity_features: np.ndarray = None,
                                     original_texts: List[str] = None) -> np.ndarray:
        """原有的分层融合逻辑"""
        # 获取pa_feature_weights配置
        pa_weights = self.config.get('advanced_feature', {}).get('pa_feature_weights', {})
        char_weight = pa_weights.get('char_weight', 1.0)
        semantic_weight = pa_weights.get('semantic_weight', 1.0)
        statistical_weight = pa_weights.get('statistical_weight', 1.0)
        complexity_weight = pa_weights.get('complexity_weight', 1.0)
        
        # 应用恶意词权重（如果提供了原始文本）
        malicious_coefficient = 1.0
        if original_texts and len(original_texts) > 0:
            # 使用第一个文本作为代表（批量处理时通常所有文本都是同一类型）
            malicious_coefficient = self._get_malicious_adjustment_coefficient(original_texts[0])
            if malicious_coefficient < 1.0:
                logger.info(f"检测到恶意词，应用恶意词权重: {malicious_coefficient}")
        
        # 记录权重应用
        logger.info(f"应用pa_feature权重: char={char_weight}, semantic={semantic_weight}, statistical={statistical_weight}, complexity={complexity_weight}, malicious={malicious_coefficient}")
        
        # 应用权重（包括恶意词权重）
        weighted_char_features = char_features * char_weight * malicious_coefficient
        weighted_semantic_features = semantic_features * semantic_weight * malicious_coefficient
        weighted_statistical_features = statistical_features * statistical_weight * malicious_coefficient
        
        # 第一层：语义特征与统计特征融合
        semantic_statistical = np.concatenate([weighted_semantic_features, weighted_statistical_features], axis=1)
        
        # 第二层：与字符特征融合
        final_features = np.concatenate([weighted_char_features, semantic_statistical], axis=1)
        
        # 第三层：与句子复杂度特征融合（如果启用）
        if complexity_features is not None and complexity_features.shape[1] > 0:
            weighted_complexity_features = complexity_features * complexity_weight * malicious_coefficient
            final_features = np.concatenate([final_features, weighted_complexity_features], axis=1)
        
        return final_features
        
    def _extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """提取语义特征"""
        features = []
        for text in texts:
            text_features = []
            
            # 主要关键词匹配
            for intent, keywords in self.primary_keywords.items():
                primary_score = self.position_weighted_matching(text, keywords)
                text_features.append(primary_score)
            
            # 次要关键词匹配
            for intent, keywords in self.secondary_keywords.items():
                secondary_score = self.context_aware_matching(text, keywords)
                text_features.append(secondary_score)
            
            # 否定关键词匹配
            for intent, keywords in self.negative_keywords.items():
                negative_score = self.fuzzy_keyword_matching(text, keywords)
                text_features.append(negative_score)
            

            
            features.append(text_features)
        
        return np.array(features)
        
    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """提取统计特征"""
        features = []
        for text in texts:
            text_features = self.extract_structural_features(text)
            features.append(text_features)
        return np.array(features)
        
    def _extract_complexity_features(self, texts: List[str]) -> np.ndarray:
        """
        提取句子复杂度特征（独立特征层）
        
        Args:
            texts: 文本列表
            
        Returns:
            句子复杂度特征矩阵
        """
        # 检查是否启用句子复杂度特征
        complexity_config = self.config.get('advanced_feature', {}).get('complexity_feature', {})
        enabled = complexity_config.get('enabled', True)  # 默认启用
        
        if not enabled:
            return np.array([]).reshape(len(texts), 0)
        
        features = []
        for text in texts:
            complexity = self.complexity_calculator.calculate(text)
            features.append([complexity])
        
        return np.array(features)

    def fit(self, texts: List[str], labels: List[str] = None):
        """训练特征提取器"""
        # 初始化TF-IDF向量器
        tfidf_config = self.config.get('advanced_feature', {}).get('tfidf', {})
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 1000),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
            min_df=tfidf_config.get('min_df', 1),
            max_df=tfidf_config.get('max_df', 0.8),
            sublinear_tf=True,
            norm='l2'
        )
        
        # 训练TF-IDF
        self.tfidf_vectorizer.fit(texts)
        logger.info("增强特征提取器训练完成")
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """转换文本为特征向量"""
        if self.tfidf_vectorizer is None:
            raise ValueError("特征提取器尚未训练")
        
        # 提取TF-IDF特征
        char_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # 提取语义特征
        semantic_features = self._extract_semantic_features(texts)
        
        # 提取统计特征
        statistical_features = self._extract_statistical_features(texts)
        
        # 提取句子复杂度特征
        complexity_features = self._extract_complexity_features(texts)
        
        # 特征融合 - 总是传递原始文本以确保恶意词权重能够应用
        enhanced_config = self.config.get('advanced_feature', {}).get('enhanced', {})
        use_sentence_embedding = enhanced_config.get('use_sentence_embedding', False)
        
        if use_sentence_embedding and texts:
            fused_features = self.hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features,
                original_texts=texts
            )
        else:
            # 即使没有启用句向量，也要传递original_texts以确保恶意词权重能够应用
            fused_features = self.hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features,
                original_texts=texts
            )
        
        return fused_features
    
    def get_sentence_embedding(self, sentences: List[str]) -> np.ndarray:
        """获取批量句向量特征"""
        if not hasattr(self, 'sentence_model'):
            try:
                from sentence_transformers import SentenceTransformer
                enhanced_config = self.config.get('advanced_feature', {}).get('enhanced', {})
                sentence_model = enhanced_config.get('sentence_model', 'shibing624/text2vec-base-chinese')
                
                # 从配置中获取网络设置
                network_timeout = enhanced_config.get('network_timeout', 30)
                allow_offline = enhanced_config.get('allow_offline', True)
                fallback_to_zero = enhanced_config.get('fallback_to_zero', True)
                
                # 设置环境变量
                import os
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(network_timeout)
                os.environ['HF_HUB_OFFLINE'] = '1' if allow_offline else '0'
                
                # 尝试加载模型
                self.sentence_model = SentenceTransformer(sentence_model)
                logger.info("句向量模型加载成功")
                
            except ImportError:
                logger.warning("sentence-transformers未安装，使用零向量代替")
                return np.zeros((len(sentences), 768))
            except Exception as e:
                logger.error(f"句向量模型加载失败: {e}")
                if fallback_to_zero:
                    logger.warning("将使用零向量代替句向量特征")
                    # 设置一个标志，避免重复尝试加载
                    self.sentence_model = None
                    return np.zeros((len(sentences), 768))
                else:
                    # 如果不允许回退，则抛出异常
                    raise e
        
        # 如果模型加载失败，直接返回零向量
        if self.sentence_model is None:
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
            if fallback_to_zero:
                logger.warning("使用零向量作为兜底")
                return np.zeros((len(sentences), 768))
            else:
                # 如果不允许回退，则抛出异常
                raise e
    
    def get_enhanced_intent_features(self, sentence: str) -> np.ndarray:
        """获取增强的意图特征（包含句向量）"""
        try:
            # 提取TF-IDF特征
            char_features = self.tfidf_vectorizer.transform([sentence]).toarray()
            
            # 提取语义特征
            semantic_features = self._extract_semantic_features([sentence])
            
            # 提取统计特征
            statistical_features = self._extract_statistical_features([sentence])
            
            # 提取句子复杂度特征
            complexity_features = self._extract_complexity_features([sentence])
            
            # 特征融合 - 传递原始文本和复杂度特征
            fused_features = self.hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features, original_texts=[sentence]
            )
            
            return fused_features
        except Exception as e:
            logger.error(f"增强意图特征提取失败: {e}")
            # 返回基础特征作为兜底
            return self.tfidf_vectorizer.transform([sentence]).toarray()
    
    def _sentence_embedding_fusion(self, char_features: np.ndarray,
                                  semantic_features: np.ndarray,
                                  statistical_features: np.ndarray,
                                  sentence_embeddings: np.ndarray,
                                  complexity_features: np.ndarray = None,
                                  original_texts: List[str] = None) -> np.ndarray:
        """批量句向量融合方法"""
        try:
            logger.info(f"开始批量句向量融合，特征维度: char={char_features.shape}, semantic={semantic_features.shape}, statistical={statistical_features.shape}, complexity={complexity_features.shape if complexity_features is not None else 'None'}")
            logger.info(f"批量句向量维度: {sentence_embeddings.shape}")
            
            # 获取pa_feature_weights配置
            pa_weights = self.config.get('advanced_feature', {}).get('pa_feature_weights', {})
            sentence_weight = pa_weights.get('sentence_weight', 0.3)
            
            logger.info(f"pa_feature句向量融合权重: sentence_weight={sentence_weight}")
            
            # 原有特征融合 - 传递original_texts以确保恶意词权重能够应用
            original_features = self._original_hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features, original_texts
            )
            logger.info(f"原有特征融合完成，维度: {original_features.shape}")
            
            # 对句向量应用权重
            weighted_sentence = sentence_embeddings * sentence_weight
            
            logger.info(f"加权融合: original={original_features.shape}, sentence={weighted_sentence.shape}")
            
            # 拼接加权特征
            combined_features = np.concatenate([original_features, weighted_sentence], axis=1)
            logger.info(f"批量句向量融合成功，最终维度: {combined_features.shape}")
            
            return combined_features
        except Exception as e:
            logger.error(f"批量句向量融合失败: {e}")
            # 回退到原有融合 - 传递original_texts以确保恶意词权重能够应用
            return self._original_hierarchical_fusion(
                char_features, semantic_features, statistical_features, complexity_features, original_texts
            )
        
    def save(self, filepath: str):
        """保存特征提取器"""
        save_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'primary_keywords': self.primary_keywords,
            'secondary_keywords': self.secondary_keywords,
            'negative_keywords': self.negative_keywords,
            'config': self.config
        }
        joblib.dump(save_data, filepath)
        logger.info(f"增强特征提取器已保存到: {filepath}")
        
    def load(self, filepath: str):
        """加载特征提取器"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        save_data = joblib.load(filepath)
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        self.primary_keywords = save_data['primary_keywords']
        self.secondary_keywords = save_data['secondary_keywords']
        self.negative_keywords = save_data['negative_keywords']
        self.config = save_data['config']
        logger.info(f"增强特征提取器已从 {filepath} 加载")
    
    def save_enhanced_feature_extractor(self, path: str):
        """保存增强的特征提取器"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存所有必要的组件，包括恶意词相关的方法
            save_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'primary_keywords': self.primary_keywords,
                'secondary_keywords': self.secondary_keywords,
                'negative_keywords': self.negative_keywords,
                'config': self.config,
                # 添加恶意词相关的方法
                'malicious_patterns': self._init_malicious_patterns(),
                'helper_patterns': self._init_helper_patterns()
            }
            
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
            self.primary_keywords = save_data['primary_keywords']
            self.secondary_keywords = save_data['secondary_keywords']
            self.negative_keywords = save_data['negative_keywords']
            self.config = save_data['config']
            
            # 加载恶意词相关的方法（如果存在）
            if 'malicious_patterns' in save_data:
                self.malicious_patterns = save_data['malicious_patterns']
            else:
                # 兼容旧版本，重新初始化恶意词模式
                self.malicious_patterns = self._init_malicious_patterns()
                
            if 'helper_patterns' in save_data:
                self.helper_patterns = save_data['helper_patterns']
            else:
                # 兼容旧版本，重新初始化辅助词模式
                self.helper_patterns = self._init_helper_patterns()
            
            logger.info(f"增强特征提取器已从 {path} 加载")
            
        except Exception as e:
            logger.error(f"增强特征提取器加载失败: {e}")
            raise
    
    def fit_transform(self, queries: List[str]) -> np.ndarray:
        """
        训练并提取特征（包装现有方法）
        
        Args:
            queries: 查询文本列表
            
        Returns:
            特征矩阵
        """
        self.fit(queries)
        return self.transform(queries)
    
    def get_feature_dim(self) -> int:
        """
        获取特征维度（包装现有方法）
        
        Returns:
            特征维度
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("特征提取器尚未训练")
        
        # 使用测试数据计算特征维度
        test_query = ["测试查询"]
        features = self.transform(test_query)
        return features.shape[1]


    def _init_evilgood_feature_from_config(self):
        """从配置文件读取evilgood_feature配置"""
        try:
            evilgood_config = self.config.get('advanced_feature', {}).get('evilgood_feature', {})
            
            # 读取调整系数
            self.malicious_adjustment_coefficient = evilgood_config.get('malicious_weight', 0.1)
            self.helper_adjustment_coefficient = evilgood_config.get('helper_weight', 0.3)
            
            # 读取模式列表
            self.malicious_patterns = evilgood_config.get('malicious_patterns', self._init_malicious_patterns())
            self.helper_patterns = evilgood_config.get('helper_patterns', self._init_helper_patterns())
            
            logger.info(f"成功读取evilgood_feature配置: malicious_weight={self.malicious_adjustment_coefficient}, helper_weight={self.helper_adjustment_coefficient}")
            logger.info(f"恶意词模式数量: {len(self.malicious_patterns)}, 辅助词模式数量: {len(self.helper_patterns)}")
            
        except Exception as e:
            logger.warning(f"读取evilgood_feature配置失败，使用默认值: {e}")
            # 使用默认值
            self.malicious_adjustment_coefficient = 0.1
            self.helper_adjustment_coefficient = 0.3
            self.malicious_patterns = self._init_malicious_patterns()
            self.helper_patterns = self._init_helper_patterns()
        
    def _get_malicious_adjustment_coefficient(self, text: str) -> float:
        """从配置文件获取恶意词调整系数"""
        # 确保属性已初始化
        if not hasattr(self, 'malicious_adjustment_coefficient'):
            self._init_evilgood_feature_from_config()
        
        if hasattr(self, 'malicious_patterns') and self.malicious_patterns:
            for pattern in self.malicious_patterns:
                if pattern in text:
                    logger.info(f"检测到恶意词模式: {pattern}, 应用权重: {self.malicious_adjustment_coefficient}")
                    return self.malicious_adjustment_coefficient
        return 1.0
        
    def _get_helper_adjustment_coefficient(self, text: str, keyword: str, pos: int) -> float:
        """从配置文件获取辅助词调整系数"""
        # 确保属性已初始化
        if not hasattr(self, 'helper_adjustment_coefficient'):
            self._init_evilgood_feature_from_config()
        
        if hasattr(self, 'helper_patterns') and self.helper_patterns:
            # 检查关键词前后的上下文
            before = text[max(0, pos-8):pos]
            after = text[pos+len(keyword):pos+len(keyword)+8]
            
            # 如果被辅助词汇包围，降低权重
            is_helper_context = any(pattern in before for pattern in self.helper_patterns)
            
            if is_helper_context:
                logger.info(f"检测到辅助词上下文，应用权重: {self.helper_adjustment_coefficient}")
                return self.helper_adjustment_coefficient
        return 1.0




class FeatureInteractionFusion:
    """特征交互融合器"""
    
    def __init__(self, fusion_method: str = 'interaction'):
        """
        初始化特征融合器
        
        Args:
            fusion_method: 融合方法 ('interaction', 'attention', 'hierarchical')
        """
        self.fusion_method = fusion_method
        
    def feature_interaction_fusion(self, char_features: np.ndarray, 
                                 semantic_features: np.ndarray, 
                                 statistical_features: np.ndarray) -> np.ndarray:
        """特征交互融合"""
        # 基础拼接
        basic_fusion = np.concatenate([char_features, semantic_features, statistical_features], axis=1)
        
        # 获取最小维度进行交互
        min_dim = min(char_features.shape[1], semantic_features.shape[1], statistical_features.shape[1])
        
        # 截取到相同维度进行交互
        char_subset = char_features[:, :min_dim]
        semantic_subset = semantic_features[:, :min_dim]
        statistical_subset = statistical_features[:, :min_dim]
        
        # 特征交互
        char_semantic_interaction = char_subset * semantic_subset  # 字符-语义交互
        char_statistical_interaction = char_subset * statistical_subset  # 字符-统计交互
        semantic_statistical_interaction = semantic_subset * statistical_subset  # 语义-统计交互
        
        # 最终融合
        final_features = np.concatenate([
            basic_fusion,
            char_semantic_interaction,
            char_statistical_interaction,
            semantic_statistical_interaction
        ], axis=1)
        
        return final_features
        
    def attention_fusion(self, char_features: np.ndarray, 
                        semantic_features: np.ndarray, 
                        statistical_features: np.ndarray) -> np.ndarray:
        """注意力机制融合"""
        # 计算注意力权重
        char_attention = np.mean(char_features, axis=1, keepdims=True)
        semantic_attention = np.mean(semantic_features, axis=1, keepdims=True)
        statistical_attention = np.mean(statistical_features, axis=1, keepdims=True)
        
        # 归一化注意力权重
        total_attention = char_attention + semantic_attention + statistical_attention
        char_weight = char_attention / (total_attention + 1e-8)
        semantic_weight = semantic_attention / (total_attention + 1e-8)
        statistical_weight = statistical_attention / (total_attention + 1e-8)
        
        # 获取最小维度进行加权融合
        min_dim = min(char_features.shape[1], semantic_features.shape[1], statistical_features.shape[1])
        
        # 截取到相同维度进行加权融合
        char_subset = char_features[:, :min_dim]
        semantic_subset = semantic_features[:, :min_dim]
        statistical_subset = statistical_features[:, :min_dim]
        
        # 加权融合
        fused_features = (char_subset * char_weight + 
                         semantic_subset * semantic_weight + 
                         statistical_subset * statistical_weight)
        
        return fused_features
        
    def hierarchical_fusion(self, char_features: np.ndarray, 
                           semantic_features: np.ndarray, 
                           statistical_features: np.ndarray) -> np.ndarray:
        """分层特征融合"""
        # 第一层：语义特征与统计特征融合
        semantic_statistical = np.concatenate([semantic_features, statistical_features], axis=1)
        
        # 第二层：与字符特征融合
        final_features = np.concatenate([char_features, semantic_statistical], axis=1)
        
        return final_features


class StructuralFeatureExtractor:
    """结构特征提取器"""
    
    def __init__(self):
        """初始化结构特征提取器"""
        self.question_words = ['什么', '如何', '为什么', '哪个', '怎么', '哪些']
        self.quantity_words = ['哪些', '多少个', '多少种', '几个', '多少', '全部', '所有']
        self.comparison_words = ['哪个', '相比', '比较', '更', '最高', '最低', '最远', '最近', '最长', '最短']
        
    def extract_question_word_features(self, text: str) -> np.ndarray:
        """提取疑问词特征"""
        features = []
        for word in self.question_words:
            count = text.count(word)
            features.append(count)
        return np.array(features)
        
    def extract_quantity_word_features(self, text: str) -> np.ndarray:
        """提取数量词特征"""
        features = []
        for word in self.quantity_words:
            count = text.count(word)
            features.append(count)
        return np.array(features)
        
    def extract_comparison_word_features(self, text: str) -> np.ndarray:
        """提取比较词特征"""
        features = []
        for word in self.comparison_words:
            count = text.count(word)
            features.append(count)
        return np.array(features)
        
    def extract_sentence_structure_features(self, text: str) -> np.ndarray:
        """提取句子结构特征"""
        features = []
        
        # 句子长度
        features.append(len(text))
        
        # 词数
        word_count = len(list(jieba.cut(text)))
        features.append(word_count)
        
        # 标点符号数
        punctuation_count = sum(1 for char in text if char in '，。！？；：""''（）【】')
        features.append(punctuation_count)
        
        # 疑问词位置（句首/句中/句尾）
        question_positions = []
        for word in self.question_words:
            if word in text:
                pos = text.find(word)
                if pos < len(text) * 0.3:
                    question_positions.append(1)  # 句首
                elif pos > len(text) * 0.7:
                    question_positions.append(3)  # 句尾
                else:
                    question_positions.append(2)  # 句中
            else:
                question_positions.append(0)  # 没有
        
        features.extend(question_positions)
        
        return np.array(features) 