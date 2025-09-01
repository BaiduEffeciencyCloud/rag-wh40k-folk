"""
自适应查询处理器
根据查询意图动态选择检索策略
"""

import logging
import numpy as np
import random
import json
import os
import yaml
from typing import Dict, List, Any, Union
from .processorinterface import QueryProcessorInterface
from .data_models import IntentResult, COTStep, PipelineConfig
from intent_training.model.intent_classifier import IntentClassifier
from intent_training.model.uncertainty_detector import UncertaintyDetector
from intent_training.feature.feature_engineering import IntentFeatureExtractor
from intent_training.predictor import IntentPredictor
from vocab_mgr.vocab_loader import VocabLoad
from engineconfig.qp_config import (
    INTENT_CLASSIFIER_CONFIG,
    UNCERTAINTY_CONFIG,
    INTENT_TYPES,
    NON_WH40K_RESPONSES
)

logger = logging.getLogger(__name__)

class AdaptiveProcessor(QueryProcessorInterface):
    """自适应查询处理器"""
    
    def __init__(self, intent_classifier_path: str = None, uncertainty_config: Dict = None):
        """初始化自适应处理器"""
        # 使用IntentPredictor替换手动加载逻辑
        try:
            self.intent_predictor = IntentPredictor.from_models_dir("models")
            logger.info("成功加载IntentPredictor")
            # 保持向后兼容：设置intent_classifier和feature_extractor属性
            self.intent_classifier = self.intent_predictor.intent_classifier
            self.feature_extractor = self.intent_predictor.feature_extractor
        except Exception as e:
            logger.warning(f"IntentPredictor加载失败，使用传统加载方式: {e}")
            # 回退到传统加载方式
            self.intent_predictor = None
            self.intent_classifier = self._load_intent_classifier(intent_classifier_path)
            self.feature_extractor = self._load_feature_extractor()

        self.uncertainty_detector = UncertaintyDetector(uncertainty_config or UNCERTAINTY_CONFIG)
        self.cot_processor = AdaptiveCOTProcessor()
        self.pipeline_generator = AdaptiveRetrievalPipeline()
        self.vocab_loader = VocabLoad()  # 初始化VocabLoad实例
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理查询并返回处理结果和管道配置
        Returns:
            {
                "processed_queries": List[Dict],  # 处理后的查询列表
                "pipeline_config": Dict,          # 检索管道配置
                "intent_result": IntentResult,    # 意图识别结果
                "uncertainty_result": Dict,       # 不确定性检测结果
                "cot_steps": List[COTStep]        # COT处理步骤
            }
        """
        # 1. 意图识别
        intent_result = self._classify_intent(query)
        
        # 2. 处理非战锤相关问题
        if intent_result.intent_type == 'unknown':
            return {
                'processed_queries': [],
                'pipeline_config': None,
                'intent_result': intent_result,
                'cot_steps': [],
                'fixed_response': self._get_wh40k_response()
            }
        
        # 3. COT处理
        cot_steps = self._process_cot(query, intent_result)
        
        # 4. 管道生成
        pipeline_config = self._generate_pipeline(query, intent_result, cot_steps)
        
        # 5. 生成处理后的查询
        processed_queries = []
        for step in cot_steps:
            processed_queries.append({
                'query': step.query,
                'step_id': step.step_id,
                'step_type': step.step_type,
                'reasoning': step.reasoning
            })
        
        return {
            'processed_queries': processed_queries,
            'pipeline_config': pipeline_config,
            'intent_result': intent_result,
            'uncertainty_result': intent_result.uncertainty_result,
            'cot_steps': cot_steps
        }
    
    def get_type(self) -> str:
        """获取处理器类型"""
        return "adaptive"
    
    def is_multi_query(self) -> bool:
        """是否生成多个查询"""
        return True
    
    def _load_intent_classifier(self, model_path: str = None) -> IntentClassifier:
        """加载意图分类器"""
        if model_path is None:
            model_path = INTENT_CLASSIFIER_CONFIG['model_path']
        
        intent_classifier = IntentClassifier(INTENT_CLASSIFIER_CONFIG)
        intent_classifier.load_model(model_path)
        return intent_classifier
    
    def _load_feature_extractor(self):
        """加载特征提取器，根据配置使用正确的特征提取器类型"""
        import joblib

        # 1. 读取配置文件，确定特征提取器类型
        config_path = INTENT_CLASSIFIER_CONFIG.get('config_path', "intent_training/config.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}，使用默认配置")
            config = {'feature_type': 'basic'}

        feature_type = config.get('feature_type', 'basic')
        logger.info(f"检测到特征提取器类型: {feature_type}")

        # 2. 根据特征类型加载相应的特征提取器
        if feature_type == 'pa':
            # 使用Position-Aware特征提取器
            try:
                # 导入PA特征提取器
                from intent_training.feature.factory import FeatureExtractorFactory
                from intent_training.feature.pa_feature import EnhancedFeatureExtractor

                # 创建PA特征提取器
                pa_extractor = EnhancedFeatureExtractor(config)

                # 加载增强特征提取器（从训练输出目录）
                intent_training_dir = "intent_training/model_output"
                if os.path.exists(intent_training_dir):
                    training_dirs = [d for d in os.listdir(intent_training_dir)
                                   if os.path.isdir(os.path.join(intent_training_dir, d))]
                    if training_dirs:
                        training_dirs.sort(reverse=True)
                        latest_dir = training_dirs[0]
                        feature_dir = os.path.join(intent_training_dir, latest_dir, 'feature')
                        intent_tfidf_name = config.get('intent_feature', {}).get('intent_tfidf', 'intent_tfidf.pkl')
                        feature_path = os.path.join(feature_dir, intent_tfidf_name)

                        if os.path.exists(feature_path):
                            pa_extractor.load_enhanced_feature_extractor(feature_path)
                            logger.info(f"成功加载Position-Aware特征提取器: {feature_path}")
                            return pa_extractor
                        else:
                            logger.warning(f"PA特征文件不存在: {feature_path}")

                # 如果找不到训练输出，尝试从models目录加载
                models_dir = "models"
                feature_path = os.path.join(models_dir, 'intent_tfidf.pkl')
                if os.path.exists(feature_path):
                    pa_extractor.load_enhanced_feature_extractor(feature_path)
                    logger.info(f"成功加载Position-Aware特征提取器: {feature_path}")
                    return pa_extractor

            except Exception as e:
                logger.warning(f"加载Position-Aware特征提取器失败: {e}")

        elif feature_type == 'advanced':
            # 使用高级特征提取器
            try:
                from intent_training.feature.factory import FeatureExtractorFactory
                from intent_training.feature.ad_feature import AdvancedFeature

                ad_extractor = AdvancedFeature(config)

                # 加载增强特征提取器
                models_dir = "models"
                feature_path = os.path.join(models_dir, 'intent_tfidf.pkl')
                if os.path.exists(feature_path):
                    ad_extractor.load(feature_path)
                    logger.info(f"成功加载高级特征提取器: {feature_path}")
                    return ad_extractor

            except Exception as e:
                logger.warning(f"加载高级特征提取器失败: {e}")

        else:
            # 使用基础特征提取器
            try:
                # 优先从models目录加载TF-IDF向量化器
                models_dir = "models"
                tfidf_path = os.path.join(models_dir, 'intent_tfidf.pkl')

                if os.path.exists(tfidf_path):
                    tfidf_vectorizer = joblib.load(tfidf_path)
                    logger.info(f"成功加载基础TF-IDF向量化器: {tfidf_path}")
                    return tfidf_vectorizer

                # 备用方案：从训练输出目录加载
                intent_training_dir = "intent_training/model_output"
                if os.path.exists(intent_training_dir):
                    training_dirs = [d for d in os.listdir(intent_training_dir)
                                   if os.path.isdir(os.path.join(intent_training_dir, d))]
                    if training_dirs:
                        training_dirs.sort(reverse=True)
                        latest_dir = training_dirs[0]
                        feature_dir = os.path.join(intent_training_dir, latest_dir, 'feature')
                        tfidf_path = os.path.join(feature_dir, 'intent_tfidf.pkl')

                        if os.path.exists(tfidf_path):
                            tfidf_vectorizer = joblib.load(tfidf_path)
                            logger.info(f"成功加载基础TF-IDF向量化器: {tfidf_path}")
                            return tfidf_vectorizer

            except Exception as e:
                logger.warning(f"加载基础TF-IDF向量化器失败: {e}")

        # 如果所有加载都失败，返回None，使用备用方案
        logger.warning("所有特征提取器加载失败，将使用备用特征提取方法")
        return None
    
    def _classify_intent(self, query: str) -> IntentResult:
        """优化的意图识别：先检查WH40K词汇，再进行ML预测"""
        # 1. 快速检查：是否包含WH40K相关词汇
        if not self._contains_wh40k_vocab(query):
            # 非WH40K查询，节省资源，直接返回unknown
            return IntentResult(
                intent_type="unknown",
                confidence=0.8,
                entities=[],
                relations=[],
                complexity_score=0.0,
                uncertainty_result={
                    "is_uncertain": False,
                    "uncertainty_score": 0.0,
                    "uncertainty_reason": "非WH40K相关查询"
                },
                probabilities=np.array([0.2] * 5)  # 默认概率分布
            )

        # 2. 是WH40K查询，才进行耗时的ML预测
        try:
            # 使用IntentPredictor进行预测（如果可用）
            if self.intent_predictor is not None:
                predict_result = self.intent_predictor.predict(query)
                intent_prediction = predict_result['intent'].lower()
                max_confidence = predict_result['confidence']
                intent_probabilities = np.array(predict_result['probabilities'])
            else:
                # 回退到传统预测方式
                features = self._extract_features(query)
                intent_probabilities = self.intent_classifier.predict_proba(features)[0]
                max_confidence = np.max(intent_probabilities)

                # 使用概率最大值确定预测意图
                max_prob_idx = np.argmax(intent_probabilities)
                intent_prediction = self.intent_classifier.classes_[max_prob_idx].lower()

        except Exception as e:
            logger.warning(f"ML模型预测失败: {str(e)}, 使用规则-based分类")
            # 模型预测失败，使用规则-based分类
            intent_prediction = self._rule_based_classify(query)
            max_confidence = 0.5  # 规则分类的默认置信度
            intent_probabilities = np.array([0.5] * 5)  # 假设有5个意图类别

        # 3. 返回ML预测结果
        return IntentResult(
            intent_type=intent_prediction,
            confidence=max_confidence,
            entities=[],
            relations=[],
            complexity_score=0.0,
            uncertainty_result={
                "is_uncertain": max_confidence < 0.5,
                "uncertainty_score": 1.0 - max_confidence,
                "uncertainty_reason": f"置信度较低({max_confidence:.2f})" if max_confidence < 0.5 else "置信度良好"
            },
            probabilities=intent_probabilities
        )
    
    def _extract_features(self, query: str) -> np.ndarray:
        """提取查询特征，支持不同类型的特征提取器"""
        if self.feature_extractor is not None:
            try:
                # 检查特征提取器类型
                if hasattr(self.feature_extractor, 'get_enhanced_intent_features'):
                    # Position-Aware 或 Advanced 特征提取器
                    logger.info("使用增强特征提取器")
                    features = self.feature_extractor.get_enhanced_intent_features(query)
                    return features

                elif hasattr(self.feature_extractor, 'transform'):
                    # 基础TF-IDF向量化器
                    logger.info("使用基础TF-IDF特征提取器")
                    features = self.feature_extractor.transform([query])
                    return features.toarray()

                elif isinstance(self.feature_extractor, dict):
                    # 字典格式的特征提取器
                    logger.info("使用字典格式特征提取器")
                    tfidf_vectorizer = self.feature_extractor.get('tfidf_vectorizer')
                    if tfidf_vectorizer is not None and hasattr(tfidf_vectorizer, 'transform'):
                        features = tfidf_vectorizer.transform([query])
                        return features.toarray()
                    else:
                        logger.warning("字典中没有找到有效的TF-IDF向量化器")

                else:
                    logger.warning(f"特征提取器类型不支持: {type(self.feature_extractor)}")

            except Exception as e:
                logger.warning(f"特征提取失败: {str(e)}，使用备用方案")

        # 备用方案：使用简单的字符级特征
        logger.info("使用备用特征提取方法")
        features = np.zeros(1000)  # 固定1000维
        for i, char in enumerate(query[:1000]):
            features[i] = ord(char) % 1000
        return features.reshape(1, -1)  # 确保是2D数组
    
    def _extract_entities(self, query: str, intent_type: str = None) -> List[str]:
        """根据意图类型提取查询中的实体"""
        entities = []
        wh40k_terms = self._load_wh40k_vocab()
        
        if intent_type == 'compare':
            # 比较类：根据比较模式识别实体
            compare_pattern = self.cot_processor._detect_compare_pattern(query)
            
            if compare_pattern == 'direct_comparison':
                # 直接比较：识别两个实体
                entities = self._extract_direct_cmp_entities(query, wh40k_terms)
            else:
                # 范围比较：识别一个主体
                entities = self._extract_range_cmp_entities(query, wh40k_terms)
        else:
            # Query/Rule/List：识别一个主体
            for term in wh40k_terms:
                if term in query:
                    entities.append(term)
                    break  # 只取第一个匹配的
        
        return entities
    
    def _extract_direct_cmp_entities(self, query: str, wh40k_terms: List[str]) -> List[str]:
        """提取直接比较的两个实体"""
        entities = []
        
        # 优先处理"和"比较模式
        if '和' in query:
            parts = query.split('和')
            if len(parts) >= 2:
                # 提取两个部分中的所有战锤实体
                for part in parts[:2]:  # 只取前两个部分
                    for term in wh40k_terms:
                        if term in part and term not in entities:
                            entities.append(term)
        # 然后处理"哪个...更"模式
        elif '哪个' in query and '更' in query:
            # 提取"哪个"和"更"之间的实体
            which_index = query.find('哪个')
            more_index = query.find('更')
            
            if which_index < more_index:
                # 提取"哪个"和"更"之间的内容作为比较对象
                between_text = query[which_index + 2:more_index].strip()
                # 在战锤词汇中查找匹配的实体
                for term in wh40k_terms:
                    if term in between_text:
                        entities.append(term)
        
        return entities
    
    def _extract_range_cmp_entities(self, query: str, wh40k_terms: List[str]) -> List[str]:
        """提取范围比较的主体"""
        entities = []
        
        # 检查是否是"哪个...最"模式
        if '哪个' in query and '最' in query:
            # 提取"哪个"和"最"之间的实体
            which_index = query.find('哪个')
            most_index = query.find('最')
            
            if which_index < most_index:
                between_text = query[which_index + 2:most_index].strip()
                for term in wh40k_terms:
                    if term in between_text:
                        entities.append(term)
                        break
        else:
            # 其他范围比较模式，识别一个主体
            for term in wh40k_terms:
                if term in query:
                    entities.append(term)
                    break
        
        return entities
    
    def _extract_relations(self, query: str) -> List[str]:
        """提取查询中的关系"""
        # 简单的关系提取
        relations = []
        # 这里可以添加关系提取逻辑
        return relations
    
    def _calculate_complexity(self, query: str) -> float:
        """计算查询复杂度"""
        # 基于查询长度和词汇复杂度计算
        length_factor = min(len(query) / 50.0, 1.0)  # 长度因子
        word_count = len(query.split())
        word_factor = min(word_count / 10.0, 1.0)  # 词汇因子
        
        return (length_factor + word_factor) / 2.0
    
    def _detect_uncertainty(self, features: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """检测不确定性"""
        return self.uncertainty_detector.detect(features.reshape(1, -1), probabilities.reshape(1, -1))
    
    def _check_keyword_match(self, query: str, intent_type: str) -> bool:
        """检查查询是否匹配指定意图的关键词"""
        query_lower = query.lower()
        
        # 首先检查是否包含战锤40K相关词汇
        wh40k_vocab = self._load_wh40k_vocab()
        wh40k_terms = []
        
        # 从词汇表中提取所有词汇
        for category, terms in wh40k_vocab.items():
            wh40k_terms.extend(terms)
        
        has_wh40k_content = any(term.lower() in query_lower for term in wh40k_terms)
        
        # 如果没有战锤内容，直接返回False
        if not has_wh40k_content:
            return False
        
        # 只有在包含战锤内容的情况下才检查关键词匹配
        if intent_type in INTENT_TYPES:
            keywords = INTENT_TYPES[intent_type]['keywords']
            return any(keyword in query_lower for keyword in keywords)
        return False
    
    def _find_keyword_match(self, query: str) -> str:
        """找到最佳的关键词匹配"""
        query_lower = query.lower()
        
        # 检查是否包含战锤40K相关词汇
        wh40k_vocab = self._load_wh40k_vocab()
        wh40k_terms = []
        
        # 从词汇表中提取所有词汇
        for category, terms in wh40k_vocab.items():
            wh40k_terms.extend(terms)
        
        has_wh40k_content = any(term.lower() in query_lower for term in wh40k_terms)
        
        # 如果没有战锤内容，直接返回non_wh40k
        if not has_wh40k_content:
            return 'non_wh40k'
        
        # 只有在包含战锤内容的情况下才进行关键词匹配
        for intent_type, config in INTENT_TYPES.items():
            keywords = config['keywords']
            for keyword in keywords:
                if keyword in query_lower:
                    return intent_type
                # 特殊处理模式匹配
                elif keyword == '当...时' and '当' in query_lower and '时' in query_lower:
                    return intent_type
        
        return 'unknown'
    
    def _rule_based_classify(self, query: str) -> str:
        """基于规则的意图分类"""
        return self._find_keyword_match(query)
    
    def _get_wh40k_response(self) -> str:
        """随机选择战锤40K阵营的回复"""
        responses = NON_WH40K_RESPONSES.values()
        return random.choice(list(responses))
    
    def _process_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """根据意图进行COT处理"""
        return self.cot_processor.process_cot(query, intent)
    
    def _generate_pipeline(self, query: str, intent: IntentResult, cot_steps: List[COTStep]) -> Dict[str, Any]:
        """生成检索管道配置"""
        return self.pipeline_generator.generate_pipeline(query, intent, cot_steps)
    
    def _contains_wh40k_vocab(self, query: str) -> bool:
        """检查查询是否包含WH40K相关词汇"""
        # 加载WH40K词汇
        wh40k_vocab = self._load_wh40k_vocab()
        if not wh40k_vocab:
            return True  # 如果无法加载词汇，默认认为是WH40K相关
        
        # 检查查询中是否包含WH40K词汇
        query_lower = query.lower()
        for category, terms in wh40k_vocab.items():
            for term in terms:
                if term.lower() in query_lower:
                    return True
        
        return False
    
    def _load_wh40k_vocab(self) -> Dict[str, List[str]]:
        """加载WH40K词汇表（使用VocabLoad）"""
        try:
            # 使用VocabLoad加载所有词典数据
            vocab_data = self.vocab_loader._load_all_vocabulary()

            # 转换为List[str]格式
            vocab_terms = {}
            for category, terms in vocab_data.items():
                if isinstance(terms, dict):
                    term_list = []
                    for term, info in terms.items():
                        if isinstance(term, str) and term.strip():
                            term_list.append(term)
                            # 添加同义词
                            if isinstance(info, dict) and 'synonyms' in info:
                                synonyms = info.get('synonyms', [])
                                if isinstance(synonyms, list):
                                    for synonym in synonyms:
                                        if isinstance(synonym, str) and synonym.strip():
                                            term_list.append(synonym)
                    vocab_terms[category] = term_list

            logger.debug(f"成功加载 {len(vocab_terms)} 个词典分类")
            return vocab_terms

        except Exception as e:
            logger.error(f"加载WH40K词汇失败: {e}")
            return {} 

 