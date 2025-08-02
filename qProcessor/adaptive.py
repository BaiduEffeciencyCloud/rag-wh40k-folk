"""
自适应查询处理器
根据查询意图动态选择检索策略
"""

import logging
import numpy as np
from typing import Dict, List, Any, Union
from .processorinterface import QueryProcessorInterface
from .data_models import IntentResult, COTStep, PipelineConfig
from intent_training.model.intent_classifier import IntentClassifier
from intent_training.model.uncertainty_detector import UncertaintyDetector
from intent_training.feature.feature_engineering import IntentFeatureExtractor
from engineconfig.qp_config import (
    INTENT_CLASSIFIER_CONFIG, 
    UNCERTAINTY_CONFIG,
    COT_TEMPLATES,
    PIPELINE_CONFIGS,
    INTENT_TYPES
)

logger = logging.getLogger(__name__)


class AdaptiveCOTProcessor:
    """自适应思维链处理器"""
    
    def __init__(self):
        self.cot_templates = COT_TEMPLATES
    
    def process_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """根据意图进行COT处理"""
        intent_type = intent.intent_type
        
        if intent_type == 'compare':
            return self._process_compare_cot(query, intent)
        elif intent_type == 'rule':
            return self._process_rule_cot(query, intent)
        elif intent_type == 'list':
            return self._process_list_cot(query, intent)
        elif intent_type == 'query':
            return self._process_query_cot(query, intent)
        else:
            return self._process_default_cot(query, intent)
    
    def _process_compare_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """处理比较类查询的COT"""
        # 检测比较模式
        pattern_type = self._detect_compare_pattern(query)
        
        # 根据模式类型选择模板
        if pattern_type == 'direct_comparison':
            template = COT_TEMPLATES['compare']['patterns'][0]  # direct_comparison
        else:
            template = COT_TEMPLATES['compare']['patterns'][1]  # range_extremum_comparison
        
        cot_steps = []
        for step_config in template['chain_of_thought']:
            step = COTStep(
                step_id=step_config['step'],
                step_type='comparison',
                query=self._generate_sub_query(query, step_config, pattern_type),
                reasoning=step_config['reasoning'],
                entities=intent.entities,
                relations=intent.relations,
                expected_info=template['needed_information'],
                dependencies=[],
                confidence=intent.confidence,
                metadata={'pattern_type': pattern_type}
            )
            cot_steps.append(step)
        
        return cot_steps
    
    def _process_rule_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """处理规则类查询的COT"""
        template = COT_TEMPLATES['rule']
        
        cot_steps = []
        for step_config in template['chain_of_thought']:
            step = COTStep(
                step_id=step_config['step'],
                step_type='rule_reasoning',
                query=self._generate_rule_sub_query(query, step_config),
                reasoning=step_config['reasoning'],
                entities=intent.entities,
                relations=intent.relations,
                expected_info=template['needed_information'],
                dependencies=[],
                confidence=intent.confidence,
                metadata={'intent': template['intent']}
            )
            cot_steps.append(step)
        
        return cot_steps
    
    def _process_list_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """处理列举类查询的COT"""
        # 使用简单的COT处理
        return self._generate_simple_cot(query, intent)
    
    def _process_query_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """处理一般查询的COT"""
        # 使用简单的COT处理
        return self._generate_simple_cot(query, intent)
    
    def _process_default_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """处理默认COT"""
        return self._generate_simple_cot(query, intent)
    
    def _detect_compare_pattern(self, query: str) -> str:
        """检测比较模式"""
        query_lower = query.lower()
        
        # 检测新模式："哪个...更"和"哪个...最"
        if '哪个' in query_lower and '更' in query_lower:
            return 'direct_comparison'  # "哪个...更"
        elif '哪个' in query_lower and '最' in query_lower:
            return 'range_extremum_comparison'  # "哪个...最"
        
        # 检测其他极值比较模式
        extremum_keywords = ['哪种', '最远', '最长', '最高', '最低']
        if any(keyword in query_lower for keyword in extremum_keywords):
            return 'range_extremum_comparison'
        else:
            return 'direct_comparison'
    
    def _generate_sub_query(self, query: str, step_config: Dict, pattern_type: str) -> str:
        """生成子查询"""
        if pattern_type == 'direct_comparison':
            # 简单的实体替换
            return query  # 暂时返回原查询，后续可以优化
        elif pattern_type == 'range_extremum_comparison':
            return query  # 暂时返回原查询，后续可以优化
        else:
            return query
    
    def _generate_rule_sub_query(self, query: str, step_config: Dict) -> str:
        """生成规则类子查询"""
        return query  # 暂时返回原查询，后续可以优化
    
    def _generate_complex_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """生成复杂COT步骤"""
        return self.process_cot(query, intent)
    
    def _generate_simple_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """生成简单COT步骤"""
        # 简单的单步COT
        step = COTStep(
            step_id=1,
            step_type='simple',
            query=query,
            reasoning='简单查询处理',
            entities=intent.entities,
            relations=intent.relations,
            expected_info=['基本信息'],
            dependencies=[],
            confidence=intent.confidence,
            metadata={'intent_type': intent.intent_type}
        )
        return [step]


class AdaptiveRetrievalPipeline:
    """自适应检索管道生成器"""
    
    def __init__(self):
        self.pipeline_configs = PIPELINE_CONFIGS
    
    def generate_pipeline(self, query: str, intent: IntentResult, cot_steps: List[COTStep]) -> Dict[str, Any]:
        """根据意图和COT步骤生成检索管道"""
        intent_type = intent.intent_type
        
        if intent_type == 'compare':
            return self._get_compare_pipeline_config(intent, cot_steps)
        else:
            return self._get_standard_pipeline_config(intent_type)
    
    def _get_compare_pipeline_config(self, intent: IntentResult, cot_steps: List[COTStep]) -> Dict[str, Any]:
        """获取比较类查询的管道配置"""
        # 检测比较模式
        pattern_type = self._detect_compare_pattern_from_cot(cot_steps)
        
        if pattern_type == 'direct_comparison':
            return {
                'search_strategy': 'multi_engine',
                'engines': [
                    {'type': 'dense', 'purpose': '实体A特征检索'},
                    {'type': 'dense', 'purpose': '实体B特征检索'}
                ],
                'post_processors': ['rerank', 'dedup', 'mmr'],
                'fusion_strategy': 'llm_aggregation'
            }
        elif pattern_type == 'range_extremum_comparison':
            return {
                'search_strategy': 'single_engine',
                'engines': [
                    {'type': 'hybrid', 'purpose': '范围极值检索'}
                ],
                'post_processors': ['rerank', 'dedup', 'mmr'],
                'fusion_strategy': 'llm_aggregation'
            }
        else:
            return self._get_standard_pipeline_config('compare')
    
    def _get_standard_pipeline_config(self, intent_type: str) -> Dict[str, Any]:
        """获取标准管道配置"""
        return PIPELINE_CONFIGS.get(intent_type, PIPELINE_CONFIGS['unknown'])
    
    def _detect_compare_pattern_from_cot(self, cot_steps: List[COTStep]) -> str:
        """从COT步骤检测比较模式"""
        if not cot_steps:
            return 'direct_comparison'
        
        # 检查是否有范围相关的推理步骤
        for step in cot_steps:
            if '范围' in step.reasoning or '极值' in step.reasoning:
                return 'range_extremum_comparison'
        
        return 'direct_comparison'
    
    def _get_pipeline_config(self, intent_type: str) -> Dict[str, Any]:
        """获取管道配置"""
        return PIPELINE_CONFIGS.get(intent_type, PIPELINE_CONFIGS['unknown'])
    
    def _build_pipeline(self, pipeline_config: Dict[str, Any], cot_steps: List[COTStep]) -> Dict[str, Any]:
        """构建检索管道"""
        return pipeline_config


class AdaptiveProcessor(QueryProcessorInterface):
    """自适应查询处理器"""
    
    def __init__(self, intent_classifier_path: str = None, uncertainty_config: Dict = None):
        """初始化自适应处理器"""
        self.intent_classifier = self._load_intent_classifier(intent_classifier_path)
        self.uncertainty_detector = UncertaintyDetector(uncertainty_config or UNCERTAINTY_CONFIG)
        self.feature_extractor = self._load_feature_extractor()  # 加载特征提取器
        self.cot_processor = AdaptiveCOTProcessor()
        self.pipeline_generator = AdaptiveRetrievalPipeline()
    
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
        intent_result = self._classify_intent_with_gradient(query)
        
        # 2. 处理非战锤相关问题
        if intent_result.intent_type == 'unknown':
            return {
                'processed_queries': [],
                'pipeline_config': None,
                'intent_result': intent_result,
                'cot_steps': [],
                'fixed_response': self._get_random_wh40k_response()
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
        """加载特征提取器"""
        import joblib
        import os
        
        # 从训练输出目录加载TF-IDF向量化器
        intent_training_dir = "intent_training/model_output"
        if os.path.exists(intent_training_dir):
            # 获取最新的训练输出目录
            training_dirs = [d for d in os.listdir(intent_training_dir) 
                           if os.path.isdir(os.path.join(intent_training_dir, d))]
            if training_dirs:
                # 按时间戳排序，取最新的
                training_dirs.sort(reverse=True)
                latest_dir = training_dirs[0]
                feature_dir = os.path.join(intent_training_dir, latest_dir, 'feature')
                tfidf_path = os.path.join(feature_dir, 'intent_tfidf.pkl')
                
                if os.path.exists(tfidf_path):
                    try:
                        tfidf_vectorizer = joblib.load(tfidf_path)
                        logger.info(f"成功加载TF-IDF向量化器: {tfidf_path}")
                        return tfidf_vectorizer
                    except Exception as e:
                        logger.warning(f"加载TF-IDF向量化器失败: {e}")
        
        # 如果加载失败，返回None，使用备用方案
        logger.warning("TF-IDF向量化器加载失败，将使用备用特征提取方法")
        return None
    
    def _classify_intent_with_gradient(self, query: str) -> IntentResult:
        """简化的意图识别：使用概率最大值 + WH40K专业术语检查"""
        # ML模型预测
        features = self._extract_features(query)
        intent_probabilities = self.intent_classifier.predict_proba(features)[0]
        max_confidence = np.max(intent_probabilities)
        
        # 使用概率最大值确定预测意图
        max_prob_idx = np.argmax(intent_probabilities)
        intent_prediction = self.intent_classifier.classes_[max_prob_idx]
        
        # 统一意图类型为小写
        intent_prediction = intent_prediction.lower()
        
        # 检查是否包含WH40K相关词汇
        if not self._contains_wh40k_vocabulary(query):
            return IntentResult(
                intent_type="unknown",  # 不包含WH40K专业术语
                confidence=0.8,
                entities=[],
                relations=[],
                complexity_score=0.0,
                uncertainty_result={
                    "is_uncertain": False,
                    "uncertainty_score": 0.0,
                    "uncertainty_reason": "不包含WH40K专业术语"
                },
                probabilities=intent_probabilities
            )
        
        # 包含WH40K专业术语，直接使用概率最大值的结果
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
        """提取查询特征"""
        if self.feature_extractor is not None:
            # 使用训练好的TF-IDF向量化器
            features = self.feature_extractor.transform([query])
            return features.toarray()
        else:
            # 备用方案：使用简单的字符级特征
            features = np.zeros(1000)  # 固定1000维
            for i, char in enumerate(query[:1000]):
                features[i] = ord(char) % 1000
            return features.reshape(1, -1)  # 确保是2D数组
    
    def _extract_entities(self, query: str, intent_type: str = None) -> List[str]:
        """根据意图类型提取查询中的实体"""
        entities = []
        wh40k_terms = self._load_wh40k_vocabulary()
        
        if intent_type == 'compare':
            # 比较类：根据比较模式识别实体
            compare_pattern = self.cot_processor._detect_compare_pattern(query)
            
            if compare_pattern == 'direct_comparison':
                # 直接比较：识别两个实体
                entities = self._extract_direct_comparison_entities(query, wh40k_terms)
            else:
                # 范围比较：识别一个主体
                entities = self._extract_range_comparison_entities(query, wh40k_terms)
        else:
            # Query/Rule/List：识别一个主体
            for term in wh40k_terms:
                if term in query:
                    entities.append(term)
                    break  # 只取第一个匹配的
        
        return entities
    
    def _extract_direct_comparison_entities(self, query: str, wh40k_terms: List[str]) -> List[str]:
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
    
    def _extract_range_comparison_entities(self, query: str, wh40k_terms: List[str]) -> List[str]:
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
        wh40k_vocab = self._load_wh40k_vocabulary()
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
    
    def _find_best_keyword_match(self, query: str) -> str:
        """找到最佳的关键词匹配"""
        query_lower = query.lower()
        
        # 检查是否包含战锤40K相关词汇
        wh40k_vocab = self._load_wh40k_vocabulary()
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
        return self._find_best_keyword_match(query)
    
    def _get_random_wh40k_response(self) -> str:
        """随机选择战锤40K阵营的回复"""
        import random
        from engineconfig.qp_config import NON_WH40K_RESPONSES
        
        responses = NON_WH40K_RESPONSES.values()
        return random.choice(list(responses))
    
    def _process_cot(self, query: str, intent: IntentResult) -> List[COTStep]:
        """根据意图进行COT处理"""
        return self.cot_processor.process_cot(query, intent)
    
    def _generate_pipeline(self, query: str, intent: IntentResult, cot_steps: List[COTStep]) -> Dict[str, Any]:
        """生成检索管道配置"""
        return self.pipeline_generator.generate_pipeline(query, intent, cot_steps)
    
    def _contains_wh40k_vocabulary(self, query: str) -> bool:
        """检查查询是否包含WH40K相关词汇"""
        # 加载WH40K词汇
        wh40k_vocab = self._load_wh40k_vocabulary()
        if not wh40k_vocab:
            return True  # 如果无法加载词汇，默认认为是WH40K相关
        
        # 检查查询中是否包含WH40K词汇
        query_lower = query.lower()
        for category, terms in wh40k_vocab.items():
            for term in terms:
                if term.lower() in query_lower:
                    return True
        
        return False
    
    def _load_wh40k_vocabulary(self) -> Dict[str, List[str]]:
        """加载WH40K词汇表"""
        try:
            import json
            import os
            
            vocab_data = {}
            vocab_dir = "dict/wh40k_vocabulary"  # 直接使用路径，不从config导入
            
            if not os.path.exists(vocab_dir):
                return {}
            
            # 加载所有JSON文件
            for filename in os.listdir(vocab_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(vocab_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            category = filename.replace('.json', '')
                            # 提取所有词汇（包括同义词）
                            terms = []
                            for term, info in data.items():
                                terms.append(term)
                                if 'synonyms' in info:
                                    terms.extend(info['synonyms'])
                            vocab_data[category] = terms
                    except Exception as e:
                        print(f"加载词汇文件 {filepath} 失败: {e}")
            
            return vocab_data
        except Exception as e:
            print(f"加载WH40K词汇失败: {e}")
            return {} 