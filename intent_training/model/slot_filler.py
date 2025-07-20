"""
槽位填充模块
实现从用户查询中提取实体和参数的槽位填充功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import json
import logging
from collections import defaultdict, Counter
import os

logger = logging.getLogger(__name__)


class Slot:
    """槽位实体类"""
    
    def __init__(self, name: str, value: str, start_pos: int, end_pos: int, confidence: float):
        """
        初始化槽位
        
        Args:
            name: 槽位名称
            value: 槽位值
            start_pos: 开始位置
            end_pos: 结束位置
            confidence: 置信度
        """
        self.name = name
        self.value = value
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = confidence
    
    def __repr__(self):
        return f"Slot(name='{self.name}', value='{self.value}', pos=({self.start_pos},{self.end_pos}), conf={self.confidence:.2f})"


class SlotFillingResult:
    """槽位填充结果类"""
    
    def __init__(self, intent: str, slots: Dict[str, Slot], confidence: float, raw_query: str):
        """
        初始化槽位填充结果
        
        Args:
            intent: 意图
            slots: 槽位字典
            confidence: 整体置信度
            raw_query: 原始查询
        """
        self.intent = intent
        self.slots = slots
        self.confidence = confidence
        self.raw_query = raw_query
    
    def __repr__(self):
        return f"SlotFillingResult(intent='{self.intent}', slots={len(self.slots)}, conf={self.confidence:.2f})"


class SlotFeatureExtractor:
    """槽位特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        slot_config = config.get('slot_feature', {})
        
        # 处理ngram_range参数
        ngram_range = slot_config.get('ngram_range', (1, 2))
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=slot_config.get('max_features', 1000),
            ngram_range=ngram_range,
            min_df=slot_config.get('min_df', 1),
            analyzer=slot_config.get('analyzer', 'char')
        )
        
        self.is_fitted = False
        self.intent_vocab = {}
        self.slot_vocab = {}
    
    def fit(self, queries: List[str], intents: List[str], slot_labels: List[Dict[str, str]]):
        """
        训练特征提取器
        
        Args:
            queries: 查询列表
            intents: 意图列表
            slot_labels: 槽位标签列表
        """
        logger.info("开始训练槽位特征提取器")
        
        # 训练TF-IDF向量化器
        self.tfidf_vectorizer.fit(queries)
        
        # 构建意图词汇表
        unique_intents = set(intents)
        self.intent_vocab = {intent: i for i, intent in enumerate(unique_intents)}
        
        # 构建槽位词汇表
        all_slot_names = set()
        for slot_dict in slot_labels:
            all_slot_names.update(slot_dict.keys())
        self.slot_vocab = {slot: i for i, slot in enumerate(all_slot_names)}
        
        self.is_fitted = True
        logger.info(f"槽位特征提取器训练完成，意图数: {len(self.intent_vocab)}, 槽位数: {len(self.slot_vocab)}")
    
    def extract_features(self, query: str, intent: str) -> np.ndarray:
        """
        提取特征向量
        
        Args:
            query: 查询文本
            intent: 意图
            
        Returns:
            特征向量
        """
        if not self.is_fitted:
            raise ValueError("特征提取器尚未训练")
        
        # TF-IDF特征
        tfidf_features = self.tfidf_vectorizer.transform([query]).toarray()[0]
        
        # 意图特征
        intent_features = np.zeros(len(self.intent_vocab))
        if intent in self.intent_vocab:
            intent_features[self.intent_vocab[intent]] = 1.0
        
        # 实体特征（固定维度）
        entity_features = self.extract_entity_features(query)
        # 使用固定大小的特征向量
        entity_feature_size = 100  # 固定大小
        entity_feature_array = np.zeros(entity_feature_size)
        for i, (key, value) in enumerate(entity_features.items()):
            if i < entity_feature_size:
                entity_feature_array[i] = value
        
        # 上下文特征（固定维度）
        context_features = self.extract_context_features(query, intent)
        context_feature_size = 50  # 固定大小
        context_feature_array = np.zeros(context_feature_size)
        for i, (key, value) in enumerate(context_features.items()):
            if i < context_feature_size:
                context_feature_array[i] = value
        
        # 合并所有特征
        all_features = np.concatenate([
            tfidf_features,
            intent_features,
            entity_feature_array,
            context_feature_array
        ])
        
        return all_features
    
    def extract_entity_features(self, query: str) -> Dict[str, float]:
        """
        提取实体特征
        
        Args:
            query: 查询文本
            
        Returns:
            实体特征字典
        """
        features = {}
        
        # 字符级特征
        for char in query:
            features[f"char_{char}"] = features.get(f"char_{char}", 0) + 1
        
        # 词汇特征
        words = query.split()
        for word in words:
            features[f"word_{word}"] = features.get(f"word_{word}", 0) + 1
        
        # 添加原始字符特征（用于测试）
        for char in query:
            features[char] = features.get(char, 0) + 1
        
        return features
    
    def extract_context_features(self, query: str, intent: str) -> Dict[str, float]:
        """
        提取上下文特征
        
        Args:
            query: 查询文本
            intent: 意图
            
        Returns:
            上下文特征字典
        """
        features = {}
        
        # 意图特征
        features[f"intent_{intent}"] = 1.0
        
        # 查询长度特征
        features["query_length"] = len(query)
        
        # 词汇数量特征
        features["word_count"] = len(query.split())
        
        return features


class SlotFiller:
    """槽位填充器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化槽位填充器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.feature_extractor = SlotFeatureExtractor(config)
        self.models = {}  # 每个槽位一个分类器
        self.slot_names = set()
        self.is_trained = False
        
    def train(self, training_data: List[Dict[str, Any]]):
        """
        训练槽位填充器
        
        Args:
            training_data: 训练数据列表，格式为 [{"query": "...", "intent": "...", "slots": {...}}]
        """
        logger.info("开始训练槽位填充器")
        
        if not training_data:
            raise ValueError("训练数据不能为空")
        
        # 准备训练数据
        queries = []
        intents = []
        slot_labels = []
        
        for item in training_data:
            queries.append(item["query"])
            intents.append(item["intent"])
            slot_labels.append(item["slots"])
            self.slot_names.update(item["slots"].keys())
        
        # 训练特征提取器
        self.feature_extractor.fit(queries, intents, slot_labels)
        
        # 为每个槽位训练一个分类器
        max_iter = self.config.get('slot_classifier_max_iter', 1000)
        
        for slot_name in self.slot_names:
            logger.info(f"训练槽位分类器: {slot_name}")
            
            # 准备该槽位的训练数据
            slot_features = []
            slot_labels_binary = []
            
            for item in training_data:
                features = self.feature_extractor.extract_features(item["query"], item["intent"])
                slot_features.append(features)
                
                # 二分类标签：该槽位是否存在
                has_slot = slot_name in item["slots"]
                slot_labels_binary.append(1 if has_slot else 0)
            
            # 训练分类器
            classifier = LogisticRegression(max_iter=max_iter, random_state=42)
            classifier.fit(slot_features, slot_labels_binary)
            
            self.models[slot_name] = classifier
        
        self.is_trained = True
        logger.info(f"槽位填充器训练完成，槽位数: {len(self.models)}")
    
    def predict(self, query: str, intent: str) -> SlotFillingResult:
        """
        预测槽位填充结果
        
        Args:
            query: 查询文本
            intent: 意图
            
        Returns:
            槽位填充结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 提取特征
        features = self.feature_extractor.extract_features(query, intent)
        
        # 预测每个槽位
        slots = {}
        slot_confidences = []
        
        for slot_name, classifier in self.models.items():
            # 预测该槽位是否存在
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0].max()
            
            if prediction == 1:  # 该槽位存在
                # 提取槽位值（简单实现：基于关键词匹配）
                slot_value = self._extract_slot_value(query, slot_name)
                if slot_value:
                    start_pos = query.find(slot_value)
                    end_pos = start_pos + len(slot_value)
                    slot = Slot(slot_name, slot_value, start_pos, end_pos, confidence)
                    slots[slot_name] = slot
                    slot_confidences.append(confidence)
        
        # 计算整体置信度
        overall_confidence = np.mean(slot_confidences) if slot_confidences else 0.0
        
        return SlotFillingResult(intent, slots, overall_confidence, query)
    
    def extract_slots(self, query: str, intent: str) -> Dict[str, Slot]:
        """
        提取槽位（不验证）
        
        Args:
            query: 查询文本
            intent: 意图
            
        Returns:
            槽位字典
        """
        result = self.predict(query, intent)
        return result.slots
    
    def validate_slots(self, slots: Dict[str, Slot], query: str) -> Dict[str, Slot]:
        """
        验证槽位
        
        Args:
            slots: 槽位字典
            query: 查询文本
            
        Returns:
            验证后的槽位字典
        """
        validated_slots = {}
        
        for slot_name, slot in slots.items():
            # 验证槽位值是否在查询中
            if slot.value in query:
                # 验证位置是否正确
                if 0 <= slot.start_pos < len(query) and slot.start_pos < slot.end_pos <= len(query):
                    validated_slots[slot_name] = slot
        
        return validated_slots
    
    def save_model(self, model_dir: str):
        """
        保存模型
        
        Args:
            model_dir: 模型保存目录
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 保存特征提取器
        joblib.dump(self.feature_extractor, f"{model_dir}/slot_feature_extractor.pkl")
        
        # 保存分类器模型
        for slot_name, classifier in self.models.items():
            joblib.dump(classifier, f"{model_dir}/{slot_name}_classifier.pkl")
        
        # 保存槽位信息
        slot_info = {
            'slot_names': list(self.slot_names),
            'config': self.config
        }
        with open(f"{model_dir}/slot_info.json", 'w', encoding='utf-8') as f:
            json.dump(slot_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"槽位填充器模型已保存到: {model_dir}")
    
    def load_model(self, model_dir: str):
        """
        加载模型
        
        Args:
            model_dir: 模型加载目录
        """
        # 加载特征提取器
        self.feature_extractor = joblib.load(f"{model_dir}/slot_feature_extractor.pkl")
        
        # 加载槽位信息
        with open(f"{model_dir}/slot_info.json", 'r', encoding='utf-8') as f:
            slot_info = json.load(f)
        
        self.slot_names = set(slot_info['slot_names'])
        self.config = slot_info['config']
        
        # 加载分类器模型
        self.models = {}
        for slot_name in self.slot_names:
            classifier_path = f"{model_dir}/{slot_name}_classifier.pkl"
            if os.path.exists(classifier_path):
                self.models[slot_name] = joblib.load(classifier_path)
        
        self.is_trained = True
        logger.info(f"槽位填充器模型已从 {model_dir} 加载")
    
    def _extract_slot_value(self, query: str, slot_name: str) -> Optional[str]:
        """
        从查询中提取槽位值（简单实现）
        
        Args:
            query: 查询文本
            slot_name: 槽位名称
            
        Returns:
            槽位值
        """
        # 简单的关键词匹配规则
        slot_keywords = {
            'entity': ['星际战士', '兽人', '灵族', '帝国', '混沌'],
            'object': ['武器', '装备', '技能', '单位'],
            'entity1': ['兽人', '灵族', '星际战士'],
            'entity2': ['兽人', '灵族', '星际战士']
        }
        
        if slot_name in slot_keywords:
            for keyword in slot_keywords[slot_name]:
                if keyword in query:
                    return keyword
        
        return None


class SlotBuffer:
    """槽位缓存管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化槽位缓存
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_size = config.get('slot_buffer_max_size', 1000)
        self.queries = []
        self.intents = []
        self.slots = []
        self.timestamps = []
        self.is_correct = []
    
    def add_slot_data(self, query: str, intent: str, slots: Dict[str, Slot], 
                     is_correct: bool = True, metadata: Optional[Dict] = None):
        """
        添加槽位数据
        
        Args:
            query: 查询文本
            intent: 意图
            slots: 槽位字典
            is_correct: 是否正确
            metadata: 元数据
        """
        # 检查缓存大小
        if len(self.queries) >= self.max_size:
            self._remove_oldest(100)  # 移除最旧的100条
        
        self.queries.append(query)
        self.intents.append(intent)
        self.slots.append(slots)
        self.timestamps.append(pd.Timestamp.now())
        self.is_correct.append(is_correct)
    
    def get_slot_data(self) -> List[Dict[str, Any]]:
        """
        获取所有槽位数据
        
        Returns:
            槽位数据列表
        """
        data = []
        for i in range(len(self.queries)):
            data.append({
                'query': self.queries[i],
                'intent': self.intents[i],
                'slots': self.slots[i],
                'timestamp': self.timestamps[i],
                'is_correct': self.is_correct[i]
            })
        return data
    
    def remove_slot_data(self, query: str):
        """
        移除指定查询的槽位数据
        
        Args:
            query: 查询文本
        """
        indices_to_remove = []
        for i, q in enumerate(self.queries):
            if q == query:
                indices_to_remove.append(i)
        
        # 从后往前删除，避免索引变化
        for i in reversed(indices_to_remove):
            del self.queries[i]
            del self.intents[i]
            del self.slots[i]
            del self.timestamps[i]
            del self.is_correct[i]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.queries:
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'intent_distribution': {},
                'slot_distribution': {}
            }
        
        total = len(self.queries)
        correct = sum(self.is_correct)
        accuracy = correct / total if total > 0 else 0.0
        
        # 意图分布
        intent_counts = Counter(self.intents)
        
        # 槽位分布
        slot_counts = Counter()
        for slots in self.slots:
            slot_counts.update(slots.keys())
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'intent_distribution': dict(intent_counts),
            'slot_distribution': dict(slot_counts)
        }
    
    def _remove_oldest(self, num_to_remove: int = 100):
        """
        移除最旧的数据
        
        Args:
            num_to_remove: 要移除的数量
        """
        if len(self.queries) <= num_to_remove:
            self.queries.clear()
            self.intents.clear()
            self.slots.clear()
            self.timestamps.clear()
            self.is_correct.clear()
        else:
            self.queries = self.queries[num_to_remove:]
            self.intents = self.intents[num_to_remove:]
            self.slots = self.slots[num_to_remove:]
            self.timestamps = self.timestamps[num_to_remove:]
            self.is_correct = self.is_correct[num_to_remove:]


class SlotFillingEvaluator:
    """槽位填充评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate(self, predictions: List[SlotFillingResult], 
                ground_truths: List[SlotFillingResult]) -> Dict[str, float]:
        """
        评估槽位填充性能
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实结果列表
            
        Returns:
            评估指标字典
        """
        if not predictions and not ground_truths:
            return {
                'slot_f1': 0.0,
                'intent_slot_f1': 0.0,
                'slot_precision': 0.0,
                'slot_recall': 0.0,
                'intent_accuracy': 0.0
            }
        
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实结果数量不匹配")
        
        # 意图准确率
        intent_correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.intent == gt.intent:
                intent_correct += 1
        intent_accuracy = intent_correct / len(predictions)
        
        # 槽位级指标
        all_pred_slots = []
        all_gt_slots = []
        
        for pred, gt in zip(predictions, ground_truths):
            # 收集所有槽位
            pred_slots = set()
            gt_slots = set()
            
            for slot_name, slot in pred.slots.items():
                pred_slots.add(f"{slot_name}:{slot.value}")
            
            for slot_name, slot in gt.slots.items():
                gt_slots.add(f"{slot_name}:{slot.value}")
            
            all_pred_slots.extend(list(pred_slots))
            all_gt_slots.extend(list(gt_slots))
        
        # 计算槽位级指标
        if all_pred_slots or all_gt_slots:
            slot_precision, slot_recall, slot_f1, _ = precision_recall_fscore_support(
                all_gt_slots, all_pred_slots, average='weighted', zero_division=0
            )
        else:
            slot_precision = slot_recall = slot_f1 = 0.0
        
        # 意图+槽位联合F1
        intent_slot_f1 = (intent_accuracy + slot_f1) / 2
        
        return {
            'slot_f1': slot_f1,
            'intent_slot_f1': intent_slot_f1,
            'slot_precision': slot_precision,
            'slot_recall': slot_recall,
            'intent_accuracy': intent_accuracy
        }
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            
        Returns:
            评估报告文本
        """
        report = f"""
槽位填充评估报告
================

意图识别准确率: {metrics['intent_accuracy']:.4f}
槽位填充精确率: {metrics['slot_precision']:.4f}
槽位填充召回率: {metrics['slot_recall']:.4f}
槽位填充F1分数: {metrics['slot_f1']:.4f}
意图+槽位联合F1: {metrics['intent_slot_f1']:.4f}

评估总结:
- 意图识别性能: {'优秀' if metrics['intent_accuracy'] >= 0.9 else '良好' if metrics['intent_accuracy'] >= 0.8 else '一般'}
- 槽位填充性能: {'优秀' if metrics['slot_f1'] >= 0.8 else '良好' if metrics['slot_f1'] >= 0.7 else '一般'}
- 整体性能: {'优秀' if metrics['intent_slot_f1'] >= 0.85 else '良好' if metrics['intent_slot_f1'] >= 0.75 else '一般'}
"""
        return report 