"""
序列槽位填充器
使用CRF进行序列标注，实现精确的槽位边界检测
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import joblib
import json
import os
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split

from .bio_annotator import BIOAnnotator
from .crf_feature_extractor import CRFFeatureExtractor

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


class SequenceSlotFiller:
    """序列槽位填充器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化序列槽位填充器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.bio_annotator = BIOAnnotator()
        self.feature_extractor = CRFFeatureExtractor(config)
        
        # CRF模型配置
        crf_config = config.get('crf_model', {})
        self.crf = CRF(
            algorithm=crf_config.get('algorithm', 'lbfgs'),
            c1=crf_config.get('c1', 0.1),
            c2=crf_config.get('c2', 0.1),
            max_iterations=crf_config.get('max_iterations', 100),
            all_possible_transitions=crf_config.get('all_possible_transitions', True)
        )
        
        self.is_trained = False
        self.label_set = set()
        self.slot_names = set()
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        训练序列槽位填充器
        
        Args:
            training_data: 训练数据列表
        """
        logger.info("开始训练序列槽位填充器")
        
        # 准备训练数据
        queries = [item['query'] for item in training_data]
        slot_labels = [item['slots'] for item in training_data]
        
        # 转换为BIO标注
        bio_training_data = self.bio_annotator.create_training_data(queries, slot_labels)
        
        if not bio_training_data:
            raise ValueError("没有有效的BIO训练数据")
        
        # 提取特征
        X_train, y_train = self._prepare_training_features(bio_training_data)
        
        # 训练CRF模型
        logger.info(f"训练CRF模型，样本数: {len(X_train)}")
        self.crf.fit(X_train, y_train)
        
        # 收集标签集
        self.label_set = set()
        for labels in y_train:
            self.label_set.update(labels)
        
        # 收集槽位名称
        self.slot_names = set()
        for label in self.label_set:
            if label.startswith('B-') or label.startswith('I-'):
                self.slot_names.add(label[2:])
        
        self.is_trained = True
        logger.info(f"序列槽位填充器训练完成，标签数: {len(self.label_set)}, 槽位数: {len(self.slot_names)}")
    
    def _prepare_training_features(self, bio_training_data: List[Tuple[List[str], List[str]]]) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
        """
        准备训练特征
        
        Args:
            bio_training_data: BIO训练数据
            
        Returns:
            特征序列和标签序列
        """
        X_train = []
        y_train = []
        
        for tokens, bio_labels in bio_training_data:
            # 提取特征
            features = self.feature_extractor.extract_sequence_features(tokens)
            X_train.append(features)
            y_train.append(bio_labels)
        
        return X_train, y_train
    
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
            raise ValueError("序列槽位填充器尚未训练")
        
        # 转换为token序列
        tokens = list(query)
        
        # 提取特征
        features = self.feature_extractor.extract_sequence_features(tokens)
        
        # 预测BIO标签
        bio_labels = self.crf.predict([features])[0]
        
        # 转换为槽位字典
        slot_dict = self.bio_annotator.convert_bio_to_slots(query, bio_labels)
        
        # 转换为Slot对象
        slots = {}
        for slot_name, slot_info_list in slot_dict.items():
            for slot_value, start_pos, end_pos in slot_info_list:
                # 计算置信度（这里使用简单的启发式方法）
                confidence = self._calculate_slot_confidence(slot_value, slot_name, bio_labels)
                
                slot = Slot(
                    name=slot_name,
                    value=slot_value,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence
                )
                slots[slot_name] = slot
        
        # 计算整体置信度
        overall_confidence = self._calculate_overall_confidence(slots, bio_labels)
        
        return SlotFillingResult(
            intent=intent,
            slots=slots,
            confidence=overall_confidence,
            raw_query=query
        )
    
    def _calculate_slot_confidence(self, slot_value: str, slot_name: str, bio_labels: List[str]) -> float:
        """
        计算槽位置信度
        
        Args:
            slot_value: 槽位值
            slot_name: 槽位名称
            bio_labels: BIO标签序列
            
        Returns:
            置信度
        """
        # 简单的置信度计算：基于标签的连续性
        b_count = sum(1 for label in bio_labels if label == f'B-{slot_name}')
        i_count = sum(1 for label in bio_labels if label == f'I-{slot_name}')
        
        total_slot_tokens = b_count + i_count
        if total_slot_tokens == 0:
            return 0.0
        
        # 基础置信度
        base_confidence = min(1.0, total_slot_tokens / len(slot_value))
        
        # 连续性奖励
        continuity_bonus = 0.1 if i_count > 0 else 0.0
        
        return min(1.0, base_confidence + continuity_bonus)
    
    def _calculate_overall_confidence(self, slots: Dict[str, Slot], bio_labels: List[str]) -> float:
        """
        计算整体置信度
        
        Args:
            slots: 槽位字典
            bio_labels: BIO标签序列
            
        Returns:
            整体置信度
        """
        if not slots:
            return 0.0
        
        # 计算平均槽位置信度
        slot_confidences = [slot.confidence for slot in slots.values()]
        avg_slot_confidence = np.mean(slot_confidences)
        
        # 计算标签质量（O标签比例）
        o_count = sum(1 for label in bio_labels if label == 'O')
        label_quality = o_count / len(bio_labels)
        
        # 综合置信度
        overall_confidence = 0.7 * avg_slot_confidence + 0.3 * label_quality
        
        return min(1.0, overall_confidence)
    
    

    
    def save_model(self, file_path: str):
        """
        保存模型
        
        Args:
            file_path: 模型保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 直接保存CRF模型到指定路径（目录已由pipeline创建）
        joblib.dump(self.crf, file_path)
        
        # 保存模型信息到同目录下的JSON文件
        model_dir = os.path.dirname(file_path)
        model_info = {
            'is_trained': self.is_trained,
            'label_set': list(self.label_set),
            'slot_names': list(self.slot_names),
            'config': self.config
        }
        
        # 使用固定的info文件名，因为它不在config.yaml中定义
        info_path = os.path.join(model_dir, 'sequence_slot_filler_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"序列槽位填充器模型已保存到: {file_path}")
    
    def load_model(self, file_path: str):
        """
        加载模型
        
        Args:
            file_path: 模型文件路径 (例如: .../sequence_slot_filler.pkl)
        """
        # 1. 加载CRF模型本身
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"序列槽位填充器模型文件不存在: {file_path}")
        
        self.crf = joblib.load(file_path)
        
        # 2. 根据模型文件路径推断并加载附带的info.json文件
        model_dir = os.path.dirname(file_path)
        # info文件名是固定的，与save_model保持一致
        info_path = os.path.join(model_dir, 'sequence_slot_filler_info.json')
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"模型信息文件不存在: {info_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # 3. 恢复模型状态
        self.is_trained = model_info.get('is_trained', False)
        self.label_set = set(model_info.get('label_set', []))
        self.slot_names = set(model_info.get('slot_names', []))
        # 优先使用加载的config，因为它包含了训练时的特定参数
        self.config = model_info.get('config', self.config)
        
        # 4. 重新初始化依赖组件
        self.bio_annotator = BIOAnnotator()
        self.feature_extractor = CRFFeatureExtractor(self.config)
        
        logger.info(f"序列槽位填充器模型已从 {file_path} 加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'is_trained': self.is_trained,
            'label_set': list(self.label_set),
            'slot_names': list(self.slot_names),
            'feature_names': self.feature_extractor.get_feature_names(),
            'config': self.config
        } 