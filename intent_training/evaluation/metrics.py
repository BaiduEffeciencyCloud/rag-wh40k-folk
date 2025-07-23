from typing import List, Dict, Any, Tuple
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


class MetricCalculator:
    """
    负责计算各种评估指标。
    所有方法设计为静态方法，方便独立调用。
    """
    @staticmethod
    def calculate_intent_accuracy(y_true: List[str], y_pred: List[str]) -> float:
        """
        计算意图识别的准确率。

        Args:
            y_true (List[str]): 真实的意图标签列表。
            y_pred (List[str]): 预测的意图标签列表。

        Returns:
            float: 意图识别的准确率。
        """
        if not y_true and not y_pred:
            return 1.0
        if len(y_true) != len(y_pred) or not y_true:
            return 0.0
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    @staticmethod
    def calculate_slot_f1(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
        """
        计算槽位填充的F1分数、精确率和召回率。
        使用seqeval库进行计算，支持实体级别的评估。

        Args:
            y_true (List[List[str]]): 真实的槽位标签列表 (BIO格式)。
            y_pred (List[List[str]]): 预测的槽位标签列表 (BIO格式)。

        Returns:
            Dict[str, Any]: 包含各个实体类别和总体平均的 precision, recall, f1-score 的字典。
        """
        if not y_true or not y_pred:
            return {}
        if len(y_true) != len(y_pred):
            raise ValueError("输入列表长度不一致")

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return report

    @staticmethod
    def calculate_frame_accuracy(intent_true: List[str], intent_pred: List[str], 
                                 slot_true: List[List[str]], slot_pred: List[List[str]]) -> float:
        """
        计算 Frame Accuracy (北极星指标)
        
        仅当意图和所有槽位都完全正确时，才算作一次成功的预测。
        
        Args:
            intent_true: 真实意图标签列表
            intent_pred: 预测意图标签列表
            slot_true: 真实槽位字典列表 (e.g., [{'loc': '北京'}, ...])
            slot_pred: 预测槽位字典列表
            
        Returns:
            Frame Accuracy (0.0 ~ 1.0)
            
        Raises:
            ValueError: 如果输入列表长度不一致
        """
        if not (len(intent_true) == len(intent_pred) == len(slot_true) == len(slot_pred)):
            return 0.0

        correct_frames = 0
        total_frames = len(intent_true)
        for i in range(total_frames):
            if intent_true[i] == intent_pred[i] and slot_true[i] == slot_pred[i]:
                correct_frames += 1
        
        return correct_frames / total_frames if total_frames > 0 else 0.0 

    @staticmethod
    def calculate_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        计算意图识别的混淆矩阵。

        Args:
            y_true (List[str]): 真实的意图标签列表。
            y_pred (List[str]): 预测的意图标签列表。

        Returns:
            Dict[str, Any]: 包含标签和矩阵的字典。
                            {
                                "labels": ["intent1", "intent2", ...],
                                "matrix": [[...], [...], ...]
                            }
        """
        if len(y_true) != len(y_pred):
            raise ValueError("输入列表长度不一致")
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            "labels": labels,
            "matrix": matrix.tolist()
        } 