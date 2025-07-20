import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化意图分类器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.classes_ = None
        self.feature_names_ = None
        
    def fit(self, X: np.ndarray, y: List[str]):
        """
        训练意图分类器
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 意图标签列表
        """
        logger.info("开始训练意图分类器")
        
        # 验证输入数据
        self._validate_input_data(X, y)
        
        # 选择分类器类型
        self.model = self._select_classifier()
        
        # 保存类别信息
        self.classes_ = sorted(list(set(y)))
        
        # 训练模型
        self.model.fit(X, y)
        
        self.is_trained = True
        logger.info("意图分类器训练完成")
        
    def predict(self, X: np.ndarray) -> List[str]:
        """
        预测意图标签
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            预测的意图标签列表
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 实现预测逻辑
        predictions = self.model.predict(X)
        return predictions.tolist()
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测意图概率
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            预测概率矩阵 (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 实现概率预测逻辑
        probabilities = self.model.predict_proba(X)
        return probabilities
        
    def evaluate(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实意图标签列表
            
        Returns:
            评估结果字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 实现评估逻辑
        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        return metrics
        
    def save_model(self, file_path: str):
        """
        保存模型
        
        Args:
            file_path: 模型保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 实现模型保存逻辑
        model_data = {
            'model': self.model,
            'classes_': self.classes_,
            'config': self.config
        }
        joblib.dump(model_data, file_path)
        
    def load_model(self, file_path: str):
        """
        加载模型
        
        Args:
            file_path: 模型文件路径
        """
        # 实现模型加载逻辑
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.classes_ = model_data['classes_']
        self.config = model_data['config']
        
        # 设置训练完成标志
        self.is_trained = True
    
    def _validate_input_data(self, X: np.ndarray, y: List[str]):
        """
        验证输入数据
        
        Args:
            X: 特征矩阵
            y: 标签列表
        """
        # 检查数据是否为空
        if len(X) == 0 or len(y) == 0:
            raise ValueError("训练数据不能为空")
        
        # 检查数据维度匹配
        if len(X) != len(y):
            raise ValueError(f"特征数量({len(X)})与标签数量({len(y)})不匹配")
        
        # 检查是否有多个类别
        unique_classes = set(y)
        if len(unique_classes) < 2:
            raise ValueError(f"至少需要2个类别，当前只有{len(unique_classes)}个类别")
    
    def _select_classifier(self) -> Any:
        """
        根据配置选择分类器
        Returns:
            分类器实例
        """
        classifier_type = self.config.get('classifier_type', 'logistic')
        random_state = self.config.get('random_state', 42)
        max_iter = self.config.get('max_iter', 1000)
        n_estimators = self.config.get('n_estimators', 100)
        if classifier_type == 'logistic':
            return LogisticRegression(random_state=random_state, max_iter=max_iter)
        elif classifier_type == 'svm':
            return SVC(random_state=random_state, probability=True, max_iter=max_iter)
        elif classifier_type == 'random_forest':
            return RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    def _calculate_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            评估指标字典
        """
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率、召回率、F1分数
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        } 