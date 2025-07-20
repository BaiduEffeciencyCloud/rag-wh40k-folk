import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UncertaintyDetector:
    """不确定性检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        unc_cfg = config.get('uncertainty', {})
        self.methods = unc_cfg.get('methods', ['entropy', 'confidence', 'margin'])
        self.weights = unc_cfg.get('combined_weight', {
            'entropy': 0.4,
            'confidence': 0.4,
            'margin': 0.2
        })
        self.review_threshold = unc_cfg.get('review_threshold', 0.5)
        self.auto_confidence_threshold = unc_cfg.get('auto_confidence_threshold', 0.85)
        
    def detect(self, features: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        检测不确定性
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            
        Returns:
            不确定性检测结果字典
        """
        logger.info("开始不确定性检测")
        
        # 计算各种不确定性指标
        uncertainty_scores = {}
        
        if 'entropy' in self.methods:
            uncertainty_scores['entropy'] = self.entropy_uncertainty(probabilities)
            
        if 'confidence' in self.methods:
            uncertainty_scores['confidence'] = self.confidence_uncertainty(probabilities)
            
        if 'margin' in self.methods:
            uncertainty_scores['margin'] = self.margin_uncertainty(probabilities)
        
        # 计算综合不确定性分数
        combined_uncertainty = self.combine_uncertainty_scores(uncertainty_scores)
        
        # 判断是否需要审核
        should_review = self.should_review(combined_uncertainty)
        
        # 判断是否可以直接信任
        is_auto_confidence = self.is_auto_confidence(probabilities)
        
        return {
            'entropy_uncertainty': uncertainty_scores.get('entropy', np.array([])),
            'confidence_uncertainty': uncertainty_scores.get('confidence', np.array([])),
            'margin_uncertainty': uncertainty_scores.get('margin', np.array([])),
            'combined_uncertainty': combined_uncertainty,
            'should_review': should_review,
            'is_auto_confidence': is_auto_confidence
        }
        
    def entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        基于熵的不确定性
        
        Args:
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            
        Returns:
            熵不确定性分数 (n_samples,)
        """
        # 避免log(0)
        eps = 1e-10
        safe_probs = probabilities + eps
        
        # 计算熵: -sum(p * log(p))
        entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1)
        
        # 归一化到[0, 1]范围
        max_entropy = np.log(probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
        
    def confidence_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        基于置信度的不确定性
        
        Args:
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            
        Returns:
            置信度不确定性分数 (n_samples,)
        """
        # 获取最高概率
        max_probs = np.max(probabilities, axis=1)
        
        # 置信度不确定性 = 1 - 最高概率
        confidence_uncertainty = 1 - max_probs
        
        return confidence_uncertainty
        
    def margin_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        基于边距的不确定性
        
        Args:
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            
        Returns:
            边距不确定性分数 (n_samples,)
        """
        # 对每个样本的概率排序
        sorted_probs = np.sort(probabilities, axis=1)
        
        # 计算最高概率和第二高概率的差值
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # 边距不确定性 = 1 - 边距
        margin_uncertainty = 1 - margin
        
        return margin_uncertainty
        
    def combine_uncertainty_scores(self, uncertainty_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        组合多种不确定性分数
        
        Args:
            uncertainty_scores: 各种不确定性分数字典
            
        Returns:
            综合不确定性分数 (n_samples,)
        """
        if not uncertainty_scores:
            return np.array([])
        
        # 获取样本数量
        n_samples = next(iter(uncertainty_scores.values())).shape[0]
        
        # 初始化综合分数
        combined = np.zeros(n_samples)
        
        # 加权组合
        for method, scores in uncertainty_scores.items():
            if method in self.weights:
                weight = self.weights[method]
                combined += weight * scores
        
        return combined
        
    def should_review(self, combined_uncertainty: np.ndarray) -> np.ndarray:
        """
        判断是否需要人工审核
        
        Args:
            combined_uncertainty: 综合不确定性分数 (n_samples,)
            
        Returns:
            是否需要审核的布尔数组 (n_samples,)
        """
        return combined_uncertainty > self.review_threshold
        
    def is_auto_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        判断是否可以直接信任（无需审核）
        
        Args:
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            
        Returns:
            是否可以直接信任的布尔数组 (n_samples,)
        """
        # 获取最高概率
        max_probs = np.max(probabilities, axis=1)
        
        # 判断是否超过自动信任阈值
        return max_probs >= self.auto_confidence_threshold


class QueryBuffer:
    """查询缓存，用于存储需要人工审核的查询"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        unc_cfg = config.get('uncertainty', {})
        self.max_size = unc_cfg.get('buffer_max_size', 1000)
        self.queries = []
        self.predictions = []
        self.uncertainty_scores = []
        self.timestamps = []
        self.metadata = []
        
    def add_query(self, query: str, prediction: str, uncertainty_score: float, metadata: Optional[Dict] = None):
        """
        添加需要审核的查询
        
        Args:
            query: 查询文本
            prediction: 预测结果
            uncertainty_score: 不确定性分数
            metadata: 额外元数据
        """
        # 添加查询信息
        self.queries.append(query)
        self.predictions.append(prediction)
        self.uncertainty_scores.append(uncertainty_score)
        self.timestamps.append(datetime.now())
        self.metadata.append(metadata or {})
        
        # 维护缓存大小
        if len(self.queries) > self.max_size:
            self._remove_oldest(len(self.queries) - self.max_size)
        
    def get_batch_for_review(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        获取需要审核的批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            需要审核的查询列表
        """
        if not self.queries:
            return []
        
        # 创建索引列表
        indices = list(range(len(self.queries)))
        
        # 按不确定性分数降序排序
        indices.sort(key=lambda i: self.uncertainty_scores[i], reverse=True)
        
        # 获取指定数量的查询
        batch_indices = indices[:batch_size]
        
        # 构建批次数据
        batch = []
        for idx in batch_indices:
            batch.append({
                'index': idx,
                'query': self.queries[idx],
                'prediction': self.predictions[idx],
                'uncertainty_score': self.uncertainty_scores[idx],
                'timestamp': self.timestamps[idx],
                'metadata': self.metadata[idx]
            })
        
        return batch
        
    def remove_queries(self, indices: List[int]):
        """
        移除已处理的查询
        
        Args:
            indices: 要移除的查询索引列表
        """
        # 按索引降序排序，避免删除时索引变化
        indices = sorted(indices, reverse=True)
        
        # 移除指定索引的查询
        for idx in indices:
            if 0 <= idx < len(self.queries):
                del self.queries[idx]
                del self.predictions[idx]
                del self.uncertainty_scores[idx]
                del self.timestamps[idx]
                del self.metadata[idx]
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self.uncertainty_scores:
            return {
                'total_queries': 0,
                'avg_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'buffer_utilization': 0.0
            }
        
        uncertainty_array = np.array(self.uncertainty_scores)
        
        return {
            'total_queries': len(self.queries),
            'avg_uncertainty': float(np.mean(uncertainty_array)),
            'max_uncertainty': float(np.max(uncertainty_array)),
            'min_uncertainty': float(np.min(uncertainty_array)),
            'buffer_utilization': len(self.queries) / self.max_size
        }
        
    def _remove_oldest(self, num_to_remove: int = 100):
        """
        移除最旧的查询
        
        Args:
            num_to_remove: 要移除的数量
        """
        # 移除最旧的查询（列表前面的元素）
        for _ in range(min(num_to_remove, len(self.queries))):
            if self.queries:
                self.queries.pop(0)
                self.predictions.pop(0)
                self.uncertainty_scores.pop(0)
                self.timestamps.pop(0)
                self.metadata.pop(0) 