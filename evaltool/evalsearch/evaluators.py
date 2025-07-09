#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估项词典注入模块
将每个评估项抽象为独立的评估器类，支持外部注入
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import os
import traceback
import numpy as np


class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, name: str, description: str, requires_gold: bool = False):
        self.name = name
        self.description = description
        self.requires_gold = requires_gold
    
    @abstractmethod
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        """
        执行评估
        
        Args:
            queries: 原始query列表
            retrieved_dict: 检索结果字典
            all_scores: 所有分数列表
            gold: gold标准答案（可选）
            engine: 检索引擎
            args: 参数
            eval_dir: 评估目录
            
        Returns:
            Tuple[Dict, Optional[str]]: (评估结果, 可视化文件路径)
        """
        pass
    
    def can_evaluate(self, args) -> bool:
        """检查是否可以执行此评估项"""
        return True


class ScoreDistributionEvaluator(BaseEvaluator):
    """分数分布评估器"""
    
    def __init__(self):
        super().__init__(
            name='score_distribution',
            description='计算检索分数分布统计',
            requires_gold=False
        )
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import DenseRetrievalMetrics
        from .visualization import plot_score_distribution
        
        try:
            # 计算分数分布
            result = DenseRetrievalMetrics.score_distribution(all_scores)
            
            # 生成可视化
            viz_path = os.path.join(eval_dir, 'score_distribution.png')
            plot_score_distribution(all_scores, viz_path)
            
            return result, viz_path
        except Exception as e:
            return {'error': str(e)}, None


class EmbeddingDistributionEvaluator(BaseEvaluator):
    """Embedding分布评估器"""
    
    def __init__(self):
        super().__init__(
            name='embedding_distribution',
            description='计算embedding分布可视化',
            requires_gold=False
        )
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import DenseRetrievalMetrics
        from .visualization import plot_embedding_distribution
        
        try:
            # 获取embeddings
            embeddings = engine.get_embeddings(queries)
            coords = DenseRetrievalMetrics.embedding_distribution(embeddings, method="tsne")
            
            result = {
                'method': 'tsne',
                'shape': coords.shape,
                'sample_count': len(queries)
            }
            
            # 生成可视化
            viz_path = os.path.join(eval_dir, 'embedding_distribution.png')
            plot_embedding_distribution(embeddings, viz_path)
            
            return result, viz_path
        except Exception as e:
            return {'error': str(e)}, None


class RecallAtKEvaluator(BaseEvaluator):
    """Recall@K评估器"""
    
    def __init__(self):
        super().__init__(
            name='recall_at_k',
            description='计算Recall@K指标',
            requires_gold=True
        )
    
    def can_evaluate(self, args) -> bool:
        return args.gold is not None
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import DenseRetrievalMetrics
        from .visualization import plot_recall_distribution
        
        if gold is None:
            return {'error': '需要gold标准答案'}, None
        
        try:
            recall_at_k = DenseRetrievalMetrics.recall_at_k(gold, retrieved_dict, args.topk)
            result = {'recall_at_k': recall_at_k}
            
            # 生成可视化
            viz_path = os.path.join(eval_dir, 'recall_distribution.png')
            recall_values = [recall_at_k] * len(queries)  # 简化处理
            plot_recall_distribution(recall_values, viz_path)
            
            return result, viz_path
        except Exception as e:
            return {'error': str(e)}, None


class MRREvaluator(BaseEvaluator):
    """MRR评估器"""
    
    def __init__(self):
        super().__init__(
            name='mrr',
            description='计算MRR (Mean Reciprocal Rank)指标',
            requires_gold=True
        )
    
    def can_evaluate(self, args) -> bool:
        return args.gold is not None
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import DenseRetrievalMetrics
        
        if gold is None:
            return {'error': '需要gold标准答案'}, None
        
        try:
            mrr = DenseRetrievalMetrics.mrr(gold, retrieved_dict, args.topk)
            result = {'mrr': mrr}
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


class PrecisionAtKEvaluator(BaseEvaluator):
    """Precision@K评估器"""
    
    def __init__(self):
        super().__init__(
            name='precision_at_k',
            description='计算Precision@K指标',
            requires_gold=True
        )
    
    def can_evaluate(self, args) -> bool:
        return args.gold is not None
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import DenseRetrievalMetrics
        
        if gold is None:
            return {'error': '需要gold标准答案'}, None
        
        try:
            precision_at_k = DenseRetrievalMetrics.precision_at_k(gold, retrieved_dict, args.topk)
            result = {'precision_at_k': precision_at_k}
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


class OverlapStatsEvaluator(BaseEvaluator):
    """多路召回重叠统计评估器"""
    
    def __init__(self):
        super().__init__(
            name='overlap_stats',
            description='计算多路召回重叠统计',
            requires_gold=False
        )
    
    def can_evaluate(self, args) -> bool:
        return args.se == 'hybrid'
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import HybridRecallMetrics
        from .visualization import plot_overlap_stats
        
        if args.se != 'hybrid':
            return {'error': '需要hybrid检索引擎'}, None
        
        try:
            # 模拟多路召回结果
            dense_results = {f"q{i+1}": retrieved_dict[f"q{i+1}"][:args.topk] for i in range(len(queries))}
            sparse_results = {f"q{i+1}": [f"sparse_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
            graph_results = {f"q{i+1}": [f"graph_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
            
            hybrid_metrics = HybridRecallMetrics(dense_results, sparse_results, graph_results, args.topk)
            overlap_stats = hybrid_metrics.overlap_stats()
            result = {'overlap_stats': overlap_stats}
            
            # 生成可视化
            viz_path = os.path.join(eval_dir, 'overlap_stats.png')
            plot_overlap_stats(overlap_stats, viz_path)
            
            return result, viz_path
        except Exception as e:
            return {'error': str(e)}, None


class HybridUnionRecallEvaluator(BaseEvaluator):
    """多路召回并集覆盖率评估器"""
    
    def __init__(self):
        super().__init__(
            name='hybrid_union_recall',
            description='计算多路召回并集覆盖率',
            requires_gold=True
        )
    
    def can_evaluate(self, args) -> bool:
        return args.gold is not None and args.se == 'hybrid'
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        from .metrics import HybridRecallMetrics
        
        if gold is None:
            return {'error': '需要gold标准答案'}, None
        if args.se != 'hybrid':
            return {'error': '需要hybrid检索引擎'}, None
        
        try:
            # 模拟多路召回结果
            dense_results = {f"q{i+1}": retrieved_dict[f"q{i+1}"][:args.topk] for i in range(len(queries))}
            sparse_results = {f"q{i+1}": [f"sparse_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
            graph_results = {f"q{i+1}": [f"graph_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
            
            union_recall = HybridRecallMetrics.hybrid_union_recall(
                gold, dense_results, sparse_results, graph_results, args.topk
            )
            result = {'hybrid_union_recall': union_recall}
            
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


class EvaluatorRegistry:
    """评估器注册表"""
    
    def __init__(self):
        self._evaluators = {}
        self._register_default_evaluators()
    
    def _register_default_evaluators(self):
        """注册默认评估器"""
        default_evaluators = [
            ScoreDistributionEvaluator(),
            EmbeddingDistributionEvaluator(),
            RecallAtKEvaluator(),
            MRREvaluator(),
            PrecisionAtKEvaluator(),
            OverlapStatsEvaluator(),
            HybridUnionRecallEvaluator(),
        ]
        
        for evaluator in default_evaluators:
            self.register(evaluator)
    
    def register(self, evaluator: BaseEvaluator):
        """注册评估器"""
        self._evaluators[evaluator.name] = evaluator
    
    def get_evaluator(self, name: str) -> Optional[BaseEvaluator]:
        """获取评估器"""
        return self._evaluators.get(name)
    
    def get_available_evaluators(self) -> List[str]:
        """获取所有可用的评估器名称"""
        return list(self._evaluators.keys())
    
    def get_evaluators_for_args(self, args) -> List[BaseEvaluator]:
        """根据参数获取可执行的评估器"""
        available_evaluators = []
        for evaluator in self._evaluators.values():
            if evaluator.can_evaluate(args):
                available_evaluators.append(evaluator)
        return available_evaluators
    
    def execute_evaluation(self, evaluator_name: str, queries: List[str], 
                          retrieved_dict: Dict[str, List[str]], all_scores: List[float],
                          gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        """执行指定评估器"""
        evaluator = self.get_evaluator(evaluator_name)
        if evaluator is None:
            return {'error': f'未找到评估器: {evaluator_name}'}, None
        
        if not evaluator.can_evaluate(args):
            return {'error': f'评估器{evaluator_name}不满足执行条件'}, None
        
        return evaluator.evaluate(queries, retrieved_dict, all_scores, gold, engine, args, eval_dir)


# 全局评估器注册表实例
evaluator_registry = EvaluatorRegistry() 