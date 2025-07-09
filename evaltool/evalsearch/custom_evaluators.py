#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义评估器示例
展示如何外部注入新的评估项
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from .evaluators import BaseEvaluator


class CustomDiversityEvaluator(BaseEvaluator):
    """自定义多样性评估器"""
    
    def __init__(self):
        super().__init__(
            name='custom_diversity',
            description='计算检索结果的多样性指标',
            requires_gold=False
        )
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        """计算多样性指标"""
        try:
            # 计算每个query的检索结果多样性
            diversity_scores = []
            for qid, doc_ids in retrieved_dict.items():
                # 简单的多样性计算：不同doc_id的数量
                unique_docs = len(set(doc_ids))
                diversity_scores.append(unique_docs / len(doc_ids) if doc_ids else 0)
            
            result = {
                'avg_diversity': np.mean(diversity_scores),
                'min_diversity': np.min(diversity_scores),
                'max_diversity': np.max(diversity_scores),
                'diversity_scores': diversity_scores
            }
            
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


class CustomLatencyEvaluator(BaseEvaluator):
    """自定义延迟评估器"""
    
    def __init__(self):
        super().__init__(
            name='custom_latency',
            description='模拟检索延迟评估',
            requires_gold=False
        )
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        """模拟延迟评估"""
        try:
            import time
            import random
            
            # 模拟检索延迟
            latencies = []
            for _ in range(len(queries)):
                # 模拟50-200ms的延迟
                latency = random.uniform(0.05, 0.2)
                latencies.append(latency)
                time.sleep(0.001)  # 避免太快
            
            result = {
                'avg_latency_ms': np.mean(latencies) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'latencies': [l * 1000 for l in latencies]
            }
            
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


class CustomNoveltyEvaluator(BaseEvaluator):
    """自定义新颖性评估器"""
    
    def __init__(self):
        super().__init__(
            name='custom_novelty',
            description='计算检索结果的新颖性（基于gold）',
            requires_gold=True
        )
    
    def can_evaluate(self, args) -> bool:
        return args.gold is not None
    
    def evaluate(self, queries: List[str], retrieved_dict: Dict[str, List[str]], 
                all_scores: List[float], gold: Optional[Dict], engine, args, eval_dir: str) -> Tuple[Dict, Optional[str]]:
        """计算新颖性指标"""
        if gold is None:
            return {'error': '需要gold标准答案'}, None
        
        try:
            novelty_scores = []
            for qid, doc_ids in retrieved_dict.items():
                if qid not in gold:
                    continue
                
                gold_docs = set(gold[qid])
                retrieved_docs = set(doc_ids[:args.topk])
                
                # 新颖性 = 非gold文档数量 / 总检索文档数量
                novel_docs = retrieved_docs - gold_docs
                novelty = len(novel_docs) / len(retrieved_docs) if retrieved_docs else 0
                novelty_scores.append(novelty)
            
            result = {
                'avg_novelty': np.mean(novelty_scores),
                'min_novelty': np.min(novelty_scores),
                'max_novelty': np.max(novelty_scores),
                'novelty_scores': novelty_scores
            }
            
            return result, None
        except Exception as e:
            return {'error': str(e)}, None


# 注册自定义评估器的函数
def register_custom_evaluators():
    """注册所有自定义评估器"""
    from .evaluators import evaluator_registry
    
    custom_evaluators = [
        CustomDiversityEvaluator(),
        CustomLatencyEvaluator(),
        CustomNoveltyEvaluator(),
    ]
    
    for evaluator in custom_evaluators:
        evaluator_registry.register(evaluator)
        print(f"注册自定义评估器: {evaluator.name} - {evaluator.description}")


# 使用示例
if __name__ == "__main__":
    # 注册自定义评估器
    register_custom_evaluators()
    
    # 查看所有可用的评估器
    from .evaluators import evaluator_registry
    print("\n所有可用评估器:")
    for name in evaluator_registry.get_available_evaluators():
        evaluator = evaluator_registry.get_evaluator(name)
        print(f"  - {name}: {evaluator.description}") 