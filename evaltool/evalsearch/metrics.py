#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估指标核心模块
基于20250707_rag检索引擎架构设计文档6.2.1章节实现
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

# 尝试导入UMAP，如果不存在则跳过
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available, UMAP visualization will be skipped")


class DenseRetrievalMetrics:
    """Dense检索评估指标类"""
    
    def __init__(self, gold: Dict[str, List[str]], retrieved: Dict[str, List[str]], k: int = 10):
        """
        初始化Dense检索评估指标
        
        Args:
            gold: 标准答案字典 {query_id: [doc_ids]}
            retrieved: 检索结果字典 {query_id: [doc_ids]}
            k: Top-K评估参数
        """
        self.gold = gold
        self.retrieved = retrieved
        self.k = k
    
    @staticmethod
    def recall_at_k(gold: Dict[str, List[str]], retrieved: Dict[str, List[str]], k: int = 5) -> float:
        """
        计算Recall@K指标
        
        Args:
            gold: 标准答案字典
            retrieved: 检索结果字典
            k: Top-K参数
            
        Returns:
            float: Recall@K值
        """
        if not gold or not retrieved:
            return 0.0
        
        total_recall = 0
        for qid in gold:
            if qid not in retrieved:
                continue
            gold_docs = set(gold[qid])
            retrieved_docs = retrieved[qid][:k]
            if any(doc in gold_docs for doc in retrieved_docs):
                total_recall += 1
        
        return total_recall / len(gold)
    
    @staticmethod
    def mrr(gold: Dict[str, List[str]], retrieved: Dict[str, List[str]], k: int = 10) -> float:
        """
        计算MRR (Mean Reciprocal Rank)指标
        
        Args:
            gold: 标准答案字典
            retrieved: 检索结果字典
            k: Top-K参数
            
        Returns:
            float: MRR值
        """
        if not gold or not retrieved:
            return 0.0
        
        mrr_total = 0
        for qid in gold:
            if qid not in retrieved:
                continue
            gold_docs = set(gold[qid])
            for rank, doc_id in enumerate(retrieved[qid][:k], 1):
                if doc_id in gold_docs:
                    mrr_total += 1 / rank
                    break
        
        return mrr_total / len(gold)
    
    @staticmethod
    def precision_at_k(gold: Dict[str, List[str]], retrieved: Dict[str, List[str]], k: int = 5) -> float:
        """
        计算Precision@K指标
        
        Args:
            gold: 标准答案字典
            retrieved: 检索结果字典
            k: Top-K参数
            
        Returns:
            float: Precision@K值
        """
        if not gold or not retrieved:
            return 0.0
        
        total_precision = 0
        for qid in gold:
            if qid not in retrieved:
                continue
            gold_docs = set(gold[qid])
            retrieved_docs = retrieved[qid][:k]
            relevant = sum(1 for doc in retrieved_docs if doc in gold_docs)
            total_precision += relevant / k
        
        return total_precision / len(gold)
    
    @staticmethod
    def score_distribution(score_list: List[float], bins: int = 50) -> Dict:
        """
        计算分数分布统计
        
        Args:
            score_list: 分数列表
            bins: 直方图分箱数
            
        Returns:
            Dict: 包含hist和bin_edges的字典
        """
        if not score_list:
            # 空数据兜底
            hist = [0] * bins
            bin_edges = np.linspace(0, 1, bins + 1)
            return {'hist': hist, 'bin_edges': bin_edges.tolist()}
        
        hist, bin_edges = np.histogram(score_list, bins=bins, range=(0, 1))
        return {
            'hist': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean': np.mean(score_list),
            'std': np.std(score_list),
            'min': np.min(score_list),
            'max': np.max(score_list)
        }
    
    @staticmethod
    def embedding_distribution(embeddings: np.ndarray, method: str = "tsne", 
                             n_components: int = 2) -> np.ndarray:
        """
        计算embedding分布（降维可视化）
        
        Args:
            embeddings: embedding向量矩阵
            method: 降维方法 ("tsne" 或 "umap")
            n_components: 降维后的维度
            
        Returns:
            np.ndarray: 降维后的坐标
        """
        if method.lower() == "tsne":
            tsne = TSNE(n_components=n_components, random_state=42)
            return tsne.fit_transform(embeddings)
        elif method.lower() == "umap":
            if not UMAP_AVAILABLE:
                raise ValueError("UMAP not available. Please install umap-learn package.")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: tsne, umap")


class HybridRecallMetrics:
    """多路召回评估指标类"""
    
    def __init__(self, dense: Dict[str, List[str]], sparse: Dict[str, List[str]], 
                 graph: Dict[str, List[str]], k: int = 10):
        """
        初始化多路召回评估指标
        
        Args:
            dense: Dense检索结果
            sparse: Sparse检索结果
            graph: Graph检索结果
            k: Top-K参数
        """
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.k = k
    
    def overlap_stats(self) -> Dict[str, Dict[str, int]]:
        """
        计算多路召回的重叠统计
        
        Returns:
            Dict: 每个query的重叠和并集统计
        """
        overlap_stats = {}
        
        for qid in self.dense:
            if qid not in self.sparse or qid not in self.graph:
                continue
            
            dense_set = set(self.dense[qid][:self.k])
            sparse_set = set(self.sparse[qid][:self.k])
            graph_set = set(self.graph[qid][:self.k])
            
            # 计算交集
            overlap = len(dense_set & sparse_set & graph_set)
            # 计算并集
            union = len(dense_set | sparse_set | graph_set)
            
            overlap_stats[qid] = {
                'overlap': overlap,
                'union': union,
                'dense_count': len(dense_set),
                'sparse_count': len(sparse_set),
                'graph_count': len(graph_set)
            }
        
        return overlap_stats
    
    @staticmethod
    def hybrid_union_recall(gold: Dict[str, List[str]], dense: Dict[str, List[str]], 
                           sparse: Dict[str, List[str]], graph: Dict[str, List[str]], 
                           k: int = 10) -> float:
        """
        计算多路召回的并集覆盖率
        
        Args:
            gold: 标准答案字典
            dense: Dense检索结果
            sparse: Sparse检索结果
            graph: Graph检索结果
            k: Top-K参数
            
        Returns:
            float: 并集覆盖率
        """
        if not gold:
            return 0.0
        
        total_coverage = 0
        for qid in gold:
            if qid not in dense or qid not in sparse or qid not in graph:
                continue
            
            gold_docs = set(gold[qid])
            dense_set = set(dense[qid][:k])
            sparse_set = set(sparse[qid][:k])
            graph_set = set(graph[qid][:k])
            
            hybrid_set = dense_set | sparse_set | graph_set
            if any(doc in gold_docs for doc in hybrid_set):
                total_coverage += 1
        
        return total_coverage / len(gold) 