#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估工厂模块
实现检索引擎和Query处理的工厂模式
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class QueryProcessor(ABC):
    """Query处理器基类"""
    
    @abstractmethod
    def process(self, queries: List[str]) -> List[str]:
        """处理query列表"""
        pass


class StraightforwardProcessor(QueryProcessor):
    """直接处理器（不做任何处理）"""
    
    def process(self, queries: List[str]) -> List[str]:
        return queries


class ExpandProcessor(QueryProcessor):
    """扩写处理器（模拟query扩写）"""
    
    def process(self, queries: List[str]) -> List[str]:
        # 这里可以集成真实的query扩写逻辑
        # 目前简单模拟：每个query生成2个扩写版本
        expanded_queries = []
        for query in queries:
            expanded_queries.append(query)  # 原query
            expanded_queries.append(f"{query} 详细")  # 扩写版本1
            expanded_queries.append(f"{query} 相关")  # 扩写版本2
        return expanded_queries


class COTProcessor(QueryProcessor):
    """思维链处理器（模拟Chain of Thought）"""
    
    def process(self, queries: List[str]) -> List[str]:
        # 这里可以集成真实的COT逻辑
        # 目前简单模拟：添加推理步骤
        cot_queries = []
        for query in queries:
            cot_queries.append(query)  # 原query
            cot_queries.append(f"思考过程：{query}")  # COT版本
        return cot_queries


class SearchEngine(ABC):
    """检索引擎基类"""
    
    @abstractmethod
    def search(self, queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """执行检索"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的embedding向量"""
        pass


class DenseSearchEngine(SearchEngine):
    """Dense检索引擎（模拟）"""
    
    def __init__(self):
        # 模拟embedding维度
        self.embedding_dim = 768
    
    def search(self, queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """模拟dense检索"""
        results = []
        for i, query in enumerate(queries):
            # 模拟检索结果
            for j in range(top_k):
                results.append({
                    "doc_id": f"doc_{i}_{j:03d}",
                    "score": np.random.uniform(0.5, 0.95),
                    "query_id": f"q{i}"
                })
        return results
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """模拟获取embedding"""
        return np.random.rand(len(texts), self.embedding_dim)


class SparseSearchEngine(SearchEngine):
    """Sparse检索引擎（模拟）"""
    
    def __init__(self):
        self.embedding_dim = 1000  # 稀疏向量维度
    
    def search(self, queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """模拟sparse检索"""
        results = []
        for i, query in enumerate(queries):
            for j in range(top_k):
                results.append({
                    "doc_id": f"doc_{i}_{j:03d}",
                    "score": np.random.uniform(0.3, 0.8),
                    "query_id": f"q{i}"
                })
        return results
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """模拟获取稀疏embedding"""
        return np.random.rand(len(texts), self.embedding_dim)


class HybridSearchEngine(SearchEngine):
    """混合检索引擎（模拟）"""
    
    def __init__(self):
        self.dense_engine = DenseSearchEngine()
        self.sparse_engine = SparseSearchEngine()
    
    def search(self, queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """模拟混合检索"""
        # 合并dense和sparse的结果
        dense_results = self.dense_engine.search(queries, top_k)
        sparse_results = self.sparse_engine.search(queries, top_k)
        
        # 简单合并，实际应该有更复杂的融合策略
        combined_results = dense_results + sparse_results
        
        # 按分数排序
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results[:top_k * len(queries)]
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取混合embedding"""
        # 返回dense embedding
        return self.dense_engine.get_embeddings(texts)


class QueryProcessorFactory:
    """Query处理器工厂"""
    
    @staticmethod
    def create_processor(processor_type: str) -> QueryProcessor:
        """创建Query处理器"""
        if processor_type == "straightforward":
            return StraightforwardProcessor()
        elif processor_type == "expand":
            return ExpandProcessor()
        elif processor_type == "cot":
            return COTProcessor()
        else:
            raise ValueError(f"不支持的处理器类型: {processor_type}")


class SearchEngineFactory:
    """检索引擎工厂"""
    
    @staticmethod
    def create_engine(engine_type: str) -> SearchEngine:
        """创建检索引擎"""
        if engine_type == "dense":
            return DenseSearchEngine()
        elif engine_type == "sparse":
            return SparseSearchEngine()
        elif engine_type == "hybrid":
            return HybridSearchEngine()
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")


def process_queries_with_factory(queries: List[str], processor_type: str) -> List[str]:
    """
    使用工厂模式处理queries
    
    Args:
        queries: 原始query列表
        processor_type: 处理器类型
        
    Returns:
        List[str]: 处理后的query列表
    """
    processor = QueryProcessorFactory.create_processor(processor_type)
    return processor.process(queries)


def build_retrieved_dict(queries: List[str], engine: SearchEngine, top_k: int) -> Tuple[Dict[str, List[str]], List[float]]:
    """
    构建检索结果字典
    
    Args:
        queries: query列表
        engine: 检索引擎
        top_k: Top-K参数
        
    Returns:
        Tuple[Dict[str, List[str]], List[float]]: (检索结果字典, 分数列表)
    """
    # 执行检索
    search_results = engine.search(queries, top_k)
    
    # 构建结果字典
    retrieved_dict = {}
    all_scores = []
    
    for i, query in enumerate(queries):
        query_id = f"q{i+1}"
        query_results = [r for r in search_results if r['query_id'] == f"q{i}"]
        
        # 提取doc_ids和scores
        doc_ids = [r['doc_id'] for r in query_results]
        scores = [r['score'] for r in query_results]
        
        retrieved_dict[query_id] = doc_ids
        all_scores.extend(scores)
    
    return retrieved_dict, all_scores 