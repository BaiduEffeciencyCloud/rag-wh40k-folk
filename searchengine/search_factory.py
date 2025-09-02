from typing import Dict, Any, Optional
from .search_interface import SearchEngineInterface
from .dense_search import DenseSearchEngine
from .hybrid_search import HybridSearchEngine
from .sparse_search import SparseSearchEngine

class SearchEngineFactory:
    """搜索引擎工厂类"""
    
    # 支持的搜索引擎类型
    SUPPORTED_TYPES = {
        'dense': DenseSearchEngine,
        'hybrid': HybridSearchEngine,
        'sparse': SparseSearchEngine
    }
    
    @classmethod
    def create(cls, engine_type: str = 'hybrid', **kwargs) -> SearchEngineInterface:
        """
        创建搜索引擎实例
        Args:
            engine_type: 搜索引擎类型 ('dense' 或 'hybrid')
            **kwargs: 初始化参数
        Returns:
            搜索引擎实例
        """
        if engine_type not in cls.SUPPORTED_TYPES:
            raise ValueError(f"不支持的搜索引擎类型: {engine_type}。支持的类型: {list(cls.SUPPORTED_TYPES.keys())}")
        
        engine_class = cls.SUPPORTED_TYPES[engine_type]
        return engine_class(**kwargs)
    
    @classmethod
    def get_supported_types(cls) -> Dict[str, str]:
        """获取支持的搜索引擎类型"""
        return {
            'dense': 'Dense向量搜索引擎',
            'hybrid': 'Hybrid混合搜索引擎（推荐）',
            'sparse': 'Sparse稀疏搜索引擎'
        }
    
 