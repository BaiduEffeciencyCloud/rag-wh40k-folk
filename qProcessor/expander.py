#使用query扩展的query processor 
from .processorinterface import QueryProcessorInterface
from typing import Dict, Any, List

class ExpanderQueryProcessor(QueryProcessorInterface):
    """查询扩写处理器，生成多个相关查询"""
    
    def __init__(self):
        # 初始化扩展策略
        self.expansion_strategies = [
            self._add_related_terms,
            self._add_synonyms,
            self._add_context_terms
        ]
    
    def process(self, query: str) -> List[Dict[str, Any]]:
        """
        对查询进行扩写，生成多个相关查询
        Args:
            query: 用户输入的查询
        Returns:
            包含多个扩写查询的字典列表
        """
        expanded_queries = self._expand_query(query)
        return [
            {
                "text": expanded_query,
                "entities": [],
                "relations": [],
                "processed": True,
                "type": "expanded",
                "original_query": query,
                "expansion_method": method
            }
            for expanded_query, method in expanded_queries
        ]
    
    def get_type(self) -> str:
        return "expander"
    
    def is_multi_query(self) -> bool:
        return True
    
    def _expand_query(self, query: str) -> List[tuple]:
        """执行查询扩写"""
        expanded_queries = []
        
        # 添加原始查询
        expanded_queries.append((query, "original"))
        
        # 应用各种扩写策略
        for strategy in self.expansion_strategies:
            expanded = strategy(query)
            if expanded and expanded != query:
                expanded_queries.append((expanded, strategy.__name__))
        
        return expanded_queries
    
    def _add_related_terms(self, query: str) -> str:
        """添加相关词"""
        # 这里可以实现更复杂的相关词查找逻辑
        related_terms = {
            "阿巴顿": "混沌领主 黑色军团",
            "黑色远征": "荷鲁斯叛乱 帝国",
            "战锤40k": "科幻 战争 宇宙"
        }
        
        for key, terms in related_terms.items():
            if key in query:
                return f"{query} {terms}"
        
        return query
    
    def _add_synonyms(self, query: str) -> str:
        """添加同义词"""
        synonyms = {
            "阿巴顿": "阿巴顿·德西乌斯",
            "黑色远征": "黑色十字军",
            "战锤": "Warhammer"
        }
        
        for original, synonym in synonyms.items():
            if original in query:
                return query.replace(original, f"{original} {synonym}")
        
        return query
    
    def _add_context_terms(self, query: str) -> str:
        """添加上下文词"""
        context_terms = {
            "阿巴顿": "背景 历史 故事",
            "黑色远征": "事件 战役 影响",
            "战锤40k": "设定 世界观 背景"
        }
        
        for key, terms in context_terms.items():
            if key in query:
                return f"{query} {terms}"
        
        return query



