import os
import sys
from typing import Dict, Any, List, Union
import pinecone
from openai import OpenAI
import logging

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, EMBADDING_MODEL, HYBRID_ALPHA, RERANK_MODEL
from .search_interface import SearchEngineInterface
from .dense_search import DenseSearchEngine
from .base_search import BaseSearchEngine

logger = logging.getLogger(__name__)

class HybridSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Hybrid混合搜索引擎，融合dense和sparse搜索"""
    
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        super().__init__(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        # 初始化dense搜索引擎作为备用
        self.dense_engine = DenseSearchEngine(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        logger.info(f"Hybrid搜索引擎初始化完成，索引: {self.index_name}, 环境: {self.pinecone_environment}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        Args:
            text: 输入文本
        Returns:
            向量表示
        """
        try:
            # 使用config中的get_embedding_model()方法，确保维度一致
            from config import get_embedding_model
            embeddings = get_embedding_model()
            embedding = embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
    
    def _build_kg_filter(self, kg_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据KG信息构建filter
        Args:
            kg_info: KG信息，包含entities和relations
        Returns:
            Pinecone filter字典
        """
        if not kg_info or not kg_info.get('entities'):
            return {}
        
        entities = kg_info.get('entities', [])
        if not entities:
            return {}
        
        # 构建OR条件，匹配任意一个实体
        filter_conditions = []
        for entity in entities:
            # 在多个字段中搜索实体
            entity_conditions = [
                {"h1": entity},
                {"h2": entity},
                {"h3": entity},
                {"h4": entity},
                {"h5": entity},
                {"h6": entity}
            ]
            filter_conditions.extend(entity_conditions)
        
        if filter_conditions:
            return {"$or": filter_conditions}
        
        return {}
    
    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        执行hybrid搜索
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数（如kg_info、filter等）
        Returns:
            搜索结果列表
        """
        try:
            # 处理查询输入
            if isinstance(query, dict):
                query_text = query.get('text', '')
                if not query_text:
                    logger.warning("结构化查询中没有找到text字段")
                    return []
            else:
                query_text = str(query)
            
            if not query_text.strip():
                logger.warning("查询文本为空")
                return []
            
            # 获取KG信息
            kg_info = kwargs.get('kg_info', {})
            
            # 构建filter
            filter_dict = kwargs.get('filter', {})
            if not filter_dict and kg_info:
                filter_dict = self._build_kg_filter(kg_info)
            
            # 获取查询向量
            query_embedding = self.get_embedding(query_text)
            
            # 构建搜索参数
            search_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "hybrid_search": True,
                "alpha": HYBRID_ALPHA
            }
            
            # 添加filter参数（如果提供）
            if filter_dict:
                search_params['filter'] = filter_dict
            
            # 添加rerank配置（如果支持）
            if RERANK_MODEL:
                search_params['rerank_config'] = {
                    "model": RERANK_MODEL,
                    "top_k": top_k
                }
            
            # 执行hybrid搜索
            response = self.index.query(**search_params)
            
            # 处理搜索结果
            results = []
            for match in response.matches:
                text = match.metadata.get('text', '')
                if not text or len(text.strip()) < 10:
                    continue
                
                result = {
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata,
                    'search_type': 'hybrid',
                    'query': query_text,
                    'kg_used': bool(filter_dict)
                }
                results.append(result)
            
            logger.info(f"Hybrid搜索完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果，KG过滤: {bool(filter_dict)}")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid搜索失败: {str(e)}")
            # 如果hybrid搜索失败，回退到dense搜索
            logger.info("回退到dense搜索")
            return self.dense_engine.search(query, top_k, **kwargs)
    
    def search_with_kg(self, query: Union[str, Dict[str, Any]], kg_info: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        使用KG信息进行搜索
        Args:
            query: 查询
            kg_info: KG信息
            top_k: 返回结果数量
        Returns:
            搜索结果列表
        """
        return self.search(query, top_k, kg_info=kg_info)
    
    def get_type(self) -> str:
        return "hybrid"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取搜索引擎能力信息"""
        return {
            "type": "hybrid",
            "description": "融合dense和sparse的混合检索",
            "supports_filtering": True,
            "supports_hybrid": True,
            "supports_kg": True,
            "supports_rerank": bool(RERANK_MODEL),
            "embedding_model": EMBADDING_MODEL,
            "hybrid_alpha": HYBRID_ALPHA,
            "rerank_model": RERANK_MODEL,
            "index_name": self.index_name
        } 