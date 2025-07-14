import os
import sys
from typing import Dict, Any, List, Union
import pinecone
from openai import OpenAI
import logging
from config import get_embedding_model

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, EMBADDING_MODEL
from .search_interface import SearchEngineInterface
from .base_search import BaseSearchEngine

logger = logging.getLogger(__name__)

class DenseSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Dense向量搜索引擎"""
    
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        super().__init__(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        logger.info(f"Dense搜索引擎初始化完成，索引: {self.index_name}, 环境: {self.pinecone_environment}")
    
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
            embeddings = get_embedding_model()
            embedding = embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
    
    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        执行dense向量搜索
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数（如filter等）
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
            
            # 获取查询向量
            query_embedding = self.get_embedding(query_text)
            
            # 构建搜索参数
            search_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            # 添加filter参数（如果提供）
            if 'filter' in kwargs:
                search_params['filter'] = kwargs['filter']
            
            # 执行搜索
            response = self.index.query(**search_params)
            
            # 处理搜索结果
            results = []
            for match in response.matches:
                text = match.metadata.get('text', '')
                if not text or len(text.strip()) < 10:
                    continue
                result = {
                    'doc_id': match.id,
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata,
                    'search_type': 'dense',
                    'query': query_text
                }
                results.append(result)
            
            logger.info(f"Dense搜索完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Dense搜索失败: {str(e)}")
            return []
    
    def get_type(self) -> str:
        return "dense"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取搜索引擎能力信息"""
        return {
            "type": "dense",
            "description": "基于向量相似度的密集检索",
            "supports_filtering": True,
            "supports_hybrid": False,
            "embedding_model": EMBADDING_MODEL,
            "index_name": self.index_name
        } 