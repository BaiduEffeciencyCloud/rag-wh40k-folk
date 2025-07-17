import os
import sys
from typing import Dict, Any, List, Union
from openai import OpenAI
import logging
from config import get_embedding_model
from config import RERANK_MODEL
from pinecone import Pinecone

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, EMBADDING_MODEL,PINECONE_API_KEY,RERANK_TOPK
from .search_interface import SearchEngineInterface
from .base_search import BaseSearchEngine

logger = logging.getLogger(__name__)

class DenseSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Dense向量搜索引擎"""
    
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        # 初始化 Pinecone 实例用于 inference.rerank
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
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
            try:
                pinecone_response = self.index.query(**search_params)
                # Pinecone 新 SDK 返回 QueryResponse 对象，需用属性访问
                if hasattr(pinecone_response, "matches"):
                    candidates = pinecone_response.matches
                else:
                    logger.error("Pinecone返回对象无matches属性")
                    candidates = []
            except Exception as e:
                logger.info(f"Pinecone index.query 调用异常: {str(e)}")
                return []
            # 执行rerank，修正参数顺序
            rerank_results = self.rerank(query_text, candidates, RERANK_TOPK, model=RERANK_MODEL)
            results = self.composite(rerank_results, query_text, 'dense')
            
            logger.info(f"Dense搜索完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Dense搜索失败: {str(e)}")
            return []
        


    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5, model: str = RERANK_MODEL, **kwargs) -> List[Dict]:
        # 1. 提取所有 text（兼容 Pinecone 返回结构）
        text2item = {item['metadata']['text']: item for item in candidates if 'metadata' in item and 'text' in item['metadata']}
        documents = list(text2item.keys())
        rerank_model = model
        # 2. 调用 Pinecone rerank
        try:
            rerank_results = self.pc.inference.rerank(
                model=rerank_model,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=True
            )
            return rerank_results
        except Exception as e:
            logger.error(f"Rerank 过程发生错误:{str(e)}")
        # 3. 组装补全后的结果
    
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