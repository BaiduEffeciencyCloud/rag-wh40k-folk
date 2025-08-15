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
from config import OPENAI_API_KEY, EMBADDING_MODEL,PINECONE_API_KEY,RERANK_TOPK,ALIYUN_RERANK_MODEL
from .search_interface import SearchEngineInterface
from .base_search import BaseSearchEngine

logger = logging.getLogger(__name__)

class DenseSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Dense向量搜索引擎"""
    
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        # 初始化 Pinecone 实例用于 inference.rerank
        #self.pc = Pinecone(api_key=PINECONE_API_KEY)
        #super().__init__(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        pass
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        Args:
            text: 输入文本
        Returns:
            向量表示
        """
        try:
            # 优先使用传入的嵌入模型实例
            if hasattr(self, 'embedding_model') and self.embedding_model:
                embedding = self.embedding_model.get_embedding(text)
                logger.info("使用传入的嵌入模型实例生成向量")
            else:
                # 使用config中的get_embedding_model()方法，确保维度一致
                embeddings = get_embedding_model()
                embedding = embeddings.get_embedding(text)
                logger.info("使用默认嵌入模型生成向量")
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
    
    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, 
               db_conn=None, embedding_model=None, rerank: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        执行dense向量搜索
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            db_conn: 数据库连接实例（必需）
            embedding_model: 嵌入模型实例
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
            
            # 验证必需参数
            if not db_conn:
                logger.error("必须提供数据库连接实例")
                return []
            
            # 使用传入的实例对象，如果没有则使用默认配置
            if embedding_model:
                self.embedding_model = embedding_model
                logger.info("使用传入的嵌入模型实例")
            
            logger.info("使用传入的数据库连接实例")
            
            # 调用数据库连接的 dense_search 方法
            if hasattr(db_conn, 'dense_search'):
                # 构建搜索参数
                filter_param = kwargs.get('filter', {})
                
                # 调用 OpenSearch 的 dense_search
                opensearch_results = db_conn.dense_search(
                    query=query_text,
                    filter=filter_param,
                    embedding_model=embedding_model,
                    top_k=top_k
                )
                
                # 格式转换：OpenSearch → Pinecone
                formatted_results = self.format_result(opensearch_results)
                
                # 根据rerank参数决定是否执行rerank
                if rerank:
                    # 执行rerank
                    rerank_results = db_conn.rerank(query_text, formatted_results, RERANK_TOPK, model=ALIYUN_RERANK_MODEL)
                    results = self.composite(rerank_results, query_text, 'dense')
                    logger.info(f"Dense搜索完成（启用rerank），查询: {query_text[:50]}...，返回 {len(results)} 个结果")
                else:
                    # 跳过rerank，需要将formatted_results转换为composite格式
                    results = self._format_without_rerank(formatted_results, query_text, 'dense')
                    logger.info(f"Dense搜索完成（跳过rerank），查询: {query_text[:50]}...，返回 {len(results)} 个结果")
                
                # 添加结果格式调试信息
                if results and len(results) > 0:
                    logger.info(f"返回结果数量: {len(results)}")
                    logger.info(f"第一个结果格式: {list(results[0].keys()) if results[0] else 'None'}")
                    logger.info(f"第一个结果内容: {results[0] if results[0] else 'None'}")
                else:
                    logger.warning("返回结果为空或None")
                
                return results
            else:
                logger.error("传入的数据库连接不支持 dense_search 方法")
                return []
                
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
            "index_name": "dynamic"  # 动态数据库连接，不固定索引名
        } 

    def format_result(self, opensearch_results: dict) -> list:
        """
        将 OpenSearch 返回结果格式化为标准搜索结果格式
        
        Args:
            opensearch_results: OpenSearch 返回的原始响应结构，包含total和hits
            
        Returns:
            list: 标准格式的搜索结果列表
        """
        formatted_results = []
        
        # 从原始响应中提取hits列表
        hits = opensearch_results.get('hits', [])
        
        for hit in hits:
            # 创建标准格式的搜索结果
            formatted_result = {
                'id': hit['_id'],
                'score': hit['_score'],
                'text': hit['_source']['text'],
                'faction': hit['_source'].get('faction', ''),
                'content_type': hit['_source'].get('content_type', ''),
                'section_heading': hit['_source'].get('section_heading', ''),
                'h1': hit['_source'].get('h1', ''),
                'h2': hit['_source'].get('h2', ''),
                'h3': hit['_source'].get('h3', ''),
                'source_file': hit['_source'].get('source_file', ''),
                'chunk_index': hit['_source'].get('chunk_index', 0),
                'created_at': hit['_source'].get('created_at', '')
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _format_without_rerank(self, formatted_results: List[Dict], query_text: str, search_engine: str) -> List[Dict]:
        """
        跳过rerank时，将formatted_results转换为与composite一致的格式
        
        Args:
            formatted_results: format_result方法返回的结果
            query_text: 查询文本
            search_engine: 搜索引擎类型
            
        Returns:
            List[Dict]: 与composite格式一致的结果列表
        """
        results = []
        for i, item in enumerate(formatted_results):
            result = {
                'doc_id': item.get('id', i),  # 使用原始ID或索引
                'text': item.get('text', ''),
                'score': item.get('score', 0.0),  # 保持原始向量分数
                'search_type': search_engine,
                'query': query_text,
                # 保留重要元数据，确保postsearch模块能正常处理
                'faction': item.get('faction', ''),
                'content_type': item.get('content_type', ''),
                'section_heading': item.get('section_heading', ''),
                'source_file': item.get('source_file', ''),
                'chunk_index': item.get('chunk_index', 0)
            }
            results.append(result)
        
        return results 