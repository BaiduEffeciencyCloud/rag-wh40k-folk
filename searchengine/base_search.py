import os
import pinecone
from openai import OpenAI
import logging
from config import RERANK_MODEL
from typing import Dict, Any, List, Union
logger = logging.getLogger(__name__)


class BaseSearchEngine:
    """搜索引擎基类"""
    def __init__(self):
        pass
        
    def composite(self, rerank_results:any, query_text:str, search_engine:str):
        """
        将 rerank API 返回的结果统一转换为标准化的检索结果列表，便于后续展示、评估和下游处理。

        参数说明
        rerank_results (any):
        rerank API 的返回结果，可能是包含 .data 属性的对象，也可能是直接的列表。
        每个元素为 dict，结构包括 index（原始文档下标）、score（相关性分数）、document（dict，含 text 字段）。
        query_text (str):原始查询文本，用于结果溯源和展示。
        search_engine (str):检索引擎类型标识（如 'dense'），用于结果标记。
        
        返回值
        results (List[Dict]):
        标准化后的检索结果列表。每个元素为 dict，包含：
            doc_id: 文档在原始候选集中的下标（即 rerank 后的 index）
            text: 文档内容文本
            score: rerank 相关性分数
            search_type: 检索类型（如 'dense'）
            query: 原始查询文本
        """
        results = []
        
        # 兼容两种格式：直接列表或包含.data属性的对象
        if hasattr(rerank_results, 'data'):
            # Pinecone 格式：rerank_results.data
            data_list = rerank_results.data
        else:
            # 直接列表格式：rerank_results 本身就是列表
            data_list = rerank_results
        
        for data in data_list:
            text = data["document"]["text"]
            if not text or len(text.strip()) < 10:
                continue
            result = {
                'doc_id': data["index"],
                'text': text,
                'score': data["score"],
                'search_type': search_engine,
                'query': query_text
            }
            results.append(result)
        return results

    def pineconeRerank(self, query: str, candidates: List[Dict], top_k: int = 5, model: str = RERANK_MODEL, **kwargs) -> List[Dict]:
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