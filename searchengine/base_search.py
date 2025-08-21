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
            return []
    
    def _log_score_analysis(self, results: List[Dict], search_type: str, query_text: str = None, **kwargs):
        """
        统一的score分析日志函数，可被所有搜索引擎复用
        
        Args:
            results: 搜索结果列表
            search_type: 搜索类型（如"Dense搜索"、"RRF混合搜索"、"Pipeline混合搜索"）
            query_text: 查询文本（可选，用于输出基础信息）
            **kwargs: 其他参数（如alpha等）
        """
        if not results:
            logger.warning(f"{search_type} - 无结果返回")
            return
        
        # 提取score信息
        scores = [r.get('score', 0) for r in results]
        min_score, max_score = min(scores), max(scores)
        avg_score = sum(scores) / len(scores)
        
        # 构建基础日志信息（如果提供了query_text）
        if query_text:
            base_info = f"{search_type}完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果"
            if 'alpha' in kwargs:
                base_info += f"，alpha: {kwargs['alpha']}"
            logger.info(base_info)
        
        # Score分析日志
        logger.info(f"🔍 {search_type} Score分析 - 范围: [{min_score:.4f}, {max_score:.4f}], 均值: {avg_score:.4f}")
        
        # Score分布日志（全部结果）
        score_list = [f"{r.get('score', 0):.4f}" for r in results]
        logger.info(f"🔍 {search_type} Score分布 - 全部结果: {score_list}")
        
        # 自动判断score类型并给出建议
        if 0 <= min_score <= max_score <= 1:
            logger.info(f"✅ {search_type} score在[0,1]范围内，可直接用于MMR相关性过滤")
        elif -1 <= min_score <= max_score <= 1:
            logger.info(f"✅ {search_type} score在[-1,1]范围内（cosine相似度），建议映射到[0,1]后用于MMR过滤")
        elif max_score > 1.0 or min_score < -1.0:
            logger.warning(f"⚠️  {search_type} score超出[-1,1]范围，可能影响MMR相关性过滤！")
            logger.warning(f"⚠️  建议在MMR过滤前对score进行归一化处理")
        else:
            logger.warning(f"⚠️  {search_type} score范围异常: [{min_score:.4f}, {max_score:.4f}]")