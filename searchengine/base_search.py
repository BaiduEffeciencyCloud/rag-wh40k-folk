import os
import pinecone
from openai import OpenAI
import logging
logger = logging.getLogger(__name__)
class BaseSearchEngine:
    """搜索引擎基类，负责pinecone和openai初始化"""
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.pinecone_environment = pinecone_environment or os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')

        # 初始化openai
        self.client = OpenAI(api_key=self.openai_api_key)
        # 新版pinecone初始化
        self.pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)

        logging.getLogger(__name__).info(
            f"[BaseSearchEngine] pinecone初始化完成，索引: {self.index_name}, 环境: {self.pinecone_environment}") 
        
    def composite(self, rerank_results:any, query_text:str, search_engine:str):
        """
        将 Pinecone rerank API 返回的结果对象（RerankResult）
        统一转换为标准化的检索结果列表，便于后续展示、评估和下游处理。

        参数说明
        rerank_results (any):
        Pinecone rerank API 的返回对象，需包含 .data 属性。每个元素为 dict，结构包括 index（原始文档下标）、score（相关性分数）、document（dict，含 text 字段）。
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
        for data in rerank_results.data:
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
            logger.info(result)
            results.append(result)
        return results