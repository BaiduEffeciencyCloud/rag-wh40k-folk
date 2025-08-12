import logging
from typing import Any, Dict, List
from embedding.em_factory import EmbeddingFactory
from conn.conn_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self, query_processor, search_engine, post_search, aggregator, db_type, embedding_model):
        self.qp = query_processor
        self.se = search_engine
        self.agg = aggregator
        self.post_search = post_search  # 新增：PostSearchPipeline实例
        self.db_type = db_type
        self.embedding_model = embedding_model

    def run(self, query: str, top_k: int = 5, **kwargs) -> Any:
        """
        RAG主流程调度：处理query、检索、聚合，返回最终结果。
        Args:
            query: 用户查询
            top_k: 检索返回数量
            **kwargs: 其他可选参数
        Returns:
            聚合后的结果
        """
        logger.info("="*60)
        logger.info("SearchController 主流程")
        logger.info(f"用户查询: {query}")
        logger.info(f"QueryProcessor: {self.qp.__class__.__name__}")
        logger.info(f"SearchEngine: {self.se.__class__.__name__}")
        logger.info(f"PostSearch: {self.post_search.__class__.__name__}")
        logger.info(f"Aggregator: {self.agg.__class__.__name__}")
        logger.info("="*60)

        # 1. 查询处理
        processed = self.qp.process(query)
        logger.info("="*60)
        logger.info("QueryProcessor 处理结果")
        logger.info(processed)
        logger.info("="*60)
         
        #数据库连接实例以及嵌入式模型实例
        conn=None
        embedding_model=None
        try:
            conn=ConnectionFactory.create_conn(self.db_type)
            embedding_model=EmbeddingFactory.create_embedding_model(self.embedding_model)
        except Exception as e:
            logger.error(f"创建连接或嵌入模型失败: {e}")
            return {"error": str(e)}

        # 2. 检索召回
        results = self.se.search(processed, top_k=top_k)

        # 2.5 Post-Search 处理（简化调用）
        if self.post_search:
            results = self.post_search.process(results)

        # 3. 聚合
        result = self.agg.aggregate(results, query, params=kwargs)

        logger.info("流程结束")
        logger.info("✅ 全流程执行完毕")
        logger.info("="*60)
        return result

    def _get_post_search_strategy(self, processed: Dict[str, Any]) -> List[str]:
        """根据处理后的查询信息获取Post-Search策略"""
        # 可以从processed中提取query_type等信息
        # 返回处理器名称列表，如['filter']、['rerank', 'mmr']等
        pass 