import logging
from typing import Any, Dict, List
from config import DB_TYPE, EMBEDDING_TYPE
from embedding.em_factory import EmbeddingFactory
from conn.conn_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self, query_processor, search_engine, post_search, aggregator):
        """
        初始化 RAG 编排器
        
        Args:
            query_processor: 查询处理器
            search_engine: 搜索引擎
            post_search: 后处理组件
            aggregator: 聚合器
        """
        self.qp = query_processor
        self.se = search_engine
        self.agg = aggregator
        self.post_search = post_search

    def run(self, query: str, intent: str, top_k: int = 5, db_type: str = None, embedding_model: str = None, rerank: bool = True, **kwargs) -> Any:
        """
        RAG主流程调度：处理query、检索、聚合，返回最终结果。
        
        Args:
            query: 用户查询
            top_k: 检索返回数量
            db_type: 数据库类型（opensearch, pinecone等）
            embedding_model: 嵌入模型类型（qwen, openai等）
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
         
        # 2. 实例化数据库连接和嵌入模型
        db_conn_instance = None
        if db_type:
            try:
                db_conn_instance = ConnectionFactory.create_conn(db_type)
                logger.info(f"成功创建数据库连接: {db_type}")
            except Exception as e:
                logger.error(f"创建数据库连接失败: {str(e)}")
                raise RuntimeError(f"无法创建数据库连接 {db_type}: {str(e)}")
        
        embedding_instance = None
        if embedding_model:
            try:
                embedding_instance = EmbeddingFactory.create_embedding_model(embedding_model)
                logger.info(f"成功创建嵌入模型: {embedding_model}")
            except Exception as e:
                logger.error(f"创建嵌入模型失败: {str(e)}")
                raise RuntimeError(f"无法创建嵌入模型 {embedding_model}: {str(e)}")
        
        # 3. 检索召回
        results = self.se.search(processed, intent, top_k=top_k, 
                               db_conn=db_conn_instance, 
                               embedding_model=embedding_instance,
                               rerank=rerank,
                               **kwargs)

        # 4. Post-Search 处理
        if self.post_search:
            logger.info("开始进行搜索后的优化处理:")
            logger.info(f"PostSearch处理前，结果数量: {len(results) if results else 0}")
            if results and len(results) > 0:
                logger.info(f"PostSearch处理前，第一个结果: {results[0]}")
            
            # 构建query_info，包含原始查询信息
            query_info = {
                'text': query,
                'original_query': query,
                'query': query
            }
            
            results = self.post_search.process(results, 
                                             embedding_model=embedding_instance,
                                             query_info=query_info)
            logger.info(f"PostSearch处理后，结果数量: {len(results) if results else 0}")
            if results and len(results) > 0:
                logger.info(f"PostSearch处理后，第一个结果: {results[0]}")

        # 5. 聚合
        logger.info("开始进行聚合处理:")
        logger.info(f"聚合前，结果数量: {len(results) if results else 0}")
        
        # 构建包含意图信息的参数
        agg_params = kwargs.copy() if kwargs else {}
        agg_params['intent'] = intent
        
        result = self.agg.aggregate(results, query, params=agg_params)
        logger.info(f"聚合后，结果: {len(result)}")

        logger.info("流程结束")
        logger.info("✅ 全流程执行完毕")
        logger.info("="*60)
        return result

    def _get_post_search_strategy(self, processed: Dict[str, Any]) -> List[str]:
        """根据处理后的查询信息获取Post-Search策略"""
        # 可以从processed中提取query_type等信息
        # 返回处理器名称列表，如['filter']、['rerank', 'mmr']等
        pass 

    def run_with_config(self, query: str, processor_config: Dict[str, Any], **kwargs) -> Any:
        """根据配置执行RAG流程"""
        # 处理嵌套的processor_config结构
        if 'processor_config' in processor_config:
            # 如果processor_config是嵌套结构，提取内部的配置
            actual_config = processor_config['processor_config']
        else:
            # 如果processor_config直接是配置，直接使用
            actual_config = processor_config
        
        # 从实际配置中提取参数
        intent = kwargs.get('intent', processor_config.get('intent_type', 'query'))
        top_k = actual_config.get('topk', 5)
        rerank = actual_config.get('rerank', True)
        
        # 从kwargs中获取db_type和embedding_model，如果没有则使用默认值
        db_type = kwargs.get('db_type', DB_TYPE)
        embedding_model = kwargs.get('embedding_model', EMBEDDING_TYPE)
        
        # 调用现有的run方法，按正确顺序传递参数
        return self.run(query, intent, top_k, db_type, embedding_model, rerank)
        