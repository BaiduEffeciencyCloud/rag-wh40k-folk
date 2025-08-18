import os
import sys
from typing import Dict, Any, List, Union
import logging

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBADDING_MODEL, HYBRID_ALPHA, HYBRID_ALGORITHM, RRF_RANK_CONSTANT, RRF_WINDOW_MULTIPLIER
from .search_interface import SearchEngineInterface
from .dense_search import DenseSearchEngine
from .base_search import BaseSearchEngine
from embedding.em_factory import EmbeddingFactory

logger = logging.getLogger(__name__)

class HybridSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Hybrid混合搜索引擎，融合dense和sparse搜索"""
    
    def __init__(self):
        super().__init__()  # 修复：正确调用父类构造函数
        # 延迟初始化dense搜索引擎（需要db_conn参数）
        self.dense_engine = None
        # 加载BM25Manager
        self.bm25_manager = None
        self._load_bm25_manager()
        logger.info("Hybrid搜索引擎初始化完成")

    def _load_bm25_manager(self):
        """加载BM25Manager和词典，参考SparseSearchEngine，自动加载白名单userdict"""
        try:
            from dataupload.bm25_config import BM25Config
            from dataupload.bm25_manager import BM25Manager
            import glob
            self.bm25_config = BM25Config()
            # 自动查找最新白名单userdict
            whitelist_files = glob.glob(os.path.join("dict", "all_docs_dict_*.txt"))
            whitelist_files = [f for f in whitelist_files if not f.endswith(".meta.json")]
            whitelist_path = None
            if whitelist_files:
                whitelist_files.sort(reverse=True)
                whitelist_path = whitelist_files[0]
                logger.info(f"检测到白名单userdict: {whitelist_path}")
            else:
                logger.info("未检测到白名单userdict，按无白名单流程加载BM25Manager")
            self.bm25_manager = BM25Manager(
                k1=self.bm25_config.k1,
                b=self.bm25_config.b,
                min_freq=self.bm25_config.min_freq,
                max_vocab_size=self.bm25_config.max_vocab_size,
                user_dict_path=whitelist_path
            )
            vocab_path = self.bm25_manager.get_latest_freq_dict_file()
            if not vocab_path or not os.path.exists(vocab_path):
                logger.error("未找到有效的词汇表文件")
                raise FileNotFoundError("未找到有效的词汇表文件")
            model_path = self.bm25_config.get_model_path()
            if os.path.exists(model_path):
                self.bm25_manager.load_model(model_path, vocab_path)
                logger.info(f"成功加载BM25模型，模型路径: {model_path}, 词典路径: {vocab_path}")
            else:
                self.bm25_manager.load_dictionary(vocab_path)
                logger.info(f"成功加载BM25词汇表，词典路径: {vocab_path}（模型文件不存在，需要重新训练）")
        except Exception as e:
            logger.error(f"加载BM25Manager失败: {str(e)}")
            self.bm25_manager = None



    def _generate_sparse_vector(self, query_text: str) -> Dict[str, List]:
        """
        生成query的sparse向量，参考SparseSearchEngine
        Args:
            query_text: 查询文本
        Returns:
            Dict[str, List]: 包含indices和values的稀疏向量
        """
        if not self.bm25_manager:
            raise RuntimeError("BM25Manager未初始化")
        try:
            sparse_vector = self.bm25_manager.get_sparse_vector(query_text)
            logger.info(f"BM25稀疏向量生成成功，indices: {sparse_vector['indices'][:5]}...")
            # Pinecone要求indices为int
            sparse_vector['indices'] = [int(i) for i in sparse_vector['indices']]
            return sparse_vector
        except Exception as e:
            logger.warning(f"生成sparse向量失败: {str(e)}")
            raise

    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        执行hybrid搜索（dense+sparse）
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数（如filter、alpha、db_conn、embedding_model等）
        Returns:
            搜索结果列表
        """
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
        
        # 获取关键参数
        filter_dict = kwargs.get('filter', {})
        alpha = kwargs.get('alpha', HYBRID_ALPHA)
        db_conn = kwargs.get('db_conn')
        embedding_model = kwargs.get('embedding_model')
        
        # 根据配置选择算法
        algorithm = kwargs.get('algorithm', HYBRID_ALGORITHM)
        
        try:
            if algorithm == 'rrf':
                # 使用RRF算法
                return self._search_with_rrf(query_text, filter_dict, top_k, alpha, db_conn, embedding_model)
            else:
                # 使用现有pipeline算法（默认）
                return self._search_with_pipeline(query_text, filter_dict, top_k, alpha, db_conn, embedding_model)
                
        except Exception as e:
            # 检查是否是核心功能失败（如embedding生成失败）
            if "embedding" in str(e).lower() or "embed" in str(e).lower():
                logger.error(f"核心功能失败，无法降级: {str(e)}")
                raise  # 重新抛出异常
            else:
                logger.warning(f"Hybrid检索失败: {str(e)}，降级为dense检索")
                # 降级时使用传入的db_conn和embedding_model
                if db_conn:
                    return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
                else:
                    logger.error("无法降级为dense检索：缺少db_conn参数")
                    return []

        # 3. 组装补全后的结果  
    
    def _fallback_to_dense(self, query_text: str, top_k: int, filter_dict: dict, db_conn, embedding_model) -> List[Dict[str, Any]]:
        """
        降级为dense检索
        """
        try:
            # 使用实例的dense_engine（如果存在）或创建新的
            if hasattr(self, 'dense_engine') and self.dense_engine:
                dense_engine = self.dense_engine
            else:
                from .dense_search import DenseSearchEngine
                dense_engine = DenseSearchEngine()
                # 确保新创建的实例有embedding_model属性
                dense_engine.embedding_model = embedding_model
            
            # 调用dense检索
            results = dense_engine.search(
                query_text, 
                top_k=top_k, 
                db_conn=db_conn, 
                embedding_model=embedding_model,
                filter=filter_dict
            )
            
            logger.info(f"Dense检索降级完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Dense检索降级失败: {str(e)}")
            return []
    
    def _search_with_rrf(self, query_text: str, filter_dict: dict, top_k: int, alpha: float, db_conn, embedding_model) -> List[Dict[str, Any]]:
        """
        使用RRF算法执行混合搜索
        """
        # 生成dense向量（如失败直接raise异常）
        try:
            # 直接使用传入的embedding_model实例
            query_embedding = embedding_model.get_embedding(query_text)
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
        
        # 生成sparse向量（如失败降级dense）
        try:
            sparse_vector = self._generate_sparse_vector(query_text)
        except Exception as e:
            logger.warning(f"sparse向量生成失败，降级为dense检索: {str(e)}")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
        
        # 计算RRF参数
        rrf_params = self._calculate_rrf_params(alpha, top_k)
        
        # 执行RRF混合搜索（如失败降级dense）
        try:
            if db_conn and hasattr(db_conn, 'hybrid_search'):
                response = db_conn.hybrid_search(
                    query=query_text,
                    filter=filter_dict,
                    top_k=top_k,
                    sparse_vector=sparse_vector,
                    dense_vector=query_embedding,
                    rrf_params=rrf_params
                )
                
                # 添加调试日志，了解返回数据的结构
                logger.debug(f"RRF混合搜索 - OpenSearch 返回的原始数据: {response}")
                logger.debug(f"RRF混合搜索 - 数据类型: {type(response)}")
                logger.debug(f"RRF混合搜索 - 数据键: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                
                if isinstance(response, dict) and 'hits' in response:
                    hits = response['hits']
                    logger.debug(f"RRF混合搜索 - hits 字段: {hits}")
                    logger.debug(f"RRF混合搜索 - hits 类型: {type(hits)}")
                    if isinstance(hits, dict) and 'hits' in hits:
                        actual_hits = hits['hits']
                        logger.debug(f"RRF混合搜索 - 实际结果数量: {len(actual_hits) if isinstance(actual_hits, list) else 'Not a list'}")
                        if actual_hits and len(actual_hits) > 0:
                            logger.debug(f"RRF混合搜索 - 第一个结果结构: {list(actual_hits[0].keys()) if isinstance(actual_hits, list) else 'Not a list'}")
                
                # 格式化结果
                formatted_results = self._format_hybrid_result(response)
                logger.info(f"RRF混合搜索完成，查询: {query_text[:50]}...，返回 {len(formatted_results)} 个结果")
                return formatted_results
            else:
                logger.warning("数据库连接不支持RRF混合搜索，降级为dense检索")
                return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
                
        except Exception as e:
            logger.warning(f"RRF混合检索失败: {str(e)}，降级为dense检索")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
    
    def _search_with_pipeline(self, query_text: str, filter_dict: dict, top_k: int, alpha: float, db_conn, embedding_model) -> List[Dict[str, Any]]:
        """
        使用现有pipeline算法执行混合搜索
        """
        # 生成dense向量（如失败直接raise异常）
        try:
            query_embedding = embedding_model.get_embedding(query_text)
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
        
        # 生成sparse向量（如失败降级dense）
        try:
            sparse_vector = self._generate_sparse_vector(query_text)
        except Exception as e:
            logger.warning(f"sparse向量生成失败，降级为dense检索: {str(e)}")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
        
        # 执行hybrid搜索（如失败降级dense）
        try:
            if db_conn and hasattr(db_conn, 'hybrid_search'):
                # 使用统一接口执行混合搜索
                response = db_conn.hybrid_search(
                    query=query_text,
                    filter=filter_dict,
                    top_k=top_k,
                    query_vector=query_embedding,
                    bm25_weight=alpha,
                    vector_weight=1.0 - alpha
                )
                
                # 添加调试日志，了解返回数据的结构
                logger.debug(f"Pipeline混合搜索 - OpenSearch 返回的原始数据: {response}")
                logger.debug(f"Pipeline混合搜索 - 数据类型: {type(response)}")
                logger.debug(f"Pipeline混合搜索 - 数据键: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                
                if isinstance(response, dict) and 'hits' in response:
                    hits = response['hits']
                    logger.debug(f"Pipeline混合搜索 - hits 字段: {hits}")
                    logger.debug(f"Pipeline混合搜索 - hits 类型: {type(hits)}")
                    if isinstance(hits, dict) and 'hits' in hits:
                        actual_hits = hits['hits']
                        logger.debug(f"Pipeline混合搜索 - 实际结果数量: {len(actual_hits) if isinstance(actual_hits, list) else 'Not a list'}")
                        if actual_hits and len(actual_hits) > 0:
                            logger.debug(f"Pipeline混合搜索 - 第一个结果结构: {list(actual_hits[0].keys()) if isinstance(actual_hits, list) else 'Not a list'}")
                
                # 格式化结果
                formatted_results = self._format_hybrid_result(response)
                logger.info(f"Pipeline Hybrid检索完成，查询: {query_text[:50]}...，返回 {len(formatted_results)} 个结果，alpha: {alpha}")
                return formatted_results
            else:
                logger.warning("数据库连接不支持混合搜索，降级为dense检索")
                return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
                
        except Exception as e:
            logger.warning(f"Pipeline混合检索失败: {str(e)}，降级为dense检索")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
    
    def _calculate_rrf_params(self, alpha: float, top_k: int) -> dict:
        """
        计算RRF算法参数
        
        RRF (Reciprocal Rank Fusion) 算法参数说明：
        
        Args:
            alpha: 混合搜索权重参数，控制BM25和向量搜索的权重分配
            top_k: 搜索返回的文档数量
            
        Returns:
            dict: 包含RRF算法参数的字典
            
        RRF参数业务意义：
        
        1. window_size (窗口大小):
           - 定义：RRF算法在计算融合分数时考虑的候选文档数量
           - 作用：控制参与排名融合的文档范围
           - 优化方向：越大越好，建议 top_k * 4
           - 原因：更大的窗口能捕获更多潜在的优质文档，给RRF算法更多选择空间
           
        2. rank_constant (排名常数):
           - 定义：RRF公式中的平滑常数，防止分母为0
           - 作用：控制排名融合的敏感度和稳定性
           - 优化方向：越小越好，建议 10 或 5
           - 原因：更小的常数让排名差异更明显，提高算法的区分度
           
        RRF公式：score = 1 / (rank_constant + rank)
        
        注意事项：
        - window_size过大可能增加计算开销，但通常值得
        - rank_constant过小可能导致排名差异过于敏感，建议不要小于5
        - 需要在效果提升和计算效率之间找到平衡点
        """
        # alpha控制稀疏向量和密集向量的权重
        # 这里可以根据alpha值调整RRF参数
        return {
            "window_size": top_k * RRF_WINDOW_MULTIPLIER,
            "rank_constant": RRF_RANK_CONSTANT
        }
    

    
    def get_type(self) -> str:
        return "hybrid"

    def get_capabilities(self) -> Dict[str, Any]:
        """获取搜索引擎能力信息"""
        return {
            "type": "hybrid",
            "description": "融合dense和sparse的混合检索",
            "supports_filtering": True,
            "supports_hybrid": True,
            "embedding_model": EMBADDING_MODEL,
            "hybrid_alpha": HYBRID_ALPHA
        } 
    
    def _format_hybrid_result(self, opensearch_results: dict) -> list:
        """
        将 OpenSearch 返回结果格式化为标准搜索结果格式
        
        Args:
            opensearch_results: OpenSearch 返回的原始响应结构，包含total和hits
            
        Returns:
            list: 标准格式的搜索结果列表
        """
        formatted_results = []
        
        # 处理不同的数据结构（RRF vs Pipeline）
        if 'hits' in opensearch_results:
            # RRF 返回的完整响应结构
            if isinstance(opensearch_results['hits'], dict) and 'hits' in opensearch_results['hits']:
                hits = opensearch_results['hits']['hits']
            else:
                # Pipeline 返回的直接 hits 结构
                hits = opensearch_results['hits']
        else:
            hits = []
        
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