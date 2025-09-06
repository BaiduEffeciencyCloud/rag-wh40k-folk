import os
import sys
from typing import Dict, Any, List, Union
import logging

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .search_interface import SearchEngineInterface
from .dense_search import DenseSearchEngine
from .base_search import BaseSearchEngine
from embedding.em_factory import EmbeddingFactory
from conn.osconn import OpenSearchConnection
import config
from storage.oss_storage import OSSStorage
logger = logging.getLogger(__name__)
from dataupload.bm25_config import BM25Config
from dataupload.bm25_manager import BM25Manager
import glob

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
            self.bm25_config = BM25Config()
            # Phase-1: 初始化云端读能力（仍回退本地）
            cloud_userdict_bytes = None
            cloud_vocab_bytes = None
            if getattr(config, 'CLOUD_STORAGE', False):
                try:
                    oss = OSSStorage(
                        access_key_id=config.OSS_ACCESS_KEY,
                        access_key_secret=config.OSS_ACCESS_KEY_SECRET,
                        endpoint=config.OSS_ENDPOINT,
                        bucket_name=config.OSS_BUCKET_NAME_DICT,
                        region=config.OSS_REGION
                    )
                    # 列举云端对象
                    try:
                        freq_keys = oss.list_objects(prefix='bm25_vocab_freq_') or []
                    except Exception:
                        freq_keys = []
                    try:
                        dict_keys = oss.list_objects(prefix='all_docs_dict_') or []
                    except Exception:
                        dict_keys = []
                    # 简单按key字面排序（YYMMDDHHMMSS即时间序）并读取bytes
                    if freq_keys:
                        freq_keys.sort()
                        latest_freq_key = freq_keys[-1]
                        try:
                            cloud_vocab_bytes = oss.load(latest_freq_key)
                        except Exception:
                            pass
                    if dict_keys:
                        dict_keys.sort()
                        latest_ud_key = dict_keys[-1]
                        try:
                            cloud_userdict_bytes = oss.load(latest_ud_key)
                        except Exception:
                            pass
                    logger.info("已启用云端读取能力（云端优先：若bytes拉取成功将直接注入BM25Manager）")
                except Exception as e:
                    logger.warning(f"云端读取初始化失败，回退本地: {e}")
            # 白名单来源：云端优先；云端失败时再进行本地glob回退
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dict_dir = os.path.join(project_root, "dict")
            whitelist_path = None
            if not cloud_userdict_bytes:
                whitelist_files = glob.glob(os.path.join(dict_dir, "all_docs_dict_*.txt"))
                whitelist_files = [f for f in whitelist_files if not f.endswith(".meta.json")]
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
            # 若云端bytes可用，优先注入BM25Manager（不触盘）
            if cloud_userdict_bytes:
                try:
                    self.bm25_manager.load_userdict_from_bytes(cloud_userdict_bytes)
                    logger.info("已从云端bytes加载userdict（白名单）")
                except Exception as e:
                    logger.warning(f"云端userdict bytes加载失败，忽略并回退本地: {e}")
            vocab_loaded_from_cloud = False
            if cloud_vocab_bytes:
                try:
                    self.bm25_manager.load_dictionary_from_bytes(cloud_vocab_bytes)
                    vocab_loaded_from_cloud = True
                    logger.info("已从云端bytes加载BM25词典")
                except Exception as e:
                    logger.warning(f"云端词典 bytes加载失败，回退本地: {e}")
            # 使用项目根目录的绝对路径
            # 从searchengine/hybrid_search.py -> 项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dict_dir = os.path.join(project_root, "dict")
            vocab_path = None
            if not vocab_loaded_from_cloud:
                vocab_path = self.bm25_manager.get_latest_freq_dict_file(dict_dir)
                if not vocab_path or not os.path.exists(vocab_path):
                    logger.error("未找到有效的词汇表文件（云端与本地均不可用）")
                    raise FileNotFoundError("未找到有效的词汇表文件")
            # 使用项目根目录的绝对路径
            model_path = os.path.join(project_root, self.bm25_config.get_model_path())
            if os.path.exists(model_path):
                # 有模型文件则优先载入模型；若无本地vocab_path且已用云端bytes加载词典，则用一个占位路径
                self.bm25_manager.load_model(model_path, vocab_path if vocab_path else os.path.join(dict_dir, "bm25_vocab_freq_latest.json"))
                logger.info("成功加载BM25模型")
            else:
                if not vocab_loaded_from_cloud:
                    self.bm25_manager.load_dictionary(vocab_path)
                    logger.info(f"成功加载BM25词汇表，词典路径: {vocab_path}（模型文件不存在，需要重新训练）")
                else:
                    logger.info("成功加载BM25词汇表（来自云端bytes，未落盘；模型文件不存在，需要重新训练）")
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

    def search(self, query, intent: str, top_k=10, db_conn=None, embedding_model=None, rerank=True, **kwargs):
        """
        执行hybrid搜索（dense+sparse）
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数（如filter、alpha、db_conn、embedding_model、rerank等）
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
        #alpha = kwargs.get('alpha', HYBRID_ALPHA)
        # db_conn 和 embedding_model 都是方法参数，不要从kwargs重新获取！
        # db_conn = kwargs.get('db_conn')  # 这行也会导致None覆盖！
        # embedding_model = kwargs.get('embedding_model')  # 这行导致了None覆盖！
        # rerank = kwargs.get('rerank', True)  # 添加rerank参数支持
        query_intent = intent
        # 动态加载intent配置
        try:
            # 创建一个临时连接实例来加载配置
            temp_conn = OpenSearchConnection()
            intent_config = temp_conn._load_intent_config(query_intent)
            alpha = intent_config.HYBRID_CONF.HYBRID_ALPHA
            algorithm = intent_config.HYBRID_CONF.HYBRID_ALGORITHM
            logger.info(f"根据intent '{query_intent}' 动态加载算法配置: {algorithm}")
        except Exception as e:
            logger.warning(f"动态加载intent配置失败，使用默认算法: {str(e)}")
            algorithm = "pipeline"
            alpha = 0.5
        
        try:
            if algorithm == 'rrf':
                # 使用RRF算法
                results = self._search_with_rrf(query_text,query_intent,filter_dict, top_k, alpha, db_conn, embedding_model, rerank=rerank)
            else:
                # 使用现有pipeline算法（默认）
                results = self._search_with_pipeline(query_text,query_intent,filter_dict, top_k, alpha, db_conn, embedding_model, rerank=rerank)
            
            # 格式化结果，保持search_type为'hybrid'供后续模块使用
            final_results = self._format_without_rerank(results, query_text, 'hybrid')
            
            if rerank:
                logger.info(f"Hybrid搜索完成（启用rerank），查询: {query_text[:50]}...，返回 {len(final_results)} 个结果")
            else:
                logger.info(f"Hybrid搜索完成（跳过rerank），查询: {query_text[:50]}...，返回 {len(final_results)} 个结果")
            
            return final_results
                
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
    
    def _search_with_rrf(self, query_text: str, intent:str,filter_dict: dict, top_k: int, alpha: float, db_conn, embedding_model, rerank: bool = False) -> List[Dict[str, Any]]:
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
        
        # 执行RRF混合搜索（如失败降级dense）
        try:
            if db_conn and hasattr(db_conn, 'hybrid_search'):
                response = db_conn.hybrid_search(
                    query=query_text,
                    intent=intent,
                    filter=filter_dict,
                    top_k=top_k,
                    rerank=rerank,
                    algorithm='rrf',
                    sparse_vector=sparse_vector,
                    dense_vector=query_embedding
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
                
                # 🔴 复用基类的score分析日志函数
                self._log_score_analysis(formatted_results, "RRF混合搜索", query_text)
                
                return formatted_results
            else:
                logger.warning("数据库连接不支持RRF混合搜索，降级为dense检索")
                return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
                
        except Exception as e:
            logger.warning(f"RRF混合检索失败: {str(e)}，降级为dense检索")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
    
    def _search_with_pipeline(self, query_text: str,intent:str, filter_dict: dict, top_k: int, alpha: float, db_conn, embedding_model, rerank: bool = False) -> List[Dict[str, Any]]:
        """
        使用现有pipeline算法执行混合搜索
        """
        # 生成dense向量（如失败直接raise异常）
        try:
            query_embedding = embedding_model.get_embedding(query_text)
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise
        
        # 执行hybrid搜索（如失败降级dense）
        try:
            if db_conn and hasattr(db_conn, 'hybrid_search'):
                # 使用统一接口执行混合搜索
                response = db_conn.hybrid_search(
                    query=query_text,
                    intent=intent,
                    filter=filter_dict,
                    top_k=top_k,
                    rerank=rerank,
                    algorithm='pipeline',
                    query_vector=query_embedding,
                    bm25_weight=alpha,
                    vector_weight=1.0 - alpha
                )
                
                if isinstance(response, dict) and 'hits' in response:
                    hits = response['hits']
                    if isinstance(hits, dict) and 'hits' in hits:
                        actual_hits = hits['hits']
                        logger.debug(f"Pipeline混合搜索 - 实际结果数量: {len(actual_hits) if isinstance(actual_hits, list) else 'Not a list'}")
                        if actual_hits and len(actual_hits) > 0:
                            logger.debug(f"Pipeline混合搜索 - 第一个结果结构: {list(actual_hits[0].keys()) if isinstance(actual_hits, list) else 'Not a list'}")
                
                # 格式化结果
                formatted_results = self._format_hybrid_result(response)
                
                # 🔴 复用基类的score分析日志函数
                self._log_score_analysis(formatted_results, "Pipeline混合搜索", query_text, alpha=alpha)
                
                return formatted_results
            else:
                logger.warning("数据库连接不支持混合搜索，降级为dense检索")
                return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
                
        except Exception as e:
            logger.warning(f"Pipeline混合检索失败: {str(e)}，降级为dense检索")
            return self._fallback_to_dense(query_text, top_k, filter_dict, db_conn, embedding_model)
    

    

    
    def get_type(self) -> str:
        return "hybrid"

 
    
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

    def _format_without_rerank(self, formatted_results: List[Dict], query_text: str, search_engine: str) -> List[Dict]:
        """
        跳过rerank时，将formatted_results转换为与composite一致的格式
        
        Args:
            formatted_results: _format_hybrid_result方法返回的结果
            query_text: 查询文本
            search_engine: 搜索引擎类型
            
        Returns:
            List[Dict]: 与composite格式一致的结果列表
        """
        results = []
        for i, item in enumerate(formatted_results):
            result = {
                'id': item.get('id', i),  # 使用原始ID或索引
                'text': item.get('text', ''),
                'score': item.get('score', 0.0),  # 保持原始向量分数
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
    
    def _format_with_rerank(self, rerank_results: any, query_text: str, search_engine: str) -> List[Dict]:
        """
        使用rerank结果，转换为标准格式
        
        Args:
            rerank_results: rerank API返回的结果
            query_text: 查询文本
            search_engine: 搜索引擎类型
            
        Returns:
            List[Dict]: 标准格式的结果列表
        """
        # 复用基类的composite方法
        return self.composite(rerank_results, query_text, search_engine)

