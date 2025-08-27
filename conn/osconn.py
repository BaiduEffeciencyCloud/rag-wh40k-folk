import logging
import os
import copy
import heapq
import json
import re
import time
from datetime import datetime
from opensearchpy import OpenSearch
from http import HTTPStatus
from .coninterface import ConnectionInterface
from config import OPENSEARCH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, QWEN_DIMENSION, ALIYUN_RERANK_MODEL, RERANK_TOPK, HYBRID_ALGORITHM, RRF_SPARSE_TERMS_BOOST, RRF_DEFAULT_RANK_CONSTANT, RRF_SPARSE_WEIGHT, RRF_DENSE_WEIGHT, RRF_WINDOW_MULTIPLIER, RRF_MAX_WINDOW_SIZE
from opensearchpy import OpenSearch
from typing import List, Dict
import dashscope
from http import HTTPStatus
from config import PIPELINE_BM25_BOOST, PIPELINE_VECTOR_BOOST, HYBRID_ALPHA
from config import QUERY_LENGTH_THRESHOLDS, BOOST_WEIGHTS, MATCH_PHRASE_CONFIG, HEADING_WEIGHTS, ENABLE_PIPELINE_CACHE
from config import SEARCH_FIELD_WEIGHTS, DEBUG_ANALYZE, SEARCH_ANALYSIS_LOG_FILE, SEARCH_ANALYSIS_LOG_MAX_SIZE, OPENSEARCH_SEARCH_ANALYZER
import json
# Opensearch数据库的处理类,负责数据转化,数据上传,数据搜索
class OpenSearchConnection(ConnectionInterface):
    def _clean_term(self, term: str) -> str:
        """
        清洗术语，防止DSL注入
        
        Args:
            term: 原始术语
            
        Returns:
            str: 清洗后的术语
        """
        if not term:
            return ""
        
        # 移除特殊字符，只保留字母、数字、中文字符和基本标点
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', str(term))
        
        # 移除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def __init__(self, index: str = None, uri: str = None, username: str = None, password: str = None):
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
        # 从config.py获取配置，如果没有传入参数的话
        self.index = index or OPENSEARCH_INDEX
        self.uri = uri or OPENSEARCH_URI
        self.username = username or OPENSEARCH_USERNAME
        self.password = password or OPENSEARCH_PASSWORD
        
        # 记录初始化信息
        self.logger.info(f"OpenSearch connection initialized - Index: {self.index}, URI: {self.uri}")
        
        # 初始化OpenSearch客户端
        try:
            self.client = OpenSearch(
                hosts=[self.uri],
                http_auth=(self.username, self.password),
                use_ssl=True,
                verify_certs=False
            )
            
            # 获取OpenSearch版本
            self.version = self._get_opensearch_version()
            self.logger.info(f"OpenSearch版本: {self.version}")
            
            # 添加管道缓存
            self._pipeline_cache = {}
            
            # 初始化搜索分析日志记录器
            self._search_analysis_logger = None
            
            # 初始化成功后立即进行心跳测试
            self.logger.info("OpenSearch客户端初始化成功，开始心跳测试...")
            heartbeat_result = self.heartbeat()
            
            if not heartbeat_result:
                raise Exception("OpenSearch心跳测试失败，连接不可用")
            
            self.logger.info("OpenSearch连接验证成功")
            
        except ImportError:
            error_msg = "opensearchpy 未安装，连接功能将不可用"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"OpenSearch客户端初始化失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        


    #从数据库中搜索数据,这个search只负责召回检索记录,不负责答案生成
    def dense_search(self, query: str, filter: dict, embedding_model=None, top_k:int=20) -> list:
        """
        执行 dense search
        
        Args:
            query: 查询文本
            filter: 过滤条件字典
            embedding_model: 嵌入模型实例，用于生成查询向量
        
        Returns:
            list: 搜索结果列表，每个结果包含id、score、text等字段
        """
        try:
            # 1. 参数验证
            if not embedding_model:
                error_msg = "embedding_model实例为空，无法生成查询向量"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not query or not query.strip():
                error_msg = "查询文本不能为空"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 2. 确保连接可用
            self._ensure_connection()
            
            # 3. 生成查询向量
            query_embedding = embedding_model.get_embedding(query)
            if not query_embedding or len(query_embedding) != 2048:
                error_msg = f"查询向量生成失败或维度不正确，期望2048维，实际{len(query_embedding) if query_embedding else 0}维"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"查询向量生成成功，维度: {len(query_embedding)}")
            
            # 4. 构建搜索查询
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                }
            }
            
            # 5. 添加过滤条件
            if filter:
                search_body["query"] = {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k
                                    }
                                }
                            }
                        ],
                        "filter": self._build_filter_query(filter)
                    }
                }
            
            # 6. 执行搜索
            self.logger.info(f"执行dense search，查询: {query[:50]}...")
            
            # 记录搜索参数（去掉embedding向量的具体数字）

            debug_search_body = copy.deepcopy(search_body)
            if 'query' in debug_search_body and 'knn' in debug_search_body['query']:
                debug_search_body['query']['knn']['embedding']['vector'] = f"[向量数据，维度: {len(query_embedding)}]"
            elif 'query' in debug_search_body and 'bool' in debug_search_body['query']:
                if 'must' in debug_search_body['query']['bool'] and debug_search_body['query']['bool']['must']:
                    for must_item in debug_search_body['query']['bool']['must']:
                        if 'knn' in must_item:
                            must_item['knn']['embedding']['vector'] = f"[向量数据，维度: {len(query_embedding)}]"
            
            self.logger.info(f"搜索参数: {debug_search_body}")
            
            # 确保使用原始的search_body进行搜索
            response = self.client.search(
                index=self.index,
                body=search_body
            )
            
            # 7. 处理结果
            # 直接返回原始的OpenSearch响应结构，格式转换由搜索引擎层处理
            hits = response.get('hits', {})
            
            # 记录前几个结果的分数，用于调试
            if hits.get('hits'):
                scores = [hit.get('_score', 0) for hit in hits['hits'][:top_k]]
                self.logger.info(f"前top_k个结果分数: {scores}")
            
            self.logger.info(f"Dense search 完成，返回 {hits.get('total', {}).get('value', 0)} 个结果")
            return hits
            
        except ValueError as e:
            # 重新抛出 ValueError，保持异常类型
            raise
        except Exception as e:
            error_msg = f"Dense search 失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
    def hybrid_search(self, query: str, filter: dict = None, top_k: int = 20, rerank: bool = False, **kwargs) -> dict:
        """
        执行混合搜索的统一接口
        
        Args:
            query: 查询文本
            filter: 过滤条件
            top_k: 返回结果数量
            rerank: 是否启用rerank，默认False
            **kwargs: 其他参数
                - 对于 pipeline: query_vector, bm25_weight, vector_weight
                - 对于 rrf: sparse_vector, dense_vector, rrf_params
                - algorithm: 可选，指定算法类型，如果不指定则使用配置默认值
                
        Returns:
            dict: 原始的 OpenSearch hits 字典，如果启用rerank则返回rerank后的结果
        """
        try:
            # 优先使用传入的算法参数，如果没有则使用配置默认值
            algorithm = kwargs.get('algorithm', HYBRID_ALGORITHM)
            
            if algorithm == 'pipeline':
                # 从 kwargs 中提取 pipeline 所需参数
                query_vector = kwargs.get('query_vector')
                # 从配置文件读取默认权重，避免硬编码
                bm25_weight = kwargs.get('bm25_weight', 1.0 - HYBRID_ALPHA)
                vector_weight = kwargs.get('vector_weight', HYBRID_ALPHA)
                
                if not query_vector:
                    raise ValueError("pipeline方式需要query_vector参数")
                    
                response = self.hybrid_search_pipeline(
                    query=query,
                    query_vector=query_vector,
                    filter=filter,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight
                )
                
            elif algorithm == 'rrf':
                # 从 kwargs 中提取 RRF 所需参数
                sparse_vector = kwargs.get('sparse_vector')
                dense_vector = kwargs.get('dense_vector')
                rrf_params = kwargs.get('rrf_params')
                
                if not sparse_vector or not dense_vector:
                    raise ValueError("RRF方式需要sparse_vector和dense_vector参数")
                    
                response = self.hybrid_search_rrf(
                    query=query,
                    sparse_vector=sparse_vector,
                    dense_vector=dense_vector,
                    filter=filter,
                    top_k=top_k,
                    rrf_params=rrf_params
                )
            else:
                raise ValueError(f"不支持的混合搜索算法: {algorithm}")
            
            # 如果启用rerank，对结果进行rerank处理
            if rerank:
                try:
                    # 将OpenSearch响应转换为rerank期望的格式
                    candidates = self._convert_response_to_rerank_candidates(response)
                    if candidates:
                        rerank_results = self.rerank(query, candidates, top_k)
                        # 将rerank结果转换回OpenSearch格式
                        return self._convert_rerank_results_to_opensearch_format(rerank_results, response)
                    else:
                        self.logger.warning("没有有效的候选文档进行rerank，返回原始结果")
                        return response
                except Exception as e:
                    self.logger.warning(f"Rerank执行失败: {str(e)}，返回原始结果")
                    return response
            else:
                # 直接返回原始 hits，不进行格式转换
                return response
            
            # 如果启用调试，收集分析信息
            if DEBUG_ANALYZE:
                self._collect_and_log_analysis(query, response, time.time())
            
        except Exception as e:
            self.logger.error(f"Hybrid搜索失败: {str(e)}")
            raise

    def _convert_response_to_list(self, response: dict) -> list:
        """
        将OpenSearch响应转换为list格式
        
        Args:
            response: OpenSearch返回的响应
            
        Returns:
            list: 转换后的结果列表
        """
        results = []
        
        # 添加调试信息
        self.logger.debug(f"Converting response: {response}")
        
        if 'hits' in response and 'hits' in response['hits']:
            self.logger.debug(f"Found {len(response['hits']['hits'])} hits")
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'text': hit['_source'].get('text', ''),
                    'faction': hit['_source'].get('faction', ''),
                    'content_type': hit['_source'].get('content_type', ''),
                    'query': ''  # 移除对response.get('query', '')的依赖
                }
                results.append(result)
                self.logger.debug(f"Converted hit: {result}")
        else:
            self.logger.debug("No hits found in response")
        
        self.logger.debug(f"Returning {len(results)} results")
        return results

    def _convert_response_to_rerank_candidates(self, response: dict) -> List[Dict]:
        """
        将OpenSearch响应转换为rerank期望的候选文档格式
        
        Args:
            response: OpenSearch返回的响应
            
        Returns:
            List[Dict]: rerank期望的候选文档格式
        """
        candidates = []
        
        self.logger.debug(f"Converting response to rerank candidates, response keys: {list(response.keys())}")
        
        if 'hits' in response and 'hits' in response['hits']:
            hits_count = len(response['hits']['hits'])
            self.logger.debug(f"Found {hits_count} hits in response")
            
            for i, hit in enumerate(response['hits']['hits']):
                self.logger.debug(f"Processing hit {i}: _id={hit.get('_id')}, _score={hit.get('_score')}")
                self.logger.debug(f"Hit _source keys: {list(hit['_source'].keys())}")
                
                text = hit['_source'].get('text', '')
                if not text:
                    self.logger.warning(f"Hit {i} has empty text field, skipping")
                    continue
                
                candidate = {
                    'text': text,
                    'score': hit['_score'],
                    'id': hit['_id'],
                    'faction': hit['_source'].get('faction', ''),
                    'content_type': hit['_source'].get('content_type', ''),
                    'section_heading': hit['_source'].get('section_heading', ''),
                    'source_file': hit['_source'].get('source_file', ''),
                    'chunk_index': hit['_source'].get('chunk_index', 0)
                }
                candidates.append(candidate)
                self.logger.debug(f"Added candidate {i}: id={candidate['id']}, text_length={len(candidate['text'])}")
        else:
            self.logger.warning("No hits found in response structure")
        
        self.logger.info(f"Converted {len(candidates)} candidates for rerank")
        return candidates

    def _convert_rerank_results_to_opensearch_format(self, rerank_results: List[Dict], original_response: dict) -> dict:
        """
        将rerank结果转换回OpenSearch格式
        
        Args:
            rerank_results: rerank返回的结果列表
            original_response: 原始的OpenSearch响应
            
        Returns:
            dict: 转换后的OpenSearch格式响应
        """
        if not rerank_results:
            return original_response
        
        # 获取原始hits列表
        original_hits = original_response.get('hits', {}).get('hits', [])
        
        # 根据rerank结果的index重新排序hits
        reranked_hits = []
        for rerank_result in rerank_results:
            index = rerank_result.get('index', 0)
            if index < len(original_hits):
                # 更新score为rerank的score
                hit = original_hits[index].copy()
                hit['_score'] = rerank_result.get('score', hit['_score'])
                reranked_hits.append(hit)
        
        # 构建新的响应
        new_response = original_response.copy()
        new_response['hits']['hits'] = reranked_hits
        new_response['hits']['total']['value'] = len(reranked_hits)
        
        return new_response
        
    def hybrid_search_pipeline(self, query: str, query_vector: list, filter: dict = None, top_k: int = 20, 
                     bm25_weight: float = 0.7, vector_weight: float = 0.3) -> dict:
        """
        执行混合搜索（BM25 + 向量搜索）- Pipeline方式
        
        Args:
            query: 查询文本
            query_vector: 查询向量（由调用方生成）
            filter: 过滤条件
            top_k: 返回结果数量
            bm25_weight: BM25搜索权重 (0.0-1.0)
            vector_weight: 向量搜索权重 (0.0-1.0)
            
        Returns:
            dict: 包含total和hits的搜索结果
        """
        try:
            # 1. 确保连接
            self._ensure_connection()
            
            # 2. 验证查询向量
            if query_vector is None:
                raise ValueError("hybrid_search 需要 query_vector 参数")
            
            if not query_vector or len(query_vector) != 2048:
                raise ValueError(f"查询向量维度不正确: {len(query_vector) if query_vector else 0}")
            
            self.logger.info(f"Hybrid搜索开始，查询: {query}，向量维度: {len(query_vector)}，OpenSearch版本: {self.version}")
            
            # 3. 创建或获取搜索管道
            pipeline_id = self._get_or_create_search_pipeline()
            
            # 4. 根据OpenSearch版本选择查询构建方式
            if self._should_use_optimized_query():
                search_body = self._build_hybrid_query_optimized(query, query_vector, filter, top_k)
            else:
                search_body = self._build_hybrid_query_with_weights(query, query_vector, bm25_weight, vector_weight, filter, top_k)
            
            # 5. 根据配置决定是否添加explain参数
            if DEBUG_ANALYZE:
                search_body["explain"] = True
            
            # 6. 执行搜索（修复：使用 search_pipeline 参数）
            response = self.client.search(
                body=search_body,
                index=self.index,
                search_pipeline=pipeline_id
            )
            
            # 7. 对搜索结果进行去重处理
            response = self._deduplicate_response(response)
            
            # 8. 如果启用调试，收集分析信息
            if DEBUG_ANALYZE:
                self._collect_and_log_analysis(query, response, time.time())
            
            # 9. 处理结果
            hits = response.get('hits', {})
            
            # 添加调试日志
            self.logger.debug(f"OpenSearch 原始响应: {response}")
            self.logger.debug(f"提取的 hits 字段: {hits}")
            self.logger.debug(f"hits 类型: {type(hits)}")
            
            self.logger.info(f"Hybrid搜索完成，查询: {query}，返回 {hits.get('total', {}).get('value', 0)} 个结果")
            return response
            
        except Exception as e:
            self.logger.error(f"Hybrid搜索失败: {str(e)}")
            raise
    

    
    def _get_or_create_search_pipeline(self) -> str:
        """
        获取或创建用于混合搜索分数归一化的搜索管道
        
        Returns:
            str: 管道ID
        """
        pipeline_id = "hybrid_normalization_pipeline"
        
        try:
            # 1. 使用正确的 Search Pipeline API 检查搜索管道是否存在
            self.client.search_pipeline.get(id=pipeline_id)
            self.logger.info(f"使用现有搜索管道: {pipeline_id}")
            
        except Exception:  # opensearchpy.NotFoundError
            # 2. 管道不存在，创建新的
            self.logger.info(f"搜索管道不存在，正在创建: {pipeline_id}")
            self._create_search_pipeline(pipeline_id)
            
        return pipeline_id
    
    def _create_search_pipeline(self, pipeline_id: str):
        """
        创建一个带有分数归一化处理器的搜索管道
        
        Args:
            pipeline_id: 管道ID
        """
        pipeline_body = {
            "description": "A search pipeline for hybrid search score processing.",
            "request_processors": [
                {
                    "script": {
                        "lang": "painless",
                        "source": """
                        // 简单的分数处理脚本
                        if (ctx.containsKey('_score')) {
                            // 保持原始分数，不做复杂处理
                            ctx._score = ctx._score;
                        }
                        """
                    }
                }
            ]
        }
        
        try:
            # 3. 使用正确的 Search Pipeline API 创建搜索管道
            self.client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
            self.logger.info(f"成功创建搜索管道: {pipeline_id}")
        except Exception as e:
            self.logger.error(f"创建搜索管道失败: {str(e)}")
            raise
    
    def _deduplicate_response(self, response: dict) -> dict:
        """
        对OpenSearch搜索结果进行去重处理
        
        Args:
            response: OpenSearch返回的原始响应
            
        Returns:
            dict: 去重后的响应
        """
        # 深拷贝response，避免修改原始数据
        import copy
        result = copy.deepcopy(response)
        
        # 获取hits数组
        if 'hits' not in result or result['hits'] is None or 'hits' not in result['hits']:
            return result
            
        hits = result['hits']['hits']
        if not hits:
            return result
            
        # 基于doc_id去重，保留最高分
        seen_docs = {}
        for hit in hits:
            doc_id = hit['_id']
            score = hit['_score']
            
            # 如果没见过这个doc_id，或者当前分数更高，则保留
            if doc_id not in seen_docs or score > seen_docs[doc_id]['_score']:
                seen_docs[doc_id] = hit
        
        # 按分数降序排列，重建hits数组
        deduplicated_hits = sorted(seen_docs.values(), key=lambda x: x['_score'], reverse=True)
        
        # 更新response
        result['hits']['hits'] = deduplicated_hits
        result['hits']['total']['value'] = len(deduplicated_hits)
        
        return result
        

    
    def _build_hybrid_query_with_weights(self, query: str, query_vector: list, bm25_weight: float, vector_weight: float, filter: dict = None, top_k: int = 20) -> dict:
        """
        构建支持动态权重的混合搜索查询
        
        Args:
            query: 查询文本
            query_vector: 查询向量
            bm25_weight: BM25搜索权重
            vector_weight: 向量搜索权重
            filter: 过滤条件
            top_k: 返回结果数量
            
        Returns:
            dict: 搜索查询体
        """
        # 基础查询结构，权重在查询层面处理
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "analyzer": OPENSEARCH_SEARCH_ANALYZER,  # 使用配置的analyzer
                                    "boost": bm25_weight  # 动态权重
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": vector_weight  # 动态权重
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # 添加过滤条件
        if filter:
            search_body["query"]["bool"]["filter"] = self._build_filter_query(filter)
        
        return search_body
    
    def _build_filter_query(self, filter: dict) -> list:
        """
        构建过滤查询
        
        Args:
            filter: 过滤条件字典
            
        Returns:
            list: 过滤查询列表
        """
        filter_queries = []
        
        for field, value in filter.items():
            if isinstance(value, str):
                filter_queries.append({"term": {field: value}})
            elif isinstance(value, list):
                filter_queries.append({"terms": {field: value}})
            elif isinstance(value, dict):
                # 支持范围查询等复杂过滤
                filter_queries.append({"range": {field: value}})
        
        return filter_queries
    
    def heartbeat(self) -> bool:
        """
        测试与OpenSearch的连接状态
        
        Returns:
            bool: True表示连接正常，False表示连接失败
        """
        try:
            if not self.client:
                self.logger.warning("OpenSearch客户端未初始化")
                return False
            
            # 使用ping方法测试连接
            response = self.client.ping()
            if response:
                self.logger.info("OpenSearch连接正常")
                return True
            else:
                self.logger.warning("OpenSearch ping失败")
                return False
                
        except Exception as e:
            self.logger.error(f"OpenSearch心跳检测失败: {str(e)}")
            return False

    def _get_opensearch_version(self) -> str:
        """
        获取OpenSearch版本信息
        
        Returns:
            str: OpenSearch版本号，如 "2.9.0"
        """
        try:
            info = self.client.info()
            version = info.get('version', {}).get('number', '0.0.0')
            self.logger.info(f"获取到OpenSearch版本: {version}")
            return version
        except Exception as e:
            self.logger.warning(f"获取OpenSearch版本失败，使用默认版本: {str(e)}")
            return "0.0.0"

    def _should_use_optimized_query(self) -> bool:
        """
        判断是否应该使用优化的查询构建方式
        
        Returns:
            bool: True表示使用_build_hybrid_query_optimized，False表示使用_build_hybrid_query_with_weights
        """
        try:
            # 解析版本号
            version_parts = self.version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            
            # OpenSearch 2.9+ 支持优化的查询方式
            if major > 2 or (major == 2 and minor >= 9):
                self.logger.info(f"OpenSearch版本 {self.version} >= 2.9，使用优化的查询构建方式")
                return True
            else:
                self.logger.info(f"OpenSearch版本 {self.version} < 2.9，使用传统权重查询构建方式")
                return False
        except (ValueError, IndexError) as e:
            self.logger.warning(f"版本解析失败，使用传统查询方式: {str(e)}")
            return False

    def adaptive_data(self, chunk: dict, chunk_id: str, bm25_manager, pre_generated_embedding: list = None) -> dict:
        """
        将单个chunk数据转换成适合OpenSearch的格式
        
        Args:
            chunk: 单个chunk数据，包含text和metadata
            chunk_id: 文档ID
            bm25_manager: 用于生成sparse向量的管理器实例
            pre_generated_embedding: 预生成的dense向量，如果提供则直接使用
            
        Returns:
            dict: 转换后的OpenSearch文档格式
            
        Raises:
            ValueError: 当pre_generated_embedding为空且无法生成向量时
            Exception: 当向量生成或数据处理失败时
        """
        # 1. 参数验证和日志记录
        if not pre_generated_embedding:
            error_msg = "pre_generated_embedding为空，无法生成dense向量"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not bm25_manager:
            error_msg = "bm25_manager实例为空，无法生成sparse向量"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"开始处理chunk {chunk_id}")
        
        try:
            # 2. 使用预生成的embedding
            dense_vector = pre_generated_embedding
            self.logger.info(f"chunk {chunk_id} 使用预生成的dense向量，维度: {len(dense_vector)}")
            
            # 3. 生成sparse向量
            sparse_vector = bm25_manager.get_sparse_vector(chunk['text'])
            
            # 转换稀疏向量格式：从 {indices: [...], values: [...]} 到 {index: value, ...}
            # 这是OpenSearch rank_features类型需要的格式
            if sparse_vector and 'indices' in sparse_vector and 'values' in sparse_vector:
                converted_sparse = {}
                for i, index in enumerate(sparse_vector['indices']):
                    converted_sparse[str(index)] = sparse_vector['values'][i]
                sparse_vector = converted_sparse
                self.logger.info(f"chunk {chunk_id} sparse向量转换完成，特征数量: {len(sparse_vector)}")
            else:
                self.logger.warning(f"chunk {chunk_id} sparse向量格式异常或为空")
                sparse_vector = {}
            
            # 4. 构造OpenSearch文档
            document = {
                'id': chunk_id,
                'text': chunk['text'],
                'embedding': dense_vector,
                'sparse': sparse_vector,
                'chunk_type': chunk.get('chunk_type', 'default'),
                'content_type': chunk.get('content_type', ''),
                'faction': chunk.get('faction', ''),
                'section_heading': chunk.get('section_heading', ''),
                # 标题层级字段
                'h1': chunk.get('h1', ''),
                'h2': chunk.get('h2', ''),
                'h3': chunk.get('h3', ''),
                'h4': chunk.get('h4', ''),
                'h5': chunk.get('h5', ''),
                'h6': chunk.get('h6', ''),
                # 其他元数据
                'source_file': chunk.get('source_file', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'created_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"chunk {chunk_id} 转换完成，文档结构: {list(document.keys())}")
            return document
            
        except Exception as e:
            error_msg = f"chunk {chunk_id} 转换失败: {str(e)}"
            self.logger.error(error_msg)
            # 直接抛出异常，让上层处理
            raise Exception(error_msg)

    def upload_data(self, index_name: str, document: dict, doc_id: str = None) -> dict:
        """上传单个文档到指定的OpenSearch索引"""
        try:
            # 准备上传参数
            upload_params = {
                'index': index_name,
                'body': document
            }
            
            # 如果提供了doc_id，则添加到参数中
            if doc_id:
                upload_params['id'] = doc_id
            
            # 执行上传
            self.logger.info(f"正在上传文档到索引 {index_name}")
            response = self.client.index(**upload_params)
            
            # 记录上传结果
            self.logger.info(f"文档上传成功: {response.get('result', 'unknown')}")
            return response
            
        except Exception as e:
            error_msg = f"文档上传失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def bulk_upload_data(self, index_name: str, documents: list, 
                         batch_size: int, 
                         doc_ids: list = None) -> dict:
        """批量上传多个文档到指定的OpenSearch索引"""
        try:
            # 验证输入参数
            if not documents:
                self.logger.warning("文档列表为空，跳过批量上传")
                return {'took': 0, 'errors': False, 'items': []}
            
            # 准备bulk API格式的数据
            bulk_data = []
            for i, doc in enumerate(documents):
                # 操作行
                operation = {
                    'index': {
                        '_index': index_name
                    }
                }
                
                # 如果提供了doc_ids，使用自定义ID
                if doc_ids and i < len(doc_ids):
                    operation['index']['_id'] = doc_ids[i]
                
                bulk_data.append(operation)
                bulk_data.append(doc)
            
            # 执行批量上传
            self.logger.info(f"正在批量上传 {len(documents)} 个文档到索引 {index_name}")
            response = self.client.bulk(body=bulk_data)
            
            # 记录上传结果
            if response.get('errors', False):
                self.logger.warning(f"批量上传完成，但有部分文档失败: {response}")
            else:
                self.logger.info(f"批量上传成功完成: {len(documents)} 个文档")
            
            return response
            
        except Exception as e:
            error_msg = f"批量上传失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def bulk_adaptive_data(self, chunks: list) -> list:
        """批量转换chunk数据（保持接口兼容性）"""
        try:
            if not chunks:
                self.logger.warning("chunks列表为空，跳过批量转换")
                return []
            
            converted_docs = []
            for chunk in chunks:
                try:
                    # 生成chunk_id
                    chunk_id = chunk.get('chunk_id', f"chunk_{len(converted_docs)}")
                    
                    # 使用现有的adaptive_data方法转换单个chunk
                    # 注意：这里需要embedding_model和bm25_manager，但接口中没有提供
                    # 为了保持兼容性，我们返回一个基本的转换结果
                    converted_doc = {
                        '_id': chunk_id,
                        'text': chunk.get('text', ''),
                        'chunk_type': chunk.get('chunk_type', 'default'),
                        'content_type': chunk.get('content_type', ''),
                        'faction': chunk.get('faction', ''),
                        'section_heading': chunk.get('section_heading', ''),
                        'h1': chunk.get('h1', ''),
                        'h2': chunk.get('h2', ''),
                        'h3': chunk.get('h3', ''),
                        'h4': chunk.get('h4', ''),
                        'h5': chunk.get('h5', ''),
                        'h6': chunk.get('h6', ''),
                        'source_file': chunk.get('source_file', ''),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    converted_docs.append(converted_doc)
                    
                except Exception as e:
                    self.logger.error(f"转换chunk失败: {str(e)}")
                    # 继续处理其他chunk
                    continue
            
            self.logger.info(f"批量转换完成: {len(converted_docs)}/{len(chunks)} 个chunk成功转换")
            return converted_docs
            
        except Exception as e:
            error_msg = f"批量转换失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_index_count(self, index_name: str) -> int:
        """获取指定 index 下的数据条目数量"""
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 使用 count API 获取文档数量
            response = self.client.count(index=index_name)
            count = response.get('count', 0)
            
            self.logger.info(f"Index {index_name} 中共有 {count} 个文档")
            return count
            
        except Exception as e:
            error_msg = f"获取 index {index_name} 文档数量失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def del_data_by_index(self, index_name: str) -> bool:
        """删除指定 index 下的所有数据"""
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 使用 delete_by_query API 删除所有文档
            # 使用 match_all 查询匹配所有文档
            response = self.client.delete_by_query(
                index=index_name,
                body={
                    "query": {
                        "match_all": {}
                    }
                }
            )
            
            # 获取删除的文档数量
            deleted_count = response.get('deleted', 0)
            total_count = response.get('total', 0)
            
            self.logger.info(f"成功删除 index {index_name} 中的 {deleted_count}/{total_count} 个文档")
            
            # 返回是否成功删除（即使删除0个文档也算成功）
            return True
            
        except Exception as e:
            error_msg = f"删除 index {index_name} 数据失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _ensure_connection(self):
        """
        确保OpenSearch连接可用
        
        Raises:
            Exception: 当连接不可用时
        """
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 执行心跳测试
            if not self.heartbeat():
                raise Exception("OpenSearch连接不可用")
                
        except Exception as e:
            error_msg = f"连接检查失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
    def rerank(self, query: str, candidates: List[Dict], top_k: int = RERANK_TOPK, model: str = None, **kwargs) -> List[Dict]:
        """
        使用 dashscope TextReRank API 重排序搜索结果
        
        Args:
            query: 查询文本
            candidates: 候选文档列表，每个文档包含 text 字段
            top_k: 重排序后返回的文档数量
            model: 重排序模型名称，默认使用 config.ALIYUN_RERANK_MODEL
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 重排序后的结果列表，格式与 pineconeRerank 兼容
        """
        self.logger.info(f"rerank 开始，query: {query[:50]}...，文档数量: {len(candidates)}，模型: {model}")
        try:
            # 1. 参数验证
            if not query or not query.strip():
                self.logger.warning("查询文本为空，返回空列表")
                return []
            
            if not candidates:
                self.logger.info("候选文档为空，返回空列表")
                return []
            
            # 2. 提取文档文本
            documents = []
            valid_candidates = []
            
            for candidate in candidates:
                if 'text' in candidate and candidate['text']:
                    documents.append(candidate['text'])
                    valid_candidates.append(candidate)
            
            if not documents:
                self.logger.warning("没有有效的文档文本，返回空列表")
                return []
            
            # 3. 设置模型参数
            rerank_model = model or ALIYUN_RERANK_MODEL
            top_n = min(top_k, len(documents))
            
            self.logger.info(f"开始重排序，查询: {query[:50]}...，文档数量: {len(documents)}，模型: {rerank_model}")
            
            # 4. 调用 dashscope TextReRank API
            response = dashscope.TextReRank.call(
                model=rerank_model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=True
            )
            
            # 5. 处理响应结果
            if response.status_code == HTTPStatus.OK:
                
                # 根据 dashscope TextReRank API 的实际返回结构处理
                # 对于 gte-rerank-v2 模型，返回结构可能是不同的
                rerank_results = []
                
                # 检查 response 的基本结构
                if hasattr(response, 'output') and hasattr(response.output, 'results'):
                    results = response.output.results
                    self.logger.info(f"找到 {len(results)} 个重排序结果")
                    
                    # 处理每个结果
                    for i, result in enumerate(results):
                        try:
                            # 根据调试结果，ReRankResult 对象有 index 和 relevance_score 属性
                            if hasattr(result, 'index'):
                                index = result.index
                            else:
                                index = i
                            
                            if hasattr(result, 'relevance_score'):
                                score = result.relevance_score
                            else:
                                self.logger.warning(f"结果 {i} 没有找到 relevance_score 属性，跳过")
                                continue
                            
                            # 验证 score 的有效性
                            if not isinstance(score, (int, float)):
                                self.logger.warning(f"结果 {i} 的 score 类型无效: {type(score)}, 值: {score}")
                                continue
                            
                            if index < len(valid_candidates):
                                # 获取原始候选文档
                                original_candidate = valid_candidates[index]
                                
                                # 创建重排序结果
                                rerank_result = {
                                    'index': index,
                                    'score': score,
                                    'document': {
                                        'text': original_candidate['text']
                                    }
                                }
                                rerank_results.append(rerank_result)
                                self.logger.info(f"成功处理结果 {i}: index={index}, score={score}")
                            else:
                                self.logger.warning(f"结果 {i} 的 index {index} 超出候选文档范围")
                                
                        except Exception as e:
                            self.logger.error(f"处理结果 {i} 时发生错误: {str(e)}")
                            continue
                    
                    # 按 score 排序
                    if rerank_results:
                        rerank_results.sort(key=lambda x: x['score'], reverse=True)
                        self.logger.info(f"重排序完成，返回 {len(rerank_results)} 个结果")
                        return rerank_results
                    else:
                        self.logger.warning("没有有效的重排序结果，使用降级策略")
                        return self._fallback_to_original_candidates(candidates, top_k)
                else:
                    self.logger.error("Response 结构不符合预期，使用降级策略")
                    return self._fallback_to_original_candidates(candidates, top_k)
            else:
                # API 调用失败，记录错误并返回原始候选文档
                self.logger.error(f"dashscope rerank API 调用失败: {response.message}")
                return self._fallback_to_original_candidates(candidates, top_k)
                
        except Exception as e:
            # 异常处理，返回原始候选文档
            self.logger.error(f"rerank 过程发生错误: {str(e)}")
            return self._fallback_to_original_candidates(candidates, top_k)
    
    def _fallback_to_original_candidates(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        降级策略：返回原始候选文档
        
        Args:
            candidates: 原始候选文档列表
            top_k: 返回的文档数量
            
        Returns:
            List[Dict]: 格式化的原始候选文档
        """
        try:
            fallback_results = []
            for i, candidate in enumerate(candidates[:top_k]):
                if 'text' in candidate:
                    fallback_result = {
                        'index': i,
                        'score': candidate.get('score', 0.0),
                        'document': {
                            'text': candidate['text']
                        }
                    }
                    fallback_results.append(fallback_result)
            
            self.logger.info(f"使用降级策略，返回 {len(fallback_results)} 个原始候选文档")
            return fallback_results
            
        except Exception as e:
            self.logger.error(f"降级策略执行失败: {str(e)}")
            return []

    def get_sample_data(self, size: int = 1) -> List[Dict]:
        """
        获取索引中的样本数据
        
        Args:
            size: 返回的样本数量，默认1条
            
        Returns:
            List[Dict]: 样本数据列表
        """
        try:
            if not self.client:
                raise Exception("OpenSearch客户端未初始化")
            
            # 构建查询，获取指定数量的样本
            search_body = {
                "size": size,
                "query": {
                    "match_all": {}
                }
            }
            
            # 执行搜索
            response = self.client.search(
                index=self.index,
                body=search_body
            )
            
            # 处理结果
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            for hit in hits:
                sample_data = {
                    '_id': hit['_id'],
                    '_score': hit['_score'],
                    '_source': hit['_source']
                }
                results.append(sample_data)
            
            self.logger.info(f"成功获取 {len(results)} 条样本数据")
            return results
            
        except Exception as e:
            error_msg = f"获取样本数据失败: {str(e)}"
            self.logger.error(error_msg)
            return []

    def _normalize_sparse_vector(self, sparse_vector: dict) -> dict:
        """
        稀疏向量预处理和格式验证
        
        Args:
            sparse_vector: 原始稀疏向量
            
        Returns:
            dict: 标准化后的稀疏向量
        """
        if not sparse_vector:
            raise ValueError("sparse_vector不能为空")
        
        # 处理不同格式的稀疏向量
        if isinstance(sparse_vector, dict) and 'indices' in sparse_vector:
            # 格式1: {indices: [...], values: [...]}
            if 'values' not in sparse_vector:
                raise ValueError("sparse_vector格式错误：缺少values字段")
            
            indices = sparse_vector['indices']
            values = sparse_vector['values']
            
            # 验证长度一致性
            if len(indices) != len(values):
                raise ValueError(f"indices和values长度不一致: indices={len(indices)}, values={len(values)}")
            
            # 验证indices不能为负数
            if any(i < 0 for i in indices):
                raise ValueError(f"检测到非法负数index: {indices}")
            
            # 权重归一化处理
            if values:
                max_value = max(values)
                if max_value > 0:
                    normalized_values = [v / max_value for v in values]
                else:
                    normalized_values = values
            else:
                normalized_values = values
            
            return {
                'indices': indices,
                'values': normalized_values
            }
        else:
            # 格式2: {term: weight} - 转换为indices格式
            if not isinstance(sparse_vector, dict):
                raise ValueError("sparse_vector必须是字典格式")
            
            # 按权重排序，取前10个最重要的术语
            sorted_terms = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if not sorted_terms:
                raise ValueError("sparse_vector中没有有效的术语")
            
            # 提取indices和values（这里indices是术语的字符串表示）
            indices = [str(term) for term, _ in sorted_terms]
            values = [weight for _, weight in sorted_terms]
            
            # 权重归一化
            max_value = max(values)
            if max_value > 0:
                normalized_values = [v / max_value for v in values]
            else:
                normalized_values = values
            
            return {
                'indices': indices,
                'values': normalized_values
            }

    def _validate_dense_vector(self, dense_vector: list):
        """
        验证密集向量格式和维度
        
        Args:
            dense_vector: 密集向量列表
            
        Raises:
            ValueError: 向量格式或维度不正确时抛出异常
        """
        if not dense_vector:
            raise ValueError("dense_vector不能为空")
        
        if not isinstance(dense_vector, list):
            raise ValueError("dense_vector必须是列表格式")
        
        # 验证维度（使用配置的维度）
        if len(dense_vector) != QWEN_DIMENSION:
            raise ValueError(f"密集向量维度应为{QWEN_DIMENSION}，实际为{len(dense_vector)}")
        
        # 验证向量元素类型
        if not all(isinstance(x, (int, float)) for x in dense_vector):
            raise ValueError("dense_vector中的所有元素必须是数字类型")

    def _configure_rrf_params(self, rrf_params: dict = None, top_k: int = 20) -> dict:
        """
        配置RRF参数，提供智能默认值
        
        Args:
            rrf_params: 用户自定义参数
            top_k: 返回结果数量
            
        Returns:
            dict: 配置完成的RRF参数
        """
        # 设置合理的默认RRF参数
        default_rrf_params = {
            "window_size": min(top_k * RRF_WINDOW_MULTIPLIER, RRF_MAX_WINDOW_SIZE),  # 使用配置的窗口倍数，但不超过最大值
            "rank_constant": RRF_DEFAULT_RANK_CONSTANT,  # 标准RRF常数
            "combination_technique": "rrf"  # 使用RRF组合技术
        }
        
        # 合并用户自定义参数
        if rrf_params:
            # 参数验证
            self._validate_rrf_params(rrf_params)
            default_rrf_params.update(rrf_params)
        
        return default_rrf_params

    def _validate_rrf_params(self, rrf_params: dict):
        """
        验证RRF参数的有效性
        
        Args:
            rrf_params: RRF参数字典
            
        Raises:
            ValueError: 参数无效时抛出异常
        """
        if 'window_size' in rrf_params:
            window_size = rrf_params['window_size']
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError("window_size必须是正整数")
            if window_size > RRF_MAX_WINDOW_SIZE:
                raise ValueError(f"window_size不能超过{RRF_MAX_WINDOW_SIZE}，过大的窗口可能影响性能")
        
        if 'rank_constant' in rrf_params:
            rank_constant = rrf_params['rank_constant']
            if not isinstance(rank_constant, (int, float)) or rank_constant <= 0:
                raise ValueError("rank_constant必须是正数")

    def _build_optimized_rrf_query(self, query: str, sparse_vector: dict, dense_vector: list, 
                                  filter: dict = None, top_k: int = 20, rrf_params: dict = None) -> dict:
        """
        构建优化的RRF查询
        
        Args:
            query: 查询文本
            sparse_vector: 稀疏向量字典
            dense_vector: 密集向量列表
            filter: 过滤条件
            top_k: 返回结果数量
            rrf_params: RRF算法参数
            
        Returns:
            dict: 优化的搜索查询体
        """
        # 构建稀疏向量查询
        sparse_query = self._build_sparse_query_optimized(sparse_vector)
        
        # 构建密集向量查询
        dense_query = self._build_dense_query(dense_vector, top_k)
        
        # 应用配置的权重
        if 'terms' in sparse_query:
            sparse_query['terms']['boost'] = RRF_SPARSE_WEIGHT
        if 'knn' in dense_query:
            dense_query['knn']['embedding']['boost'] = RRF_DENSE_WEIGHT
        
        # 构建混合查询
        search_body = {
            "size": min(top_k, 100),  # 分页保护，防止过大请求
            "query": {
                "hybrid": {
                    "queries": [
                        sparse_query,    # 稀疏向量查询
                        dense_query      # 密集向量查询
                    ]
                }
            }
        }
        
        # 添加过滤条件
        if filter:
            search_body["query"]["hybrid"]["filter"] = self._build_filter_query(filter)
        
        return search_body



    def _analyze_search_results(self, hits: dict, top_k: int):
        """
        分析搜索结果，记录性能指标
        
        Args:
            hits: 搜索结果
            top_k: 请求的结果数量
        """
        # 验证结果格式
        if 'hits' not in hits:
            self.logger.warning("RRF搜索结果格式异常，缺少'hits'字段")
            return
        
        # 记录前N个结果的分数分布
        if hits['hits']:
            scores = [hit.get('_score', 0) for hit in hits['hits'][:5]]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                self.logger.info(f"RRF搜索结果分数分布: min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}")
        
        # 记录结果数量
        total_results = hits.get('total', {}).get('value', 0)
        self.logger.info(f"RRF搜索返回 {total_results} 个结果，请求 {top_k} 个")

    def _log_performance_metrics(self, metrics: dict):
        """
        记录性能指标
        
        Args:
            metrics: 性能指标字典
        """
        operation = metrics.get('operation', 'unknown')
        duration = metrics.get('duration', 0)
        result_count = metrics.get('result_count', 0)
        query_length = metrics.get('query_length', 0)
        
        self.logger.info(f"性能指标 | {operation} | 耗时: {duration:.4f}s | 结果数: {result_count} | 查询长度: {query_length}")
        
        # 记录慢查询警告
        if duration > 5.0:  # 超过5秒的查询
            self.logger.warning(f"慢查询警告: {operation} 耗时 {duration:.4f}s")

    def _build_sparse_query_optimized(self, sparse_vector: dict) -> dict:
        """
        构建优化的稀疏向量查询，使用terms查询替代通配符匹配
        
        Args:
            sparse_vector: 稀疏向量字典
            
        Returns:
            dict: 优化的terms查询体
        """
        if not sparse_vector:
            # 兜底：返回通用文本搜索
            return {
                "match": {
                    "text": {
                        "query": "*",
                        "boost": 1.0
                    }
                }
            }
        
        # 处理不同格式的稀疏向量
        if isinstance(sparse_vector, dict) and 'indices' in sparse_vector:
            # 格式1: {indices: [...], values: [...]}
            if 'values' in sparse_vector and len(sparse_vector['indices']) == len(sparse_vector['values']):
                # 使用堆排序获取前10个最重要的术语，性能优化 O(n log k)
                term_weights = list(zip(sparse_vector['indices'], sparse_vector['values']))
                top_terms = heapq.nlargest(10, term_weights, key=lambda x: x[1])
                top_terms = [str(term) for term, _ in top_terms]
            else:
                # 只有indices，直接使用
                top_terms = [str(idx) for idx in sparse_vector['indices'][:10]]
        else:
            # 格式2: {term: weight} - 使用堆排序优化性能
            top_terms = heapq.nlargest(10, sparse_vector.items(), key=lambda x: x[1])
            top_terms = [term for term, _ in top_terms]
        
        # 术语清洗，防止DSL注入
        cleaned_terms = [self._clean_term(term) for term in top_terms if term.strip()]
        

        return {
            "terms": {
                "text": cleaned_terms,
                "boost": RRF_SPARSE_TERMS_BOOST  # 使用配置的boost值，提高稀疏向量查询权重
            }
        }

    def _create_rrf_pipeline_optimized(self, pipeline_id: str, rrf_params: dict = None):
        """
        创建优化的RRF Search Pipeline，支持版本控制和参数校验
        
        Args:
            pipeline_id: Pipeline ID
            rrf_params: RRF算法参数
        """
        # 参数校验
        required_params = ['window_size', 'rank_constant']
        for param in required_params:
            if param not in rrf_params:
                raise ValueError(f"创建RRF管道缺少必要参数: {param}")
        
        # 添加管道版本标识
        pipeline_body = {
            "description": f"RRF Pipeline v2 (constant={rrf_params['rank_constant']})",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": rrf_params["rank_constant"],
                            "window_size": rrf_params["window_size"]
                        }
                    }
                }
            ]
        }
        
        try:
            self.client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
            self.logger.info(f"RRF Pipeline创建成功: {pipeline_id}")
        except Exception as e:
            self.logger.error(f"RRF Pipeline创建失败: {str(e)}")
            raise

    def hybrid_search_rrf(self, query: str, sparse_vector: dict, dense_vector: list, 
                         filter: dict = None, top_k: int = 20, rrf_params: dict = None) -> dict:
        """
        优化后的RRF混合搜索实现
        
        Args:
            query: 查询文本
            sparse_vector: 稀疏向量字典 {term: weight}
            dense_vector: 密集向量列表
            filter: 过滤条件
            top_k: 返回结果数量
            rrf_params: RRF算法参数
            
        Returns:
            dict: 包含total和hits的搜索结果
        """
        try:
            # 1. 确保连接
            self._ensure_connection()
            
            # 2. 参数预处理和验证
            start_time = time.perf_counter()
            sparse_vector = self._normalize_sparse_vector(sparse_vector)
            self._validate_dense_vector(dense_vector)
            
            # 3. 配置RRF参数
            rrf_params = self._configure_rrf_params(rrf_params, top_k)
            
            self.logger.info(f"RRF Hybrid搜索 | 查询: '{query[:50]}...' | 稀疏术语数: {len(sparse_vector.get('indices', []))} | 密集维度: {len(dense_vector)}")
            
            # 4. 获取或创建RRF管道
            pipeline_id = self._get_or_create_rrf_pipeline(rrf_params)
            
            # 5. 构建优化查询
            search_body = self._build_optimized_rrf_query(
                query, sparse_vector, dense_vector, filter, top_k, rrf_params
            )
            
            # 6. 执行搜索
            response = self.client.search(
                body=search_body,
                index=self.index,
                search_pipeline=pipeline_id,
                request_timeout=30
            )
            
            # 7. 处理和分析结果
            hits = response.get('hits', {})
            self._analyze_search_results(hits, top_k)
            
            # 8. 记录性能指标
            duration = time.perf_counter() - start_time
            self._log_performance_metrics({
                'operation': 'hybrid_search_rrf',
                'duration': duration,
                'result_count': len(hits.get('hits', [])),
                'query_length': len(query)
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"RRF Hybrid搜索失败: {str(e)}", exc_info=True)
            raise

    def _build_hybrid_query_rrf(self, query: str, sparse_vector: dict, dense_vector: list, 
                               filter: dict = None, top_k: int = 20, rrf_params: dict = None) -> dict:
        """
        构建RRF混合搜索查询
        
        Args:
            query: 查询文本
            sparse_vector: 稀疏向量字典
            dense_vector: 密集向量列表
            filter: 过滤条件
            top_k: 返回结果数量
            rrf_params: RRF算法参数
            
        Returns:
            dict: 搜索查询体
        """
        # 构建稀疏向量查询（使用match查询替代rank_features）
        sparse_query = self._build_sparse_query(sparse_vector)
        
        # 构建密集向量查询
        dense_query = self._build_dense_query(dense_vector, top_k)
        
        # 构建混合查询（不包含rank.rrf，RRF逻辑通过Search Pipeline实现）
        search_body = {
            "size": min(top_k, 100),  # 分页保护，防止过大请求
            "query": {
                "hybrid": {
                    "queries": [
                        sparse_query,    # 稀疏向量查询
                        dense_query      # 密集向量查询
                    ]
                }
            }
        }
        
        # 添加过滤条件
        if filter:
            search_body["query"]["hybrid"]["filter"] = self._build_filter_query(filter)
        
        return search_body

    def _get_or_create_rrf_pipeline(self, rrf_params: dict = None) -> str:
        """
        获取或创建RRF Search Pipeline
        
        Args:
            rrf_params: RRF算法参数
            
        Returns:
            str: Pipeline ID
        """
        # 生成固定的Pipeline ID（基于参数）
        if rrf_params:
            rank_constant = rrf_params.get('rank_constant', RRF_DEFAULT_RANK_CONSTANT)
            window_size = rrf_params.get('window_size', RRF_MAX_WINDOW_SIZE)
        else:
            rank_constant = RRF_DEFAULT_RANK_CONSTANT
            window_size = RRF_MAX_WINDOW_SIZE
        
        # 使用固定算法生成Pipeline ID
        pipeline_id = f"rrf_pipeline_rc{rank_constant}_ws{window_size}"
        
        try:
            # 尝试获取现有Pipeline
            self.client.search_pipeline.get(id=pipeline_id)
            self.logger.info(f"✅ 复用现有RRF Pipeline: {pipeline_id}")
        except Exception:
            # Pipeline不存在，创建新的
            self.logger.info(f"🆕 创建新RRF Pipeline: {pipeline_id}")
            self._create_rrf_pipeline(pipeline_id, rrf_params)
        
        return pipeline_id
    
    def _create_rrf_pipeline(self, pipeline_id: str, rrf_params: dict = None):
        """
        创建RRF Search Pipeline
        
        Args:
            pipeline_id: Pipeline ID
            rrf_params: RRF算法参数
        """
        # 设置默认参数
        default_params = {
            "window_size": 20,
            "rank_constant": RRF_DEFAULT_RANK_CONSTANT
        }
        
        if rrf_params:
            default_params.update(rrf_params)
        
        pipeline_body = {
            "description": "RRF Pipeline for hybrid search result fusion",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": default_params["rank_constant"]
                        }
                    }
                }
            ]
        }
        
        try:
            self.client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
            self.logger.info(f"成功创建RRF Pipeline: {pipeline_id}")
        except Exception as e:
            self.logger.error(f"创建RRF Pipeline失败: {str(e)}")
            raise

    def _build_sparse_query(self, sparse_vector: dict, query_text: str = None) -> dict:
        """
        构建稀疏向量查询，默认使用match_phrase策略
        
        Args:
            sparse_vector: 稀疏向量字典
            query_text: 查询文本，用于match_phrase查询
            
        Returns:
            dict: 查询体
        """
        # 默认使用match_phrase策略进行试验
        if query_text:
            #return self._build_sparse_query_terms(sparse_vector)
            return self._build_match_phrase_query(query_text)
        else:
            # 兜底：返回通用文本搜索
            return {
                "match": {
                    "text": {
                        "query": "*",
                        "boost": 1.0
                    }
                }
            }
    
    def _build_sparse_query_terms(self, sparse_vector: dict) -> dict:
        """
        构建稀疏向量查询，使用terms策略（用于对比试验）
        
        Args:
            sparse_vector: 稀疏向量字典
            query_text: 查询文本（此方法中不使用）
            
        Returns:
            dict: 查询体
        """
        return self._build_terms_query(sparse_vector)



    def _is_query_text_quality_good(self, query_text: str) -> bool:
        """评估查询文本质量"""
        if not query_text or len(query_text.strip()) < 2:
            return False
        
        # 检查是否包含有效关键词
        meaningful_words = [word for word in query_text.split() if len(word) > 1]
        return len(meaningful_words) >= 2

    def _is_sparse_vector_quality_good(self, sparse_vector: dict) -> bool:
        """评估稀疏向量质量"""
        if not sparse_vector:
            return False
        
        # 检查是否有足够的有效术语
        if 'indices' in sparse_vector:
            return len(sparse_vector.get('indices', [])) >= 3
        else:
            return len(sparse_vector) >= 3

    def _build_match_phrase_query(self, query_text: str) -> dict:
        """构建match_phrase查询，支持模糊匹配过滤"""
        
        # 预处理查询文本
        processed_query = self._preprocess_query_text(query_text)
        
        # 构建基础查询
        query_body = {
            "match": {
                "text": {
                    "query": processed_query,
                    "boost": MATCH_PHRASE_CONFIG['default_boost'],
                    "operator": MATCH_PHRASE_CONFIG['operator'],
                    "fuzziness": MATCH_PHRASE_CONFIG['fuzziness'],
                    "max_expansions": MATCH_PHRASE_CONFIG['max_expansions'],
                    "prefix_length": MATCH_PHRASE_CONFIG['prefix_length']
                }
            }
        }
        
        # 添加模糊匹配过滤配置（DeepSeek建议）
        if MATCH_PHRASE_CONFIG.get('fuzzy_filters'):
            query_body["match"]["text"]["fuzzy_options"] = MATCH_PHRASE_CONFIG['fuzzy_filters']
        
        return query_body

    def _build_terms_query(self, sparse_vector: dict) -> dict:
        """构建terms查询（保持当前实现）"""
        if not sparse_vector:
            # 兜底：返回通用文本搜索
            return {
                "match": {
                    "text": {
                        "query": "*",
                        "boost": 1.0
                    }
                }
            }
        
        # 处理不同格式的稀疏向量
        if isinstance(sparse_vector, dict) and 'indices' in sparse_vector:
            # 格式1: {indices: [...], values: [...]}
            if 'values' in sparse_vector and len(sparse_vector['indices']) == len(sparse_vector['values']):
                # 使用堆排序获取前10个最重要的术语，性能优化 O(n log k)
                term_weights = list(zip(sparse_vector['indices'], sparse_vector['values']))
                top_terms = heapq.nlargest(10, term_weights, key=lambda x: x[1])
                top_terms = [str(term) for term, _ in top_terms]
            else:
                # 只有indices，直接使用
                top_terms = [str(idx) for idx in sparse_vector['indices'][:10]]
        else:
            # 格式2: {term: weight} - 使用堆排序优化性能
            top_terms = heapq.nlargest(10, sparse_vector.items(), key=lambda x: x[1])
            top_terms = [term for term, _ in top_terms]
        
        # 术语清洗，防止DSL注入
        cleaned_terms = [self._clean_term(term) for term in top_terms if term.strip()]
        
        return {
            "terms": {
                "text": cleaned_terms,
                "boost": RRF_SPARSE_TERMS_BOOST  # 使用配置的boost值，提高稀疏向量查询权重
            }
        }



    def _apply_fuzzy_filters(self, query_text: str) -> str:
        """应用模糊匹配过滤，保护专有名词，过滤高频词"""
        
        if not MATCH_PHRASE_CONFIG.get('fuzzy_filters'):
            return query_text
        
        # 分词处理
        words = query_text.split()
        filtered_words = []
        
        for word in words:
            # 检查词长度
            if len(word) < MATCH_PHRASE_CONFIG['term_frequency_thresholds']['min_word_length']:
                continue
            if len(word) > MATCH_PHRASE_CONFIG['term_frequency_thresholds']['max_word_length']:
                continue
            
            # 这里可以添加更复杂的词频检查逻辑
            # 例如：检查是否在专有名词词典中，或检查文档频率
            filtered_words.append(word)
        
        return ' '.join(filtered_words)

    def _get_cached_query(self, query_text: str, strategy: str) -> dict:
        """获取缓存的查询"""
        cache_key = f"{query_text}_{strategy}"
        return getattr(self, '_query_cache', {}).get(cache_key)

    def _cache_query(self, query_text: str, strategy: str, query_body: dict):
        """缓存查询"""
        if not hasattr(self, '_query_cache'):
            self._query_cache = {}
        cache_key = f"{query_text}_{strategy}"
        self._query_cache[cache_key] = query_body

    def _preprocess_query_text(self, query_text: str) -> str:
        """预处理查询文本"""
        # 1. 去除多余空格
        query_text = ' '.join(query_text.split())
        
        # 2. 特殊字符处理
        query_text = self._clean_term(query_text)
        
        # 3. 长度限制
        if len(query_text) > 200:
            query_text = query_text[:200]
        
        return query_text

    def _build_dense_query(self, dense_vector: list, top_k: int) -> dict:
        """
        构建密集向量查询
        
        Args:
            dense_vector: 密集向量列表
            top_k: 返回结果数量
            
        Returns:
            dict: 密集向量查询体
        """
        return {
            "knn": {
                "embedding": {
                    "vector": dense_vector,
                    "k": top_k
                }
            }
        }

    # ========== 混合搜索优化方法骨架 ==========
    
    def _build_heading_query(self, query: str) -> dict:
        """构建层级标题查询，权重递减"""
        
        # 使用配置权重，动态构建字段列表
        fields = [
            f"{field}^{weight}" 
            for field, weight in HEADING_WEIGHTS.items()
        ]
        
        return {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "analyzer": OPENSEARCH_SEARCH_ANALYZER,  # 使用配置的analyzer
                "tie_breaker": 0.3
            }
        }
    
    def _calculate_content_boost(self, query: str) -> float:
        """根据查询特性动态计算正文权重（带边界检查）"""

        
        query_length = len(query.split())
        
        # 获取阈值配置（带默认值）
        short_threshold = QUERY_LENGTH_THRESHOLDS.get('short', 4)
        long_threshold = QUERY_LENGTH_THRESHOLDS.get('medium', 8)
        
        # 长查询/详细描述 - 提高正文权重
        if query_length > long_threshold:
            return BOOST_WEIGHTS.get('long_query', 1.4)
        
        # 中等长度查询
        elif query_length > short_threshold:
            return BOOST_WEIGHTS.get('medium_query', 1.2)
        
        # 短查询/关键词 - 降低正文权重，侧重标题
        else:
            return BOOST_WEIGHTS.get('short_query', 0.8)
    
    def _build_content_query(self, query: str) -> dict:
        """构建正文字段查询"""
        return {
            "match": {
                "text": {
                    "query": query,
                    "analyzer": OPENSEARCH_SEARCH_ANALYZER,  # 使用配置的analyzer
                    "boost": self._calculate_content_boost(query),
                    "operator": "or"
                }
            }
        }
    
    def _build_metadata_query(self, query: str) -> dict:
        """构建元数据字段查询"""
        # 使用配置化的字段权重，构建fields列表
        fields = [f"{field}^{weight}" for field, weight in SEARCH_FIELD_WEIGHTS.items()]
        
        return {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "analyzer": OPENSEARCH_SEARCH_ANALYZER  # 使用配置的analyzer
            }
        }
    
    def _build_content_type_filter(self, query: str) -> dict:
        """根据查询内容智能添加内容类型过滤"""
        # 规则相关查询
        if any(word in query for word in ["规则", "规则书", "codex", "FAQ"]):
            return {"content_type": "corerule"}
        
        # 单位相关查询
        elif any(word in query for word in ["单位", "模型", "unit", "数据表"]):
            return {"chunk_type": "unit_data"}
        
        # 背景相关查询
        elif any(word in query for word in ["背景", "故事", "lore", "历史"]):
            return {"content_type": "background"}
        
        return {}
    
    def _build_hybrid_query_optimized(self, query: str, query_vector: list, 
                                     filter: dict = None, top_k: int = 20) -> dict:
        """构建优化后的混合搜索查询"""
        
        # 1. 层级标题查询
        heading_query = self._build_heading_query(query)
        
        # 2. 正文字段查询
        content_query = self._build_content_query(query)
        
        # 3. 元数据查询
        metadata_query = self._build_metadata_query(query)
        
        # 4. 向量搜索
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": min(100, top_k * 3),  # 扩大召回池
                    "boost": PIPELINE_VECTOR_BOOST  # 使用配置文件中的常量
                }
            }
        }
        
        # 5. BM25文本查询
        bm25_query = {
            "match": {
                "text": {
                    "query": query,
                    "analyzer": OPENSEARCH_SEARCH_ANALYZER,  # 使用配置的analyzer
                    "boost": PIPELINE_BM25_BOOST  # 使用配置文件中的常量
                }
            }
        }
        
        # 6. 组合查询
        hybrid_query = {
            "hybrid": {
                "queries": [
                    heading_query,      # 层级标题
                    bm25_query,         # BM25文本搜索
                    content_query,      # 正文内容
                    metadata_query,     # 元数据
                    knn_query          # 向量搜索
                ]
            }
        }
        
        # 7. 智能过滤（不覆盖用户显式设置的过滤条件）
        if filter:
            smart_filter = self._build_content_type_filter(query)
            if smart_filter:
                # 只添加用户未指定的过滤条件，避免覆盖用户设置
                for key, value in smart_filter.items():
                    if key not in filter:
                        filter[key] = value
            
            hybrid_query["hybrid"]["filter"] = self._build_filter_query(filter)
        
        return {
            "size": top_k,
            "query": hybrid_query  # 直接返回 hybrid_query，保持过滤条件
        }
    
    def _collect_and_log_analysis(self, query: str, response: dict, start_time: float):
        """收集和记录精简的搜索分析信息"""
        try:
            # 收集精简的explain信息
            explain_result = self._extract_explain_info(response)
            
            # 调用_analyze API
            analyze_result = self._call_analyze_api(query)
            
            # 构建精简的日志数据
            analysis_data = {
                "query": query,
                "tokens": analyze_result["tokens"],
                "results": explain_result,
                "execution_time": round(time.time() - start_time, 3),
                "result_count": response.get("hits", {}).get("total", {}).get("value", 0)
            }
            
            # 写入日志
            logger = self._get_search_analysis_logger()
            logger.log_search_analysis(analysis_data)
            
        except Exception as e:
            self.logger.warning(f"搜索分析信息收集失败: {str(e)}")
    
    def _extract_explain_info(self, response: dict) -> list:
        """从搜索响应中提取精简的explain信息"""
        explain_results = []
        seen_doc_ids = set()  # 用于去重
        
        for hit in response.get("hits", {}).get("hits", []):
            doc_id = hit["_id"]
            
            # 如果已经见过这个doc_id，跳过
            if doc_id in seen_doc_ids:
                continue
                
            seen_doc_ids.add(doc_id)
            
            # 提取基本信息
            doc_info = {
                "doc_id": doc_id,
                "score": hit["_score"]
            }
            
            # 提取explanation中的关键信息
            explanation = hit.get("_explanation", {})
            if explanation:
                # 提取文本相似度（TF-IDF相关评分）
                text_score = self._extract_text_similarity_score(explanation)
                if text_score is not None:
                    doc_info["text_similarity"] = text_score
                
                # 提取向量相似度（KNN相关评分）
                vector_score = self._extract_vector_similarity_score(explanation)
                if vector_score is not None:
                    doc_info["vector_similarity"] = vector_score
            
            explain_results.append(doc_info)
        return explain_results
    
    def _extract_text_similarity_score(self, explanation: dict) -> float:
        """从explanation中提取文本相似度评分"""
        try:
            # 查找包含文本匹配的评分部分
            if "details" in explanation:
                for detail in explanation["details"]:
                    if "description" in detail and "text:" in detail.get("description", ""):
                        return detail.get("value", 0.0)
            return None
        except Exception:
            return None
    
    def _extract_vector_similarity_score(self, explanation: dict) -> float:
        """从explanation中提取向量相似度评分"""
        try:
            # 查找包含KNN搜索的评分部分
            if "details" in explanation:
                for detail in explanation["details"]:
                    if "description" in detail and "knn search" in detail.get("description", "").lower():
                        return detail.get("value", 0.0)
            return None
        except Exception:
            return None
    
    def _call_analyze_api(self, query: str) -> dict:
        """调用OpenSearch的_analyze API，返回精简的分词结果"""
        try:
            body = {
                "analyzer": OPENSEARCH_SEARCH_ANALYZER,  # 使用配置的analyzer
                "text": query
            }
            response = self.client.indices.analyze(index=self.index, body=body)
            return {
                "tokens": [t["token"] for t in response.get("tokens", [])]
            }
        except Exception as e:
            return {
                "error": str(e),
                "tokens": []
            }
    
    def _get_search_analysis_logger(self):
        """获取搜索分析日志记录器"""
        if self._search_analysis_logger is None:
            self._search_analysis_logger = SearchAnalysisLogger()
        return self._search_analysis_logger


class SearchAnalysisLogger:
    """搜索分析日志记录器"""
    
    def __init__(self, log_file=None, max_size=None):
        self.log_file = log_file or SEARCH_ANALYSIS_LOG_FILE
        self.max_size = max_size or SEARCH_ANALYSIS_LOG_MAX_SIZE
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        # 创建日志目录
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 检查文件大小，超过限制时轮转
        if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_size:
            self._rotate_log_file()
    
    def _rotate_log_file(self):
        """轮转日志文件"""
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        base_name = os.path.splitext(os.path.basename(self.log_file))[0]
        log_dir = os.path.dirname(self.log_file)
        new_filename = os.path.join(log_dir, f"{base_name}_{timestamp}.log")
        os.rename(self.log_file, new_filename)
    
    def log_search_analysis(self, analysis_data):
        """记录搜索分析信息"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(analysis_data, ensure_ascii=False) + '\n')
        except Exception as e:
            # 静默忽略日志写入失败，不影响搜索功能
            pass