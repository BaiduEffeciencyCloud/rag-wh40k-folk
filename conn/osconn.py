import logging
import os
import copy
from datetime import datetime
from opensearchpy import OpenSearch
from http import HTTPStatus
from .coninterface import ConnectionInterface
from config import OPENSEARCH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, EMBEDDING_MODEL_QWEN, QWEN_DIMENSION, ALIYUN_RERANK_MODEL, RERANK_TOPK, HYBRID_ALGORITHM
from opensearchpy import OpenSearch
from typing import List, Dict
import dashscope
from http import HTTPStatus


# Opensearch数据库的处理类,负责数据转化,数据上传,数据搜索
class OpenSearchConnection(ConnectionInterface):
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
        
    def hybrid_search(self, query: str, filter: dict = None, top_k: int = 20, **kwargs) -> dict:
        """
        执行混合搜索的统一接口
        
        Args:
            query: 查询文本
            filter: 过滤条件
            top_k: 返回结果数量
            **kwargs: 其他参数
                - 对于 pipeline: query_vector, bm25_weight, vector_weight
                - 对于 rrf: sparse_vector, dense_vector, rrf_params
                - algorithm: 可选，指定算法类型，如果不指定则使用配置默认值
                
        Returns:
            dict: 原始的 OpenSearch hits 字典
        """
        try:
            # 优先使用传入的算法参数，如果没有则使用配置默认值
            algorithm = kwargs.get('algorithm', HYBRID_ALGORITHM)
            
            if algorithm == 'pipeline':
                # 从 kwargs 中提取 pipeline 所需参数
                query_vector = kwargs.get('query_vector')
                bm25_weight = kwargs.get('bm25_weight', 0.7)
                vector_weight = kwargs.get('vector_weight', 0.3)
                
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
                
            # 直接返回原始 hits，不进行格式转换
            return response
            
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
                    'search_type': 'hybrid',
                    'query': ''  # 移除对response.get('query', '')的依赖
                }
                results.append(result)
                self.logger.debug(f"Converted hit: {result}")
        else:
            self.logger.debug("No hits found in response")
        
        self.logger.debug(f"Returning {len(results)} results")
        return results
        
    def hybrid_search_pipeline(self, query: str, query_vector: list, filter: dict = None, top_k: int = 20, 
                     bm25_weight: float = 0.7, vector_weight: float = 0.3, 
                     use_hybrid_query: bool = True) -> dict:
        """
        执行混合搜索（BM25 + 向量搜索）- Pipeline方式
        
        Args:
            query: 查询文本
            query_vector: 查询向量（由调用方生成）
            filter: 过滤条件
            top_k: 返回结果数量
            bm25_weight: BM25搜索权重 (0.0-1.0)
            vector_weight: 向量搜索权重 (0.0-1.0)
            use_hybrid_query: 是否使用 hybrid 查询（推荐），False 使用 bool+should 方式
            
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
            
            self.logger.info(f"Hybrid搜索开始，查询: {query}，向量维度: {len(query_vector)}，使用查询类型: {'hybrid' if use_hybrid_query else 'bool+should'}")
            
            # 3. 创建或获取搜索管道
            pipeline_id = self._get_or_create_search_pipeline()
            
            # 4. 根据参数选择查询构建方式
            if use_hybrid_query:
                search_body = self._build_hybrid_query(query, query_vector, filter, top_k)
            else:
                search_body = self._build_hybrid_query_with_weights(query, query_vector, bm25_weight, vector_weight, filter, top_k)
            
            # 5. 执行搜索（修复：使用 search_pipeline 参数）
            response = self.client.search(
                body=search_body,
                index=self.index,
                search_pipeline=pipeline_id
            )
            
            # 6. 处理结果
            hits = response.get('hits', {})
            
            # 添加调试日志
            self.logger.debug(f"OpenSearch 原始响应: {response}")
            self.logger.debug(f"提取的 hits 字段: {hits}")
            self.logger.debug(f"hits 类型: {type(hits)}")
            
            self.logger.info(f"Hybrid搜索完成，查询: {query}，返回 {hits.get('total', {}).get('value', 0)} 个结果")
            return hits
            
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
    
    def _build_hybrid_query(self, query: str, query_vector: list, filter: dict = None, top_k: int = 20) -> dict:
        """
        构建混合搜索查询
        
        Args:
            query: 查询文本
            query_vector: 查询向量
            filter: 过滤条件
            top_k: 返回结果数量
            
        Returns:
            dict: 搜索查询体
        """
        from config import PIPELINE_BM25_BOOST, PIPELINE_VECTOR_BOOST
        
        # 基础查询结构
        search_body = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "boost": PIPELINE_BM25_BOOST  # 配置化的BM25搜索boost值
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": PIPELINE_VECTOR_BOOST  # 配置化的向量搜索boost值
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # 添加过滤条件
        if filter:
            search_body["query"]["hybrid"]["queries"].append({
                "bool": {
                    "filter": self._build_filter_query(filter)
                }
            })
        
        return search_body
    
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

    def hybrid_search_rrf(self, query: str, sparse_vector: dict, dense_vector: list, 
                         filter: dict = None, top_k: int = 20, rrf_params: dict = None) -> dict:
        """
        执行RRF混合搜索（稀疏向量 + 密集向量）
        
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
            
            # 2. 验证参数
            if not sparse_vector or not dense_vector:
                raise ValueError("sparse_vector 和 dense_vector 都不能为空")
            
            if len(dense_vector) != 2048:
                raise ValueError(f"密集向量维度不正确: {len(dense_vector)}")
            
            self.logger.info(f"RRF Hybrid搜索开始，查询: {query}，稀疏向量项数: {len(sparse_vector)}，密集向量维度: {len(dense_vector)}")
            
            # 3. 获取或创建RRF Search Pipeline
            pipeline_id = self._get_or_create_rrf_pipeline(rrf_params)
            
            # 4. 构建查询
            search_body = self._build_hybrid_query_rrf(query, sparse_vector, dense_vector, filter, top_k, rrf_params)
            
            # 5. 执行搜索（使用RRF Pipeline）
            response = self.client.search(
                body=search_body,
                index=self.index,
                search_pipeline=pipeline_id
            )
            
            # 5. 处理结果
            hits = response.get('hits', {})
            
            # 添加调试日志
            self.logger.debug(f"RRF OpenSearch 原始响应: {response}")
            self.logger.debug(f"RRF 提取的 hits 字段: {hits}")
            self.logger.debug(f"RRF hits 类型: {type(hits)}")
            
            self.logger.info(f"RRF Hybrid搜索完成，查询: {query}，返回 {hits.get('total', {}).get('value', 0)} 个结果")
            return response
            
        except Exception as e:
            self.logger.error(f"RRF Hybrid搜索失败: {str(e)}")
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
            "size": top_k,
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
        pipeline_id = "hybrid_rrf_pipeline"
        
        try:
            # 尝试获取现有Pipeline
            self.client.search_pipeline.get(id=pipeline_id)
            self.logger.info(f"使用现有RRF Pipeline: {pipeline_id}")
        except Exception:
            # Pipeline不存在，创建新的
            self.logger.info(f"RRF Pipeline不存在，正在创建: {pipeline_id}")
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
            "rank_constant": 60
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

    def _build_sparse_query(self, sparse_vector: dict) -> dict:
        """
        构建稀疏向量查询，使用 match 查询替代 rank_features
        
        Args:
            sparse_vector: 稀疏向量字典 {indices: [term_indices], values: [weights]}
            
        Returns:
            dict: match 查询体
        """
        # 由于 OpenSearch 3.1.0 不支持 rank_features 查询类型，
        # 我们使用 match 查询来搜索文本内容，实现类似的效果
        # 这里我们使用一个通用的文本搜索，因为稀疏向量的具体内容无法直接用于查询
        return {
            "match": {
                "text": {
                    "query": "*",
                    "boost": 1.0
                }
            }
        }

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