import logging
import os
from typing import Dict, List, Any, Union
from searchengine.base_search import BaseSearchEngine
from searchengine.search_interface import SearchEngineInterface

logger = logging.getLogger(__name__)


class SparseSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """基于BM25的稀疏检索引擎"""
    
    def __init__(self, index_name: str = None):
        """
        初始化SparseSearchEngine
        Args:
            index_name: Pinecone索引名称，如果为None则使用配置中的默认值
        """
        super().__init__(index_name)
        self.bm25_manager = None
        self.bm25_config = None
        self._load_bm25_manager()
    
    def _load_bm25_manager(self):
        """加载BM25Manager和词典"""
        try:
            # 使用BM25Config初始化BM25Manager
            from dataupload.bm25_config import BM25Config
            from dataupload.bm25_manager import BM25Manager
            
            self.bm25_config = BM25Config()
            self.bm25_manager = BM25Manager(
                k1=self.bm25_config.k1,
                b=self.bm25_config.b,
                min_freq=self.bm25_config.min_freq,
                max_vocab_size=self.bm25_config.max_vocab_size
            )
            
            # 使用BM25Manager的现成方法获取词典路径
            vocab_path = self.bm25_manager.get_latest_dict_file()
            
            if not vocab_path or not os.path.exists(vocab_path):
                logger.error("未找到有效的词汇表文件")
                raise FileNotFoundError("未找到有效的词汇表文件")
            
            # 尝试加载模型文件（如果存在）
            model_path = self.bm25_config.get_model_path()
            
            if os.path.exists(model_path):
                # 如果模型文件存在，使用load_model方法
                self.bm25_manager.load_model(model_path, vocab_path)
                logger.info(f"成功加载BM25模型，模型路径: {model_path}, 词典路径: {vocab_path}")
            else:
                # 如果模型文件不存在，只加载词汇表
                self.bm25_manager.load_dictionary(vocab_path)
                logger.info(f"成功加载BM25词汇表，词典路径: {vocab_path}（模型文件不存在，需要重新训练）")
            
        except Exception as e:
            logger.error(f"加载BM25Manager失败: {str(e)}")
            raise
    
    def _tokenize_query(self, query_text: str) -> List[str]:
        """
        对query进行分词
        Args:
            query_text: 查询文本
        Returns:
            List[str]: 分词结果列表
        """
        # 使用BM25Manager的现成分词方法
        return self.bm25_manager.tokenize_chinese(query_text)
    
    def _generate_sparse_vector(self, query_text: str) -> Dict[str, List]:
        """
        生成query的sparse向量
        Args:
            query_text: 查询文本
        Returns:
            Dict[str, List]: 包含indices和values的稀疏向量
        """
        try:
            # 分词
            tokens = self._tokenize_query(query_text)
            logger.info(f"查询分词结果: {tokens}")
            
            # 生成稀疏向量
            sparse_vector = self.bm25_manager.get_sparse_vector(query_text)
            logger.info(f"BM25稀疏向量生成成功，indices: {sparse_vector['indices'][:5]}...")
            
            # Pinecone要求sparse_vector['indices']必须为int类型，但历史上upsert.py上传时为float类型。
            # 这里做int强制转换，是为了兼容历史数据和接口，仅影响查询流程，不影响上传/构建流程。
            sparse_vector['indices'] = [int(i) for i in sparse_vector['indices']]

            return sparse_vector
            
        except Exception as e:
            logger.info(f"生成sparse向量失败: {str(e)}")
            raise  # 向上抛出，由search方法捕获
    
    def _execute_sparse_search(self, sparse_vector: Dict, query_text: str, top_k: int, **kwargs) -> List[Dict]:
        """
        执行Pinecone sparse检索
        Args:
            sparse_vector: 稀疏向量
            query_text: 原始查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数（如filter等）
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 构建搜索参数
            search_params = {
                "sparse_vector": sparse_vector,
                "top_k": top_k,
                "include_metadata": True
            }
            
            # 添加filter参数（如果提供）
            if 'filter' in kwargs:
                search_params['filter'] = kwargs['filter']
            
            # 执行搜索
            response = self.index.query(**search_params)
            logger.info(f"Pinecone sparse检索成功，返回 {len(response.matches)} 个原始结果")
            
            # 处理结果
            results = self._process_search_results(response, query_text)
            return results
            
        except Exception as e:
            logger.info(f"Pinecone sparse检索执行失败: {str(e)}")
            raise  # 向上抛出，由search方法捕获
    
    def _process_search_results(self, pinecone_response, query_text: str) -> List[Dict[str, Any]]:
        """
        处理Pinecone返回的搜索结果，确保格式与dense_search一致
        Args:
            pinecone_response: Pinecone查询响应
            query_text: 原始查询文本
        Returns:
            List[Dict[str, Any]]: 格式化的搜索结果列表
        """
        results = []
        for match in pinecone_response.matches:
            text = match.metadata.get('text', '')
            if not text or len(text.strip()) < 10:
                continue
            result = {
                'doc_id': match.id,
                'text': text,
                'score': match.score,
                'metadata': match.metadata,
                'search_type': 'sparse',  # 标识为sparse检索
                'query': query_text
            }
            results.append(result)
        return results
    
    def search(self, query: Union[str, Dict[str, Any]], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        执行sparse检索
        Args:
            query: 查询（字符串或结构化查询对象）
            top_k: 返回结果数量
            **kwargs: 其他参数（如filter等）
        Returns:
            List[Dict[str, Any]]: 搜索结果列表，每个结果包含：
            - doc_id (str): 文档ID
            - text (str): 文档文本内容
            - score (float): 相关性分数
            - metadata (Dict): 元数据
            - search_type (str): 搜索类型，固定为'sparse'
            - query (str): 原始查询文本
            失败时返回空列表
        """
        try:
            # 处理查询输入
            if isinstance(query, dict):
                query_text = query.get('text', '')
                if not query_text:
                    logger.info("结构化查询中没有找到text字段，返回空结果")
                    return []
            else:
                query_text = str(query)
            
            if not query_text.strip():
                logger.info("查询文本为空，返回空结果")
                return []
            
            # 生成sparse向量
            try:
                sparse_vector = self._generate_sparse_vector(query_text)
                logger.info(f"成功生成sparse向量，indices数量: {len(sparse_vector['indices'])}")
            except Exception as e:
                logger.info(f"生成sparse向量失败: {str(e)}，返回空结果")
                return []
            
            # 执行sparse检索
            try:
                results = self._execute_sparse_search(sparse_vector, query_text, top_k, **kwargs)
                logger.info(f"Sparse检索完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果")
                return results
            except Exception as e:
                logger.info(f"Pinecone sparse检索失败: {str(e)}，返回空结果")
                return []
                
        except Exception as e:
            logger.info(f"Sparse检索过程中发生未预期错误: {str(e)}，返回空结果")
            return []
    
    def get_type(self) -> str:
        """返回搜索引擎类型"""
        return "sparse"
    

    
    def get_sparse_vector_info(self, query_text: str) -> Dict[str, Any]:
        """
        获取query的sparse向量详细信息（调试用）
        Args:
            query_text: 查询文本
        Returns:
            Dict[str, Any]: 包含分词结果、词典映射、稀疏向量等详细信息
        """
        try:
            # 分词
            tokens = self._tokenize_query(query_text)
            
            # 生成稀疏向量
            sparse_vector = self._generate_sparse_vector(query_text)
            
            # 使用BM25Manager的现成方法获取词典统计信息
            vocab_stats = self.bm25_manager.get_vocabulary_stats()
            
            return {
                "query_text": query_text,
                "tokens": tokens,
                "sparse_vector": sparse_vector,
                "vocab_size": vocab_stats['vocabulary_size'],
                "indices_count": len(sparse_vector.get('indices', [])),
                "values_count": len(sparse_vector.get('values', []))
            }
        except Exception as e:
            logger.error(f"获取sparse向量信息失败: {str(e)}")
            return {
                "error": str(e),
                "query_text": query_text
            }
