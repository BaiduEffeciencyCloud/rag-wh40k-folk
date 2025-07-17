import os
import sys
from typing import Dict, Any, List, Union
import pinecone
from openai import OpenAI
import logging

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, EMBADDING_MODEL, HYBRID_ALPHA, RERANK_MODEL,PINECONE_API_KEY,RERANK_TOPK
from .search_interface import SearchEngineInterface
from .dense_search import DenseSearchEngine
from .base_search import BaseSearchEngine
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class HybridSearchEngine(BaseSearchEngine, SearchEngineInterface):
    """Hybrid混合搜索引擎，融合dense和sparse搜索"""
    
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        super().__init__(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        # 初始化dense搜索引擎作为降级备用
        self.dense_engine = DenseSearchEngine(pinecone_api_key, index_name, openai_api_key, pinecone_environment)
        # 加载BM25Manager
        self.bm25_manager = None
        # 初始化 Pinecone 实例用于 inference.rerank
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._load_bm25_manager()
        logger.info(f"Hybrid搜索引擎初始化完成，索引: {self.index_name}, 环境: {self.pinecone_environment}")

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

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        Args:
            text: 输入文本
        Returns:
            向量表示
        """
        try:
            from config import get_embedding_model
            embeddings = get_embedding_model()
            embedding = embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {str(e)}")
            raise

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
            **kwargs: 其他参数（如filter、alpha等）
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
        # 获取filter参数
        filter_dict = kwargs.get('filter', {})
        # 获取alpha参数
        alpha = kwargs.get('alpha', HYBRID_ALPHA)
        # 生成dense向量（如失败直接raise异常）
        query_embedding = self.get_embedding(query_text)
        # 生成sparse向量（如失败降级dense）
        try:
            sparse_vector = self._generate_sparse_vector(query_text)
        except Exception as e:
            logger.warning(f"sparse向量生成失败，降级为dense检索: {str(e)}")
            return self.dense_engine.search(query, top_k, **kwargs)
        # 构建hybrid搜索参数
        search_params = {
            "vector": query_embedding,
            "sparse_vector": sparse_vector,
            "top_k": top_k,
            "include_metadata": True,
            "alpha": alpha
        }
        if filter_dict:
            search_params['filter'] = filter_dict
        # 执行hybrid搜索（如失败降级dense）
        try:
            pinecone_response = self.index.query(**search_params)
            if hasattr(pinecone_response,"matches"):
                candidates = pinecone_response.matches
            else:
                logger.error("Pinecone返回对象无matches属性")
                raise e
        except Exception as e:
            logger.warning(f"Pinecone hybrid检索失败: {str(e)}，降级为dense检索")
            return self.dense_engine.search(query, top_k, **kwargs)
        
        #执行rerank
        rerank_results = self.rerank(query_text,candidates,RERANK_TOPK,model=RERANK_MODEL)
        # 处理搜索结果
        results = self.composite(rerank_results,query_text,"hybrid")

        logger.info(f"Hybrid检索完成，查询: {query_text[:50]}...，返回 {len(results)} 个结果，alpha: {alpha}")
        return results

        # 3. 组装补全后的结果  
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
            "hybrid_alpha": HYBRID_ALPHA,
            "index_name": self.index_name
        } 