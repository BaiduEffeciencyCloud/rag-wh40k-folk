import numpy as np
import asyncio
from typing import List, Dict, Any, Optional
from ..interfaces import PostSearchInterface
import logging
from engineconfig.post_search_config import get_processor_config



logger = logging.getLogger(__name__)

class MMRProcessor(PostSearchInterface):
    """MMR多样性处理器 - 最大化边际相关性，增加结果多样性"""
    def __init__(self, **kwargs):
        """
        初始化 MMR 处理器
        
        Args:
            **kwargs: 其他参数
        """
        pass

    def process(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """同步处理方法，保持向后兼容"""
        # 调用异步版本
        return asyncio.run(self._process_async(results, **kwargs))
    
    async def _process_async(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        # 从get_config_schema获取默认参数，kwargs中的参数优先级更高
        schema = self.get_config_schema()
        lambda_param = kwargs.get('lambda_param', schema['lambda_param']['default'])
        select_n = kwargs.get('select_n', schema['select_n']['default'])
        enable_mmr = kwargs.get('enable_mmr', schema['enable_mmr']['default'])
        min_text_length = kwargs.get('min_text_length', schema['min_text_length']['default'])
        # 零向量保护参数
        zero_vector_protection = kwargs.get('zero_vector_protection', schema['zero_vector_protection']['default'])
        zero_vector_strategy = kwargs.get('zero_vector_strategy', schema['zero_vector_strategy']['default'])
        min_norm_threshold = kwargs.get('min_norm_threshold', schema['min_norm_threshold']['default'])
        use_cache_optimization = kwargs.get('use_cache_optimization', schema['use_cache_optimization']['default'])
        output_order = kwargs.get('output_order', schema['output_order']['default'])
        reverse_order = kwargs.get('reverse_order', schema['reverse_order']['default'])
        query_info = kwargs.get('query_info', {})
        # 验证配置
        process_config = {
            'lambda_param': lambda_param,
            'select_n': select_n,
            'enable_mmr': enable_mmr,
            'min_text_length': min_text_length,
            'zero_vector_protection': zero_vector_protection,
            'zero_vector_strategy': zero_vector_strategy,
            'min_norm_threshold': min_norm_threshold,
            'use_cache_optimization': use_cache_optimization,
            'output_order': output_order,
            'reverse_order': reverse_order
        }
        is_valid, errors = self.validate_config(process_config)
        if not is_valid:
            logger.warning(f"MMRProcessor配置验证失败: {errors}")
            # 使用配置文件中的默认值继续处理
            lambda_param = schema['lambda_param']['default']
            select_n = schema['select_n']['default']
            enable_mmr = schema['enable_mmr']['default']
            min_text_length = schema['min_text_length']['default']
            zero_vector_protection = schema['zero_vector_protection']['default']
            zero_vector_strategy = schema['zero_vector_strategy']['default']
            min_norm_threshold = schema['min_norm_threshold']['default']
            use_cache_optimization = schema['use_cache_optimization']['default']
            output_order = schema['output_order']['default']
            reverse_order = schema['reverse_order']['default']
        
        if not results or not enable_mmr:
            logger.info(f"MMR处理跳过 - results数量: {len(results) if results else 0}, enable_mmr: {enable_mmr}")
            return results
        
        logger.info("开始处理query embedding:")
        logger.info(f"query_info: {query_info}")
        
        # 获取 embedding_model 实例
        logger.info("尝试获取embedding_model实例...")
        embedding_model_instance = kwargs.get('embedding_model')
        
        if not embedding_model_instance:
            logger.warning("embedding_model_instance为空，返回原始结果")
            return results
            
        logger.info("开始提取query_text...")
        query_text = self._extract_query_text(results, query_info)
        logger.info(f"提取的query_text: {query_text}")
        logger.info(f"query_text长度: {len(query_text) if query_text else 0}")
        
        if not query_text:
            logger.warning("query_text为空，返回原始结果")
            return results
            
        # 获取query embedding
        logger.info("开始获取query embedding...")
        query_emb = await self._get_embedding_async(query_text, embedding_model_instance)
        logger.info(f"query_emb类型: {type(query_emb)}")
        logger.info(f"query_emb长度: {len(query_emb) if query_emb else 0}")
        
        if query_emb is None:
            logger.warning("query_emb获取失败，返回原始结果")
            return results
            
        # 获取doc embeddings
        logger.info("开始处理文档embeddings...")
        logger.info(f"min_text_length: {min_text_length}")
        
        # 收集有效的文本和结果
        valid_texts = []
        valid_results = []
        for i, r in enumerate(results):
            text = r.get('text', '')
            if text and len(text.strip()) >= min_text_length:
                valid_texts.append(text)
                valid_results.append(r)
                logger.info(f"结果 {i+1} 有效，text长度: {len(text)}")
            else:
                logger.info(f"结果 {i+1} 跳过 - text为空或长度不足")
        
        logger.info(f"文档embedding处理准备:")
        logger.info(f"  - 总结果数: {len(results)}")
        logger.info(f"  - 有效结果数: {len(valid_results)}")
        logger.info(f"  - 有效文本数: {len(valid_texts)}")
        
        if not valid_texts:
            logger.warning("没有有效的文本，返回原始结果")
            return results
        
        # 批量异步获取embeddings
        logger.info("开始批量异步获取文档embeddings...")
        doc_embs = await self._get_embeddings_batch_async(valid_texts, embedding_model_instance)
        
        # 过滤成功的embeddings
        successful_embeddings = []
        successful_results = []
        for i, (emb, result) in enumerate(zip(doc_embs, valid_results)):
            if emb is not None:
                successful_embeddings.append(emb)
                successful_results.append(result)
                logger.info(f"结果 {i+1} embedding获取成功，长度: {len(emb)}")
            else:
                logger.warning(f"结果 {i+1} embedding获取失败")
        
        logger.info(f"批量embedding处理完成:")
        logger.info(f"  - 成功获取embedding数: {len(successful_embeddings)}")
        logger.info(f"  - 成功结果数: {len(successful_results)}")
        
        if not successful_embeddings:
            logger.warning("没有成功获取的embeddings，返回原始结果")
            return results
        
        doc_texts = valid_texts[:len(successful_embeddings)]
        doc_embs = successful_embeddings
        valid_results = successful_results
        
        if not doc_embs:
            logger.warning("没有有效的文档embeddings，返回原始结果")
            return results
            
        logger.info("开始转换为numpy数组...")
        doc_embs = np.array(doc_embs)
        query_emb = np.array(query_emb)
        logger.info(f"doc_embs数组形状: {doc_embs.shape}")
        logger.info(f"query_emb数组形状: {query_emb.shape}")
        
        # MMR selection
        logger.info("开始MMR选择...")
        logger.info(f"use_cache_optimization: {use_cache_optimization}")
        logger.info(f"lambda_param: {lambda_param}")
        logger.info(f"select_n: {select_n}")
        logger.info(f"实际select_n: {min(select_n, len(doc_embs))}")
        
        if use_cache_optimization:
            logger.info("使用缓存优化的MMR选择...")
            selected_indices = self._mmr_selection_with_ordering(
                query_emb, doc_embs, 
                top_k=len(doc_embs), 
                lambda_param=lambda_param, 
                select_n=min(select_n, len(doc_embs)),
                zero_vector_protection=zero_vector_protection,
                zero_vector_strategy=zero_vector_strategy,
                min_norm_threshold=min_norm_threshold,
                output_order=output_order,
                reverse_order=reverse_order
            )
        else:
            logger.info("使用标准MMR选择...")
            selected_indices = self._mmr_selection(
                query_emb, doc_embs, 
                top_k=len(doc_embs), 
                lambda_param=lambda_param, 
                select_n=min(select_n, len(doc_embs)),
                zero_vector_protection=zero_vector_protection,
                zero_vector_strategy=zero_vector_strategy,
                min_norm_threshold=min_norm_threshold,
                output_order=output_order,
                reverse_order=reverse_order
            )
        
        logger.info(f"MMR选择完成，selected_indices: {selected_indices}")
        logger.info(f"selected_indices类型: {type(selected_indices)}")
        logger.info(f"selected_indices长度: {len(selected_indices) if selected_indices else 0}")
        
        final_results = [valid_results[idx] for idx in selected_indices]
        logger.info(f"最终结果数量: {len(final_results)}")
        logger.info("MMR处理完成，返回结果")
        return final_results

    def _extract_query_text(self, results: List[Dict[str, Any]], query_info: Dict[str, Any]) -> str:
        if isinstance(query_info, dict) and 'text' in query_info:
            return query_info['text']
        if results and 'query' in results[0]:
            return results[0]['query']
        for key in ['original_query', 'query', 'text']:
            if key in query_info:
                return query_info[key]
        return ''

    def _get_embedding(self, text: str, embedding_model_instance) -> Optional[List[float]]:
        """同步获取embedding方法，保持向后兼容"""
        # 调用异步版本
        return asyncio.run(self._get_embedding_async(text, embedding_model_instance))
    
    async def _get_embedding_async(self, text: str, embedding_model_instance) -> Optional[List[float]]:
        """
        获取文本的 embedding 向量
        
        Args:
            text: 输入文本
            embedding_model_instance: embedding 模型实例
            
        Returns:
            embedding 向量列表，失败时返回 None
        """
        logger.info(f"_get_embedding被调用:")
        logger.info(f"  - text长度: {len(text) if text else 0}")
        logger.info(f"  - text前100字符: {text[:100] if text else 'None'}...")
        logger.info(f"  - embedding_model_instance类型: {type(embedding_model_instance)}")
        logger.info(f"  - embedding_model_instance: {embedding_model_instance}")
        
        try:
            if not embedding_model_instance or not text:
                logger.warning(f"_get_embedding参数检查失败 - embedding_model_instance: {bool(embedding_model_instance)}, text: {bool(text)}")
                return None
            
            logger.info("调用embedding_model_instance.get_embedding...")
            result = embedding_model_instance.get_embedding(text)
            logger.info(f"get_embedding调用成功，返回类型: {type(result)}")
            logger.info(f"get_embedding返回长度: {len(result) if result else 0}")
            return result
            
        except Exception as e:
            logger.error(f"MMRProcessor embedding失败: {e}")
            logger.error(f"异常类型: {type(e)}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            return None
    
    async def _get_embeddings_batch_async(self, texts: List[str], embedding_model_instance) -> List[Optional[List[float]]]:
        """
        批量异步获取文本的embedding向量
        
        Args:
            texts: 输入文本列表
            embedding_model_instance: embedding模型实例
            
        Returns:
            embedding向量列表，失败时对应位置为None
        """
        logger.info(f"开始批量异步获取 {len(texts)} 个文本的embeddings...")
        
        # 创建异步任务列表
        tasks = []
        for i, text in enumerate(texts):
            task = self._get_embedding_async(text, embedding_model_instance)
            tasks.append(task)
            logger.info(f"创建任务 {i+1}: text长度={len(text)}")
        
        # 并发执行所有任务
        logger.info("开始并发执行所有embedding任务...")
        start_time = asyncio.get_event_loop().time()
        
        try:
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time
            
            logger.info(f"批量embedding获取完成，总耗时: {total_time:.2f}秒")
            
            # 处理结果，将异常转换为None
            processed_embeddings = []
            for i, result in enumerate(embeddings):
                if isinstance(result, Exception):
                    logger.error(f"任务 {i+1} 执行失败: {result}")
                    processed_embeddings.append(None)
                else:
                    processed_embeddings.append(result)
                    logger.info(f"任务 {i+1} 执行成功，embedding长度: {len(result) if result else 0}")
            
            return processed_embeddings
            
        except Exception as e:
            logger.error(f"批量embedding获取过程中发生错误: {e}")
            # 返回None列表
            return [None] * len(texts)

    def _safe_normalize(self, vectors: np.ndarray, strategy: str = 'return_zero', min_threshold: float = 1e-8) -> np.ndarray:
        """
        安全的向量归一化，处理零向量情况
        
        Args:
            vectors: 输入向量数组
            strategy: 零向量处理策略 ('return_zero', 'return_unit', 'skip', 'log_only')
            min_threshold: 最小范数阈值
            
        Returns:
            归一化后的向量数组
        """
        if vectors.size == 0:
            return vectors
            
        # 计算每个向量的L2范数
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        
        # 检查零向量
        zero_mask = norms < min_threshold
        
        if np.any(zero_mask):
            zero_count = np.sum(zero_mask)
            logger.warning(f"MMRProcessor: 发现 {zero_count} 个零向量 (范数 < {min_threshold})")
            
            if strategy == 'log_only':
                # 只记录日志，继续使用原始向量
                pass
            elif strategy == 'skip':
                # 跳过零向量，返回None或标记
                logger.warning("MMRProcessor: 零向量策略为'skip'，但当前实现不支持跳过，使用'return_zero'策略")
                strategy = 'return_zero'
            elif strategy == 'return_unit':
                # 返回单位向量
                unit_vector = np.ones_like(vectors[0]) / np.sqrt(vectors.shape[-1])
                vectors = vectors.copy()
                vectors[zero_mask.flatten()] = unit_vector
                logger.info(f"MMRProcessor: 将 {zero_count} 个零向量替换为单位向量")
            elif strategy == 'return_zero':
                # 返回零向量（默认行为）
                logger.info(f"MMRProcessor: 保持 {zero_count} 个零向量不变")
            else:
                logger.warning(f"MMRProcessor: 未知的零向量策略 '{strategy}'，使用'return_zero'")
                strategy = 'return_zero'
        
        # 执行归一化（零向量保持为零）
        safe_norms = np.where(zero_mask, 1.0, norms)
        return vectors / safe_norms

    def _mmr_selection(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int, lambda_param: float, select_n: int, zero_vector_protection: bool, zero_vector_strategy: str, min_norm_threshold: float, output_order: str = 'mmr_score', reverse_order: bool = False) -> List[int]:
        # 处理边界情况
        if len(doc_embeddings) == 0:
            return []
        
        # 获取零向量保护配置
        # schema = self.get_config_schema() # This line is removed as per the edit hint
        # zero_protection = schema['zero_vector_protection']['default'] # This line is removed as per the edit hint
        # zero_strategy = schema['zero_vector_strategy']['default'] # This line is removed as per the edit hint
        # min_threshold = schema['min_norm_threshold']['default'] # This line is removed as per the edit hint
        
        # 使用安全的归一化函数
        if zero_vector_protection:
            q_norm = self._safe_normalize(query_embedding, zero_vector_strategy, min_norm_threshold)
            docs_norm = self._safe_normalize(doc_embeddings, zero_vector_strategy, min_norm_threshold)
        else:
            # 原有的简单归一化（向后兼容）
            def normalize(x):
                norm = np.linalg.norm(x, axis=-1, keepdims=True)
                norm = np.where(norm == 0, 1, norm)
                return x / norm
            q_norm = normalize(query_embedding)
            docs_norm = normalize(doc_embeddings)
        
        sim_to_query = docs_norm.dot(q_norm)
        top_indices = np.argsort(sim_to_query)[-top_k:][::-1]
        selected = []
        
        for _ in range(min(select_n, len(top_indices))):
            mmr_scores = []
            for idx in top_indices:
                if idx in selected:
                    continue
                rel = sim_to_query[idx]
                if selected:
                    sim_to_selected = docs_norm[idx].dot(docs_norm[selected].T)
                    div = np.max(sim_to_selected)
                else:
                    div = 0
                mmr_score = lambda_param * rel - (1 - lambda_param) * div
                mmr_scores.append((idx, mmr_score))
            if not mmr_scores:
                break
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
        return selected

    def _mmr_selection_with_cache(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int, lambda_param: float, select_n: int, zero_vector_protection: bool, zero_vector_strategy: str, min_norm_threshold: float, output_order: str = 'mmr_score', reverse_order: bool = False) -> List[int]:
        """
        带缓存优化的MMR选择算法
        
        Args:
            query_embedding: 查询向量
            doc_embeddings: 文档向量数组
            top_k: 候选文档数量
            lambda_param: MMR参数
            select_n: 选择文档数量
            zero_vector_protection: 是否启用零向量保护
            zero_vector_strategy: 零向量处理策略
            min_norm_threshold: 最小范数阈值
            
        Returns:
            选择的文档索引列表
        """
        if len(doc_embeddings) == 0:
            return []
        
        # 使用安全的归一化函数
        if zero_vector_protection:
            q_norm = self._safe_normalize(query_embedding, zero_vector_strategy, min_norm_threshold)
            docs_norm = self._safe_normalize(doc_embeddings, zero_vector_strategy, min_norm_threshold)
        else:
            # 原有的简单归一化（向后兼容）
            def normalize(x):
                norm = np.linalg.norm(x, axis=-1, keepdims=True)
                norm = np.where(norm == 0, 1, norm)
                return x / norm
            q_norm = normalize(query_embedding)
            docs_norm = normalize(doc_embeddings)
        
        sim_to_query = docs_norm.dot(q_norm)
        top_indices = np.argsort(sim_to_query)[-top_k:][::-1]
        
        selected = []
        selected_set = set()  # 使用set提高查找效率
        
        # 缓存每个候选与已选文档的最大相似度
        max_sim_cache = np.zeros(len(top_indices))
        
        for _ in range(min(select_n, len(top_indices))):
            mmr_scores = []
            for i, idx in enumerate(top_indices):
                if idx in selected_set:  # 使用set查找，O(1)复杂度
                    continue
                rel = sim_to_query[idx]
                div = max_sim_cache[i]  # 使用缓存的最大相似度
                mmr_score = lambda_param * rel - (1 - lambda_param) * div
                mmr_scores.append((idx, mmr_score))
            
            if not mmr_scores:
                break
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # 更新缓存：新选文档与所有候选的相似度
            if selected:
                new_sims = docs_norm[best_idx].dot(docs_norm[top_indices].T)
                max_sim_cache = np.maximum(max_sim_cache, new_sims)
        
        return selected

    def _mmr_selection_with_ordering(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int, lambda_param: float, select_n: int, zero_vector_protection: bool, zero_vector_strategy: str, min_norm_threshold: float, output_order: str = 'mmr_score', reverse_order: bool = False) -> List[int]:
        """
        带输出顺序优化的MMR选择算法
        
        Args:
            query_embedding: 查询向量
            doc_embeddings: 文档向量数组
            top_k: 候选文档数量
            lambda_param: MMR参数
            select_n: 选择文档数量
            zero_vector_protection: 是否启用零向量保护
            zero_vector_strategy: 零向量处理策略
            min_norm_threshold: 最小范数阈值
            output_order: 输出排序方式 ('mmr_score', 'relevance', 'diversity')
            reverse_order: 是否逆序排列
            
        Returns:
            选择的文档索引列表
        """
        # 使用缓存优化的MMR选择
        selected = self._mmr_selection_with_cache(
            query_embedding, doc_embeddings, top_k, lambda_param, select_n,
            zero_vector_protection, zero_vector_strategy, min_norm_threshold,
            output_order=output_order, reverse_order=reverse_order
        )
        
        if not selected:
            return []
        
        # 根据指定方式重新排序
        if output_order == 'relevance':
            # 使用安全的归一化函数
            if zero_vector_protection:
                q_norm = self._safe_normalize(query_embedding, zero_vector_strategy, min_norm_threshold)
                docs_norm = self._safe_normalize(doc_embeddings, zero_vector_strategy, min_norm_threshold)
            else:
                def normalize(x):
                    norm = np.linalg.norm(x, axis=-1, keepdims=True)
                    norm = np.where(norm == 0, 1, norm)
                    return x / norm
                q_norm = normalize(query_embedding)
                docs_norm = normalize(doc_embeddings)
            
            # 按相关性排序
            sim_to_query = docs_norm.dot(q_norm)
            selected_sims = sim_to_query[selected]
            sorted_indices = np.argsort(selected_sims)
            if not reverse_order:
                sorted_indices = sorted_indices[::-1]  # 降序
            return [selected[i] for i in sorted_indices]
            
        elif output_order == 'diversity':
            # 使用安全的归一化函数
            if zero_vector_protection:
                docs_norm = self._safe_normalize(doc_embeddings, zero_vector_strategy, min_norm_threshold)
            else:
                def normalize(x):
                    norm = np.linalg.norm(x, axis=-1, keepdims=True)
                    norm = np.where(norm == 0, 1, norm)
                    return x / norm
                docs_norm = normalize(doc_embeddings)
            
            # 按多样性排序（与已选文档的平均相似度）
            diversity_scores = []
            for idx in selected:
                sim_to_others = docs_norm[idx].dot(docs_norm[selected].T)
                diversity = 1 - np.mean(sim_to_others)
                diversity_scores.append(diversity)
            sorted_indices = np.argsort(diversity_scores)
            if not reverse_order:
                sorted_indices = sorted_indices[::-1]  # 降序
            return [selected[i] for i in sorted_indices]
            
        else:
            # 保持MMR选择顺序
            if reverse_order:
                return selected[::-1]
            return selected

    def get_name(self) -> str:
        return 'mmr'

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式 - 从配置文件读取默认值"""
        config = get_processor_config('mmr')
        return {
            'lambda_param': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('lambda_param', 0.7),
                'description': 'MMR lambda参数，控制相关性和多样性的平衡 (0=纯多样性, 1=纯相关性)'
            },
            'select_n': {
                'type': 'integer',
                'min': 1,
                'default': config.get('select_n', 10),
                'description': '最终选择的文档数量'
            },
            'enable_mmr': {
                'type': 'boolean',
                'default': config.get('enable_mmr', True),
                'description': '是否启用MMR处理'
            },
            'min_text_length': {
                'type': 'integer',
                'min': 1,
                'default': config.get('min_text_length', 10),
                'description': '处理的最小文本长度'
            },
            # 新增优化参数
            'cache_max_similarity': {
                'type': 'boolean',
                'default': config.get('cache_max_similarity', True),
                'description': '是否缓存最大相似度值以提高性能'
            },
            'relevance_threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('relevance_threshold', 0.3),
                'description': '相关性阈值，低于此值的结果被过滤'
            },
            'enable_relevance_filter': {
                'type': 'boolean',
                'default': config.get('enable_relevance_filter', False),
                'description': '是否启用相关性过滤'
            },
            'output_order': {
                'type': 'string',
                'options': ['mmr_score', 'relevance', 'diversity'],
                'default': config.get('output_order', 'mmr_score'),
                'description': '输出排序方式'
            },
            'reverse_order': {
                'type': 'boolean',
                'default': config.get('reverse_order', False),
                'description': '是否逆序排列结果'
            },
            'fallback_strategy': {
                'type': 'string',
                'options': ['skip', 'random', 'original'],
                'default': config.get('fallback_strategy', 'original'),
                'description': 'embedding不可用时的降级策略'
            },
            'embedding_timeout': {
                'type': 'float',
                'min': 0.1,
                'max': 60.0,
                'default': config.get('embedding_timeout', 5.0),
                'description': 'embedding获取超时时间（秒）'
            },
            'enable_vectorized_mmr': {
                'type': 'boolean',
                'default': config.get('enable_vectorized_mmr', True),
                'description': '是否启用向量化MMR算法'
            },
            'batch_size': {
                'type': 'integer',
                'min': 1,
                'max': 128,
                'default': config.get('batch_size', 32),
                'description': '批处理大小'
            },
            'use_parallel': {
                'type': 'boolean',
                'default': config.get('use_parallel', False),
                'description': '是否使用并行计算'
            },
            'max_iterations': {
                'type': 'integer',
                'min': 10,
                'max': 1000,
                'default': config.get('max_iterations', 100),
                'description': '最大迭代次数，防止无限循环'
            },
            'early_stopping': {
                'type': 'boolean',
                'default': config.get('early_stopping', True),
                'description': '是否启用早停机制'
            },
            'diversity_boost': {
                'type': 'float',
                'min': 0.1,
                'max': 5.0,
                'default': config.get('diversity_boost', 1.0),
                'description': '多样性提升因子'
            },
            # 零向量保护参数
            'zero_vector_protection': {
                'type': 'boolean',
                'default': config.get('zero_vector_protection', True),
                'description': '是否启用零向量保护'
            },
            'zero_vector_strategy': {
                'type': 'string',
                'options': ['return_zero', 'return_unit', 'skip', 'log_only'],
                'default': config.get('zero_vector_strategy', 'return_zero'),
                'description': '零向量处理策略'
            },
            'min_norm_threshold': {
                'type': 'float',
                'min': 1e-12,
                'max': 1e-6,
                'default': config.get('min_norm_threshold', 1e-8),
                'description': '最小范数阈值，低于此值视为零向量'
            },
            'use_cache_optimization': {
                'type': 'boolean',
                'default': config.get('use_cache_optimization', True),
                'description': '是否使用缓存优化的MMR算法'
            }
        }