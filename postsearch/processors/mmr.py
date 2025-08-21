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
        
        # 🔴 第一阶段：Score标准化处理（解决数值空间不一致问题）
        enable_normalization = kwargs.get('enable_score_normalization', schema['enable_score_normalization']['default'])
        normalization_method = kwargs.get('normalization_method', schema['normalization_method']['default'])
        
        if enable_normalization:
            logger.info(f"开始Score标准化处理，方法: {normalization_method}")
            results = self._normalize_score_for_filtering(results, normalization_method)
            logger.info(f"Score标准化完成，添加normalized_score字段")
        else:
            logger.info("Score标准化已禁用，跳过标准化处理")
        
        # 🔴 第二阶段：相关性过滤（使用标准化后的检索分数）
        relevance_threshold = kwargs.get('relevance_threshold', schema['relevance_threshold']['default'])
        enable_filter = kwargs.get('enable_relevance_filter', schema['enable_relevance_filter']['default'])
        
        if enable_filter:
            logger.info(f"启用相关性过滤，阈值: {relevance_threshold}")
            results = self._apply_relevance_filter(results, enable_filter, relevance_threshold)
            if not results:
                logger.warning("相关性过滤后无结果，返回空列表")
                return []
        else:
            logger.info("相关性过滤已禁用")
        
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
            logger.warning("query_text为空，应用简单选择逻辑返回结果")
            # 即使没有query_text，也要遵守select_n参数
            select_n = kwargs.get('select_n', schema['select_n']['default'])
            if len(results) > select_n:
                logger.info(f"无query情况下，限制结果数量: {len(results)} -> {select_n}")
                return results[:select_n]
            return results
            
        # 获取query embedding
        logger.info("开始获取query embedding...")
        query_emb = await self._get_embedding_async(query_text, embedding_model_instance)
        logger.info(f"query_emb类型: {type(query_emb)}")
        logger.info(f"query_emb长度: {len(query_emb) if query_emb else 0}")
        
        if query_emb is None:
            logger.warning("query_emb获取失败，返回原始结果")
            return results
        
        # 🔴 第三阶段：Top-N预选和动态λ调整
        max_candidates = kwargs.get('max_candidates', schema['max_candidates']['default'])
        dynamic_lambda = kwargs.get('dynamic_lambda', schema['dynamic_lambda']['default'])
        use_optimized_mmr = kwargs.get('use_optimized_mmr', schema['use_optimized_mmr']['default'])
        
        # Top-N预选：限制候选文档数量，避免内存开销过大
        if len(results) > max_candidates:
            logger.info(f"候选文档数量({len(results)})超过限制({max_candidates})，进行Top-N预选")
            # 按标准化后的score排序，选择前N个
            sorted_results = sorted(results, key=lambda x: x.get('normalized_score', x.get('score', 0)), reverse=True)
            results = sorted_results[:max_candidates]
            logger.info(f"Top-N预选完成，保留前{max_candidates}个文档")
        
        # 动态λ调整：根据query长度自动调整MMR参数
        if dynamic_lambda:
            original_lambda = lambda_param
            lambda_param = self._calculate_dynamic_lambda(query_text)
            if lambda_param != original_lambda:
                logger.info(f"动态λ调整: {original_lambda:.3f} -> {lambda_param:.3f} (query长度: {len(query_text)})")
        else:
            logger.info(f"使用固定λ参数: {lambda_param:.3f}")
            
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
        logger.info(f"use_optimized_mmr: {use_optimized_mmr}")
        logger.info(f"lambda_param: {lambda_param}")
        logger.info(f"select_n: {select_n}")
        logger.info(f"实际select_n: {min(select_n, len(doc_embs))}")
        
        # 🔴 第三阶段：选择优化的MMR算法
        if use_optimized_mmr:
            logger.info("使用优化的MMR选择算法（避免重复计算）...")
            selected_indices = self._mmr_selection_optimized(
                results, doc_embs, lambda_param, min(select_n, len(doc_embs))
            )
        elif use_cache_optimization:
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
        
        # 🔴 第四阶段：回退机制和稳定性增强
        min_results_threshold = kwargs.get('min_results_threshold', schema['min_results_threshold']['default'])
        fallback_strategy = kwargs.get('fallback_strategy', schema['fallback_strategy']['default'])
        fallback_levels = kwargs.get('fallback_levels', schema['fallback_levels']['default'])
        enable_fallback_monitoring = kwargs.get('enable_fallback_monitoring', schema['enable_fallback_monitoring']['default'])
        
        # 🔴 关键修复：select_n 优先级高于 min_results_threshold
        # 如果用户明确设置了 select_n，回退机制不应该超过这个限制
        user_select_n = kwargs.get('select_n')
        if user_select_n is not None and min_results_threshold > user_select_n:
            logger.info(f"用户设置select_n={user_select_n}，调整min_results_threshold: {min_results_threshold} -> {user_select_n}")
            min_results_threshold = user_select_n
        
        # 检查是否需要回退
        if len(final_results) < min_results_threshold:
            logger.warning(f"MMR结果数量({len(final_results)})低于最小阈值({min_results_threshold})，触发回退机制")
            
            # 记录回退前的状态
            original_count = len(final_results)
            
            # 应用回退机制
            final_results = self._ensure_minimum_results(
                final_results, min_results_threshold, fallback_strategy, fallback_levels
            )
            
            # 监控回退使用情况
            if enable_fallback_monitoring:
                self._monitor_fallback_usage(
                    original_count, len(final_results), fallback_strategy, 0
                )
            
            logger.info(f"回退机制完成，最终结果数量: {len(final_results)}")
        else:
            logger.info(f"结果数量({len(final_results)})满足最小阈值({min_results_threshold})，无需回退")
        
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
    
    # 🔴 第一阶段：Score标准化基础设施方法
    
    def _normalize_score_for_filtering(self, results: List[Dict], normalization_method: str = 'log_minmax') -> List[Dict]:
        """
        根据标准化方法处理score，确保数值空间一致
        
        Args:
            results: 检索结果列表
            normalization_method: 标准化方法
            
        Returns:
            标准化后的结果列表，包含normalized_score字段
        """
        if not results:
            return results
        
        # 自动检测搜索引擎类型
        search_type = self._detect_search_type(results)
        logger.info(f"检测到搜索引擎类型: {search_type}")
        
        # 根据搜索引擎类型进行标准化
        if search_type == 'dense':
            return self._normalize_dense_scores(results)
        elif search_type == 'hybrid':
            return self._normalize_hybrid_scores(results)
        elif search_type == 'sparse':
            return self._normalize_sparse_scores(results)
        else:
            # 未知类型，使用通用标准化
            return self._normalize_generic_scores(results)
    
    def _detect_search_type(self, results: List[Dict]) -> str:
        """自动检测搜索引擎类型"""
        if not results:
            return 'unknown'
        
        # 提取所有有效的score值
        scores = []
        for result in results:
            if 'score' in result and result['score'] is not None:
                scores.append(result['score'])
        
        if not scores:
            return 'unknown'
        
        # 分析score特征判断类型
        min_score = min(scores)
        max_score = max(scores)
        
        # 如果score在[0,1]范围内，可能是dense search
        if 0 <= min_score <= max_score <= 1:
            return 'dense'
        # 如果score在[-1,1]范围内，可能是cosine相似度
        elif -1 <= min_score <= max_score <= 1:
            return 'dense'
        # 如果score有极值（负值或超出正常范围），可能是hybrid search
        elif min_score < -100 or max_score > 100:
            return 'hybrid'
        # 如果score都是正值但不在[0,1]范围，可能是sparse search (BM25)
        elif min_score >= 0 and max_score > 1:
            return 'sparse'
        # 其他情况
        else:
            return 'unknown'
    
    def _normalize_dense_scores(self, results: List[Dict]) -> List[Dict]:
        """标准化dense search的score"""
        if not results:
            return results
        
        # 从配置获取默认值
        default_score = get_processor_config('mmr').get('default_normalized_score', 0.5)
        
        # Dense search的score通常已经在[0,1]范围内，但需要验证和微调
        for result in results:
            if 'score' in result and result['score'] is not None:
                score = result['score']
                # 确保score在[0,1]范围内
                normalized_score = max(0.0, min(1.0, score))
                result['normalized_score'] = normalized_score
                result['original_score'] = score
            else:
                # 如果没有score字段，使用配置的默认值
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
        
        return results
    
    def _normalize_hybrid_scores(self, results: List[Dict]) -> List[Dict]:
        """标准化hybrid search的score（适用于极端异常值）"""
        if not results:
            return results
        
        # 从配置获取默认值
        default_score = get_processor_config('mmr').get('default_normalized_score', 0.5)
        
        # 提取所有有效的score值
        scores = []
        for result in results:
            if 'score' in result and result['score'] is not None:
                scores.append(result['score'])
        
        if not scores:
            # 如果没有有效score，使用配置的默认值
            for result in results:
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
            return results
        
        # 使用对数变换 + Min-Max标准化处理极端异常值
        normalized_scores = self._apply_log_minmax_normalization(scores)
        
        # 将标准化后的score赋值给结果
        score_index = 0
        for result in results:
            if 'score' in result and result['score'] is not None:
                result['normalized_score'] = normalized_scores[score_index]
                result['original_score'] = result['score']
                score_index += 1
            else:
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
        
        return results
    
    def _normalize_sparse_scores(self, results: List[Dict]) -> List[Dict]:
        """标准化sparse search的score"""
        if not results:
            return results
        
        # 从配置获取默认值
        default_score = get_processor_config('mmr').get('default_normalized_score', 0.5)
        
        # 提取所有有效的score值
        scores = []
        for result in results:
            if 'score' in result and result['score'] is not None:
                scores.append(result['score'])
        
        if not scores:
            # 如果没有有效score，使用配置的默认值
            for result in results:
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
            return results
        
        # Sparse search通常使用Min-Max标准化
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score:
            for result in results:
                if 'score' in result and result['score'] is not None:
                    # Min-Max标准化到[0,1]
                    normalized_score = (result['score'] - min_score) / (max_score - min_score)
                    result['normalized_score'] = normalized_score
                    result['original_score'] = result['score']
                else:
                    result['normalized_score'] = default_score
                    result['original_score'] = 0.0
        else:
            # 所有score相同的情况
            for result in results:
                if 'score' in result and result['score'] is not None:
                    result['normalized_score'] = default_score
                    result['original_score'] = result['score']
                else:
                    result['normalized_score'] = default_score
                    result['original_score'] = 0.0
        
        return results
    
    def _normalize_generic_scores(self, results: List[Dict]) -> List[Dict]:
        """通用标准化方法"""
        if not results:
            return results
        
        # 从配置获取默认值
        default_score = get_processor_config('mmr').get('default_normalized_score', 0.5)
        
        # 提取所有有效的score值
        scores = []
        for result in results:
            if 'score' in result and result['score'] is not None:
                scores.append(result['score'])
        
        if not scores:
            # 如果没有有效score，使用配置的默认值
            for result in results:
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
            return results
        
        # 使用对数变换 + Min-Max标准化作为通用方法
        normalized_scores = self._apply_log_minmax_normalization(scores)
        
        # 将标准化后的score赋值给结果
        score_index = 0
        for result in results:
            if 'score' in result and result['score'] is not None:
                result['normalized_score'] = normalized_scores[score_index]
                result['original_score'] = result['score']
                score_index += 1
            else:
                result['normalized_score'] = default_score
                result['original_score'] = 0.0
        
        return results
    
    # 🔴 第二阶段：相关性过滤优化
    
    def _apply_relevance_filter(self, results: List[Dict], enable_filter: bool, relevance_threshold: float) -> List[Dict]:
        """
        使用标准化后的检索分数进行相关性过滤
        
        Args:
            results: 检索结果列表（已标准化）
            enable_filter: 是否启用过滤
            relevance_threshold: 相关性阈值（基于标准化后的分数）
            
        Returns:
            过滤后的结果列表
        """
        if not enable_filter:
            logger.info("相关性过滤已禁用，跳过过滤处理")
            return results
            
        if not results:
            logger.warning("空结果列表，无法进行相关性过滤")
            return results
        
        # 统计过滤前的结果
        original_count = len(results)
        logger.info(f"开始相关性过滤，原始结果数: {original_count}，阈值: {relevance_threshold}")
        
        # 使用标准化后的score进行过滤
        # 优先使用normalized_score，如果没有则使用原始score
        filtered_results = []
        score_stats = []
        
        for i, result in enumerate(results):
            # 优先使用标准化后的分数
            score_to_check = result.get('normalized_score', result.get('score', 0))
            score_stats.append(score_to_check)
            
            if score_to_check >= relevance_threshold:
                filtered_results.append(result)
                logger.debug(f"结果 {i+1} 通过过滤，score: {score_to_check:.4f}")
            else:
                logger.debug(f"结果 {i+1} 被过滤，score: {score_to_check:.4f} < {relevance_threshold}")
        
        # 过滤后的统计信息
        filtered_count = len(filtered_results)
        filter_ratio = filtered_count / original_count if original_count > 0 else 0
        
        # 分析score分布
        if score_stats:
            min_score = min(score_stats)
            max_score = max(score_stats)
            avg_score = sum(score_stats) / len(score_stats)
            
            logger.info(f"Score分布 - 范围: [{min_score:.4f}, {max_score:.4f}], 均值: {avg_score:.4f}")
        
        # 根据过滤效果给出不同级别的日志
        if filtered_count == 0:
            logger.error(f"相关性过滤后无结果，阈值可能过高: {relevance_threshold}")
            logger.error(f"建议降低阈值或检查score标准化是否正确")
        elif filter_ratio < 0.3:
            logger.warning(f"相关性过滤较严格: {original_count} -> {filtered_count} (保留 {filter_ratio:.1%})")
            logger.warning(f"建议考虑降低阈值从 {relevance_threshold} 到 {relevance_threshold * 0.8:.2f}")
        elif filter_ratio < 0.7:
            logger.info(f"相关性过滤适中: {original_count} -> {filtered_count} (保留 {filter_ratio:.1%})")
        else:
            logger.info(f"相关性过滤宽松: {original_count} -> {filtered_count} (保留 {filter_ratio:.1%})")
            if filter_ratio > 0.9:
                logger.info(f"建议考虑提高阈值从 {relevance_threshold} 到 {relevance_threshold * 1.2:.2f}")
        
        logger.info(f"相关性过滤完成: {original_count} -> {filtered_count}")
        return filtered_results
    
    # 🔴 第三阶段：MMR算法核心优化
    
    def _mmr_selection_optimized(self, results: List[Dict], doc_embs: np.ndarray, 
                                lambda_param: float, select_n: int) -> List[int]:
        """
        优化的MMR选择算法
        
        Args:
            results: 检索结果列表（包含score字段）
            doc_embs: 文档embedding矩阵
            lambda_param: MMR lambda参数
            select_n: 选择数量
            
        Returns:
            选择的文档索引列表
        """
        if len(results) == 0:
            return []
        
        # 1. 直接使用检索分数，避免重复计算
        relevance_scores = np.array([r.get('normalized_score', r.get('score', 0)) for r in results])
        
        # 2. 预计算文档间相似度矩阵（仅计算一次）
        # 注意：必须先对doc_embs做L2归一化，否则不是标准cosine相似度
        doc_embs_normalized = self._safe_normalize(doc_embs, 'return_unit', 1e-8)
        doc_doc_sim = np.dot(doc_embs_normalized, doc_embs_normalized.T)
        
        # 3. MMR选择逻辑
        selected = []
        for _ in range(min(select_n, len(results))):
            mmr_scores = []
            for i, result in enumerate(results):
                if i in selected:
                    continue
                
                # 使用检索分数作为相关性
                rel = relevance_scores[i]
                
                # 计算多样性（与已选文档的最大相似度）
                if selected:
                    div = max(doc_doc_sim[i, selected])
                else:
                    div = 0
                
                mmr_score = lambda_param * rel - (1 - lambda_param) * div
                mmr_scores.append((i, mmr_score))
            
            if not mmr_scores:
                break
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
        
        return selected
    
    # 🔴 第四阶段：回退机制和稳定性增强
    
    def _ensure_minimum_results(self, results: List[Dict], min_threshold: int, 
                               fallback_strategy: str, fallback_levels: int = None) -> List[Dict]:
        """
        确保返回最小数量的结果，实现多层回退机制
        
        Args:
            results: 当前结果列表
            min_threshold: 最小结果数量阈值
            fallback_strategy: 回退策略
            fallback_levels: 多级回退的级别数量
            
        Returns:
            确保满足最小数量的结果列表
        """
        # 从配置获取默认值
        if fallback_levels is None:
            fallback_levels = get_processor_config('mmr').get('fallback_levels', 3)
        
        original_count = len(results)
        
        if original_count >= min_threshold:
            logger.info(f"结果数量({original_count})已满足最小阈值({min_threshold})，无需回退")
            return results
        
        logger.warning(f"结果数量({original_count})低于最小阈值({min_threshold})，触发回退机制")
        logger.info(f"使用回退策略: {fallback_strategy}")
        
        # 执行回退策略
        if fallback_strategy == 'multi_level':
            fallback_results = self._apply_multi_level_fallback(results, min_threshold, fallback_levels)
        elif fallback_strategy == 'score_sort':
            fallback_results = self._apply_score_sort_fallback(results, min_threshold)
        elif fallback_strategy == 'random':
            fallback_results = self._apply_random_fallback(results, min_threshold)
        elif fallback_strategy == 'original':
            fallback_results = self._apply_original_fallback(results, min_threshold)
        else:
            logger.warning(f"未知的回退策略: {fallback_strategy}，使用默认策略")
            fallback_results = self._apply_multi_level_fallback(results, min_threshold, fallback_levels)
        
        # 记录回退监控统计
        final_count = len(fallback_results)
        self._monitor_fallback_usage(original_count, final_count, fallback_strategy)
        
        return fallback_results
    
    def _apply_multi_level_fallback(self, results: List[Dict], min_threshold: int, 
                                   fallback_levels: int) -> List[Dict]:
        """
        多级回退策略：逐步放宽条件，确保返回足够的结果
        
        Args:
            results: 当前结果列表
            min_threshold: 最小结果数量阈值
            fallback_levels: 回退级别数量
            
        Returns:
            回退后的结果列表
        """
        logger.info(f"开始多级回退，目标结果数: {min_threshold}，回退级别: {fallback_levels}")
        
        # 第一级：尝试降低相关性阈值
        if len(results) < min_threshold:
            logger.info("第一级回退：降低相关性阈值")
            # 这里可以重新调用相关性过滤，使用更低的阈值
            # 暂时使用score排序作为第一级回退
            fallback_results = self._apply_score_sort_fallback(results, min_threshold)
            if len(fallback_results) >= min_threshold:
                logger.info(f"第一级回退成功，获得 {len(fallback_results)} 个结果")
                return fallback_results
        
        # 第二级：使用原始score排序，不考虑MMR多样性
        if len(results) < min_threshold:
            logger.info("第二级回退：使用原始score排序")
            fallback_results = self._apply_score_sort_fallback(results, min_threshold)
            if len(fallback_results) >= min_threshold:
                logger.info(f"第二级回退成功，获得 {len(fallback_results)} 个结果")
                return fallback_results
        
        # 第三级：随机选择，确保返回结果
        if len(results) < min_threshold:
            logger.info("第三级回退：随机选择")
            fallback_results = self._apply_random_fallback(results, min_threshold)
            if len(fallback_results) >= min_threshold:
                logger.info(f"第三级回退成功，获得 {len(fallback_results)} 个结果")
                return fallback_results
        
        # 最终回退：如果结果数量仍然不足，通过重复结果来达到目标数量
        if len(results) < min_threshold:
            logger.warning(f"所有回退策略都无法达到目标数量，通过重复结果来达到目标")
            
            # 处理空结果的情况
            if len(results) == 0:
                logger.warning("原始结果为空，无法通过重复来达到目标数量")
                return []
            
            fallback_results = results.copy()
            while len(fallback_results) < min_threshold:
                # 循环添加结果，直到达到目标数量
                for result in results:
                    if len(fallback_results) >= min_threshold:
                        break
                    # 创建结果的副本，避免修改原始数据
                    result_copy = result.copy()
                    result_copy['_fallback_duplicate'] = True  # 标记为回退重复
                    fallback_results.append(result_copy)
            
            logger.info(f"最终回退完成，通过重复结果达到目标数量: {len(fallback_results)}")
            return fallback_results
        
        return results
    
    def _apply_score_sort_fallback(self, results: List[Dict], min_threshold: int) -> List[Dict]:
        """
        按score排序的回退策略
        
        Args:
            results: 当前结果列表
            min_threshold: 最小结果数量阈值
            
        Returns:
            按score排序的结果列表，如果数量不足则通过重复达到目标数量
        """
        if not results:
            return results
        
        # 优先使用normalized_score，如果没有则使用原始score
        sorted_results = sorted(results, 
                              key=lambda x: x.get('normalized_score', x.get('score', 0)), 
                              reverse=True)
        
        # 如果结果数量已经满足要求，直接返回前N个
        if len(sorted_results) >= min_threshold:
            fallback_results = sorted_results[:min_threshold]
            logger.info(f"Score排序回退：从 {len(results)} 个结果中选择前 {len(fallback_results)} 个")
            return fallback_results
        
        # 如果结果数量不足，通过重复来达到目标数量
        logger.info(f"Score排序回退：结果数量不足({len(sorted_results)} < {min_threshold})，通过重复达到目标数量")
        fallback_results = sorted_results.copy()
        
        while len(fallback_results) < min_threshold:
            for result in sorted_results:
                if len(fallback_results) >= min_threshold:
                    break
                # 创建结果的副本，避免修改原始数据
                result_copy = result.copy()
                result_copy['_fallback_duplicate'] = True  # 标记为回退重复
                fallback_results.append(result_copy)
        
        logger.info(f"Score排序回退完成：最终结果数量 {len(fallback_results)}")
        return fallback_results
    
    def _apply_random_fallback(self, results: List[Dict], min_threshold: int) -> List[Dict]:
        """
        随机选择的回退策略
        
        Args:
            results: 当前结果列表
            min_threshold: 最小结果数量阈值
            
        Returns:
            随机选择的结果列表，如果数量不足则通过重复达到目标数量
        """
        if not results:
            return results
        
        import random
        
        # 如果结果数量已经满足要求，随机选择N个
        if len(results) >= min_threshold:
            fallback_results = random.sample(results, min_threshold)
            logger.info(f"随机选择回退：从 {len(results)} 个结果中随机选择 {len(fallback_results)} 个")
            return fallback_results
        
        # 如果结果数量不足，通过重复来达到目标数量
        logger.info(f"随机选择回退：结果数量不足({len(results)} < {min_threshold})，通过重复达到目标数量")
        fallback_results = results.copy()
        
        while len(fallback_results) < min_threshold:
            for result in results:
                if len(fallback_results) >= min_threshold:
                    break
                # 创建结果的副本，避免修改原始数据
                result_copy = result.copy()
                result_copy['_fallback_duplicate'] = True  # 标记为回退重复
                fallback_results.append(result_copy)
        
        logger.info(f"随机选择回退完成：最终结果数量 {len(fallback_results)}")
        return fallback_results
    
    def _apply_original_fallback(self, results: List[Dict], min_threshold: int) -> List[Dict]:
        """
        保持原始顺序的回退策略
        
        Args:
            results: 当前结果列表
            min_threshold: 最小结果数量阈值
            
        Returns:
            保持原始顺序的结果列表，如果数量不足则通过重复达到目标数量
        """
        if not results:
            return results
        
        # 如果结果数量已经满足要求，直接返回前N个
        if len(results) >= min_threshold:
            fallback_results = results[:min_threshold]
            logger.info(f"原始顺序回退：保持原始顺序，返回前 {len(fallback_results)} 个结果")
            return fallback_results
        
        # 如果结果数量不足，通过重复来达到目标数量
        logger.info(f"原始顺序回退：结果数量不足({len(results)} < {min_threshold})，通过重复达到目标数量")
        fallback_results = results.copy()
        
        while len(fallback_results) < min_threshold:
            for result in results:
                if len(fallback_results) >= min_threshold:
                    break
                # 创建结果的副本，避免修改原始数据
                result_copy = result.copy()
                result_copy['_fallback_duplicate'] = True  # 标记为回退重复
                fallback_results.append(result_copy)
        
        logger.info(f"原始顺序回退完成：最终结果数量 {len(fallback_results)}")
        return fallback_results
    
    def _monitor_fallback_usage(self, original_count: int, final_count: int, 
                               fallback_strategy: str, fallback_level: int = 0):
        """
        监控回退机制的使用情况
        
        Args:
            original_count: 原始结果数量
            final_count: 最终结果数量
            fallback_strategy: 使用的回退策略
            fallback_level: 回退级别
        """
        if not hasattr(self, '_fallback_stats'):
            self._fallback_stats = {
                'total_fallbacks': 0,
                'strategy_usage': {},
                'level_usage': {},
                'count_improvement': []
            }
        
        # 更新统计信息
        self._fallback_stats['total_fallbacks'] += 1
        
        # 策略使用统计
        if fallback_strategy not in self._fallback_stats['strategy_usage']:
            self._fallback_stats['strategy_usage'][fallback_strategy] = 0
        self._fallback_stats['strategy_usage'][fallback_strategy] += 1
        
        # 级别使用统计
        if fallback_level not in self._fallback_stats['level_usage']:
            self._fallback_stats['level_usage'][fallback_level] = 0
        self._fallback_stats['level_usage'][fallback_level] += 1
        
        # 数量改善统计
        improvement = final_count - original_count
        self._fallback_stats['count_improvement'].append(improvement)
        
        # 记录监控信息
        logger.info(f"回退机制监控 - 策略: {fallback_strategy}, 级别: {fallback_level}")
        logger.info(f"回退机制监控 - 原始数量: {original_count}, 最终数量: {final_count}, 改善: {improvement}")
        
        # 确保统计信息被正确记录
        logger.debug(f"当前回退统计: {self._fallback_stats}")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """
        获取回退机制的统计信息
        
        Returns:
            回退机制统计信息字典
        """
        if not hasattr(self, '_fallback_stats'):
            return {
                'total_fallbacks': 0,
                'strategy_usage': {},
                'level_usage': {},
                'count_improvement': []
            }
        
        stats = self._fallback_stats.copy()
        
        # 计算平均值
        if stats['count_improvement']:
            stats['avg_improvement'] = sum(stats['count_improvement']) / len(stats['count_improvement'])
        else:
            stats['avg_improvement'] = 0
        
        return stats
    
    def _calculate_dynamic_lambda(self, query_text: str, base_lambda: float = None) -> float:
        """
        根据query长度动态调整λ参数
        
        Args:
            query_text: 查询文本
            base_lambda: 基础λ值
            
        Returns:
            调整后的λ值
        """
        # 从配置获取默认值
        if base_lambda is None:
            base_lambda = get_processor_config('mmr').get('base_lambda', 0.5)
        
        # 从配置获取阈值和调整值
        short_query_threshold = get_processor_config('mmr').get('short_query_threshold', 10)
        medium_query_threshold = get_processor_config('mmr').get('medium_query_threshold', 30)
        short_adjustment = get_processor_config('mmr').get('short_adjustment', 0.3)
        long_adjustment = get_processor_config('mmr').get('long_adjustment', 0.2)
        max_lambda = get_processor_config('mmr').get('max_lambda', 0.9)
        min_lambda = get_processor_config('mmr').get('min_lambda', 0.3)
        
        if not query_text:
            return base_lambda
        
        # 计算query长度（去除空格）
        query_length = len(query_text.strip())
        
        # 根据长度调整λ，使用配置值，不再hardcode
        if query_length <= short_query_threshold:
            # 短query（单个关键词）更看重相关性
            return min(max_lambda, base_lambda + short_adjustment)
        elif query_length <= medium_query_threshold:
            # 中等长度query，保持平衡
            return base_lambda
        else:
            # 长query（叙述型）适当降低λ，让多样性发挥作用
            return max(min_lambda, base_lambda - long_adjustment)
    
    def _apply_log_minmax_normalization(self, scores: List[float]) -> List[float]:
        """对数变换 + Min-Max标准化（适用于极端异常值）"""
        if not scores:
            return []
        
        # 处理极端异常值：对数变换
        log_scores = []
        for score in scores:
            if score > 0:
                # 正值取对数
                log_scores.append(np.log(score + 1))  # +1避免log(0)
            elif score < 0:
                # 负值取绝对值的对数，然后加负号
                log_scores.append(-np.log(abs(score) + 1))
            else:
                # 零值保持不变
                log_scores.append(0.0)
        
        # Min-Max标准化到[0,1]
        min_log_score = min(log_scores)
        max_log_score = max(log_scores)
        
        if max_log_score > min_log_score:
            normalized_scores = []
            for log_score in log_scores:
                normalized_score = (log_score - min_log_score) / (max_log_score - min_log_score)
                normalized_scores.append(normalized_score)
            return normalized_scores
        else:
            # 所有score相同的情况
            return [0.5] * len(scores)

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式 - 从配置文件读取默认值"""
        config = get_processor_config('mmr')
        return {
            # 核心MMR参数（实际使用）
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
            
            # 相关性过滤参数（实际使用）
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
            
            # Score标准化参数（实际使用）
            'enable_score_normalization': {
                'type': 'boolean',
                'default': config.get('enable_score_normalization', True),
                'description': '是否启用score标准化，解决不同搜索引擎的数值空间不一致问题'
            },
            
            # MMR算法优化参数（实际使用）
            'max_candidates': {
                'type': 'integer',
                'min': 50,
                'max': 1000,
                'default': config.get('max_candidates', 200),
                'description': '进入MMR的最大候选文档数量，避免内存开销过大'
            },
            'dynamic_lambda': {
                'type': 'boolean',
                'default': config.get('dynamic_lambda', True),
                'description': '是否启用动态λ调整，根据query长度自动调整MMR参数'
            },
            'use_optimized_mmr': {
                'type': 'boolean',
                'default': config.get('use_optimized_mmr', True),
                'description': '是否使用优化的MMR算法（避免重复计算）'
            },
            
            # 零向量保护参数（实际使用）
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
            
            # 缓存优化参数（实际使用）
            'use_cache_optimization': {
                'type': 'boolean',
                'default': config.get('use_cache_optimization', True),
                'description': '是否使用缓存优化的MMR算法'
            },
            
            # 输出排序参数（实际使用）
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
            
            # 标准化方法参数（实际使用）
            'normalization_method': {
                'type': 'string',
                'options': ['log_minmax', 'minmax', 'zscore', 'auto'],
                'default': config.get('normalization_method', 'log_minmax'),
                'description': 'score标准化方法，log_minmax适用于极端异常值'
            },
            
            # 动态λ调整参数（实际使用）
            'base_lambda': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('base_lambda', 0.5),
                'description': '动态λ调整的基础值'
            },
            'short_query_threshold': {
                'type': 'integer',
                'min': 1,
                'max': 50,
                'default': config.get('short_query_threshold', 10),
                'description': '短查询长度阈值'
            },
            'medium_query_threshold': {
                'type': 'integer',
                'min': 10,
                'max': 100,
                'default': config.get('medium_query_threshold', 30),
                'description': '中等查询长度阈值'
            },
            'short_adjustment': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('short_adjustment', 0.3),
                'description': '短查询时λ的调整值'
            },
            'long_adjustment': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('long_adjustment', 0.2),
                'description': '长查询时λ的调整值'
            },
            'max_lambda': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('max_lambda', 0.9),
                'description': '动态λ调整的最大值'
            },
            'min_lambda': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('min_lambda', 0.3),
                'description': '动态λ调整的最小值'
            },
            
            # 默认分数参数（实际使用）
            'default_normalized_score': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': config.get('default_normalized_score', 0.5),
                'description': '标准化时的默认分数'
            },
            
            # 回退机制参数（实际使用）
            'fallback_strategy': {
                'type': 'string',
                'options': ['score_sort', 'random', 'original', 'multi_level'],
                'default': config.get('fallback_strategy', 'multi_level'),
                'description': '回退策略：score_sort=按分数排序，random=随机选择，original=保持原序，multi_level=多级回退'
            },
            'min_results_threshold': {
                'type': 'integer',
                'min': 1,
                'max': 50,
                'default': config.get('min_results_threshold', 3),
                'description': '最小结果数量阈值，低于此值触发回退机制'
            },
            'fallback_levels': {
                'type': 'integer',
                'min': 1,
                'max': 5,
                'default': config.get('fallback_levels', 3),
                'description': '多级回退机制的级别数量'
            },
            'enable_fallback_monitoring': {
                'type': 'boolean',
                'default': config.get('enable_fallback_monitoring', True),
                'description': '是否启用回退机制监控和统计'
            }
        }