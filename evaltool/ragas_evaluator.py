#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于RAGAS的Dense向量检索评估模块
集成到现有评估体系中，支持RAGAS标准指标和自定义指标

⚠️ 重要提醒：RAGAS重试机制配置 ⚠️

RAGAS的重试机制依赖于RunConfig中的exception_types参数。默认情况下，RAGAS只重试Exception类型的错误，
但常见的网络错误如TimeoutError、OSError、ConnectionError等不在Exception范围内。

如果不在RunConfig中明确指定这些异常类型，重试机制将不会生效，导致：
1. 网络超时后直接失败，不会重试
2. 连接错误后直接失败，不会重试
3. 日志中看不到重试记录

正确的配置方式：
```python
run_config = RunConfig(
    max_workers=4,
    max_retries=4, 
    timeout=120,
    exception_types=(Exception, TimeoutError, OSError, ConnectionError)  # 必须指定！
)
```

参考文档：
- Python异常层次：https://docs.python.org/3/library/exceptions.html#exception-hierarchy
- RAGAS RunConfig：https://docs.ragas.io/en/latest/concepts/run_config.html
"""
import argparse
import json
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS, setup_logging, DEFAULT_MAX_WORKERS, MAX_RETRIES, CONNECT_TIMEOUT
from orchestrator.processor import RAGOrchestrator
from qProcessor.processorfactory import QueryProcessorFactory
from searchengine.search_factory import SearchEngineFactory
from aAggregation.aggfactory import AggFactory
from postsearch.factory import PostSearchFactory
from utils.llm_utils import call_llm
from llm.ragas_wrapper import create_ragas_llm

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# 全局导入Dataset
from datasets import Dataset
from conn.conn_factory import ConnectionFactory
from embedding.em_factory import EmbeddingFactory
DATASET_AVAILABLE = True

# RAGAS的工作参数, 使用config.py中的统一配置

try:
    from ragas import evaluate
    from ragas.metrics import (
        ContextRelevance,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
        AnswerCorrectness,
        ContextPrecision,
        ContextUtilization,
        AnswerSimilarity,
        BleuScore,
        RougeScore,
        ExactMatch
    )
    # 导入RAGAS embedding相关模块
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    logger.warning(f"RAGAS not available: {e}, please install: pip install ragas datasets")


class RAGASEvaluator:
    """RAGAS评估器，专门用于dense向量检索评估"""
    
    def __init__(self, orchestrator: RAGOrchestrator, db_type: str, embedding_model: str):
        """
        初始化RAGAS评估器
        Args:
            orchestrator: RAG编排器
            db_type: 数据库类型
            embedding_model: 嵌入模型名称
        """
        self.orchestrator = orchestrator
        self.db_type = db_type
        self.embedding_model = embedding_model
        
        # 配置评估专用日志
        self._setup_evaluation_logging()
        
        # 配置RAGAS embeddings
        self._configure_ragas_embeddings()
        
        # 配置RAGAS LLM
        self._configure_ragas_llm()
        
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not available, please install: pip install ragas datasets")
    
    def _setup_evaluation_logging(self):
        """设置评估专用日志"""
        try:
            # 创建evaltool/log目录
            log_dir = os.path.join(os.path.dirname(__file__), "log")
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名（yymmddhhmmss格式）
            timestamp = datetime.now().strftime('%y%m%d%H%M%S')
            log_file_path = os.path.join(log_dir, f'{timestamp}.log')
            
            # 配置日志
            setup_logging(log_file_path)
            
            logger.info(f"评估日志已配置: {log_file_path}")
            logger.info(f"开始RAGAS评估，数据库类型: {self.db_type}, 嵌入模型: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"配置评估日志失败: {e}")
            # 如果日志配置失败，继续使用默认配置
            pass
    
    def _configure_ragas_embeddings(self):
        """
        配置RAGAS使用我们的embedding模型
        确保与Pinecone数据库的1024维度一致
        """
        try:
            embedding_instance = EmbeddingFactory.create_embedding_model(self.embedding_model)
            run_config = RunConfig(timeout=CONNECT_TIMEOUT)  # 使用config.py中的超时配置
            self.ragas_embeddings = LangchainEmbeddingsWrapper(
                embeddings=embedding_instance,
                run_config=run_config
            )
            logger.info(f"RAGAS embedding配置完成，使用模型: {self.embedding_model}")
            test_embedding = self.ragas_embeddings.embed_query("测试文本")
            logger.info(f"RAGAS embedding测试成功，维度: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"RAGAS embedding配置失败: {e}")
            raise
    
    def _configure_ragas_llm(self):
        """配置RAGAS使用我们的LLM"""
        from config import LLM_TYPE, DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS
        
        try:
            self.ragas_llm = create_ragas_llm(
                llm_type=LLM_TYPE,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=MAX_ANSWER_TOKENS
            )
            logger.info(f"RAGAS LLM配置完成，使用模型: {LLM_TYPE}")
        except Exception as e:
            logger.error(f"RAGAS LLM配置失败: {e}")
            raise

    def _get_metrics_with_llm(self, metrics_list):
        """为RAGAS指标配置LLM和embeddings"""
        configured_metrics = []
        for metric in metrics_list:
            # 配置LLM（如果需要）
            if hasattr(metric, 'llm') and metric.llm is None:
                metric.llm = self.ragas_llm
                logger.debug(f"为指标 {type(metric).__name__} 配置LLM模型")
            # 配置embeddings（如果需要）
            if hasattr(metric, 'embeddings') and metric.embeddings is None:
                metric.embeddings = self.ragas_embeddings
                logger.debug(f"为指标 {type(metric).__name__} 配置embedding模型")
            configured_metrics.append(metric)
        return configured_metrics

    def _get_metrics_with_embeddings(self, metrics_list):
        """为RAGAS指标配置embeddings（保持向后兼容）"""
        configured_metrics = []
        for metric in metrics_list:
            if hasattr(metric, 'embeddings') and metric.embeddings is None:
                metric.embeddings = self.ragas_embeddings
                logger.debug(f"为指标 {type(metric).__name__} 配置embedding模型")
            configured_metrics.append(metric)
        return configured_metrics
    
    def prepare_dataset(self, queries: List[str], gold_answers: List[str], 
                       top_k: int = 5, rerank: bool = True) -> Dataset:
        """
        准备RAGAS评估数据集
        Args:
            queries: 查询列表
            gold_answers: 标准答案列表
            top_k: 检索数量
            rerank: 是否启用rerank
        Returns:
            Dataset: RAGAS格式的数据集
        """
        logger.info(f"开始准备RAGAS评估数据集...")
        logger.info(f"查询数量: {len(queries)}")
        logger.info(f"标准答案数量: {len(gold_answers)}")
        logger.info(f"检索数量: {top_k}")
        logger.info(f"启用rerank: {rerank}")
        
        questions = []
        contexts_list = []
        answers = []
        ground_truths = []
        
        for i, (query, gold_answer) in enumerate(zip(queries, gold_answers)):
            logger.info(f"处理第 {i+1}/{len(queries)} 个查询: {query}")
            logger.info(f"标准答案: {gold_answer[:100]}{'...' if len(gold_answer) > 100 else ''}")
            
            try:
                logger.info(f"调用orchestrator.run，参数: top_k={top_k}, db_type={self.db_type}, embedding_model={self.embedding_model}, rerank={rerank}")
                result = self.orchestrator.run(query, top_k=top_k, db_type=self.db_type, 
                                             embedding_model=self.embedding_model, rerank=rerank)
                
                logger.info(f"orchestrator.run返回结果类型: {type(result)}")
                logger.info(f"orchestrator.run返回结果键: {list(result.keys()) if isinstance(result, dict) else '非字典类型'}")
                
                contexts = result.get('answers', [])
                answer = result.get('final_answer', "未检索到有效答案")
                
                logger.info(f"提取的上下文数量: {len(contexts)}")
                logger.info(f"提取的答案长度: {len(answer)}")
                logger.info(f"答案内容: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
                questions.append(query)
                contexts_list.append(contexts)
                answers.append(answer)
                ground_truths.append(gold_answer)
                
            except Exception as e:
                logger.error(f"处理查询失败: {query}, 错误: {e}")
                logger.error(f"错误类型: {type(e).__name__}")
                import traceback
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                
                questions.append(query)
                contexts_list.append([])
                answers.append("处理失败")
                ground_truths.append(gold_answer)
        
        logger.info("开始构建RAGAS数据集...")
        logger.info(f"问题列表长度: {len(questions)}")
        logger.info(f"上下文列表长度: {len(contexts_list)}")
        logger.info(f"答案列表长度: {len(answers)}")
        logger.info(f"标准答案列表长度: {len(ground_truths)}")
        
        # 检查数据完整性
        for i, (q, c, a, g) in enumerate(zip(questions, contexts_list, answers, ground_truths)):
            if not q or not a or not g:
                logger.warning(f"样本 {i} 数据不完整: 问题='{q}', 答案='{a}', 标准答案='{g}'")
            if len(c) == 0:
                logger.warning(f"样本 {i} 上下文为空")
        
        dataset = Dataset.from_dict({
            "user_input": questions,
            "retrieved_contexts": contexts_list,
            "response": answers,
            "reference": ground_truths
        })
        
        logger.info(f"RAGAS数据集构建完成，列名: {dataset.column_names}")
        logger.info(f"数据集大小: {len(dataset)}")
        
        # 验证数据集结构
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"样本数据结构验证:")
            logger.info(f"  user_input: {type(sample['user_input'])}, 长度: {len(sample['user_input'])}")
            logger.info(f"  retrieved_contexts: {type(sample['retrieved_contexts'])}, 长度: {len(sample['retrieved_contexts'])}")
            logger.info(f"  response: {type(sample['response'])}, 长度: {len(sample['response'])}")
            logger.info(f"  reference: {type(sample['reference'])}, 长度: {len(sample['reference'])}")
        
        return dataset
    
    def f_extract_metrics_from_results(self, results, metric_map: Dict[str, List[str]]) -> Dict[str, float]:
        """
        从RAGAS结果中提取指标值的通用方法
        
        Args:
            results: RAGAS评估结果对象
            metric_map: 指标映射字典 {metric_name: [possible_keys]}
            
        Returns:
            Dict[str, float]: 提取的指标结果
        """
        extracted_results = {}
        for k, keys in metric_map.items():
            value = None
            found_key = None
            for real_key in keys:
                try:
                    value = results[real_key]
                    found_key = real_key
                    break
                except (KeyError, IndexError) as e:
                    logger.debug(f"键 {real_key} 不存在或访问失败: {e}")
                    continue
            
            if value is None:
                logger.error(f"指标 {k} 的所有可能键 {keys} 都不存在！")
                logger.error(f"可用的键: {list(results._scores_dict.keys()) if hasattr(results, '_scores_dict') else '无法获取'}")
                raise KeyError(f"指标 {k} 的键不存在，可用键: {list(results._scores_dict.keys()) if hasattr(results, '_scores_dict') else '无法获取'}")
            
            logger.info(f"指标 {k} 使用键 {found_key}，值: {value}")
            
            # 处理RAGAS返回的列表格式数据
            if isinstance(value, list):
                # 计算平均值
                if len(value) > 0:
                    avg_value = sum(value) / len(value)
                    logger.info(f"指标 {k} 列表长度: {len(value)}, 平均值: {avg_value}")
                    extracted_results[k] = float(avg_value)
                else:
                    logger.warning(f"指标 {k} 返回空列表，使用默认值0.0")
                    extracted_results[k] = 0.0
            else:
                # 处理单个值的情况（向后兼容）
                extracted_results[k] = float(value) if value is not None else 0.0
        
        return extracted_results
    
    def evaluate_retrieval_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估检索阶段指标
        
        Args:
            dataset: RAGAS数据集
            
        Returns:
            Dict[str, float]: 检索指标结果
        """
        logger.info("开始评估检索阶段指标...")
        
        # 检索阶段指标
        retrieval_metrics = [
            ContextRelevance(),
            ContextRecall(),
        ]
        
        # 为RAGAS指标配置LLM和embedding模型
        retrieval_metrics = self._get_metrics_with_llm(retrieval_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        # 注意：这里也需要配置正确的异常类型以确保重试机制生效
        run_config = RunConfig(
            max_workers=DEFAULT_MAX_WORKERS, 
            max_retries=MAX_RETRIES, 
            timeout=CONNECT_TIMEOUT,
            exception_types=(Exception, TimeoutError, OSError, ConnectionError)
        )
        results = evaluate(dataset, metrics=retrieval_metrics, run_config=run_config, raise_exceptions=False)
        logger.info(f'RAGAS返回的检索指标结果: {results}')
        metric_map = {
            'context_relevance': ['nv_context_relevance', 'context_relevance'],
            'context_recall': ['context_recall'],
        }
        
        retrieval_results = self.f_extract_metrics_from_results(results, metric_map)
        logger.info(f"检索指标结果: {retrieval_results}")
        return retrieval_results
    
    def evaluate_generation_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估生成阶段指标
        
        Args:
            dataset: RAGAS数据集
            
        Returns:
            Dict[str, float]: 生成指标结果
        """
        logger.info("开始评估生成阶段指标...")
        
        # 生成阶段指标
        generation_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness()
        ]
        
        # 为RAGAS指标配置LLM和embedding模型
        generation_metrics = self._get_metrics_with_llm(generation_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        # 注意：这里也需要配置正确的异常类型以确保重试机制生效
        run_config = RunConfig(
            max_workers=DEFAULT_MAX_WORKERS, 
            max_retries=MAX_RETRIES, 
            timeout=CONNECT_TIMEOUT,
            exception_types=(Exception, TimeoutError, OSError, ConnectionError)
        )
        results = evaluate(dataset, metrics=generation_metrics, run_config=run_config, raise_exceptions=False)
        logger.info(f'RAGAS返回的生成指标结果: {results}')
        metric_map = {
            'faithfulness': ['faithfulness'],
            'answer_relevancy': ['answer_relevancy'],
            'answer_correctness': ['answer_correctness'],
        }
        generation_results = self.f_extract_metrics_from_results(results, metric_map)
        
        logger.info(f"生成指标结果: {generation_results}")
        return generation_results
    
    def evaluate_advanced_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估高级RAGAS指标
        
        Args:
            dataset: RAGAS数据集
            
        Returns:
            Dict[str, float]: 高级指标结果
        """
        logger.info("开始评估高级RAGAS指标...")
        
        # 高级指标
        advanced_metrics = [
            ContextPrecision(),
            ContextUtilization(),
            AnswerSimilarity(),
            BleuScore(),
            RougeScore(),
            ExactMatch()
        ]
        
        # 为RAGAS指标配置LLM和embedding模型
        advanced_metrics = self._get_metrics_with_llm(advanced_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        # 注意：这里也需要配置正确的异常类型以确保重试机制生效
        run_config = RunConfig(
            max_workers=DEFAULT_MAX_WORKERS, 
            max_retries=MAX_RETRIES, 
            timeout=CONNECT_TIMEOUT,
            exception_types=(Exception, TimeoutError, OSError, ConnectionError)
        )
        results = evaluate(dataset, metrics=advanced_metrics, run_config=run_config, raise_exceptions=True)
        logger.info(f'RAGAS返回的高级指标结果: {results}')
        metric_map = {
            'context_precision': ['context_precision'],
            'context_utilization': ['context_utilization'],
            'answer_similarity': ['semantic_similarity', 'answer_similarity'],
            'bleu_score': ['bleu_score'],
            'rouge_score': ['rouge_score(mode=fmeasure)', 'rouge_score'],
            'exact_match': ['exact_match'],
        }
        advanced_results = self.f_extract_metrics_from_results(results, metric_map)
        
        logger.info(f"高级指标结果: {advanced_results}")
        return advanced_results
    
    def evaluate_all_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估所有RAGAS指标
        Args:
            dataset: RAGAS格式的数据集
        Returns:
            Dict[str, float]: 所有指标的结果
        """
        logger.info("开始评估所有RAGAS指标...")
        logger.info(f"数据集大小: {len(dataset)} 个样本")
        
        # 检查数据集列名
        logger.info(f"数据集列名: {dataset.column_names}")
        
        # 检查数据集样本
        if len(dataset) > 0:
            sample_item = dataset[0]
            logger.info(f"样本数据结构: {list(sample_item.keys())}")
            logger.info(f"样本问题长度: {len(sample_item.get('user_input', ''))}")
            logger.info(f"样本上下文数量: {len(sample_item.get('retrieved_contexts', []))}")
            logger.info(f"样本答案长度: {len(sample_item.get('response', ''))}")
            logger.info(f"样本标准答案长度: {len(sample_item.get('reference', ''))}")
        
        all_metrics = [
            ContextRelevance(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
            ContextPrecision(),
            ContextUtilization(),
            AnswerSimilarity(),
            BleuScore(),
            RougeScore(),
            ExactMatch()
        ]
        
        logger.info("开始配置RAGAS指标...")
        # 为RAGAS指标配置LLM和embedding模型
        all_metrics = self._get_metrics_with_llm(all_metrics)
        logger.info("RAGAS指标配置完成")
        
        # 使用RunConfig控制并发，避免API限流
        # 
        # ⚠️ 重要说明：RAGAS重试机制配置 ⚠️
        # 
        # RAGAS的重试机制依赖于exception_types参数，默认只重试Exception类型。
        # 但是常见的网络错误如TimeoutError、OSError等不在Exception范围内，
        # 如果不明确指定这些异常类型，重试机制将不会生效！
        # 
        # 异常类型层次结构：
        # - BaseException (根异常类)
        #   ├── Exception (默认重试类型)
        #   ├── TimeoutError (超时错误，需要明确指定)
        #   ├── OSError (操作系统错误，需要明确指定)
        #   └── KeyboardInterrupt (用户中断，不应重试)
        # 
        # 参考：https://docs.python.org/3/library/exceptions.html#exception-hierarchy
        # 
        # 如果后续遇到其他类型的网络错误，请在此处添加相应的异常类型
        # 确保重试机制能够正确捕获和处理这些错误
        # 
        run_config = RunConfig(
            max_workers=DEFAULT_MAX_WORKERS, 
            max_retries=MAX_RETRIES, 
            timeout=CONNECT_TIMEOUT,
            exception_types=(Exception, TimeoutError, OSError, ConnectionError)
        )
        logger.info(f"使用RunConfig: max_workers={DEFAULT_MAX_WORKERS}, max_retries={MAX_RETRIES}, timeout={CONNECT_TIMEOUT}")
        logger.info(f"重试异常类型: {run_config.exception_types}")
        
        logger.info("开始执行RAGAS评估...")
        results = evaluate(dataset, metrics=all_metrics, run_config=run_config, raise_exceptions=False)
        logger.info(f'RAGAS返回的所有指标结果: {results}')
        
        # 详细记录每个指标的结果
        try:
            if hasattr(results, '_repr_dict'):
                for metric_name, metric_value in results._repr_dict.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_value != metric_value:  # 检查是否为nan
                            logger.warning(f"指标 {metric_name} 计算结果为nan")
                        else:
                            logger.info(f"指标 {metric_name}: {metric_value:.4f}")
                    else:
                        logger.info(f"指标 {metric_name}: {metric_value}")
            else:
                logger.warning("无法获取指标结果，跳过详细日志记录")
        except Exception as e:
            logger.warning(f"记录指标日志时出错: {e}，跳过详细日志记录")
        
        metric_map = {
            'context_relevance': ['nv_context_relevance', 'context_relevance'],
            'context_recall': ['context_recall'],
            'faithfulness': ['faithfulness'],
            'answer_relevancy': ['answer_relevancy'],
            'answer_correctness': ['answer_correctness'],
            'context_precision': ['context_precision'],
            'context_utilization': ['context_utilization'],
            'answer_similarity': ['semantic_similarity', 'answer_similarity'],
            'bleu_score': ['bleu_score'],
            'rouge_score': ['rouge_score(mode=fmeasure)', 'rouge_score'],
            'exact_match': ['exact_match'],
        }
        
        logger.info("开始提取指标结果...")
        all_results = self.f_extract_metrics_from_results(results, metric_map)
        
        # 检查nan指标
        nan_metrics = [k for k, v in all_results.items() if isinstance(v, (int, float)) and v != v]
        if nan_metrics:
            logger.warning(f"发现nan指标: {nan_metrics}")
            for metric in nan_metrics:
                logger.warning(f"指标 {metric} 为nan，可能的原因:")
                if metric == 'faithfulness':
                    logger.warning("  - 数据集缺少必要的列或列名不匹配")
                    logger.warning("  - 上下文或答案为空")
                elif metric == 'answer_correctness':
                    logger.warning("  - 标准答案列缺失或为空")
                    logger.warning("  - 答案格式不匹配")
                elif metric == 'context_precision':
                    logger.warning("  - 上下文数据格式问题")
                    logger.warning("  - 检索结果为空")
                elif metric == 'context_utilization':
                    logger.warning("  - 上下文利用率计算失败")
                    logger.warning("  - 答案生成过程异常")
        
        logger.info(f"所有指标结果: {all_results}")
        logger.info("RAGAS指标评估完成")
        return all_results
    
    def generate_llm_optimization_suggestions(self, dataset: Dataset, results: Dict[str, float]) -> str:
        """
        使用LLM生成智能优化建议
        
        Args:
            dataset: RAGAS数据集
            results: 评估结果
            
        Returns:
            str: LLM生成的优化建议
        """
        try:
            # 1. 计算指标统计
            retrieval_metrics = ['context_relevance', 'context_recall', 'context_precision']
            generation_metrics = ['faithfulness', 'answer_relevancy', 'answer_correctness']
            
            retrieval_avg = sum(results.get(m, 0) for m in retrieval_metrics) / len(retrieval_metrics)
            generation_avg = sum(results.get(m, 0) for m in generation_metrics) / len(generation_metrics)
            
            # 2. 分析样本数据
            sample_analysis = {
                'total_samples': len(dataset),
                'avg_context_count': sum(len(d['retrieved_contexts']) for d in dataset) / len(dataset),
                'samples_with_zero_context': len([d for d in dataset if len(d['retrieved_contexts']) == 0]),
                'avg_answer_length': sum(len(d['response']) for d in dataset) / len(dataset),
                'samples_with_short_answers': len([d for d in dataset if len(d['response']) < 50]),
                'samples_with_long_answers': len([d for d in dataset if len(d['response']) > 500])
            }
            
            # 3. 识别问题样本
            problem_samples = []
            for i, item in enumerate(dataset):
                if len(item['retrieved_contexts']) == 0 or len(item['response']) < 20:
                    problem_samples.append({
                        'index': i,
                        'question': item['user_input'][:100] + "..." if len(item['user_input']) > 100 else item['user_input'],
                        'context_count': len(item['retrieved_contexts']),
                        'answer_length': len(item['response'])
                    })
            
            # 4. 构建LLM prompt
            prompt = f"""
你是一名经验丰富的数据分析师以及RAG系统优化专家，你的任务是:

1. 详细分析评估下面的query评估报告
2. 结合业界主流的技术方案,给出具体的,可操作的优化建议.
3. 给出可用户可参考的技术文章出处或链接

下面是query评估报告的主要内容:
<Accessment report>
## 评估指标详情
{json.dumps(results, indent=2, ensure_ascii=False)}

## 指标分析
- 检索阶段平均得分: {retrieval_avg:.4f}
- 生成阶段平均得分: {generation_avg:.4f}
- 最弱指标: {min(results.items(), key=lambda x: x[1])[0]} ({min(results.values()):.4f})
- 最强指标: {max(results.items(), key=lambda x: x[1])[0]} ({max(results.values()):.4f})

## 样本分析
{json.dumps(sample_analysis, indent=2, ensure_ascii=False)}

## 问题样本示例
{json.dumps(problem_samples[:3], indent=2, ensure_ascii=False)}
</Accessment report>

## 输出要求
1. 基于数据给出优先级排序的建议
2. 针对最弱指标给出具体改进措施,结合业界主流技术给出 1-3 条建议
3. 考虑指标间的相互影响
4. 提供可量化的改进目标
5. 针对问题样本给出具体解决方案

请输出结构化的优化建议：
"""
            
            # 5. 调用LLM
            return call_llm(
                prompt=prompt,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=MAX_ANSWER_TOKENS
            )
            
        except Exception as e:
            logger.error(f"生成LLM优化建议失败: {str(e)}")
            return "抱歉，生成优化建议时出现错误。"

    def generate_detailed_report(self, dataset: Dataset, results: Dict[str, float], 
                               output_dir: str) -> str:
        """
        生成详细评估报告
        
        Args:
            dataset: RAGAS数据集
            results: 评估结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'ragas_evaluation_report_{timestamp}.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# RAGAS Dense向量检索评估报告\n\n")
            f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**评估样本数**: {len(dataset)}\n\n")
            
            # 指标结果
            f.write("## 评估指标结果\n\n")
            f.write("| 指标 | 分数 | 说明 |\n")
            f.write("|------|------|------|\n")
            
            metric_descriptions = {
                'context_relevance': '检索上下文相关性',
                'context_recall': '检索上下文召回率',
                'faithfulness': '生成答案对检索内容的忠实度',
                'answer_relevancy': '生成答案相关性',
                'answer_correctness': '生成答案正确性',
                'context_precision': '检索上下文精确度',
                'context_utilization': '检索上下文利用率',
                'answer_similarity': '生成答案与标准答案相似度',
                'bleu_score': 'BLEU评分（机器翻译质量）',
                'rouge_score': 'ROUGE评分（文本摘要质量）',
                'exact_match': '精确匹配率'
            }
            
            for metric, score in results.items():
                description = metric_descriptions.get(metric, '')
                f.write(f"| {metric} | {score:.4f} | {description} |\n")
            
            # 详细分析
            f.write("\n## 详细分析\n\n")
            
            # 检索阶段分析
            f.write("### 检索阶段分析\n\n")
            context_relevance = results.get('context_relevance', 0)
            context_recall = results.get('context_recall', 0)
            
            f.write(f"- **上下文相关性**: {context_relevance:.4f}\n")
            if context_relevance > 0.8:
                f.write("  - ✅ 检索上下文质量很高\n")
            elif context_relevance > 0.6:
                f.write("  - ⚠️ 检索上下文质量一般，建议优化检索策略\n")
            else:
                f.write("  - ❌ 检索上下文质量较差，需要大幅优化\n")
            
            f.write(f"- **上下文召回率**: {context_recall:.4f}\n")
            if context_recall > 0.8:
                f.write("  - ✅ 召回率很高，能覆盖大部分相关信息\n")
            elif context_recall > 0.6:
                f.write("  - ⚠️ 召回率一般，可能遗漏重要信息\n")
            else:
                f.write("  - ❌ 召回率较低，需要提升检索覆盖度\n")
            
            # 生成阶段分析
            f.write("\n### 生成阶段分析\n\n")
            faithfulness = results.get('faithfulness', 0)
            answer_relevancy = results.get('answer_relevancy', 0)
            answer_correctness = results.get('answer_correctness', 0)
            
            f.write(f"- **答案忠实度**: {faithfulness:.4f}\n")
            if faithfulness > 0.8:
                f.write("  - ✅ 生成答案很好地基于检索内容\n")
            elif faithfulness > 0.6:
                f.write("  - ⚠️ 生成答案部分偏离检索内容\n")
            else:
                f.write("  - ❌ 生成答案严重偏离检索内容\n")
            
            f.write(f"- **答案相关性**: {answer_relevancy:.4f}\n")
            f.write(f"- **答案正确性**: {answer_correctness:.4f}\n")
            
            # 高级指标分析
            f.write("\n### 高级指标分析\n\n")
            context_precision = results.get('context_precision', 0)
            context_utilization = results.get('context_utilization', 0)
            answer_similarity = results.get('answer_similarity', 0)
            bleu_score = results.get('bleu_score', 0)
            rouge_score = results.get('rouge_score', 0)
            exact_match = results.get('exact_match', 0)
            
            f.write(f"- **上下文精确度**: {context_precision:.4f}\n")
            if context_precision > 0.8:
                f.write("  - ✅ 检索结果精确度很高\n")
            elif context_precision > 0.6:
                f.write("  - ⚠️ 检索结果精确度一般\n")
            else:
                f.write("  - ❌ 检索结果精确度较低\n")
            
            f.write(f"- **上下文利用率**: {context_utilization:.4f}\n")
            if context_utilization > 0.8:
                f.write("  - ✅ 生成答案很好地利用了检索内容\n")
            elif context_utilization > 0.6:
                f.write("  - ⚠️ 生成答案部分利用了检索内容\n")
            else:
                f.write("  - ❌ 生成答案很少利用检索内容\n")
            
            f.write(f"- **答案相似度**: {answer_similarity:.4f}\n")
            f.write(f"- **BLEU评分**: {bleu_score:.4f}\n")
            f.write(f"- **ROUGE评分**: {rouge_score:.4f}\n")
            f.write(f"- **精确匹配率**: {exact_match:.4f}\n")
            
            # 优化建议
            f.write("\n## 优化建议\n\n")
            
            # 使用LLM生成智能优化建议
            llm_suggestions = self.generate_llm_optimization_suggestions(dataset, results)
            f.write(llm_suggestions)
            f.write("\n")
            
            # 样本详情
            f.write("\n## 样本详情\n\n")
            f.write("| 序号 | 问题 | 生成答案 | 标准答案 | 上下文数量 |\n")
            f.write("|------|------|----------|----------|------------|\n")
            
            for i in range(min(10, len(dataset))):  # 只显示前10个样本
                question = dataset[i]['user_input']
                answer = dataset[i]['response'][:100] + "..." if len(dataset[i]['response']) > 100 else dataset[i]['response']
                ground_truth = dataset[i]['reference'][:100] + "..." if len(dataset[i]['reference']) > 100 else dataset[i]['reference']
                context_count = len(dataset[i]['retrieved_contexts'])
                
                f.write(f"| {i+1} | {question} | {answer} | {ground_truth} | {context_count} |\n")
        
        logger.info(f"详细报告已生成: {report_path}")
        return report_path
    
    def save_dataset_details(self, dataset: Dataset, output_dir: str) -> str:
        """
        保存数据集详细信息
        
        Args:
            dataset: RAGAS数据集
            output_dir: 输出目录
            
        Returns:
            str: 详情文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        details_path = os.path.join(output_dir, f'dataset_details_{timestamp}.json')
        
        details = []
        for i, item in enumerate(dataset):
            detail = {
                'index': i,
                'question': item['user_input'],
                'contexts': item['retrieved_contexts'],
                'answer': item['response'],
                'ground_truth': item['reference'],
                'context_count': len(item['retrieved_contexts'])
            }
            details.append(detail)
        
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集详情已保存: {details_path}")
        return details_path


def run_ragas_evaluation(query_file: str, gold_file: str, top_k: int = 5, 
                        output_dir: str = None, qp_type: str = 'straight', se_type: str = 'dense', ps_type='simple',agg_type: str = 'simple',db_type: str = 'opensearch',embedding_model: str = 'qwen', rerank: bool = True) -> Dict[str, Any]:
    """
    运行RAGAS评估的主函数
    
    Args:
        query_file: 查询文件路径
        gold_file: 标准答案文件路径
        top_k: 检索数量
        output_dir: 输出目录
        qp_type: query processor类型
        se_type: search engine类型
        ps_type: post search类型
        agg_type: aggregator类型
        db_type: 数据库类型
        embedding_model: 嵌入模型
        rerank: 是否启用rerank
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS not available, please install: pip install ragas datasets")
    
    # 创建输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d')
        output_dir = os.path.join('document', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    logger.info(f"加载查询文件: {query_file}")
    queries_df = pd.read_csv(query_file)
    queries = queries_df.iloc[:, 0].astype(str).tolist()
    
    logger.info(f"加载标准答案文件: {gold_file}")
    gold_df = pd.read_csv(gold_file)
    gold_answers = gold_df.iloc[:, 1].astype(str).tolist()
    
    if len(queries) != len(gold_answers):
        raise ValueError(f"查询数量({len(queries)})与标准答案数量({len(gold_answers)})不匹配")
    
    logger.info(f"共加载 {len(queries)} 个查询和标准答案对")
    
    # 初始化RAGOrchestrator（工厂组装）
    query_processor = QueryProcessorFactory.create_processor(qp_type)
    search_engine = SearchEngineFactory.create(se_type)
    aggregator = AggFactory.get_aggregator(agg_type)
    postsearch = PostSearchFactory.create_processor(ps_type)
    orchestrator = RAGOrchestrator(query_processor, search_engine, postsearch,aggregator)
    
    # 初始化RAGASEvaluator
    evaluator = RAGASEvaluator(orchestrator,db_type,embedding_model)
    
    # 准备数据集
    logger.info("准备RAGAS数据集...")
    dataset = evaluator.prepare_dataset(queries, gold_answers, top_k, rerank)
    
    # 执行评估
    logger.info("执行RAGAS评估...")
    results = evaluator.evaluate_all_metrics(dataset)
    
    # 详细日志记录
    logger.info(f"RAGAS评估完成，开始生成报告...")
    logger.info(f"结果类型: {type(results)}")
    logger.info(f"结果内容: {results}")
    
    # 生成报告
    logger.info("生成评估报告...")
    try:
        report_path = evaluator.generate_detailed_report(dataset, results, output_dir)
        logger.info(f"报告生成完成: {report_path}")
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        logger.error(f"异常详情: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise
    
    # 保存数据集详情
    logger.info("保存数据集详情...")
    try:
        details_path = evaluator.save_dataset_details(dataset, output_dir)
        logger.info(f"详情保存完成: {details_path}")
    except Exception as e:
        logger.error(f"详情保存失败: {e}")
        logger.error(f"异常详情: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise
    
    # 返回结果
    logger.info("准备返回评估结果...")
    evaluation_result = {
        'metrics': results,
        'report_path': report_path,
        'details_path': details_path,
        'sample_count': len(dataset),
        'evaluation_time': datetime.now().isoformat()
    }
    
    logger.info(f"评估结果准备完成: {evaluation_result}")
    logger.info("RAGAS评估完成")
    return evaluation_result


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='RAGAS Dense向量检索评估')
    parser.add_argument('--config_file', type=str, required=True, help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 读取并解析配置文件
        logger.info(f"读取配置文件: {args.config_file}")
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"配置文件内容: {config}")
        
        # 验证必需参数
        required_params = ['query_file', 'gold_file']
        for param in required_params:
            if param not in config:
                raise ValueError(f"配置文件缺少必需参数: {param}")
        
        # 设置默认值
        config.setdefault('top_k', 5)
        config.setdefault('output_dir', None)
        config.setdefault('qp_type', 'straight')
        config.setdefault('se_type', 'dense')
        config.setdefault('ps_type', 'simple')
        config.setdefault('agg_type', 'simple')
        config.setdefault('db_type', 'opensearch')
        config.setdefault('embedding_model', 'qwen')
        config.setdefault('rerank', True)
        
        logger.info("开始调用run_ragas_evaluation函数...")
        result = run_ragas_evaluation(
            query_file=config['query_file'],
            gold_file=config['gold_file'],
            top_k=config['top_k'],
            output_dir=config['output_dir'],
            qp_type=config['qp_type'],
            se_type=config['se_type'],
            ps_type=config['ps_type'],
            agg_type=config['agg_type'],
            db_type=config['db_type'],
            embedding_model=config['embedding_model'],
            rerank=config['rerank']
        )
        
        logger.info("run_ragas_evaluation函数调用成功，开始打印结果...")
        print("=" * 60)
        print("RAGAS评估结果摘要")
        print("=" * 60)
        for metric, score in result['metrics'].items():
            print(f"{metric}: {score:.4f}")
        print(f"报告路径: {result['report_path']}")
        print(f"详情路径: {result['details_path']}")
        logger.info("结果打印完成")
        
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {args.config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"评估失败: {e}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"完整堆栈: {traceback.format_exc()}")
        sys.exit(1) 