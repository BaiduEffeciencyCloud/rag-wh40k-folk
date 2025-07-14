#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于RAGAS的Dense向量检索评估模块
集成到现有评估体系中，支持RAGAS标准指标和自定义指标
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, get_embedding_model, EMBADDING_MODEL
# from vector_search import VectorSearch  # 移除vector_search依赖
from orchestrator.processor import RAGOrchestrator
from qProcessor.processorfactory import QueryProcessorFactory
from searchengine.search_factory import SearchEngineFactory
from aAggregation.aggfactory import AggFactory
from postsearch.factory import PostSearchFactory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局导入Dataset
from datasets import Dataset
DATASET_AVAILABLE = True

# RAGAS的工作参数, MAX_WORKERS过大可能导致超过openai一分钟内的token上限
MAX_WORKERS=4
MAX_RETRIES=3
TIMEOUT=60

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
    
    def __init__(self, orchestrator: RAGOrchestrator):
        """
        初始化RAGAS评估器
        Args:
            orchestrator: RAG主流程调度器实例
        """
        self.orchestrator = orchestrator
        self.embedding_model = get_embedding_model()
        self._configure_ragas_embeddings()
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not available, please install: pip install ragas datasets")
    
    def _configure_ragas_embeddings(self):
        """
        配置RAGAS使用我们的embedding模型
        确保与Pinecone数据库的1024维度一致
        """
        try:
            run_config = RunConfig(timeout=30)  # 设置30秒超时
            self.ragas_embeddings = LangchainEmbeddingsWrapper(
                embeddings=self.embedding_model,
                run_config=run_config
            )
            logger.info(f"RAGAS embedding配置完成，使用模型: {EMBADDING_MODEL}")
            test_embedding = self.ragas_embeddings.embed_query("测试文本")
            logger.info(f"RAGAS embedding测试成功，维度: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"RAGAS embedding配置失败: {e}")
            raise
    
    def _get_metrics_with_embeddings(self, metrics_list):
        configured_metrics = []
        for metric in metrics_list:
            if hasattr(metric, 'embeddings') and metric.embeddings is None:
                metric.embeddings = self.ragas_embeddings
                logger.debug(f"为指标 {type(metric).__name__} 配置embedding模型")
            configured_metrics.append(metric)
        return configured_metrics
    
    def prepare_dataset(self, queries: List[str], gold_answers: List[str], 
                       top_k: int = 5) -> Dataset:
        """
        准备RAGAS评估数据集
        Args:
            queries: 查询列表
            gold_answers: 标准答案列表
            top_k: 检索数量
        Returns:
            Dataset: RAGAS格式的数据集
        """
        questions = []
        contexts_list = []
        answers = []
        ground_truths = []
        for i, (query, gold_answer) in enumerate(zip(queries, gold_answers)):
            logger.info(f"处理第 {i+1}/{len(queries)} 个查询: {query}")
            try:
                result = self.orchestrator.run(query, top_k=top_k)
                contexts = result.get('answers', [])
                answer = result.get('final_answer', "未检索到有效答案")
                questions.append(query)
                contexts_list.append(contexts)
                answers.append(answer)
                ground_truths.append(gold_answer)
            except Exception as e:
                logger.error(f"处理查询失败: {query}, 错误: {e}")
                questions.append(query)
                contexts_list.append([])
                answers.append("处理失败")
                ground_truths.append(gold_answer)
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts_list,
            "answer": answers,
            "ground_truth": ground_truths
        })
        return dataset
    
    def _extract_metrics_from_results(self, results, metric_map: Dict[str, List[str]]) -> Dict[str, float]:
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
        
        # 为RAGAS指标配置embedding模型
        retrieval_metrics = self._get_metrics_with_embeddings(retrieval_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        run_config = RunConfig(max_workers=MAX_WORKERS, max_retries=MAX_RETRIES, timeout=TIMEOUT)
        results = evaluate(dataset, metrics=retrieval_metrics, run_config=run_config)
        logger.info(f'RAGAS返回的检索指标结果: {results}')
        metric_map = {
            'context_relevance': ['nv_context_relevance', 'context_relevance'],
            'context_recall': ['context_recall'],
        }
        
        retrieval_results = self._extract_metrics_from_results(results, metric_map)
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
        
        # 为RAGAS指标配置embedding模型
        generation_metrics = self._get_metrics_with_embeddings(generation_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        run_config = RunConfig(max_workers=1, max_retries=3, timeout=60)
        results = evaluate(dataset, metrics=generation_metrics, run_config=run_config)
        logger.info(f'RAGAS返回的生成指标结果: {results}')
        metric_map = {
            'faithfulness': ['faithfulness'],
            'answer_relevancy': ['answer_relevancy'],
            'answer_correctness': ['answer_correctness'],
        }
        generation_results = self._extract_metrics_from_results(results, metric_map)
        
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
        
        # 为RAGAS指标配置embedding模型
        advanced_metrics = self._get_metrics_with_embeddings(advanced_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        run_config = RunConfig(max_workers=1, max_retries=3, timeout=60)
        results = evaluate(dataset, metrics=advanced_metrics, run_config=run_config)
        logger.info(f'RAGAS返回的高级指标结果: {results}')
        metric_map = {
            'context_precision': ['context_precision'],
            'context_utilization': ['context_utilization'],
            'answer_similarity': ['semantic_similarity', 'answer_similarity'],
            'bleu_score': ['bleu_score'],
            'rouge_score': ['rouge_score(mode=fmeasure)', 'rouge_score'],
            'exact_match': ['exact_match'],
        }
        advanced_results = self._extract_metrics_from_results(results, metric_map)
        
        logger.info(f"高级指标结果: {advanced_results}")
        return advanced_results
    
    def evaluate_all_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        评估所有RAGAS指标
        
        Args:
            dataset: RAGAS数据集
            
        Returns:
            Dict[str, float]: 所有指标结果
        """
        logger.info("开始评估所有RAGAS指标...")
        
        # 所有指标
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
        
        # 为RAGAS指标配置embedding模型
        all_metrics = self._get_metrics_with_embeddings(all_metrics)
        
        # 使用RunConfig控制并发，避免API限流
        run_config = RunConfig(max_workers=1, max_retries=3, timeout=60)
        results = evaluate(dataset, metrics=all_metrics, run_config=run_config)
        logger.info(f'RAGAS返回的所有指标结果: {results}')
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
        all_results = self._extract_metrics_from_results(results, metric_map)
        
        logger.info(f"所有指标结果: {all_results}")
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
                'avg_context_count': sum(len(d['contexts']) for d in dataset) / len(dataset),
                'samples_with_zero_context': len([d for d in dataset if len(d['contexts']) == 0]),
                'avg_answer_length': sum(len(d['answer']) for d in dataset) / len(dataset),
                'samples_with_short_answers': len([d for d in dataset if len(d['answer']) < 50]),
                'samples_with_long_answers': len([d for d in dataset if len(d['answer']) > 500])
            }
            
            # 3. 识别问题样本
            problem_samples = []
            for i, item in enumerate(dataset):
                if len(item['contexts']) == 0 or len(item['answer']) < 20:
                    problem_samples.append({
                        'index': i,
                        'question': item['question'][:100] + "..." if len(item['question']) > 100 else item['question'],
                        'context_count': len(item['contexts']),
                        'answer_length': len(item['answer'])
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
            from utils.llm_utils import call_llm
            from openai import OpenAI
            from config import OPENAI_API_KEY
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            return call_llm(
                client=client,
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000
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
                question = dataset[i]['question']
                answer = dataset[i]['answer'][:100] + "..." if len(dataset[i]['answer']) > 100 else dataset[i]['answer']
                ground_truth = dataset[i]['ground_truth'][:100] + "..." if len(dataset[i]['ground_truth']) > 100 else dataset[i]['ground_truth']
                context_count = len(dataset[i]['contexts'])
                
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
                'question': item['question'],
                'contexts': item['contexts'],
                'answer': item['answer'],
                'ground_truth': item['ground_truth'],
                'context_count': len(item['contexts'])
            }
            details.append(detail)
        
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集详情已保存: {details_path}")
        return details_path


def run_ragas_evaluation(query_file: str, gold_file: str, top_k: int = 5, 
                        output_dir: str = None, qp_type: str = 'straight', se_type: str = 'dense', ps_type='simple',agg_type: str = 'simple') -> Dict[str, Any]:
    """
    运行RAGAS评估的主函数
    
    Args:
        query_file: 查询文件路径
        gold_file: 标准答案文件路径
        top_k: 检索数量
        output_dir: 输出目录
        
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
    evaluator = RAGASEvaluator(orchestrator)
    
    # 准备数据集
    logger.info("准备RAGAS数据集...")
    dataset = evaluator.prepare_dataset(queries, gold_answers, top_k)
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGAS Dense向量检索评估')
    parser.add_argument('--query', type=str, required=True, help='查询文件路径')
    parser.add_argument('--gold', type=str, required=True, help='标准答案文件路径')
    parser.add_argument('--top_k', type=int, default=5, help='检索数量')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--qp', type=str, default='straight', help='query processor类型')
    parser.add_argument('--se', type=str, default='dense', help='search engine类型')
    parser.add_argument('--ps',type=str,default='simple',help='top_k的后处理')
    parser.add_argument('--agg', type=str, default='simple', help='聚合器类型')
    
    args = parser.parse_args()
    
    try:
        logger.info("开始调用run_ragas_evaluation函数...")
        result = run_ragas_evaluation(
            query_file=args.query,
            gold_file=args.gold,
            top_k=args.top_k,
            output_dir=args.output,
            qp_type=args.qp,
            se_type=args.se,
            ps_type=args.ps,
            agg_type=args.agg
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
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"完整堆栈: {traceback.format_exc()}")
        sys.exit(1) 