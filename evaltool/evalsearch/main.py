#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估主入口
实现评估流程：参数解析->决定评估项->生成评估文件夹和空白报告->递归执行评估项并append报告->保存最终报告
"""

import sys
import os
import traceback
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .utils import (
    parse_args, load_queries, load_gold,
    create_eval_dir, create_empty_report, append_evaluation_result,
    append_error, save_report, save_input_files
)
from .factory import SearchEngineFactory, process_queries_with_factory, build_retrieved_dict
from .visualization import (
    plot_metrics_comparison, create_visualization_summary
)
from .config_loader import config_loader


def execute_evaluation_item(eval_item: str, queries: List[str], gold: Optional[Dict],
                           engine, args, eval_dir: str) -> tuple[Dict, Optional[str]]:
    """
    执行单个评估项（使用JSON配置词典）
    
    Args:
        eval_item: 评估项名称
        queries: query列表
        gold: gold标准答案（可选）
        engine: 检索引擎
        args: 参数
        eval_dir: 评估目录
        
    Returns:
        tuple[Dict, Optional[str]]: (评估结果, 可视化文件路径)
    """
    try:
        # 处理queries
        processed_queries = process_queries_with_factory(queries, args.qp)
        
        # 执行检索
        retrieved_dict, all_scores = build_retrieved_dict(processed_queries, engine, args.topk)
        
        # 根据JSON配置执行评估
        result = {}
        viz_path = None
        
        if eval_item == 'score_distribution':
            from .metrics import DenseRetrievalMetrics
            from .visualization import plot_score_distribution
            
            result = DenseRetrievalMetrics.score_distribution(all_scores)
            viz_path = os.path.join(eval_dir, 'score_distribution.png')
            plot_score_distribution(all_scores, viz_path)
            
        elif eval_item == 'embedding_distribution':
            from .metrics import DenseRetrievalMetrics
            from .visualization import plot_embedding_distribution
            
            embeddings = engine.get_embeddings(processed_queries)
            coords = DenseRetrievalMetrics.embedding_distribution(embeddings, method="tsne")
            result = {
                'method': 'tsne',
                'shape': coords.shape,
                'sample_count': len(processed_queries)
            }
            viz_path = os.path.join(eval_dir, 'embedding_distribution.png')
            plot_embedding_distribution(embeddings, viz_path)
            
        elif eval_item in ['recall_at_k', 'mrr', 'precision_at_k']:
            from .metrics import DenseRetrievalMetrics
            from .visualization import plot_recall_distribution
            
            if gold is None:
                result = {'error': '需要gold标准答案'}
            else:
                if eval_item == 'recall_at_k':
                    recall_at_k = DenseRetrievalMetrics.recall_at_k(gold, retrieved_dict, args.topk)
                    result = {'recall_at_k': recall_at_k}
                    viz_path = os.path.join(eval_dir, 'recall_distribution.png')
                    recall_values = [recall_at_k] * len(queries)
                    plot_recall_distribution(recall_values, viz_path)
                elif eval_item == 'mrr':
                    mrr = DenseRetrievalMetrics.mrr(gold, retrieved_dict, args.topk)
                    result = {'mrr': mrr}
                elif eval_item == 'precision_at_k':
                    precision_at_k = DenseRetrievalMetrics.precision_at_k(gold, retrieved_dict, args.topk)
                    result = {'precision_at_k': precision_at_k}
                    
        elif eval_item == 'overlap_stats':
            from .metrics import HybridRecallMetrics
            from .visualization import plot_overlap_stats
            
            if args.se != 'hybrid':
                result = {'error': '需要hybrid检索引擎'}
            else:
                dense_results = {f"q{i+1}": retrieved_dict[f"q{i+1}"][:args.topk] for i in range(len(queries))}
                sparse_results = {f"q{i+1}": [f"sparse_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
                graph_results = {f"q{i+1}": [f"graph_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
                
                hybrid_metrics = HybridRecallMetrics(dense_results, sparse_results, graph_results, args.topk)
                overlap_stats = hybrid_metrics.overlap_stats()
                result = {'overlap_stats': overlap_stats}
                
                viz_path = os.path.join(eval_dir, 'overlap_stats.png')
                plot_overlap_stats(overlap_stats, viz_path)
                
        elif eval_item == 'hybrid_union_recall':
            from .metrics import HybridRecallMetrics
            
            if gold is None:
                result = {'error': '需要gold标准答案'}
            elif args.se != 'hybrid':
                result = {'error': '需要hybrid检索引擎'}
            else:
                dense_results = {f"q{i+1}": retrieved_dict[f"q{i+1}"][:args.topk] for i in range(len(queries))}
                sparse_results = {f"q{i+1}": [f"sparse_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
                graph_results = {f"q{i+1}": [f"graph_doc_{i}_{j}" for j in range(args.topk)] for i in range(len(queries))}
                
                union_recall = HybridRecallMetrics.hybrid_union_recall(
                    gold, dense_results, sparse_results, graph_results, args.topk
                )
                result = {'hybrid_union_recall': union_recall}
        
        return result, viz_path
        
    except Exception as e:
        error_msg = f"执行{eval_item}时出错: {str(e)}\n{traceback.format_exc()}"
        return {'error': error_msg}, None


def main():
    """主函数"""
    try:
        # 1. 系统参数解析
        print("=== 1. 系统参数解析 ===")
        args = parse_args()
        print(f"Query文件: {args.query}")
        print(f"Gold文件: {args.gold}")
        print(f"Top-K: {args.topk}")
        print(f"Query处理模式: {args.qp}")
        print(f"检索引擎类型: {args.se}")
        
        # 2. 根据参数决定执行哪些评估项
        print("\n=== 2. 决定评估项 ===")
        eval_items = config_loader.get_available_evaluators(args)
        print(f"将执行以下评估项: {eval_items}")
        for eval_item in eval_items:
            description = config_loader.get_evaluator_description(eval_item)
            print(f"  - {eval_item}: {description}")
        
        # 打印评估项词典信息
        config_loader.print_evaluator_info(args)
        
        # 3. 生成评估文件夹和空白报告
        print("\n=== 3. 生成评估文件夹和空白报告 ===")
        eval_dir, report_path = create_eval_dir()
        report = create_empty_report(args, eval_items)
        print(f"评估目录: {eval_dir}")
        print(f"报告文件: {report_path}")
        
        # 4. 加载数据
        print("\n=== 4. 加载数据 ===")
        queries = load_queries(args.query)
        print(f"加载了 {len(queries)} 个queries")
        
        gold = None
        if args.gold and os.path.exists(args.gold):
            gold = load_gold(args.gold)
            print(f"加载了 {len(gold)} 个gold标准答案")
        
        # 5. 创建检索引擎
        print("\n=== 5. 创建检索引擎 ===")
        engine = SearchEngineFactory.create_engine(args.se)
        print(f"创建了 {args.se} 检索引擎")
        
        # 6. 递归执行评估项并append报告
        print("\n=== 6. 递归执行评估项 ===")
        for i, eval_item in enumerate(eval_items, 1):
            print(f"\n--- 执行评估项 {i}/{len(eval_items)}: {eval_item} ---")
            
            try:
                result, viz_path = execute_evaluation_item(
                    eval_item, queries, gold, engine, args, eval_dir
                )
                
                # 追加评估结果到报告
                append_evaluation_result(report, eval_item, result, viz_path)
                print(f"✓ {eval_item} 执行成功")
                
                if viz_path:
                    print(f"  生成可视化: {os.path.basename(viz_path)}")
                    
            except Exception as e:
                error_msg = f"执行{eval_item}失败: {str(e)}"
                append_error(report, eval_item, error_msg)
                print(f"✗ {eval_item} 执行失败: {str(e)}")
        
        # 7. 生成指标对比可视化
        print("\n=== 7. 生成指标对比可视化 ===")
        try:
            metrics_dict = {}
            for item, result in report['evaluation_results'].items():
                if isinstance(result, dict) and 'error' not in result:
                    # 提取数值型指标
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics_dict[f"{item}_{key}"] = value
            
            if metrics_dict:
                viz_path = os.path.join(eval_dir, 'metrics_comparison.png')
                plot_metrics_comparison(metrics_dict, viz_path)
                print(f"生成指标对比图: {os.path.basename(viz_path)}")
        except Exception as e:
            print(f"生成指标对比图失败: {str(e)}")
        
        # 8. 保存输入文件
        print("\n=== 8. 保存输入文件 ===")
        save_input_files(eval_dir, args)
        print("输入文件已保存到评估目录")
        
        # 9. 生成可视化总结
        print("\n=== 9. 生成可视化总结 ===")
        summary_path = create_visualization_summary(eval_dir, report)
        print(f"可视化总结: {os.path.basename(summary_path)}")
        
        # 10. 保存最终报告
        print("\n=== 10. 保存最终报告 ===")
        save_report(report, report_path)
        print(f"最终报告已保存: {report_path}")
        
        # 11. 输出评估结果摘要
        print("\n=== 评估结果摘要 ===")
        print(f"评估目录: {eval_dir}")
        print(f"成功执行的评估项: {len([k for k, v in report['evaluation_results'].items() if 'error' not in v])}")
        print(f"失败的评估项: {len(report.get('errors', {}))}")
        print(f"生成的可视化文件: {len(report.get('visualizations', {}))}")
        
        if report.get('errors'):
            print("\n错误信息:")
            for item, error in report['errors'].items():
                print(f"  - {item}: {error}")
        
        print(f"\n评估完成！详细结果请查看: {report_path}")
        
    except Exception as e:
        print(f"评估过程中发生严重错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 