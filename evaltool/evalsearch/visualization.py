#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_score_distribution(score_list: List[float], output_path: str, 
                           title: str = "分数分布直方图", bins: int = 50) -> None:
    """
    绘制分数分布直方图
    
    Args:
        score_list: 分数列表
        output_path: 输出文件路径
        title: 图表标题
        bins: 直方图分箱数
    """
    if not score_list:
        print("警告：分数列表为空，跳过可视化")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(score_list, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加统计信息
    mean_score = np.mean(score_list)
    std_score = np.std(score_list)
    plt.axvline(mean_score, color='red', linestyle='--', 
                label=f'均值: {mean_score:.3f}')
    plt.axvline(mean_score + std_score, color='orange', linestyle=':', 
                label=f'均值+标准差: {mean_score + std_score:.3f}')
    plt.axvline(mean_score - std_score, color='orange', linestyle=':', 
                label=f'均值-标准差: {mean_score - std_score:.3f}')
    
    plt.title(title)
    plt.xlabel('分数')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_embedding_distribution(embeddings: np.ndarray, output_path: str, 
                               labels: Optional[List[str]] = None, method: str = "tsne",
                               title: str = "Embedding分布可视化") -> None:
    """
    绘制embedding分布图
    
    Args:
        embeddings: embedding向量矩阵
        labels: 标签列表（可选）
        output_path: 输出文件路径
        method: 降维方法
        title: 图表标题
    """
    if embeddings.shape[0] == 0:
        print("警告：embedding矩阵为空，跳过可视化")
        return
    
    # 降维
    from .metrics import DenseRetrievalMetrics
    try:
        coords = DenseRetrievalMetrics.embedding_distribution(embeddings, method=method)
    except Exception as e:
        print(f"降维失败: {e}")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    if labels:
        # 如果有标签，按标签分组绘制
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(coords[mask, 0], coords[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7)
        plt.legend()
    else:
        # 无标签，统一绘制
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c='blue')
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f'{method.upper()} 维度 1')
    plt.ylabel(f'{method.upper()} 维度 2')
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_recall_distribution(recall_list: List[float], output_path: str,
                           title: str = "召回率分布") -> None:
    """
    绘制召回率分布图
    
    Args:
        recall_list: 召回率列表
        output_path: 输出文件路径
        title: 图表标题
    """
    if not recall_list:
        print("警告：召回率列表为空，跳过可视化")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图和密度图
    sns.histplot(recall_list, bins=20, kde=True, alpha=0.7)
    
    # 添加统计信息
    mean_recall = np.mean(recall_list)
    plt.axvline(mean_recall, color='red', linestyle='--', 
                label=f'平均召回率: {mean_recall:.3f}')
    
    plt.title(title)
    plt.xlabel('召回率')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overlap_stats(overlap_stats: Dict[str, Dict[str, int]], output_path: str,
                      title: str = "多路召回重叠统计") -> None:
    """
    绘制多路召回重叠统计图
    
    Args:
        overlap_stats: 重叠统计结果
        output_path: 输出文件路径
        title: 图表标题
    """
    if not overlap_stats:
        print("警告：重叠统计数据为空，跳过可视化")
        return
    
    # 提取数据
    qids = list(overlap_stats.keys())
    overlaps = [overlap_stats[qid]['overlap'] for qid in qids]
    unions = [overlap_stats[qid]['union'] for qid in qids]
    
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 重叠数量分布
    ax1.hist(overlaps, bins=min(20, len(set(overlaps))), alpha=0.7, color='skyblue')
    ax1.set_title('重叠数量分布')
    ax1.set_xlabel('重叠数量')
    ax1.set_ylabel('频次')
    ax1.grid(True, alpha=0.3)
    
    # 重叠率分布
    overlap_rates = [o/u if u > 0 else 0 for o, u in zip(overlaps, unions)]
    ax2.hist(overlap_rates, bins=20, alpha=0.7, color='lightcoral')
    ax2.set_title('重叠率分布')
    ax2.set_xlabel('重叠率 (重叠数量/并集数量)')
    ax2.set_ylabel('频次')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_dict: Dict[str, float], output_path: str,
                           title: str = "评估指标对比") -> None:
    """
    绘制评估指标对比图
    
    Args:
        metrics_dict: 指标字典 {指标名: 指标值}
        output_path: 输出文件路径
        title: 图表标题
    """
    if not metrics_dict:
        print("警告：指标数据为空，跳过可视化")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    bars = plt.bar(metrics_names, metrics_values, alpha=0.7, color='lightgreen')
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('评估指标')
    plt.ylabel('指标值')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_visualization_summary(eval_dir: str, report: Dict) -> str:
    """
    创建可视化总结报告
    
    Args:
        eval_dir: 评估目录
        report: 评估报告
        
    Returns:
        str: 可视化总结文件路径
    """
    summary_path = os.path.join(eval_dir, 'visualization_summary.md')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 评估可视化总结\n\n")
        f.write(f"生成时间: {report['report_info']['generated_at']}\n\n")
        
        # 配置信息
        f.write("## 配置信息\n")
        f.write(f"- Query文件: {report['configuration']['query_file']}\n")
        f.write(f"- Gold文件: {report['configuration']['gold_file']}\n")
        f.write(f"- Top-K: {report['configuration']['top_k']}\n")
        f.write(f"- Query处理模式: {report['configuration']['qp_type']}\n")
        f.write(f"- 检索引擎类型: {report['configuration']['se_type']}\n\n")
        
        # 评估结果
        f.write("## 评估结果\n")
        for item, result in report['evaluation_results'].items():
            f.write(f"### {item}\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    f.write(f"- {key}: {value}\n")
            else:
                f.write(f"- 结果: {result}\n")
            f.write("\n")
        
        # 可视化文件
        f.write("## 可视化文件\n")
        for item, viz_info in report['visualizations'].items():
            f.write(f"- {item}: {viz_info['file_path']}\n")
            f.write(f"  - 描述: {viz_info['description']}\n")
            f.write(f"  - 生成时间: {viz_info['generated_at']}\n\n")
        
        # 错误信息
        if report.get('errors'):
            f.write("## 错误信息\n")
            for item, error in report['errors'].items():
                f.write(f"- {item}: {error}\n")
    
    return summary_path 