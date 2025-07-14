#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估工具函数模块
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='RAG检索评估工具')
    
    # 必需参数
    parser.add_argument('--query', type=str, required=True, 
                       help='Query文件路径')
    
    # 可选参数
    parser.add_argument('--gold', type=str, default=None,
                       help='Gold标准答案文件路径（可选）')
    parser.add_argument('--topk', type=int, default=10,
                       help='Top-K评估参数，默认10')
    parser.add_argument('--qp', type=str, default='straightforward',
                       help='Query处理模式：straightforward/expand/cot')
    parser.add_argument('--se', type=str, default='dense',
                       help='检索引擎类型：dense/sparse/hybrid')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录路径（可选，默认自动生成）')
    
    return parser.parse_args()


def decide_eval_items(args: argparse.Namespace) -> List[str]:
    """
    根据参数决定执行哪些评估项（使用评估器注册表）
    
    Args:
        args: 解析后的参数
        
    Returns:
        List[str]: 需要执行的评估项列表
    """
    from .evaluators import evaluator_registry
    
    available_evaluators = evaluator_registry.get_evaluators_for_args(args)
    return [evaluator.name for evaluator in available_evaluators]


def load_queries(file_path_or_query):
    """
    加载query文件或直接处理一句query
    Args:
        file_path_or_query: 文件路径（.json结尾）或一句自然语言query
    Returns:
        List[str]: query列表
    """
    import os
    import json
    # 如果是.json结尾且文件存在，按文件处理
    if isinstance(file_path_or_query, str) and file_path_or_query.strip().endswith('.json') and os.path.exists(file_path_or_query):
        with open(file_path_or_query, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 支持列表或字典格式
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        else:
            raise ValueError(f"不支持的query文件格式: {file_path_or_query}")
    # 否则直接包装成列表
    elif isinstance(file_path_or_query, str):
        return [file_path_or_query]
    else:
        raise ValueError("query参数必须为字符串或json文件路径")


def load_gold(file_path: str) -> Dict[str, List[str]]:
    """
    加载gold标准答案文件
    
    Args:
        file_path: gold文件路径
        
    Returns:
        Dict[str, List[str]]: gold标准答案字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Gold文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Gold文件格式错误，应为字典格式: {type(data)}")
    
    return data


def create_eval_dir() -> Tuple[str, str]:
    """
    创建评估目录和报告文件路径
    
    Returns:
        Tuple[str, str]: (评估目录路径, 报告文件路径)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join('evaltool', 'dense_search', timestamp)
    
    # 确保目录存在
    os.makedirs(eval_dir, exist_ok=True)
    
    report_path = os.path.join(eval_dir, 'report.json')
    
    return eval_dir, report_path


def create_empty_report(args: argparse.Namespace, eval_items: List[str]) -> Dict:
    """
    创建空白报告结构
    
    Args:
        args: 解析后的参数
        eval_items: 评估项列表
        
    Returns:
        Dict: 空白报告结构
    """
    return {
        'report_info': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'generated_at': datetime.now().isoformat(),
            'tool_version': '1.0.0'
        },
        'configuration': {
            'query_file': args.query,
            'gold_file': args.gold,
            'top_k': args.topk,
            'qp_type': args.qp,
            'se_type': args.se
        },
        'evaluation_items': eval_items,
        'evaluation_results': {},
        'visualizations': {},
        'input_files': {},
        'errors': {}
    }


def append_evaluation_result(report: Dict, eval_item: str, result: Dict, 
                           visualization_path: Optional[str] = None) -> None:
    """
    将评估结果追加到报告中
    
    Args:
        report: 报告字典
        eval_item: 评估项名称
        result: 评估结果
        visualization_path: 可视化文件路径（可选）
    """
    # 追加评估结果
    report['evaluation_results'][eval_item] = result
    
    # 追加可视化信息
    if visualization_path:
        report['visualizations'][eval_item] = {
            'file_path': os.path.basename(visualization_path),
            'description': f'{eval_item}可视化结果',
            'generated_at': datetime.now().isoformat()
        }


def append_error(report: Dict, eval_item: str, error_msg: str) -> None:
    """
    将错误信息追加到报告中
    
    Args:
        report: 报告字典
        eval_item: 评估项名称
        error_msg: 错误信息
    """
    report['errors'][eval_item] = error_msg


def save_report(report: Dict, report_path: str) -> None:
    """
    保存最终报告
    
    Args:
        report: 报告字典
        report_path: 报告文件路径
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_input_files(eval_dir: str, args: argparse.Namespace) -> None:
    """
    保存输入文件到评估目录
    
    Args:
        eval_dir: 评估目录
        args: 解析后的参数
    """
    # 复制query文件
    if os.path.exists(args.query):
        import shutil
        query_copy = os.path.join(eval_dir, 'input_query.json')
        shutil.copy2(args.query, query_copy)
    
    # 复制gold文件
    if args.gold and os.path.exists(args.gold):
        import shutil
        gold_copy = os.path.join(eval_dir, 'input_gold.json')
        shutil.copy2(args.gold, gold_copy) 