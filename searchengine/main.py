#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
searchengine模块独立测试入口
用于调试和测试dense/hybrid等检索引擎
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchengine.dense_search import DenseSearchEngine
from searchengine.hybrid_search import HybridSearchEngine
from searchengine.search_factory import SearchEngineFactory

def print_separator(title: str):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_search_results(engine_type: str, query: str, results: List[Dict[str, Any]]):
    print(f"\n🔎 检索引擎: {engine_type}")
    print(f"📝 查询: {query}")
    print(f"📊 结果数量: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(f"内容: {result.get('text', '')[:100]}")
        print(f"分数: {result.get('score', 'N/A')}")
        print(f"元数据: {json.dumps(result.get('metadata', {}), ensure_ascii=False, indent=2)}")
        if 'kg_used' in result:
            print(f"KG过滤: {result['kg_used']}")

def main():
    parser = argparse.ArgumentParser(description="searchengine模块独立测试")
    parser.add_argument('--engine', '-e', type=str, choices=['dense', 'hybrid'], default='hybrid', help='选择检索引擎类型')
    parser.add_argument('--query', '-q', type=str, required=True, help='检索query')
    parser.add_argument('--kg', type=str, help='KG实体（逗号分隔）')
    parser.add_argument('--topk', type=int, default=5, help='返回结果数量')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出模式')
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    print_separator("searchengine模块测试")
    print(f"引擎类型: {args.engine}")
    print(f"检索query: {args.query}")
    if args.kg:
        print(f"KG实体: {args.kg}")

    # 构造KG信息
    kg_info = None
    if args.kg:
        kg_info = {'entities': [e.strip() for e in args.kg.split(',') if e.strip()]}

    # 创建引擎
    engine = SearchEngineFactory.create(args.engine)

    # 构造查询对象
    query_obj = {'text': args.query}
    
    # 检索
    if args.engine == 'hybrid' and kg_info:
        results = engine.search_with_kg(query_obj, kg_info, top_k=args.topk)
    else:
        results = engine.search(query_obj, top_k=args.topk)

    print_search_results(args.engine, args.query, results)
    print_separator("测试完成")
    print("✅ 检索测试执行完毕")

if __name__ == "__main__":
    main() 