#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SearchController: 串联qProcessor和searchengine，实现RAG主流程
"""
import sys
import os
import argparse
import json
from typing import Any, Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# qProcessor导入
from qProcessor.processorfactory import QueryProcessorFactory
# searchengine导入
from searchengine.search_factory import SearchEngineFactory

def print_separator(title: str):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="RAG SearchController 主流程")
    parser.add_argument('--query', type=str, default='', help='用户查询')
    parser.add_argument('--qp', type=str, choices=['straight', 'expand', 'cot'], default='straight', help='query processor类型')
    parser.add_argument('--se', type=str, choices=['dense', 'hybrid'], default='dense', help='search engine类型')
    parser.add_argument('--agg', type=str, default='simple', help='答案聚合器类型（预留）')
    parser.add_argument('--topk', type=int, default=5, help='检索返回数量')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出模式')
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    print_separator("SearchController 主流程")
    print(f"用户查询: {args.query}")
    print(f"QueryProcessor: {args.qp}")
    print(f"SearchEngine: {args.se}")
    print(f"Aggregator: {args.agg}")

    # 1. 组装query processor
    qp_map = {
        'straight': 'straightforward',
        'expand': 'expander',
        'cot': 'cot'
    }
    qp_type = qp_map.get(args.qp, 'straightforward')
    query_processor = QueryProcessorFactory.create_processor(qp_type)

    # 2. 组装search engine
    se_type = args.se
    search_engine = SearchEngineFactory.create(se_type)

    # 3. 查询处理
    processed = query_processor.process(args.query)
    print_separator("QueryProcessor 处理结果")
    print(json.dumps(processed, ensure_ascii=False, indent=2))

    # 4. 检索召回
    # 兼容多query和单query
    if isinstance(processed, list):
        results = []
        for q in processed:
            # q应为dict，包含'text'键
            res = search_engine.search(q, top_k=args.topk)
            results.append({'query': q, 'retrieval': res})
    else:
        results = search_engine.search(processed, top_k=args.topk)

    print_separator("SearchEngine 检索结果")
    if isinstance(results, list) and results and isinstance(results[0], dict) and 'retrieval' in results[0]:
        # 多query
        for i, item in enumerate(results, 1):
            print(f"\n--- 子查询 {i} ---")
            print(f"Query: {item['query'].get('text', item['query'])}")
            print(f"召回数量: {len(item['retrieval'])}")
            for j, doc in enumerate(item['retrieval'], 1):
                print(f"  [{j}] {doc.get('text', '')[:80]} ... (score: {doc.get('score', 'N/A')})")
    else:
        print(f"召回数量: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.get('text', '')[:80]} ... (score: {doc.get('score', 'N/A')})")

    print_separator("流程结束")
    print("✅ 全流程执行完毕")

if __name__ == "__main__":
    main() 

