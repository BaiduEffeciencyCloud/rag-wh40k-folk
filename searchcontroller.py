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
# orchestrator导入
from orchestrator.processor import RAGOrchestrator
from aAggregation.aggfactory import AggFactory
from postsearch.factory import PostSearchFactory

def print_separator(title: str):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="RAG SearchController 主流程")
    parser.add_argument('--query', type=str, default='', help='用户查询')
    parser.add_argument('--intent', type=str, default='default', help='用户查询意图')
    parser.add_argument('--qp', type=str, choices=['straight', 'expand', 'cot'], default='straight', help='query processor类型')
    parser.add_argument('--se', type=str, choices=['dense', 'sparse','hybrid'], default='dense', help='search engine类型')
    parser.add_argument('--agg', type=str, default='default', help='答案聚合器类型（预留）')
    parser.add_argument('--topk', type=int, default=5, help='检索返回数量')
    parser.add_argument('--ps', type=str,default='simple',help='对召回 TOP_K的处理')
    parser.add_argument('--rerank', action='store_true', help='是否启用rerank')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出模式')
    parser.add_argument('--db_type', type=str, choices=['opensearch', 'pinecone'], default='opensearch', help='数据库类型')
    parser.add_argument('--embedding_model', type=str, choices=['openai', 'qwen'], default='qwen', help='embedding模型')
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    print_separator("SearchController 主流程")
    print(f"用户查询: {args.query}")
    print(f"QueryProcessor: {args.qp}")
    print(f"SearchEngine: {args.se}")
    print(f"Aggregator: {args.agg}")
    print(f"PostSearch:{args.ps}")
    print(f"Rerank: {args.rerank}")
    print(f"数据库类型: {args.db_type}")
    print(f"embedding模型: {args.embedding_model}")
    print(f"topk: {args.topk}")

    intent = args.intent

    qp_type = args.qp
    query_processor = QueryProcessorFactory.create_processor(qp_type)

    # 2. 组装search engine
    se_type = args.se
    search_engine = SearchEngineFactory.create(se_type)

    # 3. 组装聚合器
    agg_type = args.agg
    aggregator = AggFactory.get_aggregator(agg_type, template=agg_type)

    ps_type = args.ps
    postsearch = PostSearchFactory.create_processor(ps_type)

    db_type = args.db_type
    embedding_model=args.embedding_model
    rerank_enabled = args.rerank
    top_k = args.topk
    # 4. 组装RAGOrchestrator（重构后不再传递db_type和embedding_model）
    orchestrator = RAGOrchestrator(query_processor, search_engine, postsearch, aggregator)

    # 5. 主流程统一调度（通过run方法传递db_type、embedding_model和rerank参数）
    result = orchestrator.run(args.query, intent=intent, top_k=top_k, db_type=db_type, embedding_model=embedding_model, rerank=rerank_enabled)

    print_separator("Orchestrator 聚合结果")
    #print(json.dumps(result, ensure_ascii=False, indent=2))
    print_separator("最终答案")
    print(result.get('final_answer', '无答案'))
    print_separator("流程结束")
    print("✅ 全流程执行完毕")

if __name__ == "__main__":
    main() 

