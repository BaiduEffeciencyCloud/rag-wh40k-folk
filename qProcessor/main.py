#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qProcessor模块独立测试入口
用于调试和测试各种query processor实例
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .processorinterface import QueryProcessorInterface
from .straightforward import StraightforwardProcessor
from .expander import ExpanderQueryProcessor
from .cot import COTProcessor
from processorfactory import QueryProcessorFactory

def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_query_results(processor_name: str, original_query: str, results):
    """打印查询处理结果"""
    print(f"\n📝 处理器: {processor_name}")
    print(f"🔍 原始查询: {original_query}")
    
    # 处理单个结果或结果列表
    if isinstance(results, dict):
        results = [results]
    
    print(f"📊 处理结果数量: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        if isinstance(result, dict):
            print(f"查询文本: {result.get('text', result.get('query', 'N/A'))}")
            print(f"查询类型: {result.get('type', 'N/A')}")
            print(f"处理状态: {result.get('processed', 'N/A')}")
            if 'entities' in result and result['entities']:
                print(f"识别实体: {result['entities']}")
            if 'relations' in result and result['relations']:
                print(f"识别关系: {result['relations']}")
            if 'metadata' in result and result['metadata']:
                print(f"元数据: {json.dumps(result['metadata'], ensure_ascii=False, indent=2)}")
        else:
            print(f"结果内容: {result}")

def test_processor(processor: QueryProcessorInterface, test_queries: List[str]):
    """测试单个处理器"""
    processor_name = processor.get_type()
    print_separator(f"测试 {processor_name}")
    
    for query in test_queries:
        try:
            results = processor.process(query)
            print_query_results(processor_name, query, results)
        except Exception as e:
            print(f"❌ 处理查询 '{query}' 时出错: {str(e)}")

def test_all_processors(test_queries: List[str]):
    """测试所有处理器"""
    print_separator("qProcessor 模块测试")
    print(f"测试查询数量: {len(test_queries)}")
    
    # 测试StraightforwardProcessor
    straightforward = StraightforwardProcessor()
    test_processor(straightforward, test_queries)
    
    # 测试ExpanderQueryProcessor
    expander = ExpanderQueryProcessor()
    test_processor(expander, test_queries)
    
    # 测试COTProcessor
    cot = COTProcessor()
    test_processor(cot, test_queries)

def test_factory(test_queries: List[str]):
    """测试工厂类"""
    print_separator("QueryProcessorFactory 测试")
    
    # 测试支持的处理器类型
    supported_types = QueryProcessorFactory.get_available_processors()
    print(f"支持的处理器类型: {supported_types}")
    
    # 测试创建各种处理器
    for processor_type in supported_types:
        try:
            processor = QueryProcessorFactory.create_processor(processor_type)
            print(f"\n✅ 成功创建 {processor_type} 处理器")
            
            # 测试第一个查询
            if test_queries:
                results = processor.process(test_queries[0])
                print(f"   处理结果数量: {len(results)}")
        except Exception as e:
            print(f"❌ 创建 {processor_type} 处理器失败: {str(e)}")

def get_test_queries() -> List[str]:
    """获取测试查询列表"""
    return [
        "帝国卫队的武器有哪些？",
        "星际战士的装备和战术",
        "混沌势力的组织结构",
        "战锤40K中的主要种族",
        "太空死灵的科技水平",
        "兽人的战斗风格",
        "灵族的文明特点",
        "钛帝国的扩张历史"
    ]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="qProcessor模块独立测试")
    parser.add_argument('--processor', '-p', type=str, 
                       choices=['straightforward', 'expander', 'cot', 'all', 'factory'],
                       default='all',
                       help='指定要测试的处理器类型')
    parser.add_argument('--query', '-q', type=str,
                       help='指定测试查询（如果不指定则使用默认测试集）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置详细输出
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # 准备测试查询
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = get_test_queries()
    
    print(f"🚀 开始qProcessor模块测试")
    print(f"📋 测试查询数量: {len(test_queries)}")
    
    try:
        if args.processor == 'all':
            test_all_processors(test_queries)
            test_factory(test_queries)
        elif args.processor == 'factory':
            test_factory(test_queries)
        else:
            # 测试单个处理器
            processor = QueryProcessorFactory.create_processor(args.processor)
            test_processor(processor, test_queries)
        
        print_separator("测试完成")
        print("✅ 所有测试执行完毕")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 