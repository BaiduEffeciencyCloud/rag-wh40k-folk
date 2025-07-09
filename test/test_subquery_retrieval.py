#!/usr/bin/env python3
"""
测试子查询召回结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_search import VectorSearch
from config import PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY

def test_subquery_retrieval():
    """测试每个子查询的召回结果"""
    
    # 初始化向量搜索
    vector_search = VectorSearch(PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY)
    
    # 子查询列表
    subqueries = [
        "猎鹰坦克是什么？",
        "火力支援是什么？", 
        "猎鹰坦克火力支援技能触发条件",
        "猎鹰坦克火力支援技能对脱离的模型提供的增益",
        "如何最有效地利用猎鹰坦克的火力支援技能"
    ]
    
    print("🔍 子查询召回测试")
    print("=" * 80)
    
    for i, query in enumerate(subqueries, 1):
        print(f"\n📝 子查询 {i}: {query}")
        print("-" * 60)
        
        try:
            # 执行检索
            results = vector_search.search(query, top_k=5)
            
            if not results:
                print("❌ 没有找到相关结果")
                continue
                
            print(f"✅ 找到 {len(results)} 个结果:")
            
            for j, result in enumerate(results, 1):
                score = result.get('score', 0)
                text = result.get('text', '')[:200]  # 只显示前200个字符
                
                print(f"  {j}. 分数: {score:.4f}")
                print(f"     内容: {text}...")
                print()
                
        except Exception as e:
            print(f"❌ 检索出错: {e}")
    
    print("=" * 80)
    print("测试完成")

if __name__ == "__main__":
    test_subquery_retrieval() 