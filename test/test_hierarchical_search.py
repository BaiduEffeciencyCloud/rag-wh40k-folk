#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from typing import List, Dict, Any
from pinecone import Pinecone
from openai import OpenAI

# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY, EMBADDING_MODEL

class HierarchicalSearchTester:
    """
    分层检索测试器
    测试通过子query定位到上层目录，然后在小范围内进行向量召回的可行性
    """
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        response = self.client.embeddings.create(
            model=EMBADDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def search_by_hierarchy_filter(self, query: str, hierarchy_filter: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        使用层级过滤进行检索
        
        Args:
            query: 查询文本
            hierarchy_filter: 层级过滤条件，如 {"h2": {"$eq": "战火军阵 (WARHOST)"}}
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=hierarchy_filter
            )
            
            search_results = []
            for match in results.matches:
                text = match.metadata.get('text', '')
                if not text or len(text.strip()) < 10:
                    continue
                search_results.append({
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return search_results
            
        except Exception as e:
            print(f"❌ 层级过滤检索出错: {str(e)}")
            return []
    
    def search_without_filter(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """不使用过滤条件的检索（对比用）"""
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            search_results = []
            for match in results.matches:
                text = match.metadata.get('text', '')
                if not text or len(text.strip()) < 10:
                    continue
                search_results.append({
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return search_results
            
        except Exception as e:
            print(f"❌ 无过滤检索出错: {str(e)}")
            return []
    
    def test_hierarchical_search(self):
        """测试分层检索的可行性"""
        
        # 测试用例
        test_cases = [
            {
                "name": "战火军阵",
                "query": "战火军阵的规则是什么",
                "hierarchy_filter": {"h2": {"$eq": "战火军阵 (WARHOST)"}},
                "expected_content": "战火军阵"
            },
            {
                "name": "凯恩化身",
                "query": "凯恩化身的属性和技能",
                "hierarchy_filter": {"h3": {"$eq": "凯恩化身 AVATAR OF KHAINE"}},
                "expected_content": "凯恩化身"
            },
            {
                "name": "猎鹰坦克",
                "query": "猎鹰坦克的装备和运输能力",
                "hierarchy_filter": {"h3": {"$eq": "猎鹰坦克 FALCON"}},
                "expected_content": "猎鹰坦克"
            }
        ]
        
        print("🔍 分层检索可行性测试")
        print("=" * 80)
        
        for test_case in test_cases:
            print(f"\n📋 测试用例: {test_case['name']}")
            print(f"🔍 查询: {test_case['query']}")
            print(f"🎯 层级过滤: {test_case['hierarchy_filter']}")
            print("-" * 60)
            
            # 1. 使用层级过滤的检索
            print("1️⃣ 使用层级过滤的检索结果:")
            filtered_results = self.search_by_hierarchy_filter(
                test_case['query'], 
                test_case['hierarchy_filter'], 
                top_k=5
            )
            
            if filtered_results:
                print(f"✅ 找到 {len(filtered_results)} 个结果:")
                for i, result in enumerate(filtered_results, 1):
                    print(f"  {i}. 分数: {result['score']:.4f}")
                    text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                    print(f"     内容: {text_preview}")
                    
                    # 检查是否包含预期内容
                    if test_case['expected_content'] in result['text']:
                        print(f"     ✅ 包含预期内容: {test_case['expected_content']}")
                    else:
                        print(f"     ❌ 未包含预期内容: {test_case['expected_content']}")
                    print()
            else:
                print("❌ 没有找到结果")
            
            # 2. 不使用过滤的检索（对比）
            print("2️⃣ 不使用过滤的检索结果（对比）:")
            unfiltered_results = self.search_without_filter(test_case['query'], top_k=5)
            
            if unfiltered_results:
                print(f"✅ 找到 {len(unfiltered_results)} 个结果:")
                for i, result in enumerate(unfiltered_results, 1):
                    print(f"  {i}. 分数: {result['score']:.4f}")
                    text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                    print(f"     内容: {text_preview}")
                    
                    # 检查是否包含预期内容
                    if test_case['expected_content'] in result['text']:
                        print(f"     ✅ 包含预期内容: {test_case['expected_content']}")
                    else:
                        print(f"     ❌ 未包含预期内容: {test_case['expected_content']}")
                    print()
            else:
                print("❌ 没有找到结果")
            
            # 3. 分析结果
            print("3️⃣ 结果分析:")
            if filtered_results and unfiltered_results:
                filtered_avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
                unfiltered_avg_score = sum(r['score'] for r in unfiltered_results) / len(unfiltered_results)
                
                print(f"   层级过滤平均分数: {filtered_avg_score:.4f}")
                print(f"   无过滤平均分数: {unfiltered_avg_score:.4f}")
                
                if filtered_avg_score > unfiltered_avg_score:
                    print("   ✅ 层级过滤提高了检索精度")
                else:
                    print("   ⚠️ 层级过滤未显著提高检索精度")
                
                # 检查相关性
                filtered_relevant = sum(1 for r in filtered_results if test_case['expected_content'] in r['text'])
                unfiltered_relevant = sum(1 for r in unfiltered_results if test_case['expected_content'] in r['text'])
                
                print(f"   层级过滤相关结果: {filtered_relevant}/{len(filtered_results)}")
                print(f"   无过滤相关结果: {unfiltered_relevant}/{len(unfiltered_results)}")
                
                if filtered_relevant > unfiltered_relevant:
                    print("   ✅ 层级过滤提高了相关性")
                elif filtered_relevant == unfiltered_relevant:
                    print("   ⚠️ 层级过滤相关性相同")
                else:
                    print("   ❌ 层级过滤降低了相关性")
            
            print("=" * 80)
    
    def test_hierarchy_structure(self):
        """测试层级结构"""
        print("\n🔍 测试层级结构")
        print("=" * 60)
        
        # 获取一些样本数据来分析层级结构
        try:
            # 使用一个通用向量进行采样
            sample_vector = [0.1] * 1536  # text-embedding-3-small的维度
            sample_results = self.index.query(
                vector=sample_vector,
                top_k=20,
                include_metadata=True
            )
            
            hierarchy_stats = {
                'h1': set(),
                'h2': set(),
                'h3': set(),
                'h4': set(),
                'h5': set(),
                'h6': set()
            }
            
            print("📊 层级结构统计:")
            for match in sample_results.matches:
                metadata = match.metadata
                for i in range(1, 7):
                    h_key = f'h{i}'
                    if h_key in metadata and metadata[h_key]:
                        hierarchy_stats[h_key].add(metadata[h_key])
            
            for level, titles in hierarchy_stats.items():
                if titles:
                    print(f"  {level}: {len(titles)} 个唯一标题")
                    # 显示前5个标题作为示例
                    sample_titles = list(titles)[:5]
                    for title in sample_titles:
                        print(f"    - {title}")
                    if len(titles) > 5:
                        print(f"    ... 还有 {len(titles) - 5} 个")
                    print()
            
        except Exception as e:
            print(f"❌ 分析层级结构时出错: {e}")

def main():
    """主函数"""
    try:
        tester = HierarchicalSearchTester()
        
        # 测试层级结构
        tester.test_hierarchy_structure()
        
        # 测试分层检索
        tester.test_hierarchical_search()
        
        print("\n✅ 分层检索可行性测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 