#!/usr/bin/env python3
"""
测试配置文件是否正确加载敏感信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    NEO4J_TEST_URI, NEO4J_TEST_USERNAME, NEO4J_TEST_PASSWORD,
    EMBADDING_MODEL, LLM_MODEL, RERANK_MODEL
)

def test_sensitive_config():
    """测试敏感信息配置"""
    print("=== 敏感信息配置测试 ===")
    
    # 检查OpenAI配置
    print(f"OpenAI API Key: {'已设置' if OPENAI_API_KEY else '未设置'}")
    
    # 检查Pinecone配置
    print(f"Pinecone API Key: {'已设置' if PINECONE_API_KEY else '未设置'}")
    print(f"Pinecone Index: {'已设置' if PINECONE_INDEX else '未设置'}")
    
    # 检查Neo4j配置
    print(f"Neo4j URI: {'已设置' if NEO4J_URI else '未设置'}")
    print(f"Neo4j Username: {'已设置' if NEO4J_USERNAME else '未设置'}")
    print(f"Neo4j Password: {'已设置' if NEO4J_PASSWORD else '未设置'}")
    
    # 检查Neo4j测试配置
    print(f"Neo4j Test URI: {'已设置' if NEO4J_TEST_URI else '未设置'}")
    print(f"Neo4j Test Username: {'已设置' if NEO4J_TEST_USERNAME else '未设置'}")
    print(f"Neo4j Test Password: {'已设置' if NEO4J_TEST_PASSWORD else '未设置'}")
    
    # 检查模型配置
    print(f"Embedding Model: {'已设置' if EMBADDING_MODEL else '未设置'}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Rerank Model: {RERANK_MODEL}")
    
    print("\n=== 配置验证结果 ===")
    
    # 验证必需的配置
    required_configs = {
        "OpenAI API Key": OPENAI_API_KEY,
        "Pinecone API Key": PINECONE_API_KEY,
        "Pinecone Index": PINECONE_INDEX,
        "Neo4j URI": NEO4J_URI,
        "Neo4j Username": NEO4J_USERNAME,
        "Neo4j Password": NEO4J_PASSWORD,
    }
    
    missing_configs = [name for name, value in required_configs.items() if not value]
    
    if missing_configs:
        print("❌ 以下必需配置未设置:")
        for config in missing_configs:
            print(f"   - {config}")
        print("\n请检查 .env 文件并设置这些配置项。")
        return False
    else:
        print("✅ 所有必需配置已正确设置")
        return True

def test_optional_config():
    """测试可选配置"""
    print("\n=== 可选配置测试 ===")
    
    optional_configs = {
        "Neo4j Test URI": NEO4J_TEST_URI,
        "Neo4j Test Username": NEO4J_TEST_USERNAME,
        "Neo4j Test Password": NEO4J_TEST_PASSWORD,
        "Embedding Model": EMBADDING_MODEL,
    }
    
    for name, value in optional_configs.items():
        status = "已设置" if value else "未设置（使用默认值）"
        print(f"{name}: {status}")

if __name__ == "__main__":
    print("开始测试配置文件...\n")
    
    # 测试敏感信息配置
    success = test_sensitive_config()
    
    # 测试可选配置
    test_optional_config()
    
    print(f"\n测试完成！{'✅ 配置正确' if success else '❌ 配置有问题'}") 