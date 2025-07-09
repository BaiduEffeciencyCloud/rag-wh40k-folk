#!/usr/bin/env python3
"""
知识图谱管理器测试脚本
"""

import sys
import os
import unittest.mock
import json
from unittest.mock import MagicMock, patch

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataupload.knowledge_graph_manager import KnowledgeGraphManager, Entity, Relationship
from dataupload.seed_neo4j import Neo4jSeeder
from config import NEO4J_TEST_URI, NEO4J_TEST_USERNAME, NEO4J_TEST_PASSWORD, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def setup_test_environment():
    """初始化并播种测试数据库"""
    print("\n--- 准备测试环境: 初始化并播种测试数据库 ---")
    seeder = Neo4jSeeder(
        uri=NEO4J_TEST_URI,
        user=NEO4J_TEST_USERNAME,
        password=NEO4J_TEST_PASSWORD,
        json_path='DATAUPLOD/knowledge_graph_patterns.json'
    )
    seeder.seed_database()
    seeder.close()
    print("--- 测试环境准备就绪 ---\n")

def get_test_kg_manager():
    """获取一个连接到测试数据库的KG管理器实例"""
    return KnowledgeGraphManager(
        neo4j_uri=NEO4J_TEST_URI,
        username=NEO4J_TEST_USERNAME,
        password=NEO4J_TEST_PASSWORD,
        environment="test"  # 确保使用测试环境的配置和前缀
    )

def test_entity_extraction():
    """测试实体提取功能"""
    print("=== 测试实体提取 ===")
    
    # 初始化知识图谱管理器（连接到测试数据库）
    kg_manager = get_test_kg_manager()
    
    # 测试文本
    test_text = """
    丑角剧团单位在近战阶段开始时，可以选择获得死亡之舞技能。
    死亡之舞技能提供额外的攻击力，允许单位在近战阶段获得+1攻击力。
    丑角剧团装备星镖手枪，射程12寸，攻击力1，力量4。
    """
    
    # 提取实体
    entities = kg_manager.extract_entities(test_text, "丑角剧团规则", {"level1": "单位数据", "level2": "丑角剧团"})
    
    print(f"提取到 {len(entities)} 个实体：")
    for entity in entities:
        print(f"- {entity.text} ({entity.type}) - 置信度: {entity.confidence:.2f}")
        if entity.aliases:
            print(f"  别名: {entity.aliases}")
        if entity.properties:
            print(f"  属性: {entity.properties}")
    
    return entities

def test_hybrid_entity_extraction():
    """测试混合模式（规则+LLM）下的实体提取功能"""
    print("\n=== 测试混合实体提取（规则+LLM） ===")

    test_text = "在指挥阶段, 丑角剧团的首席智库馆长，瓦尔迪斯·萨恩，使用了死亡之舞。"
    
    # 规则应该能识别出: "指令阶段", "丑角剧团", "死亡之舞"
    # 我们模拟LLM识别出一个新的人物实体 "瓦尔迪斯·萨恩" 和一个重复实体 "丑角剧团"
    mock_llm_response = {
        "entities": [
            {
                "text": "瓦尔迪斯·萨恩",
                "type": "unit",
                "reason": "这是一个独特的人物名称，可能是一个单位。"
            },
            {
                "text": "丑角剧团",
                "type": "faction", # LLM可能会做出错误或不同的判断
                "reason": "这是一个已知的阵营名称。"
            }
        ]
    }

    # 模拟 OpenAI API 的响应
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = json.dumps(mock_llm_response)

    # 使用 patch 来模拟 OpenAI 客户端
    with patch('DATAUPLOD.knowledge_graph_manager.OpenAI') as MockOpenAI:
        # 配置模拟实例以返回我们的模拟响应
        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.chat.completions.create.return_value = mock_response

        # 在这个上下文中，KnowledgeGraphManager 将使用我们模拟的客户端，并连接到测试数据库
        kg_manager = get_test_kg_manager()
        
        print(f"测试文本: \"{test_text}\"")
        entities = kg_manager.extract_entities(test_text)

        print(f"\n混合模式提取到 {len(entities)} 个唯一实体：")
        found_entities = {e.text: e.type for e in entities}
        for entity in entities:
            print(f"- {entity.text} ({entity.type}) [来源: {entity.properties.get('source', 'rule')}]")

        # 验证
        print("\n断言验证:")
        expected_entities = {
            "指挥阶段": "phase",
            "丑角剧团": "unit", 
            "死亡之舞": "skill",
            "瓦尔迪斯·萨恩": "unit"
            # "丑角" is not in the test text, so it should not be found.
        }
        
        all_found = True
        # Check that all expected entities are found
        for name, type in expected_entities.items():
            if name in found_entities and found_entities[name] == type:
                print(f"  ✅ 成功找到: {name} ({type})")
            else:
                print(f"  ❌ 未找到或类型错误: {name} ({type})")
                all_found = False
        
        # Check for unexpected entities
        unexpected_found = False
        for name, type in found_entities.items():
            if name not in expected_entities:
                print(f"  ❌ 发现非预期实体: {name} ({type})")
                unexpected_found = True

        if len(entities) == len(expected_entities) and all_found and not unexpected_found:
            print("  ✅ 实体数量正确，且去重成功。")
        else:
            print(f"  ❌ 实体数量或去重逻辑错误。预期 {len(expected_entities)}, 实际 {len(entities)}")

def test_relationship_extraction(entities):
    """测试关系提取功能"""
    print("\n=== 测试关系提取 ===")
    
    kg_manager = get_test_kg_manager()
    
    test_text = """
    丑角剧团单位在近战阶段开始时，可以选择获得死亡之舞技能。
    死亡之舞技能提供额外的攻击力，允许单位在近战阶段获得+1攻击力。
    丑角剧团装备星镖手枪，射程12寸，攻击力1，力量4。
    """
    
    # 提取关系
    relationships = kg_manager.extract_relationships(test_text, entities, "unit")
    
    print(f"提取到 {len(relationships)} 个关系：")
    for rel in relationships:
        print(f"- {rel.source_entity.text} --[{rel.relation_type}]--> {rel.target_entity.text}")
        if rel.properties:
            print(f"  属性: {rel.properties}")
    
    return relationships

def test_query_expansion():
    """测试查询扩展功能（模拟）"""
    print("\n=== 测试查询扩展（模拟） ===")
    
    kg_manager = get_test_kg_manager()
    
    # 模拟查询扩展
    original_query = "丑角剧团的死亡之舞技能如何使用？"
    
    # 这里只是演示，实际需要Neo4j连接
    print(f"原始查询: {original_query}")
    print("扩展查询示例:")
    print("- 丑角剧团的死亡之舞技能")
    print("- 死亡之舞技能如何使用")
    print("- troupe的death dance技能")
    print("- 近战阶段中的死亡之舞效果")

def test_config_import():
    """测试配置导入"""
    print("\n=== 测试配置导入 ===")
    
    try:
        from config import OPENAI_API_KEY, LLM_MODEL
        print(f"✅ 成功导入配置")
        print(f"   LLM_MODEL: {LLM_MODEL}")
        print(f"   OPENAI_API_KEY: {'已设置' if OPENAI_API_KEY else '未设置'}")
    except ImportError as e:
        print(f"❌ 配置导入失败: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

def main():
    """主测试函数"""
    print("🔍 知识图谱管理器测试 (数据库模式)")
    print("=" * 50)
    
    # 1. 准备测试环境
    setup_test_environment()

    # 2. 运行所有测试
    test_config_import()
    entities = test_entity_extraction()
    test_hybrid_entity_extraction()
    if entities:
        test_relationship_extraction(entities)
    test_query_expansion()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")
    print("\n注意：")
    print("- 实体提取和关系提取功能已测试")
    print("- 查询扩展需要Neo4j连接才能完整测试")
    print("- 如需完整测试，请配置Neo4j连接信息")

if __name__ == "__main__":
    main() 