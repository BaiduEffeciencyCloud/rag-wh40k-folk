import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from openai import OpenAI
from neo4j import GraphDatabase
import sys
import argparse

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import OPENAI_API_KEY, LLM_MODEL
from dataupload.data_vector_v3 import SemanticDocumentChunker

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体数据类"""
    text: str
    type: str  # unit, skill, weapon, phase, rule
    confidence: float
    aliases: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.properties is None:
            self.properties = {}

@dataclass
class Relationship:
    """关系数据类"""
    source_entity: Entity
    target_entity: Entity
    relation_type: str  # HAS_SKILL, ACTIVATES_IN, etc.
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class KnowledgeGraphManager:
    """
    知识图谱管理器 (单一数据库，通过前缀和属性实现环境隔离)
    """
    
    def __init__(self, 
                 neo4j_uri: str = None,
                 username: str = None,
                 password: str = None,
                 openai_api_key: str = None,
                 environment: str = "production",
                 patterns_config_path: str = None):
        """
        初始化知识图谱管理器
        """
        self.environment = environment.lower()
        
        # 根据环境决定连接信息和前缀
        if self.environment == "test":
            self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
            self.username = username or os.getenv("NEO4J_TEST_USERNAME", "neo4j")
            self.password = password or os.getenv("NEO4J_TEST_PASSWORD", "19811024")
            self.node_prefix = "test_"
            self.relationship_prefix = "test_"
        else: # production or development
            self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
            self.password = password or os.getenv("NEO4J_PASSWORD", "19811024")
            self.node_prefix = ""
            self.relationship_prefix = ""

        self.openai_client = OpenAI(api_key=openai_api_key or OPENAI_API_KEY)
        
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.username, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            logging.info(f"✅ Neo4j连接成功 - 环境: {self.environment}")
        except Exception as e:
            logging.error(f"❌ Neo4j连接失败: {e}")
            self.driver = None
        
        # 从Neo4j数据库动态加载模式和别名
        self._load_patterns_from_db()
        
        logger.info(f"KnowledgeGraphManager初始化完成 - 环境: {self.environment}")
    
    def _load_patterns_from_db(self):
        """从Neo4j数据库动态加载实体模式和别名映射"""
        if not self.driver:
            logging.error("Neo4j驱动未初始化，无法加载模式。")
            self.entity_patterns = {}
            self.alias_mapping = {}
            return

        self.entity_patterns = {}
        self.alias_mapping = {}
        
        logging.info("从Neo4j数据库加载实体模式...")
        with self.driver.session() as session:
            try:
                # 1. 查询所有带有 'name' 属性的节点标签 (除去了_test前缀和一些通用标签)
                labels_result = session.run("""
                    CALL db.labels() YIELD label
                    WHERE NOT label STARTS WITH 'test_' AND NOT label IN ['Entity', 'Resource']
                    RETURN label
                """)
                entity_types = [record["label"] for record in labels_result]

                # 2. 为每个标签(实体类型)查询所有节点名称
                for label in entity_types:
                    # 将标签名转为小写，以用作entity_patterns的键 (e.g., 'Faction' -> 'faction')
                    pattern_key = label.lower().replace('_', '') # 'Sub_faction' -> 'subfaction'
                    
                    query = f"MATCH (n:{label}) RETURN n.name AS name"
                    nodes_result = session.run(query)
                    self.entity_patterns[pattern_key] = [record["name"] for record in nodes_result]
                
                logging.info(f"✅ 成功加载 {len(self.entity_patterns)} 种实体类型，共 {sum(len(v) for v in self.entity_patterns.values())} 个实体。")

                # 3. 查询所有别名关系
                aliases_result = session.run("""
                    MATCH (alias)-[:ALIAS_OF]->(entity)
                    RETURN entity.name AS entity_name, COLLECT(alias.name) AS aliases
                """)
                for record in aliases_result:
                    self.alias_mapping[record["entity_name"]] = record["aliases"]
                
                logging.info(f"✅ 成功加载 {len(self.alias_mapping)} 个实体的别名映射。")

            except Exception as e:
                logging.error(f"❌ 从Neo4j加载模式时发生错误: {e}")
                self.entity_patterns = {}
                self.alias_mapping = {}

    def extract_entities(self, text: str, heading: str = None, hierarchy: Dict[str, str] = None) -> List[Entity]:
        """
        从文本中提取实体（混合模式：规则 + LLM并行，规则优先）
        
        Args:
            text: 文本内容
            heading: 章节标题
            hierarchy: 层级信息
            
        Returns:
            List[Entity]: 提取的实体列表
        """
        # 1. 首先，执行规则提取
        entities_from_rules = self._extract_entities_by_rules(text)
        
        # 2. 然后，让LLM独立地进行提取
        try:
            # 注意：这里不再传递 existing_entities
            entities_from_llm = self.extract_entities_with_llm(text)
            logging.info(f"🤖 LLM 独立发现了 {len(entities_from_llm)} 个实体。")
        except Exception as e:
            logging.error(f"❌ LLM实体提取失败: {e}")
            entities_from_llm = []

        # 3. 合并两个列表（规则提取的结果在前，保证其优先级）
        all_entities = entities_from_rules + entities_from_llm
        
        # 4. 去重和最终处理
        # _deduplicate_entities 会保留先出现的实体，从而实现"规则优先"
        final_entities = self._deduplicate_entities(all_entities)
        
        logging.info(f"📊 实体提取完成，共找到 {len(final_entities)} 个唯一实体。")
        return final_entities

    def extract_entities_with_llm(self, text: str) -> List[Entity]:
        """
        使用LLM从文本中提取所有它能识别的实体。
        采用思维链（Chain of Thought）和少量示例（Few-shot）的Prompt策略。

        Args:
            text: 文本内容
            
        Returns:
            List[Entity]: LLM发现的实体列表
        """
        if not self.openai_client:
            logging.warning("OpenAI 客户端未初始化，跳过LLM实体提取。")
            return []

        entity_types = list(self.entity_patterns.keys())

        prompt = f"""
你是一位精通《战锤40k》背景和规则的专家。你的任务是从给定的文本中，识别出所有潜在的实体。
实体类型包括：{', '.join(entity_types)}。

为了帮助你更好地完成任务，我将给你一些示例，请你学习我的思考方式：

**示例1：**
文本： "在战斗阶段，装备了动力剑的丑角剧团可以使用死亡之舞。"
你的思考过程：
1. "战斗阶段" 是一个明确的游戏阶段，应该标记为 'phase'。
2. "动力剑" 是一个武器，应该标记为 'weapon'。
3. "丑角剧团" 是一个已知的单位，应该标记为 'unit'。
4. "死亡之舞" 听起来像一个特殊能力或技能，应该标记为 'skill'。
输出JSON:
[
  {{"text": "战斗阶段", "type": "phase", "reason": "这是一个标准游戏阶段"}},
  {{"text": "动力剑", "type": "weapon", "reason": "这是一个武器名称"}},
  {{"text": "丑角剧团", "type": "unit", "reason": "这是一个单位名称"}},
  {{"text": "死亡之舞", "type": "skill", "reason": "这是一个技能名称"}}
]

**示例2：**
文本: "火龙武士是阿苏焉尼的一个支派武士。"
你的思考过程:
1. "火焰龙武士" 是一个具体的战士类型，属于一个单位，标记为 'unit'。
2. "阿苏焉尼" 是艾达灵族的一个子阵营，标记为 'sub_faction'。
输出JSON:
[
  {{"text": "火焰龙武士", "type": "unit", "reason": "这是一个具体的单位名称"}},
  {{"text": "阿苏焉尼", "type": "sub_faction", "reason": "这是一个已知的子阵营"}}
]

---
现在，请你按照上面的思考方式，分析下面的文本，并严格按照JSON格式返回结果。如果没有任何可识别的实体，则返回一个空列表 `[]`。

这是需要你分析的文本：
---
{text}
---
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}, # 使用JSON模式确保输出格式
            )
            
            response_content = response.choices[0].message.content
            # LLM可能返回一个包含根键的JSON对象，我们需要提取出列表
            response_data = json.loads(response_content)
            
            # 假设LLM可能返回如 {"new_entities": [...]} 的格式，尝试提取
            if isinstance(response_data, dict):
                # 寻找字典中的第一个值为列表的键
                for key, value in response_data.items():
                    if isinstance(value, list):
                        llm_results = value
                        break
                else:
                    llm_results = []
            elif isinstance(response_data, list):
                llm_results = response_data
            else:
                llm_results = []

            new_entities = []
            for item in llm_results:
                # 验证返回的数据
                if "text" in item and "type" in item and item["type"] in entity_types:
                    new_entities.append(Entity(
                        text=item["text"],
                        type=item["type"],
                        confidence=0.85, # 来自LLM的实体给一个较高的默认置信度
                        aliases=[], # LLM目前不负责提取别名
                        properties={"source": "llm", "reason": item.get("reason", "")}
                    ))
            
            return new_entities

        except Exception as e:
            logging.error(f"调用LLM API或解析其响应时出错: {e}")
            return []

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """对实体列表进行去重"""
        unique_entities = {}
        for entity in entities:
            # 使用实体文本作为唯一标识符
            if entity.text not in unique_entities:
                unique_entities[entity.text] = entity
        return list(unique_entities.values())

    def _extract_entities_by_rules(self, text: str) -> List[Entity]:
        """基于规则提取实体"""
        import re
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                # 对模式进行转义，避免正则表达式特殊字符导致的错误
                escaped_pattern = re.escape(pattern)
                matches = re.finditer(escaped_pattern, text, re.IGNORECASE)
                for match in matches:
                    # 计算置信度
                    confidence = min(len(match.group()) / len(text) * 2, 1.0)
                    
                    # 提取别名
                    aliases = self._extract_aliases(match.group(), entity_type)
                    
                    # 提取属性
                    properties = self._extract_properties(text, match.group(), entity_type)
                    
                    entities.append(Entity(
                        text=match.group(),
                        type=entity_type,
                        confidence=confidence,
                        aliases=aliases,
                        properties=properties
                    ))
        
        return entities
    
    def _extract_aliases(self, entity_text: str, entity_type: str) -> List[str]:
        """从加载的配置中提取实体别名"""
        # 直接使用加载的 alias_mapping
        return self.alias_mapping.get(entity_text, [])
    
    def _extract_properties(self, text: str, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """提取实体属性"""
        import re
        properties = {}
        
        # 根据实体类型提取不同属性
        if entity_type == 'weapon':
            # 提取武器属性
            range_match = re.search(r'(\d+)寸', text)
            if range_match:
                properties['range'] = int(range_match.group(1))
            
            attacks_match = re.search(r'攻击力[：:]\s*(\d+)', text)
            if attacks_match:
                properties['attacks'] = int(attacks_match.group(1))
            
            strength_match = re.search(r'力量[：:]\s*(\d+)', text)
            if strength_match:
                properties['strength'] = int(strength_match.group(1))
        
        elif entity_type == 'unit':
            # 提取单位属性
            points_match = re.search(r'(\d+)分', text)
            if points_match:
                properties['points'] = int(points_match.group(1))
            
            type_match = re.search(r'(步兵|骑兵|载具)', text)
            if type_match:
                properties['unit_type'] = type_match.group(1)
        
        elif entity_type == 'skill':
            # 提取技能属性
            phase_match = re.search(r'(近战|射击|移动|冲锋)阶段', text)
            if phase_match:
                properties['phase'] = phase_match.group(1) + '阶段'
        
        return properties
    
    def extract_relationships(self, text: str, entities: List[Entity], content_type: str = None) -> List[Relationship]:
        """
        提取实体间关系
        
        Args:
            text: 文本内容
            entities: 实体列表
            content_type: 内容类型
            
        Returns:
            List[Relationship]: 关系列表
        """
        try:
            relationships = []
            
            # 基于规则的关系提取
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j:
                        relation_type = self._determine_relationship_type(text, entity1, entity2)
                        if relation_type:
                            source, target = self._determine_relationship_direction(text, entity1, entity2, relation_type)
                            properties = self._extract_relationship_properties(text, entity1, entity2, relation_type)
                            
                            relationships.append(Relationship(
                                source_entity=source,
                                target_entity=target,
                                relation_type=relation_type,
                                properties=properties
                            ))
            
            logger.info(f"提取到 {len(relationships)} 个关系")
            return relationships
            
        except Exception as e:
            logger.error(f"关系提取失败: {e}")
            return []
    
    def _determine_relationship_type(self, text: str, entity1: Entity, entity2: Entity) -> Optional[str]:
        """确定关系类型"""
        import re
        
        # 查找两个实体之间的文本
        entity1_pos = text.find(entity1.text)
        entity2_pos = text.find(entity2.text)
        
        if entity1_pos == -1 or entity2_pos == -1:
            return None
        
        # 获取两个实体之间的文本片段
        start = min(entity1_pos, entity2_pos)
        end = max(entity1_pos + len(entity1.text), entity2_pos + len(entity2.text))
        context = text[start:end]
        
        # 根据实体类型和上下文判断关系
        if entity1.type == 'unit' and entity2.type == 'skill':
            if re.search(r'拥有|获得|使用|has|gain|use', context):
                return 'HAS_SKILL'
        
        elif entity1.type == 'skill' and entity2.type == 'phase':
            if re.search(r'在.*阶段|in.*phase|当.*时|when', context):
                return 'ACTIVATES_IN'
        
        elif entity1.type == 'unit' and entity2.type == 'weapon':
            if re.search(r'装备|携带|equipped|carry', context):
                return 'EQUIPPED_WITH'
        
        elif entity1.type == 'skill' and entity2.type in ['unit', 'weapon']:
            if re.search(r'提供|给予|造成|provides|gives|deals', context):
                return 'PROVIDES_EFFECT'
        
        elif entity1.type == 'rule' and entity2.type == 'phase':
            if re.search(r'适用于|在.*中|applies in|during', context):
                return 'APPLIES_IN'
        
        return None
    
    def _determine_relationship_direction(self, text: str, entity1: Entity, entity2: Entity, relation_type: str) -> tuple:
        """确定关系的方向"""
        # 根据关系类型确定源实体和目标实体
        if relation_type == 'HAS_SKILL':
            return (entity1, entity2) if entity1.type == 'unit' else (entity2, entity1)
        elif relation_type == 'ACTIVATES_IN':
            return (entity1, entity2) if entity1.type == 'skill' else (entity2, entity1)
        elif relation_type == 'EQUIPPED_WITH':
            return (entity1, entity2) if entity1.type == 'unit' else (entity2, entity1)
        elif relation_type == 'PROVIDES_EFFECT':
            return (entity1, entity2) if entity1.type == 'skill' else (entity2, entity1)
        elif relation_type == 'APPLIES_IN':
            return (entity1, entity2) if entity1.type == 'rule' else (entity2, entity1)
        else:
            return (entity1, entity2)
    
    def _extract_relationship_properties(self, text: str, entity1: Entity, entity2: Entity, relation_type: str) -> Dict[str, Any]:
        """提取关系属性"""
        import re
        properties = {}
        
        # 查找两个实体之间的文本
        entity1_pos = text.find(entity1.text)
        entity2_pos = text.find(entity2.text)
        
        if entity1_pos != -1 and entity2_pos != -1:
            start = min(entity1_pos, entity2_pos)
            end = max(entity1_pos + len(entity1.text), entity2_pos + len(entity2.text))
            context = text[start:end]
            
            # 根据关系类型提取属性
            if relation_type == 'HAS_SKILL':
                # 提取技能使用条件
                condition_match = re.search(r'当.*时|if.*then|when', context)
                if condition_match:
                    properties['condition'] = condition_match.group()
            
            elif relation_type == 'EQUIPPED_WITH':
                # 提取装备数量
                quantity_match = re.search(r'(\d+)个|(\d+)把|(\d+)件', context)
                if quantity_match:
                    properties['quantity'] = int(quantity_match.group(1) or quantity_match.group(2) or quantity_match.group(3))
        
        return properties
    
    def update_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship], metadata: Dict[str, Any] = None):
        """
        更新知识图谱
        
        Args:
            entities: 实体列表
            relationships: 关系列表
            metadata: 元数据
        """
        if not self.driver:
            logging.error("Neo4j连接不可用")
            return
        
        try:
            with self.driver.session() as session:
                for entity in entities:
                    self._create_entity_node(session, entity, metadata)
                for relationship in relationships:
                    self._create_relationship(session, relationship, metadata)
                self._create_alias_relationships(session, entities)
        except Exception as e:
            logging.error(f"更新知识图谱失败: {e}")
    
    def _create_entity_node(self, session, entity: Entity, metadata: Dict[str, Any] = None):
        """创建实体节点"""
        node_label = f"{self.node_prefix}{entity.type.capitalize()}"
        properties = {
            'name': entity.text, 'type': entity.type, 'confidence': entity.confidence,
            'aliases': entity.aliases, 'environment': self.environment, **entity.properties
        }
        if metadata:
            properties.update({k: metadata.get(k, '') for k in ['source_file', 'faction', 'content_type', 'created_at', 'document_name', 'faction_name']})
        
        cypher = f"""
        MERGE (e:{node_label} {{name: $name, environment: $environment}})
        SET e += $properties
        """
        session.run(cypher, name=entity.text, environment=self.environment, properties=properties)
    
    def _create_relationship(self, session, relationship: Relationship, metadata: Dict[str, Any] = None):
        """创建关系"""
        relationship_type = f"{self.relationship_prefix}{relationship.relation_type}"
        properties = {'environment': self.environment, **relationship.properties}
        if metadata:
            properties.update({k: metadata.get(k, '') for k in ['source_file', 'faction', 'created_at', 'document_name', 'faction_name']})

        source_node_label = f"{self.node_prefix}{relationship.source_entity.type.capitalize()}"
        target_node_label = f"{self.node_prefix}{relationship.target_entity.type.capitalize()}"

        cypher = f"""
        MATCH (a:{source_node_label} {{name: $source_name, environment: $environment}})
        MATCH (b:{target_node_label} {{name: $target_name, environment: $environment}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        """
        session.run(cypher, 
                   source_name=relationship.source_entity.text,
                   target_name=relationship.target_entity.text,
                   environment=self.environment,
                   properties=properties)
    
    def _create_alias_relationships(self, session, entities: List[Entity]):
        """创建别名关系"""
        alias_relationship_type = f"{self.relationship_prefix}ALIAS_OF"
        for entity in entities:
            if entity.aliases:
                for alias in entity.aliases:
                    cypher = f"""
                    MATCH (a {{name: $name1, environment: $environment}})
                    MATCH (b {{name: $alias, environment: $environment}})
                    WHERE id(a) <> id(b)
                    MERGE (a)-[r:{alias_relationship_type}]->(b)
                    SET r.environment = $environment
                    """
                    session.run(cypher, name1=entity.text, alias=alias, environment=self.environment)
    
    def query_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        查询知识图谱中的实体
        
        Args:
            query: 查询文本
            
        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        if not self.driver: return []
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (e)
                WHERE (e.name CONTAINS $query OR $query IN e.aliases) AND e.environment = $environment
                RETURN e LIMIT 10
                """
                result = session.run(cypher, query=query, environment=self.environment)
                return [record['e'] for record in result]
        except Exception as e:
            logging.error(f"查询实体失败: {e}")
            return []
    
    def expand_query_with_kg(self, query: str) -> List[str]:
        """
        基于知识图谱扩展查询
        
        Args:
            query: 原始查询
            
        Returns:
            List[str]: 扩展后的查询列表
        """
        try:
            # 1. 提取查询中的实体
            entities = self.query_entities(query)
            
            # 2. 基于实体关系生成扩展查询
            expanded_queries = [query]  # 保留原始查询
            
            for entity in entities:
                entity_name = entity['name']
                
                # 获取实体的关系
                relationships = self.query_relationships(entity_name)
                
                for rel in relationships:
                    target_name = rel['target']['name']
                    relationship_type = rel['relationship']['type']
                    
                    # 移除环境前缀以获取真实的关系类型
                    if self.relationship_prefix and relationship_type.startswith(self.relationship_prefix):
                        relationship_type = relationship_type[len(self.relationship_prefix):]
                    
                    # 生成相关查询
                    if relationship_type == 'HAS_SKILL':
                        expanded_queries.append(f"{entity_name}的{target_name}技能")
                        expanded_queries.append(f"{target_name}技能如何使用")
                    
                    elif relationship_type == 'EQUIPPED_WITH':
                        expanded_queries.append(f"{entity_name}装备的{target_name}")
                        expanded_queries.append(f"{target_name}武器属性")
                    
                    elif relationship_type == 'ACTIVATES_IN':
                        expanded_queries.append(f"{entity_name}在{target_name}中的效果")
                
                # 添加别名查询
                aliases = self.get_entity_aliases(entity_name)
                for alias in aliases:
                    if alias != entity_name:
                        expanded_queries.append(query.replace(entity_name, alias))
            
            # 去重
            unique_queries = list(set(expanded_queries))
            
            logger.info(f"查询扩展：{query} -> {len(unique_queries)} 个扩展查询")
            return unique_queries
            
        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]
    
    def query_relationships(self, entity_name: str, relation_type: str = None) -> List[Dict[str, Any]]:
        """
        查询实体的关系
        
        Args:
            entity_name: 实体名称
            relation_type: 关系类型（可选）
            
        Returns:
            List[Dict[str, Any]]: 关系查询结果
        """
        if not self.driver:
            logger.error("Neo4j连接不可用")
            return []
        
        try:
            with self.driver.session() as session:
                # 构建关系类型查询条件
                if relation_type:
                    # 添加环境前缀
                    prefixed_relation_type = f"{self.relationship_prefix}{relation_type}"
                    relationship_condition = f"r:{prefixed_relation_type}"
                else:
                    # 查询所有关系，但只限于当前环境
                    relationship_condition = "r"
                
                cypher = f"""
                MATCH (a)-[{relationship_condition}]->(b)
                WHERE (a.name = $entity_name OR $entity_name IN a.aliases)
                AND a.environment = $environment AND b.environment = $environment
                RETURN a, r, b
                """
                
                result = session.run(cypher, entity_name=entity_name, environment=self.environment)
                relationships = []
                
                for record in result:
                    relationships.append({
                        'source': record['a'],
                        'relationship': record['r'],
                        'target': record['b']
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"查询关系失败: {e}")
            return []
    
    def get_entity_aliases(self, entity_name: str) -> List[str]:
        """
        获取实体的所有别名
        
        Args:
            entity_name: 实体名称
            
        Returns:
            List[str]: 别名列表
        """
        if not self.driver:
            logger.error("Neo4j连接不可用")
            return []
        
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (e)
                WHERE (e.name = $entity_name OR $entity_name IN e.aliases)
                AND e.environment = $environment
                RETURN e.aliases
                """
                
                result = session.run(cypher, entity_name=entity_name, environment=self.environment)
                record = result.single()
                
                if record:
                    return record['e.aliases'] or []
                return []
                
        except Exception as e:
            logger.error(f"获取实体别名失败: {e}")
            return []
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def analyze_document(self, file_path: str, limit: Optional[int] = None):
        """
        分析单个文档文件，通过分块提取并打印所有识别出的实体。
        这是一个用于调试和优化的辅助方法。

        Args:
            file_path: 要分析的文档的路径。
            limit: (可选) 限制分析的块数，用于快速测试。
        """
        logging.info(f"🚀 开始分析文档: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logging.error(f"❌ 文件未找到: {file_path}")
            return
        except Exception as e:
            logging.error(f"❌ 读取文件时出错: {e}")
            return

        # 1. 使用 SemanticDocumentChunker 进行分块
        document_name = os.path.basename(file_path).split('.')[0]
        chunker = SemanticDocumentChunker(document_name=document_name)
        chunks = chunker.chunk_text(content)
        
        if limit:
            chunks = chunks[:limit]
            logging.info(f"📄 分析被限制为前 {limit} 个块。")
        else:
            logging.info(f"📄 文档被拆分为 {len(chunks)} 个块进行分析。")

        # 2. 遍历每个块，提取实体
        all_entities = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            
            logging.info(f"--- 正在分析块 {i+1}/{len(chunks)} ---")
            entities_in_chunk = self.extract_entities(chunk_text)
            all_entities.extend(entities_in_chunk)
            # 可选：如果想看每个块的结果，可以取消下面的注释
            # if entities_in_chunk:
            #     print(f"  在块 {i+1} 中找到: {[e.text for e in entities_in_chunk]}")

        # 3. 对所有找到的实体进行去重和整理
        final_entities = self._deduplicate_entities(all_entities)

        if not final_entities:
            logging.warning("在文档中没有找到任何实体。")
            return

        print("\n" + "="*50)
        print(f"📄 在《{os.path.basename(file_path)}》中找到的实体分析结果 (共 {len(final_entities)} 个唯一实体)")
        print("="*50)
        
        entities_by_type = {}
        for entity in final_entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
            
        for entity_type, entity_list in sorted(entities_by_type.items()):
            print(f"\n--- {entity_type.capitalize()} ({len(entity_list)}个) ---")
            entity_texts = [f"{e.text} ({e.properties.get('source', 'rule')})" for e in entity_list]
            print(", ".join(sorted(list(set(entity_texts))))) # 再次去重以防万一

        print("\n" + "="*50)
        logging.info(f"✅ 分析完成。")


# 当该脚本被直接执行时
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从文档中提取实体并进行分析。")
    parser.add_argument("file_path", help="要分析的文档的路径。")
    parser.add_argument("--limit", type=int, default=None, help="（可选）限制分析前N个文本块，用于快速调试。")
    
    args = parser.parse_args()

    # 初始化管理器
    kg_manager = KnowledgeGraphManager()

    # 使用 limit 参数调用分析方法
    kg_manager.analyze_document(args.file_path, limit=args.limit)

    # 关闭连接
    kg_manager.close() 