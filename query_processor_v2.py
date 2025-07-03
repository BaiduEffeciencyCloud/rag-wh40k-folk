import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from openai import OpenAI
from vector_search import VectorSearch
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    DEFAULT_TOP_K,
    LLM_MODEL
)

logger = logging.getLogger(__name__)

@dataclass
class QueryEntity:
    """查询实体"""
    text: str
    type: str  # 单位、技能、阶段、地形等
    confidence: float

@dataclass
class QueryIntent:
    """查询意图"""
    intent: str  # 规则查询、伤害计算、技能使用等
    confidence: float
    action_verbs: List[str]

@dataclass
class ParsedQuery:
    """解析后的查询"""
    original_query: str
    entities: List[QueryEntity]
    intent: QueryIntent
    filters: Dict[str, Any]  # 用于向量检索的过滤条件
    sub_queries: List[str]  # 分解后的子查询

class QueryProcessorV2:
    """
    查询处理器 v2.0
    
    新路径：
    1. Query解析：识别实体、意图、动作动词
    2. 实体识别：提取单位、技能、阶段等关键信息
    3. Filter检索：使用元数据过滤提升检索精度
    4. 结果聚合：基于解析结果整合答案
    """
    
    def __init__(self):
        """初始化查询处理器"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX
        )
        
        # 预定义的实体类型和关键词
        self.entity_patterns = {
            'unit': [
                r'丑角剧团', r'幽冥骑士', r'星际战士', r'兽人', r'钛族',
                r'死灵', r'泰伦虫族', r'混沌', r'黑暗灵族', r'机械教',
                r'战斗修女', r'灰骑士', r'死亡守望', r'血天使', r'极限战士',
                r'暗黑天使', r'太空野狼', r'白疤', r'钢铁之手', r'火蜥蜴',
                r'渡鸦守卫', r'帝国之拳', r'黑色圣堂', r'深红之拳',
                r'troupe', r'harlequin', r'space marine', r'ork', r'tau',
                r'necron', r'tyranid', r'chaos', r'drukhari', r'mechanicus'
            ],
            'skill': [
                r'死亡之舞', r'英雄之勇武', r'恶者之灾劫', r'诡术师之优雅',
                r'阔步巨兽', r'冲锋', r'攻击', r'防御', r'移动',
                r'death dance', r'heroic valour', r'malevolent malice',
                r'trickster grace', r'charge', r'attack', r'defense'
            ],
            'phase': [
                r'近战阶段', r'射击阶段', r'移动阶段', r'冲锋阶段',
                r'战斗阶段', r'回合开始', r'回合结束',
                r'fight phase', r'shooting phase', r'movement phase',
                r'charge phase', r'combat phase'
            ],
            'terrain': [
                r'地形', r'障碍物', r'建筑', r'森林', r'河流',
                r'terrain', r'obstacle', r'building', r'forest', r'river'
            ]
        }
        
        # 意图识别模式
        self.intent_patterns = {
            '规则查询': [r'如何', r'什么', r'哪些', r'是否', r'可以', r'允许'],
            '伤害计算': [r'伤害', r'攻击力', r'防御力', r'生命值', r'damage', r'attack'],
            '技能使用': [r'使用', r'激活', r'技能', r'能力', r'use', r'activate'],
            '单位信息': [r'单位', r'模型', r'属性', r'unit', r'model', r'stat']
        }
        
        logger.info("QueryProcessorV2 初始化完成")
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        解析用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            ParsedQuery: 解析结果
        """
        try:
            logger.info(f"开始解析查询: {query}")
            
            # 1. 实体识别
            entities = self._extract_entities(query)
            
            # 2. 意图识别
            intent = self._recognize_intent(query)
            
            # 3. 构建过滤条件
            filters = self._build_filters(entities, intent)
            
            # 4. 生成子查询
            sub_queries = self._generate_sub_queries(query, entities, intent)
            
            parsed_query = ParsedQuery(
                original_query=query,
                entities=entities,
                intent=intent,
                filters=filters,
                sub_queries=sub_queries
            )
            
            logger.info(f"查询解析完成: {len(entities)} 个实体, 意图: {intent.intent}")
            return parsed_query
            
        except Exception as e:
            logger.error(f"查询解析失败: {str(e)}")
            # 返回基础解析结果
            return ParsedQuery(
                original_query=query,
                entities=[],
                intent=QueryIntent(intent="规则查询", confidence=0.5, action_verbs=[]),
                filters={},
                sub_queries=[query]
            )
    
    def _extract_entities(self, query: str) -> List[QueryEntity]:
        """提取查询中的实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    # 计算置信度（基于匹配长度和位置）
                    confidence = min(len(match.group()) / len(query) * 2, 1.0)
                    
                    entities.append(QueryEntity(
                        text=match.group(),
                        type=entity_type,
                        confidence=confidence
                    ))
        
        # 去重并排序
        unique_entities = {}
        for entity in entities:
            key = (entity.text.lower(), entity.type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return sorted(unique_entities.values(), key=lambda x: x.confidence, reverse=True)
    
    def _recognize_intent(self, query: str) -> QueryIntent:
        """识别查询意图"""
        intent_scores = {}
        action_verbs = []
        
        # 计算各意图的得分
        for intent_name, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent_name] = score
        
        # 选择得分最高的意图
        if intent_scores:
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(top_intent[1] / 3, 1.0)  # 归一化置信度
        else:
            top_intent = ("规则查询", 0)
            confidence = 0.5
        
        # 提取动作动词
        action_verbs = self._extract_action_verbs(query)
        
        return QueryIntent(
            intent=top_intent[0],
            confidence=confidence,
            action_verbs=action_verbs
        )
    
    def _extract_action_verbs(self, query: str) -> List[str]:
        """提取动作动词"""
        action_verbs = []
        
        # 中文动作动词
        cn_verbs = ['获得', '使用', '激活', '选择', '进行', '造成', '受到', '允许', '禁止']
        for verb in cn_verbs:
            if verb in query:
                action_verbs.append(verb)
        
        # 英文动作动词
        en_verbs = ['get', 'use', 'activate', 'choose', 'perform', 'deal', 'take', 'allow', 'forbid']
        for verb in en_verbs:
            if re.search(r'\b' + verb + r'\b', query, re.IGNORECASE):
                action_verbs.append(verb)
        
        return action_verbs
    
    def _build_filters(self, entities: List[QueryEntity], intent: QueryIntent) -> Dict[str, Any]:
        """构建向量检索的过滤条件"""
        filters = {}
        
        # 基于实体类型构建过滤条件
        for entity in entities:
            if entity.confidence > 0.3:  # 只使用高置信度的实体
                if entity.type == 'unit':
                    filters['unit_type'] = entity.text
                elif entity.type == 'skill':
                    filters['skill_name'] = entity.text
                elif entity.type == 'phase':
                    filters['phase'] = entity.text
        
        # 基于意图添加过滤条件
        if intent.intent == '技能使用':
            filters['content_type'] = 'skill_usage'
        elif intent.intent == '伤害计算':
            filters['content_type'] = 'damage_calculation'
        
        return filters
    
    def _generate_sub_queries(self, query: str, entities: List[QueryEntity], intent: QueryIntent) -> List[str]:
        """生成子查询"""
        sub_queries = [query]  # 保留原始查询
        
        # 基于实体生成子查询
        if entities:
            entity_texts = [e.text for e in entities if e.confidence > 0.3]
            if entity_texts:
                # 组合实体生成查询
                combined_query = " ".join(entity_texts[:3])  # 最多3个实体
                if combined_query != query:
                    sub_queries.append(combined_query)
        
        # 基于意图生成子查询
        if intent.intent == '技能使用':
            skill_entities = [e.text for e in entities if e.type == 'skill']
            if skill_entities:
                sub_queries.append(f"{skill_entities[0]} 如何使用")
        
        return sub_queries
    
    def search_with_filters(self, parsed_query: ParsedQuery, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        使用过滤条件进行检索
        
        Args:
            parsed_query: 解析后的查询
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        try:
            all_results = []
            
            # 对每个子查询进行检索
            for sub_query in parsed_query.sub_queries:
                logger.info(f"检索子查询: {sub_query}")
                
                # 使用向量搜索（暂时不支持filter，后续可以扩展）
                results = self.vector_search.search(sub_query, top_k=top_k)
                
                # 应用过滤条件（在结果层面过滤）
                filtered_results = self._apply_filters(results, parsed_query.filters)
                all_results.extend(filtered_results)
            
            # 去重并排序
            unique_results = self._deduplicate_results(all_results)
            
            logger.info(f"检索完成，找到 {len(unique_results)} 个结果")
            return unique_results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用过滤条件"""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            text = result.get('text', '').lower()
            should_include = True
            
            # 检查单位类型过滤
            if 'unit_type' in filters:
                unit_type = filters['unit_type'].lower()
                if unit_type not in text:
                    should_include = False
            
            # 检查技能名称过滤
            if 'skill_name' in filters:
                skill_name = filters['skill_name'].lower()
                if skill_name not in text:
                    should_include = False
            
            # 检查阶段过滤
            if 'phase' in filters:
                phase = filters['phase'].lower()
                if phase not in text:
                    should_include = False
            
            if should_include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重并排序结果"""
        seen_texts = set()
        unique_results = []
        
        for result in results:
            text = result.get('text', '')[:100]  # 使用前100字符作为去重依据
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
        
        # 按相似度排序
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return unique_results
    
    def process_query(self, query: str) -> str:
        """
        处理用户查询的主方法
        
        Args:
            query: 用户查询
            
        Returns:
            str: 生成的答案
        """
        try:
            # 1. 解析查询
            parsed_query = self.parse_query(query)
            
            # 2. 使用过滤条件检索
            search_results = self.search_with_filters(parsed_query)
            
            if not search_results:
                return "抱歉，没有找到相关信息。"
            
            # 3. 生成答案
            answer = self._generate_answer(query, parsed_query, search_results)
            
            return answer
            
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            return f"处理查询时出现错误: {str(e)}"
    
    def _generate_answer(self, query: str, parsed_query: ParsedQuery, search_results: List[Dict[str, Any]]) -> str:
        """生成最终答案"""
        try:
            # 提取检索到的文本
            retrieved_texts = [result['text'] for result in search_results[:5]]  # 使用前5个结果
            
            # 构建prompt
            prompt = f"""你是一个专业的战锤40K规则专家。请根据提供的参考资料回答用户问题。

用户问题：{query}

解析信息：
- 识别到的实体：{[e.text for e in parsed_query.entities]}
- 查询意图：{parsed_query.intent.intent}
- 动作动词：{parsed_query.intent.action_verbs}

参考资料：
{chr(10).join([f"【{i+1}】{text}" for i, text in enumerate(retrieved_texts)])}

请根据参考资料准确回答用户问题。如果参考资料中没有相关信息，请明确说明。回答要简洁、准确、专业。"""

            # 调用LLM生成答案
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            return "生成答案时出现错误，请稍后重试。"


# 使用示例
if __name__ == "__main__":
    processor = QueryProcessorV2()
    
    # 测试查询
    test_query = "当一个丑角剧团单位在近战阶段开始时，它可以选择获得哪些能力？"
    
    print("=== 测试查询解析 ===")
    parsed = processor.parse_query(test_query)
    print(f"原始查询: {parsed.original_query}")
    print(f"识别实体: {[(e.text, e.type, e.confidence) for e in parsed.entities]}")
    print(f"查询意图: {parsed.intent.intent} (置信度: {parsed.intent.confidence})")
    print(f"动作动词: {parsed.intent.action_verbs}")
    print(f"过滤条件: {parsed.filters}")
    print(f"子查询: {parsed.sub_queries}")
    
    print("\n=== 测试完整流程 ===")
    answer = processor.process_query(test_query)
    print(f"生成的答案:\n{answer}") 