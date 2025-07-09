from .processorinterface import QueryProcessorInterface
from typing import Dict, Any, List

class COTProcessor(QueryProcessorInterface):
    """思维链处理器，将复杂查询分解为多个子查询"""
    
    def __init__(self):
        # 初始化思维链模板
        self.cot_templates = {
            "事件查询": [
                "第一步：理解事件背景和历史",
                "第二步：分析事件的关键参与者和地点",
                "第三步：总结事件的影响和结果"
            ],
            "人物查询": [
                "第一步：了解人物的基本背景",
                "第二步：分析人物的主要事迹和成就",
                "第三步：总结人物的影响和地位"
            ],
            "概念查询": [
                "第一步：理解概念的定义和含义",
                "第二步：分析概念的相关要素和特点",
                "第三步：总结概念的应用和意义"
            ]
        }
    
    def process(self, query: str) -> List[Dict[str, Any]]:
        """
        将复杂查询分解为思维链步骤
        Args:
            query: 用户输入的查询
        Returns:
            包含多个思维链步骤的字典列表
        """
        cot_steps = self._chain_of_thought(query)
        return [
            {
                "text": step_query,
                "entities": [],
                "relations": [],
                "processed": True,
                "type": "cot_step",
                "original_query": query,
                "step": i,
                "step_description": step_desc
            }
            for i, (step_query, step_desc) in enumerate(cot_steps)
        ]
    
    def get_type(self) -> str:
        return "cot"
    
    def is_multi_query(self) -> bool:
        return True
    
    def _chain_of_thought(self, query: str) -> List[tuple]:
        """执行思维链分解"""
        # 判断查询类型
        query_type = self._classify_query(query)
        
        # 获取对应的思维链模板
        template = self.cot_templates.get(query_type, self.cot_templates["概念查询"])
        
        # 生成思维链步骤
        cot_steps = []
        for i, step_template in enumerate(template):
            step_query = self._generate_step_query(query, step_template, i)
            cot_steps.append((step_query, step_template))
        
        return cot_steps
    
    def _classify_query(self, query: str) -> str:
        """分类查询类型"""
        event_keywords = ["事件", "战役", "战争", "战斗", "远征", "叛乱"]
        person_keywords = ["人物", "角色", "领袖", "指挥官", "英雄", "阿巴顿", "荷鲁斯"]
        
        for keyword in event_keywords:
            if keyword in query:
                return "事件查询"
        
        for keyword in person_keywords:
            if keyword in query:
                return "人物查询"
        
        return "概念查询"
    
    def _generate_step_query(self, original_query: str, step_template: str, step_index: int) -> str:
        """生成思维链步骤查询"""
        # 提取查询中的核心概念
        core_concepts = self._extract_core_concepts(original_query)
        
        # 根据步骤模板生成具体查询
        if step_index == 0:  # 第一步：背景理解
            return f"理解{core_concepts}的背景和历史"
        elif step_index == 1:  # 第二步：要素分析
            return f"分析{core_concepts}的关键要素和特点"
        elif step_index == 2:  # 第三步：总结影响
            return f"总结{core_concepts}的影响和意义"
        else:
            return f"深入分析{core_concepts}的相关方面"
    
    def _extract_core_concepts(self, query: str) -> str:
        """提取查询中的核心概念"""
        # 简单的核心概念提取逻辑
        # 在实际应用中可以使用更复杂的NLP技术
        
        # 移除常见的疑问词和修饰词
        question_words = ["什么是", "有哪些", "如何", "为什么", "什么时候", "在哪里"]
        for word in question_words:
            if query.startswith(word):
                query = query[len(word):]
        
        # 移除标点符号
        import re
        query = re.sub(r'[？?！!，,。.]', '', query)
        
        return query.strip() if query.strip() else "相关内容"



