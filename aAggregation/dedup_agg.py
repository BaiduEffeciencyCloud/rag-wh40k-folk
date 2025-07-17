from .agginterface import AggregationInterface
from typing import List, Dict, Any
from openai import OpenAI
import os
import logging
from config import OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

class DedupAggregation(AggregationInterface):
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_answer(self, search_results) -> List[str]:
        all_answers = self._extract_all_texts(search_results)
        deduped_answers = self._deduplicate_answers(all_answers)
        return deduped_answers

    def _extract_all_texts(self, search_results) -> List[str]:
        # 兼容 List[Dict] 和 List[List[Dict]]
        if not search_results:
            return []
        if isinstance(search_results, list) and search_results and isinstance(search_results[0], dict):
            # 单query，外层补一层
            search_results = [search_results]
        return [item['text'] for collection in search_results for item in collection if 'text' in item]

    def _deduplicate_answers(self, answers: List[str]) -> List[str]:
        # 简单去重，保留顺序，过滤None和空字符串
        seen = set()
        deduped = []
        for ans in answers:
            if ans is None or ans == '':
                continue
            if ans not in seen:
                deduped.append(ans)
                seen.add(ans)
        return deduped

    def semantic_parse(self, query: str, context: List[str]) -> str:
        """
        使用OpenAI进行语义解析，整合答案
        Args:
            query: 用户问题
            context: 上下文信息（答案片段列表）
        Returns:
            str: 解析后的最终答案
        """
        try:
            system_prompt = (
                "你是一名资深战锤40K规则专家，"
                "回答务必精准简洁，只使用并引用提供的上下文信息。"
                "每条结论后用“[证据: 来源X]”标注；"
            )
            user_prompt = f"""
## 核心要求
1.  **严格基于上下文**：仅使用提供的所有上下文信息进行回答，不引入任何外部知识或进行推测。所有答案内容必须能直接追溯到提供的文档。
2.  **完整性与整合**：务必**整合所有分散在不同章节、问答集和单位数据表中的相关信息**，确保答案的全面性和连贯性。特别注意规则之间的相互作用和优先级。
3.  **结构化回答**：根据问题的复杂度和相关信息的丰富程度，选择最合适的回答结构。先说明核心规则，再补充例外和细节。对于简单直接的问题，可以直接提供一个综合性的答案，无需强制使用所有子标题。
4.  **专业术语**: 使用战锤40K的标准术语和表达方式。
5.  **来源标注**: 每条要点末尾以 `[来源X]` 标注文档编号。

## 严格限制
-   **禁止泛化**：仅使用提供的上下文，不得引入任何外部知识或模糊泛化。如无上下文支持，明确说明无法确定。

## 建议回答结构（请根据问题的复杂度和相关信息的丰富程度，灵活选择和调整以下部分）：
### 核心规则
-   直接回答用户问题的核心内容，说明基本机制和效果。
-   **内容必须直接来源于上下文中的明确信息。** 如果能使用上下文里的原文就使用原文

### 详细说明
-   补充与核心规则相关的具体细节。

### 例外情况
-   **仔细扫描所有上下文，寻找任何明确的例外规则。**
-   特别关注包含以下关键词的内容：“但是”、“然而”、“不过”、“例外”、“特殊”、“允许”、“如果...那么”、“尽管...仍然”。
-   列出所有相关的例外规则、具体条件和效果。

### 注意事项
-   突出显示源文档中明确提及的、可能影响规则解释或应用的重要提醒、限制或潜在的冲突点。
-   **如果某个部分在提供的上下文中没有相关信息，则无需包含该部分。**

## 自我校验:
- 自我校验: 在生成最终答案之前，请进行一次内部核对：确保答案中的每一句话都能在提供的上下文信息中找到直接支持，并且没有任何矛盾或遗漏的关键信息。

## 输出格式
直接输出结构化的答案，无需添加“问题：”、“回答：”等标签。确保答案完整、准确、易于理解，且严格基于提供的上下文信息。


## 校对清单
- 已引用来源编号：X、Y、Z  
- 潜在盲区：可能存在的上下文盲区

上下文信息：
{chr(10).join([f'【{i+1}】{text}' for i, text in enumerate(context)])}

用户问题："{query}"
"""
            from utils.llm_utils import call_llm, build_messages
            
            messages = build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return call_llm(
                client=self.client,
                prompt=messages,
                temperature=0.4,
                max_tokens=3000
            )
        except Exception as e:
            logger.error(f"语义解析时出错：{str(e)}")
            return "抱歉，解析问题时出错。"

    def aggregate(self, search_results, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        answers = self.extract_answer(search_results)
        final_answer = self.semantic_parse(query, answers) if answers else "未检索到有效答案"
        return {
            'answers': answers,
            'final_answer': final_answer,
            'count': len(answers),
            'query': query
        } 