from .agginterface import AggregationInterface
from typing import List, Dict, Any
import logging
import os
import json
from utils.llm_utils import call_llm, build_messages
from config import DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS, LLM_TYPE
from llm.llm_factory import LLMFactory

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'template')

logger = logging.getLogger(__name__)

class DynamicAggregation(AggregationInterface):
    def __init__(self, template: str = 'default'):
        self.template = template or 'default'

    def extract_answer(self, search_results) -> List[str]:
        # 兼容 List[Dict] 和 List[List[Dict]]
        if not search_results:
            return []
        if isinstance(search_results, list) and search_results and isinstance(search_results[0], dict):
            search_results = [search_results]
        return [item['text'] for collection in search_results for item in collection if 'text' in item and item['text']]

    def _deduplicate_answers(self, answers: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for ans in answers:
            if ans is None or ans == '':
                continue
            if ans not in seen:
                deduped.append(ans)
                seen.add(ans)
        return deduped

    def _load_prompt(self, context: List[str], query: str) -> List[Dict[str, str]]:
        """
        1. 读取system.json作为system prompt
        2. 读取template.json作为user prompt
        3. context和query始终独立组装并拼接到user_prompt后面
        4. 使用build_messages组合为List
        5. 返回prompt列表
        """
        try:
            # 1. 生成context和query部分（与模板读取彻底解耦）
            context_str = '\n'.join([f'【{i+1}】{text}' for i, text in enumerate(context)])
            query_str = query

            # 2. 读取system prompt
            system_path = os.path.join(TEMPLATE_DIR, 'system.json')
            with open(system_path, 'r', encoding='utf-8') as f:
                system_data = json.load(f)
            system_prompt = system_data.get('system_prompt', '')

            # 3. 读取user prompt模板
            user_template_path = os.path.join(TEMPLATE_DIR, f'{self.template}.json')
            if not os.path.exists(user_template_path):
                user_template_path = os.path.join(TEMPLATE_DIR, 'default.json')
            with open(user_template_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            user_prompt = user_data.get('user_prompt', '')

            # 4. 如果user_prompt为空，自动降级为default.json
            if not user_prompt.strip() and not user_template_path.endswith('default.json'):
                with open(os.path.join(TEMPLATE_DIR, 'default.json'), 'r', encoding='utf-8') as f:
                    default_data = json.load(f)
                user_prompt = default_data.get('user_prompt', '')

            # 5. 如果default.json也为空，兜底只输出context和query
            if not user_prompt.strip():
                user_prompt = f"上下文信息:{context_str}\n用户问题：\"{query_str}\""
            # 6. 使用build_messages组合为List，确保包含type字段
            messages = build_messages(
                system_prompt={"content": system_prompt, "type": "text"},
                user_prompt={"content": f"{user_prompt}\n上下文信息:{context_str}\n用户问题：\"{query_str}\"", "type": "text"}
            )
            return messages
        except Exception as e:
            logger.error(f"加载prompt模板失败: {str(e)}")
            return [{"role": "error", "content": f"聚合出错: {str(e)}"}]

    def semantic_parse(self, query: str, context: List[str]) -> str:
        # 这里只拼装prompt，不直接调用LLM
        return self._load_prompt(context, query)

    def aggregate(self, search_results, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        answers = self.extract_answer(search_results)
        deduped = self._deduplicate_answers(answers)
        prompt_text = self.semantic_parse(query, deduped) if deduped else "未检索到有效答案"
        # 1. 生成llm实例
        llm_instance = LLMFactory.create_llm(LLM_TYPE)
        # 2. 用llm_instance.build_messages生成prompt（Qwen等自动适配格式）
        messages = llm_instance.build_messages(system_prompt="", user_prompt=prompt_text)
        # 3. 用llm_instance.call_llm生成最终答案
        try:
            final_answer = llm_instance.call_llm(
                prompt=messages
            )
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            final_answer = "聚合出错: LLM调用失败"

        return {
            'answers': deduped,
            'final_answer': final_answer,
            'count': len(deduped),
            'query': query
        } 