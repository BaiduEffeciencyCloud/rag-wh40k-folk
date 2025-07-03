import logging
from config import (
    DEFAULT_TEMPERATURE,
    OPENAI_API_KEY,
    LOG_FORMAT,
    LOG_LEVEL,
    LLM_MODEL
)
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(LOG_FORMAT)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class QueryExpander:
    def __init__(self, openai_api_key=OPENAI_API_KEY, temperature=0.7):
        self.client = OpenAI(api_key=openai_api_key)
        self.temperature = temperature
        logger.info("QueryExpander初始化完成")
        self.cache = {}
        
    def expand_query(self, query: str) -> str:
        """
        扩展查询并选择最契合用户意图的查询
        Args:
            query: 原始查询
        Returns:
            str: 最契合用户意图的扩展查询
        """
        try:
            if not query.strip():
                return ""
            logger.info(f"开始扩展查询：{query}")
            expand_prompt = f"""你是一个专业的战锤40K规则专家。请根据用户的查询生成多个相关的查询变体。\n\n重要要求：\n1. 必须完整保留原始问题中的所有专有名词、单位名称，不要替换或省略。\n2. 每个变体都要尽量具体，避免泛化（如\"获得额外攻击力\"这类模糊表述）。\n3. 变体应从不同角度或细节切入，但都要紧扣原始问题的核心意图。\n4. 优先使用战锤40K的专业术语和规则语言。\n5. 保持简洁直接，避免添加不必要的修饰词（如\"在战锤40K中\"、\"对于...而言\"等）。\n6. 使用通用的能力描述词（如\"特殊能力\"、\"技能\"、\"效果\"等）作为备选。\n7. 返回3-5个变体，每个变体都是完整问句。\n8. 无需增加背景说明，如\"在战锤40K中\"这样已经明确了问答上下文的字样\n9. 检索优化：确保变体能够有效检索到相关的规则文档。\n\n原始查询：{query}\n\n请生成相关的查询变体："""
            expand_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": expand_prompt}],
                temperature=self.temperature
            )
            expanded_queries = [q.strip() for q in expand_response.choices[0].message.content.split('\n') if q.strip()]
            logger.info(f"生成的查询变体：{expanded_queries}")
            select_prompt = f"""你是一个专业的战锤40K规则专家。请从多个查询变体中选择最契合用户原始意图的一个。\n\n选择标准：\n1. 最准确地反映原始查询的意图。\n2. 必须完整保留原始问题中的专有名词和能力名称。\n3. 变体要具体、专业，避免泛化。\n4. 能够有效检索到相关的规则信息。\n5. 优先选择简洁直接的变体，避免不必要的修饰词。\n6. 如果变体都不够理想，可以适当修改选中的变体，但保持核心意图。\n\n原始查询：{query}\n\n查询变体：\n{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(expanded_queries)])}\n\n请选择最契合原始查询意图的变体，并返回。如果所有变体都不够理想，请基于原始查询生成一个更准确的查询。"""
            select_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": select_prompt}],
                temperature=self.temperature
            )
            selected_query = select_response.choices[0].message.content.strip()
            logger.info(f"选择的最契合查询：{selected_query}")
            return selected_query
        except Exception as e:
            logger.error(f"查询扩展出错：{str(e)}")
            return query  # 如果出错，返回原始查询
    
    def clear_cache(self):
        self.cache.clear()