import logging
import re
from config import (
    DEFAULT_TEMPERATURE,
    OPENAI_API_KEY,
    LOG_FORMAT,
    LOG_LEVEL,
    LLM_MODEL
)
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class QueryCoTExpander:
    """
    使用思维链（Chain of Thought）和Few-shot技术来重写和扩展查询。
    """
    def __init__(self, openai_api_key=OPENAI_API_KEY, temperature=0.5):
        self.client = OpenAI(api_key=openai_api_key)
        self.temperature = temperature
        logger.info("QueryCoTExpander初始化完成")

    def expand_query(self, query: str, num_expansions: int = 4) -> list[str]:
        """
        使用CoT和Few-shot提示来扩展查询。

        Args:
            query: 原始查询。
            num_expansions: 希望生成的扩展查询数量。

        Returns:
            一个包含原始查询和所有扩展查询的列表。
        """
        if not query.strip():
            return []

        logger.info(f"开始使用CoT扩展查询：{query}")
        
        # 构建包含CoT和Few-shot的复杂提示
        prompt = self._build_cot_prompt(query, num_expansions)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            
            logger.debug(f"LLM原始输出:\n{content}")

            # 从LLM的响应中解析出查询
            expanded_queries = self._parse_llm_response(content)
            
            logger.info(f"成功生成的查询变体: {expanded_queries}")

            # 将原始查询也包含在内，确保它也被用于检索
            all_queries = [query] + expanded_queries
            return list(dict.fromkeys(all_queries)) # 去重并保持顺序

        except Exception as e:
            logger.error(f"查询扩展过程中发生错误: {e}", exc_info=True)
            # 如果扩展失败，则只返回原始查询
            return [query]

    def _build_cot_prompt(self, query: str, num_expansions: int) -> str:
        """构建包含CoT和Few-shot的提示"""
        return f"""你是一位顶级的战锤40K规则检索专家，你的任务是分析用户的查询，并将其重写和扩展为多个更适合向量数据库检索的变体。你需要模仿一个专家的思考过程。

### 指令
1. **思考过程（Chain of Thought）**: 请按照以下步骤进行结构化思考：
   - **第一步：实体识别与澄清** - 识别查询中的核心实体（单位、技能、武器等），并生成"X是什么？"类型的基础查询来澄清这些实体
   - **第二步：能力全集检索** - 基于识别的实体，生成"X有哪些能力？"类型的查询，确保能召回该实体的所有相关能力
   - **第三步：条件筛选** - 根据查询中的限制条件（如阶段、时机等），生成筛选性查询
   将你的完整思考过程写在 `<thinking>` 标签中。

2. **生成查询**: 在完成思考后，根据你的分析，在 `<queries>` 标签中生成 {num_expansions} 个不同的查询变体。

### 查询变体要求
* **抽象化思维**：不要被具体query中的技能名称限制，要抽象出通用的思考路径
* **从宽到窄**：先生成宽泛的能力全集查询，再生成条件筛选查询
* **实体优先**：优先澄清核心实体的定义和属性
* **能力覆盖**：确保能召回实体的所有相关能力，包括技能、规则等
* **条件筛选**：最后基于具体条件进行筛选

### 思维链模板
对于任何关于"X在Y条件下能获得什么能力"的查询，应该按以下逻辑生成：
1. X是什么？（实体定义）
2. X有哪些能力？（能力全集）
3. 这些能力中哪些在Y条件下使用？（条件筛选）

### 示例

**原始查询**: 当一个丑角剧团单位在近战阶段开始时，它可以选择获得哪些能力？

<thinking>
**第一步：实体识别与澄清**
- 核心实体: "丑角剧团单位" - 需要澄清这个单位的定义和特征
- 基础澄清查询: "丑角剧团单位是什么？"

**第二步：能力全集检索**
- 用户意图: 想了解丑角剧团单位的所有能力
- 关键查询: "丑角剧团单位有哪些能力？" - 这能召回所有相关能力，包括"死亡之舞"等技能

**第三步：条件筛选**
- 限制条件: "近战阶段开始时"
- 筛选查询: "丑角剧团单位的能力中哪些在近战阶段使用？"
</thinking>
<queries>
丑角剧团单位是什么？
丑角剧团单位有哪些能力？
丑角剧团单位的能力中哪些在近战阶段使用？
丑角剧团单位近战阶段的能力选择机制
丑角剧团单位的能力触发时机
</queries>

---

### 你的任务

**原始查询**: {query}

"""

    def _parse_llm_response(self, content: str) -> list[str]:
        """从LLM的完整响应中解析出<queries>部分。"""
        match = re.search(r'<queries>(.*?)</queries>', content, re.DOTALL | re.IGNORECASE)
        if not match:
            logger.warning("在LLM响应中未找到 <queries> 标签，将尝试按行解析。")
            # 作为备用方案，如果标签不存在，直接按行分割
            queries = [q.strip() for q in content.split('\n') if q.strip() and not q.startswith('<')]
            return queries

        queries_str = match.group(1)
        queries = [q.strip() for q in queries_str.split('\n') if q.strip()]
        return queries


def main():
    """
    主函数：用于测试QueryCoTExpander的效果
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='测试CoT查询扩展器')
    parser.add_argument('query', type=str, help='要扩展的查询')
    parser.add_argument('--num-expansions', type=int, default=4, help='生成的扩展查询数量')
    parser.add_argument('--temperature', type=float, default=0.5, help='生成温度')
    
    args = parser.parse_args()
    
    print("🤖 CoT查询扩展器测试")
    print("=" * 50)
    print(f"原始查询: {args.query}")
    print(f"扩展数量: {args.num_expansions}")
    print(f"生成温度: {args.temperature}")
    print("-" * 50)
    
    try:
        # 初始化扩展器
        expander = QueryCoTExpander(temperature=args.temperature)
        
        # 扩展查询
        expanded_queries = expander.expand_query(args.query, args.num_expansions)
        
        print("\n📝 扩展结果:")
        print("-" * 50)
        
        if len(expanded_queries) > 1:
            print(f"✅ 成功生成了 {len(expanded_queries)} 个查询变体:")
            for i, query in enumerate(expanded_queries, 1):
                print(f"{i:2d}. {query}")
        else:
            print("❌ 扩展失败，只返回原始查询")
            print(f"1. {expanded_queries[0]}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 