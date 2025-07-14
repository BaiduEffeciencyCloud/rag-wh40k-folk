"""
llm_analysis.py

主要功能：
    - 对评估流程生成的Markdown报告（visualization_summary.md）进行大模型（LLM）自动解读
    - 重点输出：
        1. 报告中的数字和图像解读
        2. query各项质量的Insight总结
        3. 针对检索/评估流程的优化建议
    - 该模块可扩展为实际调用OpenAI、Qwen等大模型API
    - 当前实现为模拟LLM调用，便于后续集成
"""
import os
from typing import Optional

# 导入全局配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) or '.')
from config import LLM_MODEL, OPENAI_API_KEY

def build_llm_prompt(md_path: str, md_text: str) -> list:
    """
    构建system, user格式的prompt，明确大模型角色和任务。
    """
    system_msg = (
        "你是一名精通数据分析，尤其擅长RAG用户query量化分析的工程师。"
        "你需要对上传的检索评估Markdown报告进行专业解读，输出：\n"
        "1. 报告中的数字和图像解读；\n"
        "2. query各项质量的Insight总结；\n"
        "3. 针对检索/评估流程的优化建议。"
    )
    user_msg = (
        f"请你阅读以下Markdown报告内容：\n{md_text}\n结合所有数字、图表和可视化，给出详细分析、结论和建议。"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def extract_md_text(md_path: str, max_chars: int = 8000) -> str:
    """
    读取Markdown文本内容，超长时截断。
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...（内容已截断，仅展示前部分）"
        return text
    except Exception as e:
        return f"[Markdown读取失败] {e}"

def analyze_report_with_llm(md_path: str, api: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    """
    对指定Markdown报告进行大模型自动解读，输出分析结果，并保存到insight.md。
    Args:
        md_path: Markdown报告文件路径
        api: 可选，指定大模型API类型（如openai、qwen等）
        save_dir: 可选，insight.md保存目录（默认为md所在目录）
    Returns:
        str: LLM分析结果文本
    """
    if not os.path.exists(md_path):
        return f"[LLM分析] 未找到Markdown报告文件: {md_path}"
    md_text = extract_md_text(md_path)
    prompt = build_llm_prompt(md_path, md_text)
    llm_result = None
    try:
        import openai
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=prompt,
            max_tokens=2048,
            temperature=0.3
        )
        llm_result = response.choices[0].message.content.strip()
    except Exception as e:
        llm_result = f"[LLM分析] API调用失败: {e}\n(可切换为本地模拟或检查API配置)"
    # 保存到insight.md
    if save_dir is None:
        save_dir = os.path.dirname(md_path)
    insight_path = os.path.join(save_dir, "insight.md")
    with open(insight_path, "w", encoding="utf-8") as f:
        f.write("# LLM自动解读报告\n\n")
        f.write(llm_result)
    return llm_result 