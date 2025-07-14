"""
llm_utils.py - 通用LLM调用工具

职责：根据传入的OpenAI client和prompt，返回结果
保持与具体业务规则的完全解耦
"""

import logging
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
from config import LLM_MODEL

logger = logging.getLogger(__name__)

def call_llm(client: OpenAI, 
             prompt: Union[str, List[Dict[str, str]]], 
             model: str = LLM_MODEL,
             temperature: float = 0.4,
             max_tokens: int = 3000,
             **kwargs) -> str:
    """
    通用LLM调用函数
    
    Args:
        client: OpenAI客户端实例
        prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
        model: 模型名称
        temperature: 生成温度
        max_tokens: 最大token数
        **kwargs: 其他OpenAI API参数
        
    Returns:
        str: LLM响应内容，失败时返回错误信息
    """
    try:
        # 处理不同的prompt格式
        if isinstance(prompt, str):
            # 字符串格式：直接作为user message
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # 列表格式：完整的messages
            messages = prompt
        else:
            raise ValueError("prompt必须是字符串或消息列表")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"LLM调用失败: {str(e)}")
        return f"抱歉，LLM调用时出错: {str(e)}"

def build_messages(system_prompt: Optional[str] = None, 
                  user_prompt: str = "",
                  **kwargs) -> List[Dict[str, str]]:
    """
    构建messages列表的辅助函数
    
    Args:
        system_prompt: 系统prompt（可选）
        user_prompt: 用户prompt
        **kwargs: 其他role的prompt（如assistant）
        
    Returns:
        List[Dict[str, str]]: 格式化的messages列表
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    
    # 支持其他role的prompt
    for role, content in kwargs.items():
        if content:
            messages.append({"role": role, "content": content})
    
    return messages
