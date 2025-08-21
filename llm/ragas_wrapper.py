#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS LLM包装器
使用RAGAS提供的LangchainLLMWrapper包装我们的LLM
"""

from ragas.llms import LangchainLLMWrapper as RAGASLangchainWrapper
from .langchain_wrapper import LangchainLLMWrapper


def create_ragas_llm(llm_type: str, **kwargs):
    """
    创建RAGAS兼容的LLM包装器
    
    Args:
        llm_type: LLM类型
        **kwargs: 其他参数
        
    Returns:
        RAGAS兼容的LLM包装器
    """
    # 创建我们的Langchain LLM包装器
    our_llm = LangchainLLMWrapper(llm_type, **kwargs)
    
    # 使用RAGAS的LangchainLLMWrapper包装
    ragas_llm = RAGASLangchainWrapper(langchain_llm=our_llm)
    
    return ragas_llm
