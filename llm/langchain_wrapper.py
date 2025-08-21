#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langchain LLM包装器
将我们的LLM工厂包装为Langchain兼容的LLM接口
"""

import json
from typing import Any, List, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .llm_factory import LLMFactory
           

class LangchainLLMWrapper(LLM):
    """将我们的LLM工厂包装为Langchain兼容的LLM"""
    
    def __init__(self, llm_type: str, **kwargs):
        super().__init__()
        # 使用object.__setattr__来设置属性，避免Pydantic验证错误
        object.__setattr__(self, 'llm_instance', LLMFactory.create_llm(llm_type, **kwargs))
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """调用LLM实例"""
        response = self.llm_instance.call_llm(prompt, **kwargs)
        
        # 只对需要JSON格式的指标进行处理
        if self._needs_json_format(prompt):
            response = self._fix_json_format(response)
        
        return response
    
    def _needs_json_format(self, prompt: str) -> bool:
        """判断是否需要JSON格式输出"""
        if prompt is None:
            return False
        json_keywords = ['statements', 'verdict', 'faithfulness', 'relevancy', 'correctness', 'analyze', 'judge', 'complexity']
        return any(keyword in prompt.lower() for keyword in json_keywords)
    
    def _fix_json_format(self, response: str) -> str:
        """修复JSON格式问题"""
        try:
            parsed = json.loads(response)
            
            # 处理 {"text": ...} 格式
            if 'text' in parsed:
                return parsed['text']
            
            # 处理 {"statements": [...]} 格式（DeepSeek特有）
            if 'statements' in parsed and isinstance(parsed['statements'], list):
                return '\n'.join(parsed['statements'])
            
            # 处理其他可能的JSON格式
            if isinstance(parsed, dict):
                # 如果是字典，尝试提取第一个字符串值
                for key, value in parsed.items():
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, list) and value and isinstance(value[0], str):
                        return '\n'.join(value)
            
        except Exception as e:
            # 如果JSON解析失败，返回原始响应
            pass
        
        return response
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "custom_llm_wrapper"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数"""
        return {
            "llm_type": getattr(self.llm_instance, 'llm_type', 'unknown'),
            "model": getattr(self.llm_instance, 'default_model', 'unknown')
        }
