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
    
    def _validate_response_length(self, response: str, max_length: int = 8000) -> str:
        """验证和截断过长的响应，防止极端情况导致解析失败"""
        if not response or len(response) <= max_length:
            return response

        # 对于过长的响应，先尝试截断JSON结构
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and 'statements' in parsed:
                # 智能截断statements数组
                statements = parsed['statements']
                if isinstance(statements, list):
                    # 计算单个statement的平均长度
                    if statements:
                        avg_length = sum(len(str(s)) for s in statements) / len(statements)
                        max_statements = int(max_length * 0.6 / avg_length)  # 保留60%的空间给其他字段
                        if len(statements) > max_statements:
                            parsed['statements'] = statements[:max_statements]
                            parsed['truncated'] = True

                    # 重新序列化为JSON
                    truncated_response = json.dumps(parsed, ensure_ascii=False)
                    if len(truncated_response) <= max_length:
                        return truncated_response

        except Exception:
            pass

        # 如果JSON处理失败，直接截断字符串
        return response[:max_length] + "..."

    def _safe_json_parse(self, response: str) -> tuple:
        """安全地解析JSON，返回 (success, parsed_data, error)"""
        if not response or not isinstance(response, str):
            return False, None, ValueError("Invalid response type")

        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict):
                return True, parsed, None
            else:
                return False, None, ValueError("Parsed data is not a dictionary")
        except json.JSONDecodeError as e:
            return False, None, e
        except Exception as e:
            return False, None, e

    def _create_fallback_format(self, original_response: str) -> str:
        """创建兜底的JSON格式，确保RAGAS能处理"""
        # 清理原始响应中的特殊字符
        cleaned_response = original_response.replace('\n', ' ').replace('\r', ' ').strip()

        # 如果原始响应看起来已经是JSON的一部分，尝试提取文本内容
        if '"text"' in original_response or '"statements"' in original_response:
            # 尝试提取可能的文本内容
            import re
            text_matches = re.findall(r'"text"\s*:\s*"([^"]*)"', original_response)
            if text_matches:
                cleaned_response = text_matches[0]

        # 创建标准的兜底格式
        fallback_data = {
            "text": cleaned_response[:2000] if len(cleaned_response) > 2000 else cleaned_response,
            "fallback": True,
            "original_length": len(original_response)
        }

        return json.dumps(fallback_data, ensure_ascii=False)

    def _fix_json_format(self, response: str) -> str:
        """修复JSON格式问题，确保符合RAGAS期望的StringIO格式"""
        # Step4: 实现真实的逻辑代码

        # 1. 首先验证和截断响应长度
        response = self._validate_response_length(response)

        # 2. 安全地解析JSON
        success, parsed, error = self._safe_json_parse(response)

        if not success:
            # JSON解析失败，使用兜底格式
            return self._create_fallback_format(response)

        # 3. 处理已解析的JSON数据
        if not isinstance(parsed, dict):
            return self._create_fallback_format(response)

        # 4. 修复QWEN statements格式 - 添加缺失的text字段
        if 'statements' in parsed and isinstance(parsed['statements'], list):
            if 'text' not in parsed:
                # 过滤掉None值和空字符串
                valid_statements = [str(s) for s in parsed['statements'] if s is not None and str(s).strip()]
                parsed['text'] = '\n'.join(valid_statements)
            return json.dumps(parsed, ensure_ascii=False)

        # 5. 修复verdict格式 - 添加缺失的text字段
        if 'verdict' in parsed and parsed['verdict'] is not None and 'text' not in parsed:
            reason = parsed.get('reason', '')
            if reason:
                parsed['text'] = str(reason)
            else:
                parsed['text'] = str(parsed['verdict'])
            return json.dumps(parsed, ensure_ascii=False)

        # 6. 如果已经有text字段，直接返回
        if 'text' in parsed:
            return json.dumps(parsed, ensure_ascii=False)

        # 7. 处理其他格式 - 尝试构建text字段
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(value, str) and value.strip():
                    parsed['text'] = str(value)
                    return json.dumps(parsed, ensure_ascii=False)
                elif isinstance(value, list) and value:
                    # 处理列表类型的值
                    valid_items = [str(item) for item in value if item is not None and str(item).strip()]
                    if valid_items:
                        parsed['text'] = '\n'.join(valid_items)
                        return json.dumps(parsed, ensure_ascii=False)

        # 8. 如果所有修复都失败，使用兜底格式
        return self._create_fallback_format(response)
    
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
