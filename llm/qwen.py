import logging
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
from .llm_interface import LLMInterface
from config import QWEN_API_KEY, QWEN_LLM_SERVER, QWEN_LLM_MODEL, DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS
from utils.retry_utils import network_retry_decorator

logger = logging.getLogger(__name__)

class QwenLLM(LLMInterface):
    """Qwen LLM实现 - 使用阿里云DashScope兼容OpenAI接口"""
    
    def __init__(self, api_key: str = None, default_model: str = None, base_url: str = None):
        """
        初始化Qwen LLM
        
        Args:
            api_key: 阿里云DashScope API密钥
            default_model: 默认模型
            base_url: API基础URL
        """
        self.api_key = api_key or QWEN_API_KEY
        self.default_model = default_model or QWEN_LLM_MODEL
        self.base_url = base_url or QWEN_LLM_SERVER
        
        if not self.api_key:
            logger.warning("Qwen API密钥未提供")
        else:
            # 初始化OpenAI客户端（使用DashScope兼容接口）
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    @network_retry_decorator(service_name="Qwen LLM")
    def call_llm(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = MAX_ANSWER_TOKENS,
                 **kwargs) -> str:
        """
        Qwen LLM调用实现
        
        Args:
            prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 其他Qwen API参数
            
        Returns:
            str: LLM响应内容
            
        Raises:
            ValueError: 参数错误
            Exception: API调用错误
        """
        if not self.api_key:
            raise ValueError("Qwen API密钥未提供")
        
        if not hasattr(self, 'client'):
            raise ValueError("Qwen客户端未初始化")
        
        # 使用初始化时的默认模型
        model_name = self.default_model
        
        # 处理不同的prompt格式
        if isinstance(prompt, str):
            # 字符串格式：直接作为user message
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # 列表格式：完整的messages
            messages = prompt
        else:
            raise ValueError("prompt必须是字符串或消息列表")
        
        # 构建请求参数
        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 添加其他参数
        request_params.update(kwargs)
        
        # 发送请求
        response = self.client.chat.completions.create(**request_params)
        # 返回响应内容
        return response.choices[0].message.content.strip()
    
    def build_messages(self, 
                      system_prompt: Optional[str] = None, 
                      user_prompt: str = "",
                      **kwargs) -> List[Dict[str, Any]]:
        """
        构建Qwen格式的messages列表，content为[{"type": "text", "text": ...}]
        """
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": str(system_prompt)}]
            })
        if user_prompt:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": str(user_prompt)}]
            })
        for role, content in kwargs.items():
            if content:
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": str(content)}]
                })
        return messages
