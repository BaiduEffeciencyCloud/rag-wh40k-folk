import logging
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
from .llm_interface import LLMInterface
from config import LLM_MODEL, OPENAI_API_KEY, DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS
from utils.retry_utils import network_retry_decorator

logger = logging.getLogger(__name__)

class OpenAILLM(LLMInterface):
    """OpenAI LLM实现"""
    
    def __init__(self, api_key: str = None, default_model: str = None):
        """
        初始化OpenAI LLM
        
        Args:
            api_key: OpenAI API密钥，默认从config获取
            default_model: 默认模型，默认从config获取
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.default_model = default_model or LLM_MODEL
        self.client = OpenAI(api_key=self.api_key)
    
    @network_retry_decorator(service_name="OpenAI LLM")
    def call_llm(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = MAX_ANSWER_TOKENS,
                 **kwargs) -> str:
        """
        OpenAI LLM调用实现
        
        Args:
            prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 其他OpenAI API参数
            
        Returns:
            str: LLM响应内容
            
        Raises:
            ValueError: 参数错误
            Exception: API调用错误
        """
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
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        logger.info(f"OpenAI响应: {response}")
        return response.choices[0].message.content.strip()
    
    def build_messages(self, 
                      system_prompt: Optional[str] = None, 
                      user_prompt: str = "",
                      **kwargs) -> List[Dict[str, str]]:
        """
        构建OpenAI格式的messages列表
        
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
