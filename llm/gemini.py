import logging
from typing import Optional, Dict, Any, List, Union
from .llm_interface import LLMInterface
from config import DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS
from utils.retry_utils import network_retry_decorator

logger = logging.getLogger(__name__)

class GeminiLLM(LLMInterface):
    """Gemini LLM实现（占位符）"""
    
    def __init__(self, api_key: str = None, default_model: str = None):
        """
        初始化Gemini LLM
        
        Args:
            api_key: Gemini API密钥
            default_model: 默认模型
        """
        self.api_key = api_key
        self.default_model = default_model or "gemini-pro"
        # TODO: 实现Gemini客户端初始化
    
    @network_retry_decorator(service_name="Gemini LLM")
    def call_llm(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = MAX_ANSWER_TOKENS,
                 **kwargs) -> str:
        """
        Gemini LLM调用实现（占位符）
        
        Args:
            prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 其他Gemini API参数
            
        Returns:
            str: LLM响应内容
            
        Raises:
            NotImplementedError: 功能尚未实现
        """
        # TODO: 实现Gemini API调用
        logger.warning("Gemini LLM实现尚未完成，返回占位符响应")
        raise NotImplementedError("Gemini LLM实现尚未完成")
    
    def build_messages(self, 
                      system_prompt: Optional[str] = None, 
                      user_prompt: str = "",
                      **kwargs) -> List[Dict[str, str]]:
        """
        构建Gemini格式的messages列表
        
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
