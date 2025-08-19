import logging
import requests
from typing import Optional, Dict, Any, List, Union
from .llm_interface import LLMInterface
from config import DEEPSEEK_API_KEY, DEEPSEEK_SERVER, DEEPSEEK_MODEL, DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS

logger = logging.getLogger(__name__)

class DeepSeekLLM(LLMInterface):
    """DeepSeek LLM实现"""
    
    def __init__(self, api_key: str = None, default_model: str = None, base_url: str = None):
        """
        初始化DeepSeek LLM
        
        Args:
            api_key: DeepSeek API密钥
            default_model: 默认模型
            base_url: API基础URL
        """
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.default_model = default_model or DEEPSEEK_MODEL
        self.base_url = base_url or DEEPSEEK_SERVER
        
        if not self.api_key:
            logger.warning("DeepSeek API密钥未提供")
        
    def call_llm(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = MAX_ANSWER_TOKENS,
                 **kwargs) -> str:
        """
        DeepSeek LLM调用实现
        
        Args:
            prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 其他DeepSeek API参数
            
        Returns:
            str: LLM响应内容，失败时返回错误信息
        """
        try:
            if not self.api_key:
                return "抱歉，DeepSeek API密钥未提供"
            
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
            
            # 构建请求数据
            request_data = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # 添加其他参数
            request_data.update(kwargs)
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            logger.info(f"DeepSeek响应: {response_data}")
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"DeepSeek响应格式异常: {response_data}")
                return "抱歉，DeepSeek响应格式异常"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API请求失败: {str(e)}")
            return f"抱歉，DeepSeek API请求失败: {str(e)}"
        except ValueError as e:
            logger.error(f"DeepSeek参数错误: {str(e)}")
            return f"抱歉，DeepSeek参数错误: {str(e)}"
        except Exception as e:
            logger.error(f"DeepSeek LLM调用失败: {str(e)}")
            return f"抱歉，DeepSeek LLM调用时出错: {str(e)}"
    
    def build_messages(self, 
                      system_prompt: Optional[str] = None, 
                      user_prompt: str = "",
                      **kwargs) -> List[Dict[str, str]]:
        """
        构建DeepSeek格式的messages列表
        
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
