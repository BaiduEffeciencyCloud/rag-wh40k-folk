from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Iterable
from config import DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS

class LLMInterface(ABC):
    """LLM接口基类，定义统一的LLM调用方法"""
    
    @abstractmethod
    def call_llm(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = MAX_ANSWER_TOKENS,
                 **kwargs) -> str:
        """
        统一的LLM调用接口
        
        Args:
            prompt: 可以是字符串（作为user message）或消息列表（完整的messages）
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 其他模型特定参数
            
        Returns:
            str: LLM响应内容，失败时返回错误信息
        """
        pass
    
    @abstractmethod
    def call_llm_stream(self,
                        prompt: Union[str, List[Dict[str, str]]],
                        temperature: float = DEFAULT_TEMPERATURE,
                        max_tokens: int = MAX_ANSWER_TOKENS,
                        *,
                        granularity: str = "chunk",
                        event_mode: bool = False,
                        fallback_to_sync: bool = True,
                        **kwargs) -> Iterable[Union[str, Dict[str, Any]]]:
        """流式返回LLM增量文本。
        Args:
            prompt: 字符串或messages列表
            temperature: 生成温度
            max_tokens: 最大token数
            granularity: "chunk" | "char" （默认chunk）
            event_mode: True 时返回事件dict，否则返回str
            fallback_to_sync: 流式失败自动回退到同步接口
            **kwargs: 其他模型特定参数
        Returns:
            可迭代对象：按片段/字符返回文本或事件
        """
        pass

    @abstractmethod
    def build_messages(self, 
                      system_prompt: Optional[str] = None, 
                      user_prompt: str = "",
                      **kwargs) -> List[Dict[str, str]]:
        """
        构建messages列表的辅助方法
        
        Args:
            system_prompt: 系统prompt（可选）
            user_prompt: 用户prompt
            **kwargs: 其他role的prompt（如assistant）
            
        Returns:
            List[Dict[str, str]]: 格式化的messages列表
        """
        pass
