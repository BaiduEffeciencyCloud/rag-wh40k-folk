from .openai import OpenAILLM
from .deepseek import DeepSeekLLM
from .gemini import GeminiLLM
from config import OPENAI_API_KEY

# 可选导入，如果配置中没有这些API密钥，则设为None
try:
    from config import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = None

try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

class LLMFactory:
    """LLM工厂类，用于创建不同类型的LLM实例"""
    
    @staticmethod
    def create_llm(llm_type: str, **kwargs):
        """
        创建LLM实例
        
        Args:
            llm_type: LLM类型 ("openai", "deepseek", "gemini")
            **kwargs: 传递给具体LLM类的参数
            
        Returns:
            LLMInterface: LLM实例
            
        Raises:
            ValueError: 不支持的LLM类型
        """
        if llm_type.lower() == "openai":
            return OpenAILLM(**kwargs)
        elif llm_type.lower() == "deepseek":
            return DeepSeekLLM(**kwargs)
        elif llm_type.lower() == "gemini":
            return GeminiLLM(**kwargs)
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
    