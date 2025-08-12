from typing import Optional
from .eminterface import EmbeddingInterface
from .open import OpenEmbedding
from .qwen import QwenEmbedding
from config import OPENAI_DIMENSION,QWEN_DIMENSION

class EmbeddingFactory:
    """嵌入模型工厂类"""
    
    def __init__(self, model_name: str = "qwen"):
        self.model_name = model_name
    
    @staticmethod
    def create_embedding_model(model_name: str = "qwen", **kwargs) -> EmbeddingInterface:
        """
        创建嵌入模型实例
        
        Args:
            model_name: 模型名称 ("openai" 或 "qwen")
            **kwargs: 自定义参数，如果不提供则从配置文件获取
            
        Returns:
            EmbeddingInterface: 嵌入模型实例
            
        Raises:
            ValueError: 当模型名称不支持或配置缺失时
        """
        if model_name == "openai":
            if kwargs:
                # 使用自定义参数
                return OpenEmbedding(**kwargs)
            else:
                # 从配置文件获取参数
                import config
                if not config.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not configured")
                return OpenEmbedding(
                    api_key=config.OPENAI_API_KEY,
                    model=config.EMBADDING_MODEL,
                    dimensions=OPENAI_DIMENSION
                )
        elif model_name == "qwen":
            if kwargs:
                # 使用自定义参数
                return QwenEmbedding(**kwargs)
            else:
                # 从配置文件获取参数
                import config
                if not config.QWEN_API_KEY:
                    raise ValueError("QWEN API key not configured")
                return QwenEmbedding(
                    api_key=config.QWEN_API_KEY,
                    model=config.EMBEDDING_MODEL_QWEN,
                    dimensions=QWEN_DIMENSION
                )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
    
    def get_embedding_model(self) -> EmbeddingInterface:
        """获取嵌入模型实例"""
        return self.create_embedding_model(self.model_name)