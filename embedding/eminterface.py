from abc import ABC, abstractmethod
from typing import List, Union

class EmbeddingInterface(ABC):
    """嵌入模型接口抽象类"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量表示"""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """获取向量维度"""
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """兼容RAGAS的方法名：获取单个文本的向量表示"""
        if text is None:
            raise TypeError("text参数不能为None")
        if not isinstance(text, str):
            raise TypeError("text参数必须是字符串类型")
        return self.get_embedding(text)
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步版本的embed_query，兼容RAGAS"""
        if text is None:
            raise TypeError("text参数不能为None")
        if not isinstance(text, str):
            raise TypeError("text参数必须是字符串类型")
        
        # 使用asyncio.run_in_executor来异步调用同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """兼容RAGAS的方法名：批量获取文本的向量表示"""
        if texts is None:
            raise TypeError("texts参数不能为None")
        if not isinstance(texts, list):
            raise TypeError("texts参数必须是列表类型")
        if not texts:  # 处理空列表情况
            return []
        
        # 验证列表中的所有元素都是字符串
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}]必须是字符串类型，当前类型: {type(text)}")
        
        return self.get_embeddings(texts)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步版本的embed_documents，兼容RAGAS"""
        if texts is None:
            raise TypeError("texts参数不能为None")
        if not isinstance(texts, list):
            raise TypeError("texts参数必须是列表类型")
        if not texts:  # 处理空列表情况
            return []
        
        # 验证列表中的所有元素都是字符串
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}]必须是字符串类型，当前类型: {type(text)}")
        
        # 使用asyncio.run_in_executor来异步调用同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embeddings, texts)