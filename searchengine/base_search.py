import os
import pinecone
from openai import OpenAI
import logging

class BaseSearchEngine:
    """搜索引擎基类，负责pinecone和openai初始化"""
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None, pinecone_environment: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.pinecone_environment = pinecone_environment or os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')

        # 初始化openai
        self.client = OpenAI(api_key=self.openai_api_key)
        # 新版pinecone初始化
        self.pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)

        logging.getLogger(__name__).info(
            f"[BaseSearchEngine] pinecone初始化完成，索引: {self.index_name}, 环境: {self.pinecone_environment}") 