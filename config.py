"""
配置文件，用于管理项目范围内的公有字段和API密钥
"""

import logging
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[logging.StreamHandler()]  # 配置StreamHandler
)

logger = logging.getLogger(__name__)

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Pinecone API配置
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not found in environment variables")
PINECONE_INDEX_NAME = "wh40kcodex"

# 模型配置
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 10

# 嵌入模型
EMBADDING_MODEL = "text-embedding-ada-002"

# LLM模型
LLM_MODEL = "gpt-3.5-turbo"

# rerank 模型
RERANK_MODEL = "bge-reranker-v2-m3"

# 日志配置
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'app.log'
LOG_LEVEL = 'INFO'

# 应用配置
APP_TITLE = "战锤40K规则助手"
APP_ICON = "⚔️"
APP_HEADER = "" 