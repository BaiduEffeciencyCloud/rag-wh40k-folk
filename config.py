"""
配置文件，用于管理项目范围内的公有字段和API密钥
"""

import logging
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

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
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Neo4j 数据库配置
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 检查Neo4j配置
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    logger.warning("Neo4j configuration incomplete. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env file")

# Neo4j 测试数据库配置 (用于单元测试和集成测试)
NEO4J_TEST_URI = os.getenv("NEO4J_TEST_URI")
NEO4J_TEST_USERNAME = os.getenv("NEO4J_TEST_USERNAME")
NEO4J_TEST_PASSWORD = os.getenv("NEO4J_TEST_PASSWORD")

# 检查Neo4j测试配置
if not all([NEO4J_TEST_URI, NEO4J_TEST_USERNAME, NEO4J_TEST_PASSWORD]):
    logger.warning("Neo4j test configuration incomplete. Please set NEO4J_TEST_URI, NEO4J_TEST_USERNAME, and NEO4J_TEST_PASSWORD in .env file")

# 模型配置
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "15"))

# 超时配置
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))  # 查询超时时间（秒）

# 答案生成配置
ANSWER_GENERATION_TEMPERATURE = float(os.getenv("ANSWER_GENERATION_TEMPERATURE", "0.3"))  # 答案生成温度
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "2000"))  # 答案生成最大token数

# 结果整合配置
DEDUPLICATION_THRESHOLD = float(os.getenv("DEDUPLICATION_THRESHOLD", "0.8"))  # 去重阈值

# 并发配置
DEFAULT_MAX_WORKERS = int(os.getenv("DEFAULT_MAX_WORKERS", "4"))  # 默认最大并发工作线程数

# 答案生成配置
MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "10"))  # 答案生成时使用的最大上下文结果数

# 嵌入模型
EMBADDING_MODEL = os.getenv("EMBADDING_MODEL", "text-embedding-3-large")

# LLM模型
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# rerank 模型
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")

# BM25相关配置
BM25_K1 = float(os.getenv('BM25_K1', '1.5'))
BM25_B = float(os.getenv('BM25_B', '0.75'))
BM25_MIN_FREQ = int(os.getenv('BM25_MIN_FREQ', '5'))
BM25_MAX_VOCAB_SIZE = int(os.getenv('BM25_MAX_VOCAB_SIZE', '10000'))
BM25_CUSTOM_DICT_PATH = os.getenv('BM25_CUSTOM_DICT_PATH', 'dicts/warhammer40k.txt')
BM25_MODEL_PATH = os.getenv('BM25_MODEL_PATH', 'models/bm25_model.pkl')
BM25_VOCAB_PATH = os.getenv('BM25_VOCAB_PATH', 'models/bm25_vocab.json')
PINECONE_SPARSE_DIMENSION = int(os.getenv('PINECONE_SPARSE_DIMENSION', '10000'))

def get_pinecone_index():
    """
    初始化并返回Pinecone索引对象。
    """
    if not PINECONE_API_KEY or not PINECONE_INDEX:
        logger.error("Pinecone API key or index name not configured.")
        raise ValueError("Pinecone configuration is missing.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    logger.info(f"Successfully connected to Pinecone index '{PINECONE_INDEX}'.")
    return index

def get_embedding_model():
    """
    初始化并返回嵌入模型对象。
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured.")
        raise ValueError("OpenAI API key is missing.")
        
    embeddings = OpenAIEmbeddings(
        model=EMBADDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        dimensions=1024  # 强制指定输出维度为1024，与index一致
    )
    logger.info(f"Embedding model '{EMBADDING_MODEL}' initialized with dimensions=1024.")
    return embeddings

# 日志配置
LOG_FORMAT = os.getenv("LOG_FORMAT", '%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE = os.getenv("LOG_FILE", 'app.log')
LOG_LEVEL = os.getenv("LOG_LEVEL", 'INFO')

# 应用配置
APP_TITLE = os.getenv("APP_TITLE", "战锤40K规则助手")
APP_ICON = os.getenv("APP_ICON", "⚔️")
APP_HEADER = os.getenv("APP_HEADER", "") 