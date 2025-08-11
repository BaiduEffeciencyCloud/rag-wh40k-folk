"""
配置文件，用于管理项目范围内的公有字段和API密钥
"""

import logging
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from dataupload.phrase_weight import PhraseWeightScorer
# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
def setup_logging(log_file_path=None):
    """设置日志配置，支持文件和控制台输出"""
    # 清除现有的handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = [logging.StreamHandler()]
    
    if log_file_path:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=handlers,
        force=True  # 强制重新配置
    )

# 默认只配置控制台输出
setup_logging()

logger = logging.getLogger(__name__)

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

#Connection 配置, 使用哪种数据库[opesearch, pinecone]
CONNECTION="opensearch"

# OpenSearch API配置
OPENSERACH_URI=os.getenv("OPENSERACH_URI")
OPENSEARCH_USERNAME=os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD=os.getenv("OPENSEARCH_PASSWORD")
OPENSERACH_INDEX="wh40k-test"

# Pinecone API配置
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not found in environment variables")
PINECONE_INDEX = "40ktest"

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
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
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
MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "20"))  # 答案生成时使用的最大上下文结果数



# OPEN AI LLM模型
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_IMAGE_MODEL=os.getenv("LLM_IMAGE_MODEL", "gpt-4o")
# 嵌入模型
EMBADDING_MODEL = os.getenv("EMBADDING_MODEL", "text-embedding-3-large")

# QWEN AI LLM 模型
EMBEDDING_MODEL_QWEN = os.getenv("EMBEDDING_MODEL_QWEN", "text-embedding-v4")
RERANK_MODEL_QWEN = os.getenv("RERANK_MODEL_QWEN", "text-embedding-v4")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_DIMENSION = 2048

# Hybrid检索配置
#当 α = 0 时，完全依赖稀疏检索（纯 BM25/TF-IDF）；
#当 α = 1 时，完全依赖密集检索（纯语义向量匹配）；
HYBRID_ALPHA = float(os.getenv('HYBRID_ALPHA', '0.3'))

# rerank 模型
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_TOPK = 20

# BM25相关配置
BM25_K1 = float(os.getenv('BM25_K1', '1.5'))
BM25_B = float(os.getenv('BM25_B', '0.75'))
BM25_MIN_FREQ = int(os.getenv('BM25_MIN_FREQ', '3'))
BM25_MAX_VOCAB_SIZE = int(os.getenv('BM25_MAX_VOCAB_SIZE', '4000'))
BM25_CUSTOM_DICT_PATH = os.getenv('BM25_CUSTOM_DICT_PATH', 'dicts/warhammer40k.txt')
BM25_MODEL_PATH = os.getenv('BM25_MODEL_PATH', 'models/bm25_model.pkl')
BM25_VOCAB_PATH = os.getenv('BM25_VOCAB_PATH', 'dict/')

# BM25文件路径配置
BM25_MODEL_DIR = os.getenv('BM25_MODEL_DIR', 'models')
BM25_VOCAB_DIR = os.getenv('BM25_VOCAB_DIR', 'dict')
BM25_MODEL_FILENAME = os.getenv('BM25_MODEL_FILENAME', 'bm25_model.pkl')
BM25_VOCAB_FILENAME_PATTERN = os.getenv('BM25_VOCAB_FILENAME_PATTERN', 'bm25_vocab_{timestamp}.json')

# BM25训练配置
BM25_ENABLE_INCREMENTAL = os.getenv('BM25_ENABLE_INCREMENTAL', 'true').lower() == 'true'
BM25_SAVE_AFTER_UPDATE = os.getenv('BM25_SAVE_AFTER_UPDATE', 'true').lower() == 'true'

# BM25性能配置
BM25_BATCH_SIZE = int(os.getenv('BM25_BATCH_SIZE', '1000'))
BM25_CACHE_SIZE = int(os.getenv('BM25_CACHE_SIZE', '1000'))



# Pinecone稀疏向量限制
PINECONE_MAX_SPARSE_VALUES = int(os.getenv('PINECONE_MAX_SPARSE_VALUES', '2048'))  # Pinecone单个稀疏向量的最大非零元素数量

PINECONE_SPARSE_DIMENSION = int(os.getenv('PINECONE_SPARSE_DIMENSION', '10000'))

phrase_weight_scorer = PhraseWeightScorer(
    df_thresholds=(1, 2, 3),
    pmi_threshold=3.0,
    boost_factors=(0.2, 0.1, 0.0, -0.5)
)

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
    # 增加shape打印，便于排查实际输出维度
    try:
        test_vec = embeddings.embed_query("test shape")
        logger.info(f"[DEBUG] embedding shape: {len(test_vec)}")
        print(f"[DEBUG] embedding shape: {len(test_vec)}")
    except Exception as e:
        logger.warning(f"[DEBUG] embedding shape 获取失败: {e}")
    return embeddings

def get_qwen_embedding_model():
    """
    初始化并返回QWEN嵌入模型对象。
    使用dashscope的TextEmbedding类方法调用。
    """
    # 导入dashscope并设置全局API key
    import dashscope
    from dashscope.embeddings import TextEmbedding
    if not QWEN_API_KEY:
        logger.error("QWEN API key not configured.")
        raise ValueError("QWEN API key is missing.")

    dashscope.api_key = QWEN_API_KEY
    
    # 记录初始化信息
    logger.info(f"QWEN embedding model '{EMBEDDING_MODEL_QWEN}' initialized with API key.")
    
    return TextEmbedding

# 日志配置
LOG_FORMAT = os.getenv("LOG_FORMAT", '%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE = os.getenv("LOG_FILE", 'app.log')
LOG_LEVEL = os.getenv("LOG_LEVEL", 'INFO')

# 应用配置
APP_TITLE = os.getenv("APP_TITLE", "战锤40K规则助手")
APP_ICON = os.getenv("APP_ICON", "⚔️")
APP_HEADER = os.getenv("APP_HEADER", "")

# 默认输入输出目录常量
INPUT_DIR = 'test/testdata'
OUTPUT_DIR = 'dict' 