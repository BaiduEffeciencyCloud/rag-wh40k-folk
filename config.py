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

# ALIYUN OSS 配置
OSS_ACCESS_KEY = os.getenv("OSS_ACCESS_KEY")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_SECRET")
OSS_ENDPOINT = "https://oss-cn-beijing.aliyuncs.com"
OSS_BUCKET_NAME_DICT = "rag-dict"
OSS_BUCKET_NAME_MODELS = "rag-models"
OSS_BUCKET_NAME_PARAMS = "rag-params"
OSS_REGION = "cn-beijing"


# OpenSearch API配置
OPENSEARCH_URI=os.getenv("OPENSEARCH_URI")
OPENSEARCH_USERNAME=os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD=os.getenv("OPENSEARCH_PASSWORD")
OPENSEARCH_INDEX="40ktest"

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

# 答案生成配置
DEFAULT_TEMPERATURE = 0.3  # 默认温度参数，控制生成文本的随机性
MAX_ANSWER_TOKENS = 3000  # 答案生成最大token数
# 并发配置
DEFAULT_MAX_WORKERS = 4  # 默认RAGAS最大并发工作线程数

# 指数退避配置
USE_EXPONENTIAL_BACKOFF = True  # 是否使用指数退避策略
BACKOFF_FACTOR = 2.0  # 退避因子
MAX_BACKOFF_DELAY = 30.0  # 最大退避延迟（秒）

# LLM类型,包括 Openai, Deepseek, Gemini,QWEN
LLM_TYPE="qwen"
EMBEDDING_TYPE="qwen"
DB_TYPE="opensearch"
# OPEN AI LLM模型
LLM_MODEL = "gpt-4o"  # OpenAI LLM模型名称
LLM_IMAGE_MODEL = "gpt-4o"  # OpenAI图像模型名称
# 嵌入模型
EMBADDING_MODEL = "text-embedding-3-large"  # OpenAI嵌入模型名称
OPENAI_DIMENSION = 2048  # OpenAI嵌入向量维度

# QWEN AI LLM 模型
EMBEDDING_MODEL_QWEN = "text-embedding-v4"  # QWEN嵌入模型名称
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")  # QWEN API密钥（保留环境变量）
QWEN_DIMENSION = 2048  # QWEN嵌入向量维度
QWEN_LLM_SERVER = "https://dashscope.aliyuncs.com/compatible-mode/v1" #QWEN 服务器地址
QWEN_LLM_MODEL = "qwen-plus" #QWEN 模型名称

# DEEPSEEK API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_SERVER="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-chat"


#检索分析调试配置
DEBUG_ANALYZE = True  # 搜索分析调试开关，默认关闭


# ========== Pipeline混合搜索配置 ==========
# Pipeline混合搜索算法参数配置
#PIPELINE_NORMALIZATION_TECHNIQUE = "min_max"  # 归一化技术：min_max, z_score, decimal_scaling, log_scale
#PIPELINE_COMBINATION_TECHNIQUE = "arithmetic_mean"  # 组合技术：arithmetic_mean, geometric_mean, harmonic_mean, weighted_sum
#PIPELINE_BM25_BOOST = 1.5  # BM25搜索boost值，影响BM25结果权重
#PIPELINE_VECTOR_BOOST = 1.0  # 向量搜索boost值，影响向量搜索结果权重

# ========== Match Phrase查询配置 ==========
# Match Phrase查询算法参数配置

# ========== 混合搜索优化配置 ==========
# OpenSearch Analyzer配置
#OPENSEARCH_INDEX_ANALYZER = "warhammer_index_analyzer"      # 索引时使用的analyzer
OPENSEARCH_SEARCH_ANALYZER = "warhammer_search_analyzer"    # 搜索时使用的analyzer

# 搜索字段权重配置
SEARCH_FIELD_WEIGHTS = {
    "section_heading": 1.0,    # 章节标题权重
    "sub_section": 1.2,        # 子章节权重（V4新增字段）
    "content_type": 1.0,       # 内容类型权重
    "chunk_type": 1.4,         # 块类型权重
    "faction": 1.3             # 阵营权重
}

# rerank 模型
RERANK_MODEL = "bge-reranker-v2-m3"  # 默认重排序模型
ALIYUN_RERANK_MODEL = "gte-rerank-v2"  # 阿里云重排序模型
RERANK_TOPK = 15  # 重排序返回结果数量

# BM25相关配置
BM25_K1 = 1.5  # BM25算法k1参数，控制词频饱和速度

# ========== 缓存配置 ==========
ENABLE_PIPELINE_CACHE = False   # 是否启用Pipeline缓存，线下开发时关闭，线上服务时打开
BM25_B = 0.75  # BM25算法b参数，控制文档长度归一化程度
BM25_MIN_FREQ = 3  # 最小词频，低于此值的词会被过滤
BM25_MAX_VOCAB_SIZE = 4000  # 最大词汇表大小
BM25_CUSTOM_DICT_PATH = 'dicts/warhammer40k.txt'  # 自定义词典路径
BM25_MODEL_PATH = 'models/bm25_model.pkl'  # BM25模型文件路径
BM25_VOCAB_PATH = 'dict/'  # 词汇表文件目录

# BM25文件路径配置
BM25_MODEL_DIR = 'models'  # 模型文件目录
BM25_VOCAB_DIR = 'dict'  # 词汇表文件目录
BM25_MODEL_FILENAME = 'bm25_model.pkl'  # 模型文件名
BM25_VOCAB_FILENAME_PATTERN = 'bm25_vocab_{timestamp}.json'  # 词汇表文件名模式

# BM25训练配置
BM25_ENABLE_INCREMENTAL = True  # 是否启用增量训练
BM25_SAVE_AFTER_UPDATE = True  # 更新后是否保存模型

# BM25性能配置
BM25_BATCH_SIZE = 1000  # 批处理大小
BM25_CACHE_SIZE = 1000  # 缓存大小

# Pinecone稀疏向量限制
PINECONE_MAX_SPARSE_VALUES = 2048  # Pinecone单个稀疏向量的最大非零元素数量

phrase_weight_scorer = PhraseWeightScorer(
    df_thresholds=(1, 2, 3),
    pmi_threshold=3.0,
    boost_factors=(0.2, 0.1, 0.0, -0.5)
)

# 默认输入输出目录常量
INPUT_DIR = 'test/testdata'
OUTPUT_DIR = 'dict' 
# 词典的导出位置
VOCAB_EXPORT_DIR = 'dict/wh40k_vocabulary'

# ========== 网络连接配置 ==========
CONNECT_TIMEOUT = 120  # 连接超时（秒）
READ_TIMEOUT = 120  # 读取超时（秒）
MAX_RETRIES = 4  # 重试次数
RETRY_DELAY = 2.0  # 重试延迟（秒）

# ========== ALIYUN配置 ==========

# === OSS配置 ===
CLOUD_STORAGE=True
# ========== OSS 内存缓存配置 ==========
# 是否启用 OSS 内存缓存
OSS_CACHE_ENABLED = True
# 缓存条目 TTL（秒）
OSS_CACHE_TTL_SECONDS = 60
# LRU 最大条目数
OSS_CACHE_MAX_ITEMS = 512
# 可选：总字节上限；None 表示不限制（暂未使用）
OSS_CACHE_MAX_BYTES = None
# 校验模式：'etag' | 'last_modified' | 'both'
OSS_CACHE_VALIDATE_MODE = 'etag'
# 允许缓存的 Key 前缀列表；为空表示不限制（全缓存）
OSS_CACHE_ALLOWED_KEY_PREFIXES = ['bm25_vocab_freq_', 'all_docs_dict_', 'vocabulary/', 'templates/','pipeline/','search/']
# 仅缓存不超过该大小（字节）的对象
OSS_CACHE_MAX_OBJECT_SIZE = 5 * 1024 * 1024

#====答案输出=====
STREAM_MODE = True


