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

# 模型配置
DEFAULT_TOP_K = 25  # 默认返回结果数量


# 答案生成配置
DEFAULT_TEMPERATURE = 0.2  # 默认温度参数，控制生成文本的随机性
MAX_ANSWER_TOKENS = 2000  # 答案生成最大token数

# 结果整合配置
DEDUPLICATION_THRESHOLD = 0.8  # 去重阈值，相似度超过此值视为重复

# 并发配置
DEFAULT_MAX_WORKERS = 4  # 默认RAGAS最大并发工作线程数





# ========== 网络连接配置 ==========
CONNECT_TIMEOUT = 120  # 连接超时（秒）
READ_TIMEOUT = 120  # 读取超时（秒）
MAX_RETRIES = 4  # 重试次数
RETRY_DELAY = 2.0  # 重试延迟（秒）

# 指数退避配置
USE_EXPONENTIAL_BACKOFF = True  # 是否使用指数退避策略
BACKOFF_FACTOR = 2.0  # 退避因子
MAX_BACKOFF_DELAY = 30.0  # 最大退避延迟（秒）

# LLM类型,包括 Openai, Deepseek, Gemini
LLM_TYPE="deepseek"
# OPEN AI LLM模型
LLM_MODEL = "gpt-4o"  # OpenAI LLM模型名称
LLM_IMAGE_MODEL = "gpt-4o"  # OpenAI图像模型名称
# 嵌入模型
EMBADDING_MODEL = "text-embedding-3-large"  # OpenAI嵌入模型名称
OPENAI_DIMENSION = 2048  # OpenAI嵌入向量维度

# QWEN AI LLM 模型
EMBEDDING_MODEL_QWEN = "text-embedding-v4"  # QWEN嵌入模型名称
RERANK_MODEL_QWEN = "text-embedding-v4"  # QWEN重排序模型名称
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")  # QWEN API密钥（保留环境变量）
QWEN_DIMENSION = 2048  # QWEN嵌入向量维度

# DEEPSEEK API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_SERVER="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-chat"

#检索分析调试配置
DEBUG_ANALYZE = True  # 搜索分析调试开关，默认关闭
# 搜索分析日志配置
SEARCH_ANALYSIS_LOG_FILE = "log/search_analysis.log"  # 搜索分析日志文件路径
SEARCH_ANALYSIS_LOG_MAX_SIZE = 5 * 1024 * 1024  # 搜索分析日志最大文件大小（5MB）

# Hybrid检索配置
#当 α = 0 时，完全依赖稀疏检索（纯 BM25/TF-IDF）；
#当 α = 1 时，完全依赖密集检索（纯语义向量匹配）；
HYBRID_ALPHA = 0.3
# 混合搜索算法选择
# 可选值: 'pipeline' (默认) 或 'rrf'
HYBRID_ALGORITHM = 'pipeline'

# ========== RRF混合搜索配置 ==========
# RRF (Reciprocal Rank Fusion) 算法参数配置
RRF_RANK_CONSTANT = 10  # RRF排名常数，越小区分度越高，建议5-15
RRF_WINDOW_MULTIPLIER = 3  # RRF窗口大小倍数，越大捕获文档越多，建议3-5
RRF_MAX_WINDOW_SIZE = 300  # RRF窗口大小最大值，防止过大的计算开销
RRF_SPARSE_TERMS_BOOST = 1.2  # RRF稀疏向量terms查询的boost值，提高稀疏向量查询权重
RRF_DEFAULT_RANK_CONSTANT = 20  # 标准RRF常数，用于RRF算法中的平滑参数

# RRF混合搜索权重配置
RRF_SPARSE_WEIGHT = 0.8  # RRF稀疏向量查询权重
RRF_DENSE_WEIGHT = 1.2 # RRF密集向量查询权重

# ========== Pipeline混合搜索配置 ==========
# Pipeline混合搜索算法参数配置
PIPELINE_NORMALIZATION_TECHNIQUE = "min_max"  # 归一化技术：min_max, z_score, decimal_scaling, log_scale
PIPELINE_COMBINATION_TECHNIQUE = "arithmetic_mean"  # 组合技术：arithmetic_mean, geometric_mean, harmonic_mean, weighted_sum
PIPELINE_BM25_BOOST = 1.5  # BM25搜索boost值，影响BM25结果权重
PIPELINE_VECTOR_BOOST = 0.8  # 向量搜索boost值，影响向量搜索结果权重

# ========== Match Phrase查询配置 ==========
# Match Phrase查询算法参数配置
MATCH_PHRASE_CONFIG = {
    'enabled': True,                    # 是否启用match_phrase查询
    'default_boost': 1.5,              # 默认boost值
    'fuzziness': 'AUTO',               # 模糊匹配级别
    'operator': 'or',                  # 操作符
    'max_expansions': 50,              # 最大扩展数
    'prefix_length': 0,                # 前缀长度
    'min_score': 0.1,                  # 最小分数阈值
    
    # 策略选择阈值
    'query_text_min_length': 2,        # 查询文本最小长度
    'sparse_vector_min_terms': 3,      # 稀疏向量最小术语数
    
    # 混合查询权重
    'match_phrase_weight': 2.0,        # match_phrase权重
    'terms_weight': 1.0,               # terms权重
    
    # 模糊匹配过滤配置（DeepSeek建议）
    'fuzzy_filters': [
        {"min_term_freq": 0.01},       # 保护低频专有名词（如"艾达灵族"）
        {"max_doc_freq": 0.5}          # 忽略高频通用词（如"单位"、"技能"）
    ],
    
    # 词频过滤阈值
    'term_frequency_thresholds': {
        'min_term_freq': 0.01,         # 最小词频，保护专有名词
        'max_doc_freq': 0.5,           # 最大文档频率，过滤高频词
        'min_word_length': 2,          # 最小词长度
        'max_word_length': 20          # 最大词长度
    }
}


# ========== 混合搜索优化配置 ==========
# 搜索字段权重配置
SEARCH_FIELD_WEIGHTS = {
    "section_heading": 1.0,    # 章节标题权重
    "sub_section": 1.3,        # 子章节权重（V4新增字段）
    "content_type": 1.0,       # 内容类型权重
    "chunk_type": 1.0,         # 块类型权重
    "faction": 1.0             # 阵营权重
}

# 标题权重配置
HEADING_WEIGHTS = {
    'h1': 1.0,
    'h2': 1.4,
    'h3': 1.8,
    'h4': 1.0,
    'h5': 0.8,
    'h6': 0.6
}

# 内容类型权重配置
CONTENT_TYPE_WEIGHTS = {
    'corerule': 1.5,
    'unit_data': 1.3,
}

# 查询长度阈值配置
QUERY_LENGTH_THRESHOLDS = {
    'short': 4,
    'medium': 8
}

# 正文权重配置
BOOST_WEIGHTS = {
    'long_query': 1.4,    # 长查询权重
    'medium_query': 1.2,  # 中等查询权重
    'short_query': 0.8    # 短查询权重
}

# rerank 模型
RERANK_MODEL = "bge-reranker-v2-m3"  # 默认重排序模型
ALIYUN_RERANK_MODEL = "gte-rerank-v2"  # 阿里云重排序模型
RERANK_TOPK = 25  # 重排序返回结果数量

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
PINECONE_SPARSE_DIMENSION = 10000  # Pinecone稀疏向量维度

phrase_weight_scorer = PhraseWeightScorer(
    df_thresholds=(1, 2, 3),
    pmi_threshold=3.0,
    boost_factors=(0.2, 0.1, 0.0, -0.5)
)

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