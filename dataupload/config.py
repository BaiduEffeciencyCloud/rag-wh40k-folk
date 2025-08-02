"""
数据处理模块配置
包含数据切片、向量上传、专业术语词典等所有数据处理相关配置

注意：部分配置与根目录 config.py 重复，后续需要统一迁移
"""

import os
from typing import Dict, Any

# ==================== 数据上传配置（新增） ====================
DATA_UPLOAD_CONFIG = {
    "chunk_size": int(os.getenv('CHUNK_SIZE', '1000')),
    "chunk_overlap": int(os.getenv('CHUNK_OVERLAP', '200')),
    "max_chunks_per_document": int(os.getenv('MAX_CHUNKS_PER_DOC', '50'))
}

# ==================== BM25配置（与根目录 config.py 重复，需要迁移） ====================
# TODO: 迁移到根目录 config.py，这里保持兼容性
BM25_CONFIG = {
    # 基础参数（与根目录重复）
    "k1": float(os.getenv('BM25_K1', '1.5')),                    # 重复：config.py BM25_K1
    "b": float(os.getenv('BM25_B', '0.75')),                      # 重复：config.py BM25_B
    "min_freq": int(os.getenv('BM25_MIN_FREQ', '3')),             # 重复：config.py BM25_MIN_FREQ
    "max_vocab_size": int(os.getenv('BM25_MAX_VOCAB_SIZE', '4000')), # 重复：config.py BM25_MAX_VOCAB_SIZE
    
    # 新增参数（专业术语词典相关）
    "min_pmi": float(os.getenv('BM25_MIN_PMI', '3.0')),           # 新增：专业术语PMI阈值
    "max_len": int(os.getenv('BM25_MAX_LEN', '6')),               # 新增：专业术语最大长度
    
    # 路径配置（与根目录重复）
    "model_dir": os.getenv('BM25_MODEL_DIR', 'models'),           # 重复：config.py BM25_MODEL_DIR
    "vocab_dir": os.getenv('BM25_VOCAB_DIR', 'dict'),             # 重复：config.py BM25_VOCAB_DIR
    "custom_dict_path": os.getenv('BM25_CUSTOM_DICT_PATH', 'dicts/warhammer40k.txt'), # 重复：config.py BM25_CUSTOM_DICT_PATH
    
    # 性能配置（与根目录重复）
    "batch_size": int(os.getenv('BM25_BATCH_SIZE', '1000')),      # 重复：config.py BM25_BATCH_SIZE
    "cache_size": int(os.getenv('BM25_CACHE_SIZE', '1000')),      # 重复：config.py BM25_CACHE_SIZE
    
    # 功能开关（与根目录重复）
    "enable_incremental": os.getenv('BM25_ENABLE_INCREMENTAL', 'true').lower() == 'true', # 重复：config.py BM25_ENABLE_INCREMENTAL
    "save_after_update": os.getenv('BM25_SAVE_AFTER_UPDATE', 'true').lower() == 'true',   # 重复：config.py BM25_SAVE_AFTER_UPDATE
}

# ==================== Pinecone配置（与根目录 config.py 重复，需要迁移） ====================
# TODO: 迁移到根目录 config.py，这里保持兼容性
PINECONE_MAX_SPARSE_VALUES = int(os.getenv('PINECONE_MAX_SPARSE_VALUES', '2048'))  # 重复：config.py PINECONE_MAX_SPARSE_VALUES

# ==================== 短语权重评分器配置（与根目录 config.py 重复，需要迁移） ====================
# TODO: 迁移到根目录 config.py，这里保持兼容性
from dataupload.phrase_weight import PhraseWeightScorer
phrase_weight_scorer = PhraseWeightScorer(
    df_thresholds=(1, 2, 3),
    pmi_threshold=3.0,
    boost_factors=(0.2, 0.1, 0.0, -0.5)
)  # 重复：config.py phrase_weight_scorer

# ==================== 专业术语词典配置（新增） ====================
WH40K_VOCAB_PATH = os.getenv('WH40K_VOCAB_PATH', 'dict/wh40k_vocabulary/')

# 功能开关
ENABLE_WH40K_WHITELIST = os.getenv('ENABLE_WH40K_WHITELIST', 'true').lower() == 'true'
ENABLE_WH40K_ENHANCEMENT = os.getenv('ENABLE_WH40K_ENHANCEMENT', 'false').lower() == 'true'
ENABLE_WH40K_SYNONYM_EXPANSION = os.getenv('ENABLE_WH40K_SYNONYM_EXPANSION', 'false').lower() == 'true'

# 权重配置
WH40K_WEIGHT_FACTOR = float(os.getenv('WH40K_WEIGHT_FACTOR', '1.5'))

# 类别权重（第二阶段）
CATEGORY_WEIGHTS = {
    "units": 2.0,
    "factions": 1.8, 
    "skills": 1.5,
    "weapons": 1.5,
    "mechanics": 1.3
}

# BM25权重增强配置（第二阶段）
BM25_WEIGHT_ENHANCEMENT_CONFIG = {
    "enable_wh40k_enhancement": ENABLE_WH40K_ENHANCEMENT,
    "category_weights": CATEGORY_WEIGHTS,
    "default_weight_factor": WH40K_WEIGHT_FACTOR
}

# 白名单生成配置
WHITELIST_GENERATION_CONFIG = {
    "include_auto_discovery": True,
    "min_freq": BM25_CONFIG["min_freq"],      # 复用BM25配置
    "min_pmi": BM25_CONFIG["min_pmi"],        # 复用BM25配置
    "max_len": BM25_CONFIG["max_len"]         # 复用BM25配置
}

# 同义词扩展配置（第三阶段）
SYNONYM_EXPANSION_CONFIG = {
    "enable_query_expansion": True,
    "enable_result_merging": True,
    "max_synonyms_per_term": 5
}

# ==================== 向量上传配置（新增） ====================
VECTOR_UPLOAD_CONFIG = {
    "batch_size": int(os.getenv('VECTOR_BATCH_SIZE', '100')),
    "max_retries": int(os.getenv('VECTOR_MAX_RETRIES', '3')),
    "timeout": int(os.getenv('VECTOR_TIMEOUT', '30')),
    "enable_hybrid_search": os.getenv('ENABLE_HYBRID_SEARCH', 'true').lower() == 'true'
}

# ==================== 存储配置（新增） ====================
STORAGE_CONFIG = {
    "pinecone_index_name": os.getenv('PINECONE_INDEX', ''),        # 复用：config.py PINECONE_INDEX
    "pinecone_namespace": os.getenv('PINECONE_NAMESPACE', 'wh40k'),
    "local_storage_path": os.getenv('LOCAL_STORAGE_PATH', 'data/chunks/'),
    "backup_enabled": os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
}

# ==================== 统一配置字典 ====================
WH40K_CONFIG = {
    "vocab_path": WH40K_VOCAB_PATH,
    "enable_wh40k_whitelist": ENABLE_WH40K_WHITELIST,
    "enable_wh40k_enhancement": ENABLE_WH40K_ENHANCEMENT,
    "enable_wh40k_synonym_expansion": ENABLE_WH40K_SYNONYM_EXPANSION,
    "wh40k_weight_factor": WH40K_WEIGHT_FACTOR,
    "category_weights": CATEGORY_WEIGHTS,
    "whitelist_generation": WHITELIST_GENERATION_CONFIG,
    "synonym_expansion": SYNONYM_EXPANSION_CONFIG,
    "data_upload": DATA_UPLOAD_CONFIG,
    "bm25": BM25_CONFIG,
    "vector_upload": VECTOR_UPLOAD_CONFIG,
    "storage": STORAGE_CONFIG
}

# ==================== 迁移标记 ====================
"""
配置迁移计划：

1. 重复配置项（需要迁移到根目录 config.py）：
   - BM25基础参数：k1, b, min_freq, max_vocab_size
   - BM25路径配置：model_dir, vocab_dir, custom_dict_path
   - BM25性能配置：batch_size, cache_size
   - BM25功能开关：enable_incremental, save_after_update
   - Pinecone配置：PINECONE_MAX_SPARSE_VALUES

2. 新增配置项（保留在 data_config.py）：
   - 专业术语词典配置：WH40K_VOCAB_PATH, ENABLE_WH40K_WHITELIST 等
   - 数据上传配置：CHUNK_SIZE, CHUNK_OVERLAP 等
   - 向量上传配置：VECTOR_BATCH_SIZE 等

3. 迁移步骤：
   - 第一阶段：保持兼容性，两个配置文件并存
   - 第二阶段：逐步迁移重复配置到根目录 config.py
   - 第三阶段：统一配置管理，data_config.py 只保留数据处理特有配置
""" 