"""
Post-Search 策略配置文件
用于管理不同查询类型对应的Post-Search处理策略和处理器参数
"""

# 处理器参数配置
from config import RERANK_MODEL
from typing import Dict, List, Any

# Post-Search 策略配置
# 根据查询类型定义对应的Post-Search处理策略
POST_SEARCH_STRATEGIES = {
    'simple_rule': [],  # 简单规则查询不需要后处理
    'complex_reasoning': ['rerank', 'mmr'],  # 复杂推理需要重排序和多样性
    'enumeration': ['dedup', 'mmr']  # 穷举查询需要去重和多样性
}


PROCESSOR_CONFIGS = {
    'mmr': {
        'enabled': True,
        'lambda_param': 0.6,  # MMR参数，平衡相关性和多样性的平衡 (0=纯多样性, 1=纯相关性)
        'select_n': 25,       # 最终选择的文档数量
        'enable_mmr': True,   # 是否启用MMR处理
        'min_text_length': 10, # 处理的最小文本长度
        
        # 相关性过滤参数（实际使用）
        'relevance_threshold': 0.2,    # 相关性阈值，低于此值的结果被过滤
        'enable_relevance_filter': True,  # 是否启用相关性过滤
        
        # Score标准化参数（实际使用）
        'enable_score_normalization': True,  # 是否启用score标准化
        'default_normalized_score': 0.5,    # 标准化时的默认分数
        'normalization_method': 'log_minmax',  # 标准化方法
        
        # 输出排序参数（实际使用）
        'output_order': 'mmr_score',   # 输出排序方式
        'reverse_order': False,        # 是否逆序排列
        
        # MMR算法优化参数（实际使用）
        'max_candidates': 200,  # 进入MMR的最大候选文档数量
        'dynamic_lambda': True,  # 是否启用动态λ调整
        'use_optimized_mmr': True,  # 是否使用优化的MMR算法
        'base_lambda': 0.5,        # 动态λ调整的基础值
        'short_query_threshold': 17,  # 短查询长度阈值
        'medium_query_threshold': 30, # 中等查询长度阈值
        
        # 动态lambda调整参数（新增）
        'short_adjustment': 0.3,        # 短查询调整值
        'long_adjustment': 0.2,         # 长查询调整值
        'max_lambda': 0.9,              # lambda最大值
        'min_lambda': 0.3,              # lambda最小值
        
        # 零向量保护参数（实际使用）
        'zero_vector_protection': True,  # 是否启用零向量保护
        'zero_vector_strategy': 'return_zero',  # 零向量处理策略
        'min_norm_threshold': 1e-8,     # 最小范数阈值
        
        # 缓存优化参数（实际使用）
        'use_cache_optimization': True,  # 是否使用缓存优化的MMR算法
        
        # 回退机制参数（实际使用）
        'fallback_levels': 3,  # 多级回退的级别数量
        'fallback_strategy': 'multi_level',  # 回退策略
        'min_results_threshold': 3,  # 最小结果数量阈值
        'enable_fallback_monitoring': True,  # 是否启用回退监控
    },
    'filter': {
        'enabled': True,
        'min_score': 0.3,  # 最小相关性分数
        'max_length': 800,  # 最大文本长度
        'exclude_patterns': [],  # 排除的文本模式
    },
    'dedup': {
        'enabled': True,
        'similarity_threshold': 0.9,  # 去重相似度阈值
        'method': 'semantic',  # 去重方法：semantic/text_hash
        'preserve_order': True,  # 是否保持原始顺序
    }
}

# 处理器执行顺序配置
PROCESSOR_ORDER = {
    'default': ['filter'],  # 默认只做过滤，等同于simple_rule
    'simple_rule': ['filter'],
    'complex_reasoning': ['filter', 'rerank', 'mmr'],
    'enumeration': ['filter', 'dedup', 'mmr']
}

# 性能配置
PERFORMANCE_CONFIG = {
    'max_processing_time': 30,  # 最大处理时间（秒）
    'enable_parallel': True,  # 是否启用并行处理
    'max_workers': 4,  # 最大工作线程数
    'cache_enabled': True,  # 是否启用缓存
    'cache_ttl': 3600,  # 缓存生存时间（秒）
}

# 错误处理配置
ERROR_HANDLING_CONFIG = {
    'continue_on_error': True,  # 处理器出错时是否继续
    'fallback_strategy': 'simple',  # 降级策略
    'log_errors': True,  # 是否记录错误
    'retry_count': 2,  # 重试次数
}

# 配置验证规则
CONFIG_VALIDATION_RULES = {
    'required_processors': ['filter'],  # 必需的处理器
    'max_strategy_length': 5,  # 最大策略长度
    'allowed_processors': ['rerank', 'mmr', 'filter', 'dedup'],  # 允许的处理器
}

def get_post_search_strategy(query_type: str) -> List[str]:
    """
    根据查询类型获取Post-Search策略
    
    Args:
        query_type: 查询类型
        
    Returns:
        List[str]: 处理策略列表
    """
    return POST_SEARCH_STRATEGIES.get(query_type, [])

def get_processor_config(processor_name: str) -> Dict[str, Any]:
    """
    获取处理器配置
    
    Args:
        processor_name: 处理器名称
        
    Returns:
        Dict[str, Any]: 处理器配置
    """
    return PROCESSOR_CONFIGS.get(processor_name, {})

def get_processor_order(query_type: str = 'default') -> List[str]:
    """
    获取处理器执行顺序
    
    Args:
        query_type: 查询类型
        
    Returns:
        List[str]: 处理器执行顺序
    """
    return PROCESSOR_ORDER.get(query_type, PROCESSOR_ORDER['default'])

def validate_strategy(strategy: List[str]) -> bool:
    """
    验证策略配置
    
    Args:
        strategy: 策略列表
        
    Returns:
        bool: 是否有效
    """
    if not strategy:
        return True
    
    if len(strategy) > CONFIG_VALIDATION_RULES['max_strategy_length']:
        return False
    
    allowed_processors = CONFIG_VALIDATION_RULES['allowed_processors']
    return all(processor in allowed_processors for processor in strategy)

def get_enabled_processors() -> List[str]:
    """
    获取已启用的处理器列表
    
    Returns:
        List[str]: 已启用的处理器列表
    """
    enabled = []
    for processor_name, config in PROCESSOR_CONFIGS.items():
        if config.get('enabled', True):
            enabled.append(processor_name)
    return enabled
