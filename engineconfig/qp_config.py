"""
查询处理配置文件
管理意图识别和自适应检索管道配置
"""

from typing import Dict, List, Any

# 意图识别配置
INTENT_CLASSIFIER_CONFIG = {
    'model_path': 'models/intent_classifier.pkl',
    'feature_extractor': 'intent_training.feature.feature_engineering',
    'high_confidence_threshold': 0.5,    # 高置信度阈值
    'low_confidence_threshold': 0.3,     # 低置信度阈值
    'auto_confidence_threshold': 0.85,   # 自动信任阈值
    'unknown_intent_label': 'unknown',
    'non_wh40k_label': 'non_wh40k'
}

# 不确定性检测配置
UNCERTAINTY_CONFIG = {
    'methods': ['entropy', 'confidence', 'margin'],
    'combined_weight': {
        'entropy': 0.4,
        'confidence': 0.4,
        'margin': 0.2
    },
    'review_threshold': 0.5,
    'auto_confidence_threshold': 0.85
}

# 意图类型定义
INTENT_TYPES = {
    'compare': {
        'keywords': ['哪个', '相比', '比较', '最远','最高','最低', '更高', '更远', '哪个更高','哪个更'],
        'description': '比较类查询，需要对比分析'
    },
    'rule': {
        'keywords': ['什么情况下', '如何触发', '规则', '何时', '条件','当...时'],
        'description': '规则类查询，需要条件推理'
    },
    'list': {
        'keywords': ['有哪些', '多少个', '列举', '哪些', '多少种'],
        'description': '列举类查询，需要全面检索'
    },
    'query': {
        'keywords': ['什么是', '如何', '解释', '说明','定义','是什么'],
        'description': '一般查询，需要概念理解'
    }
}

# COT模板配置
COT_TEMPLATES = {
    'compare': {
        'intent': 'comparison_analysis',
        'patterns': [
            {
                'type': 'direct_comparison',
                'keywords': ['哪个', '相比', '比较', '更强', '更好', '更远', '哪个更高','哪个...更'],
                'chain_of_thought': [
                    {'step': 1, 'reasoning': '识别比较对象：提取Query中的两个或多个比较实体'},
                    {'step': 2, 'reasoning': '提取比较维度：确定比较的属性、指标或特点'},
                    {'step': 3, 'reasoning': '分析差异点：对比分析关键差异和特点'}
                ],
                'needed_information': [
                    '实体A的基本信息',
                    '实体B的基本信息',
                    '比较维度分析',
                    '差异对比结果'
                ],
                'sub_queries': [
                    '<实体A> 基本信息 特点',
                    '<实体B> 基本信息 特点'
                ]
            },
            {
                'type': 'range_comparison',
                'keywords': ['哪种', '哪个', '最远', '最长', '最高', '最低', '最X',"哪个...最"],
                'pattern': '在<范围>里，哪种<对象><属性><极值>',
                'chain_of_thought': [
                    {'step': 1, 'reasoning': '识别限定范围：提取Query中的筛选条件和范围'},
                    {'step': 2, 'reasoning': '识别比较对象：确定在范围内的所有候选对象'},
                    {'step': 3, 'reasoning': '识别比较维度：确定比较的属性（如关键字,单位属性等）'},
                    {'step': 4, 'reasoning': '极值分析：在范围内寻找指定属性的极值'}
                ],
                'needed_information': [
                    '范围限定条件',
                    '候选对象列表',
                    '比较属性分析',
                    '极值排序结果'
                ],
                'sub_queries': [
                    '<范围> <对象> <属性> 排序 极值分析'
                ]
            }
        ]
    },
    'rule': {
        'intent': 'condition_trigger_effect',
        'chain_of_thought': [
            {'step': 1, 'reasoning': '识别触发条件：提取Query中的"当...时"部分，确定触发行为/阶段/状态'},
            {'step': 2, 'reasoning': '识别作用对象：确定谁执行动作，谁受到影响（单位/武器/玩家）'},
            {'step': 3, 'reasoning': '匹配规则来源：核心规则、武器特性、单位技能或分队规则'},
            {'step': 4, 'reasoning': '推导结果：描述条件满足后会发生的效果或状态变化'}
        ],
        'needed_information': [
            '触发条件涉及的规则或技能说明',
            '核心规则中相关流程说明',
            '触发后可能的状态变化或骰子投掷'
        ],
        'sub_queries': [
            '触发条件 相关规则',
            '触发后 结果或状态变化',
            '触发条件 额外效果或修正'
        ]
    }
}

# 检索管道配置
PIPELINE_CONFIGS = {
    'compare': {
        'search_strategy': 'adaptive_multi_engine',
        'patterns': {
            'direct_comparison': {
                'engines': [
                    {'type': 'dense', 'purpose': '实体A特征检索'},
                    {'type': 'dense', 'purpose': '实体B特征检索'}
                ]
            },
            'range_extremum_comparison': {
                'engines': [
                    {'type': 'hybrid', 'purpose': '范围极值检索'}
                ]
            }
        },
        'post_processors': ['dedup', 'mmr'],
        'fusion_strategy': 'llm_aggregation'
    },
    'rule': {
        'search_strategy': 'multi_engine',
        'engines': [
            {'type': 'dense', 'purpose': '规则匹配检索'},
            {'type': 'dense', 'purpose': '实体识别检索'},
            {'type': 'dense', 'purpose': '逻辑推理检索'}
        ],
        'post_processors': ['dedup','mmr'],
        'fusion_strategy': 'logical_combination'
    },
    'list': {
        'search_strategy': 'single_engine',
        'engines': [
            {'type': 'hybrid', 'purpose': '全面列举检索'}
        ],
        'post_processors': ['dedup', 'mmr'],
        'fusion_strategy': 'aggregation'
    },
    'query': {
        'search_strategy': 'single_engine',
        'engines': [
            {'type': 'dense', 'purpose': '概念理解检索'}
        ],
        'post_processors': ['dedup','mmr'],
        'fusion_strategy': 'content_aggregation'
    },
    'unknown': {
        'search_strategy': 'fallback',
        'engines': [
            {'type': 'hybrid', 'purpose': '通用检索'}
        ],
        'post_processors': ['rerank'],
        'fusion_strategy': 'simple_aggregation'
    }
}

# 性能配置
PERFORMANCE_CONFIG = {
    'max_processing_time': 30,
    'enable_parallel': True,
    'max_workers': 4,
    'cache_enabled': True,
    'cache_ttl': 3600
}

# 非战锤问题处理配置
NON_WH40K_RESPONSES = {
    'IG': '士兵,你的问题我无法回答,请去法务部报道!',
    'SpaceMarine':'这位兄弟,你的问题过于离经叛道,请去战团牧师那里忏悔',
    'CSM':'我在恐惧之眼里趴了一万年都没有听过这种问题,你最好去问战帅',
    'NEC':'思维矩阵自检中...我们的墓穴没有存储该条知识...',
    'AMECH':'01010011011001010111001001101001011011110111010101110011001000000100100001100101011100100110010101110100011010010110001101110011, 机魂不悦',
    'ED':'对不起,我在方舟的永恒回路中没有找到该问题的答案,我推荐你去问乌斯维的先知',
    'ORK':'啥? 你个小虾米都在琢磨啥?俺寻思了半天也不知道咋回答你!你换个问题吧',
    'DRUK':'我们科摩罗人可不会浪费时间在这种问题上,你想去痛苦农场吗?',
    'TS':'我向至高天发誓,我从未听说过这种问题,你最好去问马格努斯本人',
    'NIS':'咀嚼声,咀嚼声,吞咽声.....',
    'QUIN':'你从网道的哪一个角落听说了这种问题?你最好去问帷幕行者本人',
    'DA':'这,这是第一军团的秘密,你等着我去叫至高大导师!',
}

# 错误处理配置
ERROR_HANDLING_CONFIG = {
    'continue_on_error': True,
    'fallback_strategy': 'simple',
    'log_errors': True,
    'retry_count': 2
} 