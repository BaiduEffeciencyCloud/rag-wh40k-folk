"""
自适应检索数据模型
定义意图识别和COT处理相关的数据结构
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class IntentResult:
    """意图识别结果数据类"""
    intent_type: str                # 意图类型 ("compare", "rule", "list", "query", "unknown")
    confidence: float               # 置信度分数 (0-1)
    entities: List[str]             # 识别出的实体列表
    relations: List[str]            # 识别出的关系列表
    complexity_score: float         # 查询复杂度分数 (0-1)
    uncertainty_result: Dict[str, Any]  # 不确定性检测结果
    probabilities: np.ndarray       # 各意图的概率分布


@dataclass
class COTStep:
    """思维链步骤数据类"""
    step_id: int                    # 步骤序号 (1, 2, 3...)
    step_type: str                  # 步骤类型 ("background", "analysis", "comparison", "conclusion")
    query: str                      # 该步骤对应的查询文本
    reasoning: str                  # 推理说明
    entities: List[str]             # 该步骤识别出的实体
    relations: List[str]            # 该步骤识别出的关系
    expected_info: List[str]        # 期望获取的信息类型
    dependencies: List[int]         # 依赖的步骤ID列表
    confidence: float               # 该步骤的置信度 (0-1)
    metadata: Dict[str, Any]       # 额外元数据
