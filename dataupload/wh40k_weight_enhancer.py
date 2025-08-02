"""
专业术语权重增强器
为战锤40K专业术语提供权重增强功能
"""

import os
import logging
from typing import Dict, List, Any, Optional
from vocab_mgr.vocab_mgr import VocabMgr

logger = logging.getLogger(__name__)


class WH40KWeightEnhancer:
    """专业术语权重增强器"""
    
    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary/"):
        """
        初始化权重增强器
        
        Args:
            vocab_path: 专业术语词典路径
        """
        self.vocab_path = vocab_path
        self.vocab_mgr = VocabMgr(vocab_path)
        self.category_weights = {
            "units": 2.0,      # 单位最重要
            "factions": 1.8,   # 派系次之
            "skills": 1.5,     # 技能
            "weapons": 1.5,    # 武器
            "mechanics": 1.3   # 机制
        }
    
    def get_term_category(self, term: str) -> str:
        """
        获取术语所属类别
        
        Args:
            term: 术语
            
        Returns:
            术语所属类别，如果未找到则返回"general"
        """
        if not term:
            return "general"
        
        vocabulary = self.vocab_mgr.get_vocab()
        
        for category, terms in vocabulary.items():
            if isinstance(terms, dict):
                # 如果terms是字典格式（包含term_info）
                if term in terms:
                    return category
            elif isinstance(terms, list):
                # 如果terms是列表格式
                if term in terms:
                    return category
        
        return "general"
    
    def calculate_enhanced_weight(self, term: str, base_weight: float) -> float:
        """
        计算增强权重
        
        Args:
            term: 术语
            base_weight: 基础权重
            
        Returns:
            增强后的权重
        """
        if not term or base_weight == 0:
            return base_weight
        
        category = self.get_term_category(term)
        weight_factor = self.category_weights.get(category, 1.0)
        
        return base_weight * weight_factor
    
    def enhance_whitelist_weights(self, whitelist_path: str, output_path: str) -> str:
        """
        增强白名单权重
        
        Args:
            whitelist_path: 输入白名单路径
            output_path: 输出白名单路径
            
        Returns:
            增强后的白名单路径
        """
        if not os.path.exists(whitelist_path):
            raise FileNotFoundError(f"白名单文件不存在: {whitelist_path}")
        
        enhanced_phrases = []
        
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    phrase = parts[0]
                    try:
                        base_weight = int(parts[1])
                    except ValueError:
                        base_weight = 1
                    
                    # 计算增强权重
                    enhanced_weight = self.calculate_enhanced_weight(phrase, base_weight)
                    
                    # 重建行，保持原有格式
                    if len(parts) >= 3:
                        enhanced_phrases.append(f"{phrase} {int(enhanced_weight)} {parts[2]}")
                    else:
                        enhanced_phrases.append(f"{phrase} {int(enhanced_weight)} n")
        
        # 保存增强后的白名单
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(enhanced_phrases))
        
        return output_path
    
    def get_category_weights(self) -> Dict[str, float]:
        """
        获取类别权重配置
        
        Returns:
            类别权重字典
        """
        return self.category_weights.copy()
    
    def set_category_weight(self, category: str, weight: float) -> None:
        """
        设置类别权重
        
        Args:
            category: 类别
            weight: 权重
        """
        if category and weight >= 0:
            self.category_weights[category] = weight 