# -*- coding: utf-8 -*-
"""
Basic jieba tokenizer for WH40K text processing
"""

import jieba
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class JiebaTokenizer:
    """基于jieba的基础分词器"""
    
    def __init__(self, custom_dict_path: Optional[str] = None):
        """
        初始化jieba分词器
        
        Args:
            custom_dict_path: 自定义词典路径
        """
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
            logger.info(f"加载自定义词典: {custom_dict_path}")
        
    def tokenize(self, text: str) -> List[str]:
        """
        基础分词功能
        
        Args:
            text: 待分词文本
            
        Returns:
            分词结果列表
        """
        if not text or not text.strip():
            return []
        
        tokens = list(jieba.cut(text.strip()))
        return [token for token in tokens if token.strip()]
        
    def add_custom_words(self, words: List[str]):
        """
        添加自定义词汇
        
        Args:
            words: 自定义词汇列表
        """
        for word in words:
            jieba.add_word(word)
        logger.info(f"添加自定义词汇: {words}") 