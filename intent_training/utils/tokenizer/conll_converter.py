# -*- coding: utf-8 -*-
"""
CoNLL format converter for WH40K annotation
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConllConverter:
    """将分词结果转换为CoNLL格式"""
    
    def convert_to_conll(self, text: str, tokens: List[str], 
                        intent: str, entity_labels: Optional[List[str]] = None) -> str:
        """
        转换为CoNLL格式字符串
        
        Args:
            text: 原始文本
            tokens: 分词结果
            intent: 意图标签
            entity_labels: 实体标签列表
            
        Returns:
            CoNLL格式字符串
        """
        if not tokens:
            return ""
            
        # 如果没有提供实体标签，默认全部为O
        if entity_labels is None:
            entity_labels = ["O"] * len(tokens)
            
        # 确保标签数量与token数量一致
        if len(entity_labels) != len(tokens):
            entity_labels = ["O"] * len(tokens)
            
        conll_lines = []
        for i, (token, label) in enumerate(zip(tokens, entity_labels)):
            conll_lines.append(f"{token}\t{label}")
            
        return "\n".join(conll_lines)
    
    def convert_to_simple_conll(self, text: str, tokens: List[str]) -> str:
        """
        转换为简单CoNLL格式字符串（不带标签）
        
        Args:
            text: 原始文本
            tokens: 分词结果
            
        Returns:
            简单CoNLL格式字符串（只包含token）
        """
        if not tokens:
            return ""
            
        # 只返回token，不包含标签
        return "\n".join(tokens)
    
    def convert_multiple_queries_to_simple_conll(self, queries_data: List[tuple]) -> str:
        """
        将多个query转换为简单CoNLL格式，query之间用换行符分隔
        
        Args:
            queries_data: 包含(text, tokens)元组的列表
            
        Returns:
            多个query的简单CoNLL格式字符串
        """
        if not queries_data:
            return ""
            
        conll_sections = []
        for text, tokens in queries_data:
            if tokens:
                # 为每个query生成简单的token列表
                query_tokens = "\n".join(tokens)
                conll_sections.append(query_tokens)
        
        # 用双换行符分隔不同的query
        return "\n\n".join(conll_sections)
        
    def save_conll_file(self, conll_data: List[str], output_path: str):
        """
        保存CoNLL文件
        
        Args:
            conll_data: CoNLL格式数据列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in conll_data:
                    f.write(line + "\n")
            logger.info(f"CoNLL文件已保存: {output_path}")
        except Exception as e:
            logger.error(f"保存CoNLL文件失败: {e}")
            raise 