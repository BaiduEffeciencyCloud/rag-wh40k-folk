"""
专业术语白名单生成器
将战锤40K专业术语词典转换为BM25白名单格式
"""

import os
import json
from typing import List, Dict, Any
from vocab_mgr.vocab_mgr import VocabMgr


class WH40KWhitelistGenerator:
    """专业术语白名单生成器"""
    
    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary/"):
        self.vocab_path = vocab_path
        self.vocab_mgr = VocabMgr(vocab_path)
    
    def extract_wh40k_phrases(self) -> List[str]:
        """从专业术语词典中提取所有短语"""
        vocabulary = self._load_vocabulary()
        phrases = []
        
        for category, terms in vocabulary.items():
            category_phrases = self._extract_terms_from_category(terms)
            phrases.extend(category_phrases)
        
        return list(set(phrases))  # 去重
    
    def generate_wh40k_whitelist(self, output_path: str) -> str:
        """生成专业术语白名单"""
        phrases = self.extract_wh40k_phrases()
        
        # 生成BM25白名单格式：term frequency pos
        with open(output_path, 'w', encoding='utf-8') as f:
            for phrase in phrases:
                # 使用默认频率1，词性n（名词）
                # trim掉前后空格，保留中间空格
                clean_phrase = phrase.strip()
                if ' ' in clean_phrase:
                    f.write(f'"{clean_phrase}" 1 n\n')
                else:
                    f.write(f"{clean_phrase} 1 n\n")
        
        return output_path
    
    def _load_vocabulary(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载专业术语词典"""
        return self.vocab_mgr.get_vocab()
    
    def _extract_terms_from_category(self, category_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """从类别数据中提取术语"""
        terms = []
        
        # 词典格式是 {"term": {"type": "...", "frequency": ..., "synonyms": [...]}}
        for term, term_info in category_data.items():
            if isinstance(term, str) and term:
                # trim掉前后空格
                clean_term = term.strip()
                if clean_term:  # 确保trim后不为空
                    terms.append(clean_term)
                    
                    # 也添加同义词
                    if isinstance(term_info, dict) and 'synonyms' in term_info:
                        synonyms = term_info['synonyms']
                        if isinstance(synonyms, list):
                            for synonym in synonyms:
                                if isinstance(synonym, str) and synonym.strip():
                                    terms.append(synonym.strip())
        
        return terms 