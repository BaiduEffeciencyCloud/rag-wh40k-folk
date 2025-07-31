"""
专业术语词典更新器

负责从文档切片和用户查询中提取并更新专业术语
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)


class VocabUp:
    """专业术语词典更新器"""

    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary"):
        """
        初始化专业术语词典更新器

        Args:
            vocab_path: 词典文件路径
        """
        self.vocab_path = vocab_path
        self.vocabulary = self._load_vocabulary()
        logger.info(f"VocabUp初始化完成，词典路径: {vocab_path}")

    def _load_vocabulary(self) -> Dict[str, Any]:
        """加载现有专业术语词典"""
        vocabulary = {}
        
        try:
            if not os.path.exists(self.vocab_path):
                logger.warning(f"词典路径不存在: {self.vocab_path}")
                return vocabulary
            
            # 遍历词典目录下的所有JSON文件
            for filename in os.listdir(self.vocab_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.vocab_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            category = filename.replace('.json', '')
                            vocabulary[category] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"加载词典文件失败 {filename}: {e}")
                        continue
            
            logger.info(f"成功加载 {len(vocabulary)} 个词典分类")
            
        except Exception as e:
            logger.error(f"加载词典失败: {e}")
            vocabulary = {}
        
        return vocabulary

    def update_from_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        从文档切片中更新专业术语词典

        Args:
            chunks: 文档切片列表

        Returns:
            bool: 更新是否成功
        """
        try:
            if not chunks:
                logger.info("文档切片为空，跳过更新")
                return True
            
            # 从文档切片中提取专业术语
            new_terms = self._extract_terms_from_chunks(chunks)
            
            # 合并新术语到现有词典
            if new_terms:
                success = self._merge_terms(new_terms)
                if success:
                    # 保存更新后的词典
                    return self._save_vocabulary()
            
            return True
            
        except Exception as e:
            logger.error(f"从文档切片更新词典失败: {e}")
            return False

    def update_from_queries(self, queries: List[str]) -> bool:
        """
        从用户查询中更新专业术语词典

        Args:
            queries: 用户查询列表

        Returns:
            bool: 更新是否成功
        """
        try:
            if not queries:
                logger.info("用户查询为空，跳过更新")
                return True
            
            # 从用户查询中提取专业术语
            new_terms = self._extract_terms_from_queries(queries)
            
            # 合并新术语到现有词典
            if new_terms:
                success = self._merge_terms(new_terms)
                if success:
                    # 保存更新后的词典
                    return self._save_vocabulary()
            
            return True
            
        except Exception as e:
            logger.error(f"从用户查询更新词典失败: {e}")
            return False

    def _extract_terms_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从文档切片中提取专业术语"""
        extracted_terms = {}
        
        for chunk in chunks:
            if not isinstance(chunk, dict) or 'text' not in chunk:
                continue
            
            text = chunk['text']
            if not text:
                continue
            
            # 识别文本中的战锤40K专业术语
            terms = self._identify_wh40k_terms(text)
            
            # 根据切片内容类型分类术语
            content_type = chunk.get('content_type', 'general')
            section_heading = chunk.get('section_heading', '')
            
            # 确定术语分类
            category = self._determine_term_category(content_type, section_heading)
            
            if category not in extracted_terms:
                extracted_terms[category] = {}
            
            # 添加术语到对应分类
            for term in terms:
                if term not in extracted_terms[category]:
                    extracted_terms[category][term] = {
                        'type': category.upper(),
                        'frequency': 1,
                        'source': 'document_chunk'
                    }
                else:
                    extracted_terms[category][term]['frequency'] += 1
        
        logger.info(f"从文档切片中提取了 {sum(len(terms) for terms in extracted_terms.values())} 个术语")
        return extracted_terms

    def _extract_terms_from_queries(self, queries: List[str]) -> Dict[str, Any]:
        """从用户查询中提取专业术语"""
        extracted_terms = {}
        
        for query in queries:
            if not query:
                continue
            
            # 识别查询中的战锤40K专业术语
            terms = self._identify_wh40k_terms(query)
            
            # 将术语添加到general分类
            if 'general' not in extracted_terms:
                extracted_terms['general'] = {}
            
            for term in terms:
                if term not in extracted_terms['general']:
                    extracted_terms['general'][term] = {
                        'type': 'GENERAL',
                        'frequency': 1,
                        'source': 'user_query'
                    }
                else:
                    extracted_terms['general'][term]['frequency'] += 1
        
        logger.info(f"从用户查询中提取了 {sum(len(terms) for terms in extracted_terms.values())} 个术语")
        return extracted_terms

    def _merge_terms(self, new_terms: Dict[str, Any]) -> bool:
        """合并新术语到现有词典"""
        try:
            for category, terms in new_terms.items():
                if category not in self.vocabulary:
                    self.vocabulary[category] = {}
                
                for term, info in terms.items():
                    if term in self.vocabulary[category]:
                        # 更新现有术语的频率
                        self.vocabulary[category][term]['frequency'] += info['frequency']
                        logger.debug(f"更新术语频率: {term}")
                    else:
                        # 添加新术语
                        self.vocabulary[category][term] = info
                        logger.debug(f"添加新术语: {term}")
            
            logger.info(f"成功合并 {sum(len(terms) for terms in new_terms.values())} 个术语")
            return True
            
        except Exception as e:
            logger.error(f"合并术语失败: {e}")
            return False

    def _save_vocabulary(self) -> bool:
        """保存更新后的词典"""
        try:
            if not os.path.exists(self.vocab_path):
                os.makedirs(self.vocab_path, exist_ok=True)
            
            # 保存每个分类到单独的JSON文件
            for category, terms in self.vocabulary.items():
                file_path = os.path.join(self.vocab_path, f"{category}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"保存词典分类: {category}")
            
            logger.info(f"成功保存 {len(self.vocabulary)} 个词典分类")
            return True
            
        except Exception as e:
            logger.error(f"保存词典失败: {e}")
            return False

    def _identify_wh40k_terms(self, text: str) -> Set[str]:
        """识别文本中的战锤40K专业术语"""
        terms = set()
        
        if not text:
            return terms
        
        # 使用正则表达式识别可能的专业术语
        # 1. 大写字母开头的词组（可能是单位名、技能名等）
        capital_pattern = r'[A-Z][a-zA-Z\u4e00-\u9fff]+'
        capital_matches = re.findall(capital_pattern, text)
        terms.update(capital_matches)
        
        # 2. 包含数字的术语（可能是规则编号等）
        number_pattern = r'[A-Za-z\u4e00-\u9fff]+\d+'
        number_matches = re.findall(number_pattern, text)
        terms.update(number_matches)
        
        # 3. 特定格式的术语（如"XX之XX"、"XXXX"等）
        special_patterns = [
            r'[A-Za-z\u4e00-\u9fff]+之[A-Za-z\u4e00-\u9fff]+',  # XX之XX
            r'[A-Za-z\u4e00-\u9fff]{4,}',  # 4个字符以上的词组
        ]
        
        for pattern in special_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        # 过滤掉太短的术语（少于2个字符）
        terms = {term for term in terms if len(term) >= 2}
        
        return terms

    def _determine_term_category(self, content_type: str, section_heading: str) -> str:
        """根据内容类型和章节标题确定术语分类"""
        content_type = content_type.lower()
        section_heading = section_heading.lower()
        
        # 根据内容类型分类
        if 'unit' in content_type or '单位' in section_heading:
            return 'units'
        elif 'skill' in content_type or '技能' in section_heading:
            return 'skills'
        elif 'weapon' in content_type or '武器' in section_heading:
            return 'weapons'
        elif 'faction' in content_type or '派系' in section_heading:
            return 'factions'
        elif 'mechanic' in content_type or '机制' in section_heading:
            return 'mechanics'
        else:
            return 'general' 