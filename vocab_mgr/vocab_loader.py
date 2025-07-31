"""
专业术语词典加载器

负责加载和转换不同格式的专业术语词典
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class VocabLoad:
    """专业术语词典加载器"""

    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary"):
        """
        初始化专业术语词典加载器

        Args:
            vocab_path: 词典文件路径
        """
        self.vocab_path = vocab_path
        logger.info(f"VocabLoad初始化完成，词典路径: {vocab_path}")

    def load_for_jieba(self) -> List[str]:
        """
        加载jieba格式的专业术语词典

        Returns:
            List[str]: jieba格式的术语列表
        """
        jieba_terms = []
        
        try:
            # 加载所有词典分类
            vocabulary = self._load_all_vocabulary()
            
            # 转换为jieba格式
            jieba_terms = self.convert_to_jieba_format(vocabulary)
            
            logger.info(f"成功加载 {len(jieba_terms)} 个jieba格式术语")
            
        except Exception as e:
            logger.error(f"加载jieba格式词典失败: {e}")
            jieba_terms = []
        
        return jieba_terms

    def load_for_entity_recognition(self) -> Dict[str, Any]:
        """
        加载实体识别用的专业术语词典

        Returns:
            Dict[str, Any]: 实体识别词典
        """
        entity_vocab = {}
        
        try:
            # 加载所有词典分类
            vocabulary = self._load_all_vocabulary()
            
            # 转换为实体识别格式
            for category, terms in vocabulary.items():
                entity_vocab[category] = {}
                for term, info in terms.items():
                    entity_vocab[category][term] = {
                        'type': info.get('type', category.upper()),
                        'frequency': info.get('frequency', 1),
                        'description': info.get('description', ''),
                        'synonyms': info.get('synonyms', [])
                    }
            
            logger.info(f"成功加载 {len(entity_vocab)} 个实体识别词典分类")
            
        except Exception as e:
            logger.error(f"加载实体识别词典失败: {e}")
            entity_vocab = {}
        
        return entity_vocab

    def load_vocabulary_file(self, filename: str) -> Dict[str, Any]:
        """
        加载指定的词典文件

        Args:
            filename: 词典文件名

        Returns:
            Dict[str, Any]: 词典内容
        """
        vocabulary = {}
        
        try:
            file_path = os.path.join(self.vocab_path, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"词典文件不存在: {file_path}")
                return vocabulary
            
            with open(file_path, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            
            logger.debug(f"成功加载词典文件: {filename}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载词典文件失败 {filename}: {e}")
            vocabulary = {}
        except Exception as e:
            logger.error(f"加载词典文件失败 {filename}: {e}")
            vocabulary = {}
        
        return vocabulary

    def convert_to_jieba_format(self, vocabulary: Dict[str, Any]) -> List[str]:
        """
        将JSON格式词典转换为jieba格式

        Args:
            vocabulary: JSON格式词典

        Returns:
            List[str]: jieba格式术语列表
        """
        jieba_terms = []
        
        try:
            for category, terms in vocabulary.items():
                if not isinstance(terms, dict):
                    continue
                
                for term, info in terms.items():
                    if not isinstance(term, str) or not term.strip():
                        continue
                    
                    # 构建jieba格式：术语 + 词性
                    pos = 'n'  # 默认为名词
                    if isinstance(info, dict):
                        # 根据类型确定词性
                        term_type = info.get('type', '').upper()
                        if term_type in ['UNIT', 'WEAPON']:
                            pos = 'n'  # 名词
                        elif term_type == 'SKILL':
                            pos = 'n'  # 名词
                        elif term_type == 'FACTION':
                            pos = 'n'  # 名词
                        elif term_type == 'MECHANIC':
                            pos = 'n'  # 名词
                        else:
                            pos = 'n'  # 默认为名词
                    
                    jieba_terms.append(f"{term} {pos}")
            
            logger.debug(f"成功转换 {len(jieba_terms)} 个术语为jieba格式")
            
        except Exception as e:
            logger.error(f"转换jieba格式失败: {e}")
            jieba_terms = []
        
        return jieba_terms

    def save_jieba_userdict(self, terms: List[str], output_path: str) -> bool:
        """
        保存jieba用户词典文件

        Args:
            terms: 术语列表
            output_path: 输出文件路径

        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 写入jieba用户词典文件
            with open(output_path, 'w', encoding='utf-8') as f:
                for term in terms:
                    if isinstance(term, str) and term.strip():
                        f.write(f"{term}\n")
            
            logger.info(f"成功保存jieba用户词典到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存jieba用户词典失败: {e}")
            return False

    def _load_all_vocabulary(self) -> Dict[str, Any]:
        """加载所有词典分类"""
        vocabulary = {}
        
        try:
            if not os.path.exists(self.vocab_path):
                logger.warning(f"词典路径不存在: {self.vocab_path}")
                return vocabulary
            
            # 遍历词典目录下的所有JSON文件，排除combined.json
            for filename in os.listdir(self.vocab_path):
                if filename.endswith('.json') and filename != 'combined.json':
                    file_path = os.path.join(self.vocab_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            category = filename.replace('.json', '')
                            vocabulary[category] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"加载词典文件失败 {filename}: {e}")
                        continue
            
            logger.debug(f"成功加载 {len(vocabulary)} 个词典分类")
            
        except Exception as e:
            logger.error(f"加载词典失败: {e}")
            vocabulary = {}
        
        return vocabulary 