"""
专业术语词典管理器

负责加载和管理战锤40K专业术语词典
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VocabMgr:
    """专业术语词典管理器"""

    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary"):
        """
        初始化专业术语词典管理器

        Args:
            vocab_path: 词典文件路径
        """
        self.vocab_path = vocab_path
        self.vocabulary = self._load_vocabulary()
        self.userdict = self._load_userdict()
        logger.info(f"VocabMgr初始化完成，词典路径: {vocab_path}")

    def _load_vocabulary(self) -> Dict[str, Any]:
        """加载专业术语词典"""
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
                            logger.debug(f"加载词典文件: {filename}")
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"加载词典文件失败 {filename}: {e}")
                        continue
            
            logger.info(f"成功加载 {len(vocabulary)} 个词典分类")
            
        except Exception as e:
            logger.error(f"加载词典失败: {e}")
            vocabulary = {}
        
        return vocabulary

    def _load_userdict(self) -> Dict[str, Any]:
        """加载jieba用户词典"""
        userdict = {}
        
        try:
            userdict_path = os.path.join("dict", "userdict", "wh40k_userdict.txt")
            if not os.path.exists(userdict_path):
                logger.warning(f"用户词典文件不存在: {userdict_path}")
                return userdict
            
            with open(userdict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            term = parts[0]
                            pos = parts[1] if len(parts) > 1 else 'n'
                            userdict[term] = pos
            
            logger.info(f"成功加载 {len(userdict)} 个用户词典条目")
            
        except Exception as e:
            logger.error(f"加载用户词典失败: {e}")
            userdict = {}
        
        return userdict

    def get_vocab(self) -> Dict[str, Any]:
        """获取专业术语词典"""
        return self.vocabulary

    def get_userdict(self) -> Dict[str, Any]:
        """获取jieba用户词典"""
        return self.userdict

    def save_vocabulary(self, vocabulary: Dict[str, Any]) -> bool:
        """保存专业术语词典"""
        try:
            if not os.path.exists(self.vocab_path):
                os.makedirs(self.vocab_path, exist_ok=True)
            
            # 保存每个分类到单独的JSON文件
            for category, terms in vocabulary.items():
                file_path = os.path.join(self.vocab_path, f"{category}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"保存词典分类: {category}")
            
            # 更新内存中的词典
            self.vocabulary = vocabulary
            
            logger.info(f"成功保存 {len(vocabulary)} 个词典分类")
            return True
            
        except Exception as e:
            logger.error(f"保存词典失败: {e}")
            return False

    def get_vocab_for_component(self, component_name: str) -> Dict[str, Any]:
        """根据组件需求返回特定词典"""
        if component_name == "jieba_tokenizer":
            return self.get_userdict()
        elif component_name == "entity_recognition":
            return self.get_vocab()
        else:
            return self.get_vocab() 