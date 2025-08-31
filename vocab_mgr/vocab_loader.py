"""
专业术语词典加载器

负责加载和转换不同格式的专业术语词典
"""

import os
import json
import logging
from typing import Dict, Any, List
from config import VOCAB_EXPORT_DIR

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

    def export_for_opensearch(self, output_dir: str = None) -> Dict[str, Any]:
        """导出OpenSearch分析器格式文件"""
        from config import VOCAB_EXPORT_DIR
        try:
            logger.info("开始导出OpenSearch分析器格式文件")
            logger.info(f"输出目录: {output_dir}")
            
            # 参数验证
            if output_dir is None:
                output_dir = VOCAB_EXPORT_DIR
            if not output_dir:
                output_dir = VOCAB_EXPORT_DIR
            
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 1. 收集术语和同义词
            all_terms = self._collect_all_terms()
            synonyms = self._collect_synonyms()
            
            logger.info(f"收集到术语: {len(all_terms)} 个")
            logger.info(f"收集到同义词组: {len(synonyms)} 组")
            
            # 2. Solr格式检验
            validation_result = self._validate_solr_format(synonyms)
            
            if not validation_result["passed"]:
                logger.warning(f"Solr格式检验失败，错误数: {len(validation_result['errors'])}")
                for error in validation_result["errors"]:
                    logger.error(f"验证错误: {error}")
            else:
                logger.info("Solr格式检验通过")
            
            # 3. 生成文件路径
            dict_file = os.path.join(output_dir, "warhammer_dict.txt")
            synonyms_file = os.path.join(output_dir, "warhammer_synonyms.txt")
            
            # 4. 生成文件
            dict_success = self._generate_dict_file(all_terms, dict_file)
            synonyms_success = self._generate_synonyms_file(synonyms, synonyms_file)
            
            # 5. 返回结果
            success = dict_success and synonyms_success
            
            result = {
                "success": success,
                "dict_file": dict_file if dict_success else "",
                "synonyms_file": synonyms_file if synonyms_success else "",
                "stats": {
                    "total_terms": len(all_terms),
                    "total_synonyms": len(synonyms),
                    "validation_passed": validation_result["passed"],
                    "validation_errors": validation_result["errors"]
                },
                "message": "导出成功" if success else "导出失败"
            }
            
            if success:
                logger.info("OpenSearch格式导出成功")
                logger.info(f"术语文件: {dict_file}")
                logger.info(f"同义词文件: {synonyms_file}")
                
                # 显示详细的统计信息
                logger.info("=== 导出统计 ===")
                logger.info(f"术语词典: {len(all_terms)} 个唯一术语")
                logger.info(f"同义词文件: {len(synonyms)} 组同义词")
                
                if not validation_result["passed"]:
                    logger.warning(f"Solr格式验证失败，发现 {len(validation_result['errors'])} 个错误:")
                    logger.warning("")
                    
                    # 按错误类型分组显示
                    cross_group_errors = []
                    other_errors = []
                    
                    for error in validation_result["errors"]:
                        if "重复出现:" in error:
                            cross_group_errors.append(error)
                        else:
                            other_errors.append(error)
                    
                    # 显示跨组重复错误
                    if cross_group_errors:
                        logger.warning("【跨组重复错误】")
                        for error in cross_group_errors:
                            logger.warning(error)
                        logger.warning("")
                    
                    # 显示其他错误
                    if other_errors:
                        logger.warning("【其他格式错误】")
                        for error in other_errors:
                            logger.warning(f"  - {error}")
                else:
                    logger.info("Solr格式验证通过")
            else:
                logger.error("OpenSearch格式导出失败")
            
            return result
            
        except Exception as e:
            logger.error(f"导出OpenSearch格式失败: {e}")
            return {
                "success": False,
                "dict_file": "",
                "synonyms_file": "",
                "stats": {
                    "total_terms": 0,
                    "total_synonyms": 0,
                    "validation_passed": False,
                    "validation_errors": [f"导出过程发生异常: {str(e)}"]
                },
                "message": f"导出失败: {str(e)}"
            }
        
    def _generate_dict_file(self, terms: List[str], output_path: str) -> bool:
        """生成术语词典文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for term in terms:
                    if term and term.strip():
                        f.write(f"{term.strip()}\n")
            
            logger.info(f"成功生成术语词典文件: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成术语词典文件失败: {e}")
            return False
        
    def _generate_synonyms_file(self, synonyms: List[List[str]], output_path: str) -> bool:
        """生成同义词文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for group in synonyms:
                    if group and len(group) >= 2:
                        cleaned_group = [self._clean_term(term) for term in group]
                        cleaned_group = [term for term in cleaned_group if term]  # 过滤空术语
                        if len(cleaned_group) >= 2:
                            f.write(",".join(cleaned_group) + "\n")
            
            logger.info(f"成功生成同义词文件: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成同义词文件失败: {e}")
            return False
        
    def _validate_solr_format(self, synonyms: List[List[str]]) -> Dict[str, Any]:
        """Solr格式检验"""
        errors = []
        warnings = []
        
        try:
            # 检查空列表
            if not synonyms:
                return {
                    "passed": True,
                    "errors": [],
                    "warnings": []
                }
            
            # 1. 基础格式检验
            for i, group in enumerate(synonyms):
                if not isinstance(group, list):
                    errors.append(f"第{i+1}行不是有效的同义词组")
                    continue
                
                # 检查同义词组大小
                if len(group) < 2:
                    errors.append(f"第{i+1}行同义词组少于2个术语")
                    continue
                
                # 检查行长度（建议性限制，不是硬性要求）
                line = ",".join(group)
                if len(line) > 2048:  # 提高到更合理的限制
                    warnings.append(f"第{i+1}行较长({len(line)}字符)，建议拆分为多个同义词组")
                
                # 检查连续逗号
                if ",," in line:
                    errors.append(f"第{i+1}行包含连续逗号")
                
                # 检查组内重复
                unique_terms = set()
                for term in group:
                    if term in unique_terms:
                        errors.append(f"第{i+1}行内术语'{term}'重复")
                    unique_terms.add(term)
            
            # 2. 跨组重复检验
            all_terms = {}
            for i, group in enumerate(synonyms):
                for term in group:
                    if term in all_terms:
                        # 获取重复术语的详细信息
                        first_occurrence = all_terms[term]
                        errors.append(f"术语'{term}'重复出现:")
                        errors.append(f"  - 第{first_occurrence['line']+1}行 ({first_occurrence['group']})")
                        errors.append(f"  - 第{i+1}行 ({group})")
                    else:
                        all_terms[term] = {"line": i, "group": group}
            
            # 3. 字符检验
            forbidden_chars = ['\t', '\n', '\r']
            for i, group in enumerate(synonyms):
                for term in group:
                    for char in forbidden_chars:
                        if char in term:
                            errors.append(f"第{i+1}行术语'{term}'包含禁用字符: {repr(char)}")
                    
                    # 检查空术语
                    if not term or term.strip() == "":
                        errors.append(f"第{i+1}行包含空术语")
            
            logger.debug(f"Solr格式检验完成: {len(errors)} 个错误, {len(warnings)} 个警告")
            
            return {
                "passed": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Solr格式检验失败: {e}")
            return {
                "passed": False,
                "errors": [f"检验过程发生异常: {str(e)}"],
                "warnings": []
            }
        
    def _collect_all_terms(self) -> List[str]:
        """收集所有术语"""
        all_terms = set()
        
        try:
            vocabulary = self._load_all_vocabulary()
            
            for category, terms in vocabulary.items():
                if isinstance(terms, dict):
                    for term in terms.keys():
                        cleaned_term = self._clean_term(term)
                        if cleaned_term and len(cleaned_term) >= 2:
                            all_terms.add(cleaned_term)
            
            # 转换为列表并排序
            sorted_terms = sorted(list(all_terms))
            logger.debug(f"收集到术语: {len(sorted_terms)} 个")
            
            return sorted_terms
            
        except Exception as e:
            logger.error(f"收集术语失败: {e}")
            return []
        
    def _collect_synonyms(self) -> List[List[str]]:
        """收集同义词组"""
        synonym_groups = []
        
        try:
            vocabulary = self._load_all_vocabulary()
            
            # 从词典中收集同义词
            for category, terms in vocabulary.items():
                if isinstance(terms, dict):
                    for term, info in terms.items():
                        if isinstance(info, dict) and 'synonyms' in info:
                            synonyms = info['synonyms']
                            if isinstance(synonyms, list) and len(synonyms) > 0:
                                # 构建同义词组：[主术语, 同义词1, 同义词2, ...]
                                group = [self._clean_term(term)]
                                for synonym in synonyms:
                                    cleaned_synonym = self._clean_term(synonym)
                                    if cleaned_synonym and cleaned_synonym not in group:
                                        group.append(cleaned_synonym)
                                
                                # 只保留包含2个或以上术语的组
                                if len(group) >= 2:
                                    synonym_groups.append(group)
            
            logger.debug(f"收集到同义词组: {len(synonym_groups)} 组")
            return synonym_groups
            
        except Exception as e:
            logger.error(f"收集同义词失败: {e}")
            return []
        
    def _clean_term(self, term: str) -> str:
        """清理术语格式"""
        if not term:
            return ""
        
        # 去除首尾空白字符
        cleaned = term.strip()
        
        # 去除制表符、换行符等控制字符
        cleaned = cleaned.replace('\t', '').replace('\n', '').replace('\r', '')
        
        return cleaned 