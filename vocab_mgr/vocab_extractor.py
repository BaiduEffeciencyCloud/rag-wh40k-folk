"""
专业术语抽取器

负责从chunks中抽取和分类专业术语
"""

import re
import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VocabExtractor:
    """专业术语抽取器"""
    
    def __init__(self, vocab_path: str = "dict/wh40k_vocabulary"):
        self.term_patterns = self._init_patterns()
        self.category_mapping = self._init_category_mapping()
        self.vocab_path = vocab_path
        self.vocabulary = self._load_vocabulary()
    
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
    
    def _init_patterns(self) -> Dict[str, str]:
        """初始化术语识别模式"""
        return {
            "unit_pattern": r"([A-Za-z\u4e00-\u9fff]+（[A-Za-z\u4e00-\u9fff]+）|[A-Za-z\u4e00-\u9fff]+(?:\s+[A-Z]+)?)",
            "weapon_pattern": r"([A-Za-z\u4e00-\u9fff]+[·\s]+[A-Za-z\u4e00-\u9fff]+|[A-Za-z\u4e00-\u9fff]+武器)",
            "skill_pattern": r"([A-Za-z\u4e00-\u9fff]+专注|[A-Za-z\u4e00-\u9fff]+动作|[A-Za-z\u4e00-\u9fff]+能力)",
            "mechanic_pattern": r"(战斗专注|殊途同归|致命破灭|深入打击|先攻|连击|移动阶段|射击阶段|冲锋阶段|近战阶段|士气测试|装甲保护)"
        }
    
    def _init_category_mapping(self) -> Dict[str, str]:
        """初始化分类映射"""
        return {
            "人物": "units",
            "战线": "units", 
            "载具": "units",
            "骑乘": "units",
            "巨兽": "units",
            "步兵": "units",
            "运输载具": "units",
            "专属载具": "units",
            "军队规则": "mechanics",
            "分队规则": "detachments",
            "army_rule": "mechanics",
            "detachment_rule": "detachments",
            "unit": "units"
        }
    
    def _get_category_by_pattern(self, text: str) -> Optional[str]:
        """通过模式匹配获取分类"""
        # 模式匹配规则
        patterns = {
            r".*单位数据$": "units",  # 匹配任何以"单位数据"结尾的文本
            r".*军队规则$": "mechanics",  # 匹配任何以"军队规则"结尾的文本
            r".*分队规则$": "detachments",  # 匹配任何以"分队规则"结尾的文本
        }
        
        for pattern, category in patterns.items():
            if re.match(pattern, text):
                return category
        
        return None
    
    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """从chunks中抽取专业术语（包含原始文本信息）"""
        all_terms = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        for chunk in chunks:
            chunk_terms = self.extract_from_single_chunk_with_origin(chunk)
            for category, terms in chunk_terms.items():
                all_terms[category].extend(terms)
        
        # 去重（基于术语名称）
        for category in all_terms:
            seen_terms = set()
            unique_terms = []
            for term_info in all_terms[category]:
                term_name = term_info["term"]
                if term_name not in seen_terms:
                    seen_terms.add(term_name)
                    unique_terms.append(term_info)
            all_terms[category] = unique_terms
        
        return all_terms
    
    def _deduplicate_across_categories(self, all_terms: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """跨分类去重，优先保留在更合适的分类中"""
        # 定义分类优先级（数字越小优先级越高）
        category_priority = {
            "units": 1,        # 单位优先级最高
            "skills": 2,       # 技能次之
            "weapons": 3,      # 武器再次
            "detachments": 4,  # 分队
            "factions": 5,     # 种族
            "mechanics": 6     # 机制优先级最低
        }
        
        # 收集所有术语及其出现的分类
        term_categories = {}
        for category, terms in all_terms.items():
            for term in terms:
                if term not in term_categories:
                    term_categories[term] = []
                term_categories[term].append(category)
        
        # 处理重复术语
        for term, categories in term_categories.items():
            if len(categories) > 1:
                # 选择优先级最高的分类
                best_category = min(categories, key=lambda c: category_priority.get(c, 999))
                
                # 从其他分类中移除该术语
                for category in categories:
                    if category != best_category:
                        if term in all_terms[category]:
                            all_terms[category].remove(term)
        
        return all_terms
    
    def extract_from_single_chunk(self, chunk: Dict[str, Any]) -> Dict[str, List[str]]:
        """从单个chunk中抽取专业术语（只从标题获取单位信息）"""
        if not chunk:
            return {
                "units": [],
                "detachments": [],
                "factions": [],
                "mechanics": []
            }
        
        # 初始化结果
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        # 获取原始文本
        original_text = chunk.get("text", "")
        remaining_text = original_text
        
        # Step 1: 从标题中抽取（优先级最高）
        header_terms = self._extract_from_headers(chunk)
        for category, terms in header_terms.items():
            if category in result:  # 只保留我们需要的分类
                result[category].extend(terms)
                # 从剩余文本中删除已抽取的术语
                for term in terms:
                    remaining_text = self._remove_term_from_text(remaining_text, term)
        
        # Step 2: 从剩余文本中抽取（只抽取detachment、faction、mechanism，不抽取单位）
        content_type = chunk.get("content_type", "")
        if content_type == "corerule":
            text_terms = self._extract_from_text_with_mechanics(remaining_text)
        else:
            text_terms = self._extract_from_text_no_mechanics(remaining_text)
        
        # 合并文本抽取结果（排除单位抽取）
        for category, terms in text_terms.items():
            if category in result and category != "units":  # 完全跳过单位抽取
                result[category].extend(terms)
        
        # 清理和去重
        for category in result:
            result[category] = self._clean_terms(result[category])
        
        return result

    def extract_from_single_chunk_with_origin(self, chunk: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """从单个chunk中抽取专业术语（包含原始文本信息）"""
        if not chunk:
            return {
                "units": [],
                "detachments": [],
                "factions": [],
                "mechanics": []
            }
        
        # 初始化结果
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        # 获取原始文本
        original_text = chunk.get("text", "")
        remaining_text = original_text
        
        # Step 1: 从标题中抽取（优先级最高）
        header_terms = self._extract_from_headers_with_origin(chunk)
        for category, terms in header_terms.items():
            if category in result:  # 只保留我们需要的分类
                result[category].extend(terms)
                # 从剩余文本中删除已抽取的术语
                for term_info in terms:
                    remaining_text = self._remove_term_from_text(remaining_text, term_info["term"])
        
        # Step 2: 从剩余文本中抽取（只抽取detachment、faction、mechanism，不抽取单位）
        content_type = chunk.get("content_type", "")
        if content_type == "corerule":
            text_terms = self._extract_from_text_with_mechanics_with_origin(remaining_text)
        else:
            text_terms = self._extract_from_text_no_mechanics_with_origin(remaining_text)
        
        # 合并文本抽取结果（排除单位抽取）
        for category, terms in text_terms.items():
            if category in result and category != "units":  # 完全跳过单位抽取
                result[category].extend(terms)
        
        # 清理和去重
        for category in result:
            result[category] = self._clean_terms_with_origin(result[category])
        
        return result
    
    def _remove_term_from_text(self, text: str, term: str) -> str:
        """从文本中删除指定术语"""
        if not text or not term:
            return text
        
        # 使用正则表达式删除术语（包括可能的空格和标点）
        pattern = r'\b' + re.escape(term) + r'\b'
        return re.sub(pattern, '', text)
    
    def _extract_from_headers(self, chunk: Dict[str, Any]) -> Dict[str, List[str]]:
        """从标题中抽取术语（基于新的层级识别逻辑）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        h1 = chunk.get("h1", "")
        h2 = chunk.get("h2", "")
        h3 = chunk.get("h3", "")
        
        if not h1:
            return result
        
        # Step 1: 基于H1的层级识别
        h1_lower = h1.lower()
        
        # 1. 如果H1包含"军队规则"关键字
        if any(keyword in h1_lower for keyword in ["军队规则", "army rule"]):
            # 从H2、H3识别MECHANICS
            if h2:
                mechanic_terms = self._extract_chinese_part(h2)
                if mechanic_terms:
                    result["mechanics"].append(mechanic_terms)
            if h3:
                mechanic_terms = self._extract_chinese_part(h3)
                if mechanic_terms:
                    result["mechanics"].append(mechanic_terms)
        
        # 2. 如果H1包含"分队规则"关键字
        elif any(keyword in h1_lower for keyword in ["分队规则", "detachment rule"]):
            # 记录H2的分队名称(DETACHMENT)
            if h2:
                detachment_terms = self._extract_chinese_part(h2)
                if detachment_terms:
                    result["detachments"].append(detachment_terms)
        
        # 3. 如果H1包含"阿苏焉尼单位数据"关键字
        elif any(keyword in h1_lower for keyword in ["阿苏焉尼单位数据", "unit data"]):
            # 从H2、H3识别UNIT（支持同义词）
            if h2:
                unit_infos = self._extract_units_with_synonyms(h2)
                for unit_info in unit_infos:
                    if unit_info["name"]:
                        result["units"].append(unit_info["name"])
            if h3:
                unit_infos = self._extract_units_with_synonyms(h3)
                for unit_info in unit_infos:
                    if unit_info["name"]:
                        result["units"].append(unit_info["name"])
        
        # 4. 如果H1包含"派系"关键字
        elif any(keyword in h1_lower for keyword in ["派系", "faction"]):
            # 从H2识别FACTION
            if h2:
                faction_terms = self._extract_chinese_part(h2)
                if faction_terms:
                    result["factions"].append(faction_terms)
        
        # 5. 通用层级识别
        else:
            # 根据H2、H3的内容判断分类
            if h2:
                category = self._determine_category_by_priority(self._extract_chinese_part(h2), h2)
                if category and category in result:
                    result[category].append(self._extract_chinese_part(h2))
            
            if h3:
                category = self._determine_category_by_priority(self._extract_chinese_part(h3), h3)
                if category and category in result:
                    result[category].append(self._extract_chinese_part(h3))
        
        return result

    def _extract_from_headers_with_origin(self, chunk: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """从标题中抽取术语（包含原始文本信息）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        h1 = chunk.get("h1", "")
        h2 = chunk.get("h2", "")
        h3 = chunk.get("h3", "")
        
        if not h1:
            return result
        
        # Step 1: 基于H1的层级识别
        h1_lower = h1.lower()
        
        # 1. 如果H1包含"军队规则"关键字
        if any(keyword in h1_lower for keyword in ["军队规则", "army rule"]):
            # 从H2、H3识别MECHANICS
            if h2:
                mechanic_terms = self._extract_chinese_part(h2)
                if mechanic_terms:
                    result["mechanics"].append({
                        "term": mechanic_terms,
                        "original": h2
                    })
            if h3:
                mechanic_terms = self._extract_chinese_part(h3)
                if mechanic_terms:
                    result["mechanics"].append({
                        "term": mechanic_terms,
                        "original": h3
                    })
        
        # 2. 如果H1包含"分队规则"关键字
        elif any(keyword in h1_lower for keyword in ["分队规则", "detachment rule"]):
            # 记录H2的分队名称(DETACHMENT)
            if h2:
                detachment_terms = self._extract_chinese_part(h2)
                if detachment_terms:
                    result["detachments"].append({
                        "term": detachment_terms,
                        "original": h2
                    })
        
        # 3. 如果H1包含"阿苏焉尼单位数据"关键字
        elif any(keyword in h1_lower for keyword in ["阿苏焉尼单位数据", "unit data"]):
            # 只从H3识别具体单位，H2作为单位类别
            if h3:
                unit_infos = self._extract_units_with_synonyms(h3)
                for unit_info in unit_infos:
                    if unit_info["name"]:
                        result["units"].append({
                            "term": unit_info["name"],
                            "original": h3,
                            "synonyms": unit_info["synonyms"],
                            "category": h2  # H2作为单位类别
                        })
        
        # 4. 如果H1包含"派系"关键字
        elif any(keyword in h1_lower for keyword in ["派系", "faction"]):
            # 从H2识别FACTION
            if h2:
                faction_terms = self._extract_chinese_part(h2)
                if faction_terms:
                    result["factions"].append({
                        "term": faction_terms,
                        "original": h2
                    })
        
        # 5. 通用层级识别
        else:
            # 只从H3识别具体单位，H2作为单位类别
            if h3:
                category = self._determine_category_by_priority(self._extract_chinese_part(h3), h3)
                if category and category in result:
                    # 如果是单位分类，使用同义词解析
                    if category == "units":
                        unit_infos = self._extract_units_with_synonyms(h3)
                        for unit_info in unit_infos:
                            if unit_info["name"]:
                                result[category].append({
                                    "term": unit_info["name"],
                                    "original": h3,
                                    "synonyms": unit_info["synonyms"],
                                    "category": h2  # H2作为单位类别
                                })
                    else:
                        result[category].append({
                            "term": self._extract_chinese_part(h3),
                            "original": h3
                        })
        
        return result
    
    def _tokenize_header(self, header_text: str) -> List[str]:
        """对标题进行分词处理"""
        if not header_text:
            return []
        
        # 使用",|/"和空格进行分词
        tokens = re.split(r'[,|/\s]+', header_text)
        # 提取中文部分
        chinese_tokens = []
        for token in tokens:
            chinese_part = self._extract_chinese_part(token)
            if chinese_part:
                chinese_tokens.append(chinese_part)
        
        return chinese_tokens
    
    def _extract_chinese_part(self, text: str) -> str:
        """提取中文部分（包括括号和特殊符号）"""
        if not text:
            return ""
        
        # 匹配中文字符、括号、空格、点号（用于武器名称如"阿苏瓦 · 无声尖啸者"）
        pattern = r'[\u4e00-\u9fff\s\(\)（）·]+'
        matches = re.findall(pattern, text)
        
        if matches:
            result = ''.join(matches).strip()
            # 清理多余的空格
            result = re.sub(r'\s+', ' ', result).strip()
            return result
        
        return ""
    
    def _parse_unit_with_synonyms(self, text: str) -> Dict[str, Any]:
        """解析单位名称和同义词
        
        Args:
            text: 包含单位名称的文本，如"千面(维萨奇)"或"司战（维萨奇）"
            
        Returns:
            Dict包含:
            - name: 单位名称（括号前的内容）
            - synonyms: 同义词列表（括号内的内容）
        """
        if not text:
            return {"name": "", "synonyms": []}
        
        # 清理文本，去除前后空格
        text = text.strip()
        
        # 匹配括号模式：支持中英文括号
        # 匹配如：千面(维萨奇)、司战（维萨奇）、千面(维萨奇) AUTARCH
        pattern = r'^([^\(（]+)[\(（]([^\)）]*)[\)）]'
        match = re.match(pattern, text)
        
        if match:
            unit_name = match.group(1).strip()
            synonyms_text = match.group(2).strip()
            
            # 如果括号内为空，只返回单位名称，不包含同义词
            if not synonyms_text:
                return {
                    "name": unit_name,
                    "synonyms": []
                }
            
            # 处理多个同义词（用逗号分隔）
            synonyms = [s.strip() for s in synonyms_text.split(',') if s.strip()]
            
            return {
                "name": unit_name,
                "synonyms": synonyms
            }
        else:
            # 没有括号，直接返回原文本作为名称
            return {
                "name": text,
                "synonyms": []
            }
    
    def _extract_units_with_synonyms(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取单位及其同义词"""
        if not text:
            return []
        
        # 使用现有的中文提取方法
        chinese_part = self._extract_chinese_part(text)
        if not chinese_part:
            return []
        
        # 解析单位名称和同义词
        unit_info = self._parse_unit_with_synonyms(chinese_part)
        
        if unit_info["name"]:
            return [unit_info]
        
        return []
    
    def _extract_from_text_with_mechanics(self, text: str) -> Dict[str, List[str]]:
        """从文本中抽取术语（包含机制，不抽取单位）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        if not text:
            return result
        
        # 按优先级顺序抽取：detachments > mechanics（跳过单位抽取）
        # 1. 抽取分队术语
        detachment_terms = self._extract_detachment_terms(text)
        result["detachments"].extend(detachment_terms)
        
        # 2. 抽取机制术语
        mechanic_terms = self._extract_mechanic_terms(text)
        result["mechanics"].extend(mechanic_terms)
        
        return result
    
    def _extract_from_text_no_mechanics(self, text: str) -> Dict[str, List[str]]:
        """从文本中抽取术语（不包含机制，不抽取单位）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        if not text:
            return result
        
        # 只抽取分队术语（跳过单位抽取）
        # 1. 抽取分队术语
        detachment_terms = self._extract_detachment_terms(text)
        result["detachments"].extend(detachment_terms)
        
        return result

    def _extract_from_text_with_mechanics_with_origin(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """从文本中抽取术语（包含机制，不抽取单位，包含原始文本信息）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        if not text:
            return result
        
        # 按优先级顺序抽取：detachments > mechanics（跳过单位抽取）
        # 1. 抽取分队术语
        detachment_terms = self._extract_detachment_terms(text)
        for term in detachment_terms:
            result["detachments"].append({
                "term": term,
                "original": text
            })
        
        # 2. 抽取机制术语
        mechanic_terms = self._extract_mechanic_terms(text)
        for term in mechanic_terms:
            result["mechanics"].append({
                "term": term,
                "original": text
            })
        
        return result
    
    def _extract_from_text_no_mechanics_with_origin(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """从文本中抽取术语（不包含机制，不抽取单位，包含原始文本信息）"""
        result = {
            "units": [],
            "detachments": [],
            "factions": [],
            "mechanics": []
        }
        
        if not text:
            return result
        
        # 只抽取分队术语（跳过单位抽取）
        # 1. 抽取分队术语
        detachment_terms = self._extract_detachment_terms(text)
        for term in detachment_terms:
            result["detachments"].append({
                "term": term,
                "original": text
            })
        
        return result
    
    def _determine_category_by_hierarchy(self, h1: str, h2: str, h3: str) -> str:
        """根据标题层级确定分类"""
        # 首先尝试模式匹配
        if h1:
            pattern_category = self._get_category_by_pattern(h1)
            if pattern_category:
                return pattern_category
        
        if h2:
            pattern_category = self._get_category_by_pattern(h2)
            if pattern_category:
                return pattern_category
        
        # 检查h2中的关键词（精确匹配）
        if h2 in self.category_mapping:
            return self.category_mapping[h2]
        
        # 检查h1中的关键词（精确匹配）
        if h1 in self.category_mapping:
            return self.category_mapping[h1]
        
        # 根据内容判断
        if any(keyword in h2.lower() for keyword in ["人物", "战线", "载具", "骑乘","巨兽","步兵"]):
            return "units"
        elif any(keyword in h2.lower() for keyword in ["武器", "射击", "近战"]):
            return "weapons"
        elif any(keyword in h1.lower() for keyword in ["规则", "技能"]):
            return "mechanics"
        elif any(keyword in h1.lower() for keyword in ["派系", "种族"]):
            return "factions"
        
        return "general"
    
    def _determine_category_by_content_type(self, content_type: str) -> str:
        """根据内容类型确定分类（兼容性方法）"""
        if content_type == "army_rule":
            return "mechanics"
        elif content_type == "detachment_rule":
            return "detachments"
        elif content_type == "unit":
            return "units"
        else:
            return "general"
    
    def _load_known_factions(self) -> List[str]:
        """从词典加载已知派系（包括主派系和子派系）"""
        factions = []
        try:
            dict_path = "dict/wh40k_vocabulary/factions.json"
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for term, info in data.items():
                        factions.append(term)  # 主术语
                        if 'synonyms' in info:
                            factions.extend(info['synonyms'])  # 别名
                logger.debug(f"从词典加载派系术语: {len(factions)} 个")
            else:
                logger.warning(f"词典文件不存在: {dict_path}")
        except Exception as e:
            logger.error(f"加载派系词典失败: {e}")
        
        return factions
    
    def _match_known_terms(self, text: str, known_terms: List[str]) -> List[str]:
        """在文本中匹配已知术语（支持分词和优先级）"""
        # 1. 分词：用"/,|"和空格分割
        import re
        tokens = re.split(r'[/,|\s]+', text.strip())
        tokens = [token.strip() for token in tokens if token.strip()]
        
        # 2. 匹配每个分词
        matched_terms = []
        for token in tokens:
            for term in known_terms:
                if term == token:  # 精确匹配
                    matched_terms.append(term)
                    break
        
        # 3. 去重
        unique_terms = list(dict.fromkeys(matched_terms))
        
        return unique_terms
    
    def _extract_known_factions(self, faction_text: str) -> List[str]:
        """从faction文本中提取已知派系"""
        known_factions = self._load_known_factions()
        return self._match_known_terms(faction_text, known_factions)
    
    def _clean_terms(self, terms: List[str]) -> List[str]:
        """清理和过滤术语"""
        cleaned = []
        for term in terms:
            term = term.strip()
            if term and len(term) >= 2:  # 过滤空字符串和太短的术语
                cleaned.append(term)
        return list(set(cleaned))  # 去重

    def _clean_terms_with_origin(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理和过滤术语（包含原始文本信息）"""
        cleaned = []
        for term_info in terms:
            term = term_info.get("term", "").strip()
            if term and len(term) >= 2:  # 过滤空字符串和太短的术语
                cleaned.append(term_info)
        return cleaned
    
    # 添加缺失的方法以支持测试
    def _extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """从文本中抽取术语（兼容性方法）"""
        return self._extract_from_text_with_mechanics(text)
    
    def _extract_unit_terms(self, text: str) -> List[str]:
        """提取单位术语"""
        unit_pattern = self.term_patterns["unit_pattern"]
        matches = re.findall(unit_pattern, text)
        return self._clean_terms(matches)
    
    def _extract_weapon_terms(self, text: str) -> List[str]:
        """提取武器术语"""
        weapon_pattern = self.term_patterns["weapon_pattern"]
        matches = re.findall(weapon_pattern, text)
        return self._clean_terms(matches)
    
    def _extract_skill_terms(self, text: str) -> List[str]:
        """提取技能术语"""
        skill_pattern = self.term_patterns["skill_pattern"]
        matches = re.findall(skill_pattern, text)
        return self._clean_terms(matches)
    
    def _extract_faction_terms(self, text: str) -> List[str]:
        """提取派系术语（只识别已知的）"""
        known_factions = self._load_known_factions()
        return self._match_known_terms(text, known_factions)
    
    def _extract_mechanic_terms(self, text: str) -> List[str]:
        """提取机制术语"""
        mechanic_pattern = self.term_patterns["mechanic_pattern"]
        matches = re.findall(mechanic_pattern, text)
        return self._clean_terms(matches)
    
    def _categorize_terms(self, terms: List[str], chunk: Dict[str, Any]) -> Dict[str, List[str]]:
        """对术语进行分类（兼容性方法）"""
        result = {
            "units": [],
            "weapons": [],
            "skills": [],
            "factions": [],
            "mechanics": []
        }
        
        for term in terms:
            category = self._determine_category_by_hierarchy(
                chunk.get("h1", ""),
                chunk.get("h2", ""),
                chunk.get("h3", "")
            )
            if category != "general":
                result[category].append(term)
        
        return result
    
    def _determine_category_by_priority(self, chinese_part: str, full_text: str) -> str:
        """按优先级确定分类：faction > detachment > unit > mechanic"""
        if not chinese_part:
            return None
        
        # 1. 检查是否是已知派系
        known_factions = self._load_known_factions()
        if chinese_part in known_factions:
            return "factions"
        
        # 2. 检查是否是已知分队
        known_detachments = self._load_known_detachments()
        if chinese_part in known_detachments:
            return "detachments"
        
        # 3. 根据关键词判断
        text_lower = full_text.lower()
        
        # 单位相关关键词
        if any(keyword in text_lower for keyword in ["人物", "战线", "载具", "骑乘", "巨兽", "步兵"]):
            return "units"
        
        # 机制相关关键词
        if any(keyword in text_lower for keyword in ["阶段", "测试", "保护"]):
            return "mechanics"
        
        # 默认分类
        return "units"
    
    def _load_known_detachments(self) -> List[str]:
        """加载已知分队术语"""
        try:
            dict_path = os.path.join("dict", "wh40k_vocabulary", "detachment.json")
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    detachments = json.load(f)
                    terms = []
                    for key, value in detachments.items():
                        terms.append(key)
                        if "synonyms" in value:
                            terms.extend(value["synonyms"])
                    return terms
        except Exception as e:
            logger.warning(f"加载detachment词典失败: {e}")
        
        return [] 

    def _extract_detachment_terms(self, text: str) -> List[str]:
        """抽取分队术语（兼容性方法）"""
        # 从已知分队词典中识别
        known_detachments = self._load_known_detachments()
        return self._match_known_terms(text, known_detachments) 

    def _filter_clear_unit_terms(self, unit_terms: List[str], context_text: str) -> List[str]:
        """过滤出明确的单位术语"""
        clear_units = []
        context_lower = context_text.lower()
        
        for term in unit_terms:
            # 只保留包含明确单位关键词的术语
            if any(keyword in term.lower() for keyword in [
                "单位", "模型", "人物", "战线", "载具", "骑乘", "巨兽", "步兵",
                "守护者", "翔鹰", "突击蝎", "闪矛", "织", "虚空", "幽冥"
            ]):
                clear_units.append(term)
            # 或者术语在上下文中明确被标识为单位
            elif any(keyword in context_lower for keyword in [
                f"{term}单位", f"{term}模型", f"{term}人物"
            ]):
                clear_units.append(term)
        
        return clear_units 




 