import re
import os
import tiktoken
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class ContentType(Enum):
    """内容类型枚举"""
    ARMY_RULE = "army_rule"           # 军队规则
    DETACHMENT_RULE = "detachment_rule"  # 分队规则
    UNIT = "unit"                     # 单位数据
    CORERULE = "corerule"             # 核心规则


class DocumentStructureExtractor:
    """
    文档结构抽取器
    
    核心职责：
    - 预先抽取文档的标题层级结构
    - 构建完整的文档大纲
    - 为后续的切片过程提供结构指导
    """
    
    def __init__(self):
        self.structure_cache = {}
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """
        抽取文档的标题层级结构
        
        Args:
            text: 文档文本内容
            
        Returns:
            包含完整结构信息的字典
        """
        lines = text.split('\n')
        structure = {
            'headings': [],
            'hierarchy_map': {},  # 标题到层级信息的映射
            'section_boundaries': [],  # 一级标题边界
            'content_sections': {}  # 每个section的内容范围
        }
        
        current_section = None
        current_section_start = 0
        title_stack = ["" for _ in range(6)]
        
        for i, line in enumerate(lines):
            m = re.match(r'^(#{1,6})\s+(.*)', line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                
                # 更新标题栈
                title_stack[level-1] = title
                # 清空下级标题
                for j in range(level, 6):
                    title_stack[j] = ""
                
                # 构建层级信息
                hierarchy = {}
                for k in range(6):
                    if title_stack[k]:
                        hierarchy[f'level{k+1}'] = title_stack[k]
                
                heading_info = {
                    'line_number': i,
                    'title': title,
                    'level': level,
                    'hierarchy': hierarchy.copy(),
                    'title_stack': title_stack.copy()
                }
                
                structure['headings'].append(heading_info)
                structure['hierarchy_map'][title] = hierarchy
                
                # 处理一级标题边界
                if level == 1:
                    # 保存前一个section的信息
                    if current_section:
                        structure['content_sections'][current_section] = {
                            'start_line': current_section_start,
                            'end_line': i - 1,
                            'title': current_section
                        }
                    
                    current_section = title
                    current_section_start = i
                    structure['section_boundaries'].append({
                        'title': title,
                        'line_number': i,
                        'hierarchy': hierarchy
                    })
        
        # 保存最后一个section
        if current_section:
            structure['content_sections'][current_section] = {
                'start_line': current_section_start,
                'end_line': len(lines) - 1,
                'title': current_section
            }
        
        return structure
    
    def validate_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证文档结构的合理性
        
        Args:
            structure: 抽取的结构信息
            
        Returns:
            验证结果
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        headings = structure['headings']
        
        # 统计信息
        validation_result['statistics'] = {
            'total_headings': len(headings),
            'level1_count': len([h for h in headings if h['level'] == 1]),
            'level2_count': len([h for h in headings if h['level'] == 2]),
            'max_depth': max([h['level'] for h in headings]) if headings else 0
        }
        
        # 检查一级标题
        level1_headings = [h for h in headings if h['level'] == 1]
        if len(level1_headings) == 0:
            validation_result['warnings'].append("文档没有一级标题，可能影响切片质量")
        
        # 检查标题层级跳跃
        for i in range(1, len(headings)):
            prev_level = headings[i-1]['level']
            curr_level = headings[i]['level']
            if curr_level > prev_level + 1:
                validation_result['warnings'].append(
                    f"标题层级跳跃：从 {prev_level} 级跳到 {curr_level} 级 "
                    f"（{headings[i-1]['title']} -> {headings[i]['title']}）"
                )
        
        # 检查重复标题
        titles = [h['title'] for h in headings]
        duplicates = [title for title in set(titles) if titles.count(title) > 1]
        if duplicates:
            validation_result['warnings'].append(f"发现重复标题：{duplicates}")
        
        return validation_result


class StructureValidator:
    """
    结构验证器
    
    核心职责：
    - 在chunk生成过程中验证层级结构的正确性
    - 防止跨section的内容合并
    - 确保chunk的层级信息准确
    """
    
    def __init__(self, document_structure: Dict[str, Any]):
        """
        初始化结构验证器
        
        Args:
            document_structure: 文档结构信息
        """
        self.document_structure = document_structure
        self.hierarchy_map = document_structure.get('hierarchy_map', {})
        self.section_boundaries = document_structure.get('section_boundaries', [])
    
    def validate_chunk_hierarchy(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证chunk的层级信息是否正确
        
        Args:
            chunk: 待验证的chunk
            
        Returns:
            验证结果
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        chunk_hierarchy = chunk.get('hierarchy', {})
        section_heading = chunk.get('section_heading', '')
        
        # 检查section_heading是否在文档结构中
        if section_heading and section_heading not in self.hierarchy_map:
            validation_result['warnings'].append(
                f"chunk的section_heading '{section_heading}' 不在文档结构中"
            )
        
        # 检查层级信息的一致性
        if section_heading and section_heading in self.hierarchy_map:
            expected_hierarchy = self.hierarchy_map[section_heading]
            if chunk_hierarchy != expected_hierarchy:
                validation_result['warnings'].append(
                    f"chunk层级信息与文档结构不一致：\n"
                    f"期望：{expected_hierarchy}\n"
                    f"实际：{chunk_hierarchy}"
                )
        
        return validation_result
    
    def can_merge_chunks(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> bool:
        """
        判断两个chunk是否可以合并
        
        Args:
            chunk1: 第一个chunk
            chunk2: 第二个chunk
            
        Returns:
            是否可以合并
        """
        # 获取两个chunk的顶级section
        section1 = chunk1.get('hierarchy', {}).get('level1', '')
        section2 = chunk2.get('hierarchy', {}).get('level1', '')
        
        # 如果顶级section不同，禁止合并
        if section1 and section2 and section1 != section2:
            return False
        
        # 检查section_heading
        heading1 = chunk1.get('section_heading', '')
        heading2 = chunk2.get('section_heading', '')
        
        # 如果section_heading不同，禁止合并
        if heading1 and heading2 and heading1 != heading2:
            return False
        
        # 检查是否跨越了一级标题边界
        if section1 in [boundary['title'] for boundary in self.section_boundaries]:
            # 如果chunk1属于某个一级标题，chunk2必须也属于同一个
            if section2 and section2 != section1:
                return False
        
        return True
    
    def get_correct_hierarchy(self, section_heading: str) -> Dict[str, str]:
        """
        根据section_heading获取正确的层级信息
        
        Args:
            section_heading: 章节标题
            
        Returns:
            正确的层级信息
        """
        return self.hierarchy_map.get(section_heading, {})


class SemanticDocumentChunker:
    """
    语义文档切片处理类 v3.1 (Refactored)
    
    核心职责：
    - 专注于文档的语义切片和结构化处理
    - 不再负责任何数据库或外部服务（如Neo4j）的交互
    """
    
    def __init__(self,
                 encoding_name: str = 'cl100k_base',
                 max_tokens: int = 512,
                 overlap_tokens: int = 50,
                 min_semantic_units: int = 2,
                 document_name: str = "aeldari",
                 faction_name: str = "阿苏焉尼"):
        """
        初始化语义文档切片器
        
        Args:
            encoding_name: 分词器名称
            max_tokens: 最大token数
            overlap_tokens: 重叠token数
            min_semantic_units: 最小语义单元数（句子数）
            document_name: 文档名称
            faction_name: 分支势力名称
        """
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_semantic_units = min_semantic_units
        self.document_name = document_name
        self.faction_name = faction_name
        
        # 初始化目录层级栈
        self.hierarchy_stack = []
        
        # 初始化结构抽取器和验证器
        self.structure_extractor = DocumentStructureExtractor()
        self.structure_validator = None
        self.document_structure = None

    def _count_tokens(self, text: str) -> int:
        """返回文本的token数"""
        return len(self.encoder.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """
        按语义边界断句，保持语义完整性
        
        断句规则：
        1. 句号、问号、感叹号
        2. 分号（如果前后都有内容）
        3. 冒号（如果前面是标题性内容）
        4. 换行符（如果前后都有内容且不是列表项）
        """
        # 先按标点符号断句
        sentences = re.split(r'(?<=[。！？?!])\s*', text)
        
        # 进一步处理分号和冒号
        refined_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 处理分号
            if ';' in sentence:
                parts = sentence.split(';')
                for i, part in enumerate(parts):
                    if part.strip():
                        if i < len(parts) - 1:
                            refined_sentences.append(part.strip() + ';')
                        else:
                            refined_sentences.append(part.strip())
            else:
                refined_sentences.append(sentence.strip())
        
        return [s for s in refined_sentences if s.strip()]

    def _parse_heading_hierarchy(self, heading: str, level: int, hierarchy_stack=None) -> Dict[str, str]:
        """
        只生成h1~h6字段，分别对应1-6级标题。
        """
        if hierarchy_stack is None:
            return {}
        hierarchy = {}
        for i in range(6):
            hierarchy[f'h{i+1}'] = hierarchy_stack[i] if hierarchy_stack[i] else ""
        return hierarchy

    def _determine_content_type(self, heading: str, level: int, text: str) -> str:
        """
        根据标题和内容确定content_type
        
        Args:
            heading: 标题文本
            level: 标题级别
            text: 内容文本
            
        Returns:
            ContentType枚举值
        """
        if not heading:
            return ContentType.CORERULE.value
        
        heading_lower = heading.lower()
        text_lower = text.lower()
        
        # 军队规则判断
        if any(keyword in heading_lower for keyword in ['军队规则', 'army rule', '战斗专注', '殊途同归']):
            return ContentType.ARMY_RULE.value
        
        # 分队规则判断
        if any(keyword in heading_lower for keyword in ['分队规则', 'detachment', '军阵', 'host']):
            return ContentType.DETACHMENT_RULE.value
        
        # 单位数据判断
        if any(keyword in heading_lower for keyword in ['单位数据', 'unit', '人物', 'character', '步兵', 'infantry', '载具', 'vehicle']):
            return ContentType.UNIT.value
        
        # 核心规则判断
        if any(keyword in heading_lower for keyword in ['核心规则', 'core rule', '游戏规则', 'game rule']):
            return ContentType.CORERULE.value
        
        # 默认判断
        if level >= 3 and any(keyword in text_lower for keyword in ['M ', 'T ', 'SV ', 'W ', 'LD ', 'OC ']):
            return ContentType.UNIT.value
        
        return ContentType.CORERULE.value

    def _build_chunk_type(self, hierarchy: Dict[str, str], content_type: str) -> str:
        """
        构建chunk_type字符串，仅用h1~h6
        """
        if content_type == ContentType.ARMY_RULE.value:
            return f"{self.document_name}军队规则-{hierarchy.get('h1', '')}"
        elif content_type == ContentType.DETACHMENT_RULE.value:
            return f"{self.document_name}分队规则-{hierarchy.get('h1', '')}"
        elif content_type == ContentType.UNIT.value:
            # 构建单位数据的chunk_type
            parts = []
            for i in range(3):
                if hierarchy.get(f'h{i+1}', ''):
                    parts.append(hierarchy[f'h{i+1}'])
            return f"{self.document_name}单位数据-{'-'.join(parts)}"
        else:
            return f"{self.document_name}核心规则-{hierarchy.get('h1', '')}"

    def _build_faction(self) -> str:
        """
        构建faction字符串
        
        Returns:
            faction字符串，格式：文档名称-分支势力
        """
        return f"{self.document_name}-{self.faction_name}"

    def _split_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        按Markdown标题拆分，保持层级结构，记录每个section的完整标题栈快照。
        一级标题（#）出现时，强制分割section，防止串台。
        Returns:
            List of dicts with keys: heading, level, content, hierarchy_stack
        """
        sections = []
        current_heading = None
        current_content = []
        title_stack = ["" for _ in range(6)]
        lines = text.split('\n')
        
        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.*)', line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                
                # 保存前一个section（如果有的话）
                if current_heading is not None and current_content:
                    sections.append({
                        'heading': current_heading,
                        'level': current_heading.count('#'),
                        'content': '\n'.join(current_content).strip(),
                        'hierarchy_stack': title_stack.copy()
                    })
                
                # 处理新标题 - 更新标题栈
                title_stack[level-1] = title
                # 清空下级标题
                for i in range(level, 6):
                    title_stack[i] = ""
                
                current_heading = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个section
        if current_heading is not None and current_content:
            sections.append({
                'heading': current_heading,
                'level': current_heading.count('#'),
                'content': '\n'.join(current_content).strip(),
                'hierarchy_stack': title_stack.copy()
            })
        
        # 处理没有标题的内容
        if not sections and text.strip():
            sections.append({
                'heading': None,
                'level': 0,
                'content': text.strip(),
                'hierarchy_stack': ["" for _ in range(6)]
            })
        
        return sections

    def _assess_semantic_coherence(self, text: str) -> Dict[str, Any]:
        """
        评估文本的语义连贯性
        
        Returns:
            Dict with coherence metrics
        """
        sentences = self._split_sentences(text)
        
        # 基础指标
        sentence_count = len(sentences)
        avg_sentence_length = sum(len(s) for s in sentences) / max(sentence_count, 1)
        
        # 语义连贯性指标
        coherence_score = 0.0
        
        # 1. 句子数量合理性
        if sentence_count >= self.min_semantic_units:
            coherence_score += 0.3
        
        # 2. 句子长度合理性
        if 10 <= avg_sentence_length <= 200:
            coherence_score += 0.3
        
        # 3. 段落结构合理性
        if text.count('\n\n') <= 3:  # 避免过度分段
            coherence_score += 0.2
        
        # 4. 标点符号使用合理性
        punctuation_ratio = len(re.findall(r'[。！？，；：]', text)) / max(len(text), 1)
        if 0.05 <= punctuation_ratio <= 0.3:
            coherence_score += 0.2
        
        return {
            'coherence_score': min(coherence_score, 1.0),
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'punctuation_ratio': round(punctuation_ratio, 3)
        }

    def _get_last_n_tokens(self, text: str, n: int) -> str:
        """返回文本的最后n个token对应的字符串（用于overlap）"""
        tokens = self.encoder.encode(text)
        last_tokens = tokens[-n:]
        return self.encoder.decode(last_tokens)

    def chunk_text(self, text: str, extra_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        主方法：基于语义完整性的文档切片
        
        Args:
            text: 输入文本
            extra_metadata: 额外的元数据
            
        Returns:
            List of chunk dicts with semantic metadata
        """
        # 第一步：抽取文档结构
        print("正在抽取文档结构...")
        self.document_structure = self.structure_extractor.extract_structure(text)
        
        # 验证文档结构
        validation_result = self.structure_extractor.validate_structure(self.document_structure)
        if validation_result['warnings']:
            print("文档结构验证警告：")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        # 初始化结构验证器
        self.structure_validator = StructureValidator(self.document_structure)
        
        # 第二步：按标题分割文档
        sections = self._split_headings(text)
        chunks = []
        
        for section in sections:
            heading = section['heading']
            level = section['level']
            content = section['content']
            hierarchy_stack = section.get('hierarchy_stack', ["" for _ in range(6)])
            
            if not content.strip():
                continue
            
            # 解析层级信息
            hierarchy = self._parse_heading_hierarchy(heading, level, hierarchy_stack) if heading else {}
            
            # 使用结构验证器获取正确的层级信息
            if heading and heading in self.document_structure['hierarchy_map']:
                correct_hierarchy = self.structure_validator.get_correct_hierarchy(heading)
                if correct_hierarchy:
                    hierarchy = correct_hierarchy
            
            # 确定内容类型
            content_type = self._determine_content_type(heading, level, content)
            
            # 按句子分割内容
            sentences = self._split_sentences(content)
            buffer = ''
            sentence_count = 0
            
            for sentence in sentences:
                candidate = (buffer + ' ' + sentence).strip()
                candidate_tokens = self._count_tokens(candidate)
                
                # 检查是否超过token限制
                if candidate_tokens <= self.max_tokens:
                    buffer = candidate
                    sentence_count += 1
                else:
                    # 保存当前chunk
                    if buffer.strip():
                        chunk_metadata = self._create_chunk_metadata(
                            buffer, heading, level, sentence_count, hierarchy, content_type, extra_metadata
                        )
                        chunks.append(chunk_metadata)
                    
                    # 创建新的buffer，包含overlap
                    if buffer:
                        overlap = self._get_last_n_tokens(buffer, self.overlap_tokens)
                        buffer = (overlap + ' ' + sentence).strip()
                    else:
                        buffer = sentence
                    sentence_count = 1
            
            # 保存最后一个chunk
            if buffer.strip():
                chunk_metadata = self._create_chunk_metadata(
                    buffer, heading, level, sentence_count, hierarchy, content_type, extra_metadata
                )
                chunks.append(chunk_metadata)
        
        return chunks

    def _create_chunk_metadata(self, 
                              text: str, 
                              heading: Optional[str], 
                              level: int,
                              sentence_count: int,
                              hierarchy: Dict[str, str],
                              content_type: str,
                              extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建chunk的精简元数据
        """
        metadata = {
            'text': text,
            'section_heading': heading,
            'chunk_type': self._build_chunk_type(hierarchy, content_type),
            'content_type': content_type,
            'faction': self._build_faction(),
        }
        # 只保留h1~h6
        for i in range(6):
            metadata[f'h{i+1}'] = hierarchy.get(f'h{i+1}', "")
        return metadata

    def optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化chunks，合并过小的chunks，分割过大的chunks
        
        Args:
            chunks: 原始chunks列表
            
        Returns:
            优化后的chunks列表
        """
        if not self.structure_validator:
            print("警告：结构验证器未初始化，使用基础合并逻辑")
            return self._basic_optimize_chunks(chunks)
        
        optimized_chunks = []
        
        for chunk in chunks:
            text = chunk['text']
            token_count = self._count_tokens(text)
            
            # 如果chunk太小，尝试与下一个chunk合并
            if token_count < self.max_tokens * 0.3 and len(optimized_chunks) > 0:
                last_chunk = optimized_chunks[-1]
                
                # 使用结构验证器判断是否可以合并
                if self.structure_validator.can_merge_chunks(last_chunk, chunk):
                    combined_text = last_chunk['text'] + '\n\n' + text
                    combined_tokens = self._count_tokens(combined_text)
                    
                    # 如果合并后不超过限制，则合并
                    if combined_tokens <= self.max_tokens:
                        last_chunk['text'] = combined_text
                        # 其余字段无需更新
                        # 更新层级信息 - 优先使用当前chunk的层级信息（更具体）
                        if chunk.get('h1') and last_chunk.get('h1'):
                            current_hierarchy = {f'h{i+1}': chunk.get(f'h{i+1}', "") for i in range(6)}
                            last_hierarchy = {f'h{i+1}': last_chunk.get(f'h{i+1}', "") for i in range(6)}
                            current_depth = len([v for v in current_hierarchy.values() if v])
                            last_depth = len([v for v in last_hierarchy.values() if v])
                            if current_depth >= last_depth:
                                for i in range(6):
                                    last_chunk[f'h{i+1}'] = chunk.get(f'h{i+1}', "")
                                last_chunk['section_heading'] = chunk.get('section_heading', last_chunk.get('section_heading'))
                                last_chunk['chunk_type'] = chunk.get('chunk_type', last_chunk.get('chunk_type'))
                                last_chunk['content_type'] = chunk.get('content_type', last_chunk.get('content_type'))
                        continue
                else:
                    # 记录被拒绝的合并
                    print(f"拒绝合并：{chunk.get('section_heading', 'Unknown')} -> {last_chunk.get('section_heading', 'Unknown')}")
            
            # 如果chunk太大，尝试分割
            if token_count > self.max_tokens * 1.2:
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks

    def _basic_optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基础合并逻辑，不依赖已删除字段
        """
        optimized_chunks = []
        for chunk in chunks:
            text = chunk['text']
            token_count = self._count_tokens(text)
            if token_count < self.max_tokens * 0.3 and len(optimized_chunks) > 0:
                last_chunk = optimized_chunks[-1]
                combined_text = last_chunk['text'] + '\n\n' + text
                combined_tokens = self._count_tokens(combined_text)
                if combined_tokens <= self.max_tokens:
                    last_chunk['text'] = combined_text
                    continue
            if token_count > self.max_tokens * 1.2:
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        return optimized_chunks

    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分割过大的chunk
        """
        text = chunk['text']
        sentences = self._split_sentences(text)
        sub_chunks = []
        buffer = ''
        for sentence in sentences:
            candidate = (buffer + ' ' + sentence).strip()
            candidate_tokens = self._count_tokens(candidate)
            if candidate_tokens <= self.max_tokens:
                buffer = candidate
            else:
                if buffer.strip():
                    sub_chunk = chunk.copy()
                    sub_chunk['text'] = buffer
                    sub_chunks.append(sub_chunk)
                buffer = sentence
        if buffer.strip():
            sub_chunk = chunk.copy()
            sub_chunk['text'] = buffer
            sub_chunks.append(sub_chunk)
        return sub_chunks


def save_chunks_to_json(chunks: List[Dict[str, Any]], output_dir: str = "chunks") -> str:
    """保存切片结果到JSON文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"semantic_chunks_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return output_file


def save_chunks_to_text(chunks: List[Dict[str, Any]], output_dir: str = "chunks") -> str:
    """保存切片结果到文本文件，只包含核心字段"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"semantic_chunks_{timestamp}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"语义文档切片结果 - 总块数: {len(chunks)}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"----- 切片 {i} -----\n")
            for field in ["section_heading", "chunk_type", "content_type", "faction", "h1", "h2", "h3", "h4", "h5", "h6"]:
                f.write(f"{field}: {chunk.get(field, '')}\n")
            preview = chunk['text'][:200].replace('\n', ' ')
            f.write(f"内容预览: {preview} ...\n\n")
    return output_file


def main(test_file: str):
    print("\n==================== 语义文档切片器 v3.1 (Refactored) ====================")
    print("核心特性：\n- 专注于语义完整性和上下文保持\n- 强化语义连贯性评估\n- 智能chunk优化和分割\n- 结构化metadata设计（chunk_type, hierarchy, content_type, faction）\n")
    print("【命令行用法】\npython3 DATAUPLOD/data_vector_v3.py <测试文件路径>\n\n示例：\npython3 DATAUPLOD/data_vector_v3.py ../test/testdata/40kcorerule.md\n\n")

    print(f"正在处理文件：{test_file}")
    file_size = os.path.getsize(test_file)
    print(f"文件大小：{file_size} 字符\n")

    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print("开始语义切片...")
    chunker = SemanticDocumentChunker()
    print("正在抽取文档结构...")
    structure = chunker.structure_extractor.extract_structure(text)
    validation_result = chunker.structure_extractor.validate_structure(structure)
    if validation_result['warnings']:
        print("文档结构验证警告：")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    print("=== 文档结构信息 ===")
    print(f"总标题数：{validation_result['statistics']['total_headings']}")
    print(f"一级标题数：{validation_result['statistics']['level1_count']}")
    print(f"最大层级深度：{validation_result['statistics']['max_depth']}\n")
    print("一级标题列表：")
    for h in [h for h in structure['headings'] if h['level'] == 1]:
        print(f"  - {h['title']}")
    print("\n标题层级示例：")
    for h in structure['headings'][:5]:
        print(f"  {h['line_number']+1}. {'  '*(h['level']-1)}{('#'*h['level'])} {h['title']}")
        print(f"     层级：{{'h1': '{h['title_stack'][0]}', 'h2': '{h['title_stack'][1]}', 'h3': '{h['title_stack'][2]}'}}")
    
    chunks = chunker.chunk_text(text, extra_metadata={"source_file": os.path.basename(test_file), "file_path": test_file, "file_size": file_size})
    print(f"生成 {len(chunks)} 个切片")
    print("优化切片...")
    optimized_chunks = chunker.optimize_chunks(chunks)
    print(f"优化后 {len(optimized_chunks)} 个切片\n")

    print("=== 前3个切片的Metadata示例 ===\n")
    for i, chunk in enumerate(optimized_chunks[:3]):
        print(f"--- 切片 {i+1} ---")
        for field in ["section_heading", "chunk_type", "content_type", "faction", "h1", "h2", "h3", "h4", "h5", "h6"]:
            print(f"{field}: {chunk.get(field, '')}")
        preview = chunk['text'][:100].replace('\n', ' ')
        print(f"内容预览: {preview} ...\n")

    # 保存结果
    json_path = save_chunks_to_json(optimized_chunks)
    print(f"结果已保存：\nJSON文件：{json_path}\n切片完成！\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法：python3 data_vector_v3.py <测试文件路径>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    main(test_file) 