#!/usr/bin/env python3
"""
语义文档分块器 V4 (逻辑语义块版本)

核心特性：
- 从标题分块升级为逻辑语义块分块
- 智能识别关键词、技能、单位装备等边界
- 保持语义完整性和上下文连续性
- 结构化metadata设计（chunk_type, hierarchy, content_type, faction, sub_section）
"""

import re
import json
import tiktoken
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

from dataupload.config import LOGICAL_BLOCK_CONFIG
from dataupload.data_vector_v3 import SemanticDocumentChunker as SemanticDocumentChunkerV3


class DocumentStructureExtractor:
    """
    文档结构提取器
    
    核心职责：
    - 提取文档的标题层级结构
    - 识别h1-h6标题层级
    - 构建层级映射关系
    """
    
    def __init__(self):
        pass
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """
        提取文档结构
        
        Args:
            text: 要分析的文本
            
        Returns:
            文档结构信息
        """
        lines = text.split('\n')
        hierarchy_map = {}
        current_h1 = ""
        current_h2 = ""
        current_h3 = ""
        
        for line in lines:
            if line.startswith('# '):
                current_h1 = line[2:].strip()
                current_h2 = ""
                current_h3 = ""
                hierarchy_map[current_h1] = {
                    'level': 1,
                    'h2': {},
                    'h3': {}
                }
            elif line.startswith('## '):
                current_h2 = line[3:].strip()
                current_h3 = ""
                if current_h1:
                    hierarchy_map[current_h1]['h2'][current_h2] = {
                        'level': 2,
                        'h3': {}
                    }
            elif line.startswith('### '):
                current_h3 = line[4:].strip()
                if current_h1 and current_h2:
                    hierarchy_map[current_h1]['h2'][current_h2]['h3'][current_h3] = {
                        'level': 3
                    }
        
        return {
            'hierarchy_map': hierarchy_map,
            'total_h1': len([h for h in hierarchy_map.keys()]),
            'total_h2': sum(len(h['h2']) for h in hierarchy_map.values()),
            'total_h3': sum(len(h2['h3']) for h in hierarchy_map.values() for h2 in h['h2'].values())
        }


class StructureValidator:
    """
    结构验证器
    
    核心职责：
    - 验证提取的文档结构是否正确
    - 检查chunk的层级信息
    - 提供结构优化建议
    """
    
    def __init__(self, document_structure: Dict[str, Any]):
        self.document_structure = document_structure
        self.hierarchy_map = document_structure.get('hierarchy_map', {})
    
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
        
        section_heading = chunk.get('section_heading', '')
        
        # 检查section_heading是否在文档结构中
        if section_heading and section_heading not in self.hierarchy_map:
            validation_result['warnings'].append(
                f"section_heading '{section_heading}' 不在文档结构中"
                )
        
        return validation_result
    

class LogicalBlockSplitter:
    """
    逻辑语义块分割器
    
    核心职责：
    - 在 ### 层级进行逻辑语义块分割
    - 识别"关键词"、"技能"、"单位装备"等边界
    - 保持子列表与父技能的完整性
    """

    def __init__(self):
        self.boundary_keywords = LOGICAL_BLOCK_CONFIG["boundary_keywords"]

    def split_logical_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        将内容分割为逻辑语义块
        
        Args:
            content: 要分割的内容
            
        Returns:
            逻辑语义块列表，每个块包含 sub_section 和 content
        """
        lines = content.split('\n')
        blocks = []
        current_block = {'sub_section': 'basic', 'content_lines': []}
        
        # 技能合并逻辑：收集所有技能相关的内容
        skill_content_lines = []
        in_skill_section = False
        
        for line in lines:
            boundary_keyword = self._is_boundary_line(line)
            
            if boundary_keyword == '技能':
                # 如果已经在技能部分，先保存之前的技能内容
                if in_skill_section and skill_content_lines:
                    blocks.append({
                        'sub_section': '技能',
                        'content': '\n'.join(skill_content_lines).strip()
                    })
                    skill_content_lines = []
                
                # 开始新的技能部分
                in_skill_section = True
                skill_content_lines.append(line)
                
            elif boundary_keyword and boundary_keyword != '技能':
                # 如果当前在技能部分，遇到其他边界时结束技能部分
                if in_skill_section and skill_content_lines:
                    blocks.append({
                        'sub_section': '技能',
                        'content': '\n'.join(skill_content_lines).strip()
                    })
                    skill_content_lines = []
                    in_skill_section = False
                
                # 保存前一个块
                if current_block['content_lines']:
                    blocks.append({
                        'sub_section': current_block['sub_section'],
                        'content': '\n'.join(current_block['content_lines']).strip()
                    })
                
                # 开始新的边界块
                current_block = {
                    'sub_section': boundary_keyword,
                    'content_lines': [line]
                }
                
                # 立即保存边界块
                blocks.append({
                    'sub_section': current_block['sub_section'],
                    'content': current_block['content_lines'][0]
                })
                
                # 重新开始basic块
                current_block = {'sub_section': 'basic', 'content_lines': []}
                
            elif in_skill_section:
                # 在技能部分，收集所有内容直到遇到其他边界
                skill_content_lines.append(line)
                
            else:
                # 添加到当前块
                current_block['content_lines'].append(line)
        
        # 保存最后一个技能块
        if in_skill_section and skill_content_lines:
            blocks.append({
                'sub_section': '技能',
                'content': '\n'.join(skill_content_lines).strip()
            })
        
        # 保存最后一个块
        if current_block['content_lines']:
            blocks.append({
                'sub_section': current_block['sub_section'],
                'content': '\n'.join(current_block['content_lines']).strip()
            })
        
        return blocks

    def _is_boundary_line(self, line: str) -> Optional[str]:
        """
        判断是否为边界行
        
        Args:
            line: 要检查的行
            
        Returns:
            如果是边界行返回匹配的关键词，否则返回None
        """
        # 情况1: 匹配列表项中的格式：*   **关键词:** 内容
        # 使用更宽松的匹配，确保能匹配到长内容
        pattern1 = r'^\s*\*\s*\*\*(.+?)(?:\s*\([^)]+\))?\:\*\*'
        match1 = re.match(pattern1, line)
        
        if match1:
            keyword = match1.group(1).strip()
            # 精确匹配配置的关键词，但排除"核心技能"
            if keyword == "核心技能":
                return None  # "核心技能"不作为边界，而是技能内容的一部分
            # 特殊处理："技能 (Abilities):" 应该匹配为"技能"
            if keyword == "技能 (Abilities)":
                return "技能"
            matched_keyword = self._fuzzy_match_keyword(keyword)
            if matched_keyword:
                return matched_keyword
        
        # 情况2: 匹配独立的标题行，格式：**关键词:**
        pattern2 = r'^\s*\*\*(.+?)(?:\s*\([^)]+\))?\:\*\*\s*$'
        match2 = re.match(pattern2, line)
        
        if match2:
            keyword = match2.group(1).strip()
            # 精确匹配配置的关键词，但排除"核心技能"
            if keyword == "核心技能":
                return None  # "核心技能"不作为边界，而是技能内容的一部分
            # 特殊处理："技能 (Abilities):" 应该匹配为"技能"
            if keyword == "技能 (Abilities)":
                return "技能"
            matched_keyword = self._fuzzy_match_keyword(keyword)
            if matched_keyword:
                return matched_keyword
        
        return None

    def _fuzzy_match_keyword(self, keyword: str) -> Optional[str]:
        """
        精确匹配关键词
        
        Args:
            keyword: 从文档中提取的关键词
            
        Returns:
            匹配的配置关键词，如果没有匹配返回None
        """
        # 精确匹配：只有完全匹配配置的关键词才被识别
        if keyword in self.boundary_keywords:
            return keyword
        
        return None


class AdSemanticDocumentChunker:
    """
    高级语义文档分块器 V4 (Advanced)
    
    核心特性：
    - 专注于语义完整性和上下文保持
    - 强化语义连贯性评估
    - 智能chunk优化和分割
    - 结构化metadata设计（chunk_type, hierarchy, content_type, faction, sub_section）
    - 智能策略选择：逻辑语义块分块 vs V3兼容分块
    """
    
    def __init__(self,
                 document_name: str = "",
                 max_tokens: int = 512,
                 overlap_tokens: int = 50,
                 content_type: str = "unit",
                 faction_name: str = "unknown"):
        """
        初始化分块器
        
        Args:
            document_name: 文档名称
            max_tokens: 最大token数
            overlap_tokens: 重叠token数
            content_type: 内容类型
            faction_name: 分支势力名称，如果未传入则使用"unknown"
        """
        self.document_name = document_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.content_type = content_type
        self.faction_name = faction_name
        
        # 初始化组件
        self.structure_extractor = DocumentStructureExtractor()
        self.logical_block_splitter = LogicalBlockSplitter()
        self.structure_validator = None
        
        # 层级栈
        self.hierarchy_stack = ["" for _ in range(6)]
        
        # Token计数器
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        将文本分割为语义chunks
        
        Args:
            text: 要分割的文本
            metadata: 额外的元数据（保持与V3接口兼容，但不使用）
            
        Returns:
            chunks列表，每个chunk包含text和metadata
        """
        # Step 1: 策略选择
        strategy = self._choose_strategy(text)
        
        # Step 2: 根据策略执行
        if strategy == "logical_blocks":
            return self._process_with_logical_blocks(text)
        else:
            return self._process_with_v3_compatible(text, metadata)

    def _choose_strategy(self, text: str) -> str:
        """
        一次性判断文档适合的处理策略
        
        Args:
            text: 要分析的文本
            
        Returns:
            策略名称："logical_blocks" 或 "v3_compatible"
        """
        # 检查是否有H3标题
        h3_sections = self._find_h3_sections(text)
        
        if not h3_sections:
            return "v3_compatible"  # 场景A
        
        # 检查H3内容是否包含boundary_keywords
        for section in h3_sections:
            if self._contains_boundary_keywords(section):
                return "logical_blocks"  # 场景B
        
        return "v3_compatible"  # 场景C

    def _find_h3_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        查找文档中的H3标题段落
        
        Args:
            text: 要分析的文本
            
        Returns:
            H3段落列表，每个段落包含heading和content
        """
        lines = text.split('\n')
        h3_sections = []
        current_h3 = None
        current_content = []
        
        for line in lines:
            if line.startswith('### '):
                # 保存前一个H3段落
                if current_h3:
                    h3_sections.append({
                        'heading': current_h3,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # 开始新的H3段落
                current_h3 = line[4:].strip()
                current_content = []
            elif line.startswith('# ') or line.startswith('## '):
                # 遇到H1或H2标题时，结束当前H3段落
                if current_h3:
                    h3_sections.append({
                        'heading': current_h3,
                        'content': '\n'.join(current_content).strip()
                    })
                    current_h3 = None
                    current_content = []
            elif current_h3:
                current_content.append(line)
        
        # 保存最后一个H3段落
        if current_h3:
            h3_sections.append({
                'heading': current_h3,
                'content': '\n'.join(current_content).strip()
            })
        
        return h3_sections

    def _contains_boundary_keywords(self, section: Dict[str, Any]) -> bool:
        """
        检查H3段落内容是否包含boundary_keywords
        
        Args:
            section: H3段落信息
            
        Returns:
            是否包含boundary_keywords
        """
        content = section['content']
        boundary_keywords = LOGICAL_BLOCK_CONFIG["boundary_keywords"]
        
        # 检查是否包含任何boundary_keywords
        for keyword in boundary_keywords:
            # 支持两种格式：**关键词:** 和 **关键词 (Abilities):**
            if f"**{keyword}:**" in content or f"**{keyword} (" in content:
                return True
        
        return False

    def _process_with_logical_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        使用逻辑语义块分割处理文档
        
        Args:
            text: 要处理的文本
            
        Returns:
            chunks列表
        """
        # 使用现有的逻辑语义块分割逻辑
        self.document_structure = self.structure_extractor.extract_structure(text)
        self.structure_validator = StructureValidator(self.document_structure)
        
        sections = self._split_headings(text)
        all_chunks = []
        
        for section in sections:
            # 处理H1、H2、H3级别的sections
            if section['level'] in [1, 2, 3]:
                if section['level'] == 3:
                    # H3级别：使用LogicalBlockSplitter处理
                    sub_sections = self._split_logical_blocks_in_section(section)
                    for sub_section in sub_sections:
                        chunks = self._process_section_content(
                            sub_section['content'],
                            sub_section['heading'],
                            sub_section['level'],
                            sub_section['hierarchy'],
                            sub_section['sub_section'],  # 直接使用sub_section，不再需要content_type
                            {'sub_section': sub_section['sub_section']}
                        )
                        all_chunks.extend(chunks)
                else:
                    # H1、H2级别：直接处理，不经过LogicalBlockSplitter
                    if section['content'].strip():  # 只处理有内容的sections
                        # 将hierarchy_stack转换为hierarchy字典格式
                        hierarchy_dict = {}
                        for i in range(6):
                            hierarchy_dict[f'h{i+1}'] = section['hierarchy_stack'][i] if section['hierarchy_stack'][i] else ""
                        
                        chunks = self._process_section_content(
                            section['content'],
                            section['heading'],
                            section['level'],
                            hierarchy_dict,
                            'basic',  # 直接传递sub_section，不再需要content_type
                            {'sub_section': 'basic'}  # H1、H2使用basic作为sub_section
                        )
                        all_chunks.extend(chunks)
        
        return self._optimize_chunks(all_chunks)

    def _process_with_v3_compatible(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        使用V3兼容切片处理文档
        
        Args:
            text: 要处理的文本
            metadata: 额外的元数据（传递给V3 chunker）
            
        Returns:
            chunks列表
        """
        # 直接复用V3的完整逻辑，使用正确的参数
        v3_chunker = SemanticDocumentChunkerV3(
            encoding_name='cl100k_base',
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            min_semantic_units=2,
            document_name=self.document_name,
            faction_name=self.faction_name # 使用传入的faction_name
        )
        v3_chunks = v3_chunker.chunk_text(text, metadata)
        
        # 为V3的chunk添加sub_section='basic'
        return self._add_sub_section_to_v3_chunks(v3_chunks)

    def _add_sub_section_to_v3_chunks(self, v3_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为V3生成的chunk添加sub_section字段，并在text前面添加chunk_type
        
        Args:
            v3_chunks: V3生成的chunks
        
        Returns:
            添加了sub_section字段和chunk_type前缀的chunks
        """
        for chunk in v3_chunks:
            chunk['sub_section'] = 'basic'
            
            # 在text前面添加chunk_type，用"."隔开
            if 'chunk_type' in chunk and 'text' in chunk:
                chunk['text'] = f"{chunk['chunk_type']}.{chunk['text']}"
        
        return v3_chunks

    def _split_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        基于V3逻辑重构的标题分割方法
        
        核心改进：
        - 恢复V3的完整标题识别能力 (H1-H6)
        - 保持V3的层级栈管理逻辑
        - 保持现有H3处理逻辑完全不变
        - 新增H1、H2内容处理能力
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的sections列表
        """
        # 直接复用V3的成功逻辑
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
                
                # 保存前一个section（即使没有内容也要保存）
                if current_heading is not None:
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
                
                # 如果是H1标题，清空所有下级标题（包括H2、H3等）
                if level == 1:
                    for i in range(1, 6):
                        title_stack[i] = ""
                
                current_heading = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个section（即使没有内容也要保存）
        if current_heading is not None:
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

    def _split_logical_blocks_in_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        在section内进行逻辑语义块分割
        
        Args:
            section: 要分割的section
        
        Returns:
            逻辑语义块列表
        """
        content = section['content']
        if not content.strip():
            return []
        blocks = self.logical_block_splitter.split_logical_blocks(content)
        result = []
        for block in blocks:
            if block['content'].strip():
                hierarchy_dict = {}
                for i in range(6):
                    hierarchy_dict[f'h{i+1}'] = section['hierarchy_stack'][i] if section['hierarchy_stack'][i] else ""
                result.append({
                    'heading': section['heading'],
                    'level': section['level'],
                    'hierarchy': hierarchy_dict,
                    'sub_section': block['sub_section'],
                    'content': block['content']  # 只保留原始内容，不拼接chunk_type
                })
        return result


    def _process_section_content(self, content: str, section_heading: str,
                                section_level: int, hierarchy: Dict[str, str],
                                sub_section: str, additional_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理section内容，创建chunks
        
        Args:
            content: 要处理的内容
            section_heading: section标题
            section_level: section层级
            hierarchy: 层级信息
            sub_section: 子章节类型
            additional_metadata: 额外的metadata
            
        Returns:
            chunks列表
        """
        # 过滤空内容
        if not content.strip():
            return []
        
        # 创建chunk metadata
        chunk_metadata = self._create_chunk_metadata(
            content, section_heading, section_level, hierarchy,
            sub_section, additional_metadata
        )
        
        return [chunk_metadata]

    def _create_chunk_metadata(self, content: str, section_heading: str,
                              section_level: int, hierarchy: Dict[str, str],
                              sub_section: str, additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建chunk的metadata
        """
        # 构建chunk_type
        chunk_type = self._build_chunk_type(hierarchy)
        # 构建faction
        faction = self._build_faction()
        # text字段无条件拼接chunk_type和content，content左侧去空白
        text = f"{chunk_type}.{content.lstrip()}"
        return {
            'text': text,
            'section_heading': section_heading,
            'chunk_type': chunk_type,
            'content_type': 'basic',
            'faction': faction,
            'sub_section': sub_section,
            'h1': hierarchy.get('h1', ''),
            'h2': hierarchy.get('h2', ''),
            'h3': hierarchy.get('h3', ''),
            'h4': hierarchy.get('h4', ''),
            'h5': hierarchy.get('h5', ''),
            'h6': hierarchy.get('h6', '')
        }

    def _build_chunk_type(self, hierarchy: Dict[str, str]) -> str:
        """
        构建chunk_type - 直接拼接h1~h6的文本，过滤空值
        
        Args:
            hierarchy: 层级信息
            
        Returns:
            chunk_type字符串
        """
        parts = []
        for i in range(6):
            h_value = hierarchy.get(f'h{i+1}', '')
            if h_value.strip():  # 只添加非空的层级
                parts.append(h_value)
        
        return '-'.join(parts) if parts else 'unknown'

    def _build_faction(self) -> str:
        """
        构建faction
        
        Returns:
            faction字符串
        """
        # 如果传入了faction_name，优先使用
        if self.faction_name is not None:
            return self.faction_name
        
        # 否则返回"unknown"
        return "unknown"

    def _count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
            
        Returns:
            token数量
        """
        return len(self.tokenizer.encode(text))

    def _optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化chunks
        
        Args:
            chunks: 要优化的chunks
            
        Returns:
            优化后的chunks
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # 检查token数量
            token_count = self._count_tokens(chunk['text'])
            
            if token_count <= self.max_tokens:
                # 直接使用
                optimized_chunks.append(chunk)
            else:
                # 需要分割
                split_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(split_chunks)
        
        return optimized_chunks

    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分割过大的chunk
        
        Args:
            chunk: 要分割的chunk
            
        Returns:
            分割后的chunks列表
        """
        text = chunk['text']
        token_count = self._count_tokens(text)
        
        if token_count <= self.max_tokens:
            return [chunk]
        
        # 获取chunk_type，用于重新拼接text前缀
        chunk_type = chunk.get('chunk_type', '')
        
        # 提取原始content（去除chunk_type前缀）
        if chunk_type and text.startswith(chunk_type + '.'):
            original_content = text[len(chunk_type) + 1:]
        else:
            original_content = text
        
        # 按句子分割原始content
        sentences = original_content.split('。')
        split_chunks = []
        current_content = ""
        current_chunk = chunk.copy()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查添加这个句子是否会超过限制
            test_content = current_content + sentence + "。"
            test_text = f"{chunk_type}.{test_content}" if chunk_type else test_content
            test_token_count = self._count_tokens(test_text)
            
            if test_token_count <= self.max_tokens:
                current_content = test_content
            else:
                # 保存当前chunk
                if current_content:
                    final_text = f"{chunk_type}.{current_content.rstrip('。')}" if chunk_type else current_content.rstrip('。')
                    current_chunk['text'] = final_text
                    split_chunks.append(current_chunk)
                
                # 开始新的chunk
                current_content = sentence + "。"
                current_chunk = chunk.copy()
        
        # 保存最后一个chunk
        if current_content:
            final_text = f"{chunk_type}.{current_content.rstrip('。')}" if chunk_type else current_content.rstrip('。')
            current_chunk['text'] = final_text
            split_chunks.append(current_chunk)
        
        return split_chunks

    def save_to_json(self, chunks: List[Dict[str, Any]], filename: str = None) -> str:
        """
        保存chunks到JSON文件
    
    Args:
            chunks: 要保存的chunks
            filename: 文件名，如果为None则自动生成
        
    Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            filename = f"chunks/chunks_{self.document_name}_{timestamp}.json"
        
        # 确保chunks目录存在
        os.makedirs("chunks", exist_ok=True)
        
        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
        return filename


def main():
    """
    主函数，用于测试和演示
    """
    if len(sys.argv) != 2:
        print("用法: python3 data_vector_v4.py <测试文件路径>")
        print("示例: python3 data_vector_v4.py test/testdata/aeldaricodex.md")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 创建分块器
        chunker = AdSemanticDocumentChunker(document_name="aeldaricodex")

        # 进行分块
        print("开始逻辑语义块切片...")
        chunks = chunker.chunk_text(text)
        print(f"生成 {len(chunks)} 个切片")

        # 优化切片
        print("优化切片...")
        optimized_chunks = chunker._optimize_chunks(chunks)
        print(f"优化后 {len(optimized_chunks)} 个切片")

        # 统计sub_section分布
        sub_section_counts = {}
        for chunk in optimized_chunks:
            sub_section = chunk.get('sub_section', 'basic')
            sub_section_counts[sub_section] = sub_section_counts.get(sub_section, 0) + 1

        print("\n=== sub_section分布统计 ===")
        for sub_section, count in sorted(sub_section_counts.items()):
            print(f"  {sub_section}: {count}个")

        # 统计配置文件中的boundary_keywords
        print("\n=== 配置文件中的boundary_keywords ===")
        for keyword in LOGICAL_BLOCK_CONFIG["boundary_keywords"]:
            count = sub_section_counts.get(keyword, 0)
            print(f"  {keyword}: {count}个")

        # 保存结果
        output_file = chunker.save_to_json(optimized_chunks)
        print(f"\n结果已保存：")
        print(f"JSON文件：{output_file}")
        print("切片完成！")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
