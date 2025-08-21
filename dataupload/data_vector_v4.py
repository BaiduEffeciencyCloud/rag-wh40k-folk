"""
语义文档分块器 V4 - 逻辑语义块版本

核心特性：
- 从标题分块升级为逻辑语义块分块
- 智能识别关键词、技能、单位装备等边界
- 保持语义完整性和上下文连续性
- 结构化metadata设计（chunk_type, hierarchy, content_type, faction, sub_section）
"""

import re
import tiktoken
import json
import os
from typing import List, Dict, Any, Optional
from dataupload.config import LOGICAL_BLOCK_CONFIG


class DocumentStructureExtractor:
    """
    文档结构抽取器
    
    核心职责：
    - 识别文档的标题层级结构
    - 建立层级映射关系
    - 为后续分块提供结构信息
    """
    
    def __init__(self):
        self.hierarchy_stack = []
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """
        抽取文档结构
        
        Args:
            text: 文档文本
            
        Returns:
            文档结构信息
        """
        lines = text.split('\n')
        structure = {
            'hierarchy_map': {},
            'content_sections': {},
            'total_headings': 0,
            'max_depth': 0
        }
        
        current_section = None
        current_section_start = 0
        
        for line_num, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                
                # 更新层级信息 - 保持之前的层级信息，只更新当前层级
                hierarchy = current_section['hierarchy'].copy() if current_section else {}
                hierarchy[f'h{level}'] = heading
                
                # 清空当前层级之后的所有层级
                for i in range(level + 1, 7):
                    hierarchy[f'h{i}'] = ''
                
                current_section = {
                    'heading': heading,
                    'level': level,
                    'content': [],
                    'hierarchy': hierarchy
                }
            else:
                # 添加到当前section
                if current_section:
                    current_section['content'].append(line)
        
        # 保存最后一个section - 只有当有内容时才保存
        if current_section and current_section['content'] and current_section['content'] != ['']:
            content = '\n'.join(current_section['content']).strip()
            if content:  # 确保内容不为空
                structure['content_sections'][current_section['heading']] = {
                    'start_line': current_section_start,
                    'end_line': len(lines) - 1,
                    'title': current_section['heading']
                }
        
        return structure


class StructureValidator:
    """
    结构验证器
    
    核心职责：
    - 验证chunk的层级结构是否正确
    - 判断chunks是否可以合并
    - 提供正确的层级信息
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
        
        chunk_hierarchy = chunk.get('hierarchy', {})
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
        
        for line in lines:
            boundary_keyword = self._is_boundary_line(line)
            
            if boundary_keyword:
                # 保存前一个块
                if current_block['content_lines']:
                    blocks.append({
                        'sub_section': current_block['sub_section'],
                        'content': '\n'.join(current_block['content_lines']).strip()
                    })
                
                # 开始新的边界块，只包含边界行本身
                current_block = {
                    'sub_section': boundary_keyword,
                    'content_lines': [line]  # 只包含边界行
                }
                
                # 立即保存边界块
                blocks.append({
                    'sub_section': current_block['sub_section'],
                    'content': current_block['content_lines'][0]
                })
                
                # 重新开始basic块
                current_block = {'sub_section': 'basic', 'content_lines': []}
            else:
                # 添加到当前块
                current_block['content_lines'].append(line)
        
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
            # 精确匹配配置的关键词
            matched_keyword = self._fuzzy_match_keyword(keyword)
            if matched_keyword:
                return matched_keyword
        
        # 情况2: 匹配独立的标题行，格式：**关键词:**
        pattern2 = r'^\s*\*\*(.+?)(?:\s*\([^)]+\))?\:\*\*\s*$'
        match2 = re.match(pattern2, line)
        
        if match2:
            keyword = match2.group(1).strip()
            # 精确匹配配置的关键词
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
        # 只进行精确匹配，不要模糊匹配
        if keyword in self.boundary_keywords:
            return keyword
        
        return None


class SemanticDocumentChunker:
    """
    语义文档分块器 V4
    
    核心特性：
    - 专注于语义完整性和上下文保持
    - 强化语义连贯性评估
    - 智能chunk优化和分割
    - 结构化metadata设计（chunk_type, hierarchy, content_type, faction, sub_section）
    """
    
    def __init__(self, 
                 document_name: str = "",
                 max_tokens: int = 512,
                 overlap_tokens: int = 50,
                 content_type: str = "unit"):
        """
        初始化分块器
        
        Args:
            document_name: 文档名称
            max_tokens: 最大token数
            overlap_tokens: 重叠token数
            content_type: 内容类型
        """
        self.document_name = document_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.content_type = content_type
        
        # 初始化组件
        self.structure_extractor = DocumentStructureExtractor()
        self.logical_block_splitter = LogicalBlockSplitter()
        self.structure_validator = None
        
        # 层级栈
        self.hierarchy_stack = []
        
        # Token计数器
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本分割为语义chunks
        
        Args:
            text: 要分割的文本
            
        Returns:
            chunks列表，每个chunk包含text和metadata
        """
        # Step 1: 使用DocumentStructureExtractor预处理文档结构
        self.document_structure = self.structure_extractor.extract_structure(text)
        
        # Step 2: 初始化StructureValidator
        self.structure_validator = StructureValidator(self.document_structure)
        
        # Step 3: 按标题分割文本
        sections = self._split_headings(text)
        
        all_chunks = []
        
        for section in sections:
            # 在section内进行逻辑语义块分割
            sub_sections = self._split_logical_blocks_in_section(section)
            
            for sub_section in sub_sections:
                # 处理每个sub-section的内容
                chunks = self._process_section_content(
                    sub_section['content'],
                    sub_section['heading'],
                    sub_section['level'],
                    sub_section['hierarchy'],
                    self.content_type,
                    {'sub_section': sub_section['sub_section']}
                )
                all_chunks.extend(chunks)
        
        # Step 4: 优化chunks
        optimized_chunks = self.optimize_chunks(all_chunks)
        
        # 直接返回优化后的chunks，不进行扁平化
        # 因为后续的optimize_chunks调用需要原始的chunk结构
        return optimized_chunks
    
    def _split_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        按标题分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            sections列表
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                
                # 保存前一个section（只有当它有内容时才保存）
                if current_section and current_section['content']:
                    sections.append(current_section)
                
                # 开始新section
                current_section = {
                    'heading': heading,
                    'level': level,
                    'content': [],
                    'hierarchy': {f'h{level}': heading}
                }
            else:
                # 添加到当前section
                if current_section:
                    current_section['content'].append(line)
        
        # 保存最后一个section（只有当它有内容时才保存）
        if current_section and current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _split_logical_blocks_in_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        在section内进行逻辑语义块分割
        
        Args:
            section: 要分割的section
            
        Returns:
            分割后的sub-sections列表
        """
        # 只在 ### 层级进行逻辑语义块分割
        if section['level'] != 3:
            # 确保所有section都有sub_section字段，并将content转换为字符串
            section['sub_section'] = 'basic'
            # 将content从列表转换为字符串
            if isinstance(section['content'], list):
                section['content'] = '\n'.join(section['content'])
            return [section]
        
        # 使用LogicalBlockSplitter进行分割
        # 将content从列表转换为字符串
        content_text = '\n'.join(section['content']) if isinstance(section['content'], list) else str(section['content'])
        logical_blocks = self.logical_block_splitter.split_logical_blocks(content_text)
        
        sub_sections = []
        for block in logical_blocks:
            sub_sections.append({
                'heading': section['heading'],
                'level': section['level'],
                'content': block['content'],
                'hierarchy': section.get('hierarchy', {}),  # 使用get方法，提供默认值
                'sub_section': block['sub_section']
            })
        
        return sub_sections
    
    def _process_section_content(self, 
                                content: str, 
                                heading: Optional[str], 
                                level: int,
                                hierarchy: Dict[str, str],
                                content_type: str,
                                extra_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        处理section内容，应用token限制
        
        Args:
            content: section内容
            heading: 标题
            level: 标题层级
            hierarchy: 层级信息
            content_type: 内容类型
            extra_metadata: 额外元数据
            
        Returns:
            chunks列表
        """
        chunks = []
        
        # 如果content为空或只包含空白字符，不创建chunk
        if not content or not content.strip():
            return chunks
            
        token_count = self._count_tokens(content)
        
        # 如果内容小于等于max_tokens，直接作为一个chunk
        if token_count <= self.max_tokens:
            sentences = self._split_sentences(content)
            chunk = {
                'text': content,
                'metadata': self._create_chunk_metadata(
                    content, heading, level, len(sentences), hierarchy, content_type,
                    extra_metadata.get('sub_section') if extra_metadata else None,
                    extra_metadata
                )
            }
            chunks.append(chunk)
        # 如果内容大于max_tokens * 1.2，按句子分割
        elif token_count > self.max_tokens * 1.2:
            sentences = self._split_sentences(content)
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self._count_tokens(sentence)
                
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    # 保存当前chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk = {
                        'text': chunk_text,
                        'metadata': self._create_chunk_metadata(
                            chunk_text, heading, level, len(current_chunk), hierarchy, content_type,
                            extra_metadata.get('sub_section') if extra_metadata else None,
                            extra_metadata
                        )
                    }
                    chunks.append(chunk)
                    
                    # 开始新chunk，包含overlap
                    if self.overlap_tokens > 0:
                        overlap_text = self._get_last_n_tokens(chunk_text, self.overlap_tokens)
                        current_chunk = [overlap_text, sentence]
                        current_tokens = self._count_tokens(overlap_text) + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # 保存最后一个chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = {
                    'text': chunk_text,
                    'metadata': self._create_chunk_metadata(
                        chunk_text, heading, level, len(current_chunk), hierarchy, content_type,
                        extra_metadata.get('sub_section') if extra_metadata else None,
                        extra_metadata
                    )
                }
                chunks.append(chunk)
        # 如果内容在max_tokens和max_tokens * 1.2之间，保持为一个chunk
        else:
            sentences = self._split_sentences(content)
            chunk = {
                'text': content,
                'metadata': self._create_chunk_metadata(
                    content, heading, level, len(sentences), hierarchy, content_type,
                    extra_metadata.get('sub_section') if extra_metadata else None,
                    extra_metadata
                )
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_metadata(self, 
                              text: str, 
                              heading: Optional[str], 
                              level: int,
                              sentence_count: int,
                              hierarchy: Dict[str, str],
                              content_type: str,
                              sub_section: Optional[str] = None,
                              extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建chunk的精简元数据
        
        Args:
            text: chunk文本
            heading: 标题
            level: 标题层级
            sentence_count: 句子数量
            hierarchy: 层级信息
            content_type: 内容类型
            sub_section: 子部分（新增）
            extra_metadata: 额外元数据
            
        Returns:
            chunk元数据
        """
        # 修复1: sub_section名称要加到text最前面
        if sub_section and sub_section != 'basic':
            formatted_text = f"{sub_section}:{text}"
        else:
            formatted_text = text
        
        metadata = {
            'text': formatted_text,  # 修复1: 使用格式化后的text
            'section_heading': self._clean_markdown_heading(heading) if heading else None,
            'chunk_type': self._build_chunk_type(hierarchy, content_type),
            'content_type': content_type,
            'faction': self._build_faction(),
            'sub_section': sub_section if sub_section else 'basic',  # 确保sub_section不为null
            # 修复2: 删除不需要的字段 (sentence_count, token_count, created_at)
        }
        
        # 修复3: 保持完整的h1~h6层级结构
        for i in range(1, 7):
            key = f'h{i}'
            metadata[key] = hierarchy.get(key, '')  # 即使没有内容也要保留key
        
        # 添加额外元数据
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return metadata
    
    def optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化chunks，合并过小的chunks，分割过大的chunks
        
        Args:
            chunks: 原始chunks列表
            
        Returns:
            优化后的chunks列表
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # 计算token_count用于判断是否需要分割
            token_count = self._count_tokens(chunk['text'])
            
            # 如果chunk过大，进行分割
            if token_count > self.max_tokens * 1.5:
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分割过大的chunk，支持overlap保持上下文连续性
        """
        text = chunk['text']
        metadata = chunk['metadata']
        sentences = self._split_sentences(text)
        
        sub_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # 保存当前chunk
                chunk_text = ' '.join(current_chunk)
                sub_chunk = {
                    'text': chunk_text,
                    'metadata': metadata.copy()
                }
                sub_chunk['metadata']['text'] = chunk_text
                sub_chunks.append(sub_chunk)
                
                # 开始新chunk，包含overlap
                if self.overlap_tokens > 0:
                    overlap_text = self._get_last_n_tokens(chunk_text, self.overlap_tokens)
                    current_chunk = [overlap_text, sentence]
                    current_tokens = self._count_tokens(overlap_text) + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # 保存最后一个chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            sub_chunk = {
                'text': chunk_text,
                'metadata': metadata.copy()
            }
            sub_chunk['metadata']['text'] = chunk_text
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        # 简单的句子分割，按句号、问号、感叹号分割
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_last_n_tokens(self, text: str, n: int) -> str:
        """获取文本的最后n个token"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= n:
            return text
        
        # 取最后n个token
        last_tokens = tokens[-n:]
        return self.tokenizer.decode(last_tokens)
    
    def _clean_markdown_heading(self, heading: str) -> str:
        """清理markdown标题格式"""
        return re.sub(r'^#+\s*', '', heading).strip()
    
    def _build_chunk_type(self, hierarchy: Dict[str, str], content_type: str) -> str:
        """构建chunk类型"""
        h3 = hierarchy.get('h3', '')
        if h3:
            return f"{h3}_{content_type}"
        return content_type
    
    def _build_faction(self) -> str:
        """构建阵营信息"""
        return "aeldari"  # 默认阵营


def main(test_file: str):
    """
    主函数，用于测试V4逻辑语义块切片器
    """
    print("\n==================== 语义文档切片器 V4 (逻辑语义块版本) ====================")
    print("                                                 ")
    print("核心特性：")
    print("- 从标题分块升级为逻辑语义块分块")
    print("- 智能识别关键词、技能、单位装备等边界")
    print("- 保持语义完整性和上下文连续性")
    print("- 结构化metadata设计（chunk_type, hierarchy, content_type, faction, sub_section）")
    print("                                                ")
    
    print("【命令行用法】")
    print("python3 dataupload/data_vector_v4.py <测试文件路径>")
    print("")
    print("示例：")
    print("python3 dataupload/data_vector_v4.py test/testdata/aeldaricodex.md")
    print("")
    
    # 读取测试文件
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\n正在处理文件：{test_file}")
    print(f"文件大小：{len(text)} 字符")
    
    # 创建分块器
    chunker = SemanticDocumentChunker(document_name=test_file)
    
    print("\n开始逻辑语义块切片...")
    
    # 执行分块
    chunks = chunker.chunk_text(text)
    
    print(f"生成 {len(chunks)} 个切片")
    
    # 优化切片
    print("优化切片...")
    optimized_chunks = chunker.optimize_chunks(chunks)
    print(f"优化后 {len(optimized_chunks)} 个切片")
    
    # 统计sub_section分布
    print("\n=== sub_section分布统计 ===")
    sub_section_counts = {}
    for chunk in optimized_chunks:
        sub_section = chunk.get('metadata', {}).get('sub_section', '')
        if sub_section not in sub_section_counts:
            sub_section_counts[sub_section] = 0
        sub_section_counts[sub_section] += 1
    
    # 显示每个sub_section的数量
    for sub_section, count in sub_section_counts.items():
        print(f"  {sub_section}: {count}个")
    
    # 显示配置文件中的boundary_keywords
    print(f"\n=== 配置文件中的boundary_keywords ===")
    from dataupload.config import LOGICAL_BLOCK_CONFIG
    boundary_keywords = LOGICAL_BLOCK_CONFIG.get('boundary_keywords', [])
    for keyword in boundary_keywords:
        count = sub_section_counts.get(keyword, 0)
        print(f"  {keyword}: {count}个")
    
    # 保存结果
    import os
    from datetime import datetime
    
    # 创建chunks目录
    os.makedirs('chunks', exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    filename = f"chunks/chunks_aeldaricodex_{timestamp}.json"
    
    # 保存为JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(optimized_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存：")
    print(f"JSON文件：{filename}")
    print("切片完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python3 data_vector_v4.py <测试文件路径>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    
    if not os.path.exists(test_file):
        print(f"错误：文件 {test_file} 不存在")
        sys.exit(1)
    
    main(test_file)
