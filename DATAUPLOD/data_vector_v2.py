import re
import os
import tiktoken
import json
from datetime import datetime


class DocumentChunker:
    """
    文档切片处理类，基于token划分，并支持特例/覆盖规则合并、chunk_type标签和额外metadata（如section_heading、entity_name、faction、aliases）。
    """
    def __init__(self,
                 encoding_name: str = 'cl100k_base',
                 max_tokens: int = 512,
                 overlap_tokens: int = 50,
                 special_patterns: dict = None):
        # 初始化分词器
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # 定义特例模式及对应chunk_type
        self.special_patterns = special_patterns or {}

    def _count_tokens(self, text: str) -> int:
        """返回文本的token数。"""
        return len(self.encoder.encode(text))

    def _split_sentences(self, text: str) -> list:
        """按句号、问号、感叹号等断句，保留标点。"""
        pattern = r'(?<=[。！？?!])\s+'
        return re.split(pattern, text)

    def _split_headings(self, text: str) -> list:
        """按Markdown一级/二级/三级标题拆分，保留标题与内容。返回[{heading, content}]列表。"""
        # 匹配 1 到 3 级标题
        parts = re.split(r'(#{1,3}\s.*)', text)
        sections = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if re.match(r'^#{1,3}\s', part):
                heading = part.strip()
                content = parts[i+1] if i+1 < len(parts) else ''
                sections.append({'heading': heading, 'content': content.strip()})
                i += 2
            else:
                if part.strip():
                    sections.append({'heading': None, 'content': part.strip()})
                i += 1
        return sections

    def _last_n_tokens(self, text: str, n: int) -> str:
        """返回文本的最后n个token对应的字符串（用于overlap）。"""
        tokens = self.encoder.encode(text)
        last_tokens = tokens[-n:]
        return self.encoder.decode(last_tokens)

    def chunk_text(self, text: str, extra_metadata: dict = None) -> list:
        """主方法：分章节、断句后基于token进行滑动窗口切片，添加chunk_type及其他metadata。"""
        sections = self._split_headings(text)
        raw_chunks = []

        for sec in sections:
            heading = sec['heading']
            sentences = self._split_sentences(sec['content'])
            buffer = ''
            for sentence in sentences:
                candidate = (buffer + ' ' + sentence).strip()
                if self._count_tokens(candidate) <= self.max_tokens:
                    buffer = candidate
                else:
                    raw_chunks.append((heading, buffer))
                    overlap = self._last_n_tokens(buffer, self.overlap_tokens)
                    buffer = (overlap + ' ' + sentence).strip()
            if buffer:
                raw_chunks.append((heading, buffer))

        enriched = []
        for heading, chunk in raw_chunks:
            ctype = 'general'
            for pattern, tag in self.special_patterns.items():
                if re.search(pattern, chunk):
                    ctype = tag
                    break
            metadata = {
                'text': chunk,
                'chunk_type': ctype,
                'token_count': self._count_tokens(chunk),
                'section_heading': heading
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            enriched.append(metadata)

        return enriched

# 示例：chunker = DocumentChunker(special_patterns=special)
# chunks = chunker.chunk_text(markdown_text, extra_metadata)
# 支持 1~3 级标题的切分

def save_chunks_to_json(chunks, output_dir="chunks"):
    """保存切片结果到JSON文件"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"chunks_{timestamp}.json")
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return output_file

def save_chunks_to_text(chunks, output_dir="chunks"):
    """保存切片结果到文本文件，每个切片以分隔符隔开"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"chunks_{timestamp}.txt")
    
    # 保存到文本文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"总块数: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"-----{i}-----\n")
            f.write(f"类型: {chunk['chunk_type']}\n")
            f.write(f"Token数: {chunk['token_count']}\n")
            f.write(f"内容:\n{chunk['text']}\n\n")
    
    return output_file

def main(test_file: str):
    # 用户指导
    print("""
==================== 使用说明 ====================
本脚本用于测试和演示 DocumentChunker 的 Markdown 文本切片功能。

【命令行用法】
python DATAUPLOD/data_vector_v2.py <测试文件路径>

示例：
python DATAUPLOD/data_vector_v2.py ../test/testdata/40kcorerule.md

【DocumentChunker 用法简介】
- 支持基于token的滑动窗口切片，自动处理1~3级Markdown标题。
- 支持自定义特殊模式（如武器、能力、单位等规则标签）。
- 可通过extra_metadata参数为每个chunk附加自定义元数据。
- 切片结果可保存为JSON或TXT，便于后续分析和调试。

==================================================
""")
    
    # 确保文件存在
    if not os.path.exists(test_file):
        print(f"错误：找不到测试文件 {test_file}")
        return
    
    # 读取测试文档
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件时出错：{str(e)}")
        return
    
    # 定义特殊模式
    special_patterns = {
        '武器': 'weapon_rule',
        '能力': 'ability_rule',
        '单位': 'unit_rule',
        '装备': 'equipment_rule',
        '特性': 'trait_rule'
    }
    
    # 创建切片器实例
    chunker = DocumentChunker(
        max_tokens=512,
        overlap_tokens=60,
        special_patterns=special_patterns
    )
    
    # 执行切片
    print("开始处理文档...")
    chunks = chunker.chunk_text(content)
    
    # 打印统计信息
    print(f"\n处理完成！共生成 {len(chunks)} 个切片")
    print("\n切片类型统计：")
    type_stats = {}
    for chunk in chunks:
        ctype = chunk['chunk_type']
        type_stats[ctype] = type_stats.get(ctype, 0) + 1
    
    for ctype, count in type_stats.items():
        print(f"- {ctype}: {count} 个切片")
    
    # 保存结果
    json_file = save_chunks_to_json(chunks)
    txt_file = save_chunks_to_text(chunks)
    print(f"\n切片结果已保存到：")
    print(f"- JSON文件：{json_file}")
    print(f"- 文本文件：{txt_file}")
    
    # 打印前3个切片作为示例
    print("\n前3个切片示例：")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- 切片 {i} ---")
        print(f"类型: {chunk['chunk_type']}")
        print(f"Token数: {chunk['token_count']}")
        print(f"内容预览: {chunk['text'][:100]}...")

if __name__ == "__main__":
    import sys
    
    # 如果命令行提供了文件路径，则使用提供的路径
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "../test/testdata/40kcorerule.md"
    
    main(test_file) 