#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片逻辑分析测试脚本
用于分析"死亡之舞"等关键内容是否被正确切片
"""

import os
import sys
import re
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataupload.data_vector_v3 import SemanticDocumentChunker, save_chunks_to_text, save_chunks_to_json

def find_skill_content_in_text(text: str, skill_name: str) -> List[Dict[str, Any]]:
    """
    在文本中查找指定技能的内容
    
    Args:
        text: 原始文本
        skill_name: 技能名称
        
    Returns:
        包含技能内容的段落列表
    """
    lines = text.split('\n')
    skill_paragraphs = []
    current_paragraph = []
    in_skill_section = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # 检查是否包含技能名称
        if skill_name in line_stripped:
            in_skill_section = True
            current_paragraph = [line_stripped]
        elif in_skill_section:
            # 如果遇到空行或新标题，结束当前段落
            if not line_stripped or line_stripped.startswith('#'):
                if current_paragraph:
                    skill_paragraphs.append({
                        'start_line': i - len(current_paragraph) + 1,
                        'end_line': i,
                        'content': '\n'.join(current_paragraph),
                        'lines': current_paragraph
                    })
                    current_paragraph = []
                    in_skill_section = False
            else:
                current_paragraph.append(line_stripped)
    
    # 处理最后一个段落
    if current_paragraph:
        skill_paragraphs.append({
            'start_line': len(lines) - len(current_paragraph),
            'end_line': len(lines) - 1,
            'content': '\n'.join(current_paragraph),
            'lines': current_paragraph
        })
    
    return skill_paragraphs

def analyze_chunk_coverage(chunks: List[Dict[str, Any]], skill_name: str) -> Dict[str, Any]:
    """
    分析切片对指定技能的覆盖情况
    
    Args:
        chunks: 切片列表
        skill_name: 技能名称
        
    Returns:
        覆盖分析结果
    """
    coverage_analysis = {
        'skill_name': skill_name,
        'total_chunks': len(chunks),
        'containing_chunks': [],
        'missing_chunks': [],
        'coverage_rate': 0.0
    }
    
    # 检查每个切片是否包含技能名称
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        if skill_name in chunk_text:
            coverage_analysis['containing_chunks'].append({
                'chunk_index': i,
                'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                'section_heading': chunk.get('section_heading', ''),
                'content_preview': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                'full_content': chunk_text
            })
    
    # 计算覆盖率
    if coverage_analysis['containing_chunks']:
        coverage_analysis['coverage_rate'] = len(coverage_analysis['containing_chunks']) / len(chunks) * 100
    
    return coverage_analysis

def test_chunking_logic(file_path: str, skill_name: str = "死亡之舞"):
    """
    测试切片逻辑
    
    Args:
        file_path: 文档文件路径
        skill_name: 要检查的技能名称
    """
    print(f"🔍 开始分析文件: {file_path}")
    print(f"🎯 目标技能: {skill_name}")
    
    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ 文件读取成功，大小: {len(content)} 字符")
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return
    
    # 1. 在原始文本中查找技能内容
    print(f"\n📖 步骤1: 在原始文本中查找 '{skill_name}' 内容...")
    skill_paragraphs = find_skill_content_in_text(content, skill_name)
    
    if skill_paragraphs:
        print(f"✅ 在原始文本中找到 {len(skill_paragraphs)} 个包含 '{skill_name}' 的段落:")
        for i, para in enumerate(skill_paragraphs, 1):
            print(f"  段落 {i}:")
            print(f"    行号: {para['start_line']}-{para['end_line']}")
            print(f"    内容预览: {para['content'][:100]}...")
    else:
        print(f"❌ 在原始文本中未找到 '{skill_name}' 相关内容")
        return
    
    # 2. 使用切片器处理文档
    print(f"\n🔧 步骤2: 使用切片器处理文档...")
    try:
        chunker = SemanticDocumentChunker()
        chunks = chunker.chunk_text(content, {"source_file": os.path.basename(file_path)})
        print(f"✅ 切片完成，生成 {len(chunks)} 个切片")
        
        # 优化切片
        optimized_chunks = chunker.optimize_chunks(chunks)
        print(f"✅ 优化完成，优化后 {len(optimized_chunks)} 个切片")
        
    except Exception as e:
        print(f"❌ 切片处理失败: {e}")
        return
    
    # 3. 分析切片覆盖情况
    print(f"\n📊 步骤3: 分析切片对 '{skill_name}' 的覆盖情况...")
    coverage_analysis = analyze_chunk_coverage(optimized_chunks, skill_name)
    
    print(f"📈 覆盖分析结果:")
    print(f"  总切片数: {coverage_analysis['total_chunks']}")
    print(f"  包含技能的切片数: {len(coverage_analysis['containing_chunks'])}")
    print(f"  覆盖率: {coverage_analysis['coverage_rate']:.2f}%")
    
    if coverage_analysis['containing_chunks']:
        print(f"\n📋 包含 '{skill_name}' 的切片详情:")
        for i, chunk_info in enumerate(coverage_analysis['containing_chunks'], 1):
            print(f"  切片 {i}:")
            print(f"    索引: {chunk_info['chunk_index']}")
            print(f"    标题: {chunk_info['section_heading']}")
            print(f"    内容预览: {chunk_info['content_preview']}")
            print()
    else:
        print(f"❌ 切片中未找到 '{skill_name}' 相关内容")
        
        # 4. 深入分析为什么没有找到
        print(f"\n🔍 步骤4: 深入分析切片内容...")
        print("检查前10个切片的内容:")
        for i, chunk in enumerate(optimized_chunks[:10], 1):
            chunk_text = chunk.get('text', '')
            print(f"  切片 {i}:")
            print(f"    标题: {chunk.get('section_heading', '无标题')}")
            print(f"    内容长度: {len(chunk_text)} 字符")
            print(f"    内容预览: {chunk_text[:100]}...")
            print()
    
    # 5. 保存分析结果
    print(f"\n💾 步骤5: 保存分析结果...")
    try:
        # 保存切片结果
        timestamp = os.path.basename(file_path).replace('.md', '')
        save_chunks_to_text(optimized_chunks, f"test/testdata/{timestamp}_chunks.txt")
        save_chunks_to_json(optimized_chunks, f"test/testdata/{timestamp}_chunks.json")
        
        # 保存覆盖分析结果
        import json
        analysis_file = f"test/testdata/{timestamp}_coverage_analysis.json"
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(coverage_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析结果已保存到 test/testdata/ 目录")
        
    except Exception as e:
        print(f"❌ 保存分析结果失败: {e}")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_chunk_analysis.py <文档文件路径> [技能名称]")
        print("示例: python test_chunk_analysis.py DATAUPLOD/aeldari.md 死亡之舞")
        return
    
    file_path = sys.argv[1]
    skill_name = sys.argv[2] if len(sys.argv) > 2 else "死亡之舞"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 运行分析
    test_chunking_logic(file_path, skill_name)

if __name__ == "__main__":
    main() 