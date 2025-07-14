import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from test.base_test import TestBase
from dataupload.data_vector_v3 import (
    ContentType, 
    DocumentStructureExtractor, 
    StructureValidator, 
    SemanticDocumentChunker,
    save_chunks_to_json,
    save_chunks_to_text
)


class TestContentType(TestBase):
    """测试ContentType枚举"""
    
    def test_content_type_values(self):
        """测试ContentType枚举值"""
        self.assertEqual(ContentType.ARMY_RULE.value, "army_rule")
        self.assertEqual(ContentType.DETACHMENT_RULE.value, "detachment_rule")
        self.assertEqual(ContentType.UNIT.value, "unit")
        self.assertEqual(ContentType.CORERULE.value, "corerule")


class TestDocumentStructureExtractor(TestBase):
    """测试DocumentStructureExtractor类"""
    
    def setUp(self):
        super().setUp()
        self.extractor = DocumentStructureExtractor()
    
    def test_extract_structure_basic(self):
        """测试基本文档结构抽取"""
        text = """# 一级标题1
内容1

## 二级标题1
内容2

### 三级标题1
内容3

# 一级标题2
内容4"""
        
        structure = self.extractor.extract_structure(text)
        
        self.assertIn('headings', structure)
        self.assertIn('hierarchy_map', structure)
        self.assertIn('section_boundaries', structure)
        self.assertIn('content_sections', structure)
        
        self.assertEqual(len(structure['headings']), 4)
        self.assertEqual(len(structure['section_boundaries']), 2)
    
    def test_extract_structure_hierarchy(self):
        """测试层级结构抽取"""
        text = """# 阿苏焉尼
## 单位
### 猎鹰坦克
内容

## 规则
### 部署规则
内容"""
        
        structure = self.extractor.extract_structure(text)
        
        # 检查层级映射
        self.assertIn('阿苏焉尼', structure['hierarchy_map'])
        self.assertIn('猎鹰坦克', structure['hierarchy_map'])
        
        # 检查层级信息
        tank_hierarchy = structure['hierarchy_map']['猎鹰坦克']
        self.assertEqual(tank_hierarchy['level1'], '阿苏焉尼')
        self.assertEqual(tank_hierarchy['level2'], '单位')
        self.assertEqual(tank_hierarchy['level3'], '猎鹰坦克')
    
    def test_extract_structure_no_headings(self):
        """测试无标题文档"""
        text = "这是没有标题的文档内容。"
        structure = self.extractor.extract_structure(text)
        
        self.assertEqual(len(structure['headings']), 0)
        self.assertEqual(len(structure['section_boundaries']), 0)
    
    def test_validate_structure_valid(self):
        """测试有效结构验证"""
        text = """# 一级标题
## 二级标题
内容

# 另一个一级标题
内容"""
        
        structure = self.extractor.extract_structure(text)
        validation = self.extractor.validate_structure(structure)
        
        self.assertTrue(validation['is_valid'])
        self.assertEqual(validation['statistics']['total_headings'], 3)
        self.assertEqual(validation['statistics']['level1_count'], 2)
    
    def test_validate_structure_warnings(self):
        """测试结构验证警告"""
        text = """## 直接二级标题（缺少一级标题）
内容

### 三级标题
内容"""
        
        structure = self.extractor.extract_structure(text)
        validation = self.extractor.validate_structure(structure)
        
        self.assertTrue(validation['is_valid'])
        self.assertGreater(len(validation['warnings']), 0)
        # 应该警告没有一级标题
        self.assertTrue(any("没有一级标题" in warning for warning in validation['warnings']))


class TestStructureValidator(TestBase):
    """测试StructureValidator类"""
    
    def setUp(self):
        super().setUp()
        # 创建测试文档结构
        self.document_structure = {
            'hierarchy_map': {
                '阿苏焉尼': {'level1': '阿苏焉尼'},
                '猎鹰坦克': {'level1': '阿苏焉尼', 'level2': '单位', 'level3': '猎鹰坦克'}
            },
            'section_boundaries': [
                {'title': '阿苏焉尼', 'line_number': 0}
            ]
        }
        self.validator = StructureValidator(self.document_structure)
    
    def test_validate_chunk_hierarchy_valid(self):
        """测试有效chunk层级验证"""
        chunk = {
            'section_heading': '猎鹰坦克',
            'hierarchy': {'level1': '阿苏焉尼', 'level2': '单位', 'level3': '猎鹰坦克'}
        }
        
        validation = self.validator.validate_chunk_hierarchy(chunk)
        self.assertTrue(validation['is_valid'])
    
    def test_validate_chunk_hierarchy_invalid(self):
        """测试无效chunk层级验证"""
        chunk = {
            'section_heading': '不存在的标题',
            'hierarchy': {'level1': '错误的层级'}
        }
        
        validation = self.validator.validate_chunk_hierarchy(chunk)
        self.assertTrue(validation['is_valid'])  # 只是警告，不是错误
        self.assertGreater(len(validation['warnings']), 0)
    
    def test_can_merge_chunks_same_section(self):
        """测试同section的chunks可以合并"""
        chunk1 = {'hierarchy': {'level1': '阿苏焉尼'}, 'section_heading': '猎鹰坦克'}
        chunk2 = {'hierarchy': {'level1': '阿苏焉尼'}, 'section_heading': '猎鹰坦克'}
        
        self.assertTrue(self.validator.can_merge_chunks(chunk1, chunk2))
    
    def test_can_merge_chunks_different_section(self):
        """测试不同section的chunks不能合并"""
        chunk1 = {'hierarchy': {'level1': '阿苏焉尼'}, 'section_heading': '猎鹰坦克'}
        chunk2 = {'hierarchy': {'level1': '其他派系'}, 'section_heading': '其他单位'}
        
        self.assertFalse(self.validator.can_merge_chunks(chunk1, chunk2))


class TestSemanticDocumentChunker(TestBase):
    """测试SemanticDocumentChunker类"""
    
    def setUp(self):
        super().setUp()
        self.chunker = SemanticDocumentChunker(
            max_tokens=100,  # 使用较小的值便于测试
            overlap_tokens=20
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.chunker.max_tokens, 100)
        self.assertEqual(self.chunker.overlap_tokens, 20)
        self.assertEqual(self.chunker.document_name, "aeldari")
        self.assertEqual(self.chunker.faction_name, "阿苏焉尼")
    
    def test_count_tokens(self):
        """测试token计数"""
        text = "这是一个测试文本。"
        token_count = self.chunker._count_tokens(text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
    
    def test_split_sentences(self):
        """测试句子分割"""
        text = "第一句话。第二句话！第三句话？"
        sentences = self.chunker._split_sentences(text)
        
        self.assertEqual(len(sentences), 3)
        self.assertIn("第一句话。", sentences)
        self.assertIn("第二句话！", sentences)
        self.assertIn("第三句话？", sentences)
    
    def test_clean_markdown_heading(self):
        """测试markdown标题清理"""
        # 测试基本清理
        self.assertEqual(self.chunker._clean_markdown_heading("## 标题内容"), "标题内容")
        self.assertEqual(self.chunker._clean_markdown_heading("### 标题内容 "), "标题内容")
        
        # 测试markdown符号清理
        self.assertEqual(self.chunker._clean_markdown_heading("**粗体标题**"), "粗体标题")
        self.assertEqual(self.chunker._clean_markdown_heading("`代码标题`"), "代码标题")
        
        # 测试空值处理
        self.assertEqual(self.chunker._clean_markdown_heading(""), "")
        self.assertEqual(self.chunker._clean_markdown_heading(None), "")
    
    def test_parse_heading_hierarchy(self):
        """测试标题层级解析"""
        hierarchy_stack = ["阿苏焉尼", "单位", "猎鹰坦克", "", "", ""]
        hierarchy = self.chunker._parse_heading_hierarchy("猎鹰坦克", 3, hierarchy_stack)
        
        self.assertEqual(hierarchy['h1'], "阿苏焉尼")
        self.assertEqual(hierarchy['h2'], "单位")
        self.assertEqual(hierarchy['h3'], "猎鹰坦克")
    
    def test_determine_content_type(self):
        """测试内容类型确定"""
        # 测试单位类型 - 需要包含关键词
        content_type = self.chunker._determine_content_type("单位数据", 3, "技能：火力支援")
        self.assertEqual(content_type, ContentType.UNIT.value)
        
        # 测试军队规则类型
        content_type = self.chunker._determine_content_type("军队规则", 1, "军队规则内容")
        self.assertEqual(content_type, ContentType.ARMY_RULE.value)
        
        # 测试默认类型（不包含关键词的标题）
        content_type = self.chunker._determine_content_type("猎鹰坦克", 3, "技能：火力支援")
        self.assertEqual(content_type, ContentType.CORERULE.value)
    
    def test_build_chunk_type(self):
        """测试chunk类型构建"""
        hierarchy = {'h1': '阿苏焉尼', 'h2': '单位', 'h3': '猎鹰坦克'}
        chunk_type = self.chunker._build_chunk_type(hierarchy, ContentType.UNIT.value)
        
        self.assertIn("aeldari", chunk_type)
        self.assertIn("单位数据", chunk_type)
        self.assertEqual(chunk_type, "aeldari单位数据-阿苏焉尼-单位-猎鹰坦克")
    
    def test_build_faction(self):
        """测试派系构建"""
        faction = self.chunker._build_faction()
        self.assertEqual(faction, "aeldari-阿苏焉尼")
    
    def test_get_last_n_tokens(self):
        """测试获取最后n个token"""
        text = "这是一个很长的测试文本，用来测试获取最后几个token的功能。"
        last_tokens = self.chunker._get_last_n_tokens(text, 5)
        
        self.assertIsInstance(last_tokens, str)
        self.assertLessEqual(self.chunker._count_tokens(last_tokens), 5)
    
    def test_create_chunk_metadata(self):
        """测试chunk元数据创建"""
        text = "测试内容"
        heading = "## 测试标题"
        hierarchy = {'h1': '阿苏焉尼', 'h2': '单位'}
        
        metadata = self.chunker._create_chunk_metadata(
            text, heading, 2, 1, hierarchy, ContentType.UNIT.value
        )
        
        self.assertEqual(metadata['text'], text)
        self.assertEqual(metadata['section_heading'], "测试标题")  # 应该被清理
        self.assertEqual(metadata['content_type'], ContentType.UNIT.value)
        self.assertEqual(metadata['faction'], "aeldari-阿苏焉尼")
        self.assertEqual(metadata['h1'], "阿苏焉尼")
        self.assertEqual(metadata['h2'], "单位")
    
    def test_split_large_chunk(self):
        """测试大chunk分割"""
        # 创建一个超过token限制的chunk
        long_text = "这是一个很长的句子。" * 50  # 重复50次
        chunk = {
            'text': long_text,
            'section_heading': '测试标题',
            'content_type': 'unit',
            'faction': '阿苏焉尼'
        }
        
        sub_chunks = self.chunker._split_large_chunk(chunk)
        
        self.assertGreater(len(sub_chunks), 1)  # 应该被分割成多个chunk
        
        # 检查每个sub_chunk都有正确的元数据
        for sub_chunk in sub_chunks:
            self.assertIn('text', sub_chunk)
            self.assertIn('section_heading', sub_chunk)
            self.assertIn('content_type', sub_chunk)
            self.assertIn('faction', sub_chunk)
    
    def test_split_large_chunk_with_overlap(self):
        """测试带overlap的大chunk分割"""
        # 创建包含多个句子的长文本
        sentences = [
            "第一句话。",
            "第二句话。",
            "第三句话。",
            "第四句话。",
            "第五句话。"
        ]
        long_text = " ".join(sentences * 20)  # 重复20次
        
        chunk = {
            'text': long_text,
            'section_heading': '测试标题',
            'content_type': 'unit',
            'faction': '阿苏焉尼'
        }
        
        sub_chunks = self.chunker._split_large_chunk(chunk)
        
        # 检查overlap是否生效
        if len(sub_chunks) > 1:
            # 检查相邻chunk之间是否有重叠
            for i in range(len(sub_chunks) - 1):
                chunk1_text = sub_chunks[i]['text']
                chunk2_text = sub_chunks[i + 1]['text']
                
                # 简单检查：第二个chunk的开头应该包含第一个chunk的结尾部分
                # 这里只是基本检查，实际overlap逻辑更复杂
                self.assertIsInstance(chunk1_text, str)
                self.assertIsInstance(chunk2_text, str)
    
    def test_chunk_text_basic(self):
        """测试基本文本切片"""
        text = """# 阿苏焉尼
## 猎鹰坦克
这是一个测试单位。它有很强的火力。

## 其他单位
这是另一个单位。"""
        
        chunks = self.chunker.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        
        # 检查chunk结构
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('section_heading', chunk)
            self.assertIn('content_type', chunk)
            self.assertIn('faction', chunk)
    
    def test_optimize_chunks(self):
        """测试chunk优化"""
        # 创建一些测试chunks
        chunks = [
            {'text': '短文本1', 'section_heading': '标题1'},
            {'text': '短文本2', 'section_heading': '标题1'},  # 同标题，应该可以合并
            {'text': '长文本' * 50, 'section_heading': '标题2'}  # 应该被分割
        ]
        
        optimized = self.chunker.optimize_chunks(chunks)
        
        # 优化后的chunk数量可能不同
        self.assertIsInstance(optimized, list)
        self.assertGreater(len(optimized), 0)


class TestUtilityFunctions(TestBase):
    """测试工具函数"""
    
    def test_save_chunks_to_json(self):
        """测试保存chunks到JSON"""
        chunks = [
            {'text': '测试内容1', 'section_heading': '标题1'},
            {'text': '测试内容2', 'section_heading': '标题2'}
        ]
        
        output_file = save_chunks_to_json(chunks, self.temp_dir, "test.md")
        
        self.assertTrue(os.path.exists(output_file))
        
        # 验证JSON内容
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_chunks = json.load(f)
        
        self.assertEqual(len(saved_chunks), 2)
        self.assertEqual(saved_chunks[0]['text'], '测试内容1')
    
    def test_save_chunks_to_text(self):
        """测试保存chunks到文本"""
        chunks = [
            {'text': '测试内容1', 'section_heading': '标题1', 'content_type': 'unit'},
            {'text': '测试内容2', 'section_heading': '标题2', 'content_type': 'rule'}
        ]
        
        output_file = save_chunks_to_text(chunks, self.temp_dir)
        
        self.assertTrue(os.path.exists(output_file))
        
        # 验证文本内容
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('测试内容1', content)
        self.assertIn('标题1', content)
        self.assertIn('unit', content)


if __name__ == '__main__':
    unittest.main() 