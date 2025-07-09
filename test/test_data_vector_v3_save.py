import os
import re
import json
import tempfile
import shutil
from datetime import datetime
from unittest import TestCase
from dataupload.data_vector_v3 import save_chunks_to_json

class TestSaveChunksToJson(TestCase):
    def setUp(self):
        self.chunks = [
            {"text": "test1", "meta": 1},
            {"text": "test2", "meta": 2}
        ]
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_save_with_docname_and_timestamp(self):
        doc_name = "mydoc"
        # 调用重构后的接口，新增source_file参数
        output_file = save_chunks_to_json(self.chunks, output_dir=self.tmp_dir, source_file=doc_name+".md")
        # 检查文件是否存在
        self.assertTrue(os.path.exists(output_file))
        # 检查文件名格式
        basename = os.path.basename(output_file)
        # 期望格式: mydoc_YYMMDD_HHMM.json
        m = re.match(doc_name + r"_(\d{6}_\d{4})\.json", basename)
        self.assertIsNotNone(m, f"文件名格式不正确: {basename}")
        # 检查时间戳格式
        ts = m.group(1)
        try:
            datetime.strptime(ts, "%y%m%d_%H%M")
        except Exception:
            self.fail(f"时间戳格式不正确: {ts}")
        # 检查内容
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data, self.chunks)

    def test_save_with_chinese_docname(self):
        doc_name = "规则书"
        output_file = save_chunks_to_json(self.chunks, output_dir=self.tmp_dir, source_file=doc_name+".md")
        basename = os.path.basename(output_file)
        m = re.match(doc_name + r"_(\d{6}_\d{4})\.json", basename)
        self.assertIsNotNone(m, f"中文文件名格式不正确: {basename}")

    def test_save_without_source_file_fallback(self):
        """测试没有source_file参数时的回退行为"""
        output_file = save_chunks_to_json(self.chunks, output_dir=self.tmp_dir)
        basename = os.path.basename(output_file)
        # 应该回退到原来的格式: semantic_chunks_YYMMDD_HHMM.json
        m = re.match(r"semantic_chunks_(\d{6}_\d{4})\.json", basename)
        self.assertIsNotNone(m, f"回退文件名格式不正确: {basename}")

    def test_extract_filename_without_extension(self):
        """测试从带后缀的文件名中提取不带后缀的文件名"""
        test_cases = [
            ("test.md", "test"),
            ("规则书.txt", "规则书"),
            ("my_document.pdf", "my_document"),
            ("file_without_ext", "file_without_ext"),
            ("complex.name.with.dots.md", "complex.name.with.dots")
        ]
        
        for filename, expected in test_cases:
            output_file = save_chunks_to_json(self.chunks, output_dir=self.tmp_dir, source_file=filename)
            basename = os.path.basename(output_file)
            m = re.match(re.escape(expected) + r"_(\d{6}_\d{4})\.json", basename)
            self.assertIsNotNone(m, f"文件名提取不正确: {filename} -> {basename}") 