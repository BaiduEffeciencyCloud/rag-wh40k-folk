#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaltool.analyze_sparse.sparse_analyzer import SparseAnalyzer
from dataupload.bm25_manager import BM25Manager
from evaltool.analyze_sparse.config_manager import SparseAnalyzerConfig

class TestAnalyzeSparseDictLogic(unittest.TestCase):
    """测试analyze_sparse的dict参数逻辑"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_dir = tempfile.mkdtemp()
        self.dict_dir = os.path.join(self.test_dir, 'dict')
        os.makedirs(self.dict_dir, exist_ok=True)
        
        # 创建测试词典文件
        self.create_test_dict_files()
        
        # 创建测试chunk文件
        self.chunk_file = os.path.join(self.test_dir, 'test_chunks.json')
        self.create_test_chunk_file()
        
        # 创建配置
        self.config = SparseAnalyzerConfig()
        self.config.report_output_path = os.path.join(self.test_dir, 'test_report.json')
    
    def create_test_dict_files(self):
        """创建测试词典文件"""
        # 创建不同时间戳的词典文件
        dict_files = [
            ('bm25_vocab_2507060757.json', {'test1': 1.0, 'test2': 2.0}),
            ('bm25_vocab_2507060800.json', {'test1': 1.0, 'test2': 2.0, 'test3': 3.0}),
            ('bm25_vocab_2507060900.json', {'test1': 1.0, 'test2': 2.0, 'test3': 3.0, 'test4': 4.0}),
            ('custom_dict.json', {'custom1': 1.0, 'custom2': 2.0}),
        ]
        
        for filename, content in dict_files:
            filepath = os.path.join(self.dict_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
    
    def create_test_chunk_file(self):
        """创建测试chunk文件"""
        chunks = [
            {
                'chunk_id': 'test-1',
                'content': '这是一个测试chunk，包含一些测试内容。',
                'metadata': {'source_file': 'test.md'}
            },
            {
                'chunk_id': 'test-2', 
                'content': '这是另一个测试chunk，用于验证功能。',
                'metadata': {'source_file': 'test.md'}
            }
        ]
        
        with open(self.chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    def test_dict_logic_head_string(self):
        """测试传入head字符串时自动查找最新词典"""
        with patch('dataupload.bm25_manager.BM25Manager.get_latest_dict_file') as mock_get_latest:
            mock_get_latest.return_value = os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json')
            
            analyzer = SparseAnalyzer(self.config)
            # 模拟analyze方法，只测试dict_file参数处理逻辑
            dict_path = analyzer._resolve_dict_file_path('head')
            
            # 验证调用了get_latest_dict_file
            mock_get_latest.assert_called_once()
            self.assertEqual(dict_path, os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json'))
    
    def test_dict_logic_specific_file(self):
        """测试传入特定文件名时使用指定词典"""
        specific_dict = 'custom_dict.json'
        analyzer = SparseAnalyzer(self.config)
        
        # 模拟analyze方法，只测试dict_file参数处理逻辑
        dict_path = analyzer._resolve_dict_file_path(specific_dict)
        
        # 验证返回了指定的文件名（相对路径）
        self.assertEqual(dict_path, specific_dict)
    
    def test_dict_logic_none_parameter(self):
        """测试不传参数时使用默认逻辑"""
        with patch('dataupload.bm25_manager.BM25Manager.get_latest_dict_file') as mock_get_latest:
            mock_get_latest.return_value = os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json')
            
            analyzer = SparseAnalyzer(self.config)
            dict_path = analyzer._resolve_dict_file_path(None)
            
            # 验证调用了get_latest_dict_file
            mock_get_latest.assert_called_once()
            self.assertEqual(dict_path, os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json'))
    
    def test_dict_logic_full_path(self):
        """测试传入完整路径时直接使用"""
        full_path = '/absolute/path/to/custom_dict.json'
        analyzer = SparseAnalyzer(self.config)
        
        dict_path = analyzer._resolve_dict_file_path(full_path)
        
        # 验证返回了完整路径
        self.assertEqual(dict_path, full_path)
    
    def test_dict_logic_relative_path(self):
        """测试传入相对路径时正确解析"""
        relative_path = 'dict/custom_dict.json'
        analyzer = SparseAnalyzer(self.config)
        
        dict_path = analyzer._resolve_dict_file_path(relative_path)
        
        # 验证返回了相对路径
        self.assertEqual(dict_path, relative_path)
    
    def test_dict_logic_invalid_head(self):
        """测试无效的head参数"""
        analyzer = SparseAnalyzer(self.config)
        
        # 传入非'head'的字符串，应该当作文件名处理
        dict_path = analyzer._resolve_dict_file_path('invalid_head')
        
        # 验证返回了文件名
        self.assertEqual(dict_path, 'invalid_head')
    
    def test_dict_logic_case_insensitive(self):
        """测试大小写不敏感性"""
        analyzer = SparseAnalyzer(self.config)
        
        # 测试不同大小写的head，都应该触发自动查找逻辑
        with patch('dataupload.bm25_manager.BM25Manager.get_latest_dict_file') as mock_get_latest:
            mock_get_latest.return_value = os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json')
            
            # 所有大小写的head都应该调用get_latest_dict_file
            dict_path1 = analyzer._resolve_dict_file_path('HEAD')
            dict_path2 = analyzer._resolve_dict_file_path('Head')
            dict_path3 = analyzer._resolve_dict_file_path('head')
            
            # 验证都调用了get_latest_dict_file
            self.assertEqual(mock_get_latest.call_count, 3)
            self.assertEqual(dict_path1, os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json'))
            self.assertEqual(dict_path2, os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json'))
            self.assertEqual(dict_path3, os.path.join(self.dict_dir, 'bm25_vocab_2507060900.json'))
    
    def test_dict_logic_empty_string(self):
        """测试空字符串参数"""
        analyzer = SparseAnalyzer(self.config)
        
        # 空字符串应该当作文件名处理
        dict_path = analyzer._resolve_dict_file_path('')
        
        # 验证返回了空字符串
        self.assertEqual(dict_path, '')
    
    def tearDown(self):
        """清理测试文件"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 