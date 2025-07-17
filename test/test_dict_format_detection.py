#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词典格式检测逻辑
"""

import unittest
import json
import os
import tempfile
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataupload.bm25_manager import BM25Manager


class TestDictFormatDetection(unittest.TestCase):
    """测试词典格式检测"""
    
    def setUp(self):
        """测试前准备"""
        self.bm25_manager = BM25Manager()
    
    def test_traditional_format(self):
        """测试传统格式词典"""
        # 创建传统格式词典
        traditional_dict = {
            "测试词1": 0.8,
            "测试词2": 0.6,
            "测试词3": 0.4
        }
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(traditional_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            # 验证结果
            self.assertEqual(self.bm25_manager.vocabulary, traditional_dict)
            self.assertEqual(dict_data, traditional_dict)
            self.assertEqual(len(self.bm25_manager.vocabulary), 3)
            
        finally:
            os.unlink(dict_file_path)
    
    def test_freq_format(self):
        """测试带词频格式词典"""
        # 创建带词频格式词典
        freq_dict = {
            "测试词1": {
                "weight": 0.8,
                "freq": 10,
                "df": 5
            },
            "测试词2": {
                "weight": 0.6,
                "freq": 20,
                "df": 8
            },
            "测试词3": {
                "weight": 0.4,
                "freq": 5,
                "df": 3
            }
        }
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(freq_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            # 验证结果
            expected_vocabulary = {
                "测试词1": 0.8,
                "测试词2": 0.6,
                "测试词3": 0.4
            }
            self.assertEqual(self.bm25_manager.vocabulary, expected_vocabulary)
            self.assertEqual(dict_data, freq_dict)  # 原始数据保持不变
            self.assertEqual(len(self.bm25_manager.vocabulary), 3)
            
        finally:
            os.unlink(dict_file_path)
    
    def test_vocabulary_field_format(self):
        """测试包含vocabulary字段的格式"""
        # 创建包含vocabulary字段的词典
        vocab_dict = {
            "vocabulary": {
                "测试词1": 0.8,
                "测试词2": 0.6,
                "测试词3": 0.4
            },
            "metadata": {
                "created_at": "2024-12-01",
                "version": "1.0"
            }
        }
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            # 验证结果
            self.assertEqual(self.bm25_manager.vocabulary, vocab_dict["vocabulary"])
            self.assertEqual(dict_data, vocab_dict)
            self.assertEqual(len(self.bm25_manager.vocabulary), 3)
            
        finally:
            os.unlink(dict_file_path)
    
    def test_empty_dict(self):
        """测试空词典"""
        empty_dict = {}
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            # 验证结果
            self.assertEqual(self.bm25_manager.vocabulary, {})
            self.assertEqual(dict_data, {})
            self.assertEqual(len(self.bm25_manager.vocabulary), 0)
            
        finally:
            os.unlink(dict_file_path)
    
    def test_invalid_format(self):
        """测试无效格式"""
        # 创建无效格式（列表而不是字典）
        invalid_dict = ["测试词1", "测试词2", "测试词3"]
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典应该抛出异常
            with self.assertRaises(Exception) as context:
                self.bm25_manager.load_dictionary(dict_file_path)
            
            self.assertIn("词典文件格式不正确", str(context.exception))
            
        finally:
            os.unlink(dict_file_path)
    
    def test_mixed_format_detection(self):
        """测试混合格式检测"""
        # 创建一个包含不同值类型的词典（这种情况不应该发生，但测试健壮性）
        mixed_dict = {
            "测试词1": 0.8,  # 传统格式
            "测试词2": {     # 带词频格式
                "weight": 0.6,
                "freq": 20,
                "df": 8
            }
        }
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mixed_dict, f, ensure_ascii=False, indent=2)
            dict_file_path = f.name
        
        try:
            # 加载词典
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            # 由于第一个值是数字，应该被识别为传统格式
            # 但第二个值是字典，这会导致问题
            # 实际行为取决于第一个值的类型
            self.assertEqual(self.bm25_manager.vocabulary, mixed_dict)
            
        finally:
            os.unlink(dict_file_path)


if __name__ == '__main__':
    unittest.main() 