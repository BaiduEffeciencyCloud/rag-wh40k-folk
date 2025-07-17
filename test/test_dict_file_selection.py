#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词典文件选择逻辑
"""

import unittest
import json
import os
import tempfile
import shutil
import sys
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataupload.bm25_manager import BM25Manager


class TestDictFileSelection(unittest.TestCase):
    """测试词典文件选择逻辑"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.bm25_manager = BM25Manager()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_get_latest_dict_file(self):
        """测试获取最新传统格式词典文件"""
        # 创建测试文件
        files = [
            "bm25_vocab_2507071200.json",
            "bm25_vocab_2507071300.json",
            "bm25_vocab_2507071400.json",
            "bm25_vocab_freq_2507071500.json"  # 这个应该被排除
        ]
        
        for file in files:
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write('{"test": 1}')
        
        # 测试获取最新传统格式词典
        latest_file = self.bm25_manager.get_latest_dict_file(self.temp_dir)
        self.assertEqual(latest_file, os.path.join(self.temp_dir, "bm25_vocab_2507071400.json"))
    
    def test_get_latest_freq_dict_file(self):
        """测试获取最新带词频格式词典文件"""
        # 创建测试文件
        files = [
            "bm25_vocab_2507071200.json",  # 这个应该被排除
            "bm25_vocab_freq_2507071300.json",
            "bm25_vocab_freq_2507071400.json",
            "bm25_vocab_freq_2507071500.json"
        ]
        
        for file in files:
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write('{"test": {"weight": 1, "freq": 10, "df": 5}}')
        
        # 测试获取最新带词频格式词典
        latest_file = self.bm25_manager.get_latest_freq_dict_file(self.temp_dir)
        self.assertEqual(latest_file, os.path.join(self.temp_dir, "bm25_vocab_freq_2507071500.json"))
    
    def test_get_latest_dict_file_empty_dir(self):
        """测试空目录获取传统格式词典"""
        latest_file = self.bm25_manager.get_latest_dict_file(self.temp_dir)
        self.assertIsNone(latest_file)
    
    def test_get_latest_freq_dict_file_empty_dir(self):
        """测试空目录获取带词频格式词典"""
        latest_file = self.bm25_manager.get_latest_freq_dict_file(self.temp_dir)
        self.assertIsNone(latest_file)
    
    def test_get_latest_dict_file_no_freq_files(self):
        """测试没有带词频格式文件时获取传统格式词典"""
        # 只创建传统格式文件
        files = [
            "bm25_vocab_2507071200.json",
            "bm25_vocab_2507071300.json"
        ]
        
        for file in files:
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write('{"test": 1}')
        
        latest_file = self.bm25_manager.get_latest_dict_file(self.temp_dir)
        self.assertEqual(latest_file, os.path.join(self.temp_dir, "bm25_vocab_2507071300.json"))
        
        # 带词频格式应该返回None
        latest_freq_file = self.bm25_manager.get_latest_freq_dict_file(self.temp_dir)
        self.assertIsNone(latest_freq_file)
    
    def test_get_latest_freq_dict_file_no_traditional_files(self):
        """测试没有传统格式文件时获取带词频格式词典"""
        # 只创建带词频格式文件
        files = [
            "bm25_vocab_freq_2507071200.json",
            "bm25_vocab_freq_2507071300.json"
        ]
        
        for file in files:
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write('{"test": {"weight": 1, "freq": 10, "df": 5}}')
        
        latest_freq_file = self.bm25_manager.get_latest_freq_dict_file(self.temp_dir)
        self.assertEqual(latest_freq_file, os.path.join(self.temp_dir, "bm25_vocab_freq_2507071300.json"))
        
        # 传统格式应该返回None
        latest_file = self.bm25_manager.get_latest_dict_file(self.temp_dir)
        self.assertIsNone(latest_file)
    
    def test_load_dictionary_traditional_format(self):
        """测试加载传统格式词典"""
        # 创建传统格式词典文件
        traditional_dict = {"测试词1": 0.8, "测试词2": 0.6}
        dict_file = os.path.join(self.temp_dir, "bm25_vocab_2507071200.json")
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(traditional_dict, f, ensure_ascii=False, indent=2)
        
        # 加载词典
        dict_data = self.bm25_manager.load_dictionary(dict_file)
        
        # 验证结果
        self.assertEqual(self.bm25_manager.vocabulary, traditional_dict)
        self.assertEqual(dict_data, traditional_dict)
    
    def test_load_dictionary_freq_format(self):
        """测试加载带词频格式词典"""
        # 创建带词频格式词典文件
        freq_dict = {
            "测试词1": {"weight": 0.8, "freq": 10, "df": 5},
            "测试词2": {"weight": 0.6, "freq": 20, "df": 8}
        }
        dict_file = os.path.join(self.temp_dir, "bm25_vocab_freq_2507071200.json")
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(freq_dict, f, ensure_ascii=False, indent=2)
        
        # 加载词典
        dict_data = self.bm25_manager.load_dictionary(dict_file)
        
        # 验证结果
        expected_vocabulary = {"测试词1": 0.8, "测试词2": 0.6}
        self.assertEqual(self.bm25_manager.vocabulary, expected_vocabulary)
        self.assertEqual(dict_data, freq_dict)  # 原始数据保持不变


if __name__ == '__main__':
    unittest.main() 