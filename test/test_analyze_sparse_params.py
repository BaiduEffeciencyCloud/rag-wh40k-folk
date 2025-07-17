#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from evaltool.analyze_sparse.config_manager import SparseAnalyzerConfig
from dataupload.bm25_manager import BM25Manager


class TestAnalyzeSparseParams(unittest.TestCase):
    """测试analyze_sparse工具的参数覆写功能"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SparseAnalyzerConfig()
    
    def test_config_default_values(self):
        """测试配置默认值"""
        self.assertEqual(self.config.min_freq, 3)
        self.assertEqual(self.config.max_vocab_size, 10000)
    
    def test_set_bm25_params(self):
        """测试设置BM25参数"""
        # 设置自定义参数
        self.config.set_bm25_params(min_freq=2, max_vocab_size=5000)
        
        # 验证参数已正确设置
        self.assertEqual(self.config.min_freq, 2)
        self.assertEqual(self.config.max_vocab_size, 5000)
    
    def test_bm25_manager_uses_config_params(self):
        """测试BM25Manager使用配置参数"""
        # 设置自定义参数
        self.config.set_bm25_params(min_freq=1, max_vocab_size=3000)
        
        # 创建BM25Manager实例 - 直接传递参数而不是配置对象
        bm25_manager = BM25Manager(
            min_freq=self.config.min_freq,
            max_vocab_size=self.config.max_vocab_size
        )
        
        # 验证参数已正确传递
        self.assertEqual(bm25_manager.min_freq, 1)
        self.assertEqual(bm25_manager.max_vocab_size, 3000)
    
    @patch('dataupload.bm25_manager.BM25Manager')
    def test_generate_sparse_vectors_uses_config_params(self, mock_original_bm25):
        """测试生成sparse向量时使用配置参数"""
        # 设置自定义参数
        self.config.set_bm25_params(min_freq=2, max_vocab_size=4000)
        
        # 创建BM25Manager实例 - 直接传递参数
        bm25_manager = BM25Manager(
            min_freq=self.config.min_freq,
            max_vocab_size=self.config.max_vocab_size
        )
        
        # 模拟词典数据
        bm25_manager.vocabulary = {"测试": 0, "词汇": 1}
        
        # 模拟BM25Manager实例
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.is_fitted = True
        mock_bm25_instance.get_sparse_vector.return_value = {"indices": [0], "values": [1.0]}
        mock_original_bm25.return_value = mock_bm25_instance
        
        # 调用生成sparse向量，传入配置参数
        texts = ["这是一个测试文本"]
        bm25_manager.generate_sparse_vectors(texts, config=self.config)
        
        # 验证BM25Manager被正确调用，传递了配置参数
        mock_original_bm25.assert_called_once_with(
            min_freq=2,
            max_vocab_size=4000
        )
    
    def test_config_to_dict_includes_bm25_params(self):
        """测试配置转字典包含BM25参数"""
        # 设置自定义参数
        self.config.set_bm25_params(min_freq=5, max_vocab_size=8000)
        
        # 转换为字典
        config_dict = self.config.to_dict()
        
        # 验证包含BM25参数
        self.assertEqual(config_dict['min_freq'], 5)
        self.assertEqual(config_dict['max_vocab_size'], 8000)
    
    def test_config_str_representation(self):
        """测试配置的字符串表示"""
        # 设置自定义参数
        self.config.set_bm25_params(min_freq=4, max_vocab_size=6000)
        
        # 验证字符串表示
        config_str = str(self.config)
        self.assertIn("min_freq=4", config_str)
        self.assertIn("max_vocab_size=6000", config_str)


if __name__ == '__main__':
    unittest.main() 