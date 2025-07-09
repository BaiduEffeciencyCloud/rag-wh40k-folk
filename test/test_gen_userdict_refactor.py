#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tempfile
import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataupload.gen_userdict import generate_bm25_manager_from_texts
from dataupload.bm25_manager import BM25Manager

class TestGenerateBM25ManagerFromTexts(unittest.TestCase):
    """测试generate_bm25_manager_from_texts函数"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_texts = [
            "战锤40000是一款桌面战争游戏",
            "游戏包含多个种族和单位",
            "每个单位都有独特的技能和属性",
            "玩家需要制定策略来获得胜利"
        ]
    
    def test_basic_functionality(self):
        """测试基本功能"""
        bm25, vocab_dict = generate_bm25_manager_from_texts(
            self.test_texts,
            min_freq=1,  # 降低阈值以便测试
            min_pmi=1.0,
            max_len=4,
            do_filter=False  # 关闭过滤以便测试
        )
        
        # 验证返回的BM25Manager
        self.assertIsInstance(bm25, BM25Manager)
        self.assertTrue(bm25.is_fitted)
        
        # 验证词汇表字典
        self.assertIsInstance(vocab_dict, dict)
        self.assertGreater(len(vocab_dict), 0)
        
        # 验证可以生成sparse向量
        test_text = "战锤40000游戏"
        sparse_vector = bm25.get_sparse_vector(test_text)
        self.assertIsInstance(sparse_vector, dict)
        self.assertIn('indices', sparse_vector)
        self.assertIn('values', sparse_vector)
    
    def test_empty_texts(self):
        """测试空文本列表"""
        with self.assertRaises(ValueError):
            generate_bm25_manager_from_texts([])
    
    def test_min_freq_parameter(self):
        """测试min_freq参数的影响"""
        # 使用min_freq=1
        bm25_1, vocab_1 = generate_bm25_manager_from_texts(
            self.test_texts,
            min_freq=1,
            do_filter=False
        )
        
        # 使用min_freq=3
        bm25_3, vocab_3 = generate_bm25_manager_from_texts(
            self.test_texts,
            min_freq=3,
            do_filter=False
        )
        
        # min_freq=1应该产生更多词汇
        self.assertGreaterEqual(len(vocab_1), len(vocab_3))
    
    def test_filter_parameter(self):
        """测试do_filter参数的影响"""
        # 不过滤
        bm25_no_filter, vocab_no_filter = generate_bm25_manager_from_texts(
            self.test_texts,
            do_filter=False
        )
        
        # 过滤
        bm25_filter, vocab_filter = generate_bm25_manager_from_texts(
            self.test_texts,
            do_filter=True
        )
        
        # 过滤后词汇应该更少
        self.assertGreaterEqual(len(vocab_no_filter), len(vocab_filter))
    
    def test_ac_mask_parameter(self):
        """测试ac_mask参数"""
        bm25_normal, vocab_normal = generate_bm25_manager_from_texts(
            self.test_texts,
            ac_mask=False
        )
        
        bm25_masked, vocab_masked = generate_bm25_manager_from_texts(
            self.test_texts,
            ac_mask=True
        )
        
        # 两种方式都应该能正常工作
        self.assertTrue(bm25_normal.is_fitted)
        self.assertTrue(bm25_masked.is_fitted)
    
    def test_auto_threshold_parameter(self):
        """测试auto_threshold参数"""
        bm25_normal, vocab_normal = generate_bm25_manager_from_texts(
            self.test_texts,
            auto_threshold=False
        )
        
        bm25_auto, vocab_auto = generate_bm25_manager_from_texts(
            self.test_texts,
            auto_threshold=True
        )
        
        # 两种方式都应该能正常工作
        self.assertTrue(bm25_normal.is_fitted)
        self.assertTrue(bm25_auto.is_fitted)
    
    def test_sparse_vector_generation(self):
        """测试sparse向量生成"""
        bm25, vocab_dict = generate_bm25_manager_from_texts(
            self.test_texts,
            min_freq=1,
            do_filter=False
        )
        
        # 测试多个文本的sparse向量生成
        test_texts = [
            "战锤40000",
            "桌面战争游戏",
            "种族和单位",
            "策略和胜利"
        ]
        
        for text in test_texts:
            sparse_vector = bm25.get_sparse_vector(text)
            self.assertIsInstance(sparse_vector, dict)
            self.assertIn('indices', sparse_vector)
            self.assertIn('values', sparse_vector)
            self.assertEqual(len(sparse_vector['indices']), len(sparse_vector['values']))
    
    def test_vocabulary_consistency(self):
        """测试词汇表一致性"""
        bm25, vocab_dict = generate_bm25_manager_from_texts(
            self.test_texts,
            min_freq=1,
            do_filter=False
        )
        
        # 验证BM25Manager的词汇表和返回的词汇表一致
        bm25_vocab = set(bm25.vocabulary.keys())
        returned_vocab = set(vocab_dict.keys())
        
        # 返回的词汇表应该是BM25词汇表的子集（因为过滤）
        self.assertTrue(returned_vocab.issubset(bm25_vocab))

if __name__ == '__main__':
    unittest.main() 