#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DictionaryNoiseScorer 单元测试骨架
只测试接口和逻辑，不断言具体评分结果
"""

import unittest
import tempfile
import os
import json
from evaltool.analyze_sparse.dict_noise_scorer import DictionaryNoiseScorer

class TestDictionaryNoiseScorer(unittest.TestCase):
    def setUp(self):
        # 构造一个简单的带freq词典
        self.vocab = {
            "词1": {"weight": 0.5, "freq": 10, "df": 5},
            "词2": {"weight": 0.8, "freq": 1, "df": 1},
            "词3": {"weight": 0.3, "freq": 5, "df": 3},
            "词4": {"weight": 0.2, "freq": 20, "df": 8},
            "词5": {"weight": 0.9, "freq": 2, "df": 1}
        }

    def test_score_interface(self):
        scorer = DictionaryNoiseScorer(self.vocab)
        result = scorer.score()
        # 检查输出类型和主要字段
        self.assertIsInstance(result, dict)
        self.assertIn("freq_noise_score", result)
        self.assertIn("length_noise_score", result)
        self.assertIn("overall_noise_score", result)
        self.assertIn("noise_level", result)
        self.assertIn("optimization_suggestions", result)
        self.assertIn("freq_outliers", result)
        self.assertIn("abnormal_length_words", result)

    def test_from_file(self):
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            path = f.name
        try:
            scorer = DictionaryNoiseScorer.from_file(path)
            result = scorer.score()
            self.assertIsInstance(result, dict)
        finally:
            os.unlink(path)

    def test_empty_vocab(self):
        scorer = DictionaryNoiseScorer({})
        result = scorer.score()
        self.assertIsInstance(result, dict)
        # 检查所有分数字段存在
        self.assertIn("freq_noise_score", result)
        self.assertIn("length_noise_score", result)
        self.assertIn("overall_noise_score", result)
        self.assertIn("noise_level", result)

    def test_single_word(self):
        vocab = {"唯一": {"weight": 0.1, "freq": 1, "df": 1}}
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_various_word_lengths(self):
        vocab = {
            "一": {"weight": 0.1, "freq": 1, "df": 1},
            "二二": {"weight": 0.2, "freq": 2, "df": 1},
            "三三三": {"weight": 0.3, "freq": 3, "df": 1},
            "四四四四": {"weight": 0.4, "freq": 4, "df": 1},
            "五五五五五": {"weight": 0.5, "freq": 5, "df": 1}
        }
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_all_zero_freq(self):
        vocab = {f"词{i}": {"weight": 0.1, "freq": 0, "df": 1} for i in range(10)}
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_all_same_freq(self):
        vocab = {f"词{i}": {"weight": 0.1, "freq": 5, "df": 1} for i in range(10)}
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_extreme_weights(self):
        vocab = {
            "极小": {"weight": 1e-10, "freq": 1, "df": 1},
            "极大": {"weight": 1e10, "freq": 1, "df": 1},
            "正常": {"weight": 0.5, "freq": 1, "df": 1}
        }
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_extreme_lengths(self):
        vocab = {
            "一": {"weight": 0.1, "freq": 1, "df": 1},
            "超长词汇" * 20: {"weight": 0.2, "freq": 2, "df": 1},
            "正常": {"weight": 0.3, "freq": 3, "df": 1}
        }
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_large_vocab(self):
        # 10000个词，freq均匀分布
        vocab = {f"词{i}": {"weight": 0.1, "freq": i % 10 + 1, "df": 1} for i in range(10000)}
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

    def test_negative_freq(self):
        # 理论上freq不应为负，但测试健壮性
        vocab = {"负": {"weight": 0.1, "freq": -5, "df": 1}, "正": {"weight": 0.2, "freq": 5, "df": 1}}
        scorer = DictionaryNoiseScorer(vocab)
        result = scorer.score()
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main() 