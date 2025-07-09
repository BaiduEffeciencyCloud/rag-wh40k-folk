#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词典评估工具测试
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaltool.analyze_sparse.dict_eval import DictEvaluator


class TestDictEvaluator(unittest.TestCase):
    """词典评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试词典数据
        self.test_vocab_data = {
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
            },
            "测试词4": {
                "weight": 0.2,
                "freq": 30,
                "df": 12
            },
            "测试词5": {
                "weight": 0.9,
                "freq": 2,
                "df": 1
            }
        }
        
        # 创建测试词典文件
        self.test_dict_path = os.path.join(self.temp_dir, "test_vocab.json")
        with open(self.test_dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_vocab_data, f, ensure_ascii=False, indent=2)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_valid_dict(self):
        """测试使用有效词典初始化"""
        evaluator = DictEvaluator(self.test_dict_path)
        self.assertEqual(len(evaluator.vocab_data), 5)
        self.assertIn("测试词1", evaluator.vocab_data)
        self.assertEqual(evaluator.vocab_data["测试词1"]["weight"], 0.8)
    
    def test_init_with_invalid_path(self):
        """测试使用无效路径初始化"""
        invalid_path = os.path.join(self.temp_dir, "nonexistent.json")
        evaluator = DictEvaluator(invalid_path)
        self.assertEqual(len(evaluator.vocab_data), 0)
    
    def test_get_basic_stats(self):
        """测试获取基础统计信息"""
        evaluator = DictEvaluator(self.test_dict_path)
        stats = evaluator.get_basic_stats()
        
        # 检查基本结构
        self.assertIn('total_words', stats)
        self.assertIn('weight_stats', stats)
        self.assertIn('freq_stats', stats)
        self.assertIn('df_stats', stats)
        
        # 检查数值
        self.assertEqual(stats['total_words'], 5)
        self.assertAlmostEqual(stats['weight_stats']['mean'], 0.58, places=2)
        self.assertAlmostEqual(stats['freq_stats']['mean'], 13.4, places=1)
        self.assertAlmostEqual(stats['df_stats']['mean'], 5.8, places=1)
    
    def test_get_basic_stats_empty_data(self):
        """测试空数据的统计信息"""
        evaluator = DictEvaluator("nonexistent.json")
        stats = evaluator.get_basic_stats()
        self.assertEqual(stats, {})
    
    def test_analyze_noise_patterns(self):
        """测试噪声模式分析"""
        evaluator = DictEvaluator(self.test_dict_path)
        patterns = evaluator.analyze_noise_patterns()
        
        # 检查基本结构
        self.assertIn('top_weight_words', patterns)
        self.assertIn('top_freq_words', patterns)
        self.assertIn('top_df_words', patterns)
        self.assertIn('high_weight_low_freq', patterns)
        self.assertIn('low_weight_high_freq', patterns)
        self.assertIn('weight_percentiles', patterns)
        self.assertIn('freq_percentiles', patterns)
        
        # 检查排序结果
        top_weight = patterns['top_weight_words']
        self.assertEqual(top_weight[0][0], "测试词5")  # 权重最高
        self.assertEqual(top_weight[0][1]['weight'], 0.9)
        
        top_freq = patterns['top_freq_words']
        self.assertEqual(top_freq[0][0], "测试词4")  # 词频最高
        self.assertEqual(top_freq[0][1]['freq'], 30)
    
    def test_analyze_noise_patterns_empty_data(self):
        """测试空数据的噪声模式分析"""
        evaluator = DictEvaluator("nonexistent.json")
        patterns = evaluator.analyze_noise_patterns()
        self.assertEqual(patterns, {})
    
    def test_generate_report(self):
        """测试生成报告"""
        evaluator = DictEvaluator(self.test_dict_path)
        report_path = evaluator.generate_report(self.temp_dir)
        
        # 检查报告文件是否存在
        self.assertTrue(os.path.exists(report_path))
        
        # 检查报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        self.assertIn('timestamp', report)
        self.assertIn('dict_file', report)
        self.assertIn('basic_stats', report)
        self.assertIn('noise_analysis', report)
        
        # 检查摘要文件是否存在
        summary_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_dict_eval_summary.txt')]
        self.assertGreater(len(summary_files), 0)
    
    def test_generate_report_empty_data(self):
        """测试空数据生成报告"""
        evaluator = DictEvaluator("nonexistent.json")
        result = evaluator.generate_report(self.temp_dir)
        self.assertEqual(result, "无数据可分析")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_visualizations(self, mock_close, mock_savefig):
        """测试创建可视化图表"""
        evaluator = DictEvaluator(self.test_dict_path)
        evaluator.create_visualizations(self.temp_dir)
        
        # 检查是否调用了保存图表的方法
        self.assertGreater(mock_savefig.call_count, 0)
        self.assertGreater(mock_close.call_count, 0)
    
    def test_create_visualizations_empty_data(self):
        """测试空数据创建可视化"""
        evaluator = DictEvaluator("nonexistent.json")
        # 不应该抛出异常
        evaluator.create_visualizations(self.temp_dir)
    
    def test_load_dict_with_encoding_error(self):
        """测试加载词典时的编码错误处理"""
        # 创建一个损坏的JSON文件
        bad_dict_path = os.path.join(self.temp_dir, "bad_vocab.json")
        with open(bad_dict_path, 'w', encoding='utf-8') as f:
            f.write("这不是有效的JSON")
        
        evaluator = DictEvaluator(bad_dict_path)
        self.assertEqual(len(evaluator.vocab_data), 0)
    
    def test_percentile_calculations(self):
        """测试百分位数计算"""
        evaluator = DictEvaluator(self.test_dict_path)
        patterns = evaluator.analyze_noise_patterns()
        
        # 检查百分位数存在
        self.assertIn('p10', patterns['weight_percentiles'])
        self.assertIn('p25', patterns['weight_percentiles'])
        self.assertIn('p50', patterns['weight_percentiles'])
        self.assertIn('p75', patterns['weight_percentiles'])
        self.assertIn('p90', patterns['weight_percentiles'])
        
        # 检查百分位数顺序
        weight_percentiles = patterns['weight_percentiles']
        self.assertLess(weight_percentiles['p10'], weight_percentiles['p25'])
        self.assertLess(weight_percentiles['p25'], weight_percentiles['p50'])
        self.assertLess(weight_percentiles['p50'], weight_percentiles['p75'])
        self.assertLess(weight_percentiles['p75'], weight_percentiles['p90'])
    
    def test_noise_word_identification(self):
        """测试噪声词识别"""
        # 创建一个更大的测试数据集来测试噪声词识别
        large_test_data = {}
        for i in range(100):
            word = f"词{i}"
            # 创建一些明显的噪声模式
            if i < 10:  # 高权重低词频
                large_test_data[word] = {"weight": 0.9, "freq": 1, "df": 1}
            elif i < 20:  # 低权重高词频
                large_test_data[word] = {"weight": 0.1, "freq": 50, "df": 20}
            else:  # 正常分布
                large_test_data[word] = {"weight": 0.5, "freq": 10, "df": 5}
        
        large_dict_path = os.path.join(self.temp_dir, "large_vocab.json")
        with open(large_dict_path, 'w', encoding='utf-8') as f:
            json.dump(large_test_data, f, ensure_ascii=False, indent=2)
        
        evaluator = DictEvaluator(large_dict_path)
        patterns = evaluator.analyze_noise_patterns()
        
        # 检查是否识别出噪声词
        self.assertGreater(len(patterns['high_weight_low_freq']), 0)
        self.assertGreater(len(patterns['low_weight_high_freq']), 0)


if __name__ == '__main__':
    unittest.main() 