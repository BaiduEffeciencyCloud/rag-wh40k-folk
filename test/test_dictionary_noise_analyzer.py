#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock

from evaltool.analyze_sparse.config_manager import SparseAnalyzerConfig



class TestDictionaryNoiseAnalyzer(unittest.TestCase):
    """词典噪声分析器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = SparseAnalyzerConfig()
        self.analyzer = DictionaryNoiseAnalyzer(self.config)
        
        # 创建测试词典数据
        self.normal_vocabulary = {
            "测试": 10,
            "词汇": 15,
            "分析": 8,
            "功能": 12,
            "评估": 6,
            "质量": 9,
            "噪声": 5,
            "检测": 11,
            "统计": 7,
            "分布": 13
        }
        
        self.vocabulary_with_outliers = {
            "正常词汇1": 10,
            "正常词汇2": 12,
            "正常词汇3": 8,
            "异常高频词汇": 150,  # 异常高频
            "异常低频词汇": 1,    # 异常低频
            "另一个高频": 120,    # 异常高频
            "正常词汇4": 9,
            "正常词汇5": 11
        }
        
        self.vocabulary_with_abnormal_lengths = {
            "a": 5,                    # 单字符
            "正常词汇": 10,
            "另一个正常词汇": 12,
            "超长词汇这是一个非常长的词汇用于测试": 8,  # 超长
            "b": 3,                    # 单字符
            "正常": 7,
            "另一个超长词汇这也是一个非常长的词汇": 6,   # 超长
            "正常词汇2": 9
        }
    
    def test_frequency_distribution_normal_case(self):
        """测试正常词频分布分析"""
        result = self.analyzer._analyze_frequency_distribution(self.normal_vocabulary)
        self.assertIsInstance(result, dict)
        for key in ['outlier_ratio', 'skewness', 'kurtosis', 'freq_cv', 'high_freq_ratio', 'outlier_words', 'high_freq_words']:
            self.assertIn(key, result)
        self.assertIsInstance(result['outlier_words'], list)
        self.assertIsInstance(result['high_freq_words'], list)
        self.assertIsInstance(result['outlier_ratio'], float)
        self.assertIsInstance(result['freq_cv'], float)
    
    def test_frequency_distribution_with_outliers(self):
        """测试包含异常值的词频分布分析"""
        result = self.analyzer._analyze_frequency_distribution(self.vocabulary_with_outliers)
        self.assertIsInstance(result, dict)
        for key in ['outlier_ratio', 'skewness', 'kurtosis', 'freq_cv', 'high_freq_ratio', 'outlier_words', 'high_freq_words']:
            self.assertIn(key, result)
        self.assertIsInstance(result['outlier_words'], list)
        self.assertIsInstance(result['high_freq_words'], list)
        self.assertIsInstance(result['outlier_ratio'], float)
        self.assertIsInstance(result['freq_cv'], float)
    
    def test_frequency_distribution_empty_vocabulary(self):
        """测试空词典的词频分布分析"""
        result = self.analyzer._analyze_frequency_distribution({})
        self.assertIsInstance(result, dict)
        self.assertEqual(result['outlier_ratio'], 0)
        self.assertEqual(result['high_freq_ratio'], 0)
        self.assertEqual(len(result.get('outlier_words', [])), 0)
        self.assertEqual(len(result.get('high_freq_words', [])), 0)
    
    def test_frequency_distribution_single_value(self):
        """测试单一值的词频分布分析"""
        single_value_vocab = {"word1": 10, "word2": 10, "word3": 10}
        result = self.analyzer._analyze_frequency_distribution(single_value_vocab)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['outlier_ratio'], float)
        self.assertIsInstance(result['freq_cv'], float)
    
    def test_word_length_distribution_normal_case(self):
        """测试正常词汇长度分布分析"""
        result = self.analyzer._analyze_word_length_distribution(self.normal_vocabulary)
        self.assertIsInstance(result, dict)
        self.assertIn('length_stats', result)
        self.assertIn('abnormal_length_words', result)
        self.assertIn('abnormal_length_ratio', result)
        length_stats = result['length_stats']
        for key in ['avg_length', 'std_length', 'min_length', 'max_length', 'single_char_ratio', 'very_long_ratio']:
            self.assertIn(key, length_stats)
        self.assertIsInstance(result['abnormal_length_words'], list)
        self.assertIsInstance(result['abnormal_length_ratio'], float)
    
    def test_word_length_distribution_with_abnormal_lengths(self):
        """测试包含异常长度的词汇长度分布分析"""
        result = self.analyzer._analyze_word_length_distribution(self.vocabulary_with_abnormal_lengths)
        self.assertIsInstance(result, dict)
        self.assertIn('length_stats', result)
        self.assertIn('abnormal_length_words', result)
        self.assertIn('abnormal_length_ratio', result)
        self.assertIsInstance(result['abnormal_length_words'], list)
        self.assertIsInstance(result['abnormal_length_ratio'], float)
    
    def test_word_length_distribution_empty_vocabulary(self):
        """测试空词典的长度分布分析"""
        result = self.analyzer._analyze_word_length_distribution({})
        self.assertIsInstance(result, dict)
        self.assertEqual(result['abnormal_length_ratio'], 0)
        self.assertEqual(len(result['abnormal_length_words']), 0)
        length_stats = result['length_stats']
        self.assertEqual(length_stats['avg_length'], 0)
        self.assertEqual(length_stats['std_length'], 0)
        self.assertEqual(length_stats['min_length'], 0)
        self.assertEqual(length_stats['max_length'], 0)
        self.assertEqual(length_stats['single_char_ratio'], 0)
        self.assertEqual(length_stats['very_long_ratio'], 0)
    
    def test_word_length_distribution_single_length(self):
        """测试单一长度的词汇长度分布分析"""
        single_length_vocab = {"word1": 10, "word2": 10, "word3": 10}
        result = self.analyzer._analyze_word_length_distribution(single_length_vocab)
        self.assertIsInstance(result, dict)
        self.assertIn('length_stats', result)
        self.assertIn('abnormal_length_words', result)
        self.assertIn('abnormal_length_ratio', result)
        self.assertIsInstance(result['abnormal_length_words'], list)
        self.assertIsInstance(result['abnormal_length_ratio'], float)
    
    def test_generate_noise_report_normal_case(self):
        """测试正常报告生成"""
        # 模拟分析结果
        analysis_result = {
            'frequency_analysis': {
                'outlier_ratio': 0.05,
                'skewness': 2.1,
                'kurtosis': 8.5,
                'freq_cv': 1.2,
                'high_freq_ratio': 0.08,
                'outlier_words': ['word1', 'word2'],
                'high_freq_words': ['word3', 'word4'],
                'mean_freq': 10.5,
                'std_freq': 5.2,
                'median_freq': 9.0,
                'outlier_threshold_low': 0.0,
                'outlier_threshold_high': 26.1,
                'high_freq_threshold': 20.9
            },
            'length_analysis': {
                'length_stats': {
                    'avg_length': 3.2,
                    'std_length': 1.8,
                    'min_length': 1,
                    'max_length': 15,
                    'single_char_ratio': 0.12,
                    'very_long_ratio': 0.03
                },
                'abnormal_length_words': ['a', 'b'],
                'abnormal_length_ratio': 0.06
            },
            'vocabulary_size': 1000,
            'noise_score': 15.0,
            'noise_level': 'medium',
            'recommendations': ['建议1', '建议2']
        }
        
        dict_file_path = "dict/test_vocab.json"
        report_content = self.analyzer._generate_noise_report(analysis_result, dict_file_path)
        
        # 验证报告内容
        self.assertIsInstance(report_content, str)
        self.assertIn("词典噪声量化评估报告", report_content)
        self.assertIn("词频分布异常检测", report_content)
        self.assertIn("词汇长度分布分析", report_content)
        self.assertIn("综合评估", report_content)
        self.assertIn("dict/test_vocab.json", report_content)
        # 检查词汇数量，兼容带千分位分隔符的格式
        self.assertTrue("1000" in report_content or "1,000" in report_content)
    
    def test_generate_noise_report_with_high_noise(self):
        """测试高噪声报告生成"""
        # 模拟高噪声分析结果
        analysis_result = {
            'frequency_analysis': {
                'outlier_ratio': 0.25,  # 高异常值比例
                'skewness': 5.0,
                'kurtosis': 15.0,
                'freq_cv': 2.5,
                'high_freq_ratio': 0.20,
                'outlier_words': ['word1', 'word2', 'word3', 'word4', 'word5'],
                'high_freq_words': ['word6', 'word7', 'word8'],
                'mean_freq': 20.0,
                'std_freq': 50.0,
                'median_freq': 10.0,
                'outlier_threshold_low': 0.0,
                'outlier_threshold_high': 170.0,
                'high_freq_threshold': 120.0
            },
            'length_analysis': {
                'length_stats': {
                    'avg_length': 2.0,
                    'std_length': 3.0,
                    'min_length': 1,
                    'max_length': 25,
                    'single_char_ratio': 0.30,  # 高单字符比例
                    'very_long_ratio': 0.15
                },
                'abnormal_length_words': ['a', 'b', 'c', 'd', 'e'],
                'abnormal_length_ratio': 0.25  # 高异常长度比例
            },
            'vocabulary_size': 500,
            'noise_score': 45.0,
            'noise_level': 'high',
            'recommendations': ['高噪声建议1', '高噪声建议2']
        }
        
        dict_file_path = "dict/high_noise_vocab.json"
        report_content = self.analyzer._generate_noise_report(analysis_result, dict_file_path)
        
        # 验证高噪声报告内容
        self.assertIsInstance(report_content, str)
        self.assertIn("词典噪声量化评估报告", report_content)
        self.assertIn("高噪声建议1", report_content)
        self.assertIn("高噪声建议2", report_content)
    
    def test_save_noise_report(self):
        """测试报告保存功能"""
        report_content = "# 测试报告\n\n这是一个测试报告。"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name
        
        try:
            self.analyzer._save_noise_report(report_content, temp_path)
            
            # 验证文件已保存
            self.assertTrue(os.path.exists(temp_path))
            
            # 验证文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            self.assertEqual(saved_content, report_content)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_analyze_noise_integration(self):
        """测试完整的噪声分析流程"""
        # 创建临时词典文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_dict_path = f.name
            json.dump(self.vocabulary_with_outliers, f, ensure_ascii=False, indent=2)
        
        try:
            # 执行完整分析
            result = self.analyzer.analyze_noise(temp_dict_path)
            
            # 验证返回结果结构
            self.assertIn('frequency_analysis', result)
            self.assertIn('length_analysis', result)
            self.assertIn('vocabulary_size', result)
            self.assertIn('noise_score', result)
            self.assertIn('noise_level', result)
            self.assertIn('recommendations', result)
            
            # 验证数据类型
            self.assertIsInstance(result['vocabulary_size'], int)
            self.assertIsInstance(result['noise_score'], float)
            self.assertIsInstance(result['noise_level'], str)
            self.assertIsInstance(result['recommendations'], list)
            
            # 验证噪声评分范围
            self.assertGreaterEqual(result['noise_score'], 0)
            self.assertLessEqual(result['noise_score'], 100)
            
            # 验证噪声等级
            self.assertIn(result['noise_level'], ['low', 'medium', 'high'])
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_dict_path):
                os.unlink(temp_dict_path)
    
    def test_analyze_noise_with_nonexistent_file(self):
        """测试分析不存在的词典文件"""
        with self.assertRaises(Exception):
            self.analyzer.analyze_noise("nonexistent_file.json")
    
    def test_analyze_noise_with_invalid_json(self):
        """测试分析无效的JSON文件"""
        # 创建无效的JSON文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_dict_path = f.name
            f.write("invalid json content")
        
        try:
            with self.assertRaises(Exception):
                self.analyzer.analyze_noise(temp_dict_path)
        finally:
            if os.path.exists(temp_dict_path):
                os.unlink(temp_dict_path)


if __name__ == '__main__':
    unittest.main() 