#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaltool.analyze_sparse.config_manager import SparseAnalyzerConfig
from evaltool.chunk_processor import ChunkProcessor
from dataupload.bm25_manager import BM25Manager
from evaltool.analyze_sparse.sparse_analyzer import SparseAnalyzer

class TestSparseAnalyzerConfig(unittest.TestCase):
    """测试配置管理类"""
    
    def setUp(self):
        self.config = SparseAnalyzerConfig()
    
    def test_default_config(self):
        """测试默认配置"""
        self.assertEqual(self.config.min_freq, 3)
        self.assertEqual(self.config.max_vocab_size, 10000)
        self.assertEqual(self.config.input_dir, "")
        self.assertEqual(self.config.output_dir, "dict")
        self.assertTrue(self.config.include_headers)
    
    def test_set_bm25_params(self):
        """测试设置BM25参数"""
        self.config.set_bm25_params(3, 15000)
        self.assertEqual(self.config.min_freq, 3)
        self.assertEqual(self.config.max_vocab_size, 15000)
    
    def test_set_paths(self):
        """测试设置路径"""
        self.config.set_paths(input_dir="test_data", output_dir="test_dict")
        self.assertEqual(self.config.input_dir, "test_data")
        self.assertEqual(self.config.output_dir, "test_dict")
    
    def test_to_dict(self):
        """测试转换为字典"""
        config_dict = self.config.to_dict()
        self.assertIn('min_freq', config_dict)
        self.assertIn('max_vocab_size', config_dict)
        self.assertIn('input_dir', config_dict)
        self.assertIn('output_dir', config_dict)

class TestChunkProcessor(unittest.TestCase):
    """测试Chunk处理器"""
    
    def setUp(self):
        self.config = SparseAnalyzerConfig()
        self.processor = ChunkProcessor(self.config)
        
        # 创建测试chunk数据
        self.test_chunk = {
            'text': '这是一个测试文本，包含一些重要信息。',
            'section_heading': '测试章节',
            'h1': '主标题',
            'h2': '子标题',
            'source_file': 'test.md'
        }
    
    def test_extract_chunk_text_with_headers(self):
        """测试提取包含标题的文本"""
        text = self.processor.extract_chunk_text(self.test_chunk, include_headers=True)
        self.assertIn('测试章节', text)
        self.assertIn('主标题', text)
        self.assertIn('子标题', text)
        self.assertIn('这是一个测试文本', text)
    
    def test_extract_chunk_text_without_headers(self):
        """测试提取不包含标题的文本"""
        text = self.processor.extract_chunk_text(self.test_chunk, include_headers=False)
        self.assertNotIn('测试章节', text)
        self.assertNotIn('主标题', text)
        self.assertNotIn('子标题', text)
        self.assertIn('这是一个测试文本', text)
    
    def test_analyze_chunk_text(self):
        """测试分析chunk文本"""
        analysis = self.processor.analyze_chunk_text(self.test_chunk)
        
        self.assertIn('text', analysis)
        self.assertIn('text_length', analysis)
        self.assertIn('word_count', analysis)
        self.assertIn('token_count', analysis)
        self.assertIn('token_diversity', analysis)
        self.assertIn('tokens', analysis)
        
        self.assertGreater(analysis['text_length'], 0)
        self.assertGreater(analysis['token_count'], 0)
        self.assertGreaterEqual(analysis['token_diversity'], 0)
        self.assertLessEqual(analysis['token_diversity'], 1)
    
    def test_is_empty_sparse_vector(self):
        """测试判断空sparse向量"""
        # 测试空向量
        empty_vector = {'indices': [], 'values': []}
        self.assertTrue(self.processor.is_empty_sparse_vector(empty_vector))
        
        # 测试None
        self.assertTrue(self.processor.is_empty_sparse_vector(None))
        
        # 测试非空向量
        non_empty_vector = {'indices': [1, 2, 3], 'values': [0.1, 0.2, 0.3]}
        self.assertFalse(self.processor.is_empty_sparse_vector(non_empty_vector))
    
    def test_get_chunk_id(self):
        """测试生成chunk ID"""
        chunk_id = self.processor.get_chunk_id(self.test_chunk, 5)
        self.assertEqual(chunk_id, "test.md-5")
    
    def test_analyze_empty_sparse_reason(self):
        """测试分析空sparse向量原因"""
        text_analysis = {
            'tokens': ['这', '是', '一个', '测试', '文本'],
            'text': '这是一个测试文本'
        }
        vocabulary = {'这': 1, '是': 1, '一个': 1}
        
        reason = self.processor.analyze_empty_sparse_reason(
            self.test_chunk, text_analysis, vocabulary
        )
        
        self.assertIsInstance(reason, str)
        self.assertIn('覆盖率', reason)
    
    def test_calculate_sparse_stats(self):
        """测试计算sparse向量统计"""
        sparse_vectors = [
            {'indices': [1, 2, 3], 'values': [0.1, 0.2, 0.3]},
            {'indices': [2, 4], 'values': [0.4, 0.5]},
            {'indices': [], 'values': []}
        ]
        
        stats = self.processor.calculate_sparse_stats(sparse_vectors)
        
        self.assertIn('avg_indices_count', stats)
        self.assertIn('max_indices_count', stats)
        self.assertIn('min_indices_count', stats)
        self.assertIn('avg_weight', stats)
        self.assertIn('weight_distribution', stats)
        
        self.assertEqual(stats['max_indices_count'], 3)
        self.assertEqual(stats['min_indices_count'], 0)

class TestBM25Manager(unittest.TestCase):
    """测试BM25管理器"""
    
    def setUp(self):
        self.config = SparseAnalyzerConfig()
        self.bm25_manager = BM25Manager(self.config)
    
    @patch('subprocess.run')
    def test_generate_dictionary(self, mock_run):
        """测试生成词典"""
        # 模拟subprocess.run返回成功
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
                # 模拟文件系统
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['bm25_vocab_2507060757.json']

            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True

                dict_file = self.bm25_manager.generate_dictionary(min_freq=3, max_vocab_size=15000)

                # 检查文件名格式：bm25_vocab_YYMMDDHHMM.json
                self.assertIn('bm25_vocab_', dict_file)
                self.assertTrue(dict_file.endswith('.json'))
                mock_run.assert_called_once()
    
    def test_load_dictionary(self):
        """测试加载词典"""
        # 创建临时词典文件
        test_vocabulary = {'测试': 5, '文本': 3, '分析': 2}
        test_dict_data = {
            'vocabulary': test_vocabulary,
            'metadata': {'created_at': '2024-12-01'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_dict_data, f, ensure_ascii=False)
            dict_file_path = f.name
        
        try:
            dict_data = self.bm25_manager.load_dictionary(dict_file_path)
            
            self.assertEqual(dict_data['vocabulary'], test_vocabulary)
            self.assertEqual(self.bm25_manager.vocabulary, test_vocabulary)
        finally:
            os.unlink(dict_file_path)
    
    def test_get_vocabulary_stats(self):
        """测试获取词汇表统计"""
        # 设置测试词汇表
        self.bm25_manager.vocabulary = {'测试': 5, '文本': 3, '分析': 2}
        
        stats = self.bm25_manager.get_vocabulary_stats()
        
        self.assertEqual(stats['vocabulary_size'], 3)
        self.assertEqual(stats['avg_frequency'], 10/3)
        self.assertEqual(stats['max_frequency'], 5)
        self.assertEqual(stats['min_frequency'], 2)
    
    def test_analyze_vocabulary_coverage(self):
        """测试分析词汇表覆盖率"""
        # 设置测试词汇表
        self.bm25_manager.vocabulary = {'测试': 5, '文本': 3, '分析': 2}
        
        texts = ['这是一个测试文本', '另一个分析文档']
        
        coverage = self.bm25_manager.analyze_vocabulary_coverage(texts)
        
        self.assertIn('total_tokens', coverage)
        self.assertIn('unique_tokens', coverage)
        self.assertIn('covered_tokens', coverage)
        self.assertIn('coverage_rate', coverage)
        self.assertIn('missing_tokens', coverage)
        
        self.assertGreaterEqual(coverage['coverage_rate'], 0)
        self.assertLessEqual(coverage['coverage_rate'], 1)

class TestSparseAnalyzer(unittest.TestCase):
    """测试Sparse分析器"""
    
    def setUp(self):
        self.config = SparseAnalyzerConfig()
        self.analyzer = SparseAnalyzer(self.config)
        
        # 创建测试chunk数据
        self.test_chunks = [
            {
                'text': '这是第一个测试chunk，包含一些重要信息。',
                'section_heading': '第一章',
                'h1': '主标题1',
                'source_file': 'test1.md'
            },
            {
                'text': '这是第二个测试chunk，内容较少。',
                'section_heading': '第二章',
                'h1': '主标题2',
                'source_file': 'test2.md'
            }
        ]
    
    def test_create_empty_report(self):
        """测试创建空报告"""
        report = self.analyzer._create_empty_report()
        
        self.assertIn('report_info', report)
        self.assertIn('summary', report)
        self.assertIn('detailed_analysis', report)
        self.assertIn('recommendations', report)
        
        self.assertEqual(report['summary']['total_chunks'], 0)
        self.assertEqual(report['summary']['empty_sparse_count'], 0)
    
    def test_generate_recommendations(self):
        """测试生成改进建议"""
        # 测试正常情况
        normal_result = {
            'empty_sparse_rate': 0.05,
            'vocab_coverage': {'coverage_rate': 0.8},
            'vocab_stats': {'vocabulary_size': 5000}
        }
        
        recommendations = self.analyzer._generate_recommendations(normal_result)
        self.assertIn("当前配置表现良好", recommendations[0])
        
        # 测试需要调整的情况
        bad_result = {
            'empty_sparse_rate': 0.2,
            'vocab_coverage': {'coverage_rate': 0.3},
            'vocab_stats': {'vocabulary_size': 500}
        }
        
        recommendations = self.analyzer._generate_recommendations(bad_result)
        self.assertGreater(len(recommendations), 1)
    
    def test_print_summary(self):
        """测试打印摘要"""
        report = {
            'summary': {
                'total_chunks': 100,
                'empty_sparse_count': 10,
                'empty_sparse_rate': 0.1,
                'vocabulary_size': 5000,
                'vocab_coverage_rate': 0.8
            },
            'recommendations': ['建议1', '建议2']
        }
        
        # 测试不会抛出异常
        try:
            self.analyzer.print_summary(report)
        except Exception as e:
            self.fail(f"print_summary抛出异常: {e}")

if __name__ == '__main__':
    unittest.main() 