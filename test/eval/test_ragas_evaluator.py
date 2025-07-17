#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS评估器单元测试
使用mock数据避免调用真实API，测试所有评估方法和指标计算逻辑
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )

# 导入base_test确保安全删除
from test.base_test import TestBase

from datasets import Dataset
from evaltool.ragas_evaluator import RAGASEvaluator


class TestRAGASEvaluator(TestBase):
    """RAGAS评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # Mock Orchestrator
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.run.return_value = {
            'answers': ['这是检索到的上下文1', '这是检索到的上下文2'],
            'final_answer': '这是生成的答案'
        }
        # Mock embedding model
        self.mock_embedding_model = Mock()
        self.mock_embedding_model.embed_query.return_value = [0.1] * 1024  # 返回1024维向量
        # 创建测试数据集
        self.test_data = {
            "question": ["什么是战锤40K？", "帝国卫队有哪些单位？"],
            "contexts": [
                ["战锤40K是一个科幻桌面游戏", "帝国卫队是人类的军队"],
                ["帝国卫队包括步兵、坦克等", "帝国卫队是战锤40K中的军队"]
            ],
            "answer": [
                "战锤40K是一个科幻桌面游戏，帝国卫队是人类的军队。",
                "帝国卫队包括步兵、坦克等单位，是战锤40K中的军队。"
            ],
            "ground_truth": [
                "战锤40K是一个科幻桌面游戏。",
                "帝国卫队包括步兵、坦克等单位。"
            ]
        }
        self.dataset = Dataset.from_dict(self.test_data)
        # Mock RAGAS evaluate函数
        self.mock_evaluate_patcher = patch('evaltool.ragas_evaluator.evaluate')
        self.mock_evaluate = self.mock_evaluate_patcher.start()
        # Mock config模块
        self.mock_config_patcher = patch('evaltool.ragas_evaluator.get_embedding_model')
        self.mock_get_embedding_model = self.mock_config_patcher.start()
        self.mock_get_embedding_model.return_value = self.mock_embedding_model
        # Mock RAGAS embedding包装器
        self.mock_ragas_embeddings_patcher = patch('evaltool.ragas_evaluator.LangchainEmbeddingsWrapper')
        self.mock_ragas_embeddings_class = self.mock_ragas_embeddings_patcher.start()
        self.mock_ragas_embeddings = Mock()
        self.mock_ragas_embeddings.embed_query.return_value = [0.1] * 1024
        self.mock_ragas_embeddings_class.return_value = self.mock_ragas_embeddings
        # 创建评估器实例
        self.evaluator = RAGASEvaluator(self.mock_orchestrator)
    
    def tearDown(self):
        self.mock_evaluate_patcher.stop()
        self.mock_config_patcher.stop()
        self.mock_ragas_embeddings_patcher.stop()
    
    def test_init(self):
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.orchestrator, self.mock_orchestrator)
        self.assertEqual(self.evaluator.embedding_model, self.mock_embedding_model)
    
    def test_prepare_dataset_success(self):
        queries = ["什么是战锤40K？", "帝国卫队有哪些单位？"]
        gold_answers = ["战锤40K是一个科幻桌面游戏。", "帝国卫队包括步兵、坦克等单位。"]
        dataset = self.evaluator.prepare_dataset(queries, gold_answers, top_k=2)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 2)
        self.assertIn('question', dataset.column_names)
        self.assertIn('contexts', dataset.column_names)
        self.assertIn('answer', dataset.column_names)
        self.assertIn('ground_truth', dataset.column_names)
        self.assertEqual(dataset[0]['question'], queries[0])
        self.assertEqual(dataset[1]['question'], queries[1])
        self.assertEqual(len(dataset[0]['contexts']), 2)
        self.assertEqual(len(dataset[1]['contexts']), 2)
        self.mock_orchestrator.run.assert_any_call(queries[0], top_k=2)
        self.mock_orchestrator.run.assert_any_call(queries[1], top_k=2)
    
    def test_prepare_dataset_with_search_error(self):
        self.mock_orchestrator.run.side_effect = Exception("检索失败")
        queries = ["测试查询"]
        gold_answers = ["测试答案"]
        dataset = self.evaluator.prepare_dataset(queries, gold_answers, top_k=2)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['contexts'], [])
        self.assertEqual(dataset[0]['answer'], "处理失败")
    
    def test_evaluate_retrieval_metrics(self):
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: {
            'nv_context_relevance': [0.85, 0.90, 0.80],
            'context_recall': [0.78, 0.82, 0.75]
        }.get(key, [0.0]))
        mock_results.__contains__ = Mock(side_effect=lambda key: key in ['nv_context_relevance', 'context_recall'])
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_retrieval_metrics(self.dataset)
        self.assertIn('context_relevance', results)
        self.assertIn('context_recall', results)
        self.assertAlmostEqual(results['context_relevance'], 0.85, places=2)
        self.assertAlmostEqual(results['context_recall'], 0.78, places=2)
        self.mock_evaluate.assert_called_once()
    
    def test_evaluate_generation_metrics(self):
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: {
            'faithfulness': [0.92, 0.88, 0.96],
            'answer_relevancy': [0.88, 0.85, 0.91],
            'answer_correctness': [0.82, 0.79, 0.85]
        }.get(key, [0.0]))
        mock_results.__contains__ = Mock(side_effect=lambda key: key in ['faithfulness', 'answer_relevancy', 'answer_correctness'])
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_generation_metrics(self.dataset)
        self.assertIn('faithfulness', results)
        self.assertIn('answer_relevancy', results)
        self.assertIn('answer_correctness', results)
        self.assertAlmostEqual(results['faithfulness'], 0.92, places=2)
        self.assertAlmostEqual(results['answer_relevancy'], 0.88, places=2)
        self.assertAlmostEqual(results['answer_correctness'], 0.82, places=2)
    
    def test_evaluate_all_metrics(self):
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: {
            'nv_context_relevance': [0.85, 0.90, 0.80],
            'context_recall': [0.78, 0.82, 0.75],
            'faithfulness': [0.92, 0.88, 0.96],
            'answer_relevancy': [0.88, 0.85, 0.91],
            'answer_correctness': [0.82, 0.79, 0.85],
            'context_precision': [0.83, 0.87, 0.80],
            'context_utilization': [0.89, 0.85, 0.92],
            'answer_similarity': [0.86, 0.82, 0.90],
            'bleu_score': [0.75, 0.78, 0.72],
            'rouge_score': [0.81, 0.85, 0.78],
            'exact_match': [0.65, 0.70, 0.60]
        }.get(key, [0.0]))
        mock_results.__contains__ = Mock(side_effect=lambda key: key in [
            'nv_context_relevance', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_correctness',
            'context_precision', 'context_utilization', 'answer_similarity', 'bleu_score', 'rouge_score', 'exact_match'
        ])
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_all_metrics(self.dataset)
        self.assertIn('context_relevance', results)
        self.assertIn('context_recall', results)
        self.assertIn('faithfulness', results)
        self.assertIn('answer_relevancy', results)
        self.assertIn('answer_correctness', results)
        self.assertIn('context_precision', results)
        self.assertIn('context_utilization', results)
        self.assertIn('answer_similarity', results)
        self.assertIn('bleu_score', results)
        self.assertIn('rouge_score', results)
        self.assertIn('exact_match', results)
    
    def test_evaluate_all_metrics_with_missing_keys(self):
        """测试指标评估时缺少某些key"""
        # Mock RAGAS评估结果，缺少某些指标
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: {
            'nv_context_relevance': 0.85,
            'context_recall': 0.78
            # 缺少其他指标
        }.get(key, 0.0))
        mock_results.__contains__ = Mock(side_effect=lambda key: key in ['nv_context_relevance', 'context_recall'])
        self.mock_evaluate.return_value = mock_results
        
        results = self.evaluator.evaluate_all_metrics(self.dataset)
        
        # 验证结果，缺少的指标应该为0.0
        self.assertEqual(results['context_relevance'], 0.85)
        self.assertEqual(results['context_recall'], 0.78)
        self.assertEqual(results['faithfulness'], 0.0)
        self.assertEqual(results['answer_relevancy'], 0.0)
        self.assertEqual(results['answer_correctness'], 0.0)
    
    def test_generate_detailed_report(self):
        """测试详细报告生成"""
        results = {
            'context_relevance': 0.85,
            'context_recall': 0.78,
            'faithfulness': 0.92,
            'answer_relevancy': 0.88,
            'answer_correctness': 0.82
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = self.evaluator.generate_detailed_report(self.dataset, results, temp_dir)
            
            # 验证报告文件存在
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(report_path.endswith('.md'))
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('RAGAS Dense向量检索评估报告', content)
                self.assertIn('0.8500', content)
                self.assertIn('0.7800', content)
                self.assertIn('0.9200', content)
    
    def test_save_dataset_details(self):
        """测试数据集详情保存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            details_path = self.evaluator.save_dataset_details(self.dataset, temp_dir)
            
            # 验证详情文件存在
            self.assertTrue(os.path.exists(details_path))
            self.assertTrue(details_path.endswith('.json'))
            
            # 验证详情内容
            with open(details_path, 'r', encoding='utf-8') as f:
                details = json.load(f)
                self.assertEqual(len(details), 2)
                self.assertIn('index', details[0])
                self.assertIn('question', details[0])
                self.assertIn('contexts', details[0])
                self.assertIn('answer', details[0])
                self.assertIn('ground_truth', details[0])
                self.assertIn('context_count', details[0])

    def test_evaluate_all_metrics_all_zero(self):
        """测试RAGAS评估返回全0时主流程不抛异常"""
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: 0.0)
        mock_results.__contains__ = Mock(return_value=True)
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_all_metrics(self.dataset)
        self.assertTrue(all(v == 0.0 for v in results.values()))
        # 生成报告不应抛异常
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.evaluator.generate_detailed_report(self.dataset, results, temp_dir)
            except Exception as e:
                self.fail(f"全0指标时报告生成抛异常: {e}")

    def test_evaluate_all_metrics_partial_zero(self):
        """测试RAGAS评估返回部分0时主流程不抛异常"""
        mock_results = Mock()
        mock_results.__getitem__ = Mock(side_effect=lambda key: {
            'nv_context_relevance': 0.0,
            'context_recall': 0.5,
            'faithfulness': 0.0,
            'answer_relevancy': 0.7,
            'answer_correctness': 0.0
        }.get(key, 0.0))
        mock_results.__contains__ = Mock(side_effect=lambda key: key in [
            'nv_context_relevance', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_correctness'
        ])
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_all_metrics(self.dataset)
        self.assertEqual(results['context_relevance'], 0.0)
        self.assertEqual(results['context_recall'], 0.5)
        self.assertEqual(results['faithfulness'], 0.0)
        self.assertEqual(results['answer_relevancy'], 0.7)
        self.assertEqual(results['answer_correctness'], 0.0)
        # 生成报告不应抛异常
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.evaluator.generate_detailed_report(self.dataset, results, temp_dir)
            except Exception as e:
                self.fail(f"部分0指标时报告生成抛异常: {e}")

    def test_evaluate_all_metrics_empty_result(self):
        """测试RAGAS评估返回空对象时主流程兜底"""
        mock_results = Mock()
        mock_results.__getitem__ = Mock(return_value=0.0)
        mock_results.__contains__ = Mock(return_value=False)
        self.mock_evaluate.return_value = mock_results
        results = self.evaluator.evaluate_all_metrics(self.dataset)
        self.assertTrue(all(v == 0.0 for v in results.values()))
        # 生成报告不应抛异常
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.evaluator.generate_detailed_report(self.dataset, results, temp_dir)
            except Exception as e:
                self.fail(f"空对象时报告生成抛异常: {e}") 