#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS评估集成测试
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch
import tempfile
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
from test.base_test import TestBase
from evaltool.ragas_evaluator import run_ragas_evaluation

class TestRAGASEvaluationIntegration(TestBase):
    """RAGAS评估集成测试"""
    def setUp(self):
        super().setUp()
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.run.return_value = {
            'answers': ['集成检索上下文1', '集成检索上下文2'],
            'final_answer': '集成生成的答案'
        }
        self.mock_embedding_model = Mock()
        self.mock_embedding_model.embed_query.return_value = [0.1] * 1024
        self.mock_evaluate_patcher = patch('evaltool.ragas_evaluator.evaluate')
        self.mock_evaluate = self.mock_evaluate_patcher.start()
        self.mock_config_patcher = patch('evaltool.ragas_evaluator.get_embedding_model')
        self.mock_get_embedding_model = self.mock_config_patcher.start()
        self.mock_get_embedding_model.return_value = self.mock_embedding_model
        self.mock_ragas_embeddings_patcher = patch('evaltool.ragas_evaluator.LangchainEmbeddingsWrapper')
        self.mock_ragas_embeddings_class = self.mock_ragas_embeddings_patcher.start()
        self.mock_ragas_embeddings = Mock()
        self.mock_ragas_embeddings.embed_query.return_value = [0.1] * 1024
        self.mock_ragas_embeddings_class.return_value = self.mock_ragas_embeddings
        # 创建临时query_file和gold_file
        self.query_file = os.path.join(self.temp_dir, 'queries.csv')
        self.gold_file = os.path.join(self.temp_dir, 'answers.csv')
        with open(self.query_file, 'w', encoding='utf-8') as f:
            f.write('query\n')
            f.write('什么是战锤40K？\n')
            f.write('帝国卫队有哪些单位？\n')
        with open(self.gold_file, 'w', encoding='utf-8') as f:
            f.write('query,answer\n')
            f.write('什么是战锤40K？,战锤40K是一个科幻桌面游戏。\n')
            f.write('帝国卫队有哪些单位？,帝国卫队包括步兵、坦克等单位。\n')
    def tearDown(self):
        self.mock_evaluate_patcher.stop()
        self.mock_config_patcher.stop()
        self.mock_ragas_embeddings_patcher.stop()
        super().tearDown()
    def test_run_ragas_evaluation_success(self):
        with patch('evaltool.ragas_evaluator.RAGOrchestrator', return_value=self.mock_orchestrator):
            result = run_ragas_evaluation(
                query_file=self.query_file,
                gold_file=self.gold_file,
                top_k=2,
                output_dir=self.temp_dir
            )
            self.assertIn('metrics', result)
            self.assertIn('report_path', result)
            self.assertIn('details_path', result)
            self.assertIn('sample_count', result)
            self.assertIn('evaluation_time', result)
            metrics = result['metrics']
            self.assertAlmostEqual(metrics['context_relevance'], 0.85, places=2)
            self.assertAlmostEqual(metrics['context_recall'], 0.78, places=2)
            self.assertAlmostEqual(metrics['faithfulness'], 0.92, places=2)
            self.assertAlmostEqual(metrics['answer_relevancy'], 0.88, places=2)
            self.assertAlmostEqual(metrics['answer_correctness'], 0.82, places=2)
            self.assertTrue(os.path.exists(result['report_path']))
            self.assertTrue(os.path.exists(result['details_path']))
            self.assertEqual(result['sample_count'], 2)
    def test_run_ragas_evaluation_with_mismatched_data(self):
        with open(self.gold_file, 'w', encoding='utf-8') as f:
            f.write('query,answer\n')
            f.write('什么是战锤40K？,战锤40K是一个科幻桌面游戏。\n')
        with self.assertRaises(ValueError) as context:
            run_ragas_evaluation(
                query_file=self.query_file,
                gold_file=self.gold_file,
                top_k=2,
                output_dir=self.temp_dir
            )
        self.assertIn('不匹配', str(context.exception))
    def test_run_ragas_evaluation_with_invalid_files(self):
        with self.assertRaises(FileNotFoundError):
            run_ragas_evaluation(
                query_file='nonexistent.csv',
                gold_file=self.gold_file,
                top_k=2,
                output_dir=self.temp_dir
            )
    def test_factory_assemble_success(self):
        """测试run_ragas_evaluation通过工厂方法组装RAGOrchestrator"""
        with patch('evaltool.ragas_evaluator.QueryProcessorFactory') as mock_qpf, \
             patch('evaltool.ragas_evaluator.SearchEngineFactory') as mock_sef, \
             patch('evaltool.ragas_evaluator.AggFactory') as mock_aggf, \
             patch('evaltool.ragas_evaluator.PostSearchFactory') as mock_psf, \
             patch('evaltool.ragas_evaluator.RAGOrchestrator') as mock_orch:
            # mock工厂方法返回对象
            mock_qp = Mock()
            mock_se = Mock()
            mock_agg = Mock()
            mock_ps = Mock()
            mock_qpf.create_processor.return_value = mock_qp
            mock_sef.create.return_value = mock_se
            mock_aggf.get_aggregator.return_value = mock_agg
            mock_psf.create_processor.return_value = mock_ps
            mock_orch.return_value = self.mock_orchestrator
            # 调用run_ragas_evaluation
            result = run_ragas_evaluation(
                query_file=self.query_file,
                gold_file=self.gold_file,
                top_k=2,
                output_dir=self.temp_dir
            )
            # 断言工厂方法被正确调用
            mock_qpf.create_processor.assert_called()
            mock_sef.create.assert_called()
            mock_aggf.get_aggregator.assert_called()
            mock_psf.create_processor.assert_called()
            mock_orch.assert_called_with(mock_qp, mock_se, mock_agg, mock_ps)
            self.assertIn('metrics', result)
            self.assertIn('report_path', result)
            self.assertIn('details_path', result)
            self.assertIn('sample_count', result)
            self.assertIn('evaluation_time', result)
    def test_factory_assemble_invalid_type(self):
        """测试run_ragas_evaluation工厂方法异常分支"""
        with patch('evaltool.ragas_evaluator.QueryProcessorFactory') as mock_qpf, \
             patch('evaltool.ragas_evaluator.SearchEngineFactory') as mock_sef, \
             patch('evaltool.ragas_evaluator.AggFactory') as mock_aggf:
            mock_qpf.create_processor.side_effect = ValueError('不支持的query processor类型')
            with self.assertRaises(ValueError):
                run_ragas_evaluation(
                    query_file=self.query_file,
                    gold_file=self.gold_file,
                    top_k=2,
                    output_dir=self.temp_dir
                )

if __name__ == '__main__':
    unittest.main() 