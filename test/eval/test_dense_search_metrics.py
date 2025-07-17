#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索评估体系单元测试
测试JSON配置词典驱动的评估体系，包括参数解析、配置加载、评估执行、报告生成等
"""

import unittest
import tempfile
import json
import os
import sys
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入base_test确保安全删除
from test.base_test import TestBase

# 导入被测试模块
from evaltool.evalsearch.metrics import DenseRetrievalMetrics, HybridRecallMetrics
from evaltool.evalsearch.factory import SearchEngineFactory, process_queries_with_factory, build_retrieved_dict
from evaltool.evalsearch.utils import (
    parse_args, load_queries, load_gold, create_eval_dir, create_empty_report,
    append_evaluation_result, append_error, save_report, save_input_files
)
from evaltool.evalsearch.config_loader import EvaluatorConfigLoader
from evaltool.evalsearch.visualization import (
    plot_score_distribution, plot_embedding_distribution, plot_recall_distribution,
    plot_overlap_stats, plot_metrics_comparison, create_visualization_summary
)


class TestDenseRetrievalMetrics(TestBase):
    """测试DenseRetrievalMetrics类"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        self.gold = {
            "q1": ["doc_001", "doc_017", "doc_023"],
            "q2": ["doc_045", "doc_052"]
        }
        self.retrieved = {
            "q1": ["doc_017", "doc_099", "doc_001"],
            "q2": ["doc_052", "doc_045", "doc_100"]
        }
        self.k = 5
        self.metrics = DenseRetrievalMetrics(self.gold, self.retrieved, self.k)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.metrics.gold, self.gold)
        self.assertEqual(self.metrics.retrieved, self.retrieved)
        self.assertEqual(self.metrics.k, self.k)
    
    def test_recall_at_k(self):
        """测试Recall@K指标计算逻辑"""
        # 测试正常召回计算逻辑
        recall = DenseRetrievalMetrics.recall_at_k(self.gold, self.retrieved, k=3)
        
        # 测试返回值类型和范围逻辑
        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
    
    def test_recall_at_k_partial(self):
        """测试部分召回的计算逻辑"""
        partial_retrieved = {
            "q1": ["doc_999", "doc_888"],  # 都不在gold中
            "q2": ["doc_052", "doc_045"]   # 都在gold中
        }
        recall = DenseRetrievalMetrics.recall_at_k(self.gold, partial_retrieved, k=2)
        
        # 测试部分召回的计算逻辑
        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
    
    def test_mrr(self):
        """测试MRR指标计算逻辑"""
        mrr = DenseRetrievalMetrics.mrr(self.gold, self.retrieved, k=3)
        
        # 测试MRR计算逻辑
        self.assertIsInstance(mrr, float)
        self.assertGreaterEqual(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)
    
    def test_precision_at_k(self):
        """测试Precision@K指标计算逻辑"""
        precision = DenseRetrievalMetrics.precision_at_k(self.gold, self.retrieved, k=3)
        
        # 测试Precision计算逻辑
        self.assertIsInstance(precision, float)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
    
    def test_score_distribution(self):
        """测试分数分布统计逻辑"""
        score_list = [0.98, 0.85, 0.77, 0.92, 0.68]
        result = DenseRetrievalMetrics.score_distribution(score_list, bins=3)
        
        # 测试返回结构完整性
        self.assertIn('hist', result)
        self.assertIn('bin_edges', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('min', result)
        self.assertIn('max', result)
        
        # 测试数据结构逻辑
        self.assertEqual(len(result['hist']), 3)
        self.assertEqual(len(result['bin_edges']), 4)  # bins+1
        self.assertIsInstance(result['hist'], list)
        self.assertIsInstance(result['bin_edges'], list)
    
    def test_score_distribution_empty_list(self):
        """测试空分数列表的分布统计逻辑"""
        result = DenseRetrievalMetrics.score_distribution([], bins=5)
        
        # 测试空数据兜底逻辑
        self.assertIn('hist', result)
        self.assertIn('bin_edges', result)
        self.assertEqual(len(result['hist']), 5)
        
        # 空数据时只返回hist和bin_edges，不返回统计值
        self.assertNotIn('mean', result)
        self.assertNotIn('std', result)
        self.assertNotIn('min', result)
        self.assertNotIn('max', result)
    
    def test_score_distribution_low_score_queries(self):
        """测试分数分布统计能正确标记低分query"""
        # 构造5个query及其分数
        queries = [f"query_{i}" for i in range(5)]
        scores = [0.95, 0.92, 0.60, 0.55, 0.50]  # 0.60, 0.55, 0.50为低分
        # 期望均值和std
        import numpy as np
        mean = np.mean(scores)
        std = np.std(scores)
        # 调用新版score_distribution
        result = DenseRetrievalMetrics.score_distribution(scores, queries=queries, bins=3)
        self.assertIn('low_score_queries', result)
        # 检查低分query是否被正确标记
        low_score = mean - std
        expected = [(q, s) for q, s in zip(queries, scores) if s < low_score]
        actual = [(item['query'], item['score']) for item in result['low_score_queries']]
        self.assertEqual(set(expected), set(actual))
    
    def test_score_distribution_low_score_queries_percentile(self):
        """测试低分query标记：百分位+兜底逻辑"""
        # case 1: 只有一条query
        queries = ["query_0"]
        scores = [0.88]
        result = DenseRetrievalMetrics.score_distribution(scores, queries=queries, bins=3)
        self.assertEqual(result['low_score_queries'], [{'query': 'query_0', 'score': 0.88}])

        # case 2: 多条query，左侧20%分位
        queries = [f"query_{i}" for i in range(10)]
        scores = [0.95, 0.92, 0.91, 0.93, 0.94, 0.60, 0.55, 0.50, 0.89, 0.88]
        result = DenseRetrievalMetrics.score_distribution(scores, queries=queries, bins=3)
        # 20%分位数应覆盖分数最低的两条
        expected = [
            {'query': 'query_7', 'score': 0.50},
            {'query': 'query_6', 'score': 0.55}
        ]
        actual = [q for q in result['low_score_queries'] if q['score'] <= np.percentile(scores, 20)]
        self.assertTrue(all(q in actual for q in expected))
        self.assertTrue(len(result['low_score_queries']) >= 2)

        # case 3: query过少，20%分位取不到，回退为score-1std最大一条
        queries = ["query_0", "query_1"]
        scores = [0.95, 0.60]
        result = DenseRetrievalMetrics.score_distribution(scores, queries=queries, bins=3)
        # 断言应为fallback结果
        self.assertEqual(result['low_score_queries'], [{'query': 'query_1', 'score': 0.6}])
    
    def test_embedding_distribution_method_selection(self):
        """测试embedding分布方法选择逻辑"""
        import numpy as np
        embeddings = np.random.rand(50, 5)  # 50个样本，确保perplexity(30) < n_samples(50)
        
        # 测试tsne方法选择逻辑
        with patch('evaltool.evalsearch.metrics.TSNE') as mock_tsne:
            mock_tsne.return_value.fit_transform.return_value = np.random.rand(50, 2)
            result = DenseRetrievalMetrics.embedding_distribution(embeddings, method="tsne")
            
            # 验证方法调用逻辑
            mock_tsne.assert_called_once()
            self.assertIsInstance(result, np.ndarray)
    
    # def test_embedding_distribution_small_sample_exception(self):
    #     """测试小样本时的异常处理逻辑"""
    #     import numpy as np
    #     embeddings = np.random.rand(10, 5)  # 10个样本，perplexity(30) > n_samples(10)
    #     # 测试小样本时应该抛出ValueError
    #     with self.assertRaises(ValueError) as context:
    #         DenseRetrievalMetrics.embedding_distribution(embeddings, method="tsne")
    #     # 验证错误信息包含perplexity相关信息
    #     error_msg = str(context.exception)
    #     self.assertIn("perplexity", error_msg)
    #     self.assertIn("n_samples", error_msg)
    
    def test_embedding_distribution_unknown_method(self):
        """测试未知的embedding降维方法处理逻辑"""
        import numpy as np
        embeddings = np.random.rand(5, 3)
        
        # 测试异常处理逻辑
        with self.assertRaises(ValueError) as context:
            DenseRetrievalMetrics.embedding_distribution(embeddings, method="unknown")
        
        # 验证错误信息包含支持的方法
        self.assertIn("tsne", str(context.exception))
        self.assertIn("umap", str(context.exception))

    def test_embedding_distribution_too_few_samples(self):
        """测试embedding分布样本数过少时的异常和主流程提示"""
        import numpy as np
        from evaltool.evalsearch.metrics import DenseRetrievalMetrics
        embeddings = np.random.rand(1, 5)  # 只有1个样本
        with self.assertRaises(ValueError) as context:
            DenseRetrievalMetrics.embedding_distribution(embeddings, method="tsne")
        self.assertIn("样本数过少", str(context.exception))

    def test_score_distribution_integration_with_queries(self):
        """集成测试：主流程传入queries参数，报告应包含low_score_queries和最低分query"""
        import numpy as np
        queries = [f"query_{i}" for i in range(5)]
        scores = [0.95, 0.92, 0.60, 0.55, 0.50]  # 0.60, 0.55, 0.50为低分
        # 有低分query时
        result = DenseRetrievalMetrics.score_distribution(scores, queries=queries, bins=3)
        self.assertIn('low_score_queries', result)
        self.assertTrue(len(result['low_score_queries']) > 0)
        # 全部高分时
        high_scores = [0.95, 0.92, 0.91, 0.93, 0.94]
        result2 = DenseRetrievalMetrics.score_distribution(high_scores, queries=queries, bins=3)
        self.assertIn('low_score_queries', result2)
        # integration case: 严格按20%分位和兜底逻辑断言
        perc_20 = np.percentile(high_scores, 20)
        expected = [
            {'query': q, 'score': s}
            for q, s in zip(queries, high_scores) if s <= perc_20
        ]
        if expected:
            self.assertEqual(result2['low_score_queries'], expected)
        else:
            mean = np.mean(high_scores)
            std = np.std(high_scores)
            low_score = mean - std
            fallback = max(
                [{'query': q, 'score': s} for q, s in zip(queries, high_scores) if s < low_score],
                key=lambda x: x['score'], default=None
            )
            if fallback:
                self.assertEqual(result2['low_score_queries'], [fallback])
            else:
                self.assertEqual(result2['low_score_queries'], [])
        # 单条query时应始终包含该条
        single_query = ["query_0"]
        single_score = [0.99]
        result3 = DenseRetrievalMetrics.score_distribution(single_score, queries=single_query, bins=3)
        self.assertEqual(result3['low_score_queries'], [{'query': 'query_0', 'score': 0.99}])
        self.assertIn('queries', result2)
        self.assertIn('scores', result2)
        self.assertEqual(result2['queries'], queries)
        self.assertEqual(result2['scores'], high_scores)


class TestHybridRecallMetrics(TestBase):
    """测试HybridRecallMetrics类"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        self.dense = {
            "q1": ["doc_001", "doc_017", "doc_023"],
            "q2": ["doc_045", "doc_052"]
        }
        self.sparse = {
            "q1": ["doc_017", "doc_099", "doc_001"],
            "q2": ["doc_052", "doc_045", "doc_100"]
        }
        self.graph = {
            "q1": ["doc_001", "doc_017"],
            "q2": ["doc_045"]
        }
        self.k = 10
        self.metrics = HybridRecallMetrics(self.dense, self.sparse, self.graph, self.k)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.metrics.dense, self.dense)
        self.assertEqual(self.metrics.sparse, self.sparse)
        self.assertEqual(self.metrics.graph, self.graph)
        self.assertEqual(self.metrics.k, self.k)
    
    def test_overlap_stats(self):
        """测试重叠统计计算逻辑"""
        result = self.metrics.overlap_stats()
        
        # 测试返回结构逻辑
        self.assertIn('q1', result)
        self.assertIn('q2', result)
        self.assertIn('overlap', result['q1'])
        self.assertIn('union', result['q1'])
        
        # 测试数据类型逻辑
        self.assertIsInstance(result['q1']['overlap'], int)
        self.assertIsInstance(result['q1']['union'], int)
        
        # 测试数值范围逻辑
        self.assertGreaterEqual(result['q1']['overlap'], 0)
        self.assertGreaterEqual(result['q1']['union'], 0)
        self.assertLessEqual(result['q1']['overlap'], result['q1']['union'])
    
    def test_hybrid_union_recall(self):
        """测试多路召回并集覆盖率计算逻辑"""
        gold = {
            "q1": ["doc_001", "doc_017", "doc_999"],
            "q2": ["doc_045", "doc_052", "doc_888"]
        }
        
        union_recall = HybridRecallMetrics.hybrid_union_recall(
            gold, self.dense, self.sparse, self.graph, k=3
        )
        
        # 测试并集覆盖率计算逻辑
        self.assertIsInstance(union_recall, float)
        self.assertGreaterEqual(union_recall, 0.0)
        self.assertLessEqual(union_recall, 1.0)


class TestConfigLoader(TestBase):
    """测试配置加载器"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        self.config_data = {
            "evaluators": {
                "score_distribution": {
                    "name": "score_distribution",
                    "description": "计算检索分数分布统计",
                    "requires_gold": False,
                    "conditions": {"always_execute": True}
                },
                "recall_at_k": {
                    "name": "recall_at_k",
                    "description": "计算Recall@K指标",
                    "requires_gold": True,
                    "conditions": {"requires_gold_file": True}
                }
            }
        }
        
        # 创建临时配置文件
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f)
        
        self.config_loader = EvaluatorConfigLoader(self.config_file)
    
    def test_load_config(self):
        """测试配置文件加载"""
        self.assertEqual(self.config_loader.config, self.config_data)
    
    def test_get_evaluator_config(self):
        """测试获取评估器配置"""
        config = self.config_loader.get_evaluator_config("score_distribution")
        self.assertIsNotNone(config)
        self.assertEqual(config["name"], "score_distribution")
        self.assertEqual(config["description"], "计算检索分数分布统计")
    
    def test_get_all_evaluators(self):
        """测试获取所有评估器"""
        evaluators = self.config_loader.get_all_evaluators()
        self.assertEqual(len(evaluators), 2)
        self.assertIn("score_distribution", evaluators)
        self.assertIn("recall_at_k", evaluators)
    
    def test_can_evaluate_always_execute(self):
        """测试总是执行的评估器"""
        class MockArgs:
            def __init__(self):
                self.gold = None
                self.se = "dense"
        
        args = MockArgs()
        can_evaluate = self.config_loader.can_evaluate("score_distribution", args)
        self.assertTrue(can_evaluate)
    
    def test_can_evaluate_requires_gold(self):
        """测试需要gold的评估器"""
        class MockArgs:
            def __init__(self, has_gold=True):
                self.gold = "gold.json" if has_gold else None
                self.se = "dense"
        
        # 有gold文件
        args_with_gold = MockArgs(has_gold=True)
        with patch('os.path.exists', return_value=True):
            can_evaluate = self.config_loader.can_evaluate("recall_at_k", args_with_gold)
            self.assertTrue(can_evaluate)
        
        # 无gold文件
        args_without_gold = MockArgs(has_gold=False)
        can_evaluate = self.config_loader.can_evaluate("recall_at_k", args_without_gold)
        self.assertFalse(can_evaluate)
    
    def test_get_available_evaluators(self):
        """测试获取可执行的评估器"""
        class MockArgs:
            def __init__(self):
                self.gold = "gold.json"
                self.se = "dense"
        
        args = MockArgs()
        with patch('os.path.exists', return_value=True):
            available = self.config_loader.get_available_evaluators(args)
            self.assertEqual(len(available), 2)  # 两个都可以执行


class TestUtilityFunctions(TestBase):
    """测试工具函数"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        
        # 创建测试文件
        self.query_file = os.path.join(self.temp_dir, "queries.json")
        self.gold_file = os.path.join(self.temp_dir, "gold.json")
        
        # 写入测试数据
        queries_data = ["query1", "query2", "query3"]
        with open(self.query_file, 'w', encoding='utf-8') as f:
            json.dump(queries_data, f)
        
        gold_data = {
            "q1": ["doc_001", "doc_002"],
            "q2": ["doc_003", "doc_004"]
        }
        with open(self.gold_file, 'w', encoding='utf-8') as f:
            json.dump(gold_data, f)
    
    def test_load_queries(self):
        """测试加载query文件"""
        queries = load_queries(self.query_file)
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0], "query1")
    
    def test_load_queries_dict_format(self):
        """测试字典格式的query文件"""
        dict_queries = {"q1": "query1", "q2": "query2"}
        dict_file = os.path.join(self.temp_dir, "dict_queries.json")
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dict_queries, f)
        
        queries = load_queries(dict_file)
        self.assertEqual(len(queries), 2)
        self.assertIn("query1", queries)
        self.assertIn("query2", queries)
    
    def test_load_gold(self):
        """测试加载gold文件"""
        gold = load_gold(self.gold_file)
        self.assertEqual(len(gold), 2)
        self.assertIn("q1", gold)
        self.assertIn("q2", gold)
        self.assertEqual(gold["q1"], ["doc_001", "doc_002"])
    
    def test_create_eval_dir(self):
        """测试创建评估目录逻辑"""
        eval_dir, report_path = create_eval_dir()
        # 验证目录路径格式逻辑
        import re
        pattern = r"evaltool/dense_search/\d{8}_\d{6}$"
        self.assertRegex(eval_dir, pattern)
        # 验证报告路径逻辑
        self.assertIn("report.json", report_path)
        self.assertTrue(report_path.startswith(eval_dir))
    
    def test_create_empty_report(self):
        """测试创建空白报告"""
        class MockArgs:
            def __init__(self):
                self.query = "queries.json"
                self.gold = "gold.json"
                self.topk = 10
                self.qp = "straightforward"
                self.se = "dense"
        
        args = MockArgs()
        eval_items = ["score_distribution", "recall_at_k"]
        
        report = create_empty_report(args, eval_items)
        
        self.assertIn("report_info", report)
        self.assertIn("configuration", report)
        self.assertIn("evaluation_items", report)
        self.assertIn("evaluation_results", report)
        self.assertEqual(report["evaluation_items"], eval_items)
    
    def test_append_evaluation_result(self):
        """测试追加评估结果"""
        report = {
            "evaluation_results": {},
            "visualizations": {}
        }
        
        result = {"score": 0.85}
        viz_path = "score_distribution.png"
        
        append_evaluation_result(report, "score_distribution", result, viz_path)
        
        self.assertIn("score_distribution", report["evaluation_results"])
        self.assertIn("score_distribution", report["visualizations"])
        self.assertEqual(report["evaluation_results"]["score_distribution"], result)
    
    def test_append_error(self):
        """测试追加错误信息"""
        report = {"errors": {}}
        error_msg = "评估失败"
        
        append_error(report, "score_distribution", error_msg)
        
        self.assertIn("score_distribution", report["errors"])
        self.assertEqual(report["errors"]["score_distribution"], error_msg)
    
    def test_save_report(self):
        """测试保存报告"""
        report = {"test": "data"}
        report_path = os.path.join(self.temp_dir, "test_report.json")
        
        save_report(report, report_path)
        
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r', encoding='utf-8') as f:
            saved_report = json.load(f)
        self.assertEqual(saved_report, report)

    def test_load_queries_from_sentence(self):
        """测试直接传入一句自然语言query时的加载逻辑"""
        from evaltool.evalsearch.utils import load_queries
        query = "丑角剧团有哪些能力"
        queries = load_queries(query)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0], query)
        self.assertIsInstance(queries, list)


class TestFactoryFunctions(TestBase):
    """测试工厂函数"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        self.queries = ["query1", "query2"]
    
    def test_process_queries_with_factory_straightforward(self):
        """测试straightforward处理器"""
        result = process_queries_with_factory(self.queries, "straightforward")
        self.assertEqual(result, self.queries)
    
    def test_process_queries_with_factory_expand(self):
        """测试expand处理器"""
        result = process_queries_with_factory(self.queries, "expand")
        # expand处理器会为每个query生成3个版本
        self.assertEqual(len(result), 6)  # 2个query * 3个版本
    
    def test_process_queries_with_factory_cot(self):
        """测试cot处理器"""
        result = process_queries_with_factory(self.queries, "cot")
        # cot处理器会为每个query生成2个版本
        self.assertEqual(len(result), 4)  # 2个query * 2个版本
    
    def test_build_retrieved_dict(self):
        """测试构建retrieved字典"""
        mock_engine = Mock()
        mock_engine.search.return_value = [
            {"doc_id": "doc_001", "score": 0.9, "query_id": "q0"},
            {"doc_id": "doc_002", "score": 0.8, "query_id": "q0"},
            {"doc_id": "doc_003", "score": 0.7, "query_id": "q1"}
        ]
        top_k = 5
        
        retrieved, all_scores = build_retrieved_dict(self.queries, mock_engine, top_k)
        
        self.assertIn('q1', retrieved)
        self.assertIn('q2', retrieved)
        self.assertEqual(len(all_scores), 3)


class TestVisualizationFunctions(TestBase):
    """测试可视化函数"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        self.output_dir = self.temp_dir
    
    def test_plot_score_distribution(self):
        """测试分数分布可视化函数调用逻辑"""
        score_list = [0.8, 0.9, 0.7, 0.85, 0.95]
        output_path = os.path.join(self.output_dir, "score_dist.png")
        
        # 测试函数调用不抛出异常
        try:
            with patch('matplotlib.pyplot.savefig'):
                plot_score_distribution(score_list, output_path)
        except Exception as e:
            self.fail(f"plot_score_distribution抛出异常: {e}")
    
    def test_plot_score_distribution_empty(self):
        """测试空分数列表的可视化处理逻辑"""
        output_path = os.path.join(self.output_dir, "empty_score.png")
        
        # 测试空数据时的处理逻辑
        try:
            with patch('matplotlib.pyplot.savefig'):
                plot_score_distribution([], output_path)
        except Exception as e:
            self.fail(f"空数据可视化处理抛出异常: {e}")
    
    def test_plot_embedding_distribution(self):
        """测试embedding分布可视化函数调用逻辑"""
        import numpy as np
        embeddings = np.random.rand(10, 5)
        output_path = os.path.join(self.output_dir, "embedding.png")
        
        # 测试函数调用不抛出异常
        try:
            with patch('sklearn.manifold.TSNE') as mock_tsne, \
                 patch('matplotlib.pyplot.savefig'):
                mock_tsne.return_value.fit_transform.return_value = np.random.rand(10, 2)
                plot_embedding_distribution(embeddings, output_path)
        except Exception as e:
            self.fail(f"plot_embedding_distribution抛出异常: {e}")
    
    def test_create_visualization_summary(self):
        """测试创建可视化总结文件生成逻辑"""
        report = {
            "report_info": {"generated_at": "2024-12-01T12:00:00"},
            "configuration": {
                "query_file": "queries.json",
                "gold_file": "gold.json",
                "top_k": 10,
                "qp_type": "straightforward",
                "se_type": "dense"
            },
            "evaluation_results": {
                "score_distribution": {"mean": 0.85}
            },
            "visualizations": {
                "score_distribution": {
                    "file_path": "score_dist.png",
                    "description": "分数分布",
                    "generated_at": "2024-12-01T12:00:00"
                }
            }
        }
        
        # 测试文件生成逻辑
        summary_path = create_visualization_summary(self.output_dir, report)
        
        # 测试返回路径逻辑
        self.assertIsInstance(summary_path, str)
        self.assertTrue(summary_path.endswith('visualization_summary.md'))
        
        # 测试文件存在逻辑
        self.assertTrue(os.path.exists(summary_path))


class TestMainFunctionLogic(TestBase):
    """测试主函数逻辑"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        
        # 创建测试文件
        self.query_file = os.path.join(self.temp_dir, "queries.json")
        self.gold_file = os.path.join(self.temp_dir, "gold.json")
        
        queries_data = ["query1", "query2"]
        with open(self.query_file, 'w', encoding='utf-8') as f:
            json.dump(queries_data, f)
        
        gold_data = {"q1": ["doc_001"], "q2": ["doc_002"]}
        with open(self.gold_file, 'w', encoding='utf-8') as f:
            json.dump(gold_data, f)
    
    def test_parse_args(self):
        """测试参数解析"""
        test_args = [
            "--query", self.query_file,
            "--gold", self.gold_file,
            "--topk", "20",
            "--qp", "expand",
            "--se", "hybrid"
        ]
        
        with patch('sys.argv', ['test_main.py'] + test_args):
            args = parse_args()
            
            self.assertEqual(args.query, self.query_file)
            self.assertEqual(args.gold, self.gold_file)
            self.assertEqual(args.topk, 20)
            self.assertEqual(args.qp, "expand")
            self.assertEqual(args.se, "hybrid")
    
    def test_eval_items_decision_with_gold(self):
        """测试有gold时的评估项决策"""
        query_file = self.query_file
        gold_file = self.gold_file
        
        class MockArgs:
            def __init__(self):
                self.query = query_file
                self.gold = gold_file
                self.topk = 10
                self.qp = "straightforward"
                self.se = "dense"
        
        args = MockArgs()
        
        with patch('os.path.exists', return_value=True):
            from evaltool.evalsearch.config_loader import config_loader
            available_evaluators = config_loader.get_available_evaluators(args)
            
            # 应该有基础评估项和依赖gold的评估项
            self.assertIn("score_distribution", available_evaluators)
            self.assertIn("embedding_distribution", available_evaluators)
            self.assertIn("recall_at_k", available_evaluators)
            self.assertIn("mrr", available_evaluators)
            self.assertIn("precision_at_k", available_evaluators)
    
    def test_eval_items_decision_without_gold(self):
        """测试无gold时的评估项决策"""
        query_file = self.query_file
        
        class MockArgs:
            def __init__(self):
                self.query = query_file
                self.gold = None
                self.topk = 10
                self.qp = "straightforward"
                self.se = "dense"
        
        args = MockArgs()
        
        from evaltool.evalsearch.config_loader import config_loader
        available_evaluators = config_loader.get_available_evaluators(args)
        
        # 应该只有基础评估项
        self.assertIn("score_distribution", available_evaluators)
        self.assertIn("embedding_distribution", available_evaluators)
        self.assertNotIn("recall_at_k", available_evaluators)
        self.assertNotIn("mrr", available_evaluators)
    
    def test_eval_items_decision_hybrid_engine(self):
        """测试hybrid引擎时的评估项决策"""
        query_file = self.query_file
        
        class MockArgs:
            def __init__(self):
                self.query = query_file
                self.gold = None
                self.topk = 10
                self.qp = "straightforward"
                self.se = "hybrid"
        
        args = MockArgs()
        
        from evaltool.evalsearch.config_loader import config_loader
        available_evaluators = config_loader.get_available_evaluators(args)
        
        # 应该有基础评估项和hybrid评估项
        self.assertIn("score_distribution", available_evaluators)
        self.assertIn("embedding_distribution", available_evaluators)
        self.assertIn("overlap_stats", available_evaluators)


if __name__ == '__main__':
    unittest.main() 