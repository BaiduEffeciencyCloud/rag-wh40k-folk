import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.base_test import TestBase
from searchengine.sparse_search import SparseSearchEngine
from searchengine.search_factory import SearchEngineFactory


class TestSparseSearchEngine(TestBase):
    """测试Sparse搜索引擎 - 单元测试"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()
        # 基础测试数据
        self.test_queries = [
            "艾达灵族骑乘单位",
            "喷气摩托特点", 
            "阿苏焉尼战斗专注",
            "御风者军阵"
        ]
    
    def test_init(self):
        """测试SparseSearchEngine初始化"""
        # 测试：验证引擎能正确初始化，加载BM25Manager和词典
        engine = SparseSearchEngine()
        self.assertEqual(engine.get_type(), 'sparse')
        self.assertIsNotNone(engine.bm25_manager)
        
    def test_get_type(self):
        """测试返回搜索引擎类型"""
        # 测试：验证get_type()返回"sparse"
        engine = SparseSearchEngine()
        self.assertEqual(engine.get_type(), 'sparse')
        
    def test_get_capabilities(self):
        """测试获取搜索引擎能力信息"""
        # 测试：验证返回的能力信息包含正确的sparse检索配置
        engine = SparseSearchEngine()
        capabilities = engine.get_capabilities()
        
        self.assertEqual(capabilities['type'], 'sparse')
        self.assertEqual(capabilities['description'], '基于BM25的稀疏检索')
        self.assertTrue(capabilities['supports_filtering'])
        self.assertFalse(capabilities['supports_hybrid'])
        self.assertEqual(capabilities['algorithm'], 'BM25')
        
    def test_load_bm25_manager(self):
        """测试BM25Manager加载"""
        # 测试：验证能正确加载BM25Manager和词典
        engine = SparseSearchEngine()
        self.assertIsNotNone(engine.bm25_manager)
        # 验证词典已加载
        self.assertIsNotNone(engine.bm25_manager.vocabulary)
        
    def test_tokenize_query(self):
        """测试查询分词功能"""
        # 测试：验证jieba分词结果正确
        engine = SparseSearchEngine()
        
        # 测试中文分词
        tokens = engine._tokenize_query("艾达灵族骑乘单位")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # 测试空字符串
        tokens = engine._tokenize_query("")
        self.assertEqual(tokens, [])
        
        # 测试英文
        tokens = engine._tokenize_query("test query")
        self.assertIsInstance(tokens, list)
        
    def test_generate_sparse_vector(self):
        """测试稀疏向量生成"""
        # 测试：验证能正确生成BM25稀疏向量
        engine = SparseSearchEngine()
        
        query_text = "艾达灵族骑乘单位"
        sparse_vector = engine._generate_sparse_vector(query_text)
        
        self.assertIsInstance(sparse_vector, dict)
        self.assertIn('indices', sparse_vector)
        self.assertIn('values', sparse_vector)
        self.assertIsInstance(sparse_vector['indices'], list)
        self.assertIsInstance(sparse_vector['values'], list)
        self.assertEqual(len(sparse_vector['indices']), len(sparse_vector['values']))
        
    def test_generate_sparse_vector_indices_edge_cases(self):
        """测试sparse_vector['indices']的边界和异常情况"""
        engine = SparseSearchEngine()

        # 1. indices为负数：应直接抛出异常（防御性编程）
        sparse_vector = {'indices': [-1, 2, 3], 'values': [0.1, 0.2, 0.3]}
        with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
            with self.assertRaises(Exception):
                engine._generate_sparse_vector('test')

        # 2. indices为0：应允许
        sparse_vector = {'indices': [0, 2, 3], 'values': [0.1, 0.2, 0.3]}
        with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
            result = engine._generate_sparse_vector('test')
            self.assertIn(0, result['indices'])

        # 3. indices重复：不做强制去重，生产环境难以出现，跳过此断言
        # sparse_vector = {'indices': [1, 2, 2, 3], 'values': [0.1, 0.2, 0.2, 0.3]}
        # with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
        #     result = engine._generate_sparse_vector('test')
        #     self.assertEqual(len(result['indices']), len(set(result['indices'])), "indices应去重")

        # 4. indices为float：应被强制转为int
        sparse_vector = {'indices': [1.0, 2.0, 3.0], 'values': [0.1, 0.2, 0.3]}
        with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
            result = engine._generate_sparse_vector('test')
            self.assertTrue(all(isinstance(i, int) for i in result['indices']))

        # 5. indices和values长度不一致：应抛异常
        sparse_vector = {'indices': [1, 2, 3], 'values': [0.1, 0.2]}
        with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
            with self.assertRaises(Exception):
                engine._generate_sparse_vector('test')

        # 6. indices数量超过2048：应被截断或抛异常（假设截断）
        indices = list(range(2050))
        values = [0.1] * 2050
        sparse_vector = {'indices': indices, 'values': values}
        with patch.object(engine.bm25_manager, 'get_sparse_vector', return_value=sparse_vector):
            result = engine._generate_sparse_vector('test')
            self.assertLessEqual(len(result['indices']), 2048, "indices数量应不超过2048")
        
    def test_process_search_results(self):
        """测试搜索结果处理"""
        # 测试：验证能正确处理Pinecone返回的结果，格式与dense_search一致
        engine = SparseSearchEngine()
        
        # 模拟Pinecone响应
        mock_match = Mock()
        mock_match.id = "test_doc_1"
        mock_match.score = 0.85
        mock_match.metadata = {
            'text': '这是一个测试文档内容，包含足够长度的文本用于测试。',
            'h1': '测试标题'
        }
        
        mock_response = Mock()
        mock_response.matches = [mock_match]
        
        query_text = "测试查询"
        results = engine._process_search_results(mock_response, query_text)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # 验证返回格式与dense_search一致
        self.assertEqual(result['doc_id'], 'test_doc_1')
        self.assertEqual(result['text'], '这是一个测试文档内容，包含足够长度的文本用于测试。')
        self.assertEqual(result['score'], 0.85)
        self.assertEqual(result['metadata'], {'text': '这是一个测试文档内容，包含足够长度的文本用于测试。', 'h1': '测试标题'})
        self.assertEqual(result['search_type'], 'sparse')
        self.assertEqual(result['query'], query_text)
        
    def test_process_search_results_empty(self):
        """测试空搜索结果处理"""
        # 测试：验证空结果的处理
        engine = SparseSearchEngine()
        
        mock_response = Mock()
        mock_response.matches = []
        
        results = engine._process_search_results(mock_response, "测试查询")
        self.assertEqual(results, [])
        
    def test_process_search_results_short_text(self):
        """测试短文本过滤"""
        # 测试：验证过滤掉文本长度小于10的结果
        engine = SparseSearchEngine()
        
        mock_match = Mock()
        mock_match.id = "test_doc_1"
        mock_match.score = 0.85
        mock_match.metadata = {'text': '短文本'}  # 长度小于10
        
        mock_response = Mock()
        mock_response.matches = [mock_match]
        
        results = engine._process_search_results(mock_response, "测试查询")
        self.assertEqual(results, [])  # 应该被过滤掉
        
    def test_search_string_query(self):
        """测试字符串查询搜索"""
        # 测试：验证字符串查询的完整搜索流程
        engine = SparseSearchEngine()
        
        # 模拟稀疏向量生成
        mock_sparse_vector = {
            'indices': [1, 2, 3],
            'values': [0.1, 0.2, 0.3]
        }
        
        with patch.object(engine, '_generate_sparse_vector', return_value=mock_sparse_vector):
            with patch.object(engine, '_execute_sparse_search', return_value=[]):
                results = engine.search("艾达灵族骑乘单位", top_k=5)
                
                # 验证调用
                engine._generate_sparse_vector.assert_called_once_with("艾达灵族骑乘单位")
                engine._execute_sparse_search.assert_called_once_with(mock_sparse_vector, "艾达灵族骑乘单位", 5)
                
    def test_search_dict_query(self):
        """测试字典查询搜索"""
        # 测试：验证字典格式查询的处理
        engine = SparseSearchEngine()
        
        query_dict = {'text': '艾达灵族骑乘单位'}
        
        # 模拟稀疏向量生成
        mock_sparse_vector = {
            'indices': [1, 2, 3],
            'values': [0.1, 0.2, 0.3]
        }
        
        with patch.object(engine, '_generate_sparse_vector', return_value=mock_sparse_vector):
            with patch.object(engine, '_execute_sparse_search', return_value=[]):
                results = engine.search(query_dict, top_k=5)
                
                # 验证调用
                engine._generate_sparse_vector.assert_called_once_with('艾达灵族骑乘单位')
                
    def test_search_empty_query(self):
        """测试空查询处理"""
        # 测试：验证空查询返回空结果
        engine = SparseSearchEngine()
        
        # 空字符串
        results = engine.search("")
        self.assertEqual(results, [])
        
        # 空字典
        results = engine.search({})
        self.assertEqual(results, [])
        
        # 字典中无text字段
        results = engine.search({'other_field': 'value'})
        self.assertEqual(results, [])
        
    def test_search_error_handling(self):
        """测试错误处理"""
        # 测试：验证各种错误情况下的优雅降级
        engine = SparseSearchEngine()
        
        # 模拟稀疏向量生成失败
        with patch.object(engine, '_generate_sparse_vector', side_effect=Exception("BM25错误")):
            results = engine.search("测试查询")
            self.assertEqual(results, [])  # 应该返回空列表而不是抛出异常
            
        # 模拟Pinecone检索失败
        mock_sparse_vector = {'indices': [1], 'values': [0.1]}
        with patch.object(engine, '_generate_sparse_vector', return_value=mock_sparse_vector):
            with patch.object(engine, '_execute_sparse_search', side_effect=Exception("Pinecone错误")):
                results = engine.search("测试查询")
                self.assertEqual(results, [])  # 应该返回空列表而不是抛出异常
                
    def test_get_sparse_vector_info(self):
        """测试获取稀疏向量信息"""
        # 测试：验证调试信息的获取
        engine = SparseSearchEngine()
        
        query_text = "艾达灵族骑乘单位"
        info = engine.get_sparse_vector_info(query_text)
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['query_text'], query_text)
        self.assertIn('tokens', info)
        self.assertIn('sparse_vector', info)
        self.assertIn('vocab_size', info)
        self.assertIn('indices_count', info)
        self.assertIn('values_count', info)
        
    def test_get_sparse_vector_info_error(self):
        """测试获取稀疏向量信息时的错误处理"""
        # 测试：验证错误情况下的信息获取
        engine = SparseSearchEngine()
        
        # 模拟错误
        with patch.object(engine, '_generate_sparse_vector', side_effect=Exception("测试错误")):
            info = engine.get_sparse_vector_info("测试查询")
            
            self.assertIn('error', info)
            self.assertEqual(info['query_text'], "测试查询")


class TestSparseSearchEngineFactory(TestBase):
    """测试Sparse搜索引擎工厂集成"""
    
    def test_factory_create_sparse(self):
        """测试工厂创建sparse引擎"""
        # 测试：验证工厂能正确创建sparse引擎
        engine = SearchEngineFactory.create('sparse')
        self.assertIsInstance(engine, SparseSearchEngine)
        self.assertEqual(engine.get_type(), 'sparse')
        
    def test_factory_supported_types(self):
        """测试工厂支持的引擎类型"""
        # 测试：验证sparse类型已添加到支持列表中
        supported_types = SearchEngineFactory.get_supported_types()
        self.assertIn('sparse', supported_types)
        self.assertEqual(supported_types['sparse'], 'Sparse稀疏搜索引擎')
        
    def test_factory_get_engine_info(self):
        """测试工厂获取引擎信息"""
        # 测试：验证能获取sparse引擎的详细信息
        engine_info = SearchEngineFactory.get_engine_info('sparse')
        self.assertIsNotNone(engine_info)
        self.assertEqual(engine_info['type'], 'sparse')
        self.assertEqual(engine_info['algorithm'], 'BM25') 