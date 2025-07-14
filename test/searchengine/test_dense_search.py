import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchengine.search_interface import SearchEngineInterface
from searchengine.dense_search import DenseSearchEngine
from searchengine.hybrid_search import HybridSearchEngine
from searchengine.search_factory import SearchEngineFactory

class TestSearchEngineInterface(unittest.TestCase):
    """测试搜索引擎接口"""
    
    def test_interface_abstract_methods(self):
        """测试接口抽象方法"""
        # 不能直接实例化抽象类
        with self.assertRaises(TypeError):
            SearchEngineInterface()

class TestDenseSearchEngine(unittest.TestCase):
    """测试Dense搜索引擎"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_openai = Mock()
        self.mock_pinecone = Mock()
        self.mock_index = Mock()
        
        # 模拟搜索结果
        self.mock_match = Mock()
        self.mock_match.score = 0.85
        self.mock_match.metadata = {
            'text': '这是一个测试文档内容，包含足够长度的文本用于测试。',
            'h1': '测试标题',
            'h2': '测试子标题'
        }
        
        self.mock_response = Mock()
        self.mock_response.matches = [self.mock_match]
        
        self.mock_index.query.return_value = self.mock_response
        
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_init(self, mock_pinecone, mock_openai):
        """测试初始化"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = DenseSearchEngine()
        
        self.assertEqual(engine.get_type(), 'dense')
        self.assertIsNotNone(engine.client)
        self.assertIsNotNone(engine.index)
    
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_get_embedding(self, mock_pinecone, mock_openai):
        """测试获取embedding"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        # 模拟embedding响应
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        self.mock_openai.embeddings.create.return_value = mock_embedding_response
        
        engine = DenseSearchEngine()
        embedding = engine.get_embedding("测试文本")
        
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.mock_openai.embeddings.create.assert_called_once()
    
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_search_string_query(self, mock_pinecone, mock_openai):
        """测试字符串查询搜索"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        # 模拟embedding响应
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        self.mock_openai.embeddings.create.return_value = mock_embedding_response
        
        engine = DenseSearchEngine()
        results = engine.search("测试查询", top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], '这是一个测试文档内容，包含足够长度的文本用于测试。')
        self.assertEqual(results[0]['score'], 0.85)
        self.assertEqual(results[0]['search_type'], 'dense')
    
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_search_dict_query(self, mock_pinecone, mock_openai):
        """测试字典查询搜索"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        # 模拟embedding响应
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        self.mock_openai.embeddings.create.return_value = mock_embedding_response
        
        engine = DenseSearchEngine()
        query_dict = {'text': '测试查询'}
        results = engine.search(query_dict, top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['query'], '测试查询')
    
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_search_empty_query(self, mock_pinecone, mock_openai):
        """测试空查询"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = DenseSearchEngine()
        results = engine.search("")
        
        self.assertEqual(results, [])
    
    @patch('searchengine.dense_search.OpenAI')
    @patch('searchengine.dense_search.Pinecone')
    def test_get_capabilities(self, mock_pinecone, mock_openai):
        """测试获取能力信息"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = DenseSearchEngine()
        capabilities = engine.get_capabilities()
        
        self.assertEqual(capabilities['type'], 'dense')
        self.assertTrue(capabilities['supports_filtering'])
        self.assertFalse(capabilities['supports_hybrid'])

class TestHybridSearchEngine(unittest.TestCase):
    """测试Hybrid搜索引擎"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_openai = Mock()
        self.mock_pinecone = Mock()
        self.mock_index = Mock()
        
        # 模拟搜索结果
        self.mock_match = Mock()
        self.mock_match.score = 0.9
        self.mock_match.metadata = {
            'text': '这是一个hybrid测试文档内容，包含足够长度的文本用于测试。',
            'h1': '测试标题',
            'h2': '测试子标题'
        }
        
        self.mock_response = Mock()
        self.mock_response.matches = [self.mock_match]
        
        self.mock_index.query.return_value = self.mock_response
    
    @patch('searchengine.hybrid_search.OpenAI')
    @patch('searchengine.hybrid_search.Pinecone')
    @patch('searchengine.hybrid_search.DenseSearchEngine')
    def test_init(self, mock_dense_engine, mock_pinecone, mock_openai):
        """测试初始化"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = HybridSearchEngine()
        
        self.assertEqual(engine.get_type(), 'hybrid')
        self.assertIsNotNone(engine.client)
        self.assertIsNotNone(engine.index)
        self.assertIsNotNone(engine.dense_engine)
    
    @patch('searchengine.hybrid_search.OpenAI')
    @patch('searchengine.hybrid_search.Pinecone')
    @patch('searchengine.hybrid_search.DenseSearchEngine')
    def test_build_kg_filter(self, mock_dense_engine, mock_pinecone, mock_openai):
        """测试构建KG filter"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = HybridSearchEngine()
        
        # 测试有实体的KG信息
        kg_info = {'entities': ['帝国卫队', '星际战士']}
        filter_dict = engine._build_kg_filter(kg_info)
        
        self.assertIn('$or', filter_dict)
        self.assertEqual(len(filter_dict['$or']), 12)  # 2个实体 * 6个字段
        
        # 测试空KG信息
        empty_filter = engine._build_kg_filter({})
        self.assertEqual(empty_filter, {})
    
    @patch('searchengine.hybrid_search.OpenAI')
    @patch('searchengine.hybrid_search.Pinecone')
    @patch('searchengine.hybrid_search.DenseSearchEngine')
    def test_search_with_kg(self, mock_dense_engine, mock_pinecone, mock_openai):
        """测试带KG信息的搜索"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        # 模拟embedding响应
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        self.mock_openai.embeddings.create.return_value = mock_embedding_response
        
        engine = HybridSearchEngine()
        kg_info = {'entities': ['帝国卫队']}
        results = engine.search_with_kg("测试查询", kg_info, top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['search_type'], 'hybrid')
        self.assertTrue(results[0]['kg_used'])
    
    @patch('searchengine.hybrid_search.OpenAI')
    @patch('searchengine.hybrid_search.Pinecone')
    @patch('searchengine.hybrid_search.DenseSearchEngine')
    def test_get_capabilities(self, mock_dense_engine, mock_pinecone, mock_openai):
        """测试获取能力信息"""
        mock_openai.return_value = self.mock_openai
        mock_pinecone.return_value = self.mock_pinecone
        self.mock_pinecone.Index.return_value = self.mock_index
        
        engine = HybridSearchEngine()
        capabilities = engine.get_capabilities()
        
        self.assertEqual(capabilities['type'], 'hybrid')
        self.assertTrue(capabilities['supports_filtering'])
        self.assertTrue(capabilities['supports_hybrid'])
        self.assertTrue(capabilities['supports_kg'])

class TestSearchEngineFactory(unittest.TestCase):
    """测试搜索引擎工厂"""
    
    def test_get_supported_types(self):
        """测试获取支持的引擎类型"""
        types = SearchEngineFactory.get_supported_types()
        
        self.assertIn('dense', types)
        self.assertIn('hybrid', types)
        self.assertEqual(len(types), 2)
    
    def test_create_dense_engine(self):
        """测试创建dense引擎"""
        with patch('searchengine.dense_search.OpenAI'), \
             patch('searchengine.dense_search.Pinecone'):
            
            engine = SearchEngineFactory.create('dense')
            self.assertEqual(engine.get_type(), 'dense')
    
    def test_create_hybrid_engine(self):
        """测试创建hybrid引擎"""
        with patch('searchengine.hybrid_search.OpenAI'), \
             patch('searchengine.hybrid_search.Pinecone'), \
             patch('searchengine.hybrid_search.DenseSearchEngine'):
            
            engine = SearchEngineFactory.create('hybrid')
            self.assertEqual(engine.get_type(), 'hybrid')
    
    def test_create_invalid_type(self):
        """测试创建无效类型"""
        with self.assertRaises(ValueError):
            SearchEngineFactory.create('invalid_type')
    
    def test_get_engine_info(self):
        """测试获取引擎信息"""
        with patch('searchengine.dense_search.OpenAI'), \
             patch('searchengine.dense_search.Pinecone'):
            
            info = SearchEngineFactory.get_engine_info('dense')
            self.assertIsNotNone(info)
            self.assertEqual(info['type'], 'dense')
            
            # 测试无效类型
            invalid_info = SearchEngineFactory.get_engine_info('invalid')
            self.assertIsNone(invalid_info)

if __name__ == '__main__':
    unittest.main() 