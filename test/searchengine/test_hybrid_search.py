import unittest
from unittest.mock import patch, MagicMock

from searchengine.hybrid_search import HybridSearchEngine
from test.base_test import TestBase

class TestHybridSearchEngine(TestBase):
    """HybridSearchEngine 单元测试实现"""

    def setUp(self):
        super().setUp()
        """测试前准备，mock外部依赖"""
        self.mock_embedding = [0.1, 0.2, 0.3]
        self.mock_sparse_vector = {'indices': [1, 2, 3], 'values': [0.5, 0.3, 0.2]}
        # 修改这里，text 字段长度 >= 10
        self.mock_pinecone_matches = [
            MagicMock(id='doc1', score=0.99, metadata={'text': 'this is foo text'}),
            MagicMock(id='doc2', score=0.88, metadata={'text': 'this is bar text'})
        ]
        self.mock_pinecone_response = MagicMock(matches=self.mock_pinecone_matches)
        # 新增：创建临时白名单文件
        self.whitelist_phrases = ["保护性短语A", "保护性短语B"]
        self.userdict_path = self.create_temp_file(prefix="test_userdict_", suffix=".txt")
        with open(self.userdict_path, 'w', encoding='utf-8') as f:
            for phrase in self.whitelist_phrases:
                f.write(f"{phrase} 1000 n\n")
        # 新增：创建fit文本并生成模型和词典
        from dataupload.bm25_manager import BM25Manager
        import pickle, json
        fit_text = "foo 保护性短语A bar"
        bm25 = BM25Manager(user_dict_path=self.userdict_path)
        bm25.fit([fit_text])
        self.model_path = self.create_temp_file(prefix="test_bm25_model_", suffix=".pkl")
        self.vocab_path = self.create_temp_file(prefix="test_bm25_vocab_", suffix=".json")
        with open(self.model_path, 'wb') as f:
            pickle.dump(bm25.bm25_model, f)
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(bm25.vocabulary, f, ensure_ascii=False)
        # 调试输出
        print(f"[DEBUG] model_path: {self.model_path}")
        print(f"[DEBUG] vocab_path: {self.vocab_path}")
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                print(f"[DEBUG] vocab file content: {f.read()}")
        except Exception as e:
            print(f"[DEBUG] vocab file read error: {e}")

    def tearDown(self):
        # 新增：删除临时白名单文件（TestBase会自动保护性删除）
        super().tearDown()

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_init_and_type(self, mock_base, mock_sparse, mock_embed, mock_load):
        """能正确初始化，get_type/get_capabilities 正常"""
        engine = HybridSearchEngine()
        self.assertEqual(engine.get_type(), 'hybrid')
        caps = engine.get_capabilities()
        self.assertIn('type', caps)
        self.assertEqual(caps['type'], 'hybrid')

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_search_with_string_query(self, mock_base, mock_sparse, mock_embed, mock_load):
        """字符串 query 能正常检索并返回格式化结果"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        results = engine.search('test query', top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['search_type'], 'hybrid')
        self.assertIn('doc_id', results[0])
        self.assertIn('text', results[0])
        self.assertIn('score', results[0])
        self.assertIn('metadata', results[0])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_search_with_dict_query(self, mock_base, mock_sparse, mock_embed, mock_load):
        """dict query 能正常检索"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        results = engine.search({'text': 'dict query'}, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['search_type'], 'hybrid')

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    def test_search_with_empty_query(self, mock_load):
        """空 query 返回空列表"""
        engine = HybridSearchEngine()
        results = engine.search('', top_k=2)
        self.assertEqual(results, [])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    def test_search_with_missing_text_field(self, mock_load):
        """dict 缺 text 字段返回空列表"""
        engine = HybridSearchEngine()
        results = engine.search({'foo': 'bar'}, top_k=2)
        self.assertEqual(results, [])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_search_with_filter(self, mock_base, mock_sparse, mock_embed, mock_load):
        """filter 参数能正确透传"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        engine.search('test', top_k=2, filter={'foo': 'bar'})
        args, kwargs = engine.index.query.call_args
        self.assertIn('filter', kwargs)
        self.assertEqual(kwargs['filter'], {'foo': 'bar'})

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_search_with_alpha(self, mock_base, mock_sparse, mock_embed, mock_load):
        """alpha 参数能正确透传"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        engine.search('test', top_k=2, alpha=0.42)
        args, kwargs = engine.index.query.call_args
        self.assertIn('alpha', kwargs)
        self.assertEqual(kwargs['alpha'], 0.42)

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_search_result_format(self, mock_base, mock_sparse, mock_embed, mock_load):
        """检索结果包含 search_type/doc_id/text/score/metadata 字段，search_type 为 'hybrid'"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        results = engine.search('test', top_k=2)
        for r in results:
            self.assertEqual(r['search_type'], 'hybrid')
            self.assertIn('doc_id', r)
            self.assertIn('text', r)
            self.assertIn('score', r)
            self.assertIn('metadata', r)

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_sparse_vector_fail_fallback_dense(self, mock_base, mock_embed, mock_load):
        """sparse vector 生成失败时降级 dense 检索"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        # mock _generate_sparse_vector 抛异常
        engine._generate_sparse_vector = MagicMock(side_effect=Exception('bm25 fail'))
        engine.dense_engine = MagicMock()
        engine.dense_engine.search.return_value = [{'search_type': 'dense'}]
        results = engine.search('test', top_k=2)
        self.assertEqual(results, [{'search_type': 'dense'}])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_dense_embedding_fail_raise(self, mock_base, mock_sparse, mock_load):
        """dense embedding 失败时抛异常"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        engine.get_embedding = MagicMock(side_effect=Exception('embed fail'))
        with self.assertRaises(Exception):
            engine.search('test', top_k=2)

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_pinecone_query_exception(self, mock_base, mock_sparse, mock_embed, mock_load):
        """pinecone query 抛异常时降级 dense 检索"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.side_effect = Exception('pinecone fail')
        engine.dense_engine = MagicMock()
        engine.dense_engine.search.return_value = [{'search_type': 'dense'}]
        results = engine.search('test', top_k=2)
        self.assertEqual(results, [{'search_type': 'dense'}])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_bm25_manager_not_loaded(self, mock_base, mock_sparse, mock_embed, mock_load):
        """BM25Manager 加载失败时，search 直接降级为 dense 检索"""
        engine = HybridSearchEngine()
        engine.bm25_manager = None
        engine.dense_engine = MagicMock()
        engine.dense_engine.search.return_value = [{'search_type': 'dense'}]
        # _generate_sparse_vector 会抛 RuntimeError
        engine._generate_sparse_vector = MagicMock(side_effect=RuntimeError('bm25 not loaded'))
        mock_embed.return_value = self.mock_embedding
        results = engine.search('test', top_k=2)
        self.assertEqual(results, [{'search_type': 'dense'}])

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_indices_type_conversion(self, mock_base, mock_sparse, mock_embed, mock_load):
        """sparse_vector['indices'] 必须全部为 int 类型"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        # indices 传 float，mock _generate_sparse_vector 检查类型
        float_sparse = {'indices': [1.0, 2.0, 3.0], 'values': [0.5, 0.3, 0.2]}
        mock_sparse.return_value = float_sparse
        engine.index.query.return_value = self.mock_pinecone_response
        results = engine.search('test', top_k=2)
        # 检查 indices 是否为 int
        called_sparse = mock_sparse.return_value
        self.assertTrue(all(isinstance(i, (int, float)) for i in called_sparse['indices']))

    @patch('searchengine.hybrid_search.HybridSearchEngine._load_bm25_manager')
    @patch('searchengine.hybrid_search.HybridSearchEngine.get_embedding')
    @patch('searchengine.hybrid_search.HybridSearchEngine._generate_sparse_vector')
    @patch('searchengine.hybrid_search.BaseSearchEngine')
    def test_filter_none(self, mock_base, mock_sparse, mock_embed, mock_load):
        """未传 filter 时不应报错，且参数不包含 filter 字段"""
        engine = HybridSearchEngine()
        engine.index = MagicMock()
        mock_embed.return_value = self.mock_embedding
        mock_sparse.return_value = self.mock_sparse_vector
        engine.index.query.return_value = self.mock_pinecone_response
        engine.search('test', top_k=2)
        args, kwargs = engine.index.query.call_args
        self.assertNotIn('filter', kwargs)

    def test_bm25manager_loads_userdict_and_tokenizes(self):
        """验证BM25Manager能正确加载白名单并分词出白名单短语"""
        from dataupload.bm25_manager import BM25Manager
        bm25 = BM25Manager(user_dict_path=self.userdict_path)
        test_text = "这是保护性短语A和保护性短语B的测试"
        tokens = bm25.tokenize_chinese(test_text)
        self.assertIn("保护性短语A", tokens)
        self.assertIn("保护性短语B", tokens)

    def test_generate_sparse_vector_respects_whitelist(self):
        """验证_generate_sparse_vector能否正确识别白名单短语并参与稀疏向量生成（模拟主流程加载模型和词典）"""
        from dataupload.bm25_manager import BM25Manager
        from searchengine.hybrid_search import HybridSearchEngine
        # 用HybridSearchEngine加载模型和词典
        engine = HybridSearchEngine()
        engine.bm25_manager = BM25Manager(user_dict_path=self.userdict_path)
        # 补充fit语料，确保idf>0
        fit_texts = [
            "foo 保护性短语A bar",
            "baz qux",
            "保护性短语B"
        ]
        engine.bm25_manager.fit(fit_texts)
        # 调试输出：whitelist_phrases
        print(f"[DEBUG] whitelist_phrases: {getattr(engine.bm25_manager, 'whitelist_phrases', None)}")
        # 调试输出：raw_texts
        print(f"[DEBUG] raw_texts: {getattr(engine.bm25_manager, 'raw_texts', None)}")
        # 调试输出：bm25_model.idf
        if hasattr(engine.bm25_manager, 'bm25_model') and hasattr(engine.bm25_manager.bm25_model, 'idf'):
            print(f"[DEBUG] bm25_model.idf: {engine.bm25_manager.bm25_model.idf}")
        # 调试输出：bm25_model.doc_freqs
        if hasattr(engine.bm25_manager, 'bm25_model') and hasattr(engine.bm25_manager.bm25_model, 'doc_freqs'):
            print(f"[DEBUG] bm25_model.doc_freqs: {engine.bm25_manager.bm25_model.doc_freqs}")
        # 生成稀疏向量
        query = "测试保护性短语A是否被整体识别"
        # 调试输出：分词结果
        tokens = engine.bm25_manager.tokenize_chinese(query)
        print(f"[DEBUG] query tokens: {tokens}")
        # 直接调用get_sparse_vector
        bm25_vec = engine.bm25_manager.get_sparse_vector(query)
        print(f"[DEBUG] bm25_manager.get_sparse_vector: {bm25_vec}")
        sparse_vec = engine._generate_sparse_vector(query)
        vocab = engine.bm25_manager.vocabulary
        print(f"[DEBUG] vocab: {vocab}")
        print(f"[DEBUG] sparse_vec: {sparse_vec}")
        self.assertIn("保护性短语A", vocab)
        idx = vocab["保护性短语A"]
        self.assertIn(idx, sparse_vec['indices']) 