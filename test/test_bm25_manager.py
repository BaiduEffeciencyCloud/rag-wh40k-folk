import unittest
import os
import tempfile
import json
import pickle
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataupload.bm25_manager import BM25Manager


class TestBM25Manager(unittest.TestCase):
    """BM25Manager类核心功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_corpus = [
            "战锤40k是一个科幻战争游戏",
            "星际战士是帝国的精英战士",
            "混沌势力威胁着人类帝国",
            "原体是基因改造战士的创造者",
            "帝皇是人类帝国的统治者"
        ]
        self.test_text = "战锤40k中的星际战士"
        
    def test_init_with_default_params(self):
        """测试默认参数初始化"""
        bm25 = BM25Manager()
        self.assertEqual(bm25.k1, 1.5)
        self.assertEqual(bm25.b, 0.75)
        self.assertEqual(bm25.min_freq, 5)
        self.assertEqual(bm25.max_vocab_size, 10000)
        self.assertFalse(bm25.is_fitted)
        self.assertEqual(len(bm25.vocabulary), 0)
        
    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        bm25 = BM25Manager(k1=2.0, b=0.8, min_freq=3, max_vocab_size=5000)
        self.assertEqual(bm25.k1, 2.0)
        self.assertEqual(bm25.b, 0.8)
        self.assertEqual(bm25.min_freq, 3)
        self.assertEqual(bm25.max_vocab_size, 5000)
        
    def test_tokenize_chinese(self):
        """测试中文分词功能"""
        bm25 = BM25Manager()
        text = "战锤40k是一个科幻战争游戏"
        tokens = bm25.tokenize_chinese(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # 验证分词结果包含关键词
        token_text = "".join(tokens)
        self.assertIn("战锤", token_text)
        self.assertIn("科幻", token_text)
        
    def test_fit_corpus(self):
        """测试语料库训练"""
        bm25 = BM25Manager(min_freq=1)  # 降低min_freq以便测试
        bm25.fit(self.test_corpus)
        self.assertTrue(bm25.is_fitted)
        self.assertIsNotNone(bm25.bm25_model)
        self.assertGreater(len(bm25.vocabulary), 0)
        self.assertEqual(len(bm25.corpus_tokens), len(self.test_corpus))
        
    def test_get_sparse_vector_without_fit(self):
        """测试未训练时获取稀疏向量"""
        bm25 = BM25Manager()
        with self.assertRaises(ValueError) as context:
            bm25.get_sparse_vector(self.test_text)
        self.assertIn("未训练", str(context.exception))
        
    def test_get_sparse_vector_after_fit(self):
        """测试训练后获取稀疏向量"""
        bm25 = BM25Manager(min_freq=1)
        bm25.fit(self.test_corpus)
        sparse_vector = bm25.get_sparse_vector(self.test_text)
        self.assertIsInstance(sparse_vector, dict)
        self.assertIn("indices", sparse_vector)
        self.assertIn("values", sparse_vector)
        self.assertIsInstance(sparse_vector["indices"], list)
        self.assertIsInstance(sparse_vector["values"], list)
        self.assertEqual(len(sparse_vector["indices"]), len(sparse_vector["values"]))
        
    def test_save_and_load_model(self):
        """测试模型保存和加载"""
        bm25 = BM25Manager(min_freq=1)
        bm25.fit(self.test_corpus)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as vocab_file:
            vocab_path = vocab_file.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as model_file:
            model_path = model_file.name
            
        try:
            # 保存模型
            bm25.save_model(model_path, vocab_path)
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(vocab_path))
            
            # 创建新实例并加载模型
            bm25_new = BM25Manager()
            bm25_new.load_model(model_path, vocab_path)
            self.assertTrue(bm25_new.is_fitted)
            self.assertIsNotNone(bm25_new.bm25_model)
            self.assertGreater(len(bm25_new.vocabulary), 0)
            
            # 验证加载后的模型功能正常
            sparse_vector = bm25_new.get_sparse_vector(self.test_text)
            self.assertIsInstance(sparse_vector, dict)
            
        finally:
            # 清理临时文件
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
                
    def test_incremental_update(self):
        """测试增量更新功能"""
        bm25 = BM25Manager(min_freq=1)
        bm25.fit(self.test_corpus)
        original_vocab_size = len(bm25.vocabulary)
        
        new_corpus = ["新的文本内容", "包含新的词汇"]
        bm25.incremental_update(new_corpus)
        
        self.assertTrue(bm25.is_fitted)
        self.assertEqual(len(bm25.raw_texts), len(self.test_corpus) + len(new_corpus))
        self.assertEqual(len(bm25.corpus_tokens), len(self.test_corpus) + len(new_corpus))
        
    def test_get_vocabulary_stats(self):
        """测试词汇统计功能"""
        bm25 = BM25Manager(min_freq=1)
        bm25.fit(self.test_corpus)
        stats = bm25.get_vocabulary_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("vocabulary_size", stats)
        self.assertIn("avg_frequency", stats)
        self.assertIn("max_frequency", stats)
        self.assertIn("min_frequency", stats)
        self.assertIsInstance(stats["vocabulary_size"], int)
        self.assertIsInstance(stats["avg_frequency"], (int, float))
        self.assertIsInstance(stats["max_frequency"], int)
        self.assertIsInstance(stats["min_frequency"], int)
        self.assertGreater(stats["vocabulary_size"], 0)
        
    def test_min_freq_filtering(self):
        """测试最小频率过滤功能"""
        # 使用较高的min_freq，确保某些词被过滤
        bm25 = BM25Manager(min_freq=3)
        bm25.fit(self.test_corpus)
        
        # 验证词汇表大小合理
        self.assertGreater(len(bm25.vocabulary), 0)
        # 由于min_freq=3，词汇表应该比所有唯一词少
        all_tokens = []
        for text in self.test_corpus:
            all_tokens.extend(bm25.tokenize_chinese(text))
        unique_tokens = set(all_tokens)
        self.assertLessEqual(len(bm25.vocabulary), len(unique_tokens))


class TestBM25ManagerRefactorTargets(unittest.TestCase):
    """BM25Manager重构目标功能测试 - 这些测试应该失败，直到重构完成"""
    
    def setUp(self):
        """测试前准备"""
        self.test_corpus = [
            "战锤40k是一个科幻战争游戏",
            "星际战士是帝国的精英战士", 
            "混沌势力威胁着人类帝国",
            "原体是基因改造战士的创造者",
            "帝皇是人类帝国的统治者"
        ]
        
        # 创建临时测试词典文件
        self.temp_dir = tempfile.mkdtemp()
        self.dict_dir = os.path.join(self.temp_dir, "dict")
        os.makedirs(self.dict_dir, exist_ok=True)
        
        # 实现get_latest_dict_file函数，参考upsert.py中的_get_latest_vocab_path方法
        def get_latest_dict_file(dict_dir):
            """查找dict目录下最新的bm25_vocab_*.json文件"""
            if not os.path.exists(dict_dir):
                return None  # 参考upsert.py，返回None而不是抛出异常
            
            import glob
            # 查找所有bm25_vocab_*.json文件
            vocab_pattern = os.path.join(dict_dir, "bm25_vocab_*.json")
            vocab_files = glob.glob(vocab_pattern)
            
            if not vocab_files:
                return None  # 参考upsert.py，返回None而不是抛出异常
            
            # 按文件名中的时间戳排序，取最新的
            vocab_files.sort(reverse=True)
            return vocab_files[0]
        
        # 创建测试词典文件
        test_dict_file = os.path.join(self.dict_dir, "bm25_vocab_240101_120000.json")
        test_vocab = {
            "战锤": 0,
            "40k": 1, 
            "星际战士": 2,
            "帝国": 3,
            "混沌": 4,
            "原体": 5,
            "帝皇": 6
        }
        with open(test_dict_file, 'w', encoding='utf-8') as f:
            json.dump(test_vocab, f, ensure_ascii=False, indent=2)
        
        # 使用get_latest_dict_file函数获取真实的词典文件路径
        self.test_dict_file = get_latest_dict_file(self.dict_dir)
        
        # 保存get_latest_dict_file函数供测试使用
        self.get_latest_dict_file_func = get_latest_dict_file
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_latest_dict_file(self):
        """测试词典文件查找功能 - 应该失败，因为方法不存在"""
        bm25 = BM25Manager()
        
        # 测试查找最新词典文件 - 比较BM25Manager的结果与我们实现的结果
        # 查找dict文件夹下的bm25_vocab_[timestamp].json文件
        bm25_latest_dict = bm25.get_latest_dict_file(self.dict_dir)
        expected_latest_dict = self.get_latest_dict_file_func(self.dict_dir)
        self.assertEqual(bm25_latest_dict, expected_latest_dict)
        self.assertEqual(bm25_latest_dict, self.test_dict_file)
        
        # 测试多个词典文件时选择最新的
        older_dict_file = os.path.join(self.dict_dir, "bm25_vocab_240101_100000.json")
        with open(older_dict_file, 'w', encoding='utf-8') as f:
            json.dump({"older": 0}, f, ensure_ascii=False)
        
        # 重新获取最新文件（现在应该有两个文件）
        bm25_latest_dict_again = bm25.get_latest_dict_file(self.dict_dir)
        expected_latest_dict_again = self.get_latest_dict_file_func(self.dict_dir)
        self.assertEqual(bm25_latest_dict_again, expected_latest_dict_again)
        # 应该返回时间戳最新的文件（240101_120000.json）
        self.assertEqual(bm25_latest_dict_again, self.test_dict_file)
        
        # 测试空目录 - 参考upsert.py，返回None而不是抛出异常
        empty_dir = os.path.join(self.temp_dir, "empty_dict")
        os.makedirs(empty_dir, exist_ok=True)
        result = bm25.get_latest_dict_file(empty_dir)
        self.assertIsNone(result)
        
        # 测试不存在的目录 - 参考upsert.py，返回None而不是抛出异常
        result = bm25.get_latest_dict_file("/nonexistent/path")
        self.assertIsNone(result)
    
    def test_load_dictionary(self):
        """测试词典加载功能 - 应该失败，因为方法不存在"""
        bm25 = BM25Manager()
        
        # 测试加载词典
        dict_data = bm25.load_dictionary(self.test_dict_file)
        self.assertIsInstance(dict_data, dict)
        self.assertIn("战锤", bm25.vocabulary)
        self.assertEqual(bm25.vocabulary["战锤"], 0)
        
        # 测试加载不存在的文件
        with self.assertRaises(Exception):
            bm25.load_dictionary("/nonexistent/file.json")
        
        # 测试加载格式错误的文件
        bad_file = os.path.join(self.dict_dir, "bad.json")
        with open(bad_file, 'w') as f:
            f.write("invalid json")
        with self.assertRaises(Exception):
            bm25.load_dictionary(bad_file)
    
    def test_generate_sparse_vectors(self):
        """测试稀疏向量生成功能 - 应该失败，因为方法不存在"""
        bm25 = BM25Manager()
        
        # 先加载词典
        bm25.load_dictionary(self.test_dict_file)
        
        # 测试生成稀疏向量
        # 注意：此测试中涉及读取现有BM25模型文件的逻辑
        # 在evaltool/analyze_sparse/bm25_manager.py中，有以下路径需要确认：
        # 1. models/bm25_model.pkl - 现有BM25模型文件路径
        # 2. 通过get_latest_dict_file()获取的词典文件路径
        # 我不知道这些路径在您的系统中是否存在或具体位置，请手动确认
        sparse_vectors = bm25.generate_sparse_vectors(self.test_corpus)
        self.assertIsInstance(sparse_vectors, list)
        self.assertEqual(len(sparse_vectors), len(self.test_corpus))
        
        # 验证每个向量格式
        for sparse_vector in sparse_vectors:
            self.assertIsInstance(sparse_vector, dict)
            self.assertIn("indices", sparse_vector)
            self.assertIn("values", sparse_vector)
            self.assertIsInstance(sparse_vector["indices"], list)
            self.assertIsInstance(sparse_vector["values"], list)
            self.assertEqual(len(sparse_vector["indices"]), len(sparse_vector["values"]))
        
        # 测试未加载词典时的错误
        bm25_empty = BM25Manager()
        with self.assertRaises(Exception):
            bm25_empty.generate_sparse_vectors(self.test_corpus)
    
    def test_analyze_vocabulary_coverage(self):
        """测试词汇覆盖率分析功能 - 应该失败，因为方法不存在"""
        bm25 = BM25Manager()
        
        # 先加载词典
        bm25.load_dictionary(self.test_dict_file)
        
        # 测试词汇覆盖率分析
        coverage = bm25.analyze_vocabulary_coverage(self.test_corpus)
        self.assertIsInstance(coverage, dict)
        self.assertIn("total_tokens", coverage)
        self.assertIn("unique_tokens", coverage)
        self.assertIn("covered_tokens", coverage)
        self.assertIn("coverage_rate", coverage)
        self.assertIn("missing_tokens", coverage)
        
        # 验证数据类型
        self.assertIsInstance(coverage["total_tokens"], int)
        self.assertIsInstance(coverage["unique_tokens"], int)
        self.assertIsInstance(coverage["covered_tokens"], int)
        self.assertIsInstance(coverage["coverage_rate"], float)
        self.assertIsInstance(coverage["missing_tokens"], list)
        
        # 验证覆盖率计算正确
        self.assertGreaterEqual(coverage["coverage_rate"], 0.0)
        self.assertLessEqual(coverage["coverage_rate"], 1.0)
        self.assertEqual(coverage["covered_tokens"] + len(coverage["missing_tokens"]), coverage["unique_tokens"])
        
        # 测试空文本列表 - 参考evaltool实现，不包含unique_tokens字段
        empty_coverage = bm25.analyze_vocabulary_coverage([])
        self.assertEqual(empty_coverage["total_tokens"], 0)
        self.assertEqual(empty_coverage["coverage_rate"], 0)
        # 注意：evaltool的空文本返回不包含unique_tokens字段
    
    def test_get_vocabulary_stats_enhanced(self):
        """测试词汇统计功能 - 应该失败，因为方法不存在"""
        bm25 = BM25Manager()
        
        # 先加载词典
        bm25.load_dictionary(self.test_dict_file)
        
        # 测试增强的词汇统计
        stats = bm25.get_vocabulary_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("vocabulary_size", stats)
        self.assertIn("avg_frequency", stats)
        self.assertIn("max_frequency", stats)
        self.assertIn("min_frequency", stats)
        
        # 验证数据类型和值
        self.assertIsInstance(stats["vocabulary_size"], int)
        self.assertIsInstance(stats["avg_frequency"], (int, float))
        self.assertIsInstance(stats["max_frequency"], int)
        self.assertIsInstance(stats["min_frequency"], int)
        
        # 验证统计值合理
        self.assertEqual(stats["vocabulary_size"], 7)  # 测试词典中的词数
        self.assertGreaterEqual(stats["max_frequency"], stats["min_frequency"])
        self.assertGreaterEqual(stats["avg_frequency"], stats["min_frequency"])
        self.assertLessEqual(stats["avg_frequency"], stats["max_frequency"])
        
        # 测试空词典
        bm25_empty = BM25Manager()
        empty_stats = bm25_empty.get_vocabulary_stats()
        self.assertEqual(empty_stats["vocabulary_size"], 0)
        self.assertEqual(empty_stats["avg_frequency"], 0)
        self.assertEqual(empty_stats["max_frequency"], 0)
        self.assertEqual(empty_stats["min_frequency"], 0)


if __name__ == '__main__':
    unittest.main() 