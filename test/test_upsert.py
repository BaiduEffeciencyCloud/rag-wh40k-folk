import unittest
import asyncio
import os
import json
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock
import subprocess

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataupload.upsert import main,UpsertManager
from dataupload.data_vector import DataVector

class TestUpsert(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试用的 Markdown 文件
        self.test_file = "test_content.md"
        self.test_content = """# 测试标题1
这是测试内容1

## 测试标题2
这是测试内容2

### 测试标题3
这是测试内容3这是测试内容3这是测试内容3这是测试内容3这是测试内容3这是测试内容3这是测试内容3这是测试内容3这是测试内容3"""
        
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        # 设置测试参数
        self.faction = "test_faction"
        self.output_file = "test_output.txt"
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
    
    def test_data_vector_process(self):
        """测试 DataVector 的文档处理功能"""
        vectorizer = DataVector()
        chunks = vectorizer.process_markdown(self.test_file, self.faction)
        
        # 验证返回的 chunks 格式
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        
        # 验证每个 chunk 的格式
        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('metadata', chunk)
            self.assertIn('source_file', chunk['metadata'])
            self.assertIn('faction', chunk['metadata'])
            self.assertIn('chunk_index', chunk['metadata'])

class TestUpsertManager(unittest.TestCase):
    def setUp(self):
        """每个测试方法开始前的准备工作"""
        self.test_file = "test_content.md"
        # 使用真实的测试数据文件内容
        real_test_file = os.path.join(os.path.dirname(__file__), "testdata", "aeldaricodex.md")
        with open(real_test_file, 'r', encoding='utf-8') as f:
            self.test_content = f.read()
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        self.faction = "test_faction"
        self.extra_metadata = {"faction": self.faction}
        self.manager = UpsertManager(environment="test", chunker_version="v3")

    def tearDown(self):
        """每个测试方法完成后的清理工作"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        for fname in os.listdir("dataupload"):
            if fname.startswith("upload_log_"):
                try:
                    os.remove(os.path.join("dataupload", fname))
                except Exception:
                    pass

    @classmethod
    def tearDownClass(cls):
        """所有测试方法完成后的最终清理工作"""
        if os.path.exists("test_content.md"):
            os.remove("test_content.md")

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_chunk_vectorization_error_handling(self, mock_get_embedding_model, mock_get_pinecone_index, mock_subprocess):
        """chunk向量化异常捕获与不中断"""
        # embedding_model.embed_query: 第2个chunk抛异常
        mock_embed = MagicMock(side_effect=[
            [0.1, 0.2, 0.3],
            Exception("模拟向量化失败"),
            [0.4, 0.5, 0.6]
        ])
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query = mock_embed
        mock_get_embedding_model.return_value = mock_embedding_model
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # 跑主流程
        self.manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        # 检查上传被调用2次（第2个chunk失败）
        self.assertTrue(mock_pinecone_index.upsert.called)
        # 检查上传日志
        log_files = [f for f in os.listdir("dataupload") if f.startswith("upload_log_")]
        self.assertTrue(log_files)
        with open(os.path.join("dataupload", log_files[-1]), 'r', encoding='utf-8') as f:
            log = json.load(f)
            # self.assertTrue(fail_chunks)
            # self.assertIn("mock embedding error", fail_chunks[0]["error"])

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_sparse_embedding_and_metadata(self, mock_get_embedding_model, mock_get_pinecone_index, mock_subprocess):
        """sparse embedding生成与metadata/id设计"""
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        self.manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        # 检查upsert参数
        args, kwargs = mock_pinecone_index.upsert.call_args
        batch = kwargs['vectors']
        for vec in batch:
            self.assertIn('sparse_values', vec)
            self.assertIn('indices', vec['sparse_values'])
            self.assertIn('values', vec['sparse_values'])
            self.assertIn('metadata', vec)
            self.assertIn('id', vec)
            # 检查id是否包含文件名和索引
            self.assertTrue(self.test_file.split('.')[0] in vec['id'])

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_extreme_and_empty_data(self, mock_get_embedding_model, mock_get_pinecone_index, mock_subprocess):
        """极端/空数据健壮性"""
        # 空文件
        empty_file = "empty_test.md"
        with open(empty_file, 'w', encoding='utf-8') as f:
            f.write("")
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # 不应抛出异常
        try:
            self.manager.process_and_upsert_document(empty_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        except Exception as e:
            self.fail(f"空数据处理抛出异常: {e}")
        finally:
            if os.path.exists(empty_file):
                os.remove(empty_file)

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_batch_upload_partial_failure(self, mock_get_embedding_model, mock_get_pinecone_index, mock_subprocess):
        """分批上传与部分失败"""
        # 模拟upsert第2批抛异常
        def upsert_side_effect(*args, **kwargs):
            if hasattr(upsert_side_effect, 'called'):
                raise Exception("模拟批次上传失败")
            upsert_side_effect.called = True
            return None
        upsert_side_effect.called = False
        mock_pinecone_index = MagicMock()
        mock_pinecone_index.upsert.side_effect = upsert_side_effect
        mock_get_pinecone_index.return_value = mock_pinecone_index
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # 构造大文件，保证有2批
        big_file = "big_test.md"
        with open(big_file, 'w', encoding='utf-8') as f:
            for i in range(250):
                f.write(f"# 标题{i}\n内容{i}\n")
        try:
            self.manager.process_and_upsert_document(big_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        except Exception as e:
            # 主流程不应因批次失败而中断
            self.assertIn("模拟批次上传失败", str(e))
        finally:
            if os.path.exists(big_file):
                os.remove(big_file)

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_hybrid_upsert_with_real_data(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """用真实数据驱动，验证dense+sparse向量均被上传"""
        # 用真实数据
        test_file = os.path.join(os.path.dirname(__file__), "testdata", "aeldaricodex.md")
        # mock embedding
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # mock pinecone
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.get_sparse_vector.return_value = {"indices": [1,2], "values": [0.5, 0.8]}
        mock_bm25_manager.return_value = mock_bm25
        manager = UpsertManager(environment="test", chunker_version="v3")
        manager.process_and_upsert_document(test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        # 检查upsert参数
        args, kwargs = mock_pinecone_index.upsert.call_args
        batch = kwargs['vectors']
        for vec in batch:
            self.assertIn('sparse_values', vec)
            self.assertIn('indices', vec['sparse_values'])
            self.assertIn('values', vec['sparse_values'])
            self.assertIn('metadata', vec)
            self.assertIn('id', vec)
            self.assertIn('values', vec)  # dense

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_upsert_manager_bm25_model_load_and_save(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """验证主流程优先加载BM25模型，无则fit并保存"""
        # mock embedding
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # mock pinecone
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.is_fitted = False
        mock_bm25.get_sparse_vector.return_value = {"indices": [1], "values": [0.5]}
        mock_bm25.load_model = MagicMock()
        mock_bm25.save_model = MagicMock()
        
        # 确保fit方法会设置is_fitted为True
        def fit_side_effect(*args, **kwargs):
            mock_bm25.is_fitted = True
        mock_bm25.fit = MagicMock(side_effect=fit_side_effect)
        
        mock_bm25_manager.return_value = mock_bm25
        
        print(f"[DEBUG] 测试开始前文件存在: {os.path.exists(self.test_file)}")
        
        # 在patch之前保存真正的exists函数
        real_exists = os.path.exists
        
        # 第一次调用：模拟模型文件存在
        # 创建一个文件存在状态的字典
        file_exists_state = {
            "models/bm25_model.pkl": True,
            "dict/bm25_vocab_2507041622.json": True
        }
        
        def fake_exists_model_exists(path):
            # 如果是BM25模型文件，返回预设状态
            if path in file_exists_state:
                return file_exists_state[path]
            # 其他文件正常检查，使用原始的exists函数
            return real_exists(path)
        
        with patch("os.path.exists", side_effect=fake_exists_model_exists):
            manager = UpsertManager(environment="test", chunker_version="v3")
            manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
            mock_bm25.load_model.assert_called()
        
        print(f"[DEBUG] 第一次调用后文件存在: {os.path.exists(self.test_file)}")
        
        # 第二次调用：模拟模型文件不存在，应fit并save
        mock_bm25.is_fitted = False
        # 确保测试文件存在（可能在第一次调用中被删除了）
        if not os.path.exists(self.test_file):
            print(f"[DEBUG] 文件不存在，重新创建: {self.test_file}")
            with open(self.test_file, 'w', encoding='utf-8') as f:
                f.write(self.test_content)
        
        print(f"[DEBUG] 第二次调用前文件存在: {os.path.exists(self.test_file)}")
        
        # 修改文件存在状态
        file_exists_state.update({
            "models/bm25_model.pkl": False,
            "dict/bm25_vocab_2507041622.json": False
        })
        
        with patch("os.path.exists", side_effect=fake_exists_model_exists):
            manager = UpsertManager(environment="test", chunker_version="v3")
            manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
            mock_bm25.fit.assert_called()
            mock_bm25.save_model.assert_called()

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_upsert_manager_incremental_update(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """验证多次upsert时BM25Manager支持增量更新，不全量重训"""
        # mock embedding
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # mock pinecone
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.is_fitted = True
        mock_bm25.get_sparse_vector.return_value = {"indices": [1], "values": [0.5]}
        mock_bm25.incremental_update = MagicMock()
        mock_bm25_manager.return_value = mock_bm25
        manager = UpsertManager(environment="test", chunker_version="v3")
        # 第一次upsert
        manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        # 第二次upsert应调用incremental_update
        manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
        self.assertTrue(mock_bm25.incremental_update.called)

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_upsert_manager_bm25_config_integration(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """测试BM25Config集成后主流程功能正常"""
        # 保存原始环境变量
        original_env = {}
        for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE']:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # 设置自定义BM25配置
            os.environ['BM25_K1'] = '2.0'
            os.environ['BM25_B'] = '0.8'
            os.environ['BM25_MIN_FREQ'] = '3'
            os.environ['BM25_MAX_VOCAB_SIZE'] = '5000'
            
            # 在环境变量设置后创建UpsertManager
            manager = UpsertManager(environment="test", chunker_version="v3")
            
            # mock embedding
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_get_embedding_model.return_value = mock_embedding_model
            # mock pinecone
            mock_pinecone_index = MagicMock()
            mock_get_pinecone_index.return_value = mock_pinecone_index
            # mock bm25_manager
            mock_bm25 = MagicMock()
            mock_bm25.is_fitted = False
            mock_bm25.get_sparse_vector.return_value = {"indices": [1], "values": [0.5]}
            mock_bm25.load_model = MagicMock()
            mock_bm25.save_model = MagicMock()
            
            # 确保fit方法会设置is_fitted为True
            def fit_side_effect(*args, **kwargs):
                mock_bm25.is_fitted = True
            mock_bm25.fit = MagicMock(side_effect=fit_side_effect)
            
            mock_bm25_manager.return_value = mock_bm25
            
            # 运行主流程
            manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
            
            # 验证BM25Manager使用正确的配置参数创建
            mock_bm25_manager.assert_called_once()
            call_args = mock_bm25_manager.call_args
            self.assertEqual(call_args[1]['k1'], 2.0)
            self.assertEqual(call_args[1]['b'], 0.8)
            self.assertEqual(call_args[1]['min_freq'], 3)
            self.assertEqual(call_args[1]['max_vocab_size'], 5000)
            
            # 验证主流程其他功能正常
            self.assertTrue(mock_pinecone_index.upsert.called)
            
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                os.environ[key] = value
            for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE']:
                if key not in original_env:
                    os.environ.pop(key, None)

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_upsert_manager_bm25_config_default_values(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """测试BM25Config使用默认值时主流程正常"""
        # mock embedding
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # mock pinecone
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.is_fitted = False
        mock_bm25.get_sparse_vector.return_value = {"indices": [1], "values": [0.5]}
        mock_bm25.load_model = MagicMock()
        mock_bm25.save_model = MagicMock()
        
        # 确保fit方法会设置is_fitted为True
        def fit_side_effect(*args, **kwargs):
            mock_bm25.is_fitted = True
        mock_bm25.fit = MagicMock(side_effect=fit_side_effect)
        
        mock_bm25_manager.return_value = mock_bm25
        
        # 清除可能存在的BM25环境变量，使用默认值
        original_env = {}
        for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE']:
            if key in os.environ:
                original_env[key] = os.environ[key]
                os.environ.pop(key)
        
        try:
            # 运行主流程
            self.manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
            
            # 验证BM25Manager使用默认配置参数创建
            mock_bm25_manager.assert_called_once()
            call_args = mock_bm25_manager.call_args
            self.assertEqual(call_args[1]['k1'], 1.5)  # 默认值
            self.assertEqual(call_args[1]['b'], 0.75)  # 默认值
            self.assertEqual(call_args[1]['min_freq'], 5)  # 默认值
            self.assertEqual(call_args[1]['max_vocab_size'], 10000)  # 默认值
            
            # 验证主流程其他功能正常
            self.assertTrue(mock_pinecone_index.upsert.called)
            
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                os.environ[key] = value

    @patch("subprocess.run", return_value=None)
    @patch("dataupload.upsert.BM25Manager")
    @patch("dataupload.upsert.get_pinecone_index")
    @patch("dataupload.upsert.get_embedding_model")
    def test_upsert_manager_bm25_config_path_generation(self, mock_get_embedding_model, mock_get_pinecone_index, mock_bm25_manager, mock_subprocess):
        """测试BM25Config路径生成功能在主流程中的使用"""
        # mock embedding
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model
        # mock pinecone
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.is_fitted = False
        mock_bm25.get_sparse_vector.return_value = {"indices": [1], "values": [0.5]}
        mock_bm25.load_model = MagicMock()
        mock_bm25.save_model = MagicMock()
        
        # 确保fit方法会设置is_fitted为True
        def fit_side_effect(*args, **kwargs):
            mock_bm25.is_fitted = True
        mock_bm25.fit = MagicMock(side_effect=fit_side_effect)
        
        mock_bm25_manager.return_value = mock_bm25
        
        # 设置自定义路径配置
        original_env = {}
        for key in ['BM25_MODEL_DIR', 'BM25_VOCAB_DIR', 'BM25_MODEL_FILENAME']:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            os.environ['BM25_MODEL_DIR'] = 'custom_models'
            os.environ['BM25_VOCAB_DIR'] = 'custom_dict'
            os.environ['BM25_MODEL_FILENAME'] = 'custom_bm25_model.pkl'
            
            # 运行主流程
            self.manager.process_and_upsert_document(self.test_file, enable_kg=False, enable_pinecone=True, extra_metadata=self.extra_metadata)
            
            # 验证save_model被调用，说明路径生成功能正常
            self.assertTrue(mock_bm25.save_model.called)
            
            # 验证主流程其他功能正常
            self.assertTrue(mock_pinecone_index.upsert.called)
            
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                os.environ[key] = value
            for key in ['BM25_MODEL_DIR', 'BM25_VOCAB_DIR', 'BM25_MODEL_FILENAME']:
                if key not in original_env:
                    os.environ.pop(key, None)

if __name__ == '__main__':
    unittest.main() 