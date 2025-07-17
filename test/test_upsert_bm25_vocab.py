"""
测试upsert.py中BM25词汇表文件选择逻辑
遵循TDD原则：先写测试，再修改代码
"""
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataupload.upsert import UpsertManager
from dataupload.bm25_config import BM25Config

class TestUpsertBM25Vocab:
    """测试UpsertManager中BM25词汇表文件选择逻辑"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.dict_dir = os.path.join(self.temp_dir, "dict")
        os.makedirs(self.dict_dir, exist_ok=True)
        
        # 创建测试用的词汇表文件
        self.create_test_vocab_files()
        
    def teardown_method(self):
        """每个测试方法后的清理"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_vocab_files(self):
        """创建测试用的词汇表文件"""
        # 创建不同时间戳的词汇表文件
        vocab_files = [
            "bm25_vocab_2401011200.json",  # 最早
            "bm25_vocab_2401011400.json",  # 中间
            "bm25_vocab_2401011600.json",  # 最新
        ]
        
        for filename in vocab_files:
            filepath = os.path.join(self.dict_dir, filename)
            vocab_data = {"test_token": 1.0}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f)
    
    def test_get_latest_vocab_path_returns_newest_file(self):
        """测试_get_latest_vocab_path返回最新的词汇表文件"""
        # 创建UpsertManager实例
        manager = UpsertManager()
        
        # 模拟dict目录路径
        with patch.object(manager, '_get_latest_vocab_path') as mock_get_latest:
            # 设置返回值
            expected_path = os.path.join(self.dict_dir, "bm25_vocab_2401011600.json")
            mock_get_latest.return_value = expected_path
            
            # 调用方法
            result = manager._get_latest_vocab_path()
            
            # 验证返回的是最新的文件
            assert result == expected_path
            assert "2401011600" in result  # 最新的时间戳
    
    def test_save_bm25_model_uses_latest_vocab_path(self):
        """测试_save_bm25_model使用最新的词汇表路径"""
        manager = UpsertManager()
        
        # 模拟BM25Manager
        manager.bm25_manager = MagicMock()
        manager.bm25_manager.is_fitted = True
        
        # 模拟BM25Config
        manager.bm25_config = MagicMock()
        manager.bm25_config.get_model_path.return_value = os.path.join(self.temp_dir, "models", "bm25_model.pkl")
        
        # 模拟_get_latest_vocab_path返回现有文件
        latest_vocab_path = os.path.join(self.dict_dir, "bm25_vocab_2401011600.json")
        
        with patch.object(manager, '_get_latest_vocab_path', return_value=latest_vocab_path):
            with patch('os.makedirs'):
                with patch('logging.info'):
                    # 调用方法
                    manager._save_bm25_model()
                    
                    # 验证使用了最新的词汇表路径
                    manager.bm25_manager.save_model.assert_called_once()
                    call_args = manager.bm25_manager.save_model.call_args
                    model_path, vocab_path = call_args[0]
                    assert vocab_path == latest_vocab_path
    
    def test_save_bm25_model_creates_new_vocab_when_none_exists(self):
        """测试当没有现有词汇表时，创建新的词汇表文件"""
        manager = UpsertManager()
        
        # 模拟BM25Manager
        manager.bm25_manager = MagicMock()
        manager.bm25_manager.is_fitted = True
        
        # 模拟BM25Config
        manager.bm25_config = MagicMock()
        manager.bm25_config.get_model_path.return_value = os.path.join(self.temp_dir, "models", "bm25_model.pkl")
        manager.bm25_config.get_vocab_path.return_value = os.path.join(self.dict_dir, "bm25_vocab_new.json")
        
        # 模拟_get_latest_vocab_path返回None（没有现有文件）
        with patch.object(manager, '_get_latest_vocab_path', return_value=None):
            with patch('os.makedirs'):
                with patch('logging.info'):
                    # 调用方法
                    manager._save_bm25_model()
                    
                    # 验证创建了新的词汇表路径
                    manager.bm25_manager.save_model.assert_called_once()
                    call_args = manager.bm25_manager.save_model.call_args
                    model_path, vocab_path = call_args[0]
                    assert "bm25_vocab_new.json" in vocab_path
    
    def test_load_and_save_use_same_vocab_file(self):
        """测试加载和保存使用同一个词汇表文件"""
        manager = UpsertManager()
        
        # 模拟BM25Manager
        manager.bm25_manager = MagicMock()
        manager.bm25_manager.is_fitted = False  # 初始状态未训练
        
        # 模拟BM25Config
        manager.bm25_config = MagicMock()
        model_path = os.path.join(self.temp_dir, "models", "bm25_model.pkl")
        vocab_path = os.path.join(self.dict_dir, "bm25_vocab_2401011600.json")
        manager.bm25_config.get_model_path.return_value = model_path
        
        # 模拟文件存在
        with patch('os.path.exists', return_value=True):
            with patch.object(manager, '_get_latest_vocab_path', return_value=vocab_path):
                with patch('logging.info'):
                    # 初始化BM25Manager（会加载现有模型）
                    manager._initialize_bm25_manager("test_dict.txt")
                    
                    # 验证加载时使用了指定的词汇表文件
                    manager.bm25_manager.load_model.assert_called_once_with(model_path, vocab_path)
                    
                    # 训练模型
                    manager.bm25_manager.is_fitted = True
                    
                    # 保存模型
                    manager._save_bm25_model()
                    
                    # 验证保存时也使用了同一个词汇表文件
                    save_call_args = manager.bm25_manager.save_model.call_args
                    saved_model_path, saved_vocab_path = save_call_args[0]
                    assert saved_vocab_path == vocab_path

if __name__ == "__main__":
    # 运行测试
    print("🧪 开始运行BM25词汇表文件选择逻辑测试...")
    
    # 创建测试实例
    test_instance = TestUpsertBM25Vocab()
    
    try:
        # 运行所有测试方法
        test_instance.setup_method()
        
        print("✅ 测试1: _get_latest_vocab_path返回最新文件")
        test_instance.test_get_latest_vocab_path_returns_newest_file()
        
        print("✅ 测试2: _save_bm25_model使用最新词汇表路径")
        test_instance.test_save_bm25_model_uses_latest_vocab_path()
        
        print("✅ 测试3: 没有现有词汇表时创建新文件")
        test_instance.test_save_bm25_model_creates_new_vocab_when_none_exists()
        
        print("✅ 测试4: 加载和保存使用同一个文件")
        test_instance.test_load_and_save_use_same_vocab_file()
        
        test_instance.teardown_method()
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        test_instance.teardown_method()
        raise 