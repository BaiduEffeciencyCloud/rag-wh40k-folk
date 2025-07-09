#!/usr/bin/env python3
"""
BM25配置管理测试用例
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch
from datetime import datetime

class TestBM25Config(unittest.TestCase):
    """BM25配置管理测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 保存原始环境变量
        self.original_env = {}
        for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE', 
                   'BM25_MODEL_DIR', 'BM25_VOCAB_DIR', 'BM25_MODEL_FILENAME',
                   'BM25_ENABLE_INCREMENTAL', 'BM25_SAVE_AFTER_UPDATE']:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
    
    def tearDown(self):
        """测试后清理"""
        # 恢复原始环境变量
        for key, value in self.original_env.items():
            os.environ[key] = value
        # 清除测试时设置的环境变量
        for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE',
                   'BM25_MODEL_DIR', 'BM25_VOCAB_DIR', 'BM25_MODEL_FILENAME',
                   'BM25_ENABLE_INCREMENTAL', 'BM25_SAVE_AFTER_UPDATE']:
            if key in os.environ and key not in self.original_env:
                del os.environ[key]
    
    def test_bm25_config_default_values(self):
        """测试BM25配置默认值"""
        # 确保环境变量被清除
        for key in ['BM25_K1', 'BM25_B', 'BM25_MIN_FREQ', 'BM25_MAX_VOCAB_SIZE']:
            if key in os.environ:
                del os.environ[key]
        
        from dataupload.bm25_config import BM25Config
        config = BM25Config()
        
        # 验证默认值
        self.assertEqual(config.k1, 1.5)
        self.assertEqual(config.b, 0.75)
        self.assertEqual(config.min_freq, 5)
        self.assertEqual(config.max_vocab_size, 10000)
        self.assertEqual(config.model_dir, 'models')
        self.assertEqual(config.vocab_dir, 'dict')
        self.assertEqual(config.model_filename, 'bm25_model.pkl')
        self.assertTrue(config.enable_incremental)
        self.assertTrue(config.save_after_update)
    
    def test_bm25_config_environment_override(self):
        """测试环境变量覆盖配置"""
        # 设置测试环境变量
        os.environ['BM25_K1'] = '2.0'
        os.environ['BM25_B'] = '0.8'
        os.environ['BM25_MIN_FREQ'] = '3'
        os.environ['BM25_MAX_VOCAB_SIZE'] = '5000'
        os.environ['BM25_MODEL_DIR'] = 'custom_models'
        os.environ['BM25_VOCAB_DIR'] = 'custom_dict'
        os.environ['BM25_MODEL_FILENAME'] = 'custom_model.pkl'
        os.environ['BM25_ENABLE_INCREMENTAL'] = 'false'
        os.environ['BM25_SAVE_AFTER_UPDATE'] = 'false'
        
        from dataupload.bm25_config import BM25Config
        config = BM25Config()
        
        # 验证环境变量覆盖
        self.assertEqual(config.k1, 2.0)
        self.assertEqual(config.b, 0.8)
        self.assertEqual(config.min_freq, 3)
        self.assertEqual(config.max_vocab_size, 5000)
        self.assertEqual(config.model_dir, 'custom_models')
        self.assertEqual(config.vocab_dir, 'custom_dict')
        self.assertEqual(config.model_filename, 'custom_model.pkl')
        self.assertFalse(config.enable_incremental)
        self.assertFalse(config.save_after_update)
    
    def test_bm25_config_path_generation(self):
        """测试路径生成功能"""
        from dataupload.bm25_config import BM25Config
        config = BM25Config()
        
        # 测试模型路径生成
        model_path = config.get_model_path()
        expected_model_path = os.path.join('models', 'bm25_model.pkl')
        self.assertEqual(model_path, expected_model_path)
        
        # 测试词汇表路径生成（带时间戳）
        vocab_path = config.get_vocab_path()
        self.assertIn('dict/bm25_vocab_', vocab_path)
        self.assertTrue(vocab_path.endswith('.json'))
        
        # 测试自定义时间戳
        custom_timestamp = '2507051200'
        vocab_path_custom = config.get_vocab_path(custom_timestamp)
        expected_vocab_path = os.path.join('dict', f'bm25_vocab_{custom_timestamp}.json')
        self.assertEqual(vocab_path_custom, expected_vocab_path)
    
    def test_bm25_config_validation(self):
        """测试配置参数验证"""
        from dataupload.bm25_config import BM25Config
        
        # 测试正常配置
        config = BM25Config()
        self.assertTrue(config.validate())
        
        # 测试无效配置（通过环境变量设置）
        os.environ['BM25_K1'] = '-1.0'  # 无效的k1值
        os.environ['BM25_B'] = '2.0'    # 无效的b值
        
        config = BM25Config()
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_bm25_config_serialization(self):
        """测试配置序列化功能"""
        from dataupload.bm25_config import BM25Config
        config = BM25Config()
        
        # 测试转换为字典
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('k1', config_dict)
        self.assertIn('b', config_dict)
        self.assertIn('min_freq', config_dict)
        self.assertIn('max_vocab_size', config_dict)
        
        # 测试从字典创建
        new_config = BM25Config.from_dict(config_dict)
        self.assertEqual(new_config.k1, config.k1)
        self.assertEqual(new_config.b, config.b)
        self.assertEqual(new_config.min_freq, config.min_freq)
        self.assertEqual(new_config.max_vocab_size, config.max_vocab_size)
    
    def test_bm25_config_file_operations(self):
        """测试配置文件操作"""
        from dataupload.bm25_config import BM25Config
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'bm25_config.json')
            config = BM25Config()
            
            # 测试保存配置到文件
            config.save_to_file(config_file)
            self.assertTrue(os.path.exists(config_file))
            
            # 测试从文件加载配置
            loaded_config = BM25Config.load_from_file(config_file)
            self.assertEqual(loaded_config.k1, config.k1)
            self.assertEqual(loaded_config.b, config.b)
            self.assertEqual(loaded_config.min_freq, config.min_freq)
            self.assertEqual(loaded_config.max_vocab_size, config.max_vocab_size)
    
    def test_bm25_config_integration_with_manager(self):
        """测试配置与BM25Manager的集成"""
        from dataupload.bm25_config import BM25Config
        from dataupload.bm25_manager import BM25Manager
        
        # 创建配置
        config = BM25Config()
        
        # 使用配置参数创建BM25Manager
        manager = BM25Manager(
            k1=config.k1,
            b=config.b,
            min_freq=config.min_freq,
            max_vocab_size=config.max_vocab_size
        )
        
        # 验证配置参数正确传递
        self.assertEqual(manager.k1, config.k1)
        self.assertEqual(manager.b, config.b)
        self.assertEqual(manager.min_freq, config.min_freq)
        self.assertEqual(manager.max_vocab_size, config.max_vocab_size)
        
        # 验证配置路径生成
        expected_model_path = config.get_model_path()
        expected_vocab_path = config.get_vocab_path()
        
        self.assertIn('models', expected_model_path)
        self.assertIn('bm25_model.pkl', expected_model_path)
        self.assertIn('dict', expected_vocab_path)
        self.assertIn('bm25_vocab_', expected_vocab_path)
    
    def test_bm25_config_error_handling(self):
        """测试配置错误处理"""
        from dataupload.bm25_config import BM25Config
        
        # 测试无效的环境变量值
        os.environ['BM25_K1'] = 'invalid_float'
        
        with self.assertRaises(ValueError):
            config = BM25Config()
        
        # 测试无效的布尔值
        os.environ['BM25_K1'] = '1.5'  # 恢复有效值
        os.environ['BM25_ENABLE_INCREMENTAL'] = 'invalid_bool'
        
        with self.assertRaises(ValueError):
            config = BM25Config()

if __name__ == '__main__':
    unittest.main() 