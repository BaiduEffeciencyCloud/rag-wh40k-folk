#!/usr/bin/env python3
"""
BM25配置管理类
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from setuptools._distutils.util import strtobool
import config as project_config

logger = logging.getLogger(__name__)

class BM25Config:
    """BM25配置管理类"""
    
    def __init__(self):
        """初始化BM25配置"""
        try:
            # 直接使用项目 config.py 常量，缺失则退回默认
            # BM25算法参数
            self.k1 = getattr(project_config, 'BM25_K1', 1.5)
            self.b = getattr(project_config, 'BM25_B', 0.75)
            self.min_freq = getattr(project_config, 'BM25_MIN_FREQ', 5)
            self.max_vocab_size = getattr(project_config, 'BM25_MAX_VOCAB_SIZE', 10000)

            # 文件路径配置
            self.model_dir = getattr(project_config, 'BM25_MODEL_DIR', 'models')
            self.vocab_dir = getattr(project_config, 'BM25_VOCAB_DIR', 'dict')
            self.model_filename = getattr(project_config, 'BM25_MODEL_FILENAME', 'bm25_model.pkl')
            self.vocab_filename_pattern = getattr(project_config, 'BM25_VOCAB_FILENAME_PATTERN', 'bm25_vocab_{timestamp}.json')

            # 训练配置
            self.enable_incremental = getattr(project_config, 'BM25_ENABLE_INCREMENTAL', True)
            self.save_after_update = getattr(project_config, 'BM25_SAVE_AFTER_UPDATE', True)

            # 性能配置
            self.batch_size = getattr(project_config, 'BM25_BATCH_SIZE', 1000)
            self.cache_size = getattr(project_config, 'BM25_CACHE_SIZE', 1000)
            
        except Exception as e:
            logger.error(f"BM25配置初始化失败: {e}")
            raise
    
    def _get_float(self, key: str, default: float) -> float:
        """获取浮点数配置，严格校验"""
        s = os.getenv(key, str(default))
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"环境变量 {key} 值非法，无法转换为 float: {s}")
    
    def _get_int(self, key: str, default: int) -> int:
        """获取整数配置，严格校验"""
        s = os.getenv(key, str(default))
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"环境变量 {key} 值非法，无法转换为 int: {s}")
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """获取布尔值配置，严格校验"""
        s = os.getenv(key, str(default)).strip()
        try:
            return bool(strtobool(s))
        except ValueError:
            raise ValueError(f"环境变量 {key} 值非法，无法转换为 bool: {s}")
    
    def _get_str(self, key: str, default: str) -> str:
        """获取字符串配置"""
        return os.getenv(key, default)
    
    def validate(self) -> bool:
        """验证配置参数"""
        # 验证BM25算法参数
        if self.k1 <= 0:
            raise ValueError(f"BM25_K1必须大于0，当前值: {self.k1}")
        if not 0 <= self.b <= 1:
            raise ValueError(f"BM25_B必须在[0,1]范围内，当前值: {self.b}")
        if self.min_freq < 1:
            raise ValueError(f"BM25_MIN_FREQ必须大于等于1，当前值: {self.min_freq}")
        if self.max_vocab_size < 1:
            raise ValueError(f"BM25_MAX_VOCAB_SIZE必须大于等于1，当前值: {self.max_vocab_size}")
        
        # 验证性能配置
        if self.batch_size < 1:
            raise ValueError(f"BM25_BATCH_SIZE必须大于等于1，当前值: {self.batch_size}")
        if self.cache_size < 1:
            raise ValueError(f"BM25_CACHE_SIZE必须大于等于1，当前值: {self.cache_size}")
        
        return True
    
    def get_model_path(self) -> str:
        """获取模型文件路径"""
        return os.path.join(self.model_dir, self.model_filename)
    
    def get_vocab_path(self, timestamp: Optional[str] = None) -> str:
        """获取词汇表文件路径"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%y%m%d%H%M")
        return os.path.join(self.vocab_dir, self.vocab_filename_pattern.format(timestamp=timestamp))
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            'k1': self.k1,
            'b': self.b,
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size,
            'model_dir': self.model_dir,
            'vocab_dir': self.vocab_dir,
            'model_filename': self.model_filename,
            'vocab_filename_pattern': self.vocab_filename_pattern,
            'enable_incremental': self.enable_incremental,
            'save_after_update': self.save_after_update,
            'batch_size': self.batch_size,
            'cache_size': self.cache_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BM25Config':
        """从字典创建配置对象"""
        config = cls.__new__(cls)
        
        # 设置配置属性
        config.k1 = config_dict.get('k1', 1.5)
        config.b = config_dict.get('b', 0.75)
        config.min_freq = config_dict.get('min_freq', 5)
        config.max_vocab_size = config_dict.get('max_vocab_size', 10000)
        config.model_dir = config_dict.get('model_dir', 'models')
        config.vocab_dir = config_dict.get('vocab_dir', 'dict')
        config.model_filename = config_dict.get('model_filename', 'bm25_model.pkl')
        config.vocab_filename_pattern = config_dict.get('vocab_filename_pattern', 'bm25_vocab_{timestamp}.json')
        config.enable_incremental = config_dict.get('enable_incremental', True)
        config.save_after_update = config_dict.get('save_after_update', True)
        config.batch_size = config_dict.get('batch_size', 1000)
        config.cache_size = config_dict.get('cache_size', 1000)
        
        # 验证配置
        config.validate()
        
        return config
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"BM25配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存BM25配置失败: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BM25Config':
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = cls.from_dict(config_dict)
            logger.info(f"BM25配置已从文件加载: {file_path}")
            return config
        except Exception as e:
            logger.error(f"加载BM25配置失败: {e}")
            raise
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"BM25Config(k1={self.k1}, b={self.b}, min_freq={self.min_freq}, max_vocab_size={self.max_vocab_size})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"BM25Config(k1={self.k1}, b={self.b}, min_freq={self.min_freq}, max_vocab_size={self.max_vocab_size}, model_dir='{self.model_dir}', vocab_dir='{self.vocab_dir}')" 