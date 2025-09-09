#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器 - 单例模式
统一管理所有预加载的模型
"""

import os
import logging
import threading
from typing import Optional, Dict, Any

# ===== 在模块导入阶段设置国内网络必要环境变量（确保下方库初始化前生效）=====
os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '0')
os.environ.setdefault('HF_HUB_TIMEOUT', '120')

try:
    _shm_dir = '/dev/shm/hf_cache'
    _tmp_dir = '/tmp/hf_cache'
    _cache_dir = _shm_dir if os.path.isdir('/dev/shm') else _tmp_dir
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', _cache_dir)
    os.environ.setdefault('TRANSFORMERS_CACHE', _cache_dir)
except Exception:
    # 缓存目录失败则交由默认行为处理
    pass

from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models: Dict[str, Any] = {}
            self._initialized = True
            logger.info("ModelManager 初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，清理资源"""
        self.cleanup()
        return False  # 不抑制异常
    
    def cleanup(self):
        """手动清理资源"""
        try:
            logger.info("🧹 开始清理 ModelManager 资源...")
            
            # 清理模型资源
            for model_name, model in self.models.items():
                try:
                    # 如果模型有清理方法，调用它
                    if hasattr(model, 'cpu'):
                        model.cpu()  # 将模型移到CPU
                    if hasattr(model, 'eval'):
                        model.eval()  # 设置为评估模式
                    logger.info(f"模型 {model_name} 资源已清理")
                except Exception as e:
                    logger.warning(f"清理模型 {model_name} 时出错: {e}")
            
            # 清空模型字典
            self.models.clear()
            
            # 重置初始化状态
            self._initialized = False
            
            logger.info("✅ ModelManager 资源清理完成")
            
        except Exception as e:
            logger.error(f"ModelManager 资源清理失败: {e}")
    
    def load_sentence_transformer(self, model_name: str = 'shibing624/text2vec-base-chinese') -> bool:
        """预加载 SentenceTransformer 模型"""
        try:
            if model_name in self.models:
                logger.info(f"模型 {model_name} 已存在，跳过加载")
                return True
            
            logger.info(f"开始预加载模型: {model_name}")
            # ===== 国内网络优化：必要环境变量设置（仅在未显式设置时注入）=====
            # 镜像端点
            os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', 'https://hf-mirror.com')
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            # 传输优化与超时
            os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
            os.environ.setdefault('HF_HUB_OFFLINE', '0')
            os.environ.setdefault('HF_HUB_TIMEOUT', '30')
            
            # 临时/内存缓存目录（不做长期落盘）：优先 /dev/shm，其次 /tmp
            try:
                shm_dir = '/dev/shm/hf_cache'
                tmp_dir = '/tmp/hf_cache'
                cache_dir = shm_dir if os.path.isdir('/dev/shm') else tmp_dir
                os.makedirs(cache_dir, exist_ok=True)
                os.environ.setdefault('HUGGINGFACE_HUB_CACHE', cache_dir)
                os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
                logger.info(f"HuggingFace 缓存目录: {cache_dir}")
            except Exception as _e:
                logger.warning(f"缓存目录初始化失败，继续使用默认缓存: {_e}")

            # 导入并加载模型（保持调用方式不变）
            model = SentenceTransformer(model_name)
            self.models[model_name] = model
            
            logger.info(f"模型 {model_name} 预加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型 {model_name} 预加载失败: {e}")
            return False
    
    def get_sentence_transformer(self, model_name: str = 'shibing624/text2vec-base-chinese') -> Optional[Any]:
        """获取预加载的 SentenceTransformer 模型"""
        return self.models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self.models
    
    def get_loaded_models(self) -> list:
        """获取已加载的模型列表"""
        return list(self.models.keys())
