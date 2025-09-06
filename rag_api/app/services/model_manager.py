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
            
            # 设置中国镜像站点
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            os.environ['HF_HUB_OFFLINE'] = '0'
            
            # 导入并加载模型

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
