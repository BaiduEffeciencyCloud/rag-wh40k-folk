#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预加载服务
在服务启动时预加载所有需要的模型
"""

import logging
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class ModelService:
    """模型预加载服务"""
    
    def __init__(self):
        self.model_manager = ModelManager()
    
    async def preload_models(self):
        """预加载所有需要的模型"""
        logger.info("🚀 开始预加载模型...")
        
        # 预加载 SentenceTransformer 模型
        success = self.model_manager.load_sentence_transformer()
        
        if success:
            logger.info("✅ 模型预加载完成")
        else:
            logger.warning("⚠️ 模型预加载失败，将使用懒加载模式")
        
        return success
    
    async def cleanup_models(self):
        """清理模型资源"""
        logger.info("🧹 开始清理模型服务资源...")
        
        try:
            # 清理 ModelManager 资源
            self.model_manager.cleanup()
            logger.info("✅ 模型服务资源清理完成")
        except Exception as e:
            logger.error(f"模型服务资源清理失败: {e}")
    
    def get_model_manager(self) -> ModelManager:
        """获取模型管理器实例"""
        return self.model_manager
