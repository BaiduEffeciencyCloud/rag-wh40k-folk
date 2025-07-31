"""
战锤40K专业术语词典管理模块

提供专业术语词典的加载、更新、管理功能
"""

from .vocab_mgr import VocabMgr
from .vocab_updater import VocabUp
from .vocab_loader import VocabLoad

__all__ = ['VocabMgr', 'VocabUp', 'VocabLoad'] 