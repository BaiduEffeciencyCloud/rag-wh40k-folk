"""
配置化存储选择器
"""

import config
from .storage_interface import StorageInterface
from .oss_storage import OSSStorage
from dataupload.raw_chunk_local_storage import RawChunkStorage
import logging
from typing import Any, Optional

class ConfigurableStorage:
    """配置化存储选择器"""
    
    def __init__(self, **kwargs):
        """
        初始化配置化存储
        
        Args:
            **kwargs: 存储配置参数
        """
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self._storage = None
    
    def get_storage(self) -> StorageInterface:
        """根据配置获取存储实例"""
        if self._storage is None:
            if config.CLOUD_STORAGE:
                self.logger.info("使用OSS存储")
                self._storage = OSSStorage(**self.kwargs)
            else:
                self.logger.info("使用本地存储")
                # 过滤掉OSS相关的参数
                local_kwargs = {k: v for k, v in self.kwargs.items() 
                               if k not in ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name', 'region']}
                self._storage = RawChunkStorage(**local_kwargs)
        
        return self._storage
    
    def storage(self, *, data: Any, path: str, **kwargs) -> bool:
        """存储数据"""
        # 统一关键字参数传递，避免位置参数误用
        return self.get_storage().storage(data=data, oss_path=path, **kwargs)
    
    def load(self, path: str, **kwargs) -> Optional[bytes]:
        """加载数据"""
        return self.get_storage().load(path, **kwargs)
    
    def exists(self, path: str, **kwargs) -> bool:
        """检查文件是否存在"""
        return self.get_storage().exists(path, **kwargs)
    
    def delete(self, path: str, **kwargs) -> bool:
        """删除文件"""
        return self.get_storage().delete(path, **kwargs)
