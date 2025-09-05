"""
内存缓存管理器
"""

from typing import Optional, Dict, Any
import logging

class MemoryCache:
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 100):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存项数量
        """
        self.cache = {}
        self.metadata = {}
        self.max_size = max_size
        self.access_count = {}
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[bytes]:
        """获取缓存内容"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: bytes, metadata: dict = None):
        """设置缓存内容"""
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        
        self.cache[key] = value
        if metadata:
            self.metadata[key] = metadata
        self.access_count[key] = 1
    
    def _evict_least_used(self):
        """淘汰最少使用的缓存项"""
        if not self.access_count:
            return
        
        least_used_key = min(self.access_count.keys(), 
                           key=lambda k: self.access_count[k])
        del self.cache[least_used_key]
        del self.access_count[least_used_key]
        if least_used_key in self.metadata:
            del self.metadata[least_used_key]

class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        self.cache = MemoryCache()
        self.logger = logging.getLogger(__name__)
    
    def is_file_changed(self, oss_path: str, local_etag: str = None) -> bool:
        """检查OSS文件是否发生变化"""
        try:
            # 获取OSS文件元数据
            oss_metadata = self.get_oss_metadata(oss_path)
            cached_metadata = self.cache.metadata.get(oss_path)
            
            if not cached_metadata:
                return True  # 首次访问，需要下载
            
            # 比较ETag
            if oss_metadata.get('etag') != cached_metadata.get('etag'):
                return True
            
            # 比较最后修改时间
            if oss_metadata.get('last_modified') != cached_metadata.get('last_modified'):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查文件变化失败: {oss_path}, 错误: {e}")
            return True  # 出错时重新下载
    
    def get_oss_metadata(self, oss_path: str) -> Dict[str, Any]:
        """获取OSS文件元数据"""
        # 这里需要实际的OSS客户端来获取元数据
        # 暂时返回模拟数据
        return {
            'etag': 'mock_etag',
            'last_modified': 'mock_timestamp',
            'size': 1024
        }
