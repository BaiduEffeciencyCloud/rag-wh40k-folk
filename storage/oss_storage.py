"""
阿里云OSS存储实现
"""

import oss2
from .storage_interface import StorageInterface
import os
import tempfile
from typing import Any, Optional, Dict
import logging
import time
import threading
import config

class OSSStorage(StorageInterface):
    """阿里云OSS存储实现"""
    # 进程内共享缓存（跨实例复用）
    _CACHE: Dict[str, Dict[str, Any]] = {}
    _LOCK = threading.Lock()
    
    def __init__(self, 
                 access_key_id: str,
                 access_key_secret: str,
                 endpoint: str,
                 bucket_name: str,
                 region: str = "cn-hangzhou"):
        """
        初始化OSS存储
        
        Args:
            access_key_id: 阿里云AccessKey ID
            access_key_secret: 阿里云AccessKey Secret
            endpoint: OSS服务端点
            bucket_name: 存储桶名称
            region: 地域
        """
        # 按照CSDN文章的方式初始化
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        # 缓存改为进程级共享（见类变量 _CACHE / _LOCK）
    
    def storage(self, *, data: Any, oss_path: str, **kwargs) -> bool:
        """上传数据到OSS"""
        try:
            # 处理不同类型的数据
            if isinstance(data, str):
                content = data.encode('utf-8')
            elif isinstance(data, bytes):
                content = data
            elif hasattr(data, 'read'):
                content = data.read()
            else:
                content = str(data).encode('utf-8')
            
            # 上传到OSS (使用oss2的方式)
            result = self.bucket.put_object(oss_path, content)
            self.logger.info(f"文件上传成功: {oss_path}, 状态码: {result.status}")
            # 缓存失效（写后）
            with OSSStorage._LOCK:
                if oss_path in OSSStorage._CACHE:
                    OSSStorage._CACHE.pop(oss_path, None)
                    self.logger.info(f"缓存失效(写入后): {oss_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件上传失败: {oss_path}, 错误: {e}")
            return False
    
    def load(self, oss_path: str, **kwargs) -> Optional[bytes]:
        """从OSS下载文件，带内存缓存（ETag/TTL/LRU）。"""
        use_cache = getattr(config, 'OSS_CACHE_ENABLED', True)
        allowed_prefixes = getattr(config, 'OSS_CACHE_ALLOWED_KEY_PREFIXES', [])
        ttl = getattr(config, 'OSS_CACHE_TTL_SECONDS', 60)
        max_items = getattr(config, 'OSS_CACHE_MAX_ITEMS', 512)
        max_bytes = getattr(config, 'OSS_CACHE_MAX_BYTES', None)
        validate_mode = getattr(config, 'OSS_CACHE_VALIDATE_MODE', 'etag')
        max_obj_size = getattr(config, 'OSS_CACHE_MAX_OBJECT_SIZE', 5 * 1024 * 1024)

        def _allowed(key: str) -> bool:
            if not allowed_prefixes:
                return True
            return any(key.startswith(p) for p in allowed_prefixes)

        now = time.time()
        etag = None
        cached = None

        if use_cache and _allowed(oss_path):
            with OSSStorage._LOCK:
                cached = OSSStorage._CACHE.get(oss_path)
                if cached:
                    etag = cached.get('etag')
                    cached['last'] = now

        headers = {}
        if use_cache and etag and validate_mode in ('etag', 'both'):
            # 规范化 If-None-Match（带引号）
            quoted = etag if etag.startswith('"') and etag.endswith('"') else f'"{etag}"'
            headers['If-None-Match'] = quoted
            self.logger.info(f"发送条件请求 If-None-Match={quoted} key={oss_path}")

        try:
            # 条件请求
            result = self.bucket.get_object(oss_path, headers=headers) if headers else self.bucket.get_object(oss_path)
            status = getattr(result, 'status', 200)
            resp_etag = None
            try:
                if hasattr(result, 'headers') and result.headers:
                    # 兼容不同大小写与带引号的ETag
                    resp_etag = result.headers.get('etag') or result.headers.get('ETag')
                    if isinstance(resp_etag, str) and resp_etag.startswith('"') and resp_etag.endswith('"'):
                        resp_etag = resp_etag.strip('"')
            except Exception:
                resp_etag = None

            if status == 304 and cached is not None:
                # 未修改，返回缓存
                self.logger.info(f"缓存命中(304): {oss_path}")
                return cached.get('bytes')

            # 读取主体
            content = result.read()
            # 可选：对象过大不入缓存
            if use_cache and _allowed(oss_path) and (max_obj_size is None or len(content) <= max_obj_size):
                with OSSStorage._LOCK:
                    # LRU 淘汰
                    if max_items and len(OSSStorage._CACHE) >= max_items and oss_path not in OSSStorage._CACHE:
                        # 淘汰 last 最小者
                        oldest_key = min(OSSStorage._CACHE.items(), key=lambda kv: kv[1].get('last', kv[1].get('ts', 0)))[0]
                        OSSStorage._CACHE.pop(oldest_key, None)
                    OSSStorage._CACHE[oss_path] = {
                        'etag': (resp_etag or etag),
                        'bytes': content,
                        'ts': now,
                        'last': now,
                    }
                    self.logger.info(f"缓存更新: key={oss_path}, etag={(resp_etag or etag)} bytes={len(content)}")
            else:
                if use_cache and not _allowed(oss_path):
                    self.logger.info(f"缓存跳过(前缀不允许): key={oss_path}")
                if use_cache and max_obj_size is not None and len(content) > max_obj_size:
                    self.logger.info(f"缓存跳过(对象过大): key={oss_path}, size={len(content)} > {max_obj_size}")
            self.logger.info(f"文件下载成功: {oss_path}")
            return content

        except Exception as e:
            # 识别 SDK 将 304 当异常抛出的情况
            is_304 = False
            try:
                status = getattr(e, 'status', None)
                if status == 304:
                    is_304 = True
            except Exception:
                pass
            if not is_304:
                try:
                    msg = str(e)
                    if ("status': 304" in msg) or ("status: 304" in msg) or (" 304" in msg):
                        is_304 = True
                except Exception:
                    pass
            if is_304 and cached is not None and 'bytes' in cached:
                self.logger.info(f"缓存命中(304): {oss_path}")
                return cached.get('bytes')
            # 其他异常：若有缓存则降级返回缓存
            if cached is not None and 'bytes' in cached:
                self.logger.warning(f"下载异常，返回缓存: {oss_path}, 错误: {e}")
                return cached.get('bytes')
            self.logger.error(f"文件下载失败: {oss_path}, 错误: {e}")
            return None
    
    def exists(self, oss_path: str, **kwargs) -> bool:
        """检查OSS文件是否存在"""
        try:
            # 使用oss2的方式检查文件是否存在
            return self.bucket.object_exists(oss_path)
            
        except Exception as e:
            self.logger.error(f"检查文件存在性失败: {oss_path}, 错误: {e}")
            return False
    
    def delete(self, oss_path: str, **kwargs) -> bool:
        """删除OSS文件"""
        try:
            # 使用oss2的方式删除文件
            self.bucket.delete_object(oss_path)
            self.logger.info(f"文件删除成功: {oss_path}")
            # 缓存失效（删后）
            with OSSStorage._LOCK:
                if oss_path in OSSStorage._CACHE:
                    OSSStorage._CACHE.pop(oss_path, None)
                    self.logger.info(f"缓存失效(删除后): {oss_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件删除失败: {oss_path}, 错误: {e}")
            return False

    # ===== 对外更优雅的API =====
    def get_object_bytes(self, key: str) -> Optional[bytes]:
        """获取对象字节（封装缓存/条件请求细节）。"""
        return self.load(key)

    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> list:
        """列举Bucket根目录下指定前缀的对象Key列表。
        云端优先读取流程依赖该方法获取最新的 userdict 与 freq vocab。
        失败返回空列表，不抛异常。
        """
        try:
            # 使用oss2的List Objects接口
            result = self.bucket.list_objects(prefix=prefix, max_keys=max_keys)
            # 兼容不同SDK版本的返回结构
            object_list = getattr(result, 'object_list', None)
            if object_list is None and hasattr(result, 'objects'):
                object_list = result.objects
            if not object_list:
                return []
            keys = []
            for obj in object_list:
                key = getattr(obj, 'key', None)
                if key:
                    keys.append(key)
            self.logger.info(f"列举对象完成: prefix={prefix}, count={len(keys)}")
            return keys
        except Exception as e:
            self.logger.warning(f"列举对象失败: prefix={prefix}, 错误: {e}")
            return []
