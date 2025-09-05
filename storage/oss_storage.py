"""
阿里云OSS存储实现
"""

import oss2
from .storage_interface import StorageInterface
import os
import tempfile
from typing import Any, Optional, Dict
import logging

class OSSStorage(StorageInterface):
    """阿里云OSS存储实现"""
    
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
    
    def storage(self, data: Any, oss_path: str, **kwargs) -> bool:
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
            return True
            
        except Exception as e:
            self.logger.error(f"文件上传失败: {oss_path}, 错误: {e}")
            return False
    
    def load(self, oss_path: str, **kwargs) -> Optional[bytes]:
        """从OSS下载文件"""
        try:
            # 使用oss2的方式下载
            result = self.bucket.get_object(oss_path)
            content = result.read()
            self.logger.info(f"文件下载成功: {oss_path}")
            return content
            
        except Exception as e:
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
            return True
            
        except Exception as e:
            self.logger.error(f"文件删除失败: {oss_path}, 错误: {e}")
            return False
