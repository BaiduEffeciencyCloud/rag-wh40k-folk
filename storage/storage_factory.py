import config
from dataupload.raw_chunk_local_storage import RawChunkStorage
from .storage_interface import StorageInterface
from .oss_storage import OSSStorage
from .configurable_storage import ConfigurableStorage

class StorageFactory:
    """
    存储工厂，支持配置化存储选择
    """
    @staticmethod
    def create_storage(**kwargs) -> StorageInterface:
        """
        创建存储实例（根据config.CLOUD_STORAGE自动选择）
        
        Args:
            **kwargs: 存储配置参数
        """
        if config.CLOUD_STORAGE:
            # 使用OSS存储
            return OSSStorage(
                access_key_id=kwargs.get('access_key_id'),
                access_key_secret=kwargs.get('access_key_secret'),
                endpoint=kwargs.get('endpoint'),
                bucket_name=kwargs.get('bucket_name'),
                region=kwargs.get('region', 'oss-cn-hangzhou')
            )
        else:
            # 使用本地存储
            return RawChunkStorage(**kwargs)
    
    @staticmethod
    def create_configurable_storage(**kwargs) -> ConfigurableStorage:
        """
        创建配置化存储实例（支持运行时切换）
        
        Args:
            **kwargs: 存储配置参数
        """
        return ConfigurableStorage(**kwargs) 