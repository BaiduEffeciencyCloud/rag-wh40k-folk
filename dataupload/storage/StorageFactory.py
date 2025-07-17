from .Local_RawChunk_Storage import LocalStorage
from .StorageInterface import StorageInterface

class StorageFactory:
    """
    存储工厂，根据参数生成对应的存储实现类
    """
    @staticmethod
    def Create_storage(storage_type: str = "local", **kwargs) -> StorageInterface:
        if storage_type == "local":
            return LocalStorage(**kwargs)
        # 未来可扩展: if storage_type == "mongo": ...
        # 未来可扩展: if storage_type == "s3": ...
        raise ValueError(f"不支持的存储类型: {storage_type}") 