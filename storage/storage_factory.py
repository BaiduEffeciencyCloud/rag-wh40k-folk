from dataupload.raw_chunk_local_storage import RawChunkStorage
from .storage_interface import StorageInterface

class StorageFactory:
    """
    存储工厂，根据参数生成对应的存储实现类
    """
    @staticmethod
    def create_storage(storage_type: str = "local", **kwargs) -> StorageInterface:
        if storage_type == "local":
            return RawChunkStorage(**kwargs)
        # 未来可扩展: if storage_type == "mongo": ...
        # 未来可扩展: if storage_type == "s3": ...
        raise ValueError(f"不支持的存储类型: {storage_type}") 