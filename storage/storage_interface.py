from abc import ABC, abstractmethod
from typing import Any

class StorageInterface(ABC):
    """
    存储接口基类，所有具体存储方式需继承并实现storage方法。
    """
    @abstractmethod
    def storage(self, data: Any, *args, **kwargs):
        raise NotImplementedError("storage方法需由子类实现")

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        raise NotImplementedError("load方法需由子类实现") 
    
    @abstractmethod
    def exists(self, path: str, *args, **kwargs) -> bool:
        """判断指定路径/资源是否存在"""
        raise NotImplementedError("exists方法需由子类实现")

    @abstractmethod
    def delete(self, path: str, *args, **kwargs):
        """删除指定路径/资源"""
        raise NotImplementedError("delete方法需由子类实现")