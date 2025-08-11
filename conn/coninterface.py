from abc import ABC, abstractmethod

class ConnectionInterface(ABC):
    @abstractmethod
    def search(self, query: str,filter:dict) -> list:
        pass
    
    @abstractmethod
    def adaptive_data(self, chunks: list) -> dict:
        pass

    @abstractmethod
    def upload_data(self, data: dict) -> dict:
        pass
    
