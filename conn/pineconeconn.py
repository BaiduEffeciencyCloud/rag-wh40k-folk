from .coninterface import ConnectionInterface


class PineconeConnection(ConnectionInterface):
    def __init__(self, index: str = None, api_key: str = None):
        self.index = index
        self.api_key = api_key

    def search(self, query: str, filter: dict) -> list:
        pass
    
    def adaptive_data(self, chunks: list) -> dict:
        pass

    def upload_data(self, data: dict) -> dict:
        pass