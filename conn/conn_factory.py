from .osconn import OpenSearchConnection
from .pineconeconn import PineconeConnection

class ConnectionFactory:
    def __init__(self, connection_type: str):
        self.connection_type = connection_type
    @staticmethod
    def create_conn(self):
        if self.connection_type == "opensearch":
            return OpenSearchConnection()
        elif self.connection_type == "pinecone":
            return PineconeConnection()
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")