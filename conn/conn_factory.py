from .osconn import OpenSearchConnection
from .pineconeconn import PineconeConnection

class ConnectionFactory:
    def __init__(self, connection_type: str):
        self.connection_type = connection_type
    
    @staticmethod
    def create_conn(connection_type: str):
        if connection_type == "opensearch":
            return OpenSearchConnection()
        elif connection_type == "pinecone":
            return PineconeConnection()
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")