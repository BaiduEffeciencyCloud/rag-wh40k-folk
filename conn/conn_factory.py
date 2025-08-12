from .osconn import OpenSearchConnection
from .pineconeconn import PineconeConnection
from config import PINECONE_API_KEY, PINECONE_INDEX, OPENSEARCH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD,OPENSEARCH_INDEX


class ConnectionFactory:
    def __init__(self, connection_type: str):
        self.connection_type = connection_type
    
    @staticmethod
    def create_conn(connection_type: str):
        if connection_type == "opensearch":
            return OpenSearchConnection(
                index=OPENSEARCH_INDEX,
                uri=OPENSEARCH_URI,
                username=OPENSEARCH_USERNAME,
                password=OPENSEARCH_PASSWORD
            )
        elif connection_type == "pinecone":
            return PineconeConnection(
                index=PINECONE_INDEX,
                api_key=PINECONE_API_KEY
            )
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")