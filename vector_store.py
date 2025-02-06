import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def create_collection(self, collection_name: str):
        try:
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"Collection {collection_name} might already exist: {str(e)}")
            return self.client.get_collection(collection_name)

    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        try:
            collection = self.client.get_collection(collection_name)
            
            ids = [str(i) for i in range(len(documents))]
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query_similar(self, collection_name: str, query_embedding: List[float], n_results: int = 5):
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise 