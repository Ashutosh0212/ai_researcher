import logging
from typing import List, Optional
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAssistant:
    def __init__(self, collection_name: str = "research_papers"):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.collection_name = collection_name
        self.vector_store.create_collection(collection_name)

    def process_documents(self, file_paths: List[str]):
        try:
            all_chunks = []
            for file_path in file_paths:
                chunks = self.document_processor.load_document(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}")

            texts = [chunk['content'] for chunk in all_chunks]
            embeddings = self.embedding_manager.generate_embeddings(texts)
            
            self.vector_store.add_documents(
                self.collection_name,
                all_chunks,
                embeddings
            )
            logger.info("Documents processed and stored successfully")
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise

    def query_research(self, query: str, n_results: int = 5):
        try:
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            results = self.vector_store.query_similar(
                self.collection_name,
                query_embedding,
                n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            raise 