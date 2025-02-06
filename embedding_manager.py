import logging
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        try:
            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} texts)")
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    logger.info(f"Successfully processed batch {i//batch_size + 1}")
                    # Small delay to prevent overwhelming the Ollama server
                    time.sleep(0.5)
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
                    # Retry with smaller batch size if batch fails
                    if batch_size > 1:
                        logger.info("Retrying with smaller batch size...")
                        for text in batch:
                            try:
                                single_embedding = self.embeddings.embed_documents([text])
                                embeddings.extend(single_embedding)
                                time.sleep(0.5)
                            except Exception as single_error:
                                logger.error(f"Error processing single text: {str(single_error)}")
                    
            if not embeddings:
                raise ValueError("No embeddings were generated successfully")
                
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise 