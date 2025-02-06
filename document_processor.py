import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            return [{
                'content': chunk.page_content,
                'metadata': {
                    **chunk.metadata,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            } for chunk in chunks]

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise 