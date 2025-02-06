import logging
from research_assistant import ResearchAssistant
from pathlib import Path
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_documents_from_folder(folder_path: str) -> list:
    """Scan and return all supported document files from the given folder."""
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    documents = []
    
    try:
        folder = Path(folder_path)
        if not folder.exists():
            folder.mkdir(parents=True)
            logger.info(f"Created reference folder at {folder}")
            return documents

        logger.info(f"Scanning folder: {folder}")
        for file in folder.rglob('*'):
            if file.suffix.lower() in supported_extensions:
                documents.append(str(file))
                logger.info(f"Found document: {file.name}")
        
        logger.info(f"Total documents found: {len(documents)}")
    
    except Exception as e:
        logger.error(f"Error scanning reference folder: {str(e)}")
    
    return documents

def main():
    print("\n=== Research Paper Analysis System ===\n")
    
    # Initialize the research assistant
    assistant = ResearchAssistant()

    # Set up reference folder path
    reference_folder = os.path.join(os.getcwd(), "reference_docs")
    
    try:
        # Get all documents from the reference folder
        print("\nScanning for documents...")
        file_paths = get_documents_from_folder(reference_folder)
        
        if not file_paths:
            print("\nNo documents found in the reference folder.")
            print(f"Please add your research papers to: {reference_folder}")
            return

        print(f"\nFound {len(file_paths)} documents to process.")
        
        # Process documents
        print("\nProcessing documents and generating embeddings...")
        print("This may take a while depending on the number and size of documents.")
        print("Please wait...\n")
        
        assistant.process_documents(file_paths)
        print("\nDocument processing completed!")

        while True:
            print("\n=== Query Interface ===")
            print("Enter your research question (or 'quit' to exit)")
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            print("\nSearching for relevant information...")
            results = assistant.query_research(query)
            
            if results['documents'] and results['documents'][0]:
                print("\nFound relevant information:")
                print("----------------------------")
                for idx, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    print(f"\nResult {idx + 1} (Similarity: {1 - score:.2f}):")
                    print("---")
                    print(doc)
                    print("---")
            else:
                print("\nNo relevant information found for your query.")
            
            time.sleep(1)  # Small delay for better user experience

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print("\nAn error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main() 