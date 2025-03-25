"""
RAG System Module

This module integrates all components of the RAG system:
- Document processing
- Text chunking
- Embedding generation
- Vector storage
- Query processing
- Response generation using LLM
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from document_processor import process_file
from text_processor import split_into_chunks, get_chunk_metadata
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system that integrates document processing,
    embedding generation, vector storage, and LLM response generation.
    """
    
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k: int = 5,
                 vector_store_path: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model_name (str): Name of the embedding model to use
            chunk_size (int): Size of document chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
            top_k (int): Number of similar documents to retrieve
            vector_store_path (Optional[str]): Path to load vector store from
        """
        logger.info("Initializing RAG system")
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
        
        # Load or create vector store
        if vector_store_path and os.path.exists(vector_store_path):
            try:
                self.vector_store = VectorStore.load(vector_store_path)
                logger.info(f"Loaded vector store from {vector_store_path}")
            except Exception as e:
                logger.warning(f"Failed to load vector store from {vector_store_path}: {str(e)}")
                logger.info("Creating new vector store instead")
                dimension = self.embedding_generator.get_embedding_dimension()
                self.vector_store = VectorStore(dimension=dimension)
        else:
            dimension = self.embedding_generator.get_embedding_dimension()
            self.vector_store = VectorStore(dimension=dimension)
            logger.info(f"Created new vector store with dimension {dimension}")
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vector_store_path = vector_store_path
    
    def ingest_document(self, file_path: str) -> int:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            int: Number of chunks ingested
        """
        logger.info(f"Ingesting document: {file_path}")
        
        try:
            # Process the file based on its type
            document_text = process_file(file_path)
            
            # Split the document into chunks
            chunks = split_into_chunks(
                document_text, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            # Create metadata for each chunk
            metadata_list = [
                get_chunk_metadata(chunk, i, source_doc=file_path) 
                for i, chunk in enumerate(chunks)
            ]
            
            # Generate embeddings for the chunks
            embeddings = self.embedding_generator.generate(chunks)
            
            # Add chunks and embeddings to the vector store
            self.vector_store.add_documents(chunks, embeddings, metadata_list)
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            raise
    
    def ingest_directory(self, directory_path: str) -> int:
        """
        Ingest all supported documents in a directory.
        
        Args:
            directory_path (str): Path to the directory
            
        Returns:
            int: Total number of chunks ingested
        """
        logger.info(f"Ingesting documents from directory: {directory_path}")
        
        supported_extensions = [
            '.pdf', '.docx', '.csv', 
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
            '.mp4', '.avi', '.mov', '.mkv', '.wmv'
        ]
        
        total_chunks = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() in supported_extensions:
                    try:
                        chunks_ingested = self.ingest_document(file_path)
                        total_chunks += chunks_ingested
                    except Exception as e:
                        logger.error(f"Error ingesting {file_path}: {str(e)}")
                        # Continue with other files
        
        logger.info(f"Completed ingestion of directory. Total chunks: {total_chunks}")
        return total_chunks
    
    def process_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Process a query and retrieve relevant document chunks.
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict]: List of retrieved documents
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate(query)[0]
            
            # Retrieve similar documents
            results = self.vector_store.search(query_embedding, top_k=self.top_k)
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def generate_response(self, query: str, llm_interface) -> Tuple[str, List[Dict]]:
        """
        Generate a response to a query using RAG.
        
        Args:
            query (str): User query
            llm_interface: Interface to the LLM
            
        Returns:
            Tuple[str, List[Dict]]: Generated response and retrieved contexts
        """
        logger.info(f"Generating response for query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.process_query(query)
        
        # Extract the text from the retrieved documents
        contexts = [doc['document'] for doc in retrieved_docs]
        
        # Generate response using the LLM
        response = llm_interface.generate_response(query, contexts)
        
        return response, retrieved_docs
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path (Optional[str]): Path to save to. If None, uses the path from initialization.
        """
        save_path = path or self.vector_store_path
        
        if not save_path:
            raise ValueError("No path specified for saving vector store")
        
        self.vector_store.save(save_path)
        logger.info(f"Vector store saved to {save_path}")
