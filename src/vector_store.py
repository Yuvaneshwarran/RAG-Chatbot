"""
Vector Store Module

This module implements a vector database using FAISS for efficient similarity search:
- Storing document chunks and their embeddings
- Searching for similar documents given a query embedding
- Saving and loading the index to/from disk
"""

import faiss
import numpy as np
import pickle
import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, dimension=384):
        """
        Initialize the vector store with the specified embedding dimension.
        
        Args:
            dimension (int): Dimension of the embeddings (default is 384 for 'all-MiniLM-L6-v2')
        """
        logger.info(f"Initializing VectorStore with dimension {dimension}")
        
        # Initialize a flat L2 index (exact search)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
        # Storage for documents and metadata
        self.documents = []
        self.metadata = []
        
        # Track index size
        self.index_size = 0
    
    def add_documents(self, documents, embeddings, metadata=None):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents (List[str]): List of document texts
            embeddings (numpy.ndarray): Embeddings for the documents
            metadata (List[Dict], optional): Metadata for each document
            
        Returns:
            List[int]: List of indices for the added documents
        """
        # Validate inputs
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) must match number of embeddings ({len(embeddings)})")
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        if len(metadata) != len(documents):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of documents ({len(documents)})")
        
        # Record the starting index for the new documents
        start_idx = len(self.documents)
        
        # Add documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Convert embeddings to the correct format
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add embeddings to the index
        try:
            start_time = time.time()
            self.index.add(embeddings_array)
            elapsed_time = time.time() - start_time
            
            # Update index size
            num_added = len(documents)
            self.index_size += num_added
            
            logger.info(f"Added {num_added} documents to the vector store in {elapsed_time:.2f} seconds")
            
            # Return indices of the added documents
            return list(range(start_idx, start_idx + num_added))
        
        except Exception as e:
            # Revert document and metadata additions if index update fails
            self.documents = self.documents[:-len(documents)]
            self.metadata = self.metadata[:-len(metadata)]
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
    
    def search(self, query_embedding, top_k=5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding (numpy.ndarray): Embedding of the query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of dictionaries containing the retrieved documents, metadata, and similarity scores
        """
        if self.index_size == 0:
            logger.warning("Search performed on empty index")
            return []
        
        # Ensure the query embedding is the right shape and type
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Perform the search
        try:
            start_time = time.time()
            distances, indices = self.index.search(query_embedding, min(top_k, self.index_size))
            elapsed_time = time.time() - start_time
            
            logger.info(f"Search completed in {elapsed_time:.4f} seconds")
            
            # Prepare the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(distances[0][i]),
                        'index': int(idx)
                    })
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            raise
    
    def get_document(self, index):
        """
        Get a document by its index.
        
        Args:
            index (int): Document index
            
        Returns:
            Tuple[str, Dict]: Document text and metadata
        """
        if 0 <= index < len(self.documents):
            return self.documents[index], self.metadata[index]
        else:
            raise IndexError(f"Document index {index} out of range")
    
    def save(self, directory):
        """
        Save the vector store to disk.
        
        Args:
            directory (str): Directory to save to
        """
        logger.info(f"Saving vector store to {directory}")
        
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Save the FAISS index
            index_path = os.path.join(directory, 'index.faiss')
            faiss.write_index(self.index, index_path)
            
            # Save the documents and metadata
            data_path = os.path.join(directory, 'documents.pkl')
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_size': self.index_size
                }, f)
            
            logger.info(f"Vector store saved successfully with {self.index_size} documents")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    @classmethod
    def load(cls, directory):
        """
        Load a vector store from disk.
        
        Args:
            directory (str): Directory to load from
            
        Returns:
            VectorStore: Loaded vector store
        """
        logger.info(f"Loading vector store from {directory}")
        
        try:
            # Load the documents and metadata
            data_path = os.path.join(directory, 'documents.pkl')
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Create a new instance with the right dimension
            instance = cls(dimension=data['dimension'])
            
            # Load the index
            index_path = os.path.join(directory, 'index.faiss')
            instance.index = faiss.read_index(index_path)
            
            # Set the instance variables
            instance.documents = data['documents']
            instance.metadata = data['metadata']
            instance.index_size = data['index_size']
            
            logger.info(f"Vector store loaded successfully with {instance.index_size} documents")
            return instance
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def get_index_size(self):
        """
        Get the number of documents in the index.
        
        Returns:
            int: Number of documents
        """
        return self.index_size
