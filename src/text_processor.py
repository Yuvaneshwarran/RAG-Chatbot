"""
Text Processor Module

This module handles text preprocessing and chunking for better retrieval:
- Text cleaning (removing extra whitespace, special characters)
- Splitting text into chunks with configurable overlap
- Basic text normalization
"""

import re
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def split_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks of approximately equal size with overlap.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Target size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    # Clean the text first
    text = clean_text(text)
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed the chunk size,
        # save the current chunk and start a new one
        if current_length + sentence_length > chunk_size and current_chunk:
            # Join the sentences in the current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk with overlap
            # Find sentences to keep for overlap
            overlap_length = 0
            sentences_to_keep = []
            
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= chunk_overlap:
                    sentences_to_keep.insert(0, s)
                    overlap_length += len(s) + 1  # +1 for the space
                else:
                    break
            
            # Initialize the new chunk with the overlapping sentences
            current_chunk = sentences_to_keep
            current_length = overlap_length
        
        # Add the current sentence to the chunk
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for the space
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def get_chunk_metadata(chunk, index, source_doc=None):
    """
    Generate metadata for a chunk.
    
    Args:
        chunk (str): The text chunk
        index (int): Chunk index
        source_doc (str, optional): Source document path
        
    Returns:
        dict: Metadata for the chunk
    """
    metadata = {
        'chunk_id': index,
        'chunk_length': len(chunk),
        'word_count': len(chunk.split()),
    }
    
    if source_doc:
        metadata['source'] = source_doc
    
    return metadata
