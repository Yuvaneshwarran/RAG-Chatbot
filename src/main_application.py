"""
Main Application

This script demonstrates how to use the RAG system components together.
It provides a command-line interface for:
- Ingesting documents
- Querying the system
- Managing the vector store
"""

import os
import argparse
import logging
from typing import List, Optional

from rag_system import RAGSystem
from llm_interface import OpenAIInterface, AnthropicInterface, LocalModelInterface, CustomAPIInterface
from gemini_interface import GeminiInterface
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_llm_interface(args):
    """Set up the LLM interface based on command line arguments."""
    if args.llm_provider == "openai":
        return OpenAIInterface(api_key=args.api_key, model=args.model_name)
    elif args.llm_provider == "anthropic":
        return AnthropicInterface(api_key=args.api_key, model=args.model_name)
    elif args.llm_provider == "gemini":
        return GeminiInterface(api_key=args.api_key, model=args.model_name)
    elif args.llm_provider == "local":
        return LocalModelInterface(model_path=args.model_path)
    elif args.llm_provider == "custom":
        headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        return CustomAPIInterface(api_url=args.api_url, headers=headers)
    else:
        raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")

def ingest_command(args):
    """Handle the ingestion command."""
    # Setup RAG system
    rag_system = RAGSystem(
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store_path
    )
    
    # Ingest file or directory
    if os.path.isfile(args.path):
        chunks = rag_system.ingest_document(args.path)
        logger.info(f"Ingested {chunks} chunks from file {args.path}")
    elif os.path.isdir(args.path):
        chunks = rag_system.ingest_directory(args.path)
        logger.info(f"Ingested {chunks} total chunks from directory {args.path}")
    else:
        logger.error(f"Path not found: {args.path}")
        return
    
    # Save the vector store
    rag_system.save_vector_store()
    logger.info(f"Vector store saved to {args.vector_store_path}")

def query_command(args):
    """Handle the query command."""
    # Setup RAG system
    rag_system = RAGSystem(
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
        vector_store_path=args.vector_store_path
    )
    
    # Setup LLM interface
    llm_interface = setup_llm_interface(args)
    
    # Process the query
    response, retrieved_docs = rag_system.generate_response(args.query, llm_interface)
    
    # Print the response
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)
    
    # Print sources if requested
    if args.show_sources:
        print("\n" + "="*50)
        print("SOURCES:")
        print("="*50)
        for i, doc in enumerate(retrieved_docs):
            print(f"Source {i+1}:")
            print(f"Score: {doc['score']:.4f}")
            if 'source' in doc['metadata']:
                print(f"File: {doc['metadata']['source']}")
            print("-"*30)
            print(doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'])
            print()

def main():
    """Main entry point for the application."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Parser for the "ingest" command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the system")
    ingest_parser.add_argument("path", help="Path to file or directory to ingest")
    ingest_parser.add_argument("--vector-store-path", default="./vector_store", help="Path to save vector store")
    ingest_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model to use")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Size of document chunks")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    # Parser for the "query" command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", help="Query to process")
    query_parser.add_argument("--vector-store-path", default="./vector_store", help="Path to vector store")
    query_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model to use")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--show-sources", action="store_true", help="Show source documents")
    
    # LLM provider options
    query_parser.add_argument("--llm-provider", default="openai", choices=["openai", "anthropic", "gemini", "local", "custom"], help="LLM provider to use")    
    query_parser.add_argument("--api-key", help="API key for the LLM provider")
    query_parser.add_argument("--model-name", default="gpt-4-turbo", help="Model name for the LLM provider")
    query_parser.add_argument("--model-path", help="Path to local model file")
    query_parser.add_argument("--api-url", help="URL for custom API endpoint")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
