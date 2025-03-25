"""
RAG System API

This module provides a REST API for the RAG system:
- Document ingestion endpoint
- Query endpoint
- System status endpoint
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import time
from typing import Dict, Any, List, Optional

from rag_system import RAGSystem
from llm_interface import OpenAIInterface, AnthropicInterface, CustomAPIInterface
from gemini_interface import GeminiInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  # Enable CORS for all routes

# Initialize RAG system
try:
    rag_system = RAGSystem(
        embedding_model_name=os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        chunk_size=int(os.environ.get('CHUNK_SIZE', '1000')),
        chunk_overlap=int(os.environ.get('CHUNK_OVERLAP', '200')),
        top_k=int(os.environ.get('TOP_K', '5')),
        vector_store_path=os.environ.get('VECTOR_STORE_PATH', './vector_store')
    )
except FileNotFoundError:
    # Vector store doesn't exist yet, create a new one
    logger.info("Vector store not found. Creating a new one.")
    vector_store_path = os.environ.get('VECTOR_STORE_PATH', './vector_store')
    rag_system = RAGSystem(
        embedding_model_name=os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        chunk_size=int(os.environ.get('CHUNK_SIZE', '1000')),
        chunk_overlap=int(os.environ.get('CHUNK_OVERLAP', '200')),
        top_k=int(os.environ.get('TOP_K', '5')),
        vector_store_path=None  # Don't try to load an existing store
    )
    # Set the path for future saves
    rag_system.vector_store_path = vector_store_path

# Cache for LLM interfaces
llm_interfaces = {}

def get_llm_interface(provider: str, api_key: Optional[str] = None, model_name: Optional[str] = None):
    """Get or create an LLM interface based on provider."""
    cache_key = f"{provider}:{model_name}"
    
    if cache_key in llm_interfaces:
        return llm_interfaces[cache_key]
    
    if provider == "openai":
        interface = OpenAIInterface(
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            model=model_name or "gpt-4-turbo"
        )
    elif provider == "anthropic":
        interface = AnthropicInterface(
            api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
            model=model_name or "claude-3-opus-20240229"
        )
    elif provider == "gemini":
        interface = GeminiInterface(
            api_key=api_key or os.environ.get('GEMINI_API_KEY'),
            model=model_name or "gemini-pro"
        )
    elif provider == "custom":
        interface = CustomAPIInterface(
            api_url=os.environ.get('CUSTOM_API_URL', ''),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    llm_interfaces[cache_key] = interface
    return interface

@app.route('/api/query', methods=['POST'])
def query():
    """Process a query and return the response."""
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query_text = data['query']
        provider = data.get('provider', 'openai')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        show_sources = data.get('show_sources', True)
        
        # Get LLM interface
        try:
            llm_interface = get_llm_interface(provider, api_key, model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Error initializing LLM interface: {str(e)}"}), 500
        
        # Process query
        start_time = time.time()
        response, retrieved_docs = rag_system.generate_response(query_text, llm_interface)
        processing_time = time.time() - start_time
        
        # Prepare sources if requested
        sources = []
        if show_sources:
            for doc in retrieved_docs:
                source_info = {
                    "text": doc['document'][:500] + "..." if len(doc['document']) > 500 else doc['document'],
                    "score": doc['score'],
                    "metadata": doc['metadata']
                }
                sources.append(source_info)
        
        result = {
            "response": response,
            "processing_time": processing_time,
            "sources": sources if show_sources else []
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Ingest a document or directory."""
    try:
        data = request.json
        
        if not data or 'path' not in data:
            return jsonify({"error": "Path is required"}), 400
        
        path = data['path']
        
        if not os.path.exists(path):
            return jsonify({"error": f"Path not found: {path}"}), 404
        
        # Ingest file or directory
        start_time = time.time()
        
        if os.path.isfile(path):
            chunks = rag_system.ingest_document(path)
            message = f"Ingested {chunks} chunks from file {path}"
        elif os.path.isdir(path):
            chunks = rag_system.ingest_directory(path)
            message = f"Ingested {chunks} total chunks from directory {path}"
        
        processing_time = time.time() - start_time
        
        # Save the vector store
        rag_system.save_vector_store()
        
        result = {
            "message": message,
            "chunks_ingested": chunks,
            "processing_time": processing_time
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    try:
        index_size = rag_system.vector_store.get_index_size()
        embedding_model = rag_system.embedding_generator.model.get_sentence_embedding_dimension()
        
        result = {
            "status": "online",
            "index_size": index_size,
            "embedding_dimension": embedding_model,
            "chunk_size": rag_system.chunk_size,
            "chunk_overlap": rag_system.chunk_overlap,
            "top_k": rag_system.top_k
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)