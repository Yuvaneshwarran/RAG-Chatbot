"""
LLM Interface Module

This module provides interfaces to different LLM providers:
- OpenAI API
- Anthropic API
- Local models (via LlamaCpp or similar)
- Custom API endpoints
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseLLMInterface:
    """Base class for LLM interfaces."""
    
    def __init__(self):
        pass
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Generated response
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _create_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Create a prompt for the LLM with the query and contexts.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Formatted prompt
        """
        # Join contexts with separators
        context_text = "\n\n---\n\n".join(contexts)
        
        # Create prompt
        prompt = f"""Answer the question based on the provided context. If the context doesn't contain the relevant information to answer the question, acknowledge that and provide the best response you can based on your knowledge. 

Context information:
{context_text}

Question: {query}

Answer:"""
        
        return prompt


class OpenAIInterface(BaseLLMInterface):
    """Interface for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the OpenAI interface.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, tries to get from environment.
            model (str): Model to use
        """
        super().__init__()
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
        
        self.model = model
        logger.info(f"Initialized OpenAI interface with model {model}")
        
        # Import optional dependencies
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.use_client = True
        except ImportError:
            logger.warning("OpenAI Python package not found, falling back to direct API calls")
            self.use_client = False
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Generated response
        """
        prompt = self._create_prompt(query, contexts)
        
        try:
            start_time = time.time()
            
            if self.use_client:
                # Use the OpenAI client
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                result = response.choices[0].message.content
            else:
                # Direct API call
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()["choices"][0]["message"]["content"]
            
            elapsed_time = time.time() - start_time
            logger.info(f"OpenAI response generated in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise


class AnthropicInterface(BaseLLMInterface):
    """Interface for Anthropic API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize the Anthropic interface.
        
        Args:
            api_key (Optional[str]): Anthropic API key. If None, tries to get from environment.
            model (str): Model to use
        """
        super().__init__()
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment")
        
        self.model = model
        logger.info(f"Initialized Anthropic interface with model {model}")
        
        # Import optional dependencies
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.use_client = True
        except ImportError:
            logger.warning("Anthropic Python package not found, falling back to direct API calls")
            self.use_client = False
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the Anthropic API.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Generated response
        """
        prompt = self._create_prompt(query, contexts)
        
        try:
            start_time = time.time()
            
            if self.use_client:
                # Use the Anthropic client
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result = response.content[0].text
            else:
                # Direct API call
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                payload = {
                    "model": self.model,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()["content"][0]["text"]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Anthropic response generated in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with Anthropic: {str(e)}")
            raise


class LocalModelInterface(BaseLLMInterface):
    """Interface for local models using LlamaCpp or similar."""
    
    def __init__(self, model_path: str, context_length: int = 4096):
        """
        Initialize the local model interface.
        
        Args:
            model_path (str): Path to the model file
            context_length (int): Maximum context length for the model
        """
        super().__init__()
        
        self.model_path = model_path
        self.context_length = context_length
        
        # Import optional dependencies
        try:
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path, n_ctx=context_length)
            logger.info(f"Initialized local model from {model_path}")
        except ImportError:
            raise ImportError("llama_cpp package not found. Install it to use local models.")
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the local model.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Generated response
        """
        prompt = self._create_prompt(query, contexts)
        
        try:
            start_time = time.time()
            
            # Generate response
            output = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.3,
                stop=["Question:", "Context information:"]
            )
            
            result = output["choices"][0]["text"]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Local model response generated in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with local model: {str(e)}")
            raise


class CustomAPIInterface(BaseLLMInterface):
    """Interface for custom API endpoints."""
    
    def __init__(self, api_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the custom API interface.
        
        Args:
            api_url (str): URL of the API endpoint
            headers (Optional[Dict[str, str]]): Headers to include in the request
        """
        super().__init__()
        
        self.api_url = api_url
        self.headers = headers or {}
        logger.info(f"Initialized custom API interface with URL {api_url}")
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the custom API.
        
        Args:
            query (str): User query
            contexts (List[str]): Relevant document chunks
            
        Returns:
            str: Generated response
        """
        prompt = self._create_prompt(query, contexts)
        
        try:
            start_time = time.time()
            
            # Prepare the payload
            payload = {
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.3
            }
            
            # Make the API call
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
            
            # Parse the response
            result = response.json().get("response")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Custom API response generated in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with custom API: {str(e)}")
            raise
