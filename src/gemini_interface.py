"""
Gemini Interface Module

This module provides an interface to Google's Gemini API.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union

from llm_interface import BaseLLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiInterface(BaseLLMInterface):
    """Interface for Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize the Gemini interface.
        
        Args:
            api_key (Optional[str]): Gemini API key. If None, tries to get from environment.
            model (str): Model to use
        """
        super().__init__()
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment")
        
        self.model = model
        logger.info(f"Initialized Gemini interface with model {model}")
        
        # Import optional dependencies
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.use_client = True
        except ImportError:
            logger.warning("Google Generative AI package not found, falling back to direct API calls")
            self.use_client = False
            import requests
            self.requests = requests
    
    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate a response using the Gemini API.
        
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
                # Use the Gemini client
                model = self.genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                result = response.text
            else:
                # Direct API call
                url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json"
                }
                params = {
                    "key": self.api_key
                }
                
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 1024
                    }
                }
                
                response = self.requests.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
                result = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Gemini response generated in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            raise