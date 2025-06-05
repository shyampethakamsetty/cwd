"""LLM client for handling interactions with Azure OpenAI."""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('llm_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with Azure OpenAI."""
    
    def __init__(self, model: Optional[str] = None):
        """Initialize the Azure OpenAI client.
        
        Args:
            model: Optional override for the deployment name. If not provided, uses AZURE_OPENAI_DEPLOYMENT_NAME from env.
        """
        # Load environment variables
        load_dotenv()
        
        # Get Azure OpenAI configuration from environment
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment_name = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, api_base, deployment_name]):
            raise ValueError(
                "Azure OpenAI configuration not found in environment variables. "
                "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME"
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        self.model = deployment_name
        
        logger.info(f"Initialized Azure OpenAI client with deployment: {self.model}")
        
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using Azure OpenAI.
        
        Args:
            system_prompt: The system prompt to guide the model's behavior
            user_prompt: The user's input prompt
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response string
        """
        try:
            logger.info(f"Generating response using deployment: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise 