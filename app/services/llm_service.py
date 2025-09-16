import os
from openai import AzureOpenAI

class LLMService:
    def __init__(self, vector_store_service):
        # Initialize Azure OpenAI client
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2023-05-15"
        )
        self.client = client       
        # Store reference to vector store for RAG
        # Initialize client memory storage

    def _get_client_memory(self, client_id):
        # Retrieve conversation history for specific client
        # Return formatted context

    def _update_client_memory(self, client_id, query, response):
        # Store new interaction in client's history

    def _retrieve_similar_posts(self, query, top_k=3):
        # Use vector_store_service to find similar posts
        # Format them as examples

    def _build_prompt(self, query, client_id, is_pro_user):
        # Get client memory
        # Get similar posts examples
        # Assemble complete prompt with system instructions

    def generate_post(self, query, client_id, is_pro_user=False):
        # Build the prompt
        # Call Azure OpenAI
        # Process response
        # Update memory
        # Return formatted result