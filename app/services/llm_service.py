import os
from openai import AzureOpenAI
from typing import Dict, List, Optional
import json
import time
from ..core.config import settings
from pydantic import SecretStr

class LLMService:
    def __init__(self, vector_store_service):
        # Initialize Azure OpenAI client using settings from config
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.client = client
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        
        # Store reference to vector store for RAG
        self.vector_store_service = vector_store_service
        
        # Initialize client memory storage
        self.client_memory: Dict[str, List[Dict]] = {}

    def _get_client_memory(self, client_id: str) -> str:
        """Retrieve conversation history for specific client and format as context"""
        if client_id not in self.client_memory or not self.client_memory[client_id]:
            return ""
        
        # Get last 5 interactions to avoid context length issues
        recent_history = self.client_memory[client_id][-5:]
        
        formatted_history = ""
        for interaction in recent_history:
            formatted_history += f"User request: {interaction['query']}\n"
            formatted_history += f"Generated post: {interaction['response']}\n\n"
            
        return formatted_history

    def _update_client_memory(self, client_id: str, query: str, response: str) -> None:
        """Store new interaction in client's history"""
        if client_id not in self.client_memory:
            self.client_memory[client_id] = []
            
        self.client_memory[client_id].append({
            "timestamp": time.time(),
            "query": query,
            "response": response
        })
        
        # Keep memory manageable - limit to last 20 interactions
        if len(self.client_memory[client_id]) > 20:
            self.client_memory[client_id] = self.client_memory[client_id][-20:]

    def _retrieve_similar_posts(self, query: str, top_k: int = 3) -> str:
        """Use vector_store_service to find similar posts and format them as examples"""
        try:
            similar_docs = self.vector_store_service.search_similar_posts(query, k=top_k)
            
            if not similar_docs:
                return ""
                
            examples = "Here are some example LinkedIn posts that might be relevant:\n\n"
            
            for i, doc in enumerate(similar_docs, 1):
                # Extract post content and any available metadata
                post_content = doc.page_content
                author = doc.metadata.get('profile_name', 'Unknown Author') if hasattr(doc, 'metadata') else 'Unknown Author'
                
                examples += f"Example {i} (by {author}):\n{post_content}\n\n"
                
            return examples
        except Exception as e:
            print(f"Error retrieving similar posts: {str(e)}")
            return ""

    def _build_prompt(self, query: str, client_id: str, is_pro_user: bool = False) -> List[Dict[str, str]]:
        """Build the complete prompt with system instructions, examples, and user query"""
        # Get client memory for personalization
        client_history = self._get_client_memory(client_id)
        
        # Get similar posts as examples
        similar_posts = self._retrieve_similar_posts(query)
        
        # Define base system prompt
        system_prompt = """You are LinkedOut, an AI assistant specialized in creating engaging LinkedIn posts.
Follow these guidelines:
- Write in a professional but conversational tone
- Include relevant hashtags at the end
- Keep posts concise (under 1300 characters)
- Focus on providing value to the reader
- Avoid clichés and generic corporate language
- Format with appropriate line breaks and emojis where natural"""
        
        # Add pro features for premium users
        if is_pro_user:
            system_prompt += """
- For PRO users: Include more sophisticated content structures
- For PRO users: Add a hook at the beginning and call-to-action at the end
- For PRO users: Optimize for maximum engagement with advanced storytelling techniques"""
        
        # Combine all context elements
        full_context = system_prompt
        d
        if client_history:
            full_context += "\n\nPREVIOUS INTERACTIONS WITH THIS USER:\n" + client_history
            
        if similar_posts:
            full_context += "\n\n" + similar_posts
        
        # Build the messages array for the API call
        messages = [
            {"role": "system", "content": full_context},
            {"role": "user", "content": query}
        ]
        
        return messages

    def generate_post(self, query: str, client_id: str, is_pro_user: bool = False) -> str:
        """Generate a LinkedIn post based on user query"""
        try:
            # Build the prompt
            messages = self._build_prompt(query, client_id, is_pro_user)
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,  # Type: List[Dict[str, str]]
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            # Handle None case for type checking
            if generated_text is None:
                generated_text = "I couldn't generate a LinkedIn post at this time. Please try again."
            
            # Update client memory
            self._update_client_memory(client_id, query, generated_text)
            
            return generated_text
            
        except Exception as e:
            error_message = f"Error generating post: {str(e)}"
            print(error_message)
            return "I'm sorry, I encountered an error while generating your LinkedIn post. Please try again later."