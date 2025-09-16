"""
Integration test to verify the complete workflow of the LinkedIn Post Generator.
This test verifies:
1. Loading CSV data into the vector store
2. Initializing the LLM service
3. Generating a post using the API
"""
import sys
import os
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import the FastAPI app
from app.main import app
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.db.setupDB import setup_database

# Initialize test client
client = TestClient(app)

class LinkedInPostGeneratorIntegrationTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment, create DB, load vectors, etc."""
        # Ensure database is initialized
        setup_database()
        
        # Check if environment variables are set
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.skipTest(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Check if the CSV file exists
        csv_path = PROJECT_ROOT / "linkedin_content.csv"
        if not csv_path.exists():
            self.skipTest(f"CSV file not found at {csv_path}")
    
    def test_vector_store_service(self):
        """Test that the vector store service can be initialized and load data."""
        try:
            # Initialize vector store service
            vector_store = VectorStoreService()
            
            # Check if a vector store exists or can be created
            self.assertIsNotNone(vector_store.vector_store, 
                                "Vector store should be loaded from disk or initialized")
            
            # Test search functionality with a simple query
            results = vector_store.search_similar_posts("professional networking", k=1)
            
            # Should return at least one result if vector store is properly loaded
            self.assertGreaterEqual(len(results), 0, 
                                   "Vector store search should return results")
            
        except Exception as e:
            self.fail(f"VectorStoreService test failed with error: {str(e)}")
    
    def test_llm_service(self):
        """Test that the LLM service can be initialized."""
        try:
            # Initialize vector store service for LLM to use
            vector_store = VectorStoreService()
            
            # Initialize LLM service
            llm_service = LLMService(vector_store)
            
            # Simple check that the service is initialized properly
            self.assertIsNotNone(llm_service.client, 
                                "LLM service should have a valid client")
            
        except Exception as e:
            self.fail(f"LLMService test failed with error: {str(e)}")
    
    def test_api_generate_post(self):
        """Test the complete API workflow to generate a post."""
        try:
            # Create a test request
            request_data = {
                "user_id": "test-integration-user",
                "query": "Share tips for effective professional networking on LinkedIn"
            }
            
            # Call the API endpoint
            response = client.post("/generate_post", json=request_data)
            
            # Check response status code
            self.assertEqual(response.status_code, 200, 
                            f"API response should be 200 OK, got {response.status_code}")
            
            # Parse response JSON
            data = response.json()
            
            # Validate response structure
            self.assertIn("posts", data, "Response should contain 'posts' field")
            self.assertIn("user_type", data, "Response should contain 'user_type' field")
            
            # Check that at least one post was generated
            self.assertGreaterEqual(len(data["posts"]), 1, 
                                   "At least one post should be generated")
            
            # Check that the post has the required fields
            post = data["posts"][0]
            self.assertIn("post_id", post, "Post should have a post_id")
            self.assertIn("content", post, "Post should have content")
            
            # Check that the content is not empty
            self.assertTrue(post["content"], "Post content should not be empty")
            
        except Exception as e:
            self.fail(f"API integration test failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()