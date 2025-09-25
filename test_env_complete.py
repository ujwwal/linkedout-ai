"""
Test file to check if both OpenAI deployment and embedding functionality in the .env file are working correctly
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_env_variables():
    """Test if all environment variables are loaded correctly"""
    print("\n🧪 Testing Environment Variables...")
    print("=" * 60)
    
    try:
        from app.core.config import settings
        
        env_vars = {
            'AZURE_OPENAI_API_KEY': settings.AZURE_OPENAI_API_KEY,
            'AZURE_OPENAI_ENDPOINT': settings.AZURE_OPENAI_ENDPOINT,
            'AZURE_OPENAI_DEPLOYMENT_NAME': settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME': settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            'AZURE_OPENAI_API_VERSION': settings.AZURE_OPENAI_API_VERSION,
            'FAISS_INDEX_PATH': settings.FAISS_INDEX_PATH,
        }
        
        all_set = True
        for var_name, var_value in env_vars.items():
            if var_value:
                # Mask API key for security
                if 'API_KEY' in var_name:
                    masked_value = f"{str(var_value)[:5]}...{str(var_value)[-4:]}"
                    print(f"✅ {var_name}: {masked_value}")
                else:
                    print(f"✅ {var_name}: {var_value}")
            else:
                print(f"❌ {var_name}: NOT SET")
                all_set = False
        
        if all_set:
            print("\n✅ All environment variables are properly set!")
        else:
            print("\n❌ Some environment variables are missing!")
        
        return all_set
    
    except Exception as e:
        print(f"\n❌ Error loading environment variables: {e}")
        return False

def test_chat_completion():
    """Test chat completion with the main deployment"""
    print("\n🔄 Testing Chat Completion (Main Deployment)...")
    print("=" * 60)
    
    try:
        from app.core.config import settings
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        print(f"🔧 Using deployment: {deployment_name}")
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a short LinkedIn post about AI."}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            print(f"\n✅ Chat completion successful!")
            print(f"📝 Generated content:\n\"{content}\"")
            return True
        else:
            print("\n❌ No response received from the model.")
            return False
    
    except Exception as e:
        print(f"\n❌ Chat completion failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation with the embedding deployment"""
    print("\n🧠 Testing Embedding Generation...")
    print("=" * 60)
    
    try:
        from app.core.config import settings
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        print(f"🔧 Using embedding deployment: {embedding_deployment}")
        
        test_text = "This is a test for LinkedIn post embedding."
        print(f"📝 Input text: \"{test_text}\"")
        
        response = client.embeddings.create(
            input=test_text,
            model=embedding_deployment
        )
        
        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            dimensions = len(embedding)
            print(f"\n✅ Embedding generated successfully!")
            print(f"📊 Embedding dimensions: {dimensions}")
            print(f"📊 First 5 values: {embedding[:5]}")
            return True
        else:
            print("\n❌ No embedding data received.")
            return False
    
    except Exception as e:
        print(f"\n❌ Embedding generation failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        return False

def test_vector_store():
    """Test vector store functionality with embeddings"""
    print("\n📂 Testing Vector Store...")
    print("=" * 60)
    
    try:
        from app.services.vector_store import VectorStoreService
        
        print("🔧 Initializing Vector Store Service...")
        vector_store = VectorStoreService()
        
        # Check if FAISS index exists
        from app.core.config import settings
        faiss_path = Path(settings.FAISS_INDEX_PATH)
        if faiss_path.exists():
            print(f"✅ FAISS index found at: {faiss_path}")
        else:
            print(f"⚠️ FAISS index not found at: {faiss_path}")
            print("   This is normal if you haven't loaded any data yet.")
        
        # Test similarity search
        test_query = "professional networking tips"
        print(f"\n🔍 Testing similarity search with query: \"{test_query}\"")
        
        results = vector_store.search_similar_posts(test_query, k=2)
        
        if results:
            print(f"✅ Found {len(results)} similar documents.")
            for i, doc in enumerate(results):
                print(f"   Document {i+1}: {doc.page_content[:100]}...")
        else:
            print("⚠️ No similar documents found.")
            print("   This is normal if no data has been loaded into the vector store.")
        
        print("\n✅ Vector store initialized and functioning correctly.")
        return True
    
    except Exception as e:
        print(f"\n❌ Vector store test failed: {e}")
        return False

def test_llm_service():
    """Test LLM service initialization and post generation"""
    print("\n🤖 Testing LLM Service...")
    print("=" * 60)
    
    try:
        from app.services.vector_store import VectorStoreService
        from app.services.llm_service import LLMService
        
        print("🔧 Initializing services...")
        vector_store = VectorStoreService()
        llm_service = LLMService(vector_store)
        
        print("\n🔍 Testing post generation...")
        test_query = "Create a post about leadership skills"
        client_id = "test_client_123"
        
        print(f"📝 Query: \"{test_query}\"")
        print(f"👤 Client ID: {client_id}")
        
        post = llm_service.generate_post(
            query=test_query,
            client_id=client_id,
            is_pro_user=True
        )
        
        if post:
            print(f"\n✅ Post generated successfully!")
            print(f"📝 Generated post:\n\"{post[:200]}...\"")
            return True
        else:
            print("\n❌ No post was generated.")
            return False
    
    except Exception as e:
        import traceback
        print(f"\n❌ LLM service test failed: {e}")
        print("\n🔍 Detailed error:")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🚀 LinkedIn Post Generator - Environment Test")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_env_variables),
        ("Chat Completion", test_chat_completion),
        ("Embedding Generation", test_embedding_generation),
        ("Vector Store", test_vector_store),
        ("LLM Service", test_llm_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📌 Running test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            import traceback
            print(f"\n❌ Test crashed with error: {e}")
            traceback.print_exc()
            results.append((test_name, False))
        print("\n" + "-" * 60)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your environment is configured correctly.")
        print("🚀 LinkedIn Post Generator should work properly.")
    else:
        print("\n⚠️ Some tests failed. Review the issues above and fix your configuration.")

if __name__ == "__main__":
    main()