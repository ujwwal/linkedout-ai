"""
Test script to verify environment variables and Azure OpenAI connection
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_env_loading():
    """Test if environment variables are loaded correctly"""
    print("🧪 Testing Environment Variables...")
    print("=" * 50)
    
    try:
        from app.core.config import settings
        
        # Test each required environment variable
        env_vars = {
            'AZURE_OPENAI_API_KEY': settings.AZURE_OPENAI_API_KEY,
            'AZURE_OPENAI_ENDPOINT': settings.AZURE_OPENAI_ENDPOINT,
            'AZURE_OPENAI_DEPLOYMENT_NAME': settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME': settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            'AZURE_OPENAI_API_VERSION': settings.AZURE_OPENAI_API_VERSION,
            'FAISS_INDEX_PATH': settings.FAISS_INDEX_PATH,
        }
        
        for var_name, var_value in env_vars.items():
            if var_value:
                # Mask API key for security
                if 'API_KEY' in var_name:
                    masked_value = f"{str(var_value)[:8]}...{str(var_value)[-4:]}"
                    print(f"✅ {var_name}: {masked_value}")
                else:
                    print(f"✅ {var_name}: {var_value}")
            else:
                print(f"❌ {var_name}: NOT SET")
                
        print("\n✅ Environment variables loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading environment variables: {e}")
        return False

def test_azure_openai_connection():
    """Test Azure OpenAI connection"""
    print("\n🔗 Testing Azure OpenAI Connection...")
    print("=" * 50)
    
    try:
        from app.core.config import settings
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        print(f"🔧 Using deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        print(f"🔧 API Version: {settings.AZURE_OPENAI_API_VERSION}")
        print(f"🔧 Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        
        # Test a simple completion
        print("\n📡 Testing API call...")
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": "Say 'Hello, Azure OpenAI is working!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        if response.choices and response.choices[0].message:
            print(f"✅ API Response: {response.choices[0].message.content}")
            print("✅ Azure OpenAI connection successful!")
            return True
        else:
            print("❌ No response from Azure OpenAI")
            return False
            
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        return False

def test_vector_store_service():
    """Test Vector Store Service initialization"""
    print("\n🗂️ Testing Vector Store Service...")
    print("=" * 50)
    
    try:
        from app.services.vector_store import VectorStoreService
        
        vector_store = VectorStoreService()
        print("✅ Vector Store Service initialized successfully!")
        
        # Check if FAISS index exists
        faiss_path = Path("data/faiss_index")
        if faiss_path.exists():
            print(f"✅ FAISS index found at: {faiss_path}")
        else:
            print(f"⚠️ FAISS index not found at: {faiss_path}")
            print("   This is normal if you haven't loaded any data yet.")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector Store Service failed: {e}")
        return False

def test_llm_service():
    """Test LLM Service initialization"""
    print("\n🤖 Testing LLM Service...")
    print("=" * 50)
    
    try:
        from app.services.vector_store import VectorStoreService
        from app.services.llm_service import LLMService
        
        vector_store = VectorStoreService()
        llm_service = LLMService(vector_store)
        print("✅ LLM Service initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ LLM Service failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LinkedIn Post Generator - Environment Test")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_env_loading),
        ("Azure OpenAI Connection", test_azure_openai_connection),
        ("Vector Store Service", test_vector_store_service),
        ("LLM Service", test_llm_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
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
        print("🎉 All tests passed! Your environment is configured correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()