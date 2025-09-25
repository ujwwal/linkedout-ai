"""
Script to help find the correct Azure OpenAI deployment name
"""
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_deployment_name(deployment_name):
    """Test a specific deployment name"""
    try:
        from app.core.config import settings
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        print(f"🧪 Testing deployment: {deployment_name}")
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.1
        )
        
        if response.choices and response.choices[0].message:
            print(f"✅ SUCCESS: {deployment_name} works!")
            return True
        else:
            print(f"❌ FAILED: {deployment_name} - No response")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "DeploymentNotFound" in error_msg:
            print(f"❌ FAILED: {deployment_name} - Deployment not found")
        elif "InvalidRequestError" in error_msg:
            print(f"⚠️ PARTIAL: {deployment_name} - Deployment exists but request invalid")
            return True  # Deployment exists, just request format issue
        else:
            print(f"❌ FAILED: {deployment_name} - {error_msg}")
        return False

def main():
    """Test common deployment names"""
    print("🔍 Finding Correct Azure OpenAI Deployment Name")
    print("=" * 55)
    
    # Common deployment names to test
    common_names = [
        "gpt-4",
        "gpt-4o", 
        "gpt-4-turbo",
        "gpt-4-32k",
        "gpt-35-turbo",
        "gpt-3.5-turbo",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4o-mini",
        # Your current embedding deployment name (might be wrong assignment)
        "gpt-4.1"
    ]
    
    working_deployments = []
    
    for name in common_names:
        if test_deployment_name(name):
            working_deployments.append(name)
        print()  # Add spacing
    
    print("📊 Results")
    print("=" * 55)
    
    if working_deployments:
        print("✅ Working deployment names found:")
        for name in working_deployments:
            print(f"   - {name}")
        
        print(f"\n💡 Update your .env file with one of these names:")
        print(f"   AZURE_OPENAI_DEPLOYMENT_NAME={working_deployments[0]}")
        
    else:
        print("❌ No working deployment names found.")
        print("\n🔧 Please check your Azure OpenAI resource and verify:")
        print("   1. You have created a model deployment")
        print("   2. The deployment is in the same region as your endpoint")
        print("   3. The deployment name is correct")
        print("\n📍 To find your deployment names:")
        print("   1. Go to Azure Portal")
        print("   2. Navigate to your Azure OpenAI resource")
        print("   3. Go to 'Model deployments' section")
        print("   4. Copy the exact deployment name from there")

if __name__ == "__main__":
    main()