"""
Quick test to verify embeddings are working correctly
"""
from langchain_huggingface import HuggingFaceEmbeddings

print("Testing HuggingFace Embeddings (No API Key Needed)")
print("="*60)

try:
    print("\n📦 Loading embedding model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("✅ Model loaded successfully!")
    
    print("\n🧪 Testing embedding generation...")
    test_embedding = embeddings.embed_query("This is a test sentence for embeddings")
    
    print(f"✅ SUCCESS!")
    print(f"   Embedding dimension: {len(test_embedding)}")
    print(f"   First 5 values: {test_embedding[:5]}")
    
    print("\n" + "="*60)
    print("✅ HuggingFace embeddings are working perfectly!")
    print("   No API key needed, reliable, and fast!")
    print("="*60)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\n💡 Solution: Install sentence-transformers and langchain-huggingface")
    print("   pip install sentence-transformers langchain-huggingface")
    import traceback
    traceback.print_exc()

