import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

print(f"Key starts with: {key[:5] if key else 'None'}")

models_to_test = [
    "models/text-embedding-004",
    "models/embedding-001",
    "text-embedding-004",
    "embedding-001"
]

for m in models_to_test:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=m, google_api_key=key)
        res = embeddings.embed_query("hello")
        print(f"[SUCCESS] {m} works! Length: {len(res)}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
