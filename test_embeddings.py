import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Try different model names
models_to_try = [
    "text-embedding-004",
    "models/text-embedding-004",
    "embedding-001",
    "models/embedding-001"
]

for model in models_to_try:
    try:
        print(f"Trying model: {model}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        result = embeddings.embed_query("test")
        print(f"✓ Success with '{model}' - dimension: {len(result)}")
        break
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}")
        print()
