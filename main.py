

# 2. Set your API keys here securely (or use getpass for input)

import os
import getpass

os.environ["GEMINI_API_KEY"] = "AIzaSyCnlEcrEJYNTYsdxM9RoDSkwdSK-ndJ33U"
os.environ["PINECONE_API_KEY"] = "pcsk_5EwLbD_C8ugsDhESaeBmXGRz2HWaJnp7f7swgEYTftAi8LNzS8YTfpCF3vgaStiuEteuYu"
os.environ["API_BEARER_TOKEN"] = "abc_123"

# --------------------------------------------------------

# 3. Imports and setup

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import requests
import PyPDF2
import docx
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from sentence_transformers import SentenceTransformer

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize embedding model (using sentence-transformers as free alternative)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------------------------------------

# 4. Pinecone Initialization (updated for current pinecone package)

def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "policy-docs"
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,  # Changed for sentence-transformers model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {index_name}")
        
        # Wait for index to be ready
        import time
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index is ready!")
    else:
        print(f"Using existing index: {index_name}")
    
    index = pc.Index(index_name)
    return pc, index

pc, index = init_pinecone()
print("Pinecone initialized and index connected.")

# --------------------------------------------------------

# 5. Text extraction (modified to work with local policy.pdf)

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(docx_path: str) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported document format: {ext}")

# --------------------------------------------------------

# 6. Text chunking utility

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# --------------------------------------------------------

# 7. Embeddings & Gemini querying

def get_embedding(text: str) -> List[float]:
    """Get embeddings using sentence-transformers (free alternative)"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def query_gemini(question: str, context_clauses: List[str]) -> str:
    """Query Gemini Pro model"""
    prompt = f"""
You are an expert assistant who answers insurance policy questions precisely and cites the clauses.

Question: {question}

Use ONLY the following clauses and explicitly mention or quote them in your answer:

{chr(10).join([f"- {clause}" for clause in context_clauses])}

Answer:
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --------------------------------------------------------

# 8. Pinecone upsert and query functions (updated for latest version)

def upsert_chunks(chunks: List[str]):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    # Upsert in batches of 100 (Pinecone recommendation)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"Upserted {len(chunks)} chunks to Pinecone.")

def query_chunks(query: str, top_k: int = 5) -> List[str]:
    query_embedding = get_embedding(query)
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match.metadata.get("text", "") for match in query_response.matches]

# --------------------------------------------------------

# 9. Initialize the policy document once at startup

def initialize_policy_document():
    """Extract and upsert the policy.pdf document once at startup"""
    policy_file = "policy.pdf"
    
    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"Policy file {policy_file} not found in the current directory")
    
    print(f"Loading policy document: {policy_file}")
    
    # Extract text from the policy document
    full_text = extract_text(policy_file)
    print(f"Extracted {len(full_text)} characters from policy document")
    
    # Chunk the text
    chunks = chunk_text(full_text)
    print(f"Created {len(chunks)} chunks")
    
    # Check if index has any vectors before trying to delete
    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print("Clearing existing vectors from index...")
            index.delete(delete_all=True)
            print("Cleared existing vectors from index")
        else:
            print("Index is empty, no need to clear")
    except Exception as e:
        print(f"Could not check/clear index stats: {e}")
        print("Proceeding with upsert...")
    
    # Upsert chunks to Pinecone
    upsert_chunks(chunks)
    print("Policy document successfully indexed")

# Initialize the policy document
initialize_policy_document()

# --------------------------------------------------------

# 10. FastAPI auth (Bearer token)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid or missing authorization token"
        )
    return True

# --------------------------------------------------------

# 11. FastAPI app setup (simplified since policy.pdf is pre-loaded)

app = FastAPI(title="Insurance Policy RAG API with Gemini", version="1.0.0")

class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    """
    Query the pre-loaded policy document with questions
    """
    answers = []
    
    for question in req.questions:
        try:
            # Query Pinecone for relevant clauses
            relevant_clauses = query_chunks(question)
            
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document for this question.")
                continue
            
            # Query Gemini for answer with citations
            answer = query_gemini(question, relevant_clauses)
            answers.append(answer)
            
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Insurance Policy RAG API with Gemini is running"}

@app.get("/info")
async def get_info():
    """Get information about the loaded policy"""
    try:
        stats = index.describe_index_stats()
        return {
            "status": "ready",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gemini-pro"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --------------------------------------------------------

# 12. Run FastAPI app with ngrok tunnel in Colab

nest_asyncio.apply()

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"\n🚀 Public URL for API: {public_url}")
print(f"📋 API Documentation: {public_url}/docs")
print(f"❤️  Health Check: {public_url}/health")
print(f"ℹ️  Index Info: {public_url}/info")

# Run the FastAPI app
uvicorn.run(app, host="0.0.0.0", port=8000)
