# main.py - Railway deployment version

import os
from typing import List
import time

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Document processing
import PyPDF2
import docx

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Initialize AI models and services
def initialize_services():
    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-pro')
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return gemini_model, embedding_model

# Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "policy-docs-gemini"
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {index_name}")
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index is ready!")
    else:
        print(f"Using existing index: {index_name}")
        # Check dimensions
        index_info = pc.describe_index(index_name)
        if index_info.dimension != 384:
            print(f"Deleting and recreating index with correct dimensions...")
            pc.delete_index(index_name)
            
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created new index: {index_name}")
            
            print("Waiting for index to be ready...")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index is ready!")
    
    index = pc.Index(index_name)
    return pc, index

# Text extraction functions
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

# Text chunking
def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# AI functions
def get_embedding(text: str, embedding_model) -> List[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def query_gemini(question: str, context_clauses: List[str], gemini_model) -> str:
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

# Pinecone operations
def upsert_chunks(chunks: List[str], index, embedding_model):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, embedding_model)
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"Upserted {len(chunks)} chunks to Pinecone.")

def query_chunks(query: str, index, embedding_model, top_k: int = 5) -> List[str]:
    query_embedding = get_embedding(query, embedding_model)
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match.metadata.get("text", "") for match in query_response.matches]

# Initialize document processing
def initialize_policy_document(index, embedding_model):
    policy_file = "policy.pdf"
    
    if not os.path.exists(policy_file):
        print(f"Warning: Policy file {policy_file} not found. Skipping document initialization.")
        return False
    
    print(f"Loading policy document: {policy_file}")
    
    full_text = extract_text(policy_file)
    print(f"Extracted {len(full_text)} characters from policy document")
    
    chunks = chunk_text(full_text)
    print(f"Created {len(chunks)} chunks")
    
    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print("Clearing existing vectors from index...")
            index.delete(delete_all=True)
            print("Cleared existing vectors from index")
    except Exception as e:
        print(f"Could not check/clear index stats: {e}")
    
    upsert_chunks(chunks, index, embedding_model)
    print("Policy document successfully indexed")
    return True

# Global variables for models
gemini_model = None
embedding_model = None
pc = None
index = None

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Gemini", version="1.0.0")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid or missing authorization token"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Startup event
@app.on_event("startup")
async def startup_event():
    global gemini_model, embedding_model, pc, index
    
    print("Initializing services...")
    
    # Check required environment variables
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY", "API_BEARER_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise Exception(f"Missing required environment variables: {missing_vars}")
    
    # Initialize AI models
    gemini_model, embedding_model = initialize_services()
    print("AI models initialized")
    
    # Initialize Pinecone
    pc, index = init_pinecone()
    print("Pinecone initialized")
    
    # Initialize policy document
    doc_loaded = initialize_policy_document(index, embedding_model)
    if doc_loaded:
        print("Startup complete - Policy document loaded and indexed")
    else:
        print("Startup complete - No policy document found, API ready for manual document upload")

# API endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    global gemini_model, embedding_model, index
    
    answers = []
    
    for question in req.questions:
        try:
            relevant_clauses = query_chunks(question, index, embedding_model)
            
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document for this question.")
                continue
            
            answer = query_gemini(question, relevant_clauses, gemini_model)
            answers.append(answer)
            
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Insurance Policy RAG API with Gemini is running"}

@app.get("/info")
async def get_info():
    global index
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

# For Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
