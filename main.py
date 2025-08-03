# main.py - Ultra-lightweight Railway deployment (Uses Gemini for embeddings)

import os
from typing import List
import time
import hashlib
import numpy as np

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Gemini", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
gemini_model = None
pc = None
index = None

# Initialize AI models and services
def initialize_services():
    global gemini_model
    
    if gemini_model is None:
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("Gemini model initialized")
    
    return gemini_model

# Initialize Pinecone
def init_pinecone():
    global pc, index
    
    if pc is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if index is None:
        index_name = "policy-docs-gemini-hash"
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=512,  # Using simple hash-based embeddings
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

# Simple embedding function using text hashing (no ML dependencies)
def get_simple_embedding(text: str) -> List[float]:
    """Create a simple embedding using text hashing and basic features"""
    try:
        # Normalize text
        text = text.lower().strip()
        
        # Create multiple hash-based features
        embeddings = []
        
        # Hash-based features (first 256 dimensions)
        for i in range(8):
            hash_obj = hashlib.md5(f"{text}_{i}".encode())
            hash_bytes = hash_obj.digest()
            # Convert to normalized floats
            for j in range(0, len(hash_bytes), 4):
                if len(embeddings) < 256:
                    val = int.from_bytes(hash_bytes[j:j+4], byteorder='big', signed=False)
                    normalized_val = (val / (2**32 - 1)) * 2 - 1  # Normalize to [-1, 1]
                    embeddings.append(normalized_val)
        
        # Simple statistical features (remaining 256 dimensions)
        words = text.split()
        
        # Add basic text statistics
        features = [
            len(text) / 1000.0,  # Text length
            len(words) / 100.0,  # Word count
            sum(len(word) for word in words) / max(len(words), 1) / 10.0,  # Avg word length
            len(set(words)) / max(len(words), 1),  # Unique word ratio
        ]
        
        # Pad with word-based hashes
        for word in words[:252]:  # Take first 252 words
            word_hash = hash(word) % 10000
            features.append(word_hash / 10000.0)
        
        # Pad to exactly 256 features
        while len(features) < 256:
            features.append(0.0)
        
        # Combine hash and statistical features
        embeddings.extend(features[:256])
        
        # Ensure exactly 512 dimensions
        return embeddings[:512]
        
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return zero vector if error
        return [0.0] * 512

# Alternative: Use Gemini for embeddings (API-based, no local ML dependencies)
def get_gemini_embedding(text: str, gemini_model) -> List[float]:
    """Use Gemini to create embeddings via API"""
    try:
        # Use Gemini to generate a semantic representation
        prompt = f"Create a semantic summary of this text in exactly 50 keywords, separated by commas: {text[:1000]}"
        response = gemini_model.generate_content(prompt)
        keywords = response.text.strip()
        
        # Convert keywords to embedding using simple hashing
        return get_simple_embedding(keywords)
        
    except Exception as e:
        print(f"Error getting Gemini embedding: {e}")
        # Fallback to simple embedding
        return get_simple_embedding(text)

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
def query_chunks(query: str, index, gemini_model, top_k: int = 5) -> List[str]:
    # Try Gemini-based embedding first, fallback to simple embedding
    try:
        query_embedding = get_gemini_embedding(query, gemini_model)
    except:
        query_embedding = get_simple_embedding(query)
    
    if not query_embedding:
        return []
    
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match.metadata.get("text", "") for match in query_response.matches]

# Pinecone upsert functions
def upsert_chunks(chunks: List[str], index, gemini_model):
    vectors = []
    for i, chunk in enumerate(chunks):
        # Try Gemini-based embedding first, fallback to simple embedding
        try:
            embedding = get_gemini_embedding(chunk, gemini_model)
        except:
            embedding = get_simple_embedding(chunk)
            
        if embedding is not None:
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

# Initialize policy document function
def initialize_policy_document():
    """Extract and upsert the policy.pdf document once at startup"""
    global index, gemini_model
    
    policy_file = "policy.pdf"
    
    if not os.path.exists(policy_file):
        print(f"Warning: Policy file {policy_file} not found. Skipping document initialization.")
        return
    
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
    upsert_chunks(chunks, index, gemini_model)
    print("Policy document successfully indexed")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    # Debug logging
    print(f"DEBUG: Received token: {token}")
    print(f"DEBUG: Expected token: {expected_token}")
    print(f"DEBUG: All env vars: {dict(os.environ)}")
    
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API_BEARER_TOKEN not configured. Available env vars: {list(os.environ.keys())}"
        )
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"Invalid token. Expected: {expected_token}, Got: {token}"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize services once at startup for Railway
print("Initializing services...")
gemini_model = initialize_services()
pc, index = init_pinecone()

# Initialize policy document if available
initialize_policy_document()

print("All services initialized successfully!")

# API endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    # Use pre-initialized services (Railway pattern)
    global gemini_model, pc, index
    
    answers = []
    
    for question in req.questions:
        try:
            relevant_clauses = query_chunks(question, index, gemini_model)
            
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
            "embedding_model": "gemini-enhanced-hash-embeddings",
            "llm_model": "gemini-pro"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# For Railway deployment - run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
