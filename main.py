# main.py - Fixed Railway deployment for Insurance Policy RAG API with Hugging Face

import os
from typing import List
import time
import hashlib
import traceback
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
import requests

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Hugging Face", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
hf_api_key = None
pc = None
index = None
initialization_status = {
    "huggingface": False,
    "pinecone": False,
    "document": False,
    "error": None
}

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

# Initialize AI models and services
def initialize_services():
    global hf_api_key, initialization_status
    
    try:
        if hf_api_key is None:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
            
            hf_api_key = api_key
            logger.info("Hugging Face API key initialized successfully")
            initialization_status["huggingface"] = True
        
        return hf_api_key
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face: {e}")
        initialization_status["error"] = str(e)
        raise e

# Initialize Pinecone
def init_pinecone():
    global pc, index, initialization_status
    
    try:
        logger.info("🔧 Initializing Pinecone...")
        if pc is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            pc = Pinecone(api_key=api_key)
            logger.info("✓ Pinecone client initialized")
        
        if index is None:
            index_name = "policy-docs-hf-embeddings"
            
            # Check if index exists
            try:
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                logger.info(f"📋 Found existing indexes: {existing_indexes}")
                
                if index_name not in existing_indexes:
                    logger.info(f"🏗 Creating new index: {index_name}")
                    pc.create_index(
                        name=index_name,
                        dimension=384,  # Using sentence-transformers/all-MiniLM-L6-v2 embeddings
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    logger.info(f"✓ Created new index: {index_name}")
                    
                    # Wait for index to be ready
                    logger.info("⏳ Waiting for index to be ready...")
                    max_retries = 60
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            if pc.describe_index(index_name).status['ready']:
                                break
                        except Exception as wait_error:
                            logger.warning(f"Waiting for index, attempt {retry_count}: {wait_error}")
                        time.sleep(2)
                        retry_count += 1
                        if retry_count % 10 == 0:
                            logger.info(f"Still waiting... ({retry_count}/{max_retries})")
                    
                    if retry_count >= max_retries:
                        raise Exception("Index creation timeout")
                    logger.info("✓ Index is ready!")
                else:
                    logger.info(f"✓ Using existing index: {index_name}")
                
                index = pc.Index(index_name)
                logger.info("✓ Connected to index")
                initialization_status["pinecone"] = True
                
            except Exception as e:
                logger.error(f"❌ Error with Pinecone index: {e}")
                raise e
        
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        initialization_status["error"] = str(e)
        raise e

# Text extraction functions
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise e

def extract_text_from_docx(docx_path: str) -> str:
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise e

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

# Simple embedding function using text hashing (fallback)
def get_simple_embedding(text: str) -> List[float]:
    """Create a simple embedding using text hashing - EXACTLY 384 dimensions for fallback"""
    try:
        text = text.lower().strip()
        embeddings = []
        
        # Hash-based features (192 dimensions)
        for i in range(12):  # 12 hash iterations
            hash_obj = hashlib.md5(f"{text}_{i}".encode())
            hash_bytes = hash_obj.digest()
            for byte_val in hash_bytes:
                if len(embeddings) < 192:
                    normalized_val = (byte_val / 255.0) * 2 - 1
                    embeddings.append(normalized_val)
        
        # Word-based features (192 dimensions)
        words = text.split()
        word_features = []
        
        # Basic text statistics
        word_features.extend([
            len(text) / 1000.0,
            len(words) / 100.0,
            sum(len(word) for word in words) / max(len(words), 1) / 10.0,
            len(set(words)) / max(len(words), 1),
            text.count(' ') / max(len(text), 1),
            text.count('.') / max(len(text), 1),
            text.count(',') / max(len(text), 1),
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            len([w for w in words if len(w) > 5]) / max(len(words), 1)
        ])
        
        # Hash features for remaining dimensions
        for i in range(182):  # 192 - 10 = 182
            if i < len(words):
                word_hash = hash(f"{words[i]}_{i}") % 10000
                word_features.append(word_hash / 10000.0)
            else:
                if i < len(text):
                    char_hash = hash(f"{text[i]}_{i}") % 10000
                    word_features.append(char_hash / 10000.0)
                else:
                    word_features.append(0.0)
        
        word_features = word_features[:192]
        while len(word_features) < 192:
            word_features.append(0.0)
        
        embeddings.extend(word_features)
        
        # Ensure exactly 384 dimensions
        embeddings = embeddings[:384]
        while len(embeddings) < 384:
            embeddings.append(0.0)
            
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating simple embedding: {e}")
        return [0.0] * 384

# Hugging Face API call for embeddings
def get_hf_embedding(text: str, hf_api_key: str) -> List[float]:
    """Use Hugging Face API to create embeddings - 384 dimensions"""
    try:
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        
        # Truncate text if too long
        text = text[:500]  # Limit text length for API
        
        response = requests.post(
            HF_EMBEDDING_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()
            if isinstance(embedding, list) and len(embedding) > 0:
                # Handle different response formats
                if isinstance(embedding[0], list):
                    embedding = embedding[0]  # Take first embedding if batch
                
                # Ensure exactly 384 dimensions
                if len(embedding) == 384:
                    return embedding
                elif len(embedding) > 384:
                    return embedding[:384]
                else:
                    # Pad if smaller
                    embedding.extend([0.0] * (384 - len(embedding)))
                    return embedding
            else:
                logger.warning("Invalid embedding format from HF API")
                return get_simple_embedding(text)
        else:
            logger.warning(f"HF API error: {response.status_code} - {response.text}")
            return get_simple_embedding(text)
            
    except Exception as e:
        logger.error(f"Error getting HF embedding: {e}")
        return get_simple_embedding(text)

# Hugging Face API call for text generation
def query_huggingface(question: str, context_clauses: List[str], hf_api_key: str) -> str:
    """Use Hugging Face API for text generation"""
    
    # Create a focused prompt
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])  # Limit context
    
    prompt = f"""<s>[INST] You are an expert insurance assistant. Answer the question using only the provided policy clauses.

Question: {question}

Policy Clauses:
{context}

Provide a clear, specific answer citing the relevant clauses. [/INST]"""
    
    try:
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                return generated_text.strip()
            else:
                return "Error: Invalid response format from Hugging Face API"
        else:
            logger.error(f"HF API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from Hugging Face API (Status: {response.status_code})"
            
    except Exception as e:
        logger.error(f"Error generating HF response: {e}")
        return f"Error generating response: {str(e)}"

# Pinecone operations
def query_chunks(query: str, index, hf_api_key: str, top_k: int = 5) -> List[str]:
    try:
        # Get embedding for query
        query_embedding = get_hf_embedding(query, hf_api_key)
        
        if not query_embedding:
            return []
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata.get("text", "") for match in query_response.matches]
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

# Pinecone upsert functions
def upsert_chunks(chunks: List[str], index, hf_api_key: str):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_hf_embedding(chunk, hf_api_key)
                
            if embedding is not None:
                vectors.append({
                    "id": f"chunk-{i}-{int(time.time())}",
                    "values": embedding,
                    "metadata": {"text": chunk}
                })
            
            # Add delay to avoid rate limiting
            if i % 10 == 0:
                time.sleep(1)
        
        # Upsert in batches
        batch_size = 50  # Smaller batches for stability
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            time.sleep(0.5)  # Small delay between batches
        
        logger.info(f"Upserted {len(chunks)} chunks to Pinecone.")
    except Exception as e:
        logger.error(f"Error upserting chunks: {e}")
        raise e

# Initialize policy document function
def initialize_policy_document():
    """Extract and upsert the policy.pdf document once at startup"""
    global index, hf_api_key, initialization_status
    
    try:
        policy_file = "policy.pdf"
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        if not os.path.exists(policy_file):
            logger.error(f"Policy file {policy_file} not found in {os.getcwd()}")
            logger.error(f"Available files: {os.listdir('.')}")
            return
        
        file_stat = os.stat(policy_file)
        logger.info(f"Policy file found - Size: {file_stat.st_size} bytes")
        
        logger.info(f"Loading policy document: {policy_file}")
        
        # Extract text from the policy document
        full_text = extract_text(policy_file)
        logger.info(f"Extracted {len(full_text)} characters from policy document")
        
        # Chunk the text
        chunks = chunk_text(full_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Check if index has any vectors before trying to delete
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Clearing existing vectors from index...")
                index.delete(delete_all=True)
                logger.info("Cleared existing vectors from index")
                time.sleep(5)
            else:
                logger.info("Index is empty, no need to clear")
        except Exception as e:
            logger.warning(f"Could not check/clear index stats: {e}")
        
        # Upsert chunks to Pinecone
        upsert_chunks(chunks, index, hf_api_key)
        logger.info("Policy document successfully indexed")
        initialization_status["document"] = True
        
    except Exception as e:
        logger.error(f"Error initializing policy document: {e}")
        initialization_status["error"] = str(e)
        raise e

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not expected_token:
        logger.error("API_BEARER_TOKEN not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if token != expected_token:
        logger.warning(f"Invalid token received")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid authentication token"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize services once at startup for Railway
@app.on_event("startup")
async def startup_event():
    global hf_api_key, pc, index, initialization_status
    
    logger.info("=== STARTUP: Initializing services ===")
    
    try:
        # Initialize Hugging Face
        hf_api_key = initialize_services()
        logger.info("✓ Hugging Face initialized successfully")
        
        # Initialize Pinecone
        pc, index = init_pinecone()
        logger.info("✓ Pinecone initialized successfully")
        
        # Initialize policy document if available (non-blocking)
        try:
            initialize_policy_document()
            logger.info("✓ Policy document initialized successfully")
        except Exception as doc_error:
            logger.warning(f"⚠ Policy document initialization failed: {doc_error}")
            logger.info("Continuing without document...")
        
        logger.info("=== STARTUP COMPLETE ===")
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        logger.warning("⚠ Continuing with limited functionality...")

# API endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    # Check if services are initialized
    global hf_api_key, pc, index
    
    if not hf_api_key or not index:
        logger.error("Services not properly initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not properly initialized. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            relevant_clauses = query_chunks(question, index, hf_api_key)
            
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document for this question.")
                continue
            
            answer = query_huggingface(question, relevant_clauses, hf_api_key)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Insurance Policy RAG API with Hugging Face is running",
        "initialization_status": initialization_status,
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "has_hf_key": bool(os.getenv("HUGGINGFACE_API_KEY")),
            "has_pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
            "has_bearer_token": bool(os.getenv("API_BEARER_TOKEN"))
        }
    }

@app.get("/info")
async def get_info():
    global index
    try:
        if not index:
            return {"status": "error", "message": "Index not initialized"}
            
        stats = index.describe_index_stats()
        return {
            "status": "ready",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "mistralai/Mistral-7B-Instruct-v0.1",
            "initialization_status": initialization_status
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Insurance Policy RAG API with Hugging Face",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info", 
            "query": "/hackrx/run (POST)"
        }
    }

# For Railway deployment - run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
