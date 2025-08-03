# main.py - Railway-optimized Insurance Policy RAG API

import os
from typing import List
import time
import hashlib
import traceback
import logging
from contextlib import asynccontextmanager

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
import pinecone
import openai

# Global variables for models
openai_client = None
pc = None
index = None
initialization_status = {
    "openai": False,
    "pinecone": False,
    "document": False,
    "error": None,
    "startup_complete": False
}

# Startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== RAILWAY STARTUP: Initializing services ===")
    await startup_services()
    logger.info("=== RAILWAY STARTUP COMPLETE ===")
    
    yield
    
    # Shutdown
    logger.info("=== RAILWAY SHUTDOWN ===")

# FastAPI setup with lifespan
app = FastAPI(
    title="Insurance Policy RAG API with OpenAI", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup_services():
    global openai_client, pc, index, initialization_status
    
    try:
        # Quick health check endpoint should work immediately
        initialization_status["startup_complete"] = True
        
        # Initialize OpenAI (fast)
        await initialize_openai()
        logger.info("✓ OpenAI initialized")
        
        # Initialize Pinecone (can be slow)
        await initialize_pinecone()
        logger.info("✓ Pinecone initialized")
        
        # Initialize document (slowest - make it non-blocking)
        try:
            await initialize_document()
            logger.info("✓ Document initialized")
        except Exception as doc_error:
            logger.warning(f"⚠ Document initialization failed: {doc_error}")
            logger.info("Service will continue without pre-loaded document")
            initialization_status["error"] = f"Document init failed: {str(doc_error)}"
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        # Don't raise - let the service start anyway for health checks

async def initialize_openai():
    global openai_client, initialization_status
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Use legacy OpenAI client (v0.28.x) - more stable
        openai.api_key = api_key
        openai_client = openai
        
        # Test the client with a simple request
        test_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        logger.info("✓ OpenAI client initialized and tested successfully")
        initialization_status["openai"] = True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI: {e}")
        initialization_status["error"] = str(e)
        raise e

async def initialize_pinecone():
    global pc, index, initialization_status
    
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # Add environment support
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        # Initialize Pinecone (v2.x API)
        pinecone.init(api_key=api_key, environment=environment)
        
        index_name = "policy-docs-gemini-hash"
        
        # Check existing indexes
        existing_indexes = pinecone.list_indexes()
        logger.info(f"📋 Found existing indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            logger.info(f"Creating index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=512,
                metric="cosine"
            )
            
            # Wait for index to be ready
            max_wait = 30
            waited = 0
            while waited < max_wait:
                try:
                    if pinecone.describe_index(index_name).status['ready']:
                        break
                except:
                    pass
                time.sleep(2)
                waited += 2
                if waited % 10 == 0:
                    logger.info(f"Still waiting for index... ({waited}/{max_wait}s)")
            
            if waited >= max_wait:
                logger.warning("Index creation timeout - will retry later")
        
        index = pinecone.Index(index_name)
        pc = pinecone  # Keep reference for compatibility
        initialization_status["pinecone"] = True
        logger.info("✓ Pinecone initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        initialization_status["error"] = str(e)
        raise e

async def initialize_document():
    global index, openai_client, initialization_status
    
    try:
        policy_file = "policy.pdf"
        
        if not os.path.exists(policy_file):
            logger.warning(f"Policy file {policy_file} not found")
            return
        
        # Extract and process document
        full_text = extract_text(policy_file)
        chunks = chunk_text(full_text)
        
        # Clear existing vectors
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                index.delete(delete_all=True)
                time.sleep(2)
        except:
            pass
        
        # Upsert chunks
        upsert_chunks(chunks, index, openai_client)
        initialization_status["document"] = True
        
    except Exception as e:
        logger.error(f"Error initializing document: {e}")
        raise e

# Text extraction functions (unchanged)
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

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

def get_simple_embedding(text: str) -> List[float]:
    """Create a simple embedding using text hashing - EXACTLY 512 dimensions"""
    try:
        text = text.lower().strip()
        embeddings = []
        
        # Hash-based features (256 dimensions)
        for i in range(16):
            hash_obj = hashlib.md5(f"{text}_{i}".encode())
            hash_bytes = hash_obj.digest()
            for byte_val in hash_bytes:
                if len(embeddings) < 256:
                    normalized_val = (byte_val / 255.0) * 2 - 1
                    embeddings.append(normalized_val)
        
        # Word-based features (256 dimensions)
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
        
        # Fill remaining features
        for i in range(246):
            if i < len(words):
                word_hash = hash(f"{words[i]}_{i}") % 10000
                word_features.append(word_hash / 10000.0)
            else:
                if i < len(text):
                    char_hash = hash(f"{text[i]}_{i}") % 10000
                    word_features.append(char_hash / 10000.0)
                else:
                    word_features.append(0.0)
        
        word_features = word_features[:256]
        while len(word_features) < 256:
            word_features.append(0.0)
        
        embeddings.extend(word_features)
        
        # Ensure exactly 512 dimensions
        embeddings = embeddings[:512]
        while len(embeddings) < 512:
            embeddings.append(0.0)
            
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return [0.0] * 512

def get_openai_embedding(text: str, openai_client) -> List[float]:
    """Use OpenAI to create embeddings via API"""
    try:
        # Using legacy OpenAI client (v0.28.x)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f"Create a semantic summary of this text in exactly 50 keywords, separated by commas: {text[:1000]}"
            }],
            max_tokens=150,
            temperature=0.1
        )
        keywords = response['choices'][0]['message']['content'].strip()
        embedding = get_simple_embedding(keywords)
        
        if len(embedding) != 512:
            if len(embedding) < 512:
                embedding.extend([0.0] * (512 - len(embedding)))
            else:
                embedding = embedding[:512]
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error getting OpenAI embedding: {e}")
        return get_simple_embedding(text)

def query_openai(question: str, context_clauses: List[str], openai_client) -> str:
    prompt = f"""
You are an expert assistant who answers insurance policy questions precisely and cites the clauses.

Question: {question}

Use ONLY the following clauses and explicitly mention or quote them in your answer:

{chr(10).join([f"- {clause}" for clause in context_clauses])}

Answer:
"""
    
    try:
        # Using legacy OpenAI client (v0.28.x)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        return response['choices'][0]['message']['content']
            
    except Exception as e:
        logger.error(f"Error generating OpenAI response: {e}")
        return f"Error generating response: {str(e)}"

def query_chunks(query: str, index, openai_client, top_k: int = 5) -> List[str]:
    try:
        try:
            query_embedding = get_openai_embedding(query, openai_client)
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
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

def upsert_chunks(chunks: List[str], index, openai_client):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_openai_embedding(chunk, openai_client)
            except:
                embedding = get_simple_embedding(chunk)
                
            if embedding is not None:
                vectors.append({
                    "id": f"chunk-{i}-{int(time.time())}",
                    "values": embedding,
                    "metadata": {"text": chunk}
                })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(chunks)} chunks to Pinecone.")
    except Exception as e:
        logger.error(f"Error upserting chunks: {e}")
        raise e

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
            detail="Invalid authentication token"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# API endpoints
@app.get("/health")
async def health_check():
    """Fast health check that always responds quickly"""
    return {
        "status": "healthy" if initialization_status["startup_complete"] else "starting",
        "message": "Insurance Policy RAG API is running",
        "initialization_status": initialization_status,
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
            "has_bearer_token": bool(os.getenv("API_BEARER_TOKEN")),
            "pinecone_env": os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    if not openai_client or not index:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not properly initialized. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            relevant_clauses = query_chunks(question, index, openai_client)
            
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document for this question.")
                continue
            
            answer = query_openai(question, relevant_clauses, openai_client)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

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
            "embedding_model": "openai-enhanced-hash-embeddings",
            "llm_model": "gpt-3.5-turbo",
            "initialization_status": initialization_status
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Insurance Policy RAG API with OpenAI",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "query": "/hackrx/run (POST)"
        }
    }

# Railway/Docker deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📋 Files in directory: {os.listdir('.')}")
    
    # Log environment status
    logger.info(f"🔑 Environment check:")
    logger.info(f"  - OPENAI_API_KEY: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    logger.info(f"  - PINECONE_API_KEY: {'✓' if os.getenv('PINECONE_API_KEY') else '✗'}")
    logger.info(f"  - PINECONE_ENVIRONMENT: {os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')}")
    logger.info(f"  - API_BEARER_TOKEN: {'✓' if os.getenv('API_BEARER_TOKEN') else '✗'}")
    logger.info(f"  - policy.pdf: {'✓' if os.path.exists('policy.pdf') else '✗'}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        timeout_keep_alive=65,
        access_log=True
    )
