import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import time
import gc
import hashlib

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai
from pinecone import Pinecone
import numpy as np
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with more explicit configuration
app = FastAPI(
    title="Railway Semantic Search API", 
    version="1.0.0",
    debug=True,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")

# Global variables for models
embedding_model = None
pinecone_client = None
pc_index = None

class QueryRequest(BaseModel):
    document_url: Optional[HttpUrl] = None
    questions: List[str]

class Answer(BaseModel):
    answer: str

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not API_BEARER_TOKEN or token != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# Initialize models and connections
def initialize_models():
    global embedding_model, pinecone_client, pc_index
    
    try:
        # Initialize Gemini
        if not API_BEARER_TOKEN:
            raise ValueError("API_BEARER_TOKEN not found in environment variables")
        # Initialize Gemini API
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize embedding model (no API key needed - open source model)
        logger.info("Loading paraphrase-MiniLM-L6-v2 model...")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pinecone_client.Index("first")
        logger.info("Pinecone connection established")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def extract_text_from_pdf(pdf_url: str) -> str:
    """Extract text from PDF URL"""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")

def chunk_text(text: str, chunk_size: 400, overlap: 50) -> List[str]:
    """Chunk text with overlap using tiktoken for token counting"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback to simple text chunking
        words = text.split()
        chunk_size_words = chunk_size // 4  # Rough approximation
        overlap_words = overlap // 4
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size_words
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - overlap_words
        
        return chunks

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks"""
    try:
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

def store_embeddings_in_pinecone(chunks: List[str], embeddings: List[List[float]], doc_id: str):
    """Store embeddings in Pinecone"""
    try:
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk[:1000],  # Limit metadata size
                    "doc_id": doc_id,
                    "chunk_index": i
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            pc_index.upsert(vectors=batch)
        
        logger.info(f"Stored {len(vectors)} vectors in Pinecone")
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(e)}")

def search_similar_chunks(query: str, top_k: int = 15) -> List[str]:
    """Search for similar chunks in Pinecone - FIXED VERSION"""
    try:
        logger.info(f"Searching for query: '{query}'")
        
        # Generate query embedding using the SAME model as indexing
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0].tolist()
        
        # Search in Pinecone
        search_results = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False  # Don't need the vectors back
        )
        
        logger.info(f"Pinecone returned {len(search_results.matches)} matches")
        
        # Log all results for debugging
        contexts = []
        for i, match in enumerate(search_results.matches):
            score = match.score
            text = match.metadata.get('text', '') if match.metadata else ''
            logger.info(f"Match {i}: ID={match.id}, Score={score:.4f}, Text preview: {text[:100]}...")
            
            # SIGNIFICANTLY lowered threshold - was 0.3, now 0.1
            if score > 0.1:
                contexts.append(text)
                logger.info(f"✓ Added context {i} with score {score:.4f}")
            else:
                logger.info(f"✗ Rejected context {i} with score {score:.4f} (below 0.1 threshold)")
        
        if not contexts:
            logger.warning("No contexts found above threshold!")
            # If no good matches, take the top 3 anyway for debugging
            for match in search_results.matches[:3]:
                if match.metadata and match.metadata.get('text'):
                    contexts.append(match.metadata['text'])
                    logger.info(f"Added fallback context with score {match.score:.4f}")
        
        logger.info(f"Returning {len(contexts)} contexts total")
        return contexts
        
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        return []

def generate_answer_with_gemini(question: str, contexts: List[str]) -> str:
    """Generate answer using Gemini 1.5 Flash - IMPROVED VERSION"""
    try:
        # Use more contexts for better answers
        context_text = "\n\n".join(contexts[:5])  # Use top 5 contexts instead of 3
        
        logger.info(f"Generating answer for: {question}")
        logger.info(f"Using {len(contexts)} contexts, total length: {len(context_text)} chars")
        
        prompt = f"""Based on the following context from documents, provide a comprehensive and accurate answer to the question. If you cannot find the specific information in the context, try to provide related information that might be helpful, and clearly state what information is missing.

Context:
{context_text}

Question: {question}

Answer (be specific and detailed):"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.2,  # Slightly higher for more natural responses
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
        )
        
        answer = response.text.strip()
        logger.info(f"Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"Error generating answer: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("=== APPLICATION STARTUP ===")
    logger.info(f"PORT: {os.getenv('PORT', '8000')}")
    logger.info(f"Environment variables check:")
    logger.info(f"  API_BEARER_TOKEN: {'SET' if API_BEARER_TOKEN else 'MISSING'}")
    logger.info(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'MISSING'}")
    logger.info(f"  PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'MISSING'}")
    
    try:
        initialize_models()
        logger.info("=== STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"=== STARTUP FAILED: {e} ===")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint - no auth required"""
    return {"message": "API is working!", "timestamp": time.time()}

# NEW DEBUG ENDPOINT
@app.post("/test-search")
async def test_search_endpoint(
    query: str,
    token: str = Depends(verify_token)
):
    """Debug endpoint to test search functionality"""
    try:
        contexts = search_similar_chunks(query, top_k=10)
        return {
            "query": query,
            "contexts_found": len(contexts),
            "contexts": contexts[:3],  # Return first 3 for inspection
            "index_stats": pc_index.describe_index_stats()
        }
    except Exception as e:
        logger.error(f"Error in test search: {e}")
        return {"error": str(e)}

@app.post("/api/hackrx/run", response_model=QueryResponse)
async def process_query_api(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main API endpoint for processing queries (original path)"""
    return await process_query_logic(request)

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing queries"""
    return await process_query_logic(request)

async def process_query_logic(request: QueryRequest):
    """Shared logic for processing queries - IMPROVED VERSION"""
    try:
        start_time = time.time()
        
        # Only process document if URL is provided
        if request.document_url:
            # Extract text from PDF
            logger.info(f"Processing document: {request.document_url}")
            pdf_text = extract_text_from_pdf(str(request.document_url))
            
            if not pdf_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
            
            # Chunk the text
            chunks = chunk_text(pdf_text)
            logger.info(f"Created {len(chunks)} chunks from {len(pdf_text)} characters")
            
            # Generate embeddings
            embeddings = generate_embeddings(chunks)
            
            # Store in Pinecone with unique doc ID
            doc_id = f"doc_{int(time.time())}"
            store_embeddings_in_pinecone(chunks, embeddings, doc_id)
            logger.info(f"Indexed new document with ID: {doc_id}")
        else:
            logger.info("No document URL provided, searching existing index")
        
        # Check if index has data
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        logger.info(f"Pinecone index has {total_vectors} total vectors")
        
        if total_vectors == 0:
            logger.warning("Pinecone index is empty!")
            return QueryResponse(answers=["The document index is empty. Please index some documents first."] * len(request.questions))
        
        # Process questions in parallel
        async def process_question(question: str) -> str:
            try:
                logger.info(f"Processing question: {question}")
                
                # Search for relevant contexts
                contexts = search_similar_chunks(question, top_k=15)
                
                if not contexts:
                    logger.warning(f"No contexts found for question: {question}")
                    return "I couldn't find relevant information in the indexed documents for this question."
                
                # Generate answer
                answer_text = generate_answer_with_gemini(question, contexts)
                
                return answer_text
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                return f"Error processing question: {str(e)}"
        
        # Process all questions
        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Clean up memory
        gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Railway Semantic Search API",
        "version": "1.0.0",
        "status": "active",
        "available_endpoints": [
            "GET /",
            "GET /health",
            "GET /debug",
            "POST /test-search",
            "POST /hackrx/run",
            "POST /api/hackrx/run"
        ]
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check app status - ENHANCED"""
    try:
        # Check Pinecone index stats
        index_stats = pc_index.describe_index_stats()
        
        # Test embedding generation
        test_embedding = embedding_model.encode(["test query"], convert_to_tensor=False)[0]
        
        return {
            "app_status": "running",
            "models_loaded": embedding_model is not None,
            "pinecone_connected": pc_index is not None,
            "embedding_model_working": len(test_embedding) > 0,
            "embedding_dimension": len(test_embedding),
            "pinecone_stats": {
                "total_vectors": index_stats.total_vector_count,
                "namespaces": index_stats.namespaces,
                "dimension": index_stats.dimension
            },
            "available_routes": [str(route.path) for route in app.routes],
            "environment_vars": {
                "API_BEARER_TOKEN": "***" if API_BEARER_TOKEN else "MISSING",
                "GEMINI_API_KEY": "***" if GEMINI_API_KEY else "MISSING",
                "PINECONE_API_KEY": "***" if PINECONE_API_KEY else "MISSING",
                "PORT": os.getenv("PORT", "8000")
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "app_status": "running",
            "models_loaded": embedding_model is not None,
            "pinecone_connected": pc_index is not None
        }

# Add a catch-all route for debugging
from fastapi import Request

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    """Catch-all route to see what requests are being made"""
    return {
        "error": f"Route not found: {request.method} /{path}",
        "available_routes": [
            "GET /",
            "GET /health", 
            "GET /test",
            "GET /debug",
            "GET /docs",
            "POST /test-search",
            "POST /hackrx/run",
            "POST /api/hackrx/run"
        ],
        "request_info": {
            "method": request.method,
            "path": path,
            "headers": dict(request.headers)
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Available routes:")
    for route in app.routes:
        logger.info(f"  {route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False, 
        log_level="info",
        access_log=True
    )
