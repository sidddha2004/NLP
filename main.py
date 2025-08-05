# main.py - Phase 2 Query Service for Railway
import os
from typing import List
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
import google.generativeai as genai

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables - only what's needed for queries
gemini_model = None
index = None
initialization_status = {"ready": False, "error": None}

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def initialize_query_services():
    """Initialize only query-related services - fast startup"""
    global gemini_model, index, initialization_status
    
    try:
        logger.info("🚀 Initializing Phase 2 Query Service...")
        
        # Initialize Gemini for queries only
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found")
            
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        logger.info("✓ Gemini initialized")
        
        # Connect to existing Pinecone index (read-only)
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found")
            
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = os.getenv("PINECONE_INDEX", "policy-docs-production")
        index = pc.Index(index_name)
        logger.info(f"✓ Connected to Pinecone index: {index_name}")
        
        # Verify index has data
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count
        logger.info(f"✓ Index contains {vector_count} vectors")
        
        if vector_count == 0:
            logger.warning("⚠️ Index is empty - documents may not be processed yet")
        
        initialization_status["ready"] = True
        logger.info("🎯 Phase 2 Query Service ready!")
        
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        initialization_status["error"] = error_msg
        raise

def get_query_embedding(query: str) -> List[float]:
    """Generate embedding for user query using Google's model"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"  # Optimized for queries
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Query embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

def search_relevant_chunks(query: str, top_k=5) -> List[str]:
    """Search for relevant document chunks in Pinecone"""
    try:
        query_embedding = get_query_embedding(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [match.metadata.get('text', '') for match in results.matches if match.metadata]
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate final answer using Gemini"""
    try:
        if not context_chunks:
            return "No relevant information found in the policy documents."
        
        context = "\n".join([f"- {chunk}" for chunk in context_chunks[:5]])  # Limit context
        
        prompt = f"""Based on the following insurance policy excerpts, answer the question accurately and cite specific clauses:

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Cite specific policy sections when possible
- If the context doesn't contain enough information, say so
- Be concise and accurate

Answer:"""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return f"Error generating answer: Unable to process your question at this time."

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        initialize_query_services()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue anyway to allow health checks
        yield
    finally:
        # Cleanup (if needed)
        logger.info("Shutting down...")

# FastAPI app with lifespan
app = FastAPI(
    title="Insurance Policy Query API - Phase 2",
    version="2.0.0",
    description="Lightweight query service for insurance policy RAG system",
    lifespan=lifespan
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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API bearer token"""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid authentication token"
        )
    return True

# API Endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    """Process user queries and return answers"""
    
    # Check if service is ready
    if not initialization_status["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {initialization_status.get('error', 'Unknown error')}"
        )
    
    if not req.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Search for relevant chunks
            relevant_chunks = search_relevant_chunks(question, top_k=5)
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Unexpected error processing question: {e}")
            answers.append("I apologize, but I encountered an error processing your question. Please try again.")
    
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    """Health check endpoint - always responds quickly"""
    return {
        "status": "healthy" if initialization_status["ready"] else "initializing",
        "message": "Phase 2 Query Service",
        "ready": initialization_status["ready"],
        "error": initialization_status.get("error"),
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
            "has_pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
            "has_bearer_token": bool(os.getenv("API_BEARER_TOKEN")),
            "pinecone_index": os.getenv("PINECONE_INDEX", "policy-docs-production")
        }
    }

@app.get("/status")
async def detailed_status():
    """Detailed status including vector database info"""
    try:
        if not index:
            return {"status": "error", "message": "Index not initialized"}
            
        stats = index.describe_index_stats()
        return {
            "status": "operational" if initialization_status["ready"] else "initializing",
            "initialization": initialization_status,
            "vector_database": {
                "total_vectors": stats.total_vector_count,
                "index_fullness": stats.index_fullness,
                "ready": stats.total_vector_count > 0
            },
            "models": {
                "llm": "gemini-1.5-pro",
                "embedding": "models/embedding-001"
            }
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insurance Policy Query API - Phase 2",
        "version": "2.0.0",
        "description": "Lightweight query service for processed insurance documents",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "query": "/hackrx/run (POST)"
        },
        "phase": "Query Service Only - Documents processed separately"
    }

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
