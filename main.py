# Phase 2: Query & Retrieval API (Deploy on Railway)
# This is a lightweight API that only handles queries and retrieval

import os
from typing import List
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI and vector database - only query functionality
from pinecone import Pinecone
import google.generativeai as genai

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG Query API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
gemini_model = None
pc = None
index = None
initialization_status = {
    "gemini": False,
    "pinecone": False,
    "ready": False,
    "error": None
}

class QueryService:
    def __init__(self):
        self.gemini_model = None
        self.index = None
        
    def initialize_services(self):
        """Initialize Gemini and Pinecone services"""
        global gemini_model, pc, index, initialization_status
        
        try:
            # Initialize Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_model = gemini_model
            logger.info("✓ Gemini model initialized successfully")
            initialization_status["gemini"] = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            initialization_status["error"] = str(e)
            raise e
    
    def connect_to_pinecone(self):
        """Connect to existing Pinecone index"""
        global pc, index, initialization_status
        
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            pc = Pinecone(api_key=api_key)
            logger.info("✓ Pinecone client initialized")
            
            # Connect to existing index
            index_name = "policy-docs-gemini-hash"
            
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                raise Exception(f"Index '{index_name}' not found. Please run Phase 1 first to create and populate the index.")
            
            index = pc.Index(index_name)
            self.index = index
            logger.info(f"✓ Connected to existing index: {index_name}")
            
            # Verify index has data
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                raise Exception("Index is empty. Please run Phase 1 first to populate the index with document embeddings.")
            
            logger.info(f"✓ Index ready with {stats.total_vector_count} vectors")
            initialization_status["pinecone"] = True
            initialization_status["ready"] = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            initialization_status["error"] = str(e)
            raise e
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for query using same method as Phase 1"""
        try:
            # Use Gemini to enhance query understanding
            prompt = f"""
            Analyze this insurance policy question and extract the key concepts and terms that should be searched for.
            Focus on: policy terms, coverage types, conditions, exclusions, procedures, and related legal concepts.
            
            Question: {query}
            
            Provide exactly 50 key search terms and concepts related to this question, separated by commas:
            """
            
            response = self.gemini_model.generate_content(prompt)
            search_terms = response.text.strip()
            
            # Convert to embedding using same method as Phase 1
            embedding = self._text_to_embedding(search_terms + " " + query)
            
            if len(embedding) != 768:
                if len(embedding) < 768:
                    embedding.extend([0.0] * (768 - len(embedding)))
                else:
                    embedding = embedding[:768]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            # Fallback to simple embedding
            return self._text_to_embedding(query)
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to 768-dimensional embedding (same as Phase 1)"""
        import hashlib
        
        embeddings = []
        text = text.lower().strip()
        
        # Multiple hash functions for different aspects (matching Phase 1)
        hash_functions = [
            lambda t, i: hashlib.md5(f"{t}_{i}".encode()).digest(),
            lambda t, i: hashlib.sha1(f"{t}_{i}".encode()).digest(),
            lambda t, i: hashlib.sha256(f"{t}_{i}".encode()).digest(),
        ]
        
        # Generate embeddings from different hash functions
        for hash_func in hash_functions:
            for i in range(256):
                if len(embeddings) >= 768:
                    break
                hash_bytes = hash_func(text, i)
                for byte_val in hash_bytes:
                    if len(embeddings) >= 768:
                        break
                    normalized_val = (byte_val / 255.0) * 2 - 1
                    embeddings.append(normalized_val)
        
        # Ensure exactly 768 dimensions
        embeddings = embeddings[:768]
        while len(embeddings) < 768:
            embeddings.append(0.0)
        
        return embeddings
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant chunks in Pinecone"""
        try:
            query_embedding = self.create_query_embedding(query)
            
            if not query_embedding:
                return []
            
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            relevant_texts = []
            for match in query_response.matches:
                if match.score > 0.3:  # Similarity threshold
                    text = match.metadata.get("text", "")
                    if text:
                        relevant_texts.append(text)
            
            logger.info(f"Found {len(relevant_texts)} relevant chunks for query")
            return relevant_texts
            
        except Exception as e:
            logger.error(f"Error searching for relevant chunks: {e}")
            return []
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer using Gemini with retrieved context"""
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the policy document to answer your question. Please try rephrasing your question or contact support for assistance."
            
            context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
            
            prompt = f"""
You are an expert insurance policy assistant. Answer the question based ONLY on the provided policy context.

Question: {question}

Policy Context:
{context}

Instructions:
1. Answer based strictly on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Quote specific policy clauses when relevant
4. Be precise and avoid speculation
5. If there are exclusions or conditions, mention them clearly
6. Use clear, professional language

Answer:
"""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try again or contact support."

# Initialize service
query_service = QueryService()

# Authentication
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
        logger.warning("Invalid token received")
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=== PHASE 2 API STARTUP ===")
    
    try:
        # Initialize services
        query_service.initialize_services()
        query_service.connect_to_pinecone()
        
        logger.info("🎉 Phase 2 API ready to serve queries!")
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        # Don't raise - let health checks show the error

# Main API endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    """Process insurance policy questions"""
    
    if not initialization_status["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Search for relevant chunks
            relevant_chunks = query_service.search_relevant_chunks(question, top_k=5)
            
            # Generate answer
            answer = query_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Quick health check"""
    return {
        "status": "healthy" if initialization_status["ready"] else "initializing",
        "message": "Phase 2 - Insurance Policy Query API",
        "initialization_status": initialization_status,
        "phase": 2,
        "description": "Query and retrieval only - no document processing"
    }

@app.get("/status")
async def detailed_status():
    """Detailed status information"""
    try:
        status_info = {
            "phase": 2,
            "services": initialization_status,
            "ready": initialization_status["ready"]
        }
        
        if query_service.index:
            try:
                stats = query_service.index.describe_index_stats()
                status_info["vector_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "index_fullness": stats.index_fullness
                }
            except Exception as stats_error:
                status_info["vector_stats"] = {"error": str(stats_error)}
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting detailed status: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/info")
async def get_info():
    """API information"""
    try:
        if not query_service.index:
            return {"status": "error", "message": "Index not connected"}
            
        stats = query_service.index.describe_index_stats()
        return {
            "phase": 2,
            "status": "ready",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "gemini-enhanced-semantic-embeddings",
            "llm_model": "gemini-1.5-flash",
            "functionality": "query_and_retrieval_only",
            "description": "Lightweight API for querying pre-indexed policy documents"
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    """Root endpoint information"""
    return {
        "message": "Insurance Policy RAG Query API - Phase 2",
        "version": "2.0.0",
        "phase": 2,
        "description": "Query and retrieval only - documents are pre-indexed in Phase 1",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "info": "/info",
            "query": "/hackrx/run (POST)"
        },
        "note": "Ensure Phase 1 has been run to populate the Pinecone index"
    }

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
