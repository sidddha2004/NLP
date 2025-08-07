import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import time
import gc
import re
from collections import defaultdict

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

# Initialize FastAPI app
app = FastAPI(
    title="Railway Semantic Search API", 
    version="2.0.0",
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

class QueryResponse(BaseModel):
    answers: List[str]

class SearchDebugRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

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

def initialize_models():
    """Initialize models and connections"""
    global embedding_model, pinecone_client, pc_index
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize embedding model with exact same settings as Phase 1
        logger.info("Loading paraphrase-MiniLM-L6-v2 model...")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pinecone_client.Index("first")
        logger.info("Pinecone connection established")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def preprocess_query(query: str) -> str:
    """Preprocess query to match document preprocessing"""
    # Clean the query similar to how documents were processed
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    return query

def expand_query(query: str) -> List[str]:
    """Create query variations for better matching"""
    variations = [query]
    
    # Add simplified version
    simple_query = re.sub(r'[^\w\s]', ' ', query.lower())
    simple_query = re.sub(r'\s+', ' ', simple_query).strip()
    if simple_query != query.lower():
        variations.append(simple_query)
    
    # Add key terms only
    words = simple_query.split()
    if len(words) > 3:
        key_terms = ' '.join(words[:3])  # First 3 words
        variations.append(key_terms)
    
    return list(set(variations))  # Remove duplicates

def advanced_search_similar_chunks(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """Optimized search with reduced complexity for speed"""
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Preprocess query
        processed_query = preprocess_query(query)
        
        # Reduced to max 2 variations for speed
        query_variations = [processed_query]
        
        # Add only one simplified variation if query is complex
        simple_query = re.sub(r'[^\w\s]', ' ', processed_query.lower())
        simple_query = re.sub(r'\s+', ' ', simple_query).strip()
        if simple_query != processed_query.lower() and len(processed_query.split()) > 2:
            query_variations.append(simple_query)
        
        all_results = []
        
        # Search with fewer variations
        for i, query_var in enumerate(query_variations):
            # Generate embedding
            query_embedding = embedding_model.encode(
                [query_var], 
                convert_to_tensor=False,
                normalize_embeddings=True
            )[0].tolist()
            
            # Single Pinecone search with higher top_k to get good results in one go
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Collect results
            for match in search_results.matches:
                if match.metadata and match.score > 0.1:  # Early filtering
                    all_results.append({
                        'id': match.id,
                        'score': match.score,
                        'text': match.metadata.get('text', ''),
                        'doc_id': match.metadata.get('doc_id', ''),
                        'chunk_index': match.metadata.get('chunk_index', 0),
                        'query_variant': i
                    })
        
        # Quick deduplication and sorting
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids and len(unique_results) < 12:  # Limit results
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} matches")
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def filter_and_rank_contexts(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Streamlined context filtering for speed"""
    if not search_results:
        return []
    
    contexts = []
    query_words = set(query.lower().split())
    
    # Single-pass filtering with combined criteria
    for result in search_results[:10]:  # Only check top 10 for speed
        if result['score'] > 0.15:  # High confidence
            contexts.append(result['text'])
        elif result['score'] > 0.1:  # Medium confidence, check keywords
            text_words = set(result['text'].lower().split())
            overlap = len(query_words.intersection(text_words))
            if overlap >= 2:  # At least 2 word overlap
                contexts.append(result['text'])
        
        if len(contexts) >= 5:  # Stop early when we have enough
            break
    
    # Fallback if no good contexts found
    if not contexts and search_results:
        contexts = [r['text'] for r in search_results[:3]]
        logger.info("Using fallback contexts")
    
    logger.info(f"Selected {len(contexts)} contexts")
    return contexts

def generate_comprehensive_answer(question: str, contexts: List[str]) -> str:
    """Generate concise, focused answers quickly"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Use fewer contexts for speed and conciseness
        selected_contexts = contexts[:4]  # Max 4 contexts instead of 7
        context_text = "\n\n".join([f"Context {i+1}: {ctx[:500]}" for i, ctx in enumerate(selected_contexts)])
        
        logger.info(f"Generating answer with {len(selected_contexts)} contexts")
        
        # Optimized prompt for concise answers
        prompt = f"""Based on the provided contexts, give a direct and concise answer to the question.

CONTEXTS:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Provide a focused, direct answer (2-4 sentences maximum)
- Only include the most relevant information
- Be specific and factual
- If information is not available, state it clearly and briefly

CONCISE ANSWER:"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,  # Lower temperature for more focused responses
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 512,  # Reduced from 2048 to 512
            }
        )
        
        answer = response.text.strip()
        logger.info(f"Generated answer: {len(answer)} chars")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("=== APPLICATION STARTUP ===")
    logger.info(f"Environment variables:")
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

@app.post("/debug-search")
async def debug_search_endpoint(
    request: SearchDebugRequest,
    token: str = Depends(verify_token)
):
    """Enhanced debug endpoint for search testing"""
    try:
        # Get search results
        search_results = advanced_search_similar_chunks(request.query, request.top_k)
        
        # Get index stats
        index_stats = pc_index.describe_index_stats()
        
        # Process contexts
        contexts = filter_and_rank_contexts(search_results, request.query)
        
        return {
            "query": request.query,
            "query_variations": expand_query(preprocess_query(request.query)),
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension,
                "namespaces": index_stats.namespaces
            },
            "search_results": {
                "total_matches": len(search_results),
                "matches": [
                    {
                        "id": r['id'],
                        "score": r['score'],
                        "doc_id": r['doc_id'],
                        "text_preview": r['text'][:200] + "..." if len(r['text']) > 200 else r['text']
                    }
                    for r in search_results[:5]
                ]
            },
            "filtered_contexts": {
                "count": len(contexts),
                "contexts": [ctx[:300] + "..." if len(ctx) > 300 else ctx for ctx in contexts[:3]]
            }
        }
    except Exception as e:
        logger.error(f"Error in debug search: {e}")
        return {"error": str(e)}

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing queries - ENHANCED VERSION"""
    try:
        start_time = time.time()
        
        # Check index status first
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        
        logger.info(f"Processing {len(request.questions)} questions")
        logger.info(f"Index contains {total_vectors} vectors")
        
        if total_vectors == 0:
            return QueryResponse(
                answers=["The document index is empty. Please index some documents first."] * len(request.questions)
            )
        
        # Handle document URL if provided
        if request.document_url:
            logger.info(f"Document URL provided: {request.document_url}")
            # Add document processing logic here if needed
            # For now, we'll search the existing index
        
        # Process each question
        async def process_single_question(question: str) -> str:
            try:
                logger.info(f"Processing: {question}")
                
                # Optimized search with reduced complexity
                search_results = advanced_search_similar_chunks(question, top_k=12)
                
                if not search_results:
                    return f"No relevant information found for: '{question}'"
                
                # Streamlined context filtering
                contexts = filter_and_rank_contexts(search_results, question)
                
                if not contexts:
                    return f"Found potentially related information, but not relevant enough for: '{question}'"
                
                # Generate concise answer
                answer = generate_comprehensive_answer(question, contexts)
                return answer
                
            except Exception as e:
                logger.error(f"Error processing '{question}': {e}")
                return f"Error processing: {str(e)}"
        
        # Process all questions concurrently
        tasks = [process_single_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Clean up
        gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hackrx/run", response_model=QueryResponse)
async def process_query_api(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Alternative endpoint path"""
    return await process_query(request, token)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Railway Semantic Search API - Enhanced Version",
        "version": "2.0.0",
        "status": "active",
        "endpoints": [
            "GET /",
            "GET /health",
            "POST /debug-search",
            "POST /hackrx/run",
            "POST /api/hackrx/run"
        ]
    }

@app.get("/debug")
async def debug_info():
    """Enhanced debug endpoint"""
    try:
        index_stats = pc_index.describe_index_stats()
        
        # Test embedding
        test_embedding = embedding_model.encode(["test"], normalize_embeddings=True)[0]
        
        return {
            "app_status": "running",
            "version": "2.0.0",
            "models": {
                "embedding_loaded": embedding_model is not None,
                "embedding_dimension": len(test_embedding),
                "pinecone_connected": pc_index is not None
            },
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension,
                "namespaces": index_stats.namespaces
            },
            "config": {
                "API_BEARER_TOKEN": "SET" if API_BEARER_TOKEN else "MISSING",
                "GEMINI_API_KEY": "SET" if GEMINI_API_KEY else "MISSING",
                "PINECONE_API_KEY": "SET" if PINECONE_API_KEY else "MISSING"
            }
        }
    except Exception as e:
        return {"error": str(e), "app_status": "error"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
