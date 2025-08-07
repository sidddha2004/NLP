import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import time
import gc
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    version="2.2.0",
    debug=False,  # Disable debug mode for production
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
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for CPU-bound tasks

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
        # Set to use CPU for consistent performance
        embedding_model = embedding_model.to('cpu')
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

# Async wrapper for embedding generation
async def generate_embedding_async(text: str) -> List[float]:
    """Generate embedding asynchronously using thread pool"""
    loop = asyncio.get_event_loop()
    
    def _generate_embedding():
        return embedding_model.encode(
            [text], 
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False  # Disable progress bar for speed
        )[0].tolist()
    
    return await loop.run_in_executor(executor, _generate_embedding)

async def fast_search_similar_chunks(query: str, top_k: int = 12) -> List[Dict[str, Any]]:
    """Optimized search with better accuracy and minimal variations"""
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Preprocess query
        processed_query = preprocess_query(query)
        
        # Generate two query variants for better recall
        query_variants = [processed_query]
        
        # Add a simplified variant for better matching
        simple_query = re.sub(r'[^\w\s]', ' ', processed_query.lower())
        simple_query = re.sub(r'\s+', ' ', simple_query).strip()
        if simple_query != processed_query.lower() and len(processed_query.split()) > 2:
            query_variants.append(simple_query)
        
        all_results = []
        
        # Search with both variants but process efficiently
        for i, query_var in enumerate(query_variants):
            # Generate embedding asynchronously
            query_embedding = await generate_embedding_async(query_var)
            
            # Pinecone search with optimized parameters
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Process results quickly
            for match in search_results.matches:
                if match.metadata and match.score > 0.08:  # Lower threshold for better recall
                    all_results.append({
                        'id': match.id,
                        'score': match.score,
                        'text': match.metadata.get('text', ''),
                        'doc_id': match.metadata.get('doc_id', ''),
                        'chunk_index': match.metadata.get('chunk_index', 0),
                        'query_variant': i
                    })
        
        # Quick deduplication while preserving best scores
        seen_ids = {}
        for result in all_results:
            if result['id'] not in seen_ids or result['score'] > seen_ids[result['id']]['score']:
                seen_ids[result['id']] = result
        
        # Sort by score and return top results
        unique_results = sorted(seen_ids.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        
        logger.info(f"Found {len(unique_results)} unique matches")
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def smart_filter_contexts(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Enhanced context filtering with better accuracy"""
    if not search_results:
        return []
    
    contexts = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Score contexts based on multiple criteria
    scored_contexts = []
    
    for result in search_results[:8]:  # Check more results for better accuracy
        score = result['score']
        text = result['text']
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Calculate enhanced relevance score
        relevance_score = score
        
        # Boost score for exact phrase matches
        if query_lower in text_lower:
            relevance_score += 0.1
        
        # Boost score for keyword overlap
        word_overlap = len(query_words.intersection(text_words))
        if len(query_words) > 0:
            overlap_ratio = word_overlap / len(query_words)
            relevance_score += overlap_ratio * 0.05
        
        # Boost score for important keywords (domain-specific terms)
        important_keywords = ['railway', 'train', 'station', 'track', 'signal', 'safety', 'maintenance', 'schedule']
        for keyword in important_keywords:
            if keyword in text_lower and keyword in query_lower:
                relevance_score += 0.02
        
        scored_contexts.append({
            'text': text,
            'relevance_score': relevance_score,
            'original_score': score
        })
    
    # Sort by enhanced relevance score
    scored_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Select contexts with adaptive thresholds
    for ctx in scored_contexts:
        if len(contexts) >= 5:  # Limit to 5 contexts for balance
            break
            
        if ctx['relevance_score'] > 0.15:  # High relevance
            contexts.append(ctx['text'])
        elif ctx['relevance_score'] > 0.12 and len(contexts) < 3:  # Medium relevance, ensure minimum contexts
            contexts.append(ctx['text'])
    
    # Ensure we have at least one context if results exist
    if not contexts and scored_contexts:
        contexts = [scored_contexts[0]['text']]
        logger.info("Using fallback context with highest score")
    
    logger.info(f"Selected {len(contexts)} contexts with enhanced filtering")
    return contexts

async def generate_fast_answer(question: str, contexts: List[str]) -> str:
    """Generate answers with optimized Gemini settings"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Use only top 3 contexts and limit their length
        selected_contexts = contexts[:3]
        context_text = "\n\n".join([f"Context {i+1}: {ctx[:300]}" for i, ctx in enumerate(selected_contexts)])
        
        logger.info(f"Generating answer with {len(selected_contexts)} contexts")
        
        # Optimized prompt for speed and conciseness
        prompt = f"""Answer the question directly and concisely based on the provided contexts.

CONTEXTS:
{context_text}

QUESTION: {question}

Provide a direct answer in 2-3 sentences maximum. Be specific and factual.

ANSWER:"""

        # Run Gemini API call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate_content():
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'top_k': 20,
                    'max_output_tokens': 256,  # Further reduced for speed
                }
            )
            return response.text.strip()
        
        answer = await loop.run_in_executor(executor, _generate_content)
        
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
    """Debug endpoint for search testing"""
    try:
        # Get search results
        search_results = await fast_search_similar_chunks(request.query, request.top_k)
        
        # Get index stats
        index_stats = pc_index.describe_index_stats()
        
        # Process contexts
        contexts = smart_filter_contexts(search_results, request.query)
        
        return {
            "query": request.query,
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension
            },
            "search_results": {
                "total_matches": len(search_results),
                "matches": [
                    {
                        "id": r['id'],
                        "score": r['score'],
                        "doc_id": r['doc_id'],
                        "text_preview": r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
                    }
                    for r in search_results[:3]
                ]
            },
            "filtered_contexts": {
                "count": len(contexts),
                "contexts": [ctx[:200] + "..." if len(ctx) > 200 else ctx for ctx in contexts[:2]]
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
    """Main endpoint for processing queries - SPEED OPTIMIZED VERSION"""
    try:
        start_time = time.time()
        
        # Quick index check
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        
        logger.info(f"Processing {len(request.questions)} questions")
        logger.info(f"Index contains {total_vectors} vectors")
        
        if total_vectors == 0:
            return QueryResponse(
                answers=["The document index is empty. Please index some documents first."] * len(request.questions)
            )
        
        # Process each question with optimized pipeline
        async def process_single_question(question: str) -> str:
            try:
                question_start = time.time()
                logger.info(f"Processing: {question}")
                
                # Fast search with better coverage
                search_results = await fast_search_similar_chunks(question, top_k=8)
                
                if not search_results:
                    return f"No relevant information found for: '{question}'"
                
                # Enhanced context filtering
                contexts = smart_filter_contexts(search_results, question)
                
                if not contexts:
                    return f"Found potentially related information, but not relevant enough for: '{question}'"
                
                # Generate enhanced answer
                answer = await generate_enhanced_answer(question, contexts)
                
                question_time = time.time() - question_start
                logger.info(f"Question processed in {question_time:.2f}s")
                
                return answer
                
            except Exception as e:
                logger.error(f"Error processing '{question}': {e}")
                return f"Error processing: {str(e)}"
        
        # Process questions with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
        
        async def process_with_semaphore(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)
        
        # Process all questions
        tasks = [process_with_semaphore(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Quick cleanup
        gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"TOTAL: Processed {len(request.questions)} questions in {processing_time:.2f}s")
        
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
        "message": "Railway Semantic Search API - Speed + Accuracy Optimized",
        "version": "2.2.0",
        "status": "active",
        "optimizations": "Async processing, enhanced context filtering, better search coverage"
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint"""
    try:
        index_stats = pc_index.describe_index_stats()
        
        # Test embedding
        test_embedding = embedding_model.encode(["test"], normalize_embeddings=True)[0]
        
        return {
            "app_status": "running",
            "version": "2.2.0",
            "models": {
                "embedding_loaded": embedding_model is not None,
                "embedding_dimension": len(test_embedding),
                "pinecone_connected": pc_index is not None
            },
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension
            },
            "config": {
                "API_BEARER_TOKEN": "SET" if API_BEARER_TOKEN else "MISSING",
                "GEMINI_API_KEY": "SET" if GEMINI_API_KEY else "MISSING",
                "PINECONE_API_KEY": "SET" if PINECONE_API_KEY else "MISSING"
            }
        }
    except Exception as e:
        return {"error": str(e), "app_status": "error"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    executor.shutdown(wait=False)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
