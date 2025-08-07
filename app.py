import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import time
import gc
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from functools import lru_cache

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
    version="2.3.0",
    debug=False,
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
executor = ThreadPoolExecutor(max_workers=8)  # Increased for better concurrency

# Caching for better performance
search_cache = {}
auth_cache = {}
CACHE_TTL = 300  # 5 minutes
MAX_CACHE_SIZE = 1000

# Updated request model to match your format
class QueryRequest(BaseModel):
    documents: Optional[str] = None  # Ignored in Phase 2
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class SearchDebugRequest(BaseModel):
    query: str
    top_k: Optional[int] = 15

# Authentication middleware with caching
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    current_time = time.time()
    
    # Check cache first
    if token in auth_cache:
        if current_time - auth_cache[token] < CACHE_TTL:
            return token
    
    if not API_BEARER_TOKEN or token != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Cache successful auth
    auth_cache[token] = current_time
    return token

def initialize_models():
    """Initialize models and connections - UNCHANGED to maintain compatibility"""
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
        
        # Warm up the model
        _ = embedding_model.encode(["warmup"], normalize_embeddings=True)
        logger.info("Embedding model loaded and warmed up successfully")
        
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pinecone_client.Index("first")
        logger.info("Pinecone connection established")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def preprocess_query(query: str) -> str:
    """Preprocess query to match document preprocessing - UNCHANGED to maintain compatibility"""
    # Clean the query similar to how documents were processed
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    return query

def get_cache_key(query: str, top_k: int) -> str:
    """Generate cache key for search results"""
    return hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()

def clean_cache():
    """Clean old cache entries"""
    current_time = time.time()
    global search_cache, auth_cache
    
    # Clean search cache
    if len(search_cache) > MAX_CACHE_SIZE:
        # Keep only recent entries
        search_cache = {k: v for k, v in list(search_cache.items())[-MAX_CACHE_SIZE//2:]}
    
    # Clean auth cache
    auth_cache = {k: v for k, v in auth_cache.items() if current_time - v < CACHE_TTL}

# Async wrapper for embedding generation - UNCHANGED to maintain compatibility
async def generate_embedding_async(text: str) -> List[float]:
    """Generate embedding asynchronously using thread pool - UNCHANGED to maintain compatibility"""
    loop = asyncio.get_event_loop()
    
    def _generate_embedding():
        return embedding_model.encode(
            [text], 
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].tolist()
    
    return await loop.run_in_executor(executor, _generate_embedding)

async def ultra_enhanced_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Ultra-enhanced multi-stage search with caching"""
    try:
        # Check cache first
        cache_key = get_cache_key(query, top_k)
        if cache_key in search_cache:
            logger.info(f"Cache hit for query: '{query[:50]}...'")
            return search_cache[cache_key]
        
        logger.info(f"Multi-stage search for: '{query}'")
        
        # Stage 1: Broader retrieval with higher top_k
        processed_query = preprocess_query(query)
        query_embedding = await generate_embedding_async(processed_query)
        
        # Initial search with higher top_k for better coverage
        search_results = pc_index.query(
            vector=query_embedding,
            top_k=min(top_k * 2, 40),  # Get more results initially
            include_metadata=True,
            include_values=False
        )
        
        # Stage 2: Enhanced scoring and filtering
        results = []
        query_words = set(query.lower().split())
        query_bigrams = set([f"{query.lower().split()[i]} {query.lower().split()[i+1]}" 
                            for i in range(len(query.lower().split())-1)])
        
        for match in search_results.matches:
            if match.metadata and match.score > 0.05:  # Lower threshold for broader search
                text = match.metadata.get('text', '')
                text_lower = text.lower()
                text_words = set(text_lower.split())
                
                # Multi-factor scoring
                keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
                
                # Bigram matching for phrase detection
                bigram_score = 0
                if query_bigrams:
                    text_bigrams = set([f"{text_lower.split()[i]} {text_lower.split()[i+1]}" 
                                       for i in range(len(text_lower.split())-1)])
                    bigram_matches = query_bigrams.intersection(text_bigrams)
                    bigram_score = len(bigram_matches) / len(query_bigrams)
                
                # Exact phrase bonus
                phrase_bonus = 0.1 if query.lower() in text_lower else 0
                
                # Position-based scoring (earlier mentions get slight boost)
                position_score = 0
                first_match_pos = text_lower.find(query.lower().split()[0]) if query.lower().split() else -1
                if first_match_pos != -1:
                    position_score = max(0, (500 - first_match_pos) / 5000)  # Normalize to 0-0.1
                
                # Combined enhanced score
                enhanced_score = (match.score + 
                                (keyword_overlap * 0.15) + 
                                (bigram_score * 0.2) + 
                                phrase_bonus + 
                                position_score)
                
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'enhanced_score': enhanced_score,
                    'text': text,
                    'doc_id': match.metadata.get('doc_id', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'keyword_overlap': keyword_overlap,
                    'bigram_score': bigram_score,
                    'phrase_bonus': phrase_bonus > 0
                })
        
        # Stage 3: Smart deduplication and final selection
        results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        # Remove very similar texts (deduplication)
        final_results = []
        seen_texts = set()
        
        for result in results:
            text_hash = hashlib.md5(result['text'][:200].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                final_results.append(result)
                
                if len(final_results) >= top_k:
                    break
        
        # Cache results
        search_cache[cache_key] = final_results
        
        logger.info(f"Multi-stage search completed: {len(final_results)} high-quality matches")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in ultra-enhanced search: {e}")
        return []

def ultra_smart_context_selection(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Ultra-smart context selection with advanced filtering"""
    if not search_results:
        return []
    
    contexts = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    # Enhanced selection with multiple criteria
    for i, result in enumerate(search_results):
        text = result['text']
        enhanced_score = result['enhanced_score']
        keyword_overlap = result.get('keyword_overlap', 0)
        bigram_score = result.get('bigram_score', 0)
        has_phrase = result.get('phrase_bonus', False)
        
        # Multi-tier selection with refined thresholds
        select_context = False
        
        if enhanced_score > 0.4 and keyword_overlap > 0.4:  # Tier 1: Excellent match
            select_context = True
        elif enhanced_score > 0.3 and (keyword_overlap > 0.3 or bigram_score > 0.3):  # Tier 2: Very good
            select_context = True
        elif enhanced_score > 0.25 and has_phrase:  # Tier 3: Has exact phrase
            select_context = True
        elif enhanced_score > 0.2 and keyword_overlap > 0.2 and i < 8:  # Tier 4: Good with position bonus
            # Additional relevance check
            important_terms = ['policy', 'coverage', 'premium', 'benefit', 'waiting', 'period', 
                             'mediclaim', 'insurance', 'claim', 'discount', 'hospital', 'medical',
                             'treatment', 'cashless', 'reimbursement', 'exclusion', 'condition']
            if any(term in text.lower() for term in important_terms):
                select_context = True
        
        if select_context:
            contexts.append(text)
        
        # Dynamic limit based on query complexity
        max_contexts = 7 if len(query.split()) > 10 else 6
        if len(contexts) >= max_contexts:
            break
    
    # Ensure we have at least one context
    if not contexts and search_results:
        contexts = [search_results[0]['text']]
        logger.info("Using fallback context with best score")
    
    logger.info(f"Ultra-smart selection: {len(contexts)} contexts from {len(search_results)} candidates")
    return contexts

async def ultra_optimized_answer_generation(question: str, contexts: List[str]) -> str:
    """Ultra-optimized answer generation with improved prompting"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Optimized context processing
        selected_contexts = contexts[:5]  # Increased from 4
        
        # Smart truncation with better preservation
        processed_contexts = []
        for i, ctx in enumerate(selected_contexts):
            if len(ctx) > 450:  # Slightly increased limit
                # Prioritize sentences with query terms
                sentences = ctx.split('. ')
                question_words = set(question.lower().split())
                
                scored_sentences = []
                for sentence in sentences:
                    sentence_words = set(sentence.lower().split())
                    overlap = len(question_words.intersection(sentence_words))
                    scored_sentences.append((sentence, overlap))
                
                # Sort by relevance and take top sentences
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                top_sentences = [s[0] for s in scored_sentences[:3]]
                
                if top_sentences and scored_sentences[0][1] > 0:
                    ctx = '. '.join(top_sentences)
                else:
                    # Fallback: take beginning which often contains key info
                    ctx = ctx[:450]
            
            processed_contexts.append(f"Context {i+1}: {ctx}")
        
        context_text = "\n\n".join(processed_contexts)
        
        logger.info(f"Generating answer with {len(selected_contexts)} ultra-optimized contexts")
        
        # Ultra-refined prompt for maximum accuracy and conciseness
        prompt = f"""You are a precise insurance policy expert. Answer the question with exact details from the provided contexts.

CONTEXTS:
{context_text}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
- Give a direct, factual answer using ONLY information from the contexts
- Include specific numbers, percentages, time periods, and conditions mentioned
- If exact information isn't in contexts, state "The provided information doesn't specify..."
- Keep answer concise but complete (2-3 sentences maximum)
- Be precise with terms like "up to", "minimum", "maximum", "after", "before"
- Focus ONLY on what the question asks - avoid extra details

ANSWER:"""

        # Optimized Gemini call with better config
        loop = asyncio.get_event_loop()
        
        def _generate_content():
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.02,  # Very low for maximum consistency
                    'top_p': 0.6,        # More focused
                    'top_k': 10,         # More focused
                    'max_output_tokens': 250,  # Optimized for concise answers
                }
            )
            return response.text.strip()
        
        answer = await loop.run_in_executor(executor, _generate_content)
        
        # Post-process for consistency and conciseness
        answer = re.sub(r'\n+', ' ', answer)  # Remove newlines
        answer = re.sub(r'\s+', ' ', answer)  # Normalize spaces
        answer = answer.replace('**', '').replace('*', '')  # Remove markdown
        
        logger.info(f"Ultra-optimized answer generated: {len(answer)} chars")
        return answer
        
    except Exception as e:
        logger.error(f"Error in ultra-optimized generation: {e}")
        return f"Error generating answer: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("=== ENHANCED APPLICATION STARTUP ===")
    logger.info(f"Environment variables:")
    logger.info(f"  API_BEARER_TOKEN: {'SET' if API_BEARER_TOKEN else 'MISSING'}")
    logger.info(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'MISSING'}")
    logger.info(f"  PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'MISSING'}")
    
    try:
        initialize_models()
        logger.info("=== ENHANCED STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"=== STARTUP FAILED: {e} ===")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.3.0", "timestamp": time.time()}

@app.get("/cache-stats")
async def cache_stats(token: str = Depends(verify_token)):
    """Cache statistics endpoint"""
    return {
        "search_cache_size": len(search_cache),
        "auth_cache_size": len(auth_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_ttl": CACHE_TTL
    }

@app.post("/clear-cache")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear cache endpoint"""
    global search_cache, auth_cache
    search_cache.clear()
    auth_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.post("/debug-search")
async def debug_search_endpoint(
    request: SearchDebugRequest,
    token: str = Depends(verify_token)
):
    """Enhanced debug endpoint"""
    try:
        start_time = time.time()
        
        # Get enhanced search results
        search_results = await ultra_enhanced_search(request.query, request.top_k)
        
        # Get index stats
        index_stats = pc_index.describe_index_stats()
        
        # Process contexts
        contexts = ultra_smart_context_selection(search_results, request.query)
        
        search_time = time.time() - start_time
        
        return {
            "query": request.query,
            "performance": {
                "search_time_ms": round(search_time * 1000, 2),
                "cache_hit": get_cache_key(request.query, request.top_k) in search_cache
            },
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
                        "enhanced_score": r['enhanced_score'],
                        "keyword_overlap": r['keyword_overlap'],
                        "bigram_score": r.get('bigram_score', 0),
                        "has_phrase": r.get('phrase_bonus', False),
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
        logger.error(f"Error in enhanced debug search: {e}")
        return {"error": str(e)}

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """ULTRA-ENHANCED main endpoint for processing queries"""
    try:
        start_time = time.time()
        
        # Log document field (ignored)
        if request.documents:
            logger.info(f"Document field received (ignored): {request.documents[:100]}...")
        
        # Quick index check
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        
        logger.info(f"Processing {len(request.questions)} questions with ultra-enhancements")
        logger.info(f"Index contains {total_vectors} vectors")
        
        if total_vectors == 0:
            return QueryResponse(
                answers=["The document index is empty. Please index some documents first."] * len(request.questions)
            )
        
        # Ultra-enhanced question processing
        async def process_single_question(question: str) -> str:
            try:
                question_start = time.time()
                logger.info(f"Ultra-processing: {question}")
                
                # Ultra-enhanced search
                search_results = await ultra_enhanced_search(question, top_k=15)
                
                if not search_results:
                    return f"No relevant information found for: '{question}'"
                
                # Ultra-smart context selection
                contexts = ultra_smart_context_selection(search_results, question)
                
                if not contexts:
                    return f"Found related information, but not specific enough for: '{question}'"
                
                # Ultra-optimized answer generation
                answer = await ultra_optimized_answer_generation(question, contexts)
                
                question_time = time.time() - question_start
                logger.info(f"Ultra-processed in {question_time:.2f}s")
                
                return answer
                
            except Exception as e:
                logger.error(f"Error in ultra-processing '{question}': {e}")
                return f"Error processing: {str(e)}"
        
        # Process with increased concurrency
        semaphore = asyncio.Semaphore(6)  # Increased concurrency
        
        async def process_with_semaphore(question: str) -> str:
            async with semaphore:
                return await process_single_question(question)
        
        # Process all questions
        tasks = [process_with_semaphore(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Cleanup cache if needed
        if len(search_cache) > MAX_CACHE_SIZE * 0.8:
            clean_cache()
        
        # Quick garbage collection
        gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"ULTRA-ENHANCED TOTAL: {len(request.questions)} questions in {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in ultra-enhanced processing: {e}")
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
        "message": "Railway Semantic Search API - Ultra Enhanced Version",
        "version": "2.3.0",
        "status": "active",
        "enhancements": [
            "Multi-stage search pipeline",
            "Advanced caching system",
            "Enhanced scoring algorithms",
            "Ultra-smart context selection",
            "Optimized answer generation",
            "Improved concurrency"
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
            "app_status": "ultra-enhanced",
            "version": "2.3.0",
            "performance": {
                "search_cache_size": len(search_cache),
                "auth_cache_size": len(auth_cache),
                "max_workers": executor._max_workers
            },
            "models": {
                "embedding_loaded": embedding_model is not None,
                "embedding_dimension": len(test_embedding),
                "pinecone_connected": pc_index is not None,
                "embedding_model": "paraphrase-MiniLM-L6-v2"
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
    """Enhanced cleanup on shutdown"""
    executor.shutdown(wait=False)
    search_cache.clear()
    auth_cache.clear()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
