'''
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
executor = ThreadPoolExecutor(max_workers=6)  # Increased for better concurrency

# Updated request model to match your format
class QueryRequest(BaseModel):
    documents: Optional[str] = None  # Ignored in Phase 2
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
    """Preprocess query to match document preprocessing - UNCHANGED to maintain compatibility"""
    # Clean the query similar to how documents were processed
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    return query

# Async wrapper for embedding generation
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

async def enhanced_search_similar_chunks(query: str, top_k: int = 12) -> List[Dict[str, Any]]:
    """Enhanced search with better result processing"""
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Preprocess query (unchanged to maintain compatibility)
        processed_query = preprocess_query(query)
        
        # Generate embedding asynchronously (unchanged)
        query_embedding = await generate_embedding_async(processed_query)
        
        # Search with slightly higher top_k for better selection
        search_results = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        # Enhanced result processing
        results = []
        query_words = set(query.lower().split())
        
        for match in search_results.matches:
            if match.metadata and match.score > 0.08:  # Slightly lower threshold
                text = match.metadata.get('text', '')
                
                # Calculate keyword overlap bonus
                text_words = set(text.lower().split())
                keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
                
                # Enhanced scoring
                enhanced_score = match.score + (keyword_overlap * 0.1)
                
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'enhanced_score': enhanced_score,
                    'text': text,
                    'doc_id': match.metadata.get('doc_id', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'keyword_overlap': keyword_overlap
                })
        
        # Sort by enhanced score
        results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        logger.info(f"Found {len(results)} matches")
        return results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def smart_context_selection(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Smarter context selection with improved filtering"""
    if not search_results:
        return []
    
    contexts = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    # Enhanced context selection logic
    for result in search_results:
        text = result['text']
        score = result['enhanced_score']
        keyword_overlap = result.get('keyword_overlap', 0)
        
        # Multi-tier selection criteria
        if score > 0.3 and keyword_overlap > 0.3:  # High confidence
            contexts.append(text)
        elif score > 0.2 and keyword_overlap > 0.2:  # Medium confidence
            contexts.append(text)
        elif score > 0.15 and keyword_overlap > 0.1:  # Lower confidence but some relevance
            # Additional check for important terms
            important_terms = ['policy', 'coverage', 'premium', 'benefit', 'waiting', 'period', 
                             'mediclaim', 'insurance', 'claim', 'discount', 'hospital']
            if any(term in text.lower() for term in important_terms):
                contexts.append(text)
        
        # Limit contexts but allow more for complex queries
        max_contexts = 6 if len(query.split()) > 8 else 5
        if len(contexts) >= max_contexts:
            break
    
    # Fallback with best result
    if not contexts and search_results:
        contexts = [search_results[0]['text']]
        logger.info("Using fallback context")
    
    logger.info(f"Selected {len(contexts)} contexts with improved filtering")
    return contexts

async def generate_enhanced_answer(question: str, contexts: List[str]) -> str:
    """Enhanced answer generation with better prompting"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Use more contexts but truncate smartly
        selected_contexts = contexts[:4]  # Increased from 3
        
        # Smart truncation - keep important parts
        processed_contexts = []
        for i, ctx in enumerate(selected_contexts):
            if len(ctx) > 400:  # Increased from 300
                # Try to keep the most relevant part
                sentences = ctx.split('. ')
                relevant_sentences = []
                question_words = set(question.lower().split())
                
                for sentence in sentences:
                    sentence_words = set(sentence.lower().split())
                    if question_words.intersection(sentence_words):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    ctx = '. '.join(relevant_sentences[:2])  # Keep top 2 relevant sentences
                else:
                    ctx = ctx[:400]  # Fallback to truncation
            
            processed_contexts.append(f"Context {i+1}: {ctx}")
        
        context_text = "\n\n".join(processed_contexts)
        
        logger.info(f"Generating answer with {len(selected_contexts)} enhanced contexts")
        
        # Improved prompt for better accuracy
        prompt = f"""You are an expert insurance policy analyst. Answer the question accurately and concisely based on the provided contexts from insurance policy documents.

CONTEXTS:
{context_text}

QUESTION: {question}

Instructions:
- Provide a direct, factual answer based on the contexts
- Be specific about numbers, periods, percentages, and conditions
- If the information is not in the contexts, state that clearly
- Keep the answer concise but complete (2-4 sentences)
- Focus on the exact details requested in the question

ANSWER:"""

        # Run Gemini API call in thread pool
        loop = asyncio.get_event_loop()
        
        def _generate_content():
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.05,  # Lower for more consistent answers
                    'top_p': 0.7,        # More focused
                    'top_k': 15,         # More focused
                    'max_output_tokens': 300,  # Slightly increased for complete answers
                }
            )
            return response.text.strip()
        
        answer = await loop.run_in_executor(executor, _generate_content)
        
        # Post-process answer for consistency
        answer = re.sub(r'\n+', ' ', answer)  # Remove multiple newlines
        answer = re.sub(r'\s+', ' ', answer)  # Normalize spaces
        
        logger.info(f"Generated enhanced answer: {len(answer)} chars")
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
        search_results = await enhanced_search_similar_chunks(request.query, request.top_k)
        
        # Get index stats
        index_stats = pc_index.describe_index_stats()
        
        # Process contexts
        contexts = smart_context_selection(search_results, request.query)
        
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
                        "enhanced_score": r['enhanced_score'],
                        "keyword_overlap": r['keyword_overlap'],
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
    """Main endpoint for processing queries - ENHANCED VERSION"""
    try:
        start_time = time.time()
        
        # Log the document field (ignored as mentioned)
        if request.documents:
            logger.info(f"Document field received (ignored): {request.documents[:100]}...")
        
        # Quick index check
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        
        logger.info(f"Processing {len(request.questions)} questions")
        logger.info(f"Index contains {total_vectors} vectors")
        
        if total_vectors == 0:
            return QueryResponse(
                answers=["The document index is empty. Please index some documents first."] * len(request.questions)
            )
        
        # Enhanced question processing
        async def process_single_question(question: str) -> str:
            try:
                question_start = time.time()
                logger.info(f"Processing: {question}")
                
                # Enhanced search with better result selection
                search_results = await enhanced_search_similar_chunks(question, top_k=10)
                
                if not search_results:
                    return f"No relevant information found for: '{question}'"
                
                # Smart context selection
                contexts = smart_context_selection(search_results, question)
                
                if not contexts:
                    return f"Found potentially related information, but not specific enough for: '{question}'"
                
                # Generate enhanced answer
                answer = await generate_enhanced_answer(question, contexts)
                
                question_time = time.time() - question_start
                logger.info(f"Question processed in {question_time:.2f}s")
                
                return answer
                
            except Exception as e:
                logger.error(f"Error processing '{question}': {e}")
                return f"Error processing: {str(e)}"
        
        # Process questions with optimized concurrency
        semaphore = asyncio.Semaphore(4)  # Slightly increased concurrency
        
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
        "message": "Railway Semantic Search API - Enhanced Version",
        "version": "2.2.0",
        "status": "active",
        "enhancements": "Improved accuracy, smart context selection, enhanced scoring"
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
'''
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import gc
import re
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== APPLICATION STARTUP - ULTRA OPTIMIZED VERSION 4.0 ===")
    logger.info(f"Performance Configuration:")
    logger.info(f"  MAX_CONCURRENT_SEARCHES: {MAX_CONCURRENT_SEARCHES}")
    logger.info(f"  MAX_CONCURRENT_GENERATIONS: {MAX_CONCURRENT_GENERATIONS}")
    logger.info(f"  CACHE_SIZE: {CACHE_SIZE}")
    logger.info(f"  PINECONE_TOP_K: {PINECONE_TOP_K}")
    logger.info(f"Environment variables:")
    logger.info(f"  API_BEARER_TOKEN: {'SET' if API_BEARER_TOKEN else 'MISSING'}")
    logger.info(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'MISSING'}")
    logger.info(f"  PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'MISSING'}")
    
    try:
        initialize_models()
        logger.info("=== ULTRA OPTIMIZED STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"=== STARTUP FAILED: {e} ===")
        raise
    
    yield
    
    # Shutdown
    global executor
    if executor:
        executor.shutdown(wait=True)
    logger.info("Enhanced application shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Railway Semantic Search API - Ultra Optimized", 
    version="4.0.0",
    debug=False,  # Disable debug for production performance
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")

# Enhanced performance configuration
MAX_CONCURRENT_SEARCHES = 8  # Increased for better parallelism
MAX_CONCURRENT_GENERATIONS = 5  # Increased concurrent generations
CACHE_SIZE = 2000  # Increased cache size
EMBEDDING_BATCH_SIZE = 32  # Increased batch size
PINECONE_TOP_K = 25  # Increased records from Pinecone
ENHANCED_SEARCH_DEPTH = 30  # For multi-stage searching

# Global variables for models and caching
embedding_model = None
pinecone_client = None
pc_index = None
search_cache = {}
embedding_cache = {}
query_enhancement_cache = {}
executor = None

# Create lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== APPLICATION STARTUP - ULTRA OPTIMIZED VERSION 4.0 ===")
    logger.info(f"Performance Configuration:")
    logger.info(f"  MAX_CONCURRENT_SEARCHES: {MAX_CONCURRENT_SEARCHES}")
    logger.info(f"  MAX_CONCURRENT_GENERATIONS: {MAX_CONCURRENT_GENERATIONS}")
    logger.info(f"  CACHE_SIZE: {CACHE_SIZE}")
    logger.info(f"  PINECONE_TOP_K: {PINECONE_TOP_K}")
    logger.info(f"Environment variables:")
    logger.info(f"  API_BEARER_TOKEN: {'SET' if API_BEARER_TOKEN else 'MISSING'}")
    logger.info(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'MISSING'}")
    logger.info(f"  PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'MISSING'}")
    
    try:
        initialize_models()
        logger.info("=== ULTRA OPTIMIZED STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"=== STARTUP FAILED: {e} ===")
        raise
    
    yield
    
    # Shutdown
    global executor
    if executor:
        executor.shutdown(wait=True)
    logger.info("Enhanced application shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Railway Semantic Search API - Ultra Optimized", 
    version="4.0.0",
    debug=False,  # Disable debug for production performance
    docs_url="/docs",
    redoc_url="/redoc",
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

class QueryRequest(BaseModel):
    document_url: Optional[HttpUrl] = None
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class SearchDebugRequest(BaseModel):
    query: str
    top_k: Optional[int] = PINECONE_TOP_K

# Authentication middleware with enhanced caching
_auth_cache = {}

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    # Enhanced cache with TTL
    current_time = time.time()
    if token in _auth_cache:
        cache_entry = _auth_cache[token]
        if cache_entry['valid'] and (current_time - cache_entry['timestamp']) < 3600:  # 1 hour TTL
            return token
        else:
            _auth_cache.pop(token, None)  # Remove expired entry
    
    # Verify and cache result
    is_valid = API_BEARER_TOKEN and token == API_BEARER_TOKEN
    _auth_cache[token] = {'valid': is_valid, 'timestamp': current_time}
    
    # Clean old cache entries periodically
    if len(_auth_cache) > 200:  # Increased cache size
        cutoff_time = current_time - 3600
        _auth_cache = {k: v for k, v in _auth_cache.items() if v['timestamp'] > cutoff_time}
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def initialize_models():
    """Initialize models and connections with enhanced optimizations"""
    global embedding_model, pinecone_client, pc_index, executor
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize embedding model with enhanced device optimization
        logger.info("Loading paraphrase-MiniLM-L6-v2 model with enhanced optimizations...")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        
        # Optimize model for inference with warmup
        embedding_model.eval()
        # Warmup the model with a sample embedding
        warmup_text = ["sample warmup text for model initialization"]
        _ = embedding_model.encode(warmup_text, normalize_embeddings=True, show_progress_bar=False)
        logger.info("Model warmed up successfully")
        
        if hasattr(embedding_model, '_target_device'):
            logger.info(f"Model device: {embedding_model._target_device}")
        
        logger.info("Embedding model loaded and optimized successfully")
        
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pinecone_client.Index("first")
        logger.info("Pinecone connection established")
        
        # Initialize enhanced thread pool
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SEARCHES)
        logger.info(f"Enhanced thread pool initialized with {MAX_CONCURRENT_SEARCHES} workers")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

@lru_cache(maxsize=CACHE_SIZE)
def preprocess_query_cached(query: str) -> str:
    """Enhanced cached query preprocessing with better normalization"""
    # Remove extra whitespace and normalize
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    
    # Handle common abbreviations and improve query quality
    query = re.sub(r'\b(qty|quantity)\b', 'quantity', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(info|information)\b', 'information', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(req|requirement)\b', 'requirement', query, flags=re.IGNORECASE)
    
    return query

def get_query_hash(query: str) -> str:
    """Generate hash for query caching with better collision resistance"""
    return hashlib.sha256(query.encode()).hexdigest()[:16]  # Use SHA256 for better distribution

def get_cached_embedding(query: str) -> Optional[List[float]]:
    """Get cached embedding with hit rate tracking"""
    query_hash = get_query_hash(query)
    return embedding_cache.get(query_hash)

def cache_embedding(query: str, embedding: List[float]):
    """Enhanced cache embedding with LRU-style management"""
    query_hash = get_query_hash(query)
    
    # Manage cache size with LRU-like behavior
    if len(embedding_cache) >= CACHE_SIZE:
        # Remove oldest 25% of entries
        old_keys = list(embedding_cache.keys())[:CACHE_SIZE // 4]
        for key in old_keys:
            embedding_cache.pop(key, None)
    
    embedding_cache[query_hash] = embedding

def enhanced_query_expansion(query: str) -> List[str]:
    """Enhanced query expansion with semantic variations and context-aware improvements"""
    cache_key = get_query_hash(f"expand_{query}")
    if cache_key in query_enhancement_cache:
        return query_enhancement_cache[cache_key]
    
    variations = [query]
    
    # Add simplified version
    simple_query = re.sub(r'[^\w\s]', ' ', query.lower())
    simple_query = re.sub(r'\s+', ' ', simple_query).strip()
    if simple_query != query.lower() and len(query.split()) > 1:
        variations.append(simple_query)
    
    # Add keyword-focused version
    keywords = [word for word in query.split() if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'how', 'why', 'which', 'that', 'this', 'with', 'from']]
    if len(keywords) >= 2:
        keyword_query = ' '.join(keywords)
        if keyword_query not in variations:
            variations.append(keyword_query)
    
    # Add question-to-statement conversion for better semantic matching
    if query.lower().startswith(('what is', 'what are', 'how is', 'how are')):
        statement_version = query.lower().replace('what is', '').replace('what are', '').replace('how is', '').replace('how are', '').strip()
        if statement_version and len(statement_version) > 5:
            variations.append(statement_version)
    
    # Cache the result
    query_enhancement_cache[cache_key] = variations
    return variations

def batch_encode_queries_enhanced(queries: List[str]) -> List[List[float]]:
    """Enhanced batch encoding with better performance optimization"""
    # Check cache first
    cached_embeddings = {}
    queries_to_encode = []
    query_indices = []
    
    for i, query in enumerate(queries):
        cached = get_cached_embedding(query)
        if cached is not None:
            cached_embeddings[i] = cached
        else:
            queries_to_encode.append(query)
            query_indices.append(i)
    
    # Batch encode uncached queries with optimized settings
    if queries_to_encode:
        logger.info(f"Encoding {len(queries_to_encode)} queries in optimized batch")
        
        # Use optimized encoding parameters for speed
        new_embeddings = embedding_model.encode(
            queries_to_encode,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False
        )
        
        # Cache new embeddings
        for i, embedding in enumerate(new_embeddings):
            query_idx = query_indices[i]
            embedding_list = embedding.tolist()
            cache_embedding(queries[query_idx], embedding_list)
            cached_embeddings[query_idx] = embedding_list
    
    # Return embeddings in original order
    return [cached_embeddings[i] for i in range(len(queries))]

async def enhanced_parallel_search(queries: List[str], top_k: int = PINECONE_TOP_K) -> List[Dict[str, Any]]:
    """Enhanced parallel search with multi-stage retrieval and improved accuracy"""
    try:
        logger.info(f"Enhanced parallel search for {len(queries)} query variations with top_k={top_k}")
        
        # Batch encode all queries
        query_embeddings = batch_encode_queries_enhanced(queries)
        
        # Multi-stage search: First stage with higher top_k for broader retrieval
        stage1_top_k = min(top_k * 2, ENHANCED_SEARCH_DEPTH)  # Get more candidates initially
        
        # Prepare search tasks with enhanced caching
        search_tasks = []
        cached_results = []
        
        for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
            cache_key = f"{get_query_hash(query)}_{stage1_top_k}"
            if cache_key in search_cache:
                cached_results.extend(search_cache[cache_key])
                continue
            
            search_tasks.append({
                'query': query,
                'embedding': embedding,
                'cache_key': cache_key,
                'index': i
            })
        
        # Execute searches in parallel with enhanced concurrency
        all_results = list(cached_results)
        
        if search_tasks:
            async def search_single_enhanced(task):
                try:
                    # Stage 1: Broader search
                    search_results = pc_index.query(
                        vector=task['embedding'],
                        top_k=stage1_top_k,
                        include_metadata=True,
                        include_values=False
                    )
                    
                    results = []
                    for match in search_results.matches:
                        if match.metadata and match.score > 0.05:  # Lower threshold for initial retrieval
                            results.append({
                                'id': match.id,
                                'score': match.score,
                                'text': match.metadata.get('text', ''),
                                'doc_id': match.metadata.get('doc_id', ''),
                                'chunk_index': match.metadata.get('chunk_index', 0),
                                'query_variant': task['index'],
                                'original_query': task['query']
                            })
                    
                    # Cache results
                    search_cache[task['cache_key']] = results
                    return results
                    
                except Exception as e:
                    logger.error(f"Enhanced search error for query {task['index']}: {e}")
                    return []
            
            # Execute searches with increased concurrency
            search_coroutines = [search_single_enhanced(task) for task in search_tasks]
            search_results_list = await asyncio.gather(*search_coroutines, return_exceptions=True)
            
            # Collect all results
            for results in search_results_list:
                if isinstance(results, list):
                    all_results.extend(results)
        
        # Enhanced deduplication with semantic clustering
        return enhanced_result_deduplication(all_results, top_k)
        
    except Exception as e:
        logger.error(f"Error in enhanced parallel search: {e}")
        return []

def enhanced_result_deduplication(results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Enhanced result deduplication with semantic similarity and score-based ranking"""
    if not results:
        return []
    
    # Sort by score first
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Enhanced deduplication with text similarity check
    unique_results = []
    seen_ids = set()
    text_fingerprints = set()
    
    def text_fingerprint(text: str) -> str:
        # Create a more sophisticated fingerprint for duplicate detection
        words = text.lower().split()
        # Use first and last few words as fingerprint
        if len(words) > 10:
            fingerprint_words = words[:5] + words[-3:]
        else:
            fingerprint_words = words
        return ' '.join(sorted(fingerprint_words))
    
    for result in results:
        if result['id'] in seen_ids:
            continue
            
        text_fp = text_fingerprint(result['text'])
        if text_fp in text_fingerprints and len(result['text']) < 200:  # Skip short duplicates
            continue
            
        seen_ids.add(result['id'])
        text_fingerprints.add(text_fp)
        unique_results.append(result)
        
        if len(unique_results) >= target_count:
            break
    
    logger.info(f"Deduplicated to {len(unique_results)} unique results from {len(results)} total")
    return unique_results

def enhanced_context_filtering(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Enhanced context filtering with advanced semantic scoring and relevance ranking"""
    if not search_results:
        return []
    
    query_words = set(query.lower().split())
    query_keywords = [word for word in query_words if len(word) > 3]
    
    # Multi-tier filtering with enhanced scoring
    scored_contexts = []
    
    for result in search_results:
        text = result['text']
        score = result['score']
        text_words = set(text.lower().split())
        text_lower = text.lower()
        
        # Enhanced scoring algorithm
        base_score = score
        
        # Keyword matching bonus
        keyword_matches = sum(1 for kw in query_keywords if kw in text_lower)
        keyword_bonus = keyword_matches * 0.1
        
        # Exact phrase matching bonus
        phrase_bonus = 0.2 if query.lower() in text_lower else 0
        
        # Length penalty for very short or very long texts
        text_len = len(text.split())
        if text_len < 10:
            length_penalty = -0.1
        elif text_len > 200:
            length_penalty = -0.05
        else:
            length_penalty = 0
        
        # Question-answer format bonus
        qa_bonus = 0.1 if any(marker in text_lower for marker in ['answer:', 'solution:', 'result:', 'definition:']) else 0
        
        # Calculate final score
        final_score = base_score + keyword_bonus + phrase_bonus + length_penalty + qa_bonus
        
        scored_contexts.append({
            'text': text,
            'score': final_score,
            'original_score': score,
            'length': len(text)
        })
    
    # Sort by enhanced score and select top contexts
    scored_contexts.sort(key=lambda x: x['score'], reverse=True)
    
    # Select contexts with diversity in mind
    selected_contexts = []
    used_fingerprints = set()
    
    for ctx in scored_contexts:
        # Simple diversity check
        text_start = ctx['text'][:100].lower()
        if text_start not in used_fingerprints:
            selected_contexts.append(ctx['text'])
            used_fingerprints.add(text_start)
            
            if len(selected_contexts) >= 5:  # Increased context count for better accuracy
                break
    
    # Fallback if no good matches
    if not selected_contexts and search_results:
        selected_contexts = [r['text'] for r in search_results[:3]]
    
    logger.info(f"Selected {len(selected_contexts)} enhanced contexts from {len(search_results)} results")
    return selected_contexts

async def ultra_optimized_answer_generation(question: str, contexts: List[str]) -> str:
    """Ultra-optimized answer generation using Gemini 2.5 Flash Lite with enhanced prompting"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Select optimal number of contexts based on their quality
        selected_contexts = contexts[:4]  # Slightly increased for better accuracy
        
        # Create optimized context text with better formatting
        context_text = ""
        for i, ctx in enumerate(selected_contexts):
            # Truncate very long contexts while preserving important information
            if len(ctx) > 500:
                # Try to find sentence breaks near 400 chars
                truncated = ctx[:400]
                last_sentence = truncated.rfind('.')
                if last_sentence > 300:
                    ctx = truncated[:last_sentence + 1]
                else:
                    ctx = truncated + "..."
            
            context_text += f"Context {i+1}: {ctx}\n\n"
        
        logger.info(f"Generating answer with {len(selected_contexts)} contexts using Gemini 2.5 Flash Lite")
        
        # Enhanced prompt engineering for better accuracy and speed
        prompt = f"""You are a precise information assistant. Answer the question using only the provided contexts.

{context_text.strip()}

Question: {question}

Instructions:
- Provide a direct, factual answer in 2-4 sentences
- Use information ONLY from the contexts above
- If the contexts don't contain the answer, say so clearly
- Be specific and cite relevant details when available

Answer:"""

        # Use Gemini 2.5 Flash Lite with optimized settings
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,  # Low temperature for consistency
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 400,  # Optimized for speed vs quality
            }
        )
        
        answer = response.text.strip()
        
        # Quick post-processing to ensure answer quality
        if len(answer) < 10:
            return "The provided context doesn't contain sufficient information to answer this question."
        
        logger.info(f"Generated enhanced answer: {len(answer)} chars")
        return answer
        
    except Exception as e:
        logger.info(f"Error in ultra-optimized generation: {e}")
        return f"Error generating answer. Please try again."

# Enhanced background task for cache management
async def enhanced_cache_cleanup():
    """Enhanced cache cleanup with better memory management"""
    global search_cache, embedding_cache, query_enhancement_cache
    
    current_time = time.time()
    
    # Smart cache cleanup based on usage patterns
    for cache_dict, name in [(search_cache, "search"), (embedding_cache, "embedding"), (query_enhancement_cache, "query_enhancement")]:
        if len(cache_dict) > CACHE_SIZE:
            # Remove oldest 30% of entries
            old_keys = list(cache_dict.keys())[:len(cache_dict) // 3]
            for key in old_keys:
                cache_dict.pop(key, None)
            logger.info(f"Cleaned {name} cache, remaining: {len(cache_dict)}")
    
    # Force garbage collection for better memory management
    gc.collect()

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": time.time(), 
        "version": "4.0.0",
        "optimizations": "ultra-enhanced",
        "gemini_model": "gemini-2.5-flash-lite"
    }

@app.post("/debug-search")
async def debug_search_endpoint(
    request: SearchDebugRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Enhanced debug endpoint with comprehensive performance metrics"""
    try:
        start_time = time.time()
        
        # Schedule enhanced cache cleanup
        background_tasks.add_task(enhanced_cache_cleanup)
        
        # Process query with enhanced expansion
        processed_query = preprocess_query_cached(request.query)
        query_variations = enhanced_query_expansion(processed_query)
        
        # Get enhanced search results with timing
        search_start = time.time()
        search_results = await enhanced_parallel_search(query_variations, request.top_k)
        search_time = time.time() - search_start
        
        # Get index stats
        index_stats = pc_index.describe_index_stats()
        
        # Process contexts with enhanced filtering
        context_start = time.time()
        contexts = enhanced_context_filtering(search_results, request.query)
        context_time = time.time() - context_start
        
        total_time = time.time() - start_time
        
        return {
            "query": request.query,
            "processed_query": processed_query,
            "query_variations": query_variations,
            "performance": {
                "total_time_ms": round(total_time * 1000, 2),
                "search_time_ms": round(search_time * 1000, 2),
                "context_time_ms": round(context_time * 1000, 2),
                "cache_stats": {
                    "search_cache_size": len(search_cache),
                    "embedding_cache_size": len(embedding_cache),
                    "query_enhancement_cache_size": len(query_enhancement_cache)
                }
            },
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension,
                "namespaces": index_stats.namespaces
            },
            "search_results": {
                "total_matches": len(search_results),
                "enhanced_top_k": request.top_k,
                "matches": [
                    {
                        "id": r['id'],
                        "score": round(r['score'], 4),
                        "doc_id": r['doc_id'],
                        "query_variant": r.get('query_variant', 'N/A'),
                        "text_preview": r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
                    }
                    for r in search_results[:8]  # Show more results
                ]
            },
            "enhanced_contexts": {
                "count": len(contexts),
                "contexts": [ctx[:200] + "..." if len(ctx) > 200 else ctx for ctx in contexts[:4]]
            }
        }
    except Exception as e:
        logger.error(f"Error in enhanced debug search: {e}")
        return {"error": str(e)}

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query_ultra_optimized(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Ultra-optimized main endpoint with all enhancements"""
    try:
        start_time = time.time()
        
        # Schedule enhanced background cleanup
        background_tasks.add_task(enhanced_cache_cleanup)
        
        # Enhanced index check
        index_stats = pc_index.describe_index_stats()
        total_vectors = index_stats.total_vector_count
        
        logger.info(f"Processing {len(request.questions)} questions (Index: {total_vectors} vectors) - Ultra Optimized")
        
        if total_vectors == 0:
            return QueryResponse(
                answers=["The document index is empty. Please index some documents first."] * len(request.questions)
            )
        
        # Ultra-optimized question processing with enhanced pipeline
        async def process_single_question_ultra(question: str) -> str:
            try:
                question_start = time.time()
                logger.info(f"Ultra processing: {question[:60]}...")
                
                # Enhanced query preprocessing and expansion
                processed_query = preprocess_query_cached(question)
                query_variations = enhanced_query_expansion(processed_query)
                
                # Enhanced parallel search with increased retrieval
                search_results = await enhanced_parallel_search(query_variations, top_k=PINECONE_TOP_K)
                
                if not search_results:
                    return f"No relevant information found for: '{question}'"
                
                # Enhanced context filtering
                contexts = enhanced_context_filtering(search_results, question)
                
                if not contexts:
                    return f"Found potentially related information, but not specific enough for: '{question}'"
                
                # Ultra-optimized answer generation with Gemini 2.5 Flash Lite
                answer = await ultra_optimized_answer_generation(question, contexts)
                
                question_time = time.time() - question_start
                logger.info(f"Ultra question processed in {question_time:.2f}s")
                
                return answer
                
            except Exception as e:
                logger.error(f"Error in ultra processing '{question[:30]}...': {e}")
                return f"Error processing question: {str(e)}"
        
        # Process questions with enhanced concurrency control
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        
        async def bounded_ultra_process(question: str) -> str:
            async with semaphore:
                return await process_single_question_ultra(question)
        
        # Execute with enhanced concurrency
        tasks = [bounded_ultra_process(q) for q in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Enhanced exception handling
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Exception in ultra question {i}: {answer}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        # Enhanced cleanup
        gc.collect()
        
        processing_time = time.time() - start_time
        avg_time_per_question = processing_time / len(request.questions)
        
        logger.info(f"Ultra processed {len(request.questions)} questions in {processing_time:.2f}s "
                   f"(avg: {avg_time_per_question:.2f}s per question)")
        
        return QueryResponse(answers=processed_answers)
        
    except Exception as e:
        logger.error(f"Error in ultra main processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hackrx/run", response_model=QueryResponse)
async def process_query_api_ultra(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Alternative endpoint path - Ultra optimized"""
    return await process_query_ultra_optimized(request, background_tasks, token)

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Railway Semantic Search API - Ultra Optimized Version",
        "version": "4.0.0",
        "status": "active",
        "gemini_model": "gemini-2.5-flash-lite",
        "enhancements": [
            "Gemini 2.5 Flash Lite integration",
            "Enhanced parallel search execution",
            "Increased Pinecone retrieval (25+ records)",
            "Advanced embedding caching with SHA256",
            "Smart query expansion with semantic variations",
            "Multi-stage search with broader initial retrieval",
            "Enhanced context filtering with relevance scoring",
            "Ultra-optimized answer generation",
            "Improved concurrent processing (8 searches, 5 generations)",
            "Advanced deduplication with text fingerprinting",
            "Enhanced cache management with TTL",
            "Model warmup for reduced latency"
        ],
        "performance_improvements": {
            "pinecone_top_k": PINECONE_TOP_K,
            "max_concurrent_searches": MAX_CONCURRENT_SEARCHES,
            "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
            "cache_size": CACHE_SIZE,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE
        },
        "endpoints": [
            "GET /",
            "GET /health", 
            "GET /debug",
            "POST /debug-search",
            "POST /hackrx/run",
            "POST /api/hackrx/run"
        ]
    }

@app.get("/debug")
async def debug_info_ultra():
    """Ultra-enhanced debug endpoint with comprehensive metrics"""
    try:
        index_stats = pc_index.describe_index_stats()
        
        # Test embedding with performance timing
        test_start = time.time()
        test_embedding = embedding_model.encode(["test query performance"], normalize_embeddings=True)[0]
        embedding_time = (time.time() - test_start) * 1000
        
        return {
            "app_status": "running",
            "version": "4.0.0",
            "gemini_model": "gemini-2.5-flash-lite",
            "models": {
                "embedding_loaded": embedding_model is not None,
                "embedding_dimension": len(test_embedding),
                "embedding_test_time_ms": round(embedding_time, 2),
                "pinecone_connected": pc_index is not None
            },
            "ultra_performance": {
                "search_cache_size": len(search_cache),
                "embedding_cache_size": len(embedding_cache),
                "query_enhancement_cache_size": len(query_enhancement_cache),
                "max_concurrent_searches": MAX_CONCURRENT_SEARCHES,
                "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
                "cache_size_limit": CACHE_SIZE,
                "pinecone_top_k": PINECONE_TOP_K,
                "enhanced_search_depth": ENHANCED_SEARCH_DEPTH,
                "embedding_batch_size": EMBEDDING_BATCH_SIZE
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
            },
            "optimizations_active": {
                "model_warmup": "enabled",
                "enhanced_caching": "enabled",
                "multi_stage_search": "enabled",
                "advanced_deduplication": "enabled",
                "semantic_query_expansion": "enabled",
                "ultra_answer_generation": "enabled"
            }
        }
    except Exception as e:
        return {"error": str(e), "app_status": "error"}

# Additional utility endpoints for monitoring and optimization

@app.get("/cache-stats")
async def get_cache_statistics(token: str = Depends(verify_token)):
    """Get detailed cache statistics for monitoring"""
    return {
        "timestamp": time.time(),
        "cache_statistics": {
            "search_cache": {
                "size": len(search_cache),
                "max_size": CACHE_SIZE
            },
            "embedding_cache": {
                "size": len(embedding_cache),
                "max_size": CACHE_SIZE
            },
            "query_enhancement_cache": {
                "size": len(query_enhancement_cache),
                "max_size": CACHE_SIZE
            },
            "auth_cache": {
                "size": len(_auth_cache),
                "max_size": 200
            }
        },
        "performance_config": {
            "pinecone_top_k": PINECONE_TOP_K,
            "enhanced_search_depth": ENHANCED_SEARCH_DEPTH,
            "max_concurrent_searches": MAX_CONCURRENT_SEARCHES,
            "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE
        }
    }

@app.post("/clear-cache")
async def clear_all_caches(token: str = Depends(verify_token)):
    """Clear all caches for fresh start (admin endpoint)"""
    global search_cache, embedding_cache, query_enhancement_cache
    
    search_cache.clear()
    embedding_cache.clear()
    query_enhancement_cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    return {
        "status": "success",
        "message": "All caches cleared successfully",
        "timestamp": time.time()
    }

if _name_ == "_main_":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False, 
        log_level="info",
        workers=1,  # Single worker for optimal memory usage with enhanced caching
        access_log=False,  # Disable access logs for better performance
        # Enhanced uvicorn settings for better performance
        loop="asyncio",
        http="httptools"
    )
