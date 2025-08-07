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

def advanced_search_similar_chunks(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Advanced search with multiple strategies"""
    try:
        logger.info(f"Advanced search for query: '{query}'")
        
        # Preprocess query
        processed_query = preprocess_query(query)
        query_variations = expand_query(processed_query)
        
        all_results = []
        
        # Search with each query variation
        for i, query_var in enumerate(query_variations):
            logger.info(f"Searching with variation {i+1}: '{query_var}'")
            
            # Generate embedding with same normalization as indexing
            query_embedding = embedding_model.encode(
                [query_var], 
                convert_to_tensor=False,
                normalize_embeddings=True  # Match indexing normalization
            )[0].tolist()
            
            # Search in Pinecone
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Collect results with query variant info
            for match in search_results.matches:
                if match.metadata:
                    all_results.append({
                        'id': match.id,
                        'score': match.score,
                        'text': match.metadata.get('text', ''),
                        'doc_id': match.metadata.get('doc_id', ''),
                        'chunk_index': match.metadata.get('chunk_index', 0),
                        'query_variant': i,
                        'metadata': match.metadata
                    })
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Log results for debugging
        logger.info(f"Found {len(unique_results)} unique matches")
        for i, result in enumerate(unique_results[:5]):
            logger.info(f"  Match {i}: ID={result['id']}, Score={result['score']:.4f}, "
                       f"Query variant={result['query_variant']}")
            logger.info(f"    Text preview: {result['text'][:100]}...")
        
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        return []

def filter_and_rank_contexts(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Filter and rank contexts using multiple criteria"""
    if not search_results:
        return []
    
    # Multiple filtering strategies
    contexts = []
    
    # Strategy 1: High similarity score
    high_score_results = [r for r in search_results if r['score'] > 0.15]
    logger.info(f"High score results (>0.15): {len(high_score_results)}")
    
    # Strategy 2: Keyword matching
    query_words = set(query.lower().split())
    keyword_results = []
    for result in search_results[:15]:  # Check top 15
        text_words = set(result['text'].lower().split())
        overlap = len(query_words.intersection(text_words))
        if overlap >= 2 or result['score'] > 0.1:  # At least 2 word overlap or decent score
            keyword_results.append(result)
    
    logger.info(f"Keyword matching results: {len(keyword_results)}")
    
    # Combine strategies
    combined_results = high_score_results if high_score_results else keyword_results
    
    # If still no results, take top scoring ones anyway
    if not combined_results and search_results:
        combined_results = search_results[:5]
        logger.info("Using fallback: top 5 results regardless of score")
    
    # Extract contexts
    for result in combined_results[:10]:  # Top 10 contexts max
        if result['text'] and len(result['text'].strip()) > 50:
            contexts.append(result['text'])
    
    logger.info(f"Final contexts selected: {len(contexts)}")
    return contexts

def generate_comprehensive_answer(question: str, contexts: List[str]) -> str:
    """Generate comprehensive answer with improved prompting"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Prepare context with better formatting
        numbered_contexts = []
        for i, context in enumerate(contexts[:7], 1):  # Use up to 7 contexts
            numbered_contexts.append(f"Context {i}:\n{context}\n")
        
        context_text = "\n".join(numbered_contexts)
        
        logger.info(f"Generating answer for: '{question}'")
        logger.info(f"Using {len(contexts)} contexts, total length: {len(context_text)} chars")
        
        # Improved prompt with specific instructions
        prompt = f"""You are a helpful AI assistant. Based on the provided contexts from documents, answer the user's question comprehensively and accurately.

CONTEXTS:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a detailed answer based on the information found in the contexts
2. If you find relevant information, synthesize it from multiple contexts if needed
3. Be specific and cite relevant details
4. If the exact answer isn't available but related information exists, provide that and explain what's missing
5. Only say you cannot answer if absolutely no relevant information is found
6. Structure your answer clearly with key points

ANSWER:"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
        
        answer = response.text.strip()
        logger.info(f"Generated answer length: {len(answer)} chars")
        logger.info(f"Answer preview: {answer[:150]}...")
        
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
                logger.info(f"\n--- Processing Question ---")
                logger.info(f"Question: {question}")
                
                # Advanced search
                search_results = advanced_search_similar_chunks(question, top_k=25)
                
                if not search_results:
                    logger.warning("No search results found")
                    return f"I couldn't find any relevant information for the question: '{question}'"
                
                # Filter and rank contexts
                contexts = filter_and_rank_contexts(search_results, question)
                
                if not contexts:
                    logger.warning("No contexts passed filtering")
                    return f"I found some potentially related information, but it doesn't seem relevant enough to answer: '{question}'"
                
                # Generate answer
                answer = generate_comprehensive_answer(question, contexts)
                
                logger.info(f"Successfully generated answer for: {question}")
                return answer
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                return f"Error processing the question '{question}': {str(e)}"
        
        # Process all questions concurrently
        tasks = [process_single_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Clean up
        gc.collect()
        
        processing_time = time.time() - start_time
        logger.info(f"\n=== PROCESSING COMPLETE ===")
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"Questions processed: {len(request.questions)}")
        
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
