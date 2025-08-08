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

async def enhanced_search_similar_chunks(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """Enhanced search with better result processing for detailed answers"""
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Preprocess query (unchanged to maintain compatibility)
        processed_query = preprocess_query(query)
        
        # Generate embedding asynchronously (unchanged)
        query_embedding = await generate_embedding_async(processed_query)
        
        # Search with higher top_k for better selection
        search_results = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        # Enhanced result processing with better scoring
        results = []
        query_words = set(query.lower().split())
        
        # Extract key terms that often indicate important information
        important_patterns = [
            r'\d+\s*(days?|months?|years?)', r'\d+%', r'sum insured', r'premium', 
            r'waiting period', r'grace period', r'coverage', r'benefit', r'limit',
            r'condition', r'eligibility', r'defined', r'means', r'includes'
        ]
        
        for match in search_results.matches:
            if match.metadata and match.score > 0.06:  # Lower threshold for more results
                text = match.metadata.get('text', '')
                
                # Calculate keyword overlap bonus
                text_words = set(text.lower().split())
                keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
                
                # Bonus for containing important patterns
                pattern_bonus = 0
                for pattern in important_patterns:
                    if re.search(pattern, text.lower()):
                        pattern_bonus += 0.05
                
                # Enhanced scoring with pattern bonus
                enhanced_score = match.score + (keyword_overlap * 0.12) + pattern_bonus
                
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'enhanced_score': enhanced_score,
                    'text': text,
                    'doc_id': match.metadata.get('doc_id', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'keyword_overlap': keyword_overlap,
                    'pattern_bonus': pattern_bonus
                })
        
        # Sort by enhanced score
        results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        logger.info(f"Found {len(results)} matches")
        return results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def smart_context_selection(search_results: List[Dict[str, Any]], query: str) -> List[str]:
    """Enhanced context selection for comprehensive answers"""
    if not search_results:
        return []
    
    contexts = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    # Enhanced context selection with better criteria
    for result in search_results:
        text = result['text']
        score = result['enhanced_score']
        keyword_overlap = result.get('keyword_overlap', 0)
        pattern_bonus = result.get('pattern_bonus', 0)
        
        # Multi-tier selection criteria with pattern consideration
        if score > 0.35 or (score > 0.25 and pattern_bonus > 0.1):  # High confidence
            contexts.append(text)
        elif score > 0.22 and keyword_overlap > 0.2:  # Medium confidence
            contexts.append(text)
        elif score > 0.15 and (keyword_overlap > 0.15 or pattern_bonus > 0.05):  # Lower confidence but relevant
            # Additional check for policy-specific terms
            important_terms = [
                'policy', 'coverage', 'premium', 'benefit', 'waiting', 'period', 'grace',
                'mediclaim', 'insurance', 'claim', 'discount', 'hospital', 'treatment',
                'sum insured', 'eligibility', 'condition', 'limit', 'defined', 'means',
                'days', 'months', 'years', 'percentage', 'expenses', 'reimbursement'
            ]
            if any(term in text.lower() for term in important_terms):
                contexts.append(text)
        
        # Allow more contexts for comprehensive answers
        max_contexts = 8 if len(query.split()) > 6 else 6
        if len(contexts) >= max_contexts:
            break
    
    # Fallback with best results
    if not contexts and search_results:
        contexts = [search_results[0]['text']]
        if len(search_results) > 1:
            contexts.append(search_results[1]['text'])
        logger.info("Using fallback contexts")
    
    logger.info(f"Selected {len(contexts)} contexts for comprehensive answer")
    return contexts

async def generate_enhanced_answer(question: str, contexts: List[str]) -> str:
    """Enhanced answer generation optimized for comprehensive policy responses"""
    try:
        if not contexts:
            return "I couldn't find relevant information in the indexed documents to answer this question."
        
        # Use more contexts for comprehensive answers
        selected_contexts = contexts[:6]  # Increased for better coverage
        
        # Enhanced smart truncation - preserve important details
        processed_contexts = []
        for i, ctx in enumerate(selected_contexts):
            if len(ctx) > 600:  # Increased limit for more detail
                # Prioritize sentences with numbers, percentages, and key terms
                sentences = ctx.split('. ')
                scored_sentences = []
                question_words = set(question.lower().split())
                
                # Important patterns for policy documents
                important_patterns = [
                    r'\d+\s*(?:days?|months?|years?)', r'\d+%', r'\d+\s*(?:lakhs?|crores?)',
                    r'sum insured', r'premium', r'waiting period', r'grace period',
                    r'coverage', r'benefit', r'limit', r'condition', r'eligibility'
                ]
                
                for sentence in sentences:
                    score = 0
                    sentence_words = set(sentence.lower().split())
                    
                    # Score based on question word overlap
                    overlap = len(question_words.intersection(sentence_words))
                    score += overlap * 2
                    
                    # Bonus for important patterns
                    for pattern in important_patterns:
                        if re.search(pattern, sentence.lower()):
                            score += 3
                    
                    # Bonus for definition patterns
                    if re.search(r'(?:is defined|means|includes|shall mean)', sentence.lower()):
                        score += 2
                    
                    scored_sentences.append((sentence, score))
                
                # Sort by score and take top sentences
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                relevant_sentences = [s[0] for s in scored_sentences[:3]]  # Top 3 sentences
                
                if relevant_sentences:
                    ctx = '. '.join(relevant_sentences)
                else:
                    ctx = ctx[:600]  # Fallback to truncation
            
            processed_contexts.append(ctx)
        
        context_text = "\n\n".join(processed_contexts)
        
        logger.info(f"Generating comprehensive answer with {len(selected_contexts)} contexts")
        
        # Enhanced prompt for detailed, comprehensive answers
        prompt = f"""Based on the policy information provided, answer the question with complete details including specific numbers, conditions, and requirements.

POLICY INFORMATION:
{context_text}

QUESTION: {question}

Instructions:
- Provide a comprehensive answer with all relevant details
- Include specific numbers, percentages, time periods, and amounts when mentioned
- Include all conditions, requirements, and exceptions
- Be precise about eligibility criteria and limitations
- Structure the answer clearly and logically
- Do not reference sources or document locations
- If multiple aspects are covered, address them all
- Use exact terminology from the policy when relevant

DETAILED ANSWER:"""

        # Run Gemini API call in thread pool
        loop = asyncio.get_event_loop()
        
        def _generate_content():
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.02,  # Very low for consistency
                    'top_p': 0.8,        # Slightly higher for comprehensive answers
                    'top_k': 20,         # More options for detailed responses
                    'max_output_tokens': 400,  # Increased for comprehensive answers
                }
            )
            return response.text.strip()
        
        answer = await loop.run_in_executor(executor, _generate_content)
        
        # Enhanced post-processing for policy answers
        answer = re.sub(r'\n+', ' ', answer)  # Remove multiple newlines
        answer = re.sub(r'\s+', ' ', answer)  # Normalize spaces
        
        # Clean up any remaining artifacts
        answer = re.sub(r'(?:Context \d+:?|Based on the policy|According to|As per)', '', answer)
        answer = answer.strip()
        
        logger.info(f"Generated comprehensive answer: {len(answer)} chars")
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

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
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
                
                # Enhanced search with optimized parameters
                search_results = await enhanced_search_similar_chunks(question, top_k=15)
                
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
    


    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
