import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRX Document Q&A System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# In-memory storage for documents and embeddings
document_store: Dict[str, Dict] = {}
MAX_DOCUMENTS = 20  # Good balance for memory

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "default_token")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

class DocumentRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the PDF document")
    questions: List[str] = Field(..., min_length=1, max_length=10, description="List of questions")

class DocumentResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials

def generate_document_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

async def download_pdf(url: str) -> bytes:
    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(str(url)) as response:
                if response.status != 200:
                    raise HTTPException(400, f"Failed to download PDF: HTTP {response.status}")
                content = await response.read()
                if len(content) == 0:
                    raise HTTPException(400, "Downloaded file is empty")
                return content
    except aiohttp.ClientError as e:
        raise HTTPException(400, f"Network error downloading PDF: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_parts = []
        max_pages = min(40, len(pdf_reader.pages))  # Good balance
        
        for page_num in range(max_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            raise ValueError("No text could be extracted from PDF")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(400, f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Create optimized chunks for Gemini embedding API"""
    words = text.split()
    if len(words) == 0:
        return [text] if text.strip() else []
    
    chunks = []
    step_size = max(chunk_size - overlap, 200)
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_size >= len(words):
            break
    
    return chunks if chunks else ([text] if text.strip() else [])

async def create_gemini_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Create embeddings using Gemini's embedding API"""
    try:
        # Process in batches to avoid API limits
        batch_size = 10  # Gemini embedding API batch limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Use Gemini's embedding model
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model="models/embedding-001",  # Gemini's embedding model
                    content=batch,
                    task_type="retrieval_document"  # Optimized for document retrieval
                )
                
                # Extract embeddings from result
                if hasattr(result, 'embedding'):
                    # Single embedding
                    all_embeddings.append(result['embedding'])
                elif hasattr(result, 'embeddings'):
                    # Multiple embeddings
                    all_embeddings.extend(result['embeddings'])
                else:
                    # Handle different response formats
                    batch_embeddings = result if isinstance(result, list) else [result]
                    all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect API limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to create embeddings for batch {i//batch_size + 1}: {e}")
                # Continue with other batches
                continue
        
        if not all_embeddings:
            return None
            
        logger.info(f"Created {len(all_embeddings)} embeddings using Gemini API")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error creating Gemini embeddings: {e}")
        return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        # Convert to numpy arrays for easier computation
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

async def get_relevant_chunks_with_gemini_embeddings(question: str, chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
    """Use Gemini embeddings for semantic similarity"""
    if not embeddings or len(chunks) == 0:
        return chunks[:top_k]
    
    try:
        # Create embedding for the question using Gemini
        question_result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=[question],
            task_type="retrieval_query"  # Optimized for queries
        )
        
        # Extract question embedding
        if hasattr(question_result, 'embedding'):
            question_embedding = question_result.embedding
        elif hasattr(question_result, 'embeddings'):
            question_embedding = question_result.embeddings[0]
        else:
            question_embedding = question_result[0] if isinstance(question_result, list) else question_result
        
        # Calculate similarities with all chunk embeddings
        similarities = []
        for i, chunk_embedding in enumerate(embeddings):
            if i < len(chunks):  # Ensure we don't go out of bounds
                similarity = cosine_similarity(question_embedding, chunk_embedding)
                similarities.append((similarity, i))
        
        # Sort by similarity and get top chunks
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in similarities[:top_k]]
        
        relevant_chunks = [chunks[i] for i in top_indices if i < len(chunks)]
        similarity_scores = [score for score, _ in similarities[:len(relevant_chunks)]]
        
        logger.info(f"Selected {len(relevant_chunks)} relevant chunks with Gemini embeddings, similarities: {similarity_scores}")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in Gemini embedding similarity search: {e}")
        # Fallback to keyword-based matching
        return await get_relevant_chunks_keyword_fallback(question, chunks, top_k)

async def get_relevant_chunks_keyword_fallback(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Fallback method using keyword matching"""
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    try:
        # Simple but effective keyword-based relevance scoring
        question_words = set(word.lower().strip('.,!?":;()[]{}') for word in question.split() if len(word) > 2)
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(word.lower().strip('.,!?":;()[]{}') for word in chunk.split() if len(word) > 2)
            
            # Calculate relevance score
            common_words = question_words.intersection(chunk_words)
            score = len(common_words)
            
            # Boost score for exact phrase matches
            question_lower = question.lower()
            chunk_lower = chunk.lower()
            for word in question_words:
                if word in chunk_lower:
                    score += 0.5
            
            chunk_scores.append((score, i))
        
        # Sort by score and return top chunks
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        relevant_indices = [idx for _, idx in chunk_scores[:top_k]]
        relevant_chunks = [chunks[i] for i in relevant_indices]
        
        logger.info(f"Selected {len(relevant_chunks)} relevant chunks using keyword fallback")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in keyword fallback: {e}")
        return chunks[:top_k]

async def generate_answer(question: str, context: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit context length
        limited_context = context[:4500] if len(context) > 4500 else context
        
        prompt = f"""Based on the provided context, answer the question concisely and accurately. If the context doesn't contain enough information to answer the question, say so.

Context:
{limited_context}

Question: {question}

Answer:"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=400,
                    temperature=0.2,
                    top_p=0.8,
                )
            ),
            timeout=25.0
        )
        
        return response.text.strip() if response.text else "No answer generated."
        
    except asyncio.TimeoutError:
        logger.error("Answer generation timed out")
        return "Sorry, the answer generation timed out. Please try again."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."

async def process_document_and_questions(url: str, questions: List[str]) -> List[str]:
    doc_hash = generate_document_hash(str(url))
    
    # Clean up old documents if we're at capacity
    if len(document_store) >= MAX_DOCUMENTS and doc_hash not in document_store:
        # Remove oldest document
        oldest_doc = min(document_store.items(), key=lambda x: x[1]['processed_at'])
        del document_store[oldest_doc[0]]
        logger.info(f"Removed oldest document to free memory")
    
    # Check if document is already processed
    if doc_hash not in document_store:
        logger.info(f"Processing new document: {doc_hash}")
        try:
            pdf_content = await download_pdf(str(url))
            text = extract_text_from_pdf(pdf_content)
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Limit number of chunks to manage API usage
            chunks = chunks[:60]  # Reasonable limit
            
            # Create embeddings using Gemini API
            embeddings = await create_gemini_embeddings(chunks)
            
            # Store in memory
            document_store[doc_hash] = {
                'chunks': chunks,
                'embeddings': embeddings,
                'url': str(url),
                'processed_at': datetime.utcnow().isoformat()
            }
            
            embed_status = "with Gemini embeddings" if embeddings else "without embeddings (will use keyword matching)"
            logger.info(f"Document processed successfully. Chunks: {len(chunks)} {embed_status}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info(f"Using cached document: {doc_hash}")
    
    document_data = document_store[doc_hash]
    chunks = document_data['chunks']
    embeddings = document_data.get('embeddings')
    
    async def process_single_question(question: str) -> str:
        try:
            # Try Gemini embeddings first, fallback to keyword matching
            if embeddings:
                relevant_chunks = await get_relevant_chunks_with_gemini_embeddings(question, chunks, embeddings, top_k=3)
            else:
                relevant_chunks = await get_relevant_chunks_keyword_fallback(question, chunks, top_k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            context = "\n\n".join(relevant_chunks)
            answer = await generate_answer(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with controlled concurrency
    semaphore = asyncio.Semaphore(3)  # Allow 3 concurrent questions
    
    async def process_with_semaphore(question: str) -> str:
        async with semaphore:
            return await process_single_question(question)
    
    # Process questions concurrently but controlled
    tasks = [process_with_semaphore(q) for q in questions]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions in the results
    final_answers = []
    for answer in answers:
        if isinstance(answer, Exception):
            logger.error(f"Error in question processing: {answer}")
            final_answers.append("An error occurred while processing this question.")
        else:
            final_answers.append(answer)
    
    return final_answers

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cached_documents": len(document_store),
        "max_documents": MAX_DOCUMENTS,
        "gemini_configured": bool(GEMINI_API_KEY),
        "embedding_method": "gemini_api",
        "features": ["gemini_embeddings", "gemini_qa", "pdf_processing"]
    }

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System (Gemini Embeddings - Ultra Lightweight)",
        "version": "1.0.0",
        "status": "running",
        "embedding_method": "gemini_api",
        "advantages": ["ultra_small_image_size", "no_ml_dependencies", "high_accuracy_embeddings"]
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(
    request: DocumentRequest,
    token: str = Depends(verify_token)
) -> DocumentResponse:
    try:
        logger.info(f"Processing request for document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        answers = await asyncio.wait_for(
            process_document_and_questions(request.documents, request.questions),
            timeout=300.0
        )
        
        logger.info(f"Successfully processed {len(answers)} answers")
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Request processing timed out")
        raise HTTPException(408, "Request processing timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"An unexpected error occurred: {str(e)}")

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")
    logger.info("Using Gemini API for embeddings - ultra lightweight deployment!")
    logger.info(f"Max documents in cache: {MAX_DOCUMENTS}")
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")
    document_store.clear()
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
