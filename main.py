import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# Initialize the lightweight sentence transformer model
try:
    # Load the lightweight model (only ~90MB)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer: {e}")
    sentence_model = None

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
        max_pages = min(50, len(pdf_reader.pages))
        
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
    """Create optimized chunks for sentence transformer processing"""
    words = text.split()
    if len(words) == 0:
        return [text] if text.strip() else []
    
    chunks = []
    step_size = max(chunk_size - overlap, 100)
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_size >= len(words):
            break
    
    return chunks if chunks else ([text] if text.strip() else [])

def create_embeddings(texts: List[str]) -> np.ndarray:
    """Create embeddings for text chunks using sentence transformer"""
    if not sentence_model:
        raise ValueError("Sentence transformer model not available")
    
    try:
        # Process in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate texts to avoid model limits (all-MiniLM-L6-v2 has 512 token limit)
            truncated_batch = [text[:2000] for text in batch]  # ~500 tokens approx
            
            embeddings = sentence_model.encode(
                truncated_batch, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=min(len(truncated_batch), 16)  # Small batch size for memory efficiency
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise ValueError(f"Failed to create embeddings: {str(e)}")

async def get_relevant_chunks_semantic(question: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> List[str]:
    """Use sentence transformer for semantic similarity instead of Gemini ranking"""
    if not sentence_model or len(chunks) == 0:
        return chunks[:top_k]
    
    try:
        # Create embedding for the question
        question_embedding = sentence_model.encode([question[:2000]], convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return the most relevant chunks
        relevant_chunks = [chunks[i] for i in top_indices if i < len(chunks)]
        
        logger.info(f"Selected {len(relevant_chunks)} relevant chunks with similarities: {similarities[top_indices][:len(relevant_chunks)]}")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in semantic similarity search: {e}")
        # Fallback to first chunks
        return chunks[:top_k]

async def get_relevant_chunks_fallback(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Fallback method using Gemini for chunk selection when sentence transformer fails"""
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit chunks to prevent token overflow
        limited_chunks = chunks[:12]
        
        chunks_text = ""
        for i, chunk in enumerate(limited_chunks):
            chunks_text += f"CHUNK {i+1}:\n{chunk[:600]}\n\n---\n\n"
        
        prompt = f"""Question: "{question}"

Below are text chunks. Rate each chunk's relevance to answering the question (1-10).
Return ONLY the numbers of the {top_k} most relevant chunks, comma-separated.

{chunks_text}

Response format: "1,3,7" (just the numbers)"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=50,
                    temperature=0.1
                )
            ),
            timeout=20.0
        )
        
        # Parse response
        try:
            chunk_indices = [int(x.strip()) - 1 for x in response.text.strip().split(',')]
            relevant_chunks = [limited_chunks[i] for i in chunk_indices if 0 <= i < len(limited_chunks)]
            return relevant_chunks[:top_k] if relevant_chunks else chunks[:top_k]
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse chunk selection: {e}")
            return chunks[:top_k]
            
    except Exception as e:
        logger.error(f"Error in fallback chunk selection: {e}")
        return chunks[:top_k]

async def generate_answer(question: str, context: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit context length
        limited_context = context[:4000] if len(context) > 4000 else context
        
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
                    max_output_tokens=500,
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
    
    # Check if document is already processed
    if doc_hash not in document_store:
        logger.info(f"Processing new document: {doc_hash}")
        try:
            pdf_content = await download_pdf(str(url))
            text = extract_text_from_pdf(pdf_content)
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Create embeddings if sentence transformer is available
            embeddings = None
            if sentence_model:
                try:
                    embeddings = await asyncio.to_thread(create_embeddings, chunks)
                    logger.info(f"Created embeddings for {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Failed to create embeddings, will use fallback: {e}")
                    embeddings = None
            
            # Store in memory
            document_store[doc_hash] = {
                'chunks': chunks,
                'embeddings': embeddings,
                'url': str(url),
                'processed_at': datetime.utcnow().isoformat()
            }
            logger.info(f"Document processed successfully. Chunks: {len(chunks)}")
            
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
            # Try semantic similarity first, fallback to Gemini-based selection
            if embeddings is not None and sentence_model:
                relevant_chunks = await get_relevant_chunks_semantic(question, chunks, embeddings, top_k=3)
            else:
                relevant_chunks = await get_relevant_chunks_fallback(question, chunks, top_k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            context = "\n\n".join(relevant_chunks)
            answer = await generate_answer(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with limited concurrency
    semaphore = asyncio.Semaphore(2)
    
    async def process_with_semaphore(question: str) -> str:
        async with semaphore:
            return await process_single_question(question)
    
    answers = []
    for question in questions:
        try:
            answer = await process_with_semaphore(question)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append("An error occurred while processing this question.")
    
    return answers

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cached_documents": len(document_store),
        "gemini_configured": bool(GEMINI_API_KEY),
        "sentence_transformer_available": sentence_model is not None,
        "model_info": "all-MiniLM-L6-v2" if sentence_model else "not_loaded"
    }

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System (Optimized with Sentence Transformers)",
        "version": "1.0.0",
        "status": "running",
        "features": ["semantic_similarity", "gemini_qa", "pdf_processing"]
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

# Memory-efficient cleanup
async def cleanup_old_documents():
    while True:
        try:
            await asyncio.sleep(3600)  # Clean up every hour
            if len(document_store) > 30:  # Reduced limit to manage memory better
                sorted_docs = sorted(
                    document_store.items(), 
                    key=lambda x: x[1]['processed_at']
                )
                for doc_id, _ in sorted_docs[:-15]:  # Keep only 15 most recent
                    del document_store[doc_id]
                logger.info(f"Cleaned up old documents. Current count: {len(document_store)}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")
    logger.info(f"Sentence transformer model status: {'Loaded' if sentence_model else 'Failed to load'}")
    asyncio.create_task(cleanup_old_documents())
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")
    document_store.clear()
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
