import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
from typing import List, Dict
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

# In-memory storage instead of Pinecone
document_store: Dict[str, Dict] = {}

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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Create larger chunks since we're using Gemini for both embedding and answering"""
    words = text.split()
    if len(words) == 0:
        return [text] if text.strip() else []
    
    chunks = []
    step_size = max(chunk_size - overlap, 100)  # Ensure minimum step size
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_size >= len(words):
            break
    
    return chunks if chunks else ([text] if text.strip() else [])

async def get_relevant_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Use Gemini to find most relevant chunks instead of vector similarity"""
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit chunks to prevent token overflow
        limited_chunks = chunks[:15]  # Process max 15 chunks at once
        
        # Create a prompt to rank chunks by relevance
        chunks_text = ""
        for i, chunk in enumerate(limited_chunks):
            chunks_text += f"CHUNK {i+1}:\n{chunk[:800]}\n\n---\n\n"  # Limit each chunk to 800 chars
        
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
            timeout=30.0
        )
        
        # Parse the response to get chunk indices
        try:
            chunk_indices = [int(x.strip()) - 1 for x in response.text.strip().split(',')]
            relevant_chunks = [limited_chunks[i] for i in chunk_indices if 0 <= i < len(limited_chunks)]
            return relevant_chunks[:top_k] if relevant_chunks else chunks[:top_k]
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse chunk selection response: {e}")
            return chunks[:top_k]
            
    except asyncio.TimeoutError:
        logger.warning("Chunk selection timed out, using first chunks")
        return chunks[:top_k]
    except Exception as e:
        logger.error(f"Error in chunk selection: {e}")
        return chunks[:top_k]

async def generate_answer(question: str, context: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit context length to prevent token overflow
        limited_context = context[:4000] if len(context) > 4000 else context
        
        prompt = f"""Based on the provided context, answer the question concisely and accurately.

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
            timeout=30.0
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
            
            # Store in memory
            document_store[doc_hash] = {
                'chunks': chunks,
                'url': str(url),
                'processed_at': datetime.utcnow().isoformat()
            }
            logger.info(f"Document processed successfully. Chunks: {len(chunks)}")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info(f"Using cached document: {doc_hash}")
    
    chunks = document_store[doc_hash]['chunks']
    
    async def process_single_question(question: str) -> str:
        try:
            # Get relevant chunks using Gemini
            relevant_chunks = await get_relevant_chunks(question, chunks, top_k=3)
            
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
    
    # Process all questions
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
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System (Optimized)",
        "version": "1.0.0",
        "status": "running"
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
            timeout=300.0  # 5 minutes timeout
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

# Background task for cleanup
async def cleanup_old_documents():
    while True:
        try:
            await asyncio.sleep(3600)  # Clean up every hour
            if len(document_store) > 50:  # Keep only 50 most recent documents
                # Remove oldest documents
                sorted_docs = sorted(
                    document_store.items(), 
                    key=lambda x: x[1]['processed_at']
                )
                for doc_id, _ in sorted_docs[:-25]:  # Keep only 25 most recent
                    del document_store[doc_id]
                logger.info(f"Cleaned up old documents. Current count: {len(document_store)}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")
    # Start cleanup task in background
    asyncio.create_task(cleanup_old_documents())
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")
    # Clear memory
    document_store.clear()
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
