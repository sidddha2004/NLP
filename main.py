import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRX Document Q&A System with Pinecone", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "default_token")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

# Validate required environment variables
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY environment variable")

# Initialize services
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = None
index = None
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Gemini embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        import time
        time.sleep(10)
    
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("Pinecone initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise ValueError(f"Pinecone initialization failed: {e}")

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
    """Generate a consistent hash for document URL"""
    return hashlib.sha256(url.encode()).hexdigest()

def generate_chunk_id(doc_hash: str, chunk_index: int) -> str:
    """Generate unique ID for each chunk"""
    return f"{doc_hash}_chunk_{chunk_index}"

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

async def create_gemini_embedding(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """Create a single embedding using Gemini API"""
    try:
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        
        # Extract embedding from result
        if hasattr(result, 'embedding'):
            return result.embedding
        elif isinstance(result, dict) and 'embedding' in result:
            return result['embedding']
        else:
            logger.warning(f"Unexpected embedding result format: {type(result)}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Gemini embedding: {e}")
        return None

async def check_document_exists_in_pinecone(doc_hash: str) -> bool:
    """Check if document already exists in Pinecone"""
    try:
        # Query for any vector with the document hash prefix
        query_result = index.query(
            vector=[0.0] * 768,  # Dummy vector for metadata search
            filter={"document_hash": doc_hash},
            top_k=1,
            include_metadata=True
        )
        
        return len(query_result['matches']) > 0
        
    except Exception as e:
        logger.error(f"Error checking document existence in Pinecone: {e}")
        return False

async def store_document_in_pinecone(doc_hash: str, chunks: List[str], url: str) -> bool:
    """Store document chunks and their embeddings in Pinecone"""
    try:
        logger.info(f"Processing {len(chunks)} chunks for Pinecone storage")
        
        # Process chunks in batches
        batch_size = 10
        vectors_to_upsert = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Create embeddings for the batch
            for j, chunk in enumerate(batch_chunks):
                chunk_index = i + j
                embedding = await create_gemini_embedding(chunk, "retrieval_document")
                
                if embedding:
                    vector_id = generate_chunk_id(doc_hash, chunk_index)
                    metadata = {
                        "document_hash": doc_hash,
                        "chunk_index": chunk_index,
                        "text": chunk,
                        "url": url,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                
                # Small delay to respect API limits
                await asyncio.sleep(0.1)
            
            # Upsert batch to Pinecone
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert[-len(batch_chunks):])
                logger.info(f"Upserted batch {(i//batch_size) + 1} to Pinecone")
        
        logger.info(f"Successfully stored {len(vectors_to_upsert)} chunks in Pinecone for document {doc_hash}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing document in Pinecone: {e}")
        return False

async def retrieve_relevant_chunks_from_pinecone(question: str, doc_hash: str, top_k: int = 3) -> List[str]:
    """Retrieve relevant chunks from Pinecone using question embedding"""
    try:
        # Create embedding for the question
        question_embedding = await create_gemini_embedding(question, "retrieval_query")
        
        if not question_embedding:
            logger.warning("Failed to create question embedding, using fallback")
            return []
        
        # Query Pinecone for similar chunks
        query_result = index.query(
            vector=question_embedding,
            filter={"document_hash": doc_hash},
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract relevant chunks
        relevant_chunks = []
        for match in query_result['matches']:
            if match['metadata'] and 'text' in match['metadata']:
                chunk_text = match['metadata']['text']
                score = match['score']
                relevant_chunks.append(chunk_text)
                logger.debug(f"Retrieved chunk with similarity score: {score:.4f}")
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks from Pinecone")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {e}")
        return []

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
    logger.info(f"Processing document with hash: {doc_hash}")
    
    # Check if document exists in Pinecone
    document_exists = await check_document_exists_in_pinecone(doc_hash)
    
    if not document_exists:
        logger.info("Document not found in Pinecone, processing and storing...")
        
        try:
            # Download and process the document
            pdf_content = await download_pdf(str(url))
            text = extract_text_from_pdf(pdf_content)
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Limit number of chunks for efficiency
            chunks = chunks[:60]
            
            # Store in Pinecone
            success = await store_document_in_pinecone(doc_hash, chunks, str(url))
            
            if not success:
                raise HTTPException(500, "Failed to store document in Pinecone")
                
            logger.info(f"Document successfully processed and stored in Pinecone. Total chunks: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info("Document found in Pinecone, using existing data")
    
    # Process questions using Pinecone retrieval
    async def process_single_question(question: str) -> str:
        try:
            # Retrieve relevant chunks from Pinecone
            relevant_chunks = await retrieve_relevant_chunks_from_pinecone(question, doc_hash, top_k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            context = "\n\n".join(relevant_chunks)
            answer = await generate_answer(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with controlled concurrency
    semaphore = asyncio.Semaphore(3)
    
    async def process_with_semaphore(question: str) -> str:
        async with semaphore:
            return await process_single_question(question)
    
    # Process questions concurrently
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
    try:
        # Test Pinecone connection
        pinecone_status = "connected"
        try:
            index.describe_index_stats()
        except:
            pinecone_status = "disconnected"
            
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "gemini_configured": bool(GEMINI_API_KEY),
            "pinecone_status": pinecone_status,
            "pinecone_index": PINECONE_INDEX_NAME,
            "embedding_method": "gemini_api",
            "vector_database": "pinecone"
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System with Pinecone Vector Database",
        "version": "1.0.0",
        "status": "running",
        "features": ["pinecone_storage", "gemini_embeddings", "gemini_qa", "pdf_processing"],
        "advantages": ["persistent_storage", "fast_retrieval", "scalable", "cost_effective"]
    }

@app.post("/hackrx/run", response_model=DocumentResponse)
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

# Additional endpoint to check document status
@app.get("/document-status/{doc_hash}")
async def get_document_status(doc_hash: str, token: str = Depends(verify_token)):
    try:
        exists = await check_document_exists_in_pinecone(doc_hash)
        
        if exists:
            # Get document stats from Pinecone
            query_result = index.query(
                vector=[0.0] * 768,
                filter={"document_hash": doc_hash},
                top_k=100,  # Get more to count total chunks
                include_metadata=True
            )
            
            chunk_count = len(query_result['matches'])
            creation_date = None
            url = None
            
            if query_result['matches']:
                first_match = query_result['matches'][0]
                if first_match['metadata']:
                    creation_date = first_match['metadata'].get('created_at')
                    url = first_match['metadata'].get('url')
            
            return {
                "document_hash": doc_hash,
                "exists": True,
                "chunk_count": chunk_count,
                "created_at": creation_date,
                "url": url
            }
        else:
            return {
                "document_hash": doc_hash,
                "exists": False
            }
            
    except Exception as e:
        raise HTTPException(500, f"Error checking document status: {str(e)}")

# Endpoint to delete document from Pinecone (optional)
@app.delete("/document/{doc_hash}")
async def delete_document(doc_hash: str, token: str = Depends(verify_token)):
    try:
        # Delete all vectors for this document
        index.delete(filter={"document_hash": doc_hash})
        
        return {
            "message": f"Document {doc_hash} deleted successfully",
            "deleted_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error deleting document: {str(e)}")

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")
    logger.info("Using Gemini API for embeddings + Pinecone for storage")
    logger.info(f"Pinecone index: {PINECONE_INDEX_NAME}")
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
