import os
import hashlib
import logging
import asyncio
import aiohttp
import io
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX Document Q&A System",
    description="RAG-based document Q&A with smart caching",
    version="1.0.0"
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    
    return response

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

# Global variables for models and clients
embedding_model = None
pc_client = None
index = None
executor = ThreadPoolExecutor(max_workers=8)  # Increased workers for better concurrency
model_lock = threading.Lock()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "028694bf504e52fe16bde850cea655c4d1fe6b7068383fdaf110d3e561e878b6")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

if not all([PINECONE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic models
class DocumentRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the PDF document")
    questions: List[str] = Field(..., min_items=1, max_items=10, description="List of questions")

class DocumentResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication"""
    if credentials.credentials != HACKRX_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

async def initialize_services():
    """Initialize all services asynchronously"""
    global embedding_model, pc_client, index
    
    try:
        logger.info("Initializing services...")
        
        # Initialize Pinecone with older API
        pinecone.init(api_key=PINECONE_API_KEY, environment='us-east-1-aws')
        
        # Check if index exists, create if not
        try:
            # For pinecone-client 2.2.4, use different approach
            existing_indexes = pinecone.list_indexes()
            
            if PINECONE_INDEX_NAME in existing_indexes:
                index = pinecone.Index(PINECONE_INDEX_NAME)
                logger.info(f"Connected to existing Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.warning(f"Index not found, creating new index: {PINECONE_INDEX_NAME}")
                # Create index with older API
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,
                    metric='cosine'
                )
                # Wait for index to be ready
                await asyncio.sleep(15)
                index = pinecone.Index(PINECONE_INDEX_NAME)
                logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
                
        except Exception as e:
            logger.error(f"Error with Pinecone index operations: {e}")
            # Try to connect anyway
            try:
                index = pinecone.Index(PINECONE_INDEX_NAME)
                logger.info(f"Connected to index despite list error")
            except Exception as inner_e:
                logger.error(f"Failed to connect to index: {inner_e}")
                raise RuntimeError(f"Could not connect to or create Pinecone index: {e}")
        
        # Initialize embedding model with retry logic - run in thread to avoid blocking
        with model_lock:
            if embedding_model is None:
                logger.info("Loading sentence transformer model...")
                max_retries = 2  # Reduced retries
                for attempt in range(max_retries):
                    try:
                        # Load model in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        embedding_model = await loop.run_in_executor(
                            executor, 
                            lambda: SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                        )
                        logger.info("Embedding model initialized successfully")
                        break
                    except Exception as model_error:
                        logger.warning(f"Model loading attempt {attempt + 1} failed: {model_error}")
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to load embedding model after {max_retries} attempts: {model_error}")
                        await asyncio.sleep(2)  # Reduced wait time
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def generate_document_hash(url: str) -> str:
    """Generate SHA-256 hash for document URL"""
    return hashlib.sha256(url.encode()).hexdigest()[:16]

async def download_pdf(url: str) -> bytes:
    """Download PDF from URL with timeout and error handling"""
    try:
        # Reduced timeout for faster failures
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            logger.info(f"Downloading PDF from: {str(url)[:100]}...")
            async with session.get(str(url)) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download PDF: HTTP {response.status}"
                    )
                
                content = await response.read()
                if len(content) == 0:
                    raise HTTPException(status_code=400, detail="Downloaded file is empty")
                
                logger.info(f"Downloaded PDF: {len(content)} bytes")
                return content
                
    except asyncio.TimeoutError:
        logger.error("PDF download timeout")
        raise HTTPException(status_code=408, detail="PDF download timeout")
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes with improved error handling and speed"""
    try:
        logger.info("Extracting text from PDF...")
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF has no pages")
        
        # Process pages in parallel for faster extraction
        text_parts = []
        max_pages = min(50, len(pdf_reader.pages))  # Limit to first 50 pages for speed
        
        for page_num in range(max_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            raise ValueError("No text could be extracted from PDF")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted text from {len(text_parts)} pages, {len(full_text)} characters")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    """Split text into overlapping chunks - optimized for speed"""
    words = text.split()
    if len(words) == 0:
        return [text]
    
    chunks = []
    step_size = chunk_size - overlap
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        
        if i + chunk_size >= len(words):
            break
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks if chunks else [text]

async def embed_chunks_async(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks asynchronously"""
    try:
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Run embedding generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def generate_embeddings():
            with model_lock:
                embeddings = embedding_model.encode(
                    chunks, 
                    convert_to_tensor=False,
                    show_progress_bar=False,  # Disable progress bar for speed
                    batch_size=32  # Process in batches
                )
            return embeddings
        
        embeddings = await loop.run_in_executor(executor, generate_embeddings)
        
        # Ensure we return a list of lists
        if len(embeddings.shape) == 1:
            result = [embeddings.tolist()]
        else:
            result = embeddings.tolist()
            
        logger.info(f"Generated {len(result)} embeddings")
        return result
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

def check_document_exists(doc_hash: str) -> bool:
    """Check if document embeddings already exist in Pinecone"""
    try:
        namespace = f"doc_{doc_hash}"
        # Updated for older Pinecone API
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        return namespace in namespaces and namespaces[namespace].get('vector_count', 0) > 0
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        return False

async def store_embeddings_async(doc_hash: str, chunks: List[str], embeddings: List[List[float]], url: str):
    """Store embeddings in Pinecone with metadata - async version"""
    try:
        if not chunks or not embeddings:
            raise ValueError("No chunks or embeddings to store")
            
        logger.info(f"Storing {len(embeddings)} embeddings...")
        namespace = f"doc_{doc_hash}"
        vectors = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{doc_hash}_{i}"
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': chunk,
                    'chunk_index': i,
                    'document_url': str(url),
                    'doc_hash': doc_hash,
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
        
        # Store embeddings in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def upsert_vectors():
            # Batch upsert in chunks of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
        
        await loop.run_in_executor(executor, upsert_vectors)
        
        logger.info(f"Stored {len(vectors)} embeddings for document {doc_hash}")
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to store document embeddings")

async def search_similar_chunks_async(question: str, doc_hash: str, top_k: int = 3) -> List[str]:
    """Search for similar chunks in Pinecone - async version"""
    try:
        logger.info(f"Searching for relevant chunks for question: {question[:50]}...")
        
        # Generate question embedding in executor
        loop = asyncio.get_event_loop()
        
        def generate_question_embedding():
            with model_lock:
                return embedding_model.encode([question], show_progress_bar=False)[0].tolist()
        
        question_embedding = await loop.run_in_executor(executor, generate_question_embedding)
        
        namespace = f"doc_{doc_hash}"
        
        # Search in Pinecone in executor
        def search_pinecone():
            return index.query(
                vector=question_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
        
        search_results = await loop.run_in_executor(executor, search_pinecone)
        
        # Extract text from results
        relevant_chunks = []
        for match in search_results.matches:
            if match.metadata and 'text' in match.metadata:
                relevant_chunks.append(match.metadata['text'])
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error searching similar chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to search document")

async def generate_answer_optimized(question: str, context: str) -> str:
    """Generate answer using Gemini with optimized prompt and settings"""
    try:
        # Use faster model and optimized settings
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Shorter, more direct prompt for faster processing
        prompt = f"""Answer this question based on the context provided. Be concise and direct.

Context: {context[:2000]}  

Question: {question}

Answer:"""

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def generate_content():
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,  # Reduced for faster response
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40
                )
            )
        
        response = await loop.run_in_executor(executor, generate_content)
        
        answer = response.text.strip()
        logger.info(f"Generated answer for question: {question[:30]}... (Length: {len(answer)})")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return "Sorry, I encountered an error while generating the answer. Please try again."

async def process_document_and_questions(url: str, questions: List[str]) -> List[str]:
    """Main processing pipeline - optimized for speed"""
    doc_hash = generate_document_hash(str(url))
    logger.info(f"Processing document hash: {doc_hash}")
    
    # Check if document already processed
    if not check_document_exists(doc_hash):
        logger.info(f"Processing new document: {doc_hash}")
        
        # Download and process document
        pdf_content = await download_pdf(str(url))
        text = extract_text_from_pdf(pdf_content)
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created from the document")
        
        # Generate and store embeddings asynchronously
        embeddings = await embed_chunks_async(chunks)
        await store_embeddings_async(doc_hash, chunks, embeddings, url)
        
        logger.info(f"Document {doc_hash} processed and cached")
    else:
        logger.info(f"Using cached document: {doc_hash}")
    
    # Process questions concurrently for faster results
    async def process_single_question(question: str) -> str:
        try:
            # Retrieve relevant chunks
            relevant_chunks = await search_similar_chunks_async(question, doc_hash, top_k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            # Combine chunks as context (limit size for faster processing)
            context = "\n\n".join(relevant_chunks)[:3000]  # Limit context size
            
            # Generate answer
            answer = await generate_answer_optimized(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions concurrently with limited concurrency
    semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
    
    async def process_with_semaphore(question: str) -> str:
        async with semaphore:
            return await process_single_question(question)
    
    # Execute all questions concurrently
    logger.info(f"Processing {len(questions)} questions concurrently...")
    tasks = [process_with_semaphore(q) for q in questions]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions in the results
    final_answers = []
    for i, answer in enumerate(answers):
        if isinstance(answer, Exception):
            logger.error(f"Error processing question {i}: {answer}")
            final_answers.append("An error occurred while processing this question.")
        else:
            final_answers.append(answer)
    
    logger.info("All questions processed successfully")
    return final_answers

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting HackRX Q&A System initialization...")
        await initialize_services()
        logger.info("HackRX Q&A System started successfully")
    except Exception as e:
        logger.error(f"Failed to start services: {e}", exc_info=True)
        # Don't raise here to allow the app to start and show health status

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=True)
    logger.info("HackRX Q&A System shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {
        "pinecone": "connected" if index is not None else "disconnected",
        "embedding_model": "loaded" if embedding_model is not None else "not_loaded",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured"
    }
    
    overall_status = "healthy" if all(
        status in ["connected", "loaded", "configured"] 
        for status in services_status.values()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRX Document Q&A System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs", 
            "main_endpoint": "/hackrx/run",
            "debug": "/debug"
        },
        "status": "running"
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    return {
        "embedding_model_loaded": embedding_model is not None,
        "pinecone_connected": index is not None,
        "gemini_configured": bool(GEMINI_API_KEY),
        "environment": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "bearer_token_set": bool(HACKRX_BEARER_TOKEN)
        },
        "performance": {
            "thread_pool_workers": executor._max_workers if executor else 0,
            "model_device": "cpu"  # We're forcing CPU for consistency
        }
    }

@app.post("/hackrx/run-single", response_model=Dict)
async def process_single_question(
    request: Dict,
    token: str = Depends(verify_token)
) -> Dict:
    """Test endpoint for processing a single question - useful for debugging performance"""
    try:
        document_url = request.get("documents")
        question = request.get("question")
        
        if not document_url or not question:
            raise HTTPException(status_code=400, detail="Both 'documents' and 'question' are required")
        
        logger.info(f"Processing single question: {question[:50]}...")
        start_time = datetime.utcnow()
        
        # Process just one question
        answers = await process_document_and_questions(document_url, [question])
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "question": question,
            "answer": answers[0] if answers else "No answer generated",
            "processing_time_seconds": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error in single question processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test", response_model=Dict)
async def test_endpoint():
    """Test endpoint without authentication for debugging"""
    return {
        "status": "success",
        "message": "Test endpoint is working",
        "services": {
            "embedding_model": embedding_model is not None,
            "pinecone": index is not None,
            "gemini": bool(GEMINI_API_KEY)
        }
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(
    request: DocumentRequest,
    token: str = Depends(verify_token)
) -> DocumentResponse:
    """Main endpoint for document Q&A processing"""
    try:
        logger.info(f"Received request: {len(request.questions)} questions for document")
        logger.info(f"Document URL: {str(request.documents)[:100]}...")
        
        start_time = datetime.utcnow()
        
        # Validate that services are initialized
        if embedding_model is None:
            logger.error("Embedding model not initialized")
            raise HTTPException(
                status_code=503,
                detail="Embedding model not initialized. Check logs for details."
            )
            
        if index is None:
            logger.error("Pinecone index not initialized")
            raise HTTPException(
                status_code=503,
                detail="Vector database not initialized. Check logs for details."
            )
        
        # Process document and questions
        answers = await process_document_and_questions(request.documents, request.questions)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
        
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /hackrx/run: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting HackRX Q&A System on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
