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
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX Document Q&A System",
    description="RAG-based document Q&A with smart caching",
    version="1.0.0"
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

# Global variables for models and clients
embedding_model = None
pc_client = None
index = None
executor = ThreadPoolExecutor(max_workers=4)
model_lock = threading.Lock()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "028694bf504e52fe16bde850cea655c4d1fe6b7068383fdaf110d3e561e878b6")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

if not all([PINECONE_API_KEY, GEMINI_API_KEY, HACKRX_BEARER_TOKEN]):
    raise ValueError("Missing required environment variables")

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
        
        # Initialize Pinecone
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists, create if not
        try:
            index = pc_client.Index(PINECONE_INDEX_NAME)
            logger.info(f"Connected to existing Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.warning(f"Index not found, creating new index: {e}")
            pc_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            import time
            time.sleep(10)
            index = pc_client.Index(PINECONE_INDEX_NAME)
            logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Initialize embedding model with retry logic
        with model_lock:
            if embedding_model is None:
                logger.info("Loading sentence transformer model...")
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                        logger.info("Embedding model initialized successfully")
                        break
                    except Exception as model_error:
                        logger.warning(f"Model loading attempt {attempt + 1} failed: {model_error}")
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to load embedding model after {max_retries} attempts: {model_error}")
                        await asyncio.sleep(5)
        
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
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(str(url)) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download PDF: HTTP {response.status}"
                    )
                
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and not str(url).lower().endswith('.pdf'):
                    logger.warning(f"Content-Type: {content_type}, URL: {url}")
                
                content = await response.read()
                if len(content) == 0:
                    raise HTTPException(status_code=400, detail="Downloaded file is empty")
                
                return content
                
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="PDF download timeout")
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF has no pages")
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        
        if i + chunk_size >= len(words):
            break
    
    return chunks if chunks else [text]  # Return original text if chunking fails

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks"""
    try:
        with model_lock:
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

def check_document_exists(doc_hash: str) -> bool:
    """Check if document embeddings already exist in Pinecone"""
    try:
        namespace = f"doc_{doc_hash}"
        stats = index.describe_index_stats()
        return namespace in stats.get('namespaces', {})
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        return False

def store_embeddings(doc_hash: str, chunks: List[str], embeddings: List[List[float]], url: str):
    """Store embeddings in Pinecone with metadata"""
    try:
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
        
        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        logger.info(f"Stored {len(vectors)} embeddings for document {doc_hash}")
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to store document embeddings")

def search_similar_chunks(question: str, doc_hash: str, top_k: int = 5) -> List[str]:
    """Search for similar chunks in Pinecone"""
    try:
        # Generate question embedding
        with model_lock:
            question_embedding = embedding_model.encode([question])[0].tolist()
        
        namespace = f"doc_{doc_hash}"
        
        # Search in Pinecone
        search_results = index.query(
            vector=question_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Extract text from results
        relevant_chunks = []
        for match in search_results.matches:
            if 'text' in match.metadata:
                relevant_chunks.append(match.metadata['text'])
        
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error searching similar chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to search document")

async def generate_answer(question: str, context: str) -> str:
    """Generate answer using Gemini 2.5 Flash Lite"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Based on the following context from a document, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Provide a direct, factual answer based only on the given context
- If the context doesn't contain enough information to answer the question, say "The provided document doesn't contain sufficient information to answer this question."
- Keep the answer concise but complete
- Do not make assumptions beyond what's stated in the context

Answer:"""

        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.1,
                )
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return "Sorry, I encountered an error while generating the answer. Please try again."

async def process_document_and_questions(url: str, questions: List[str]) -> List[str]:
    """Main processing pipeline"""
    doc_hash = generate_document_hash(str(url))
    
    # Check if document already processed
    if not check_document_exists(doc_hash):
        logger.info(f"Processing new document: {doc_hash}")
        
        # Download and process document
        pdf_content = await download_pdf(str(url))
        text = extract_text_from_pdf(pdf_content)
        chunks = chunk_text(text)
        
        # Generate and store embeddings
        embeddings = embed_chunks(chunks)
        store_embeddings(doc_hash, chunks, embeddings, url)
        
        logger.info(f"Document {doc_hash} processed and cached")
    else:
        logger.info(f"Using cached document: {doc_hash}")
    
    # Process questions
    answers = []
    for question in questions:
        try:
            # Retrieve relevant chunks
            relevant_chunks = search_similar_chunks(question, doc_hash)
            
            if not relevant_chunks:
                answers.append("No relevant information found in the document.")
                continue
            
            # Combine chunks as context
            context = "\n\n".join(relevant_chunks)
            
            # Generate answer
            answer = await generate_answer(question, context)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            answers.append("An error occurred while processing this question.")
    
    return answers

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_services()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {
        "pinecone": "connected" if index is not None else "disconnected",
        "embedding_model": "loaded" if embedding_model is not None else "not_loaded",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured"
    }
    
    return HealthResponse(
        status="healthy" if all(status == "connected" or status == "loaded" or status == "configured" 
                              for status in services_status.values()) else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )

@app.post("/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(
    request: DocumentRequest,
    token: str = Depends(verify_token)
) -> DocumentResponse:
    """Main endpoint for document Q&A processing"""
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        start_time = datetime.utcnow()
        
        # Process document and questions
        answers = await process_document_and_questions(request.documents, request.questions)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /hackrx/run: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRX Document Q&A System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
