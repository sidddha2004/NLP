# main.py - Fixed Railway deployment for Insurance Policy RAG API

import os
from typing import List
import time
import hashlib
import traceback
import logging
import asyncio

# Add the import for Sentence Transformers
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Gemini", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
gemini_model = None
sentence_transformer_model = None
pc = None
index = None
initialization_status = {
    "gemini": False,
    "sentence_transformer": False,
    "pinecone": False,
    "document": False,
    "document_processing": False,
    "error": None
}

# Initialize AI models and services
def initialize_gemini():
    global gemini_model, initialization_status
    
    try:
        if gemini_model is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized successfully")
            initialization_status["gemini"] = True
        
        return gemini_model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        initialization_status["error"] = str(e)
        raise e

# New function to initialize Sentence Transformer
def initialize_sentence_transformer():
    global sentence_transformer_model, initialization_status
    try:
        if sentence_transformer_model is None:
            model_name = "all-MiniLM-L6-v2"
            sentence_transformer_model = SentenceTransformer(model_name)
            logger.info(f"Sentence Transformer model '{model_name}' initialized successfully")
            initialization_status["sentence_transformer"] = True
        return sentence_transformer_model
    except Exception as e:
        logger.error(f"Failed to initialize Sentence Transformer model: {e}")
        initialization_status["error"] = str(e)
        raise e

# Initialize Pinecone
def init_pinecone():
    global pc, index, initialization_status
    
    try:
        logger.info("🔧 Initializing Pinecone...")
        if pc is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            pc = Pinecone(api_key=api_key)
            logger.info("✓ Pinecone client initialized")
        
        if index is None:
            index_name = "policy-docs-gemini-st"
            
            # Check if index exists
            try:
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                logger.info(f"📋 Found existing indexes: {existing_indexes}")
                
                if index_name not in existing_indexes:
                    logger.info(f"🏗 Creating new index: {index_name}")
                    pc.create_index(
                        name=index_name,
                        dimension=384,  # Change dimension to 384 for all-MiniLM-L6-v2
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    logger.info(f"✓ Created new index: {index_name}")
                    
                    # Wait for index to be ready
                    logger.info("⏳ Waiting for index to be ready...")
                    max_retries = 30
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            if pc.describe_index(index_name).status['ready']:
                                break
                        except Exception as wait_error:
                            logger.warning(f"Waiting for index, attempt {retry_count}: {wait_error}")
                        time.sleep(2)
                        retry_count += 1
                        if retry_count % 5 == 0:
                            logger.info(f"Still waiting... ({retry_count}/{max_retries})")
                    
                    if retry_count >= max_retries:
                        raise Exception("Index creation timeout")
                    logger.info("✓ Index is ready!")
                else:
                    logger.info(f"✓ Using existing index: {index_name}")
                
                index = pc.Index(index_name)
                logger.info("✓ Connected to index")
                initialization_status["pinecone"] = True
                
            except Exception as e:
                logger.error(f"❌ Error with Pinecone index: {e}")
                raise e
        
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        initialization_status["error"] = str(e)
        raise e

# Text extraction functions
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise e

def extract_text_from_docx(docx_path: str) -> str:
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise e

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported document format: {ext}")

# Text chunking
def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

def get_sentence_transformer_embedding(text: str) -> List[float]:
    """Create a semantic embedding using a Sentence Transformer model"""
    global sentence_transformer_model
    try:
        if sentence_transformer_model is None:
            raise RuntimeError("Sentence Transformer model is not initialized")
        embedding = sentence_transformer_model.encode(text).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error creating Sentence Transformer embedding: {e}")
        return []

def query_gemini(question: str, context_clauses: List[str], gemini_model) -> str:
    prompt = f"""
You are an expert assistant who answers insurance policy questions precisely and cites the clauses.

Question: {question}

Use ONLY the following clauses and explicitly mention or quote them in your answer:

{chr(10).join([f"- {clause}" for clause in context_clauses])}

Answer:
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return f"Error generating response: {str(e)}"

# Pinecone operations
def query_chunks(query: str, index, top_k: int = 5) -> List[str]:
    try:
        # Use the Sentence Transformer embedding function
        query_embedding = get_sentence_transformer_embedding(query)
        
        if not query_embedding:
            return []
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata.get("text", "") for match in query_response.matches]
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

# Pinecone upsert functions - now with batching
async def upsert_chunks_async(chunks: List[str], index):
    """Async version with rate limiting and batching"""
    try:
        vectors = []
        batch_size = 10
        
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = []
            
            for j, chunk in enumerate(batch_chunks):
                try:
                    # Use the Sentence Transformer embedding function
                    embedding = get_sentence_transformer_embedding(chunk)
                    
                    if embedding:
                        batch_vectors.append({
                            "id": f"chunk-{i+j}-{int(time.time())}",
                            "values": embedding,
                            "metadata": {"text": chunk}
                        })
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk {i+j}: {chunk_error}")
                    continue
            
            # Upsert this batch
            if batch_vectors:
                try:
                    index.upsert(vectors=batch_vectors)
                    logger.info(f"✓ Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_vectors)} vectors)")
                except Exception as upsert_error:
                    logger.error(f"Error upserting batch: {upsert_error}")
            
            # Small delay between batches to avoid overwhelming the API
            await asyncio.sleep(0.5)
        
        logger.info(f"✓ Completed upserting all chunks")
        
    except Exception as e:
        logger.error(f"Error in async upsert: {e}")
        raise e

# Background task for document processing
async def process_documents_background():
    """Process documents in the background after startup"""
    global index, gemini_model, initialization_status
    
    try:
        logger.info("🔄 Starting background document processing...")
        initialization_status["document_processing"] = True
        
        policy_file = "policy.pdf"
        
        # Debug: List all files in current directory
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        if not os.path.exists(policy_file):
            logger.error(f"Policy file {policy_file} not found in {os.getcwd()}")
            logger.error(f"Available files: {os.listdir('.')}")
            initialization_status["document_processing"] = False
            return
        
        # Check file permissions and size
        file_stat = os.stat(policy_file)
        logger.info(f"Policy file found - Size: {file_stat.st_size} bytes, Permissions: {oct(file_stat.st_mode)}")
        
        logger.info(f"Loading policy document: {policy_file}")
        
        # Extract text from the policy document
        full_text = extract_text(policy_file)
        logger.info(f"Extracted {len(full_text)} characters from policy document")
        
        # Chunk the text
        chunks = chunk_text(full_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Check if index has any vectors before trying to delete
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Clearing existing vectors from index...")
                index.delete(delete_all=True)
                logger.info("Cleared existing vectors from index")
                await asyncio.sleep(5)  # Wait a bit after clearing
            else:
                logger.info("Index is empty, no need to clear")
        except Exception as e:
            logger.warning(f"Could not check/clear index stats: {e}")
            logger.info("Proceeding with upsert...")
        
        # Upsert chunks to Pinecone (async with rate limiting)
        await upsert_chunks_async(chunks, index)
        logger.info("✅ Policy document successfully indexed in background")
        initialization_status["document"] = True
        initialization_status["document_processing"] = False
        
    except Exception as e:
        logger.error(f"❌ Error in background document processing: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        initialization_status["document_processing"] = False

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not expected_token:
        logger.error("API_BEARER_TOKEN not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if token != expected_token:
        logger.warning(f"Invalid token received")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid authentication token"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# FIXED: Non-blocking startup event
@app.on_event("startup")
async def startup_event():
    global gemini_model, pc, index, sentence_transformer_model, initialization_status
    
    logger.info("=== STARTUP: Initializing core services ===")
    
    try:
        # Initialize Gemini (fast)
        gemini_model = initialize_gemini()
        logger.info("✓ Gemini initialized successfully")
        
        # Initialize Sentence Transformer (download model)
        sentence_transformer_model = initialize_sentence_transformer()
        logger.info("✓ Sentence Transformer initialized successfully")
        
        # Initialize Pinecone (fast)
        pc, index = init_pinecone()
        logger.info("✓ Pinecone initialized successfully")
        
        logger.info("=== CORE STARTUP COMPLETE ===")
        logger.info("🚀 Server is ready to accept requests")
        
        # Start document processing in background (non-blocking)
        logger.info("🔄 Starting document processing in background...")
        asyncio.create_task(process_documents_background())
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        logger.warning("⚠ Continuing with limited functionality...")

# API endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    # Check if services are initialized
    global gemini_model, pc, index
    
    if not gemini_model or not index:
        logger.error("Services not properly initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not properly initialized. Status: {initialization_status}"
        )
    
    # Check if document is still processing
    if initialization_status["document_processing"]:
        logger.info("Document still processing, but can answer with available data")
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            relevant_clauses = query_chunks(question, index)
            
            if not relevant_clauses:
                if initialization_status["document_processing"]:
                    answers.append("Document is still being processed. Please try again in a few moments.")
                elif not initialization_status["document"]:
                    answers.append("Policy document not yet available. Please contact support.")
                else:
                    answers.append("No relevant information found in the policy document for this question.")
                continue
            
            answer = query_gemini(question, relevant_clauses, gemini_model)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    """Health check endpoint - always responds quickly"""
    return {
        "status": "healthy", 
        "message": "Insurance Policy RAG API with Gemini is running",
        "initialization_status": initialization_status,
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
            "has_pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
            "has_bearer_token": bool(os.getenv("API_BEARER_TOKEN"))
        }
    }

@app.get("/status")
async def get_detailed_status():
    """Detailed status endpoint"""
    global index
    try:
        status_info = {
            "services": initialization_status,
            "server_ready": bool(gemini_model and index and sentence_transformer_model),
            "document_ready": initialization_status["document"],
            "document_processing": initialization_status["document_processing"]
        }
        
        if index:
            try:
                stats = index.describe_index_stats()
                status_info["vector_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "index_fullness": stats.index_fullness
                }
            except Exception as stats_error:
                status_info["vector_stats"] = {"error": str(stats_error)}
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting detailed status: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/info")
async def get_info():
    global index
    try:
        if not index:
            return {"status": "error", "message": "Index not initialized"}
            
        stats = index.describe_index_stats()
        return {
            "status": "ready",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gemini-1.5-flash",
            "initialization_status": initialization_status
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Insurance Policy RAG API with Gemini",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "info": "/info",
            "query": "/hackrx/run (POST)"
        }
    }

# For Railway deployment - run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
