# main.py - Insurance Policy RAG API with Gemini 1.5 Flash and Enhanced Prompting

import os
from typing import List
import time
import hashlib
import traceback
import logging
import asyncio

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
app = FastAPI(title="Insurance Policy RAG API with Gemini 1.5 Flash", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
gemini_model = None
pc = None
index = None
initialization_status = {
    "gemini": False,
    "pinecone": False,
    "document": False,
    "document_processing": False,
    "error": None
}

# Initialize AI models and services
def initialize_services():
    global gemini_model, initialization_status
    
    try:
        if gemini_model is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Configure Gemini with 1.5 Flash model
            genai.configure(api_key=api_key)
            
            # Use Gemini 1.5 Flash - reliable and fast
            gemini_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent, factual responses
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1000,  # Limit output for concise answers
                    response_mime_type="text/plain",
                )
            )
            logger.info("Gemini 1.5 Flash model initialized successfully")
            initialization_status["gemini"] = True
        
        return gemini_model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini 1.5 Flash: {e}")
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
            index_name = "policy-docs-gemini-1-5-flash"
            
            # Check if index exists
            try:
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                logger.info(f"📋 Found existing indexes: {existing_indexes}")
                
                if index_name not in existing_indexes:
                    logger.info(f"🏗 Creating new index: {index_name}")
                    pc.create_index(
                        name=index_name,
                        dimension=768,  # Enhanced embedding dimensions
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

# Text chunking with enhanced parameters
def chunk_text(text: str, chunk_size=800, overlap=100) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# Enhanced embedding function using Gemini 1.5 Flash
def get_enhanced_embedding(text: str, gemini_model) -> List[float]:
    """Create enhanced embeddings using Gemini 1.5 Flash - EXACTLY 768 dimensions"""
    try:
        # Use Gemini 1.5 Flash to create semantic keywords and concepts
        embedding_prompt = f"""
Analyze this insurance policy text and extract exactly 60 key semantic concepts, keywords, and important phrases that capture the meaning. Focus on insurance terms, coverage details, conditions, and important clauses.

Text: {text[:1500]}

Provide exactly 60 items separated by commas, focusing on the most important insurance-related concepts:
"""
        
        response = gemini_model.generate_content(embedding_prompt)
        semantic_keywords = response.text.strip()
        
        # Create embedding from semantic keywords
        embedding = create_semantic_embedding(semantic_keywords, text)
        
        # Ensure exactly 768 dimensions
        if len(embedding) != 768:
            embedding = embedding[:768]
            while len(embedding) < 768:
                embedding.append(0.0)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error creating Gemini 1.5 Flash embedding: {e}")
        # Fallback to basic embedding
        return create_basic_embedding(text)

def create_semantic_embedding(keywords: str, original_text: str) -> List[float]:
    """Create semantic embedding from Gemini-generated keywords"""
    embedding = []
    
    # Parse keywords
    keyword_list = [k.strip().lower() for k in keywords.split(',')]
    
    # Method 1: Keyword-based features (384 dimensions)
    for i in range(384):
        if i < len(keyword_list):
            keyword = keyword_list[i]
            # Hash each keyword to create feature
            hash_val = hash(f"{keyword}_{i}") % 10000
            embedding.append((hash_val / 10000.0) * 2 - 1)  # Normalize to [-1, 1]
        else:
            embedding.append(0.0)
    
    # Method 2: Text statistics and patterns (384 dimensions)
    text_features = []
    words = original_text.lower().split()
    
    # Insurance-specific text statistics (first 20 features)
    text_features.extend([
        len(original_text) / 2000.0,
        len(words) / 200.0,
        len(set(words)) / max(len(words), 1),
        original_text.count('insurance') / max(len(words), 1),
        original_text.count('coverage') / max(len(words), 1),
        original_text.count('policy') / max(len(words), 1),
        original_text.count('claim') / max(len(words), 1),
        original_text.count('premium') / max(len(words), 1),
        original_text.count('deductible') / max(len(words), 1),
        original_text.count('benefit') / max(len(words), 1),
        original_text.count('exclusion') / max(len(words), 1),
        original_text.count('condition') / max(len(words), 1),
        original_text.count('limit') / max(len(words), 1),
        original_text.count('amount') / max(len(words), 1),
        original_text.count('$') / max(len(original_text), 1),
        original_text.count('%') / max(len(original_text), 1),
        original_text.count('.') / max(len(original_text), 1),
        sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1),
        sum(1 for c in original_text if c.isdigit()) / max(len(original_text), 1),
        len([w for w in words if len(w) > 8]) / max(len(words), 1)
    ])
    
    # Additional hash-based features (364 more features)
    for i in range(364):
        if i < len(words):
            word_hash = hash(f"{words[i]}_{i}_enhanced") % 10000
            text_features.append((word_hash / 10000.0) * 2 - 1)
        else:
            # Use character-level hashing if we run out of words
            char_idx = i % max(len(original_text), 1)
            char_hash = hash(f"{original_text[char_idx]}_{i}") % 10000
            text_features.append((char_hash / 10000.0) * 2 - 1)
    
    # Ensure exactly 384 text features
    text_features = text_features[:384]
    while len(text_features) < 384:
        text_features.append(0.0)
    
    embedding.extend(text_features)
    return embedding

def create_basic_embedding(text: str) -> List[float]:
    """Fallback basic embedding - 768 dimensions"""
    embeddings = []
    text = text.lower().strip()
    
    # Method 1: Hash-based features (384 dimensions)
    for i in range(24):
        hash_obj = hashlib.md5(f"{text}_{i}".encode())
        hash_bytes = hash_obj.digest()
        for byte_val in hash_bytes:
            if len(embeddings) < 384:
                normalized_val = (byte_val / 255.0) * 2 - 1
                embeddings.append(normalized_val)
    
    # Method 2: Word and text features (384 dimensions)
    words = text.split()
    for i in range(384):
        if i < len(words):
            word_hash = hash(f"{words[i]}_{i}") % 10000
            embeddings.append((word_hash / 10000.0) * 2 - 1)
        else:
            embeddings.append(0.0)
    
    return embeddings[:768]

# ENHANCED QUERY FUNCTION WITH DIRECT PROMPTING
def query_gemini_enhanced(question: str, context_clauses: List[str], gemini_model) -> str:
    """Enhanced query function with DIRECT ANSWER prompting for Gemini 1.5 Flash"""
    
    # Create context from relevant clauses
    context = "\n\n".join([f"CLAUSE {i+1}:\n{clause}" for i, clause in enumerate(context_clauses)])
    
    # ENHANCED PROMPT FOR DIRECT ANSWERS
    prompt = f"""You are a precise insurance policy assistant. Answer the user's question directly and concisely using ONLY the provided policy clauses.

INSTRUCTIONS FOR DIRECT ANSWERS:
- Give a direct, clear answer in 2-3 sentences maximum
- Quote specific policy text when relevant using quotation marks
- If coverage exists, state what is covered and any limits/conditions
- If not covered, state this clearly
- Do not add disclaimers or suggest consulting agents
- Be factual and specific
- Start with the answer, not with phrases like "Based on the policy"

QUESTION: {question}

RELEVANT POLICY CLAUSES:
{context}

DIRECT ANSWER:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
        # Clean up the answer to ensure it's direct
        answer = clean_answer(answer)
        return answer
        
    except Exception as e:
        logger.error(f"Error generating Gemini 1.5 Flash response: {e}")
        return f"Unable to process your question at this time. Please try again."

def clean_answer(answer: str) -> str:
    """Clean and format the answer to be more direct"""
    # Remove common filler phrases that make answers less direct
    filler_phrases = [
        "Based on the provided policy clauses,",
        "According to the policy document,",
        "The policy states that",
        "Please note that",
        "It's important to understand that",
        "You should be aware that",
        "Keep in mind that",
        "It should be noted that",
        "Based on the policy,",
        "According to the clauses,",
        "The document indicates that",
        "As stated in the policy,"
    ]
    
    cleaned = answer
    for phrase in filler_phrases:
        cleaned = cleaned.replace(phrase, "").strip()
    
    # Ensure answer starts with a capital letter
    if cleaned and not cleaned[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Remove excessive spacing
    cleaned = " ".join(cleaned.split())
    
    return cleaned

# Query chunks with enhanced embeddings
def query_chunks(query: str, index, gemini_model, top_k: int = 3) -> List[str]:
    """Query chunks using enhanced embeddings"""
    try:
        # Use enhanced embedding for better semantic matching
        query_embedding = get_enhanced_embedding(query, gemini_model)
        
        if not query_embedding:
            return []
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Filter and return relevant chunks
        relevant_chunks = []
        for match in query_response.matches:
            if match.score > 0.7:  # Only include highly relevant matches
                relevant_chunks.append(match.metadata.get("text", ""))
        
        return relevant_chunks if relevant_chunks else [match.metadata.get("text", "") for match in query_response.matches[:2]]
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

# Enhanced upsert functions
async def upsert_chunks_enhanced_async(chunks: List[str], index, gemini_model):
    """Enhanced async upsert with better embeddings using Gemini 1.5 Flash"""
    try:
        batch_size = 5  # Smaller batches for API rate limiting
        logger.info(f"Processing {len(chunks)} chunks with Gemini 1.5 Flash embeddings in batches of {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = []
            
            for j, chunk in enumerate(batch_chunks):
                try:
                    # Use enhanced embedding with Gemini 1.5 Flash
                    embedding = get_enhanced_embedding(chunk, gemini_model)
                    
                    if embedding is not None:
                        batch_vectors.append({
                            "id": f"gemini15-chunk-{i+j}-{int(time.time())}",
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
                    logger.info(f"✓ Upserted Gemini 1.5 Flash batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_vectors)} vectors)")
                except Exception as upsert_error:
                    logger.error(f"Error upserting batch: {upsert_error}")
            
            # Delay between batches for API rate limiting
            await asyncio.sleep(1.0)
        
        logger.info(f"✅ Completed upserting all chunks with Gemini 1.5 Flash embeddings")
        
    except Exception as e:
        logger.error(f"Error in enhanced async upsert: {e}")
        raise e

# Background task for document processing
async def process_documents_background():
    """Process documents in the background after startup"""
    global index, gemini_model, initialization_status
    
    try:
        logger.info("🔄 Starting enhanced background document processing with Gemini 1.5 Flash...")
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
        
        # Chunk the text with better parameters
        chunks = chunk_text(full_text, chunk_size=800, overlap=100)
        logger.info(f"Created {len(chunks)} enhanced chunks")
        
        # Clear existing vectors
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
        
        # Upsert chunks with enhanced embeddings using Gemini 1.5 Flash
        await upsert_chunks_enhanced_async(chunks, index, gemini_model)
        logger.info("✅ Policy document successfully indexed with Gemini 1.5 Flash enhanced embeddings")
        initialization_status["document"] = True
        initialization_status["document_processing"] = False
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced background document processing: {e}")
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

# Startup event
@app.on_event("startup")
async def startup_event():
    global gemini_model, pc, index, initialization_status
    
    logger.info("=== STARTUP: Initializing enhanced services with Gemini 1.5 Flash ===")
    
    try:
        # Initialize Gemini 1.5 Flash
        gemini_model = initialize_services()
        logger.info("✅ Gemini 1.5 Flash initialized successfully")
        
        # Initialize Pinecone
        pc, index = init_pinecone()
        logger.info("✅ Pinecone initialized successfully")
        
        logger.info("=== ENHANCED STARTUP COMPLETE ===")
        logger.info("🚀 Server is ready with Gemini 1.5 Flash and Enhanced Prompting")
        
        # Start enhanced document processing
        asyncio.create_task(process_documents_background())
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)

# Enhanced API endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    global gemini_model, pc, index
    
    if not gemini_model or not index:
        logger.error("Services not properly initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not properly initialized. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question with Gemini 1.5 Flash: {question[:100]}...")
            
            # Get relevant clauses using enhanced embeddings
            relevant_clauses = query_chunks(question, index, gemini_model, top_k=3)
            
            if not relevant_clauses:
                if initialization_status["document_processing"]:
                    answers.append("Document is still being processed. Please try again in a few moments.")
                elif not initialization_status["document"]:
                    answers.append("Policy document not yet available.")
                else:
                    answers.append("No relevant information found for this question.")
                continue
            
            # Generate enhanced answer with direct prompting
            answer = query_gemini_enhanced(question, relevant_clauses, gemini_model)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append("Unable to process your question. Please try again.")
    
    return QueryResponse(answers=answers)

# Health and status endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Insurance Policy RAG API with Gemini 1.5 Flash",
        "model": "gemini-1.5-flash",
        "features": ["Enhanced Prompting", "Direct Answers", "Clean Responses"],
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
    global index
    try:
        status_info = {
            "services": initialization_status,
            "model": "gemini-1.5-flash",
            "embedding_dimensions": 768,
            "server_ready": bool(gemini_model and index),
            "document_ready": initialization_status["document"],
            "document_processing": initialization_status["document_processing"],
            "features": {
                "enhanced_prompting": True,
                "direct_answers": True,
                "clean_responses": True,
                "semantic_embeddings": True
            }
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
            "embedding_model": "gemini-1.5-flash-enhanced-embeddings",
            "llm_model": "gemini-1.5-flash",
            "embedding_dimensions": 768,
            "prompt_version": "enhanced-direct-answers-v2",
            "initialization_status": initialization_status
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Enhanced Insurance Policy RAG API",
        "model": "Gemini 1.5 Flash (gemini-1.5-flash)",
        "version": "2.0.0",
        "features": [
            "Enhanced semantic embeddings with Gemini 1.5 Flash",
            "Direct answer prompting (2-3 sentences max)",
            "Improved context retrieval",
            "Clean, concise responses without filler",
            "Insurance-specific term recognition",
            "768-dimensional embeddings"
        ],
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "info": "/info",
            "query": "/hackrx/run (POST)"
        }
    }

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
