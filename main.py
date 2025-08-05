# main.py - Enhanced Insurance Policy RAG API with Gemini 1.5 Flash

import os
from typing import List, Dict, Any
import time
import hashlib
import traceback
import logging
import asyncio
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("✓ pdfplumber available for enhanced PDF extraction")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("⚠ pdfplumber not available, using PyPDF2 fallback")

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Sentence transformers for proper embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("✓ sentence-transformers available for proper embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠ sentence-transformers not available, using fallback embeddings")

# FastAPI setup
app = FastAPI(title="Enhanced Insurance Policy RAG API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
gemini_model = None
pc = None
index = None
embedding_model = None
initialization_status = {
    "gemini": False,
    "pinecone": False,
    "embeddings": False,
    "document": False,
    "document_processing": False,
    "error": None
}

# Initialize embedding model
def initialize_embedding_model():
    global embedding_model, initialization_status
    
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a good insurance/legal domain model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Initialized sentence-transformers embedding model")
            initialization_status["embeddings"] = True
            return embedding_model
        else:
            logger.warning("⚠ Using fallback embedding method")
            initialization_status["embeddings"] = True
            return None
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        initialization_status["error"] = str(e)
        return None

# Initialize Gemini
def initialize_gemini():
    global gemini_model, initialization_status
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2000,  # Increased for better responses
                response_mime_type="text/plain",
            )
        )
        logger.info("✓ Gemini 1.5 Flash model initialized successfully")
        initialization_status["gemini"] = True
        return gemini_model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        initialization_status["error"] = str(e)
        raise e

# Initialize Pinecone
def initialize_pinecone():
    global pc, index, initialization_status
    
    try:
        logger.info("🔧 Initializing Pinecone...")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        pc = Pinecone(api_key=api_key)
        logger.info("✓ Pinecone client initialized")
        
        index_name = "enhanced-policy-docs"
        
        # Determine embedding dimensions
        embedding_dim = 384 if SENTENCE_TRANSFORMERS_AVAILABLE else 768
        
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        logger.info(f"📋 Found existing indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            logger.info(f"🏗 Creating new index: {index_name} with {embedding_dim} dimensions")
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"✓ Created new index: {index_name}")
            
            # Wait for index to be ready
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if pc.describe_index(index_name).status['ready']:
                        break
                except Exception:
                    pass
                time.sleep(2)
                retry_count += 1
                if retry_count % 5 == 0:
                    logger.info(f"Still waiting for index... ({retry_count}/{max_retries})")
            
            if retry_count >= max_retries:
                raise Exception("Index creation timeout")
            logger.info("✓ Index is ready!")
        else:
            logger.info(f"✓ Using existing index: {index_name}")
        
        index = pc.Index(index_name)
        logger.info("✓ Connected to index")
        initialization_status["pinecone"] = True
        
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        initialization_status["error"] = str(e)
        raise e

# Enhanced PDF text extraction
def extract_text_from_pdf_enhanced(pdf_path: str) -> str:
    """Enhanced PDF extraction with pdfplumber fallback to PyPDF2"""
    try:
        if PDFPLUMBER_AVAILABLE:
            logger.info("Using pdfplumber for PDF extraction")
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"Error extracting page {page_num + 1}: {page_error}")
                        continue
            return text
        else:
            logger.info("Using PyPDF2 for PDF extraction")
            return extract_text_from_pdf_fallback(pdf_path)
    except Exception as e:
        logger.error(f"Error with enhanced PDF extraction: {e}")
        return extract_text_from_pdf_fallback(pdf_path)

def extract_text_from_pdf_fallback(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2"""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as page_error:
                    logger.warning(f"Error extracting page {page_num + 1}: {page_error}")
                    continue
        return text
    except Exception as e:
        logger.error(f"Error with PyPDF2 extraction: {e}")
        raise e

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for para_num, para in enumerate(doc.paragraphs):
            if para.text.strip():
                text += f"{para.text}\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise e

def extract_text(file_path: str) -> str:
    """Main text extraction function"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf_enhanced(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported document format: {ext}")

# Improved text chunking
def smart_chunk_text(text: str, chunk_size=1200, overlap=200) -> List[str]:
    """Intelligent text chunking that respects sentence and paragraph boundaries"""
    
    # Clean the text
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r' +', ' ', text)  # Normalize spaces
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size
        if len(current_chunk + paragraph) > chunk_size and current_chunk:
            # Try to split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
            
            # Add complete sentences to chunk
            chunk_text = ""
            for sentence in sentences:
                if len(chunk_text + sentence) <= chunk_size:
                    chunk_text += sentence + " "
                else:
                    if chunk_text:
                        chunks.append(chunk_text.strip())
                    chunk_text = sentence + " "
            
            if chunk_text:
                chunks.append(chunk_text.strip())
            
            # Start new chunk with current paragraph
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add remaining content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_words = chunks[i-1].split()
                overlap_words = prev_words[-overlap//10:] if len(prev_words) > overlap//10 else prev_words
                overlap_text = " ".join(overlap_words)
                overlapped_chunks.append(overlap_text + " " + chunk)
        chunks = overlapped_chunks
    
    logger.info(f"Created {len(chunks)} smart chunks from text")
    return chunks

# Proper embedding functions
def get_embedding(text: str) -> List[float]:
    """Get embedding using sentence-transformers or fallback"""
    global embedding_model
    
    try:
        if embedding_model is not None and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use proper semantic embeddings
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        else:
            # Use improved fallback
            return get_fallback_embedding(text)
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return get_fallback_embedding(text)

def get_fallback_embedding(text: str) -> List[float]:
    """Improved fallback embedding function"""
    # Use Gemini to extract semantic features
    try:
        if gemini_model:
            prompt = f"""Extract 50 key insurance terms, concepts, and important phrases from this text. Focus on coverage details, limits, conditions, exclusions, and financial terms.

Text: {text[:1000]}

Return exactly 50 comma-separated terms:"""
            
            response = gemini_model.generate_content(prompt)
            semantic_terms = response.text.strip()
            
            # Create embedding from semantic terms
            embedding = []
            terms = [term.strip().lower() for term in semantic_terms.split(',')]
            
            # Create hash-based features from semantic terms
            for i in range(768):
                if i < len(terms):
                    term_hash = hash(f"{terms[i]}_{i}_semantic") % 10000
                    embedding.append((term_hash / 10000.0) * 2 - 1)
                else:
                    # Use text statistics for remaining dimensions
                    char_idx = i % max(len(text), 1)
                    char_hash = hash(f"{text[char_idx]}_{i}") % 10000
                    embedding.append((char_hash / 10000.0) * 2 - 1)
            
            return embedding[:768]
    except Exception as e:
        logger.warning(f"Gemini fallback embedding failed: {e}")
    
    # Basic fallback
    return create_basic_embedding(text)

def create_basic_embedding(text: str) -> List[float]:
    """Basic embedding as last resort"""
    text_lower = text.lower()
    embedding = []
    
    # Create features based on text characteristics
    for i in range(768):
        # Use multiple hash functions for better distribution
        hash1 = hash(f"{text_lower}_{i}_v1") % 10000
        hash2 = hash(f"{text_lower}_{i}_v2") % 10000
        combined = (hash1 + hash2) / 20000.0 * 2 - 1
        embedding.append(combined)
    
    return embedding

# Enhanced query processing
def query_chunks(query: str, index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query chunks with improved similarity handling"""
    try:
        query_embedding = get_embedding(query)
        
        if not query_embedding:
            return []
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Dynamic threshold based on score distribution
        matches = query_response.matches
        if not matches:
            return []
        
        scores = [match.score for match in matches]
        max_score = max(scores)
        
        # Use adaptive threshold
        if max_score > 0.8:
            threshold = 0.7
        elif max_score > 0.6:
            threshold = 0.5
        else:
            threshold = 0.3
        
        relevant_chunks = []
        for match in matches:
            if match.score > threshold:
                relevant_chunks.append({
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "id": match.id
                })
        
        # If no chunks meet threshold, return top 2
        if not relevant_chunks:
            relevant_chunks = [
                {
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "id": match.id
                }
                for match in matches[:2]
            ]
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for query")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

# Enhanced Gemini querying
def query_gemini_enhanced(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Enhanced query function with better prompting"""
    
    if not context_chunks:
        return "I couldn't find relevant information in the policy document to answer your question."
    
    # Prepare context with chunk information
    context_text = ""
    for i, chunk_data in enumerate(context_chunks):
        context_text += f"\n--- POLICY SECTION {i+1} (Relevance: {chunk_data['score']:.2f}) ---\n"
        context_text += chunk_data['text'] + "\n"
    
    # Enhanced prompt for better answers
    prompt = f"""You are an expert insurance policy analyst. Analyze the provided policy sections and answer the user's question directly and accurately.

INSTRUCTIONS:
- Provide a clear, specific answer based ONLY on the policy sections provided
- If you find specific coverage limits, exclusions, or conditions, quote them exactly
- If the information isn't in the provided sections, say so clearly
- Use bullet points for multiple items (like exclusions or coverage types)
- Include specific dollar amounts, percentages, or limits when mentioned
- Be concise but complete

USER QUESTION: {question}

POLICY SECTIONS:
{context_text}

ANSWER:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
        # Clean up the response
        answer = clean_and_format_answer(answer)
        
        logger.info(f"Generated answer length: {len(answer)} characters")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "I encountered an error while processing your question. Please try again."

def clean_and_format_answer(answer: str) -> str:
    """Clean and format the answer for better presentation"""
    
    # Remove redundant phrases
    cleanup_phrases = [
        "Based on the provided policy sections,",
        "According to the policy document,",
        "The policy states that",
        "Based on the policy sections provided,",
        "From the policy sections,",
        "According to the sections provided,"
    ]
    
    cleaned = answer
    for phrase in cleanup_phrases:
        cleaned = cleaned.replace(phrase, "").strip()
    
    # Ensure proper capitalization
    if cleaned and not cleaned[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    
    return cleaned

# Enhanced document processing
async def upsert_chunks_async(chunks: List[str], index):
    """Async upsert with proper embeddings"""
    try:
        batch_size = 10
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = []
            
            for j, chunk in enumerate(batch_chunks):
                try:
                    embedding = get_embedding(chunk)
                    
                    if embedding:
                        batch_vectors.append({
                            "id": f"enhanced-chunk-{i+j}-{int(time.time())}",
                            "values": embedding,
                            "metadata": {
                                "text": chunk,
                                "chunk_index": i + j,
                                "timestamp": time.time()
                            }
                        })
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk {i+j}: {chunk_error}")
                    continue
            
            if batch_vectors:
                try:
                    index.upsert(vectors=batch_vectors)
                    logger.info(f"✓ Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_vectors)} vectors)")
                except Exception as upsert_error:
                    logger.error(f"Error upserting batch: {upsert_error}")
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        logger.info("✅ Completed upserting all chunks")
        
    except Exception as e:
        logger.error(f"Error in async upsert: {e}")
        raise e

# Enhanced background document processing
async def process_documents_background():
    """Enhanced background document processing"""
    global index, gemini_model, initialization_status
    
    try:
        logger.info("🔄 Starting enhanced background document processing...")
        initialization_status["document_processing"] = True
        
        policy_file = "policy.pdf"
        
        if not os.path.exists(policy_file):
            logger.error(f"Policy file {policy_file} not found")
            # List available files for debugging
            available_files = [f for f in os.listdir('.') if f.endswith(('.pdf', '.docx'))]
            logger.info(f"Available document files: {available_files}")
            initialization_status["document_processing"] = False
            return
        
        logger.info(f"Processing policy document: {policy_file}")
        
        # Extract text
        full_text = extract_text(policy_file)
        logger.info(f"Extracted {len(full_text)} characters from {policy_file}")
        
        # Log some content for debugging
        logger.info(f"First 300 characters: {full_text[:300]}")
        logger.info(f"Contains 'coverage': {'coverage' in full_text.lower()}")
        logger.info(f"Contains 'limit': {'limit' in full_text.lower()}")
        logger.info(f"Contains 'exclusion': {'exclusion' in full_text.lower()}")
        
        # Smart chunking
        chunks = smart_chunk_text(full_text, chunk_size=1200, overlap=200)
        logger.info(f"Created {len(chunks)} smart chunks")
        
        # Clear existing vectors
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Clearing existing vectors...")
                index.delete(delete_all=True)
                await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")
        
        # Upsert chunks
        await upsert_chunks_async(chunks, index)
        
        logger.info("✅ Document processing completed successfully")
        initialization_status["document"] = True
        initialization_status["document_processing"] = False
        
    except Exception as e:
        logger.error(f"❌ Error in document processing: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        initialization_status["document_processing"] = False

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if token != expected_token:
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
    global gemini_model, pc, index, embedding_model
    
    logger.info("=== STARTUP: Initializing Enhanced Services ===")
    
    try:
        # Initialize embedding model first
        embedding_model = initialize_embedding_model()
        
        # Initialize Gemini
        gemini_model = initialize_gemini()
        
        # Initialize Pinecone
        pc, index = initialize_pinecone()
        
        logger.info("=== STARTUP COMPLETE ===")
        logger.info("🚀 Enhanced server ready")
        
        # Start document processing
        asyncio.create_task(process_documents_background())
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)

# Main API endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    if not gemini_model or not index:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not initialized. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question}")
            
            # Get relevant chunks
            relevant_chunks = query_chunks(question, index, top_k=5)
            
            if not relevant_chunks:
                if initialization_status["document_processing"]:
                    answers.append("The document is still being processed. Please try again in a few moments.")
                elif not initialization_status["document"]:
                    answers.append("The policy document has not been loaded yet.")
                else:
                    answers.append("I couldn't find relevant information in the policy document to answer your question.")
                continue
            
            # Generate answer
            answer = query_gemini_enhanced(question, relevant_chunks)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            answers.append("I encountered an error while processing your question. Please try again.")
    
    return QueryResponse(answers=answers)

# Debug endpoints
@app.get("/debug/pdf-content")
async def debug_pdf_content():
    """Debug endpoint to check PDF extraction"""
    try:
        policy_file = "policy.pdf"
        if not os.path.exists(policy_file):
            return {"error": "Policy file not found", "available_files": os.listdir('.')}
        
        full_text = extract_text(policy_file)
        
        return {
            "file_exists": True,
            "text_length": len(full_text),
            "first_500_chars": full_text[:500],
            "last_500_chars": full_text[-500:] if len(full_text) > 500 else full_text,
            "word_count": len(full_text.split()),
            "contains_keywords": {
                "coverage": full_text.lower().count("coverage"),
                "limit": full_text.lower().count("limit"),
                "exclusion": full_text.lower().count("exclusion"),
                "policy": full_text.lower().count("policy"),
                "premium": full_text.lower().count("premium"),
                "deductible": full_text.lower().count("deductible")
            }
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/debug/vectors")
async def debug_vectors():
    """Debug vector storage"""
    try:
        if not index:
            return {"error": "Index not initialized"}
        
        stats = index.describe_index_stats()
        
        # Sample query to see stored content
        sample_embedding = get_embedding("coverage limit policy")
        sample_query = index.query(
            vector=sample_embedding,
            top_k=3,
            include_metadata=True
        )
        
        return {
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "fallback",
            "sample_chunks": [
                {
                    "id": match.id,
                    "score": match.score,
                    "text_preview": match.metadata.get("text", "")[:200] + "..."
                }
                for match in sample_query.matches
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/test-query")
async def debug_test_query(question: str = "What is the coverage limit?"):
    """Test query processing"""
    try:
        relevant_chunks = query_chunks(question, index, top_k=3)
        
        return {
            "question": question,
            "chunks_found": len(relevant_chunks),
            "chunks": [
                {
                    "score": chunk["score"],
                    "text_preview": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
                }
                for chunk in relevant_chunks
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Health endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "features": [
            "Enhanced PDF extraction with pdfplumber",
            "Proper semantic embeddings with sentence-transformers",
            "Smart text chunking",
            "Improved query processing",
            "Better Gemini prompting"
        ],
        "initialization_status": initialization_status,
        "dependencies": {
            "pdfplumber": PDFPLUMBER_AVAILABLE,
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE
        }
    }

@app.get("/status")
async def get_status():
    global index
    
    status_info = {
        "services": initialization_status,
        "server_ready": bool(gemini_model and index),
        "embedding_model": "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "fallback"
    }
    
    if index:
        try:
            stats = index.describe_index_stats()
            status_info["vector_stats"] = {
                "total_vectors": stats.total_vector_count,
                "index_fullness": stats.index_fullness
            }
        except:
            status_info["vector_stats"] = {"error": "Could not fetch stats"}
    
    return status_info

@app.get("/")
async def root():
    return {
        "message": "Enhanced Insurance Policy RAG API",
        "version": "3.0.0",
        "model": "Gemini 1.5 Flash",
        "embeddings": "sentence-transformers (all-MiniLM-L6-v2)" if SENTENCE_TRANSFORMERS_AVAILABLE else "fallback",
        "features": [
            "Enhanced PDF extraction",
            "Proper semantic embeddings", 
            "Smart text chunking",
            "Improved query processing",
            "Better answer generation"
        ],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "query": "/hackrx/run (POST)",
            "debug": ["/debug/pdf-content", "/debug/vectors", "/debug/test-query"]
        }
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
