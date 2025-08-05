# main.py - Optimized Insurance Policy RAG API with Proper Embeddings
import os
from typing import List, Dict, Any
import time
import logging
import asyncio
import re

# Optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx

# Enhanced PDF processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, using PyPDF2 fallback")

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Proper embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.error("sentence-transformers not available - embeddings will be degraded")

# FastAPI setup
app = FastAPI(
    title="Optimized Insurance Policy RAG API",
    version="2.5.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None
)

# Optimized CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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

# Initialize embedding model (optimized)
def initialize_embedding_model():
    global embedding_model, initialization_status
    
    try:
        if EMBEDDINGS_AVAILABLE:
            # Use compact but effective model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Initialized sentence transformers with all-MiniLM-L6-v2")
            initialization_status["embeddings"] = True
            return embedding_model
        else:
            logger.error("sentence-transformers not available")
            initialization_status["embeddings"] = False
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
                max_output_tokens=1500,
                response_mime_type="text/plain",
            )
        )
        logger.info("✓ Gemini 1.5 Flash initialized")
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
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        pc = Pinecone(api_key=api_key)
        index_name = "optimized-policy-docs"
        
        # Use 384 dimensions for all-MiniLM-L6-v2
        embedding_dim = 384
        
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Wait for index readiness
            max_retries = 30
            for i in range(max_retries):
                try:
                    if pc.describe_index(index_name).status['ready']:
                        break
                except:
                    pass
                time.sleep(2)
                if i % 5 == 0:
                    logger.info(f"Waiting for index... ({i}/{max_retries})")
            else:
                raise Exception("Index creation timeout")
        
        index = pc.Index(index_name)
        initialization_status["pinecone"] = True
        logger.info("✓ Pinecone initialized")
        
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        initialization_status["error"] = str(e)
        raise e

# Enhanced PDF extraction
def extract_text_from_pdf(pdf_path: str) -> str:
    """Enhanced PDF text extraction"""
    try:
        if PDFPLUMBER_AVAILABLE:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
            return text
        else:
            # Fallback to PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise e

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise e

def extract_text(file_path: str) -> str:
    """Main text extraction function"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

# Smart text chunking
def smart_chunk_text(text: str, chunk_size=1000, overlap=150) -> List[str]:
    """Intelligent text chunking"""
    # Clean text
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle large paragraphs
            if len(paragraph) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk + sentence) <= chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev_words = chunks[i-1].split()[-overlap//10:]
                overlap_text = " ".join(prev_words)
                overlapped.append(overlap_text + " " + chunk)
        chunks = overlapped
    
    return chunks

# Proper embedding function
def get_embedding(text: str) -> List[float]:
    """Get proper embeddings"""
    try:
        if embedding_model and EMBEDDINGS_AVAILABLE:
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        else:
            # Fallback using Gemini for semantic extraction
            return get_gemini_enhanced_embedding(text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return get_gemini_enhanced_embedding(text)

def get_gemini_enhanced_embedding(text: str) -> List[float]:
    """Enhanced embedding using Gemini semantic analysis"""
    try:
        if gemini_model:
            prompt = f"""Extract exactly 50 key insurance terms, concepts, and important phrases from this text. Focus on coverage, limits, exclusions, conditions, and financial terms.

Text: {text[:1200]}

Return exactly 50 comma-separated terms:"""
            
            response = gemini_model.generate_content(prompt)
            semantic_terms = response.text.strip()
            
            # Create semantic embedding
            terms = [term.strip().lower() for term in semantic_terms.split(',')[:50]]
            embedding = []
            
            # Create 384-dimensional embedding
            for i in range(384):
                if i < len(terms):
                    # Use term-based hashing
                    term_hash = hash(f"{terms[i]}_insurance_{i}") % 10000
                    embedding.append((term_hash / 10000.0) * 2 - 1)
                else:
                    # Use text statistical features
                    feature_hash = hash(f"{text[i % len(text)]}_stat_{i}") % 10000
                    embedding.append((feature_hash / 10000.0) * 2 - 1)
            
            return embedding
    except Exception as e:
        logger.warning(f"Gemini embedding failed: {e}")
    
    # Final fallback
    return create_basic_embedding(text)

def create_basic_embedding(text: str) -> List[float]:
    """Basic embedding fallback"""
    embedding = []
    text_lower = text.lower()
    
    for i in range(384):
        combined_hash = hash(f"{text_lower}_{i}_basic") % 10000
        embedding.append((combined_hash / 10000.0) * 2 - 1)
    
    return embedding

# Query processing
def query_chunks(query: str, index, top_k: int = 4) -> List[Dict[str, Any]]:
    """Enhanced chunk querying"""
    try:
        query_embedding = get_embedding(query)
        
        if not query_embedding:
            return []
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Dynamic threshold
        matches = results.matches
        if not matches:
            return []
        
        max_score = max([m.score for m in matches])
        threshold = 0.6 if max_score > 0.8 else 0.4
        
        relevant_chunks = []
        for match in matches:
            if match.score > threshold:
                relevant_chunks.append({
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "id": match.id
                })
        
        # Return top chunks or fallback to best matches
        return relevant_chunks if relevant_chunks else [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "id": match.id
            }
            for match in matches[:2]
        ]
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

# Enhanced answer generation
def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate comprehensive answer"""
    
    if not context_chunks:
        return "I couldn't find relevant information in the policy document to answer your question."
    
    # Prepare context
    context_text = ""
    for i, chunk_data in enumerate(context_chunks[:3]):  # Use top 3 chunks
        context_text += f"\n--- POLICY SECTION {i+1} ---\n"
        context_text += chunk_data['text'] + "\n"
    
    prompt = f"""You are an expert insurance policy analyst. Answer the user's question using ONLY the provided policy sections.

INSTRUCTIONS:
- Provide specific, accurate answers based on the policy text
- Quote exact amounts, percentages, or limits when available
- If coverage limits exist, state them clearly
- If exclusions are mentioned, list them with bullet points
- If information is not in the provided sections, state this clearly
- Be comprehensive but concise

QUESTION: {question}

POLICY SECTIONS:
{context_text}

DETAILED ANSWER:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
        # Clean answer
        answer = re.sub(r'^(Based on the.*?[,:])\s*', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "I encountered an error while processing your question. Please try again."

# Document processing
async def upsert_chunks_async(chunks: List[str], index):
    """Efficient chunk upserting"""
    try:
        batch_size = 8  # Optimized batch size
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = []
            
            for j, chunk in enumerate(batch_chunks):
                try:
                    embedding = get_embedding(chunk)
                    
                    if embedding:
                        batch_vectors.append({
                            "id": f"chunk-{i+j}-{int(time.time())}",
                            "values": embedding,
                            "metadata": {
                                "text": chunk,
                                "chunk_index": i + j,
                                "timestamp": time.time()
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error processing chunk {i+j}: {e}")
                    continue
            
            if batch_vectors:
                try:
                    index.upsert(vectors=batch_vectors)
                    logger.info(f"✓ Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} uploaded ({len(batch_vectors)} vectors)")
                except Exception as e:
                    logger.error(f"Upsert failed: {e}")
            
            await asyncio.sleep(0.3)  # Rate limiting
        
        logger.info("✅ All chunks processed successfully")
        
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}")
        raise e

# Background document processing
async def process_documents_background():
    """Process policy document"""
    global initialization_status
    
    try:
        logger.info("🔄 Starting document processing...")
        initialization_status["document_processing"] = True
        
        policy_file = "policy.pdf"
        
        if not os.path.exists(policy_file):
            logger.error(f"Policy file not found: {policy_file}")
            initialization_status["document_processing"] = False
            return
        
        # Extract text
        full_text = extract_text(policy_file)
        logger.info(f"Extracted {len(full_text)} characters")
        
        # Analyze content
        logger.info(f"Content analysis:")
        logger.info(f"- Contains 'coverage': {full_text.lower().count('coverage')}")
        logger.info(f"- Contains 'limit': {full_text.lower().count('limit')}")
        logger.info(f"- Contains 'exclusion': {full_text.lower().count('exclusion')}")
        
        # Create chunks
        chunks = smart_chunk_text(full_text, chunk_size=1000, overlap=150)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Clear existing data
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Clearing existing vectors...")
                index.delete(delete_all=True)
                await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")
        
        # Process chunks
        await upsert_chunks_async(chunks, index)
        
        initialization_status["document"] = True
        initialization_status["document_processing"] = False
        logger.info("✅ Document processing completed")
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        initialization_status["error"] = str(e)
        initialization_status["document_processing"] = False

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected = os.getenv("API_BEARER_TOKEN")
    if not expected or credentials.credentials != expected:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    return True

# Models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Startup
@app.on_event("startup")
async def startup_event():
    global gemini_model, pc, index, embedding_model
    
    logger.info("=== OPTIMIZED STARTUP ===")
    
    try:
        # Initialize services
        embedding_model = initialize_embedding_model()
        gemini_model = initialize_gemini()
        pc, index = initialize_pinecone()
        
        logger.info("✅ All services initialized")
        
        # Start document processing
        asyncio.create_task(process_documents_background())
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        initialization_status["error"] = str(e)

# Main endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    
    if not all([gemini_model, index]):
        raise HTTPException(
            status_code=503,
            detail=f"Services not ready: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing: {question}")
            
            # Get relevant chunks
            relevant_chunks = query_chunks(question, index, top_k=4)
            
            if not relevant_chunks:
                if initialization_status["document_processing"]:
                    answers.append("Document is still being processed. Please try again shortly.")
                else:
                    answers.append("No relevant information found in the policy document.")
                continue
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append("An error occurred while processing your question.")
    
    return QueryResponse(answers=answers)

# Health and debug endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialization": initialization_status,
        "features": [
            "Enhanced PDF extraction",
            "Proper semantic embeddings",
            "Smart text chunking",
            "Optimized querying"
        ]
    }

@app.get("/debug/content")
async def debug_content():
    """Debug PDF content extraction"""
    try:
        if not os.path.exists("policy.pdf"):
            return {"error": "Policy file not found"}
        
        text = extract_text("policy.pdf")
        return {
            "text_length": len(text),
            "first_300": text[:300],
            "keyword_counts": {
                "coverage": text.lower().count("coverage"),
                "limit": text.lower().count("limit"),
                "exclusion": text.lower().count("exclusion"),
                "policy": text.lower().count("policy"),
                "premium": text.lower().count("premium")
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/vectors")
async def debug_vectors():
    """Debug vector storage"""
    try:
        if not index:
            return {"error": "Index not initialized"}
        
        stats = index.describe_index_stats()
        
        # Test query
        test_embedding = get_embedding("coverage limit")
        results = index.query(vector=test_embedding, top_k=3, include_metadata=True)
        
        return {
            "total_vectors": stats.total_vector_count,
            "sample_results": [
                {
                    "id": match.id,
                    "score": match.score,
                    "text_preview": match.metadata.get("text", "")[:200]
                }
                for match in results.matches
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Optimized Insurance Policy RAG API",
        "version": "2.5.0",
        "status": initialization_status,
        "endpoints": {
            "query": "/hackrx/run (POST)",
            "health": "/health",
            "debug": ["/debug/content", "/debug/vectors"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
