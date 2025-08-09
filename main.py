import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import numpy as np
import re
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

def clean_text(text: str) -> str:
    """Clean and normalize extracted text for better processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'\.([A-Z])', r'. \1', text)  # Add space after periods
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # Add space between text and numbers
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # Add space between numbers and text
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
                    cleaned_text = clean_text(page_text)
                    if cleaned_text and len(cleaned_text) > 50:  # Only add substantial text
                        text_parts.append(cleaned_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            raise ValueError("No text could be extracted from PDF")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(400, f"Error extracting text from PDF: {str(e)}")

def create_semantic_chunks(text: str, target_chunk_size: int = 1000, min_chunk_size: int = 200) -> List[str]:
    """Create semantically meaningful chunks that preserve context"""
    if not text.strip():
        return []
    
    # First, split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        # Fallback to sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        paragraphs = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed target size
        if len(current_chunk) + len(paragraph) > target_chunk_size:
            # If current chunk has content, save it
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If paragraph itself is too long, split it
            if len(paragraph) > target_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > target_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Even single sentence is too long, force split
                            words = sentence.split()
                            for i in range(0, len(words), 150):
                                chunk_words = words[i:i + 200]
                                chunks.append(' '.join(chunk_words))
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk if it has content
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks and merge them with adjacent chunks
    filtered_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < min_chunk_size and i > 0:
            # Merge with previous chunk if it won't become too large
            if len(filtered_chunks[-1]) + len(chunk) < target_chunk_size * 1.5:
                filtered_chunks[-1] += "\n\n" + chunk
            else:
                filtered_chunks.append(chunk)
        else:
            filtered_chunks.append(chunk)
    
    return filtered_chunks if filtered_chunks else ([text] if text.strip() else [])

async def create_gemini_embedding(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """Create a single embedding using Gemini API with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Truncate text if too long for embedding
            if len(text) > 2000:
                text = text[:2000]
            
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
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"All embedding attempts failed for text: {text[:100]}...")
                return None
    
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
    """Store document chunks and their embeddings in Pinecone with better metadata"""
    try:
        logger.info(f"Processing {len(chunks)} chunks for Pinecone storage")
        
        # Process chunks in smaller batches to ensure quality
        batch_size = 5
        successful_uploads = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            vectors_to_upsert = []
            
            # Create embeddings for the batch
            for j, chunk in enumerate(batch_chunks):
                chunk_index = i + j
                embedding = await create_gemini_embedding(chunk, "retrieval_document")
                
                if embedding:
                    vector_id = generate_chunk_id(doc_hash, chunk_index)
                    
                    # Enhanced metadata for better retrieval
                    metadata = {
                        "document_hash": doc_hash,
                        "chunk_index": chunk_index,
                        "text": chunk,
                        "url": url,
                        "created_at": datetime.utcnow().isoformat(),
                        "chunk_length": len(chunk),
                        "word_count": len(chunk.split()),
                        # Add semantic indicators
                        "has_numbers": any(char.isdigit() for char in chunk),
                        "has_questions": '?' in chunk,
                        "chunk_type": "paragraph" if '\n' in chunk else "sentence"
                    }
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                    successful_uploads += 1
                else:
                    logger.warning(f"Failed to create embedding for chunk {chunk_index}")
                
                # Respect API rate limits
                await asyncio.sleep(0.15)
            
            # Upsert batch to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    logger.info(f"Upserted batch {(i//batch_size) + 1} to Pinecone ({len(vectors_to_upsert)} vectors)")
                except Exception as e:
                    logger.error(f"Failed to upsert batch {(i//batch_size) + 1}: {e}")
        
        logger.info(f"Successfully stored {successful_uploads} chunks in Pinecone for document {doc_hash}")
        return successful_uploads > 0
        
    except Exception as e:
        logger.error(f"Error storing document in Pinecone: {e}")
        return False

async def retrieve_relevant_chunks_from_pinecone(question: str, doc_hash: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks with enhanced filtering and scoring"""
    try:
        # Create embedding for the question
        question_embedding = await create_gemini_embedding(question, "retrieval_query")
        
        if not question_embedding:
            logger.warning("Failed to create question embedding")
            return []
        
        # Query Pinecone for similar chunks with higher top_k for better selection
        query_result = index.query(
            vector=question_embedding,
            filter={"document_hash": doc_hash},
            top_k=min(top_k * 2, 15),  # Get more candidates for filtering
            include_metadata=True
        )
        
        # Process and rank results
        candidates = []
        for match in query_result['matches']:
            if match['metadata'] and 'text' in match['metadata']:
                chunk_data = {
                    'text': match['metadata']['text'],
                    'score': match['score'],
                    'metadata': match['metadata'],
                    'chunk_index': match['metadata'].get('chunk_index', 0)
                }
                candidates.append(chunk_data)
        
        # Enhanced filtering and ranking
        question_lower = question.lower()
        
        # Boost scores based on question relevance
        for candidate in candidates:
            text_lower = candidate['text'].lower()
            
            # Boost if chunk contains question keywords
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            text_words = set(re.findall(r'\b\w+\b', text_lower))
            overlap = len(question_words.intersection(text_words))
            keyword_boost = min(overlap * 0.1, 0.3)
            
            # Boost for specific question types
            if '?' in question:
                if candidate['metadata'].get('has_questions', False):
                    keyword_boost += 0.1
            
            if any(char.isdigit() for char in question):
                if candidate['metadata'].get('has_numbers', False):
                    keyword_boost += 0.1
            
            # Apply boost
            candidate['adjusted_score'] = candidate['score'] + keyword_boost
        
        # Sort by adjusted score and take top_k
        candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)
        selected_candidates = candidates[:top_k]
        
        # Ensure chunks are in logical order for context
        selected_candidates.sort(key=lambda x: x['chunk_index'])
        
        logger.info(f"Retrieved {len(selected_candidates)} relevant chunks from Pinecone")
        for i, candidate in enumerate(selected_candidates):
            logger.debug(f"Chunk {i+1}: score={candidate['score']:.4f}, adjusted={candidate['adjusted_score']:.4f}")
        
        return selected_candidates
        
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {e}")
        return []

def analyze_question_type(question: str) -> Dict[str, Any]:
    """Analyze question to provide better context for answer generation"""
    question_lower = question.lower().strip()
    
    analysis = {
        'is_yes_no': any(question_lower.startswith(word) for word in ['is', 'are', 'can', 'does', 'do', 'will', 'would', 'should', 'could']),
        'is_what': question_lower.startswith(('what', 'which')),
        'is_how': question_lower.startswith('how'),
        'is_why': question_lower.startswith('why'),
        'is_when': question_lower.startswith('when'),
        'is_where': question_lower.startswith('where'),
        'is_who': question_lower.startswith('who'),
        'asks_for_list': any(word in question_lower for word in ['list', 'types', 'kinds', 'categories', 'examples']),
        'asks_for_number': any(word in question_lower for word in ['how many', 'number of', 'count', 'quantity']),
        'asks_for_definition': any(word in question_lower for word in ['define', 'definition', 'meaning', 'what is', 'what are'])
    }
    
    return analysis

async def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    try:
        if not relevant_chunks:
            return "The document does not provide this information."
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Analyze question type for better prompting
        question_analysis = analyze_question_type(question)
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk_data in enumerate(relevant_chunks):
            context_parts.append(f"[Context {i+1}]: {chunk_data['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Limit context length but preserve important information
        if len(context) > 5000:
            # Prioritize highest scoring chunks
            sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('adjusted_score', x['score']), reverse=True)
            priority_context = []
            current_length = 0
            
            for chunk_data in sorted_chunks:
                chunk_text = f"[Context]: {chunk_data['text']}"
                if current_length + len(chunk_text) <= 4500:
                    priority_context.append(chunk_text)
                    current_length += len(chunk_text)
                else:
                    break
            
            context = "\n\n".join(priority_context)
        
        # Build dynamic prompt based on question type
        base_rules = """You are an expert assistant providing precise answers based solely on the provided document context.

CRITICAL RULES:
1. Use ONLY the provided context to answer - never add external knowledge
2. If the answer is not in the context, respond exactly: "The document does not provide this information."
3. Never make assumptions, add examples, or guess missing details
4. Preserve exact terminology, numbers, and conditions from the context
5. Be precise and factual - avoid generalizations"""

        question_specific_guidance = ""
        if question_analysis['is_yes_no']:
            question_specific_guidance = "\n6. For Yes/No questions: Start with 'Yes' or 'No', then provide supporting details from the context."
        elif question_analysis['asks_for_list']:
            question_specific_guidance = "\n6. For list questions: Provide items exactly as mentioned in the context, maintaining original order and terminology."
        elif question_analysis['asks_for_number']:
            question_specific_guidance = "\n6. For numerical questions: Provide exact numbers from the context. If ranges or approximations are given, state them precisely."
        elif question_analysis['asks_for_definition']:
            question_specific_guidance = "\n6. For definition questions: Use the exact wording and explanation provided in the context."
        
        prompt = f"""{base_rules}{question_specific_guidance}

Context from document:
{context}

Question: {question}

Answer:"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.1,  # Lower temperature for more consistent answers
                    top_p=0.9,
                    top_k=20
                )
            ),
            timeout=30.0
        )
        
        answer = response.text.strip() if response.text else "No answer generated."
        
        # Post-process answer for consistency
        if not answer or answer.lower().startswith("i don't") or answer.lower().startswith("i cannot"):
            return "The document does not provide this information."
        
        return answer
        
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
            
            # Use improved chunking strategy
            chunks = create_semantic_chunks(text, target_chunk_size=900, min_chunk_size=200)
            
            if not chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Limit number of chunks but ensure we get the most important ones
            if len(chunks) > 80:
                # Keep first chunks (usually intro/summary) and distribute the rest
                important_chunks = chunks[:20] + chunks[20::max(1, (len(chunks)-20)//40)][:60]
                chunks = important_chunks
            
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
    
    # Process questions using enhanced retrieval
    async def process_single_question(question: str) -> str:
        try:
            # Retrieve relevant chunks from Pinecone
            relevant_chunks = await retrieve_relevant_chunks_from_pinecone(question, doc_hash, top_k=4)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            answer = await generate_answer(question, relevant_chunks)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with controlled concurrency
    semaphore = asyncio.Semaphore(2)  # Reduced for better accuracy
    
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
            "vector_database": "pinecone",
            "optimization_features": ["semantic_chunking", "enhanced_retrieval", "question_analysis", "improved_prompts"]
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System with Pinecone Vector Database - Optimized for Accuracy",
        "version": "1.0.0",
        "status": "running",
        "features": ["pinecone_storage", "gemini_embeddings", "gemini_qa", "pdf_processing"],
        "optimizations": ["semantic_chunking", "enhanced_text_cleaning", "question_type_analysis", "improved_retrieval", "better_prompting"],
        "advantages": ["persistent_storage", "fast_retrieval", "scalable", "cost_effective", "high_accuracy"]
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
                "url": url,
                "optimization_level": "enhanced"
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
    logger.info("Optimizations enabled: semantic chunking, enhanced retrieval, question analysis")
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
