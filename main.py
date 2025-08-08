import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter

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
        max_pages = min(100, len(pdf_reader.pages))  # Increased for better accuracy
        
        for page_num in range(max_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Clean up the text
                    page_text = re.sub(r'\s+', ' ', page_text.strip())
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            raise ValueError("No text could be extracted from PDF")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(400, f"Error extracting text from PDF: {str(e)}")

def chunk_text_smart(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """Smart chunking with sentence awareness and metadata"""
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_sentences = []
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
            current_sentences.append(sentence)
        else:
            if current_chunk.strip():
                # Extract keywords for better retrieval
                keywords = extract_keywords(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'keywords': keywords,
                    'length': len(current_chunk)
                })
            
            # Start new chunk with overlap
            overlap_text = " ".join(current_sentences[-2:]) if len(current_sentences) >= 2 else ""
            current_chunk = overlap_text + " " + sentence + " "
            current_sentences = current_sentences[-2:] + [sentence] if len(current_sentences) >= 2 else [sentence]
    
    # Add final chunk
    if current_chunk.strip():
        keywords = extract_keywords(current_chunk)
        chunks.append({
            'text': current_chunk.strip(),
            'sentences': current_sentences,
            'keywords': keywords,
            'length': len(current_chunk)
        })
    
    return chunks

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Extract important keywords from text"""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
    
    # Extract words (alphanumeric, length > 2)
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
    
    # Filter stop words and count
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    return [word for word, count in word_counts.most_common(top_k)]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity using keyword overlap"""
    keywords1 = set(extract_keywords(text1, 20))
    keywords2 = set(extract_keywords(text2, 20))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

async def get_relevant_chunks_hybrid(question: str, chunks: List[Dict], top_k: int = 5) -> List[str]:
    """Hybrid approach: keyword similarity + Gemini ranking for better accuracy"""
    
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return [chunk['text'] for chunk in chunks]
    
    # Step 1: Filter chunks using keyword similarity
    question_keywords = set(extract_keywords(question, 15))
    
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        # Keyword similarity score
        chunk_keywords = set(chunk['keywords'])
        keyword_score = len(question_keywords.intersection(chunk_keywords)) / max(len(question_keywords), 1)
        
        # Text similarity score
        text_similarity = calculate_text_similarity(question, chunk['text'])
        
        # Combined score
        combined_score = (keyword_score * 0.6) + (text_similarity * 0.4)
        
        chunk_scores.append((i, combined_score, chunk))
    
    # Sort by score and take top candidates (more than final top_k)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = chunk_scores[:min(10, len(chunk_scores))]  # Take top 10 candidates
    
    # Step 2: Use Gemini to rank the top candidates
    try:
        candidate_chunks = [item[2] for item in top_candidates]
        gemini_ranked = await rank_chunks_with_gemini(question, candidate_chunks, top_k)
        return gemini_ranked
    except Exception as e:
        logger.error(f"Error in Gemini ranking: {e}")
        # Fallback to keyword-based ranking
        return [item[2]['text'] for item in top_candidates[:top_k]]

async def rank_chunks_with_gemini(question: str, chunks: List[Dict], top_k: int) -> List[str]:
    """Use Gemini to rank pre-filtered chunks"""
    
    if len(chunks) <= top_k:
        return [chunk['text'] for chunk in chunks]
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create ranking prompt with shortened chunks
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            shortened_text = chunk['text'][:600]  # Limit each chunk
            chunks_text += f"CHUNK {i+1}:\n{shortened_text}\n\n---\n\n"
        
        prompt = f"""Question: "{question}"

Rank these text chunks by relevance to answering the question. Consider:
1. Direct relevance to the question
2. Specific information that helps answer the question
3. Context that supports the answer

{chunks_text}

Return ONLY the {top_k} most relevant chunk numbers in order of relevance, comma-separated.
Example: "3,1,7"
Response:"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.1
                )
            ),
            timeout=30.0
        )
        
        # Parse response
        try:
            chunk_indices = [int(x.strip()) - 1 for x in response.text.strip().split(',')]
            relevant_chunks = [chunks[i]['text'] for i in chunk_indices if 0 <= i < len(chunks)]
            
            # If we don't get enough, fill with remaining chunks
            if len(relevant_chunks) < top_k:
                remaining = [chunks[i]['text'] for i in range(len(chunks)) if i not in [idx for idx in chunk_indices if 0 <= idx < len(chunks)]]
                relevant_chunks.extend(remaining[:top_k - len(relevant_chunks)])
            
            return relevant_chunks[:top_k]
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse Gemini ranking: {e}")
            return [chunk['text'] for chunk in chunks[:top_k]]
            
    except Exception as e:
        logger.error(f"Error in Gemini ranking: {e}")
        return [chunk['text'] for chunk in chunks[:top_k]]

async def generate_answer_enhanced(question: str, context: str, chunks_metadata: List[Dict] = None) -> str:
    """Enhanced answer generation with better prompting"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit context length
        limited_context = context[:5000] if len(context) > 5000 else context
        
        prompt = f"""You are an expert document analyst. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
1. Be specific and accurate
2. Quote relevant parts when helpful
3. If the answer isn't clearly in the context, say "The document doesn't contain enough information to answer this question"
4. Be concise but comprehensive
5. Use bullet points for multiple related points

CONTEXT:
{limited_context}

QUESTION: {question}

ANSWER:"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,  # Increased for better answers
                    temperature=0.1,  # Lower temperature for accuracy
                    top_p=0.9,
                )
            ),
            timeout=45.0  # Increased timeout
        )
        
        answer = response.text.strip() if response.text else "No answer generated."
        
        # Post-process the answer
        if len(answer) < 10:
            return "The document doesn't contain enough information to provide a meaningful answer to this question."
        
        return answer
        
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
            chunks = chunk_text_smart(text)  # Using smart chunking
            
            if not chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Store in memory with enhanced metadata
            document_store[doc_hash] = {
                'chunks': chunks,
                'url': str(url),
                'processed_at': datetime.utcnow().isoformat(),
                'total_text_length': len(text),
                'chunk_count': len(chunks)
            }
            logger.info(f"Document processed successfully. Chunks: {len(chunks)}, Text length: {len(text)}")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info(f"Using cached document: {doc_hash}")
    
    chunks = document_store[doc_hash]['chunks']
    
    async def process_single_question(question: str) -> str:
        try:
            # Get relevant chunks using hybrid approach
            relevant_chunk_texts = await get_relevant_chunks_hybrid(question, chunks, top_k=5)
            
            if not relevant_chunk_texts:
                return "No relevant information found in the document."
            
            context = "\n\n---\n\n".join(relevant_chunk_texts)
            
            # Generate enhanced answer
            answer = await generate_answer_enhanced(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with limited concurrency
    semaphore = asyncio.Semaphore(3)  # Increased concurrency
    
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
        "message": "HackRX Document Q&A System (High Accuracy)",
        "version": "2.0.0",
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
            timeout=600.0  # 10 minutes timeout for accuracy
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
            await asyncio.sleep(1800)  # Clean up every 30 minutes
            if len(document_store) > 30:  # Keep only 30 most recent documents
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
    logger.info("Application starting up with enhanced accuracy features...")
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
