import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import json
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Set
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

app = FastAPI(title="HackRX Document Q&A System - Final Optimized Version", version="2.0.0")

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
    """Advanced text cleaning and normalization"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'\.([A-Z])', r'. \1', text)  # Add space after periods
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # Add space between text and numbers
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # Add space between numbers and text
    
    # Fix broken words from PDF extraction
    text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)  # Fix hyphenated words
    
    # Remove excessive punctuation but preserve important ones
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

def recursive_character_text_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Recursive character-based text splitting with hierarchical delimiters"""
    if not text.strip():
        return []
    
    # Hierarchical separators in order of preference
    separators = [
        "\n\n\n",  # Multiple paragraph breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence endings
        "! ",      # Exclamation sentence endings
        "? ",      # Question sentence endings
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Spaces
        ""         # Character level (last resort)
    ]
    
    def _split_text_recursive(text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[0]
        new_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator)
        
        # Rejoin separator (except for empty string separator)
        if separator != "":
            splits = [s + separator if i < len(splits) - 1 else s for i, s in enumerate(splits)]
        
        good_splits = []
        for s in splits:
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    # Merge accumulated good splits
                    merged_text = "".join(good_splits)
                    final_chunks.extend(_merge_splits(merged_text, chunk_size, chunk_overlap))
                    good_splits = []
                
                # Recursively split the large chunk
                if new_separators:
                    final_chunks.extend(_split_text_recursive(s, new_separators))
                else:
                    # Last resort: force split by character
                    final_chunks.extend(_force_split(s, chunk_size, chunk_overlap))
        
        # Handle remaining good splits
        if good_splits:
            merged_text = "".join(good_splits)
            final_chunks.extend(_merge_splits(merged_text, chunk_size, chunk_overlap))
        
        return final_chunks
    
    def _merge_splits(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Merge small splits and create overlapping chunks"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this would be the last chunk and it's small, extend previous chunk
            if end >= len(text):
                chunk = text[start:]
                if len(chunk) > 100 or not chunks:  # Avoid tiny trailing chunks
                    chunks.append(chunk)
                else:
                    # Merge with previous chunk
                    if chunks:
                        chunks[-1] += " " + chunk
                break
            else:
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - chunk_overlap
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _force_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Force split long text that can't be split by separators"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - chunk_overlap
            
        return chunks
    
    # Start recursive splitting
    chunks = _split_text_recursive(text, separators)
    
    # Filter out very small chunks and clean up
    filtered_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) >= 100:  # Minimum chunk size
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def create_sliding_window_chunks(chunks: List[str], window_overlap: int = 150) -> List[str]:
    """Create sliding window overlaps between chunks to preserve context"""
    if len(chunks) <= 1:
        return chunks
    
    enhanced_chunks = []
    
    for i, chunk in enumerate(chunks):
        enhanced_chunk = chunk
        
        # Add context from previous chunk
        if i > 0:
            prev_chunk = chunks[i-1]
            overlap_text = prev_chunk[-window_overlap:] if len(prev_chunk) > window_overlap else prev_chunk
            # Find a good break point
            overlap_words = overlap_text.split()
            if len(overlap_words) > 10:
                overlap_text = " ".join(overlap_words[-10:])
            enhanced_chunk = f"[Previous context: {overlap_text}] {chunk}"
        
        # Add context from next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i+1]
            preview_text = next_chunk[:window_overlap] if len(next_chunk) > window_overlap else next_chunk
            # Find a good break point
            preview_words = preview_text.split()
            if len(preview_words) > 10:
                preview_text = " ".join(preview_words[:10])
            enhanced_chunk = f"{enhanced_chunk} [Next context: {preview_text}]"
        
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

async def generate_query_variations(question: str) -> List[str]:
    """Generate multiple variations of a question for better retrieval"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Generate 2-3 different variations of this question that maintain the same meaning but use different wording and phrasing. This will help find more relevant information in a document.

Original question: {question}

Requirements:
- Maintain the exact same intent and meaning
- Use synonyms and alternative phrasings
- Keep questions concise and clear
- Focus on the core information being asked
- Return only the alternative questions, one per line

Alternative questions:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        if response.text:
            variations = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            # Include original question and limit to 3 total variations
            all_questions = [question] + variations[:2]
            return all_questions
        
    except Exception as e:
        logger.warning(f"Failed to generate query variations: {e}")
    
    return [question]  # Fallback to original question

async def decompose_complex_question(question: str) -> List[str]:
    """Break down complex questions into simpler sub-questions"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Check if question is complex enough to decompose
        question_lower = question.lower()
        complexity_indicators = ['and', 'or', 'also', 'additionally', 'furthermore', 'moreover', 'what are', 'how does', 'explain']
        
        if not any(indicator in question_lower for indicator in complexity_indicators) or len(question.split()) < 8:
            return [question]
        
        prompt = f"""Analyze this question and break it down into 2-3 simpler sub-questions if it's complex. Each sub-question should be answerable independently and together they should cover the original question completely.

Original question: {question}

If the question is simple and doesn't need decomposition, return just the original question.
If it can be broken down, return the sub-questions, one per line.

Sub-questions:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250,
                temperature=0.3,
                top_p=0.8
            )
        )
        
        if response.text:
            sub_questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            return sub_questions[:3] if len(sub_questions) > 1 else [question]
        
    except Exception as e:
        logger.warning(f"Failed to decompose question: {e}")
    
    return [question]

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
        query_result = index.query(
            vector=[0.0] * 768,
            filter={"document_hash": doc_hash},
            top_k=1,
            include_metadata=True
        )
        
        return len(query_result['matches']) > 0
        
    except Exception as e:
        logger.error(f"Error checking document existence in Pinecone: {e}")
        return False

async def store_document_in_pinecone(doc_hash: str, chunks: List[str], url: str) -> bool:
    """Store document chunks with enhanced metadata"""
    try:
        logger.info(f"Processing {len(chunks)} chunks for Pinecone storage")
        
        batch_size = 5
        successful_uploads = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            vectors_to_upsert = []
            
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
                        "has_numbers": any(char.isdigit() for char in chunk),
                        "has_questions": '?' in chunk,
                        "has_context_markers": '[Previous context:' in chunk or '[Next context:' in chunk,
                        "chunk_type": "enhanced" if '[Previous context:' in chunk or '[Next context:' in chunk else "standard"
                    }
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                    successful_uploads += 1
                else:
                    logger.warning(f"Failed to create embedding for chunk {chunk_index}")
                
                await asyncio.sleep(0.15)
            
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

async def multi_query_retrieval(questions: List[str], doc_hash: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks using multiple query variations"""
    all_chunks = []
    seen_chunk_ids = set()
    
    for question in questions:
        try:
            question_embedding = await create_gemini_embedding(question, "retrieval_query")
            
            if not question_embedding:
                continue
            
            query_result = index.query(
                vector=question_embedding,
                filter={"document_hash": doc_hash},
                top_k=top_k * 2,
                include_metadata=True
            )
            
            for match in query_result['matches']:
                chunk_id = match['id']
                if chunk_id not in seen_chunk_ids and match['metadata'] and 'text' in match['metadata']:
                    chunk_data = {
                        'id': chunk_id,
                        'text': match['metadata']['text'],
                        'score': match['score'],
                        'metadata': match['metadata'],
                        'chunk_index': match['metadata'].get('chunk_index', 0),
                        'query_source': question
                    }
                    all_chunks.append(chunk_data)
                    seen_chunk_ids.add(chunk_id)
        
        except Exception as e:
            logger.warning(f"Failed to retrieve for query '{question}': {e}")
    
    # Sort by score and take top chunks
    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    return all_chunks[:top_k]

async def summarize_context_if_needed(context: str, question: str, max_length: int = 4000) -> str:
    """Summarize context if it's too long while preserving key information"""
    if len(context) <= max_length:
        return context
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Summarize the following context to make it more concise while preserving ALL factual information, numbers, names, dates, and specific details that could be relevant to answering this question: "{question}" 

Context to summarize:
{context[:6000]}

Requirements:
- Keep all specific facts, numbers, names, dates, and technical terms
- Maintain the logical structure and relationships
- Remove redundant information and verbose explanations
- Ensure the summary can still answer the specific question asked
- Target length: around {max_length} characters

Summarized context:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.1,
                top_p=0.8
            )
        )
        
        if response.text and len(response.text.strip()) > 100:
            return response.text.strip()
        
    except Exception as e:
        logger.warning(f"Context summarization failed: {e}")
    
    # Fallback: truncate intelligently
    sentences = context.split('. ')
    truncated = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            truncated.append(sentence)
            current_length += len(sentence)
        else:
            break
    
    return '. '.join(truncated)

def analyze_question_type(question: str) -> Dict[str, Any]:
    """Enhanced question type analysis"""
    question_lower = question.lower().strip()
    
    analysis = {
        'is_yes_no': any(question_lower.startswith(word) for word in ['is', 'are', 'can', 'does', 'do', 'will', 'would', 'should', 'could']),
        'is_what': question_lower.startswith(('what', 'which')),
        'is_how': question_lower.startswith('how'),
        'is_why': question_lower.startswith('why'),
        'is_when': question_lower.startswith('when'),
        'is_where': question_lower.startswith('where'),
        'is_who': question_lower.startswith('who'),
        'asks_for_list': any(word in question_lower for word in ['list', 'types', 'kinds', 'categories', 'examples', 'what are']),
        'asks_for_number': any(word in question_lower for word in ['how many', 'number of', 'count', 'quantity', 'amount']),
        'asks_for_definition': any(word in question_lower for word in ['define', 'definition', 'meaning', 'what is', 'what are']),
        'is_complex': len(question.split()) > 12 or any(word in question_lower for word in ['and', 'or', 'also', 'additionally'])
    }
    
    return analysis

async def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    try:
        if not relevant_chunks:
            return "The document does not provide this information."
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Analyze question for better prompting
        question_analysis = analyze_question_type(question)
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk_data in enumerate(relevant_chunks):
            text = chunk_data['text']
            # Remove context markers for cleaner presentation
            text = re.sub(r'\[Previous context:.*?\]', '', text)
            text = re.sub(r'\[Next context:.*?\]', '', text)
            text = text.strip()
            context_parts.append(f"[Section {i+1}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Summarize context if too long
        context = await summarize_context_if_needed(context, question, 4500)
        
        # Enhanced prompting based on question type
        base_rules = """You are an expert document analyst providing precise answers based solely on the provided document sections.

CRITICAL RULES:
1. Use ONLY the provided sections to answer - never add external knowledge
2. If the answer is not in the sections, respond exactly: "The document does not provide this information."
3. Never make assumptions, add examples, or guess missing details
4. Preserve exact terminology, numbers, and conditions from the sections
5. Be precise and factual - avoid generalizations
6. Always maintain the original meaning and context from the document"""

        question_specific_guidance = ""
        if question_analysis['is_yes_no']:
            question_specific_guidance = "\n7. For Yes/No questions: Start with 'Yes' or 'No', then provide supporting details from the sections."
        elif question_analysis['asks_for_list']:
            question_specific_guidance = "\n7. For list questions: Provide items exactly as mentioned, maintaining original order and terminology."
        elif question_analysis['asks_for_number']:
            question_specific_guidance = "\n7. For numerical questions: Provide exact numbers from the sections. If ranges or approximations are given, state them precisely."
        elif question_analysis['asks_for_definition']:
            question_specific_guidance = "\n7. For definition questions: Use the exact wording and explanation provided in the sections."
        elif question_analysis['is_complex']:
            question_specific_guidance = "\n7. For complex questions: Address each part systematically using information from the sections."

        prompt = f"""{base_rules}{question_specific_guidance}

Document sections:
{context}

Question: {question}

Answer:"""

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=600,
                    temperature=0.1,
                    top_p=0.9,
                    top_k=20
                )
            ),
            timeout=35.0
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
            
            # Use recursive character text splitting
            base_chunks = recursive_character_text_split(text, chunk_size=900, chunk_overlap=150)
            
            if not base_chunks:
                raise HTTPException(400, "No text chunks could be created from the PDF")
            
            # Apply sliding window chunking for better context preservation
            enhanced_chunks = create_sliding_window_chunks(base_chunks, window_overlap=100)
            
            # Limit chunks but ensure quality
            if len(enhanced_chunks) > 70:
                # Keep important chunks: first 15 (intro), last 10 (conclusion), and distribute middle
                important_chunks = (
                    enhanced_chunks[:15] + 
                    enhanced_chunks[15::max(1, (len(enhanced_chunks)-25)//45)][:45] + 
                    enhanced_chunks[-10:]
                )
                enhanced_chunks = important_chunks
            
            # Store in Pinecone
            success = await store_document_in_pinecone(doc_hash, enhanced_chunks, str(url))
            
            if not success:
                raise HTTPException(500, "Failed to store document in Pinecone")
                
            logger.info(f"Document successfully processed and stored. Base chunks: {len(base_chunks)}, Enhanced chunks: {len(enhanced_chunks)}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info("Document found in Pinecone, using existing data")
    
    # Process questions with enhanced strategies
    async def process_single_question(question: str) -> str:
        try:
            # Step 1: Question decomposition for complex questions
            sub_questions = await decompose_complex_question(question)
            logger.info(f"Question decomposed into {len(sub_questions)} parts: {sub_questions}")
            
            # Step 2: Generate query variations for better retrieval
            all_queries = []
            for sub_q in sub_questions:
                variations = await generate_query_variations(sub_q)
                all_queries.extend(variations)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q.lower() not in seen:
                    unique_queries.append(q)
                    seen.add(q.lower())
            
            # Limit to most relevant queries
            unique_queries = unique_queries[:6]
            logger.info(f"Using {len(unique_queries)} unique queries for retrieval")
            
            # Step 3: Multi-query retrieval
            relevant_chunks = await multi_query_retrieval(unique_queries, doc_hash, top_k=5)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            # Step 4: Generate answer with enhanced context
            answer = await generate_answer(question, relevant_chunks)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    # Process questions with controlled concurrency
    semaphore = asyncio.Semaphore(2)
    
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
            "version": "2.0.0",
            "optimization_features": [
                "recursive_character_text_splitting",
                "sliding_window_chunking", 
                "multi_query_retrieval",
                "question_decomposition",
                "context_summarization",
                "enhanced_prompting"
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System - Final Optimized Version with Advanced Accuracy Features",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "pinecone_storage", 
            "gemini_embeddings", 
            "gemini_qa", 
            "pdf_processing"
        ],
        "accuracy_optimizations": [
            "recursive_character_text_splitting",
            "sliding_window_chunking",
            "multi_query_retrieval", 
            "question_decomposition",
            "context_summarization",
            "enhanced_question_analysis",
            "improved_prompting_strategies"
        ],
        "advantages": [
            "persistent_storage", 
            "fast_retrieval", 
            "scalable", 
            "cost_effective", 
            "high_accuracy",
            "context_preservation",
            "intelligent_chunking"
        ]
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
            timeout=400.0  # Increased timeout for enhanced processing
        )
        
        logger.info(f"Successfully processed {len(answers)} answers with advanced optimization")
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Request processing timed out")
        raise HTTPException(408, "Request processing timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"An unexpected error occurred: {str(e)}")

# Enhanced document status endpoint
@app.get("/document-status/{doc_hash}")
async def get_document_status(doc_hash: str, token: str = Depends(verify_token)):
    try:
        exists = await check_document_exists_in_pinecone(doc_hash)
        
        if exists:
            # Get comprehensive document stats
            query_result = index.query(
                vector=[0.0] * 768,
                filter={"document_hash": doc_hash},
                top_k=100,
                include_metadata=True
            )
            
            chunk_count = len(query_result['matches'])
            enhanced_chunks = 0
            standard_chunks = 0
            creation_date = None
            url = None
            
            if query_result['matches']:
                first_match = query_result['matches'][0]
                if first_match['metadata']:
                    creation_date = first_match['metadata'].get('created_at')
                    url = first_match['metadata'].get('url')
                
                # Count chunk types
                for match in query_result['matches']:
                    if match['metadata'].get('chunk_type') == 'enhanced':
                        enhanced_chunks += 1
                    else:
                        standard_chunks += 1
            
            return {
                "document_hash": doc_hash,
                "exists": True,
                "total_chunks": chunk_count,
                "enhanced_chunks": enhanced_chunks,
                "standard_chunks": standard_chunks,
                "created_at": creation_date,
                "url": url,
                "optimization_level": "advanced_v2.0",
                "features_enabled": [
                    "recursive_splitting",
                    "sliding_windows", 
                    "multi_query_retrieval",
                    "context_preservation"
                ]
            }
        else:
            return {
                "document_hash": doc_hash,
                "exists": False
            }
            
    except Exception as e:
        raise HTTPException(500, f"Error checking document status: {str(e)}")

# Endpoint to delete document from Pinecone
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

# New endpoint to test query variations (for debugging)
@app.post("/debug/query-variations")
async def debug_query_variations(
    question: str,
    token: str = Depends(verify_token)
):
    try:
        variations = await generate_query_variations(question)
        decomposed = await decompose_complex_question(question)
        analysis = analyze_question_type(question)
        
        return {
            "original_question": question,
            "query_variations": variations,
            "decomposed_questions": decomposed,
            "question_analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error in debug endpoint: {str(e)}")

@app.on_event("startup")
async def startup():
    logger.info("HackRX Document Q&A System v2.0 starting up...")
    logger.info("Advanced Features: Recursive Text Splitting + Multi-Query Retrieval + Context Optimization")
    logger.info(f"Pinecone index: {PINECONE_INDEX_NAME}")
    logger.info("All optimization strategies loaded successfully")
    logger.info("System ready for high-accuracy document Q&A processing")

@app.on_event("shutdown")
async def shutdown():
    logger.info("HackRX Document Q&A System v2.0 shutting down...")
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
