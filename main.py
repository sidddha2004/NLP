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

app = FastAPI(title="HackRX Document Q&A System - Semantic Enhanced Version", version="2.1.0")

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

# NEW FEATURE 1: SEMANTIC QUERY EXPANSION
async def semantic_query_expansion(question: str) -> List[str]:
    """Generate semantic variations and expansions of the query"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Expand this question into 3-4 semantic variations that capture the same meaning using different terminology, synonyms, and related concepts.

Original question: {question}

Requirements:
- Use domain-specific synonyms and technical terms
- Include related concepts and alternative phrasings
- Cover both formal and informal ways of asking the same thing
- Maintain the exact same intent and information need
- Focus on terms that might appear in professional/business documents

Generate semantic variations:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.6,
                top_p=0.9
            )
        )
        
        if response.text:
            variations = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            # Include original question and limit variations
            all_variations = [question] + variations[:3]
            return all_variations
        
    except Exception as e:
        logger.warning(f"Failed to generate semantic variations: {e}")
    
    return [question]

# NEW FEATURE 2: NER ENHANCEMENT WITH METADATA TAGGING
def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using regex patterns"""
    entities = {
        'people': [],
        'organizations': [],
        'dates': [],
        'numbers': [],
        'locations': [],
        'monetary': []
    }
    
    # Person names (capitalized words, often with titles)
    person_pattern = r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|CEO|CFO|CTO|President|Director)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\b'
    people = re.findall(person_pattern, text)
    entities['people'] = list(set([name.strip() for name in people if len(name.strip()) > 2]))
    
    # Organizations (Inc, Corp, Ltd, LLC, Company patterns)
    org_pattern = r'\b([A-Z][A-Za-z\s&]+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|Company|Co\.?|Group|Holdings|Partners|Solutions|Technologies|Systems|Services))\b'
    orgs = re.findall(org_pattern, text)
    entities['organizations'] = list(set([org.strip() for org in orgs]))
    
    # Dates (various formats)
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        r'\bQ[1-4]\s+\d{4}\b',
        r'\b\d{4}\b(?=\s*(?:fiscal|quarter|year|annual))'
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    entities['dates'] = list(set(dates))
    
    # Numbers and percentages
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%|percent|percentage)?\b'
    numbers = re.findall(number_pattern, text)
    entities['numbers'] = list(set(numbers))
    
    # Monetary amounts
    monetary_pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|M|B|T))?\b'
    monetary = re.findall(monetary_pattern, text)
    entities['monetary'] = list(set(monetary))
    
    # Locations (basic pattern)
    location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s*,\s*([A-Z]{2}|[A-Z][a-z]+)\b'
    locations = re.findall(location_pattern, text)
    entities['locations'] = list(set([f"{loc[0]}, {loc[1]}" for loc in locations]))
    
    return entities

async def generate_query_variations(question: str) -> List[str]:
    """Enhanced query variations with semantic expansion"""
    try:
        # Get both semantic variations and original variations
        semantic_variations = await semantic_query_expansion(question)
        
        # Original variation logic (kept for backward compatibility)
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
        
        traditional_variations = []
        if response.text:
            traditional_variations = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        
        # Combine and deduplicate
        all_variations = semantic_variations + traditional_variations
        seen = set()
        unique_variations = []
        for var in all_variations:
            if var.lower() not in seen:
                unique_variations.append(var)
                seen.add(var.lower())
        
        return unique_variations[:5]  # Limit to top 5 variations
        
    except Exception as e:
        logger.warning(f"Failed to generate query variations: {e}")
    
    return [question]

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
                model="models/text-embedding-004",  # Updated to use correct Gemini embedding model
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
    """Store document chunks with enhanced metadata including NER"""
    try:
        logger.info(f"Processing {len(chunks)} chunks for Pinecone storage with NER enhancement")
        
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
                    
                    # Extract named entities for enhanced metadata
                    entities = extract_named_entities(chunk)
                    
                    # Enhanced metadata with NER information
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
                        "chunk_type": "enhanced" if '[Previous context:' in chunk or '[Next context:' in chunk else "standard",
                        # NER metadata
                        "people": entities['people'][:5],  # Limit to prevent metadata size issues
                        "organizations": entities['organizations'][:5],
                        "dates": entities['dates'][:5],
                        "numbers": entities['numbers'][:5],
                        "locations": entities['locations'][:3],
                        "monetary": entities['monetary'][:5],
                        "entity_count": sum(len(v) for v in entities.values()),
                        "has_people": len(entities['people']) > 0,
                        "has_organizations": len(entities['organizations']) > 0,
                        "has_dates": len(entities['dates']) > 0,
                        "has_monetary": len(entities['monetary']) > 0
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
                    logger.info(f"Upserted batch {(i//batch_size) + 1} to Pinecone ({len(vectors_to_upsert)} vectors with NER metadata)")
                except Exception as e:
                    logger.error(f"Failed to upsert batch {(i//batch_size) + 1}: {e}")
        
        logger.info(f"Successfully stored {successful_uploads} chunks with NER enhancement in Pinecone for document {doc_hash}")
        return successful_uploads > 0
        
    except Exception as e:
        logger.error(f"Error storing document in Pinecone: {e}")
        return False

async def multi_query_retrieval(questions: List[str], doc_hash: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Enhanced retrieval using multiple query variations with NER-based filtering"""
    all_chunks = []
    seen_chunk_ids = set()
    
    # Extract entities from all questions for filtering
    question_entities = {}
    for question in questions:
        entities = extract_named_entities(question)
        for entity_type, entity_list in entities.items():
            if entity_list:
                if entity_type not in question_entities:
                    question_entities[entity_type] = []
                question_entities[entity_type].extend(entity_list)
    
    for question in questions:
        try:
            question_embedding = await create_gemini_embedding(question, "retrieval_query")
            
            if not question_embedding:
                continue
            
            # Base query without entity filtering
            query_result = index.query(
                vector=question_embedding,
                filter={"document_hash": doc_hash},
                top_k=top_k * 3,  # Get more results for better filtering
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
                    
                    # Boost score for entity matches
                    entity_boost = 0.0
                    chunk_metadata = match['metadata']
                    
                    # Check for entity matches and boost relevance
                    for entity_type, query_entities in question_entities.items():
                        if query_entities:
                            chunk_entities = chunk_metadata.get(entity_type, [])
                            if chunk_entities:
                                # Check for exact matches
                                for query_entity in query_entities:
                                    for chunk_entity in chunk_entities:
                                        if query_entity.lower() in chunk_entity.lower() or chunk_entity.lower() in query_entity.lower():
                                            entity_boost += 0.1
                    
                    # Apply entity boost to score
                    chunk_data['score'] += entity_boost
                    chunk_data['entity_boost'] = entity_boost
                    
                    all_chunks.append(chunk_data)
                    seen_chunk_ids.add(chunk_id)
        
        except Exception as e:
            logger.warning(f"Failed to retrieve for query '{question}': {e}")
    
    # Sort by boosted score and take top chunks
    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    return all_chunks[:top_k * 2]  # Return more chunks for re-ranking

# NEW FEATURE 3: CONTEXTUAL RE-RANKING
async def contextual_rerank(question: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Re-rank retrieved chunks using contextual relevance scoring"""
    if len(chunks) <= top_k:
        return chunks
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare chunks for ranking
        chunk_texts = []
        for i, chunk in enumerate(chunks[:8]):  # Limit to top 8 for re-ranking
            # Clean chunk text for ranking
            text = chunk['text']
            text = re.sub(r'\[Previous context:.*?\]', '', text)
            text = re.sub(r'\[Next context:.*?\]', '', text)
            text = text.strip()
            chunk_texts.append(f"Chunk {i+1}: {text[:300]}...")  # Truncate for efficiency
        
        chunks_text = "\n\n".join(chunk_texts)
        
        prompt = f"""Rank these document chunks by their relevance to answering this specific question. Consider semantic relevance, factual content, and direct answerability.

Question: {question}

Document chunks to rank:
{chunks_text}

Instructions:
- Rate each chunk's relevance on a scale of 1-10
- Consider: Does it contain information that directly answers the question?
- Consider: How specific and detailed is the relevant information?
- Consider: Does it contain the exact facts, names, numbers, or concepts asked about?

Return ONLY the chunk numbers in order of relevance (most relevant first), separated by commas.
Example format: 3, 1, 5, 2, 7

Ranking:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.1,
                top_p=0.8
            )
        )
        
        if response.text:
            # Parse the ranking
            ranking_text = response.text.strip()
            try:
                # Extract numbers from the ranking response
                ranked_indices = []
                for item in ranking_text.split(','):
                    item = item.strip()
                    if item.isdigit():
                        idx = int(item) - 1  # Convert to 0-based index
                        if 0 <= idx < len(chunks[:8]):
                            ranked_indices.append(idx)
                
                # Re-order chunks based on ranking
                if ranked_indices:
                    reranked_chunks = []
                    for idx in ranked_indices[:top_k]:
                        if idx < len(chunks):
                            chunk = chunks[idx].copy()
                            chunk['rerank_score'] = len(ranked_indices) - ranked_indices.index(idx)  # Higher score for higher rank
                            reranked_chunks.append(chunk)
                    
                    # Fill remaining slots with original order if needed
                    while len(reranked_chunks) < top_k and len(reranked_chunks) < len(chunks):
                        for chunk in chunks:
                            if chunk not in reranked_chunks:
                                chunk_copy = chunk.copy()
                                chunk_copy['rerank_score'] = 0
                                reranked_chunks.append(chunk_copy)
                                if len(reranked_chunks) >= top_k:
                                    break
                    
                    logger.info(f"Successfully re-ranked {len(reranked_chunks)} chunks using contextual analysis")
                    return reranked_chunks[:top_k]
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse re-ranking results: {parse_error}")
        
    except Exception as e:
        logger.warning(f"Contextual re-ranking failed: {e}")
    
    # Fallback: return original chunks with score-based ranking
    for chunk in chunks:
        chunk['rerank_score'] = 0
    return chunks[:top_k]

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
        
        # Prepare context from chunks with re-ranking information
        context_parts = []
        for i, chunk_data in enumerate(relevant_chunks):
            text = chunk_data['text']
            # Remove context markers for cleaner presentation
            text = re.sub(r'\[Previous context:.*?\]', '', text)
            text = re.sub(r'\[Next context:.*?\]', '', text)
            text = text.strip()
            
            # Add relevance indicators
            relevance_info = ""
            if 'rerank_score' in chunk_data and chunk_data['rerank_score'] > 0:
                relevance_info = f" [High relevance: {chunk_data['rerank_score']}]"
            elif 'entity_boost' in chunk_data and chunk_data['entity_boost'] > 0:
                relevance_info = f" [Entity match: +{chunk_data['entity_boost']:.1f}]"
            
            context_parts.append(f"[Section {i+1}{relevance_info}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Summarize context if too long
        context = await summarize_context_if_needed(context, question, 4500)
        
        # Enhanced prompting based on question type and semantic analysis
        base_rules = """You are an expert document analyst providing precise answers based solely on the provided document sections.

CRITICAL RULES:
1. Use ONLY the provided sections to answer - never add external knowledge
2. If the answer is not in the sections, respond exactly: "The document does not provide this information."
3. Never make assumptions, add examples, or guess missing details
4. Preserve exact terminology, numbers, and conditions from the sections
5. Be precise and factual - avoid generalizations
6. Always maintain the original meaning and context from the document
7. Pay special attention to sections marked with [High relevance] or [Entity match] as they are most relevant"""

        question_specific_guidance = ""
        if question_analysis['is_yes_no']:
            question_specific_guidance = "\n8. For Yes/No questions: Start with 'Yes' or 'No', then provide supporting details from the sections."
        elif question_analysis['asks_for_list']:
            question_specific_guidance = "\n8. For list questions: Provide items exactly as mentioned, maintaining original order and terminology."
        elif question_analysis['asks_for_number']:
            question_specific_guidance = "\n8. For numerical questions: Provide exact numbers from the sections. If ranges or approximations are given, state them precisely."
        elif question_analysis['asks_for_definition']:
            question_specific_guidance = "\n8. For definition questions: Use the exact wording and explanation provided in the sections."
        elif question_analysis['is_complex']:
            question_specific_guidance = "\n8. For complex questions: Address each part systematically using information from the sections."

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
            
            # Store in Pinecone with NER enhancement
            success = await store_document_in_pinecone(doc_hash, enhanced_chunks, str(url))
            
            if not success:
                raise HTTPException(500, "Failed to store document in Pinecone")
                
            logger.info(f"Document successfully processed and stored with NER. Base chunks: {len(base_chunks)}, Enhanced chunks: {len(enhanced_chunks)}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    else:
        logger.info("Document found in Pinecone, using existing data with NER metadata")
    
    # Process questions with ENHANCED semantic strategies
    async def process_single_question(question: str) -> str:
        try:
            logger.info(f"Processing question with semantic enhancement: {question}")
            
            # Step 1: Question decomposition for complex questions
            sub_questions = await decompose_complex_question(question)
            logger.info(f"Question decomposed into {len(sub_questions)} parts")
            
            # Step 2: Generate semantic query variations (NEW FEATURE 1)
            all_queries = []
            for sub_q in sub_questions:
                variations = await generate_query_variations(sub_q)  # Now includes semantic expansion
                all_queries.extend(variations)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q.lower() not in seen:
                    unique_queries.append(q)
                    seen.add(q.lower())
            
            # Limit to most relevant queries
            unique_queries = unique_queries[:7]  # Increased for semantic variations
            logger.info(f"Using {len(unique_queries)} semantic query variations for enhanced retrieval")
            
            # Step 3: Multi-query retrieval with NER enhancement (FEATURE 2)
            relevant_chunks = await multi_query_retrieval(unique_queries, doc_hash, top_k=6)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            logger.info(f"Retrieved {len(relevant_chunks)} chunks, applying contextual re-ranking...")
            
            # Step 4: Contextual re-ranking (NEW FEATURE 3)
            reranked_chunks = await contextual_rerank(question, relevant_chunks, top_k=4)
            
            logger.info(f"Re-ranked to top {len(reranked_chunks)} most contextually relevant chunks")
            
            # Step 5: Generate answer with enhanced context and ranking information
            answer = await generate_answer(question, reranked_chunks)
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
            "version": "2.1.0",
            "optimization_features": [
                "recursive_character_text_splitting",
                "sliding_window_chunking", 
                "multi_query_retrieval",
                "question_decomposition",
                "context_summarization",
                "enhanced_prompting",
                "semantic_query_expansion",  # NEW
                "ner_metadata_enhancement",  # NEW
                "contextual_reranking"      # NEW
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System - Semantic Enhanced Version with Advanced AI Features",
        "version": "2.1.0",
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
            "improved_prompting_strategies",
            "semantic_query_expansion",      # NEW FEATURE 1
            "ner_metadata_enhancement",      # NEW FEATURE 2  
            "contextual_chunk_reranking"     # NEW FEATURE 3
        ],
        "semantic_enhancements": [
            "domain_specific_synonyms",
            "technical_term_matching", 
            "concept_expansion",
            "named_entity_recognition",
            "contextual_relevance_scoring",
            "intelligent_result_ranking"
        ],
        "advantages": [
            "persistent_storage", 
            "fast_retrieval", 
            "scalable", 
            "cost_effective", 
            "high_accuracy",
            "context_preservation",
            "intelligent_chunking",
            "semantic_understanding",  # NEW
            "entity_aware_search",     # NEW
            "relevance_optimized"      # NEW
        ]
    }

@app.post("/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(
    request: DocumentRequest,
    token: str = Depends(verify_token)
) -> DocumentResponse:
    try:
        logger.info(f"Processing request with semantic enhancement for document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        answers = await asyncio.wait_for(
            process_document_and_questions(request.documents, request.questions),
            timeout=420.0  # Increased timeout for enhanced processing
        )
        
        logger.info(f"Successfully processed {len(answers)} answers with semantic optimization features")
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Request processing timed out")
        raise HTTPException(408, "Request processing timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"An unexpected error occurred: {str(e)}")

# Enhanced document status endpoint with NER information
@app.get("/document-status/{doc_hash}")
async def get_document_status(doc_hash: str, token: str = Depends(verify_token)):
    try:
        exists = await check_document_exists_in_pinecone(doc_hash)
        
        if exists:
            # Get comprehensive document stats with NER information
            query_result = index.query(
                vector=[0.0] * 768,
                filter={"document_hash": doc_hash},
                top_k=100,
                include_metadata=True
            )
            
            chunk_count = len(query_result['matches'])
            enhanced_chunks = 0
            standard_chunks = 0
            entity_stats = {
                'people': 0, 'organizations': 0, 'dates': 0, 
                'numbers': 0, 'locations': 0, 'monetary': 0
            }
            creation_date = None
            url = None
            
            if query_result['matches']:
                first_match = query_result['matches'][0]
                if first_match['metadata']:
                    creation_date = first_match['metadata'].get('created_at')
                    url = first_match['metadata'].get('url')
                
                # Count chunk types and entities
                for match in query_result['matches']:
                    if match['metadata'].get('chunk_type') == 'enhanced':
                        enhanced_chunks += 1
                    else:
                        standard_chunks += 1
                    
                    # Count entities
                    for entity_type in entity_stats:
                        if match['metadata'].get(f'has_{entity_type}'):
                            entity_stats[entity_type] += 1
            
            return {
                "document_hash": doc_hash,
                "exists": True,
                "total_chunks": chunk_count,
                "enhanced_chunks": enhanced_chunks,
                "standard_chunks": standard_chunks,
                "created_at": creation_date,
                "url": url,
                "optimization_level": "semantic_enhanced_v2.1",
                "entity_statistics": entity_stats,
                "features_enabled": [
                    "recursive_splitting",
                    "sliding_windows", 
                    "multi_query_retrieval",
                    "context_preservation",
                    "semantic_expansion",    # NEW
                    "ner_enhancement",       # NEW  
                    "contextual_reranking"   # NEW
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

# Enhanced debugging endpoint with semantic features
@app.post("/debug/semantic-analysis")
async def debug_semantic_analysis(
    question: str,
    token: str = Depends(verify_token)
):
    try:
        # Test all semantic enhancement features
        semantic_variations = await semantic_query_expansion(question)
        traditional_variations = await generate_query_variations(question)
        decomposed = await decompose_complex_question(question)
        analysis = analyze_question_type(question)
        entities = extract_named_entities(question)
        
        return {
            "original_question": question,
            "semantic_variations": semantic_variations,
            "traditional_variations": traditional_variations, 
            "combined_unique_queries": list(set(semantic_variations + traditional_variations)),
            "decomposed_questions": decomposed,
            "question_analysis": analysis,
            "extracted_entities": entities,
            "entity_count": sum(len(v) for v in entities.values()),
            "semantic_features_status": {
                "semantic_expansion": "active",
                "ner_extraction": "active",
                "question_analysis": "active",
                "query_decomposition": "active"
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error in semantic debug endpoint: {str(e)}")

# New endpoint to test re-ranking (for debugging)
@app.post("/debug/test-reranking")
async def debug_test_reranking(
    question: str,
    doc_hash: str,
    token: str = Depends(verify_token)
):
    try:
        # Get some chunks for testing
        query_embedding = await create_gemini_embedding(question, "retrieval_query")
        
        if not query_embedding:
            return {"error": "Could not create query embedding"}
        
        # Get chunks
        query_result = index.query(
            vector=query_embedding,
            filter={"document_hash": doc_hash},
            top_k=8,
            include_metadata=True
        )
        
        if not query_result['matches']:
            return {"error": "No chunks found for document"}
        
        # Convert to chunk format
        chunks = []
        for match in query_result['matches']:
            if match['metadata'] and 'text' in match['metadata']:
                chunks.append({
                    'id': match['id'],
                    'text': match['metadata']['text'],
                    'score': match['score'],
                    'metadata': match['metadata']
                })
        
        # Test re-ranking
        original_order = [chunk['id'] for chunk in chunks]
        reranked_chunks = await contextual_rerank(question, chunks, top_k=4)
        reranked_order = [chunk['id'] for chunk in reranked_chunks]
        
        return {
            "question": question,
            "document_hash": doc_hash,
            "original_chunk_order": original_order,
            "reranked_chunk_order": reranked_order,
            "reranking_applied": original_order != reranked_order,
            "reranked_scores": [chunk.get('rerank_score', 0) for chunk in reranked_chunks]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error in reranking debug endpoint: {str(e)}")

@app.on_event("startup")
async def startup():
    logger.info("HackRX Document Q&A System v2.1 (Semantic Enhanced) starting up...")
    logger.info("🚀 NEW SEMANTIC FEATURES LOADED:")
    logger.info("   ✓ Semantic Query Expansion - Domain-specific synonym matching")
    logger.info("   ✓ NER Enhancement - Named entity recognition and metadata tagging") 
    logger.info("   ✓ Contextual Re-ranking - AI-powered relevance scoring")
    logger.info(f"Pinecone index: {PINECONE_INDEX_NAME}")
    logger.info("All optimization strategies loaded successfully")
    logger.info("System ready for ultra-high-accuracy document Q&A processing with semantic understanding")

@app.on_event("shutdown")
async def shutdown():
    logger.info("HackRX Document Q&A System v2.1 (Semantic Enhanced) shutting down...")
    logger.info("Shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
