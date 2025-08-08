import os
import hashlib
import logging
import asyncio
import aiohttp
import io
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
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
executor = ThreadPoolExecutor(max_workers=4)
model_lock = threading.Lock()

# Global variables
embedding_model = None
pc_client = None
index = None

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "default_token")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

if not all([PINECONE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

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

@app.on_event("startup")
async def startup():
    global embedding_model, pc_client, index
    try:
        logger.info("Initializing services...")
        
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        
        try:
            existing_indexes = pc_client.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if PINECONE_INDEX_NAME in index_names:
                index = pc_client.Index(PINECONE_INDEX_NAME)
            else:
                from pinecone import ServerlessSpec
                pc_client.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                await asyncio.sleep(15)
                index = pc_client.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            logger.error(f"Pinecone error: {e}")
            index = pc_client.Index(PINECONE_INDEX_NAME)
        
        loop = asyncio.get_event_loop()
        embedding_model = await loop.run_in_executor(
            executor, 
            lambda: SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        )
        
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

def generate_document_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

async def download_pdf(url: str) -> bytes:
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(str(url)) as response:
            if response.status != 200:
                raise HTTPException(400, f"Failed to download PDF: HTTP {response.status}")
            content = await response.read()
            if len(content) == 0:
                raise HTTPException(400, "Downloaded file is empty")
            return content

def extract_text_from_pdf(pdf_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    text_parts = []
    max_pages = min(20, len(pdf_reader.pages))
    
    for page_num in range(max_pages):
        try:
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            continue
    
    if not text_parts:
        raise ValueError("No text could be extracted from PDF")
    
    return "\n\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
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
    
    return chunks if chunks else [text]

async def embed_chunks_async(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    
    loop = asyncio.get_event_loop()
    
    def generate_embeddings():
        with model_lock:
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False, show_progress_bar=False)
        return embeddings
    
    embeddings = await loop.run_in_executor(executor, generate_embeddings)
    
    if len(embeddings.shape) == 1:
        result = [embeddings.tolist()]
    else:
        result = embeddings.tolist()
        
    return result

def check_document_exists(doc_hash: str) -> bool:
    try:
        namespace = f"doc_{doc_hash}"
        stats = index.describe_index_stats()
        namespaces = stats.namespaces or {}
        return namespace in namespaces and namespaces[namespace].vector_count > 0
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        return False

async def store_embeddings_async(doc_hash: str, chunks: List[str], embeddings: List[List[float]], url: str):
    if not chunks or not embeddings:
        raise ValueError("No chunks or embeddings to store")
        
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
                'doc_hash': doc_hash
            }
        })
    
    loop = asyncio.get_event_loop()
    
    def upsert_vectors():
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
    
    await loop.run_in_executor(executor, upsert_vectors)

async def search_similar_chunks_async(question: str, doc_hash: str, top_k: int = 3) -> List[str]:
    loop = asyncio.get_event_loop()
    
    def generate_question_embedding():
        with model_lock:
            return embedding_model.encode([question], show_progress_bar=False)[0].tolist()
    
    question_embedding = await loop.run_in_executor(executor, generate_question_embedding)
    
    namespace = f"doc_{doc_hash}"
    
    def search_pinecone():
        return index.query(
            vector=question_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
    
    search_results = await loop.run_in_executor(executor, search_pinecone)
    
    relevant_chunks = []
    for match in search_results.matches:
        if match.metadata and 'text' in match.metadata:
            relevant_chunks.append(match.metadata['text'])
    
    return relevant_chunks

async def generate_answer_optimized(question: str, context: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Answer this question based on the context provided. Be concise and direct.

Context: {context[:2000]}  

Question: {question}

Answer:"""

        loop = asyncio.get_event_loop()
        
        def generate_content():
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40
                )
            )
        
        response = await loop.run_in_executor(executor, generate_content)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."

async def process_document_and_questions(url: str, questions: List[str]) -> List[str]:
    doc_hash = generate_document_hash(str(url))
    
    if not check_document_exists(doc_hash):
        pdf_content = await download_pdf(str(url))
        text = extract_text_from_pdf(pdf_content)
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(400, "No text chunks could be created")
        
        embeddings = await embed_chunks_async(chunks)
        await store_embeddings_async(doc_hash, chunks, embeddings, url)
    
    async def process_single_question(question: str) -> str:
        try:
            relevant_chunks = await search_similar_chunks_async(question, doc_hash, top_k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            context = "\n\n".join(relevant_chunks)[:3000]
            answer = await generate_answer_optimized(question, context)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return "An error occurred while processing this question."
    
    semaphore = asyncio.Semaphore(3)
    
    async def process_with_semaphore(question: str) -> str:
        async with semaphore:
            return await process_single_question(question)
    
    tasks = [process_with_semaphore(q) for q in questions]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_answers = []
    for i, answer in enumerate(answers):
        if isinstance(answer, Exception):
            logger.error(f"Error processing question {i}: {answer}")
            final_answers.append("An error occurred while processing this question.")
        else:
            final_answers.append(answer)
    
    return final_answers

@app.get("/health")
async def health_check():
    services_status = {
        "pinecone": "connected" if index is not None else "disconnected",
        "embedding_model": "loaded" if embedding_model is not None else "not_loaded",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured"
    }
    
    overall_status = "healthy" if all(
        status in ["connected", "loaded", "configured"] 
        for status in services_status.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services_status
    }

@app.get("/")
async def root():
    return {
        "message": "HackRX Document Q&A System",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(
    request: DocumentRequest,
    token: str = Depends(verify_token)
) -> DocumentResponse:
    try:
        if embedding_model is None:
            raise HTTPException(503, "Embedding model not initialized")
            
        if index is None:
            raise HTTPException(503, "Vector database not initialized")
        
        try:
            answers = await asyncio.wait_for(
                process_document_and_questions(request.documents, request.questions),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(408, "Request processing timed out")
        
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
