# phase2_query_service.py - Improved Query Service for Insurance Policy RAG
# Updated with async client cleanup, graceful startup error handling,
# consistent env vars, configurable model name, and index existence check.

import os
import hashlib
import logging
from typing import List
import traceback
import asyncio
import httpx
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai  # sync SDK, but async Gemini calls via httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
gemini_model = None
pc = None
index = None
initialization_status = {"gemini": False, "pinecone": False, "error": None}

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = OPENAI_API_KEY.strip() if OPENAI_API_KEY else None
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY = PINECONE_API_KEY.strip() if PINECONE_API_KEY else None
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
MODEL_NAME = os.getenv("GEMINI_MODEL", "google/gemini-2.0-flash")

if not OPENAI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is missing!")
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable is missing!")
if not API_BEARER_TOKEN:
    logger.error("API_BEARER_TOKEN environment variable is missing!")

async_client = httpx.AsyncClient()
embedding_cache = {}

# Async Gemini query via OpenRouter API
async def query_gemini_async(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Gemini API key missing")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,  # Configurable
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    resp = await async_client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        logger.error(f"Gemini API failed: {resp.status_code} {resp.text}")
        raise RuntimeError(f"Gemini API failed: {resp.status_code} {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

# Phase 1 compatible simple embedding
def get_simple_embedding(text: str) -> List[float]:
    try:
        text = text.lower().strip()
        embeddings = []
        for i in range(16):
            hash_obj = hashlib.md5(f"{text}_{i}".encode())
            hash_bytes = hash_obj.digest()
            for byte_val in hash_bytes:
                if len(embeddings) < 256:
                    normalized_val = (byte_val / 255.0) * 2 - 1
                    embeddings.append(normalized_val)
        words = text.split()
        word_features = []
        word_features.extend([
            len(text)/1000.0,
            len(words)/100.0,
            sum(len(w) for w in words)/max(len(words),1)/10.0,
            len(set(words))/max(len(words),1),
            text.count(' ')/max(len(text),1),
            text.count('.')/max(len(text),1),
            text.count(',')/max(len(text),1),
            sum(1 for c in text if c.isupper())/max(len(text),1),
            sum(1 for c in text if c.isdigit())/max(len(text),1),
            len([w for w in words if len(w) > 5])/max(len(words),1),
        ])
        for i in range(246):
            if i < len(words):
                word_hash = hash(f"{words[i]}_{i}") % 10000
                word_features.append(word_hash / 10000.0)
            elif i < len(text):
                char_hash = hash(f"{text[i]}_{i}") % 10000
                word_features.append(char_hash / 10000.0)
            else:
                word_features.append(0.0)
        word_features = word_features[:256]
        while len(word_features) < 256:
            word_features.append(0.0)
        embeddings.extend(word_features)
        embeddings = embeddings[:512]
        while len(embeddings) < 512:
            embeddings.append(0.0)
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return [0.0] * 512

async def get_gemini_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        prompt = f"Extract 50 focused and relevant keywords from the following insurance text, separated by commas:\n{text[:1000]}"
        keywords = await query_gemini_async(prompt)
        embedding = get_simple_embedding(keywords)
        if len(embedding) != 512:
            embedding.extend([0.0] * (512 - len(embedding)))
            embedding = embedding[:512]
        embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        logger.error(f"Error getting Gemini embedding: {e}")
        return get_simple_embedding(text)

async def query_chunks(query: str, index, top_k: int = 5) -> List[str]:
    try:
        embedding = await get_gemini_embedding(query)
        response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        chunks = [hit.metadata.get("text", "") for hit in response.matches]
        logger.info(f"Found {len(chunks)} chunks for query")
        return chunks
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

async def generate_answer(question: str, context_clauses: List[str]) -> str:
    if not context_clauses:
        return "No relevant information found in the document."
    prompt = f"""
You are a professional assistant specialized in insurance policy analysis.
Answer the question strictly based on the provided clauses below.
Do NOT speculate or include information outside these clauses.
Quote or reference the clauses in your response where possible.

Clauses:
{chr(10).join(['- ' + c for c in context_clauses])}

Question:
{question}

Answer clearly, concisely, and cite relevant clauses:
"""
    try:
        answer = await query_gemini_async(prompt)
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

# Security using Bearer token
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = API_BEARER_TOKEN
    if not expected_token:
        logger.error("API_BEARER_TOKEN not configured")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="API_BEARER_TOKEN not configured")
    if token != expected_token:
        logger.warning("Invalid token received")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid authentication token")
    return True

# Pydantic request/response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# FastAPI app setup
app = FastAPI(title="Insurance Policy Query Service", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def initialize_gemini():
    global gemini_model
    try:
        if gemini_model is None:
            if not OPENAI_API_KEY:
                raise ValueError("GEMINI_API_KEY env var not set")
            genai.configure(api_key=OPENAI_API_KEY)
            gemini_model = genai.GenerativeModel(MODEL_NAME)
            logger.info("Gemini model initialized")
        return gemini_model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        raise

def initialize_pinecone():
    global pc, index
    try:
        if pc is None:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY env var not set")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client initialized")
        if index is None:
            index_name = "policy-docs-gemini-hash"
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            logger.info(f"Available indexes: {existing_indexes}")
            if index_name not in existing_indexes:
                raise ValueError(f"Pinecone index '{index_name}' not found. Populate Phase 1 first.")
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                logger.warning("Pinecone index is empty")
            else:
                logger.info(f"Pinecone index ready with {stats.total_vector_count} vectors")
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    global gemini_model, pc, index
    logger.info("=== Starting up Query Service ===")
    try:
        gemini_model = initialize_gemini()
        pc, index = initialize_pinecone()
        initialization_status["gemini"] = True
        initialization_status["pinecone"] = True
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        initialization_status["error"] = str(e)

@app.on_event("shutdown")
async def shutdown_event():
    await async_client.aclose()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    if not (index and gemini_model):
        raise HTTPException(status_code=503, detail="Service is initializing or unavailable")
    answers = []
    for question in req.questions:
        context_clauses = await query_chunks(question, index)
        answer = await generate_answer(question, context_clauses)
        answers.append(answer)
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_ready": initialization_status.get("gemini", False),
        "pinecone_ready": initialization_status.get("pinecone", False),
        "error": initialization_status.get("error")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
