# main.py - Railway deployment for Insurance Policy RAG API using Horizon Beta via OpenRouter and your Pinecone API key

import os
from typing import List
import time
import hashlib
import traceback
import logging
import requests
import json

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import docx

from pinecone import Pinecone, ServerlessSpec

# --- API KEYS: Set them here directly ---
OPENROUTER_API_KEY = "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".strip()
PINECONE_API_KEY = "pcsk_2HMPt3_6R2wiF8G1zmHjMaAQmJh69wEFDD16YtJkk3YrTC9wvTD5EiaLVZpLve4Up8nFbt".strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/horizon-beta"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Gemini 2.0 Flash", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = None
index = None
initialization_status = {
    "pinecone": False,
    "document": False,
    "error": None
}

# --- LLM via OpenRouter ---
def query_openrouter(prompt: str, model=OPENROUTER_MODEL, max_tokens=1024, temperature=0.0) -> str:
    # Debug print to verify API key presence
    logger.debug(f"Using OpenRouter API Key: '{OPENROUTER_API_KEY}' (length: {len(OPENROUTER_API_KEY)})")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter API key is missing or empty")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
    if resp.status_code != 200:
        logger.error(f"OpenRouter error: {resp.status_code} - {resp.text}")
        raise RuntimeError(f"OpenRouter error: {resp.status_code} - {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

# Pinecone initialization
def init_pinecone():
    global pc, index, initialization_status
    try:
        logger.info("🔧 Initializing Pinecone...")
        if pc is None:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("✓ Pinecone client initialized")
        if index is None:
            index_name = "policy-docs-gemini-hash"
            try:
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                logger.info(f"📋 Found existing indexes: {existing_indexes}")
                if index_name not in existing_indexes:
                    logger.info(f"🏗 Creating new index: {index_name}")
                    pc.create_index(
                        name=index_name,
                        dimension=512,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    logger.info(f"✓ Created new index: {index_name}")
                    logger.info("⏳ Waiting for index to be ready...")
                    max_retries = 60
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            if pc.describe_index(index_name).status['ready']:
                                break
                        except Exception as wait_error:
                            logger.warning(f"Waiting for index, attempt {retry_count}: {wait_error}")
                        time.sleep(2)
                        retry_count += 1
                        if retry_count % 10 == 0:
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

# --- Document processing ---
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

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# --- Simple embedding (hash-based) ---
def get_simple_embedding(text: str) -> List[float]:
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
        len(text) / 1000.0,
        len(words) / 100.0,
        sum(len(word) for word in words) / max(len(words), 1) / 10.0,
        len(set(words)) / max(len(words), 1),
        text.count(' ') / max(len(text), 1),
        text.count('.') / max(len(text), 1),
        text.count(',') / max(len(text), 1),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        len([w for w in words if len(w) > 5]) / max(len(words), 1)
    ])
    for i in range(246):
        if i < len(words):
            word_hash = hash(f"{words[i]}_{i}") % 10000
            word_features.append(word_hash / 10000.0)
        else:
            if i < len(text):
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

def get_gemini_embedding(text: str) -> List[float]:
    try:
        prompt = f"Summarize this insurance text with exactly 50 comma-separated keywords: {text[:1000]}"
        keywords = query_openrouter(prompt, max_tokens=100)
        embedding = get_simple_embedding(keywords)
        if len(embedding) != 512:
            embedding += [0.0] * (512 - len(embedding))
            embedding = embedding[:512]
        return embedding
    except Exception as e:
        logger.error(f"Error getting Gemini embedding: {e}")
        return get_simple_embedding(text)

def query_gemini(question: str, context_clauses: List[str]) -> str:
    prompt = f"""
You are an expert assistant who answers insurance policy questions precisely and cites the clauses.

Question: {question}

Use ONLY the following clauses and explicitly mention or quote them in your answer:

{chr(10).join([f"- {clause}" for clause in context_clauses])}

Answer:
"""
    try:
        return query_openrouter(prompt)
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return f"Error generating response: {str(e)}"

def query_chunks(query: str, index, top_k: int = 5) -> List[str]:
    try:
        query_embedding = get_gemini_embedding(query)
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

def upsert_chunks(chunks: List[str], index):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_gemini_embedding(chunk)
            except:
                embedding = get_simple_embedding(chunk)
            if embedding is not None:
                vectors.append({
                    "id": f"chunk-{i}-{int(time.time())}",
                    "values": embedding,
                    "metadata": {"text": chunk}
                })
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        logger.info(f"Upserted {len(chunks)} chunks to Pinecone.")
    except Exception as e:
        logger.error(f"Error upserting chunks: {e}")
        raise e

def initialize_policy_document():
    global index, initialization_status
    try:
        policy_file = "policy.pdf"
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        if not os.path.exists(policy_file):
            logger.error(f"Policy file {policy_file} not found in {os.getcwd()}")
            logger.error(f"Available files: {os.listdir('.')}")
            return
        file_stat = os.stat(policy_file)
        logger.info(f"Policy file found - Size: {file_stat.st_size} bytes, Permissions: {oct(file_stat.st_mode)}")
        logger.info(f"Loading policy document: {policy_file}")
        full_text = extract_text(policy_file)
        logger.info(f"Extracted {len(full_text)} characters from policy document")
        chunks = chunk_text(full_text)
        logger.info(f"Created {len(chunks)} chunks")
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                logger.info("Clearing existing vectors from index...")
                index.delete(delete_all=True)
                logger.info("Cleared existing vectors from index")
                time.sleep(5)
            else:
                logger.info("Index is empty, no need to clear")
        except Exception as e:
            logger.warning(f"Could not check/clear index stats: {e}")
            logger.info("Proceeding with upsert...")
        upsert_chunks(chunks, index)
        logger.info("Policy document successfully indexed")
        initialization_status["document"] = True
    except Exception as e:
        logger.error(f"Error initializing policy document: {e}")
        initialization_status["error"] = str(e)
        raise e

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

class QueryRequest(BaseModel):
    questions: List[str]
class QueryResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
async def startup_event():
    global pc, index, initialization_status
    logger.info("=== STARTUP: Initializing services ===")
    try:
        pc, index = init_pinecone()
        logger.info("✓ Pinecone initialized successfully")
        try:
            initialize_policy_document()
            logger.info("✓ Policy document initialized successfully")
        except Exception as doc_error:
            logger.warning(f"⚠ Policy document initialization failed: {doc_error}")
            logger.info("Continuing without document...")
        logger.info("=== STARTUP COMPLETE ===")
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        logger.warning("⚠ Continuing with limited functionality...")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    global pc, index
    if not index:
        logger.error("Services not properly initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services not properly initialized. Status: {initialization_status}"
        )
    answers = []
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            relevant_clauses = query_chunks(question, index)
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document for this question.")
                continue
            answer = query_gemini(question, relevant_clauses)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append(f"Error processing question: {str(e)}")
    return QueryResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Insurance Policy RAG API with Gemini 2.0 Flash is running",
        "initialization_status": initialization_status,
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "has_pinecone_key": True,
            "has_bearer_token": bool(os.getenv("API_BEARER_TOKEN"))
        }
    }

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
            "embedding_model": "gemini-enhanced-hash-embeddings",
            "llm_model": OPENROUTER_MODEL,
            "initialization_status": initialization_status
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Insurance Policy RAG API with Gemini 2.0 Flash",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "query": "/hackrx/run (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
