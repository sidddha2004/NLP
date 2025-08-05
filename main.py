# main.py – Insurance Policy RAG API (Gemini 1.5 + Semantic Embeddings & Readiness Probes)

import os
import time
import traceback
import logging
import asyncio
from typing import List
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import PyPDF2
import docx

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
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

# Environment variables
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")  # default embedding model

# Log env var presence
if not GEMINI_API_KEY:
    logger.error("⚠️ GEMINI_API_KEY is missing in environment.")
if not PINECONE_API_KEY:
    logger.error("⚠️ PINECONE_API_KEY is missing in environment.")
if not API_BEARER_TOKEN:
    logger.error("⚠️ API_BEARER_TOKEN is missing in environment.")

# Create FastAPI app with recommended CORS middleware
app = FastAPI(
    title="Insurance Policy RAG API – Gemini 1.5 & Semantic Embeddings",
    version="2.1"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Lifespan handler to replace deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model, pc, index
    try:
        # Initialize Gemini LLM
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1000,
                response_mime_type="text/plain",
            )
        )
        initialization_status["gemini"] = True
        logger.info("✅ Gemini 1.5 Flash initialized.")

        # Initialize Pinecone client and index
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("✅ Pinecone client connected.")

        index_name = "policy-docs-embedding-001"
        existing_indexes = [ix.name for ix in pc.list_indexes()]
        logger.info(f"Existing Pinecone indexes: {existing_indexes}")
        if index_name not in existing_indexes:
            pc.create_index(
                index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Created Pinecone index '{index_name}'. Waiting for readiness...")
            for i in range(30):
                try:
                    if pc.describe_index(index_name).status.get("ready", False):
                        break
                except Exception:
                    pass
                await asyncio.sleep(2)
            else:
                raise RuntimeError("Pinecone index creation timed out.")
        index = pc.Index(index_name)
        initialization_status["pinecone"] = True
        logger.info(f"✅ Pinecone index '{index_name}' ready.")

        # Start document ingestion in background
        asyncio.create_task(process_documents_background())

        yield  # control handed back to FastAPI here

    finally:
        # Place for cleanup on shutdown if needed
        logger.info("Shutdown initiated, performing cleanup if necessary.")

app.router.lifespan_context = lifespan


# Embedding utilities
@lru_cache()
def get_embed_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(EMBED_MODEL)


async def get_embedding(text: str, task="RETRIEVAL_DOCUMENT") -> List[float]:
    model = get_embed_model()
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: model.embed_content(content=text, task_type=task),
        )
        vec = resp.get("embedding")
        if not vec or len(vec) != 768:
            raise ValueError("Unexpected embedding length received.")
        return vec
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return [0.0] * 768


async def embed_batch(texts: List[str], task="RETRIEVAL_DOCUMENT") -> List[List[float]]:
    coros = [get_embedding(t, task) for t in texts]
    return await asyncio.gather(*coros)


# Document extraction and chunking
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    raise ValueError(f"Unsupported file format: {ext}")


def chunk_text(text: str, size=800, overlap=100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks


# Document ingestion
async def upsert_chunks(chunks: List[str]):
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = []
        embeddings = await embed_batch(batch)
        for j, embedding in enumerate(embeddings):
            vectors.append(
                {
                    "id": f"doc-{i+j}-{int(time.time())}",
                    "values": embedding,
                    "metadata": {"text": batch[j]},
                }
            )
        index.upsert(vectors=vectors)
        logger.info(f"Upserted batch {i // batch_size + 1}")


async def process_documents_background():
    try:
        initialization_status["document_processing"] = True
        policy_file = "policy.pdf"
        if not os.path.exists(policy_file):
            logger.error(f"Policy file {policy_file} not found. Skipping ingestion.")
            return
        text = extract_text(policy_file)
        chunks = chunk_text(text)
        logger.info(f"Extracted {len(chunks)} chunks from document.")
        # Optional: clear old vectors before reindexing
        try:
            index.delete(delete_all=True)
            logger.info("Cleared existing data from index.")
            await asyncio.sleep(5)
        except Exception:
            pass
        await upsert_chunks(chunks)
        initialization_status["document"] = True
        logger.info("Document ingestion complete.")
    except Exception as e:
        logger.error(f"Document ingestion error: {e}")
        initialization_status["error"] = str(e)
    finally:
        initialization_status["document_processing"] = False


# Query / retrieval
async def query_chunks(query: str, top_k: int = 3) -> List[str]:
    try:
        embedding = await get_embedding(query, task="RETRIEVAL_QUERY")
        response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [match.metadata.get("text", "") for match in response.matches]
    except Exception as e:
        logger.error(f"Query error: {e}")
        return []


def answer_with_llm(question: str, clauses: List[str]) -> str:
    context = "\n\n".join(f"CLAUSE {i+1}:\n{clause}" for i, clause in enumerate(clauses))
    prompt = f"""You are an insurance-policy assistant.
Use only the given clauses to answer concisely in 3 sentences or less.

QUESTION:
{question}

CLAUSES:
{context}

ANSWER:
"""
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "Unable to answer at this time."


# Security dependency
security = HTTPBearer()


def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")
    return True


# API request/response models
class QueryRequest(BaseModel):
    questions: List[str]


class QueryResponse(BaseModel):
    answers: List[str]


# API endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_questions(req: QueryRequest, _: bool = Depends(verify_token)):
    if not (gemini_model and index):
        raise HTTPException(status_code=503, detail="Service unavailable")
    answers = []
    for question in req.questions:
        clauses = await query_chunks(question)
        if clauses:
            answers.append(answer_with_llm(question, clauses))
        else:
            if initialization_status["document_processing"]:
                answers.append("Document is still processing. Please try again shortly.")
            elif not initialization_status["document"]:
                answers.append("Policy document not yet available.")
            else:
                answers.append("No relevant information found.")
    return QueryResponse(answers=answers)


@app.get("/health")
async def health():
    # Basic health status for liveness probe
    status = "healthy" if initialization_status["gemini"] and initialization_status["pinecone"] else "starting"
    return {
        "status": status,
        "services": initialization_status,
        "model": "gemini-1.5-flash",
        "index": index.name if index else None,
        "env": {
            "API_BEARER_TOKEN_SET": bool(API_BEARER_TOKEN),
            "GEMINI_API_KEY_SET": bool(GEMINI_API_KEY),
            "PINECONE_API_KEY_SET": bool(PINECONE_API_KEY),
            "PORT": os.getenv("PORT", "8000"),
        },
    }


@app.get("/ready")
async def ready():
    # Readiness probe: only 200 when document indexed and ready for queries
    if initialization_status["document"]:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Not ready")


@app.get("/")
async def root():
    return {
        "message": "Insurance Policy RAG API – Gemini 1.5 Flash + Semantic Embeddings",
        "version": "2.1"
    }


# Run with uvicorn via command line or production launch command
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
