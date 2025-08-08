# main.py
import os
import hashlib
import logging
import asyncio
import aiohttp
import io
import re
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import PyPDF2
import google.generativeai as genai

# Optional import - if sentence-transformers isn't present, we fall back to keywords
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

app = FastAPI(title="HackRX Document Q&A System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# In-memory store
document_store: Dict[str, Dict] = {}

# Env vars
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "default_token")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")
genai.configure(api_key=GEMINI_API_KEY)

# Try load a pre-downloaded sentence-transformers model
st_model = None
if SentenceTransformer is not None:
    try:
        MODEL_PATH = os.getenv("ST_MODEL_PATH", "./model")  # builder stage copies it here
        st_model = SentenceTransformer(MODEL_PATH)
        logger.info("SentenceTransformer model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.warning("SentenceTransformer model not available at %s: %s", MODEL_PATH, e)
        st_model = None
else:
    logger.info("sentence-transformers not installed; using keyword fallback.")

# Request/Response models
class DocumentRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the PDF document")
    questions: List[str] = Field(..., min_length=1, max_length=10, description="List of questions")

class DocumentResponse(BaseModel):
    answers: List[str]

# Auth
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials

# Helpers
def generate_document_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

async def download_pdf(url: str) -> bytes:
    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(str(url)) as resp:
                if resp.status != 200:
                    raise HTTPException(400, f"Failed to download PDF: HTTP {resp.status}")
                content = await resp.read()
                if not content:
                    raise HTTPException(400, "Downloaded file is empty")
                return content
    except aiohttp.ClientError as e:
        raise HTTPException(400, f"Network error downloading PDF: {e}")

def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 200) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_parts = []
        pages_to_read = min(max_pages, len(reader.pages))
        for i in range(pages_to_read):
            try:
                p = reader.pages[i]
                txt = p.extract_text()
                if txt:
                    txt = re.sub(r'\s+', ' ', txt.strip())
                    text_parts.append(txt)
            except Exception as e:
                logger.warning("Failed to read page %d: %s", i + 1, e)
                continue
        if not text_parts:
            raise ValueError("No text could be extracted from PDF")
        return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(400, f"Error extracting text from PDF: {e}")

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
        'is','are','was','were','be','been','have','has','had','do','does','did',
        'this','that','these','those','it','its'
    }
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(top_k)]

def chunk_text_smart(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """Split by sentences, keep sentence boundaries, attach keywords metadata"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, current_sentences = [], "", []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
            current_sentences.append(sentence)
        else:
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "sentences": current_sentences.copy(),
                    "keywords": extract_keywords(current_chunk)
                })
            overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
            current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
            current_sentences = overlap_sentences + [sentence]
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "sentences": current_sentences,
            "keywords": extract_keywords(current_chunk)
        })
    return chunks

def calculate_keyword_similarity(q: str, chunk: Dict) -> float:
    qk = set(extract_keywords(q, top_k=15))
    ck = set(chunk.get("keywords", []))
    if not qk or not ck:
        return 0.0
    return len(qk & ck) / len(qk | ck)

# Gemini parsing helper (robust)
def parse_genai_text(resp) -> str:
    # genai responses have changed over time; try common fields
    if resp is None:
        return ""
    # try .text
    t = getattr(resp, "text", None)
    if t:
        return t
    # try .output_text or .content...
    t = getattr(resp, "output_text", None)
    if t:
        return t
    # as a fallback, string-convert
    try:
        return str(resp)
    except Exception:
        return ""

# Ranking chunks with Gemini (re-rank small candidate set)
async def rank_chunks_with_gemini(question: str, chunk_texts: List[str], top_k: int) -> List[str]:
    if not chunk_texts:
        return []
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_chunks = ""
        for i, t in enumerate(chunk_texts):
            preview = t[:800].replace("\n", " ")
            prompt_chunks += f"CHUNK {i+1}:\n{preview}\n\n---\n\n"
        prompt = (
            f'Question: "{question}"\n\n'
            "Rank these text chunks by relevance to answering the question. Return only the top "
            f"{top_k} chunk numbers in order, comma-separated. Example: \"3,1,2\"\n\n"
            f"{prompt_chunks}\nResponse:"
        )
        # Use thread to run sync method if needed
        resp = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=80, temperature=0.0)
        )
        text = parse_genai_text(resp).strip()
        # parse numbers
        raw = re.findall(r'\d+', text)
        idxs = []
        for r in raw:
            try:
                i = int(r) - 1
                if 0 <= i < len(chunk_texts):
                    idxs.append(i)
            except:
                continue
        # fill with top candidates if not enough parsed
        if len(idxs) < min(top_k, len(chunk_texts)):
            for i in range(len(chunk_texts)):
                if i not in idxs:
                    idxs.append(i)
                if len(idxs) >= top_k:
                    break
        ranked = [chunk_texts[i] for i in idxs[:top_k]]
        return ranked
    except Exception as e:
        logger.warning("Gemini re-ranking failed: %s", e)
        return chunk_texts[:top_k]

# Hybrid retrieval: sentence-transformers semantic similarity (if available) + keyword fallback + Gemini re-rank
async def get_relevant_chunks_hybrid(question: str, chunks: List[Dict], top_k: int = 5) -> List[str]:
    if not chunks:
        return []
    chunk_texts = [c["text"] for c in chunks]

    # If model present, use embeddings (fast)
    try:
        if st_model is not None and util is not None:
            q_emb = st_model.encode(question, convert_to_tensor=True)
            c_embs = st_model.encode(chunk_texts, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(q_emb, c_embs)[0]
            scored = sorted(zip(chunk_texts, scores.tolist()), key=lambda x: x[1], reverse=True)
            top_candidates = [t for t, _ in scored[: min(len(scored), max(10, top_k))]]
        else:
            # Keyword score
            scored = [(c["text"], calculate_keyword_similarity(question, c)) for c in chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [t for t, _ in scored[: min(len(scored), max(10, top_k))]]
    except Exception as e:
        logger.warning("Semantic scoring failed, falling back to keywords: %s", e)
        scored = [(c["text"], calculate_keyword_similarity(question, c)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [t for t, _ in scored[: min(len(scored), max(10, top_k))]]

    # Gemini re-rank the top candidates for better precision
    try:
        reranked = await rank_chunks_with_gemini(question, top_candidates, top_k)
        return reranked
    except Exception as e:
        logger.warning("Gemini ranking failed; returning top candidates: %s", e)
        return top_candidates[:top_k]

# Answer generation using Gemini with context
async def generate_answer_enhanced(question: str, context: str, max_tokens: int = 600) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # keep context trimmed
        limited_context = context if len(context) <= 4500 else context[-4500:]
        prompt = (
            "You are an expert document analyst. Answer only from the provided context.\n\n"
            "INSTRUCTIONS:\n"
            "1) If answer is not in context, reply: \"The document doesn't contain enough information to answer this question.\"\n"
            "2) Be concise.\n"
            "3) Quote short relevant snippets when helpful.\n\n"
            f"CONTEXT:\n{limited_context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        resp = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=0.05, top_p=0.9)
        )
        text = parse_genai_text(resp).strip()
        if not text or len(text) < 6:
            return "The document doesn't contain enough information to provide a meaningful answer to this question."
        return text
    except asyncio.TimeoutError:
        logger.error("Answer generation timed out")
        return "Sorry, the answer generation timed out. Please try again."
    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return "Sorry, I encountered an error while generating the answer."

# Main processing pipeline
async def process_document_and_questions(url: str, questions: List[str]) -> List[str]:
    doc_hash = generate_document_hash(str(url))
    if doc_hash not in document_store:
        logger.info("Processing new document: %s", url)
        pdf_content = await download_pdf(url)
        text = extract_text_from_pdf(pdf_content)
        chunks = chunk_text_smart(text)
        if not chunks:
            raise HTTPException(400, "No text chunks could be created from the PDF")
        document_store[doc_hash] = {
            "url": str(url),
            "chunks": chunks,
            "processed_at": datetime.utcnow().isoformat(),
            "total_text_length": len(text),
            "chunk_count": len(chunks),
        }
        logger.info("Document stored: %s chunks", len(chunks))
    else:
        logger.info("Using cached document: %s", doc_hash)

    chunks = document_store[doc_hash]["chunks"]

    # concurrency limit
    semaphore = asyncio.Semaphore(3)

    async def process_single(q: str) -> str:
        async with semaphore:
            try:
                relevant = await get_relevant_chunks_hybrid(q, chunks, top_k=5)
                if not relevant:
                    return "No relevant information found in the document."
                context = "\n\n---\n\n".join(relevant)
                answer = await generate_answer_enhanced(q, context)
                return answer
            except Exception as e:
                logger.exception("Error while processing question '%s': %s", q, e)
                return "An error occurred while processing this question."

    tasks = [asyncio.create_task(process_single(q)) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results

# Background cleanup task
async def cleanup_old_documents():
    while True:
        try:
            await asyncio.sleep(1800)  # every 30 min
            max_keep = 30
            if len(document_store) > max_keep:
                sorted_docs = sorted(document_store.items(), key=lambda x: x[1].get("processed_at", ""))
                # keep newest 15, remove older ones to reduce memory
                to_delete = sorted_docs[:-15]
                for doc_id, _ in to_delete:
                    document_store.pop(doc_id, None)
                logger.info("Cleaned up old documents, remaining: %d", len(document_store))
        except Exception as e:
            logger.exception("Cleanup task failed: %s", e)

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "HackRX Document Q&A System", "version": "2.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cached_documents": len(document_store),
        "st_model_loaded": bool(st_model),
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document_qa(request: DocumentRequest, token: str = Depends(verify_token)):
    try:
        answers = await asyncio.wait_for(process_document_and_questions(request.documents, request.questions), timeout=600.0)
        return DocumentResponse(answers=answers)
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(408, "Request processing timed out")
    except Exception as e:
        logger.exception("Unexpected error in /run: %s", e)
        raise HTTPException(500, f"Unexpected server error: {e}")

# Startup / shutdown
@app.on_event("startup")
async def on_startup():
    logger.info("Starting HackRX app (hybrid retrieval). st_model_loaded=%s", bool(st_model))
    asyncio.create_task(cleanup_old_documents())

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down HackRX app. Clearing document cache.")
    document_store.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info")
