# main.py – Insurance Policy RAG API (Gemini 1.5 Flash + TRUE semantic embeddings)

import os, time, hashlib, traceback, logging, asyncio
from typing import List
from functools import lru_cache

# ────────────────────────────────────────────────  FastAPI  ────────────────────────────────────────────────
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ────────────────────────────────────────────────  Docs  ───────────────────────────────────────────────────
import PyPDF2, docx

# ────────────────────────────────────────────────  AI / DB  ───────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# ────────────────────────────────────────────────  Logging  ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────  Globals  ───────────────────────────────────────────────
gemini_model = None          # LLM (1.5-flash) – for answers
pc = None                    # Pinecone client
index = None                 # Pinecone index handle
initialization_status = {
    "gemini": False,
    "pinecone": False,
    "document": False,
    "document_processing": False,
    "error": None
}

# ─────────────────────────────────────────────  ENV VARS  ────────────────────────────────────────────────
API_BEARER_TOKEN   = os.getenv("API_BEARER_TOKEN")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
EMBED_MODEL        = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")   # 768-dim

if not GEMINI_API_KEY:   logger.error("⚠️  GEMINI_API_KEY missing")
if not PINECONE_API_KEY: logger.error("⚠️  PINECONE_API_KEY missing")
if not API_BEARER_TOKEN: logger.error("⚠️  API_BEARER_TOKEN missing")

# ────────────────────────────────────────────────  FastAPI app  ───────────────────────────────────────────
app = FastAPI(title="Insurance Policy RAG API – Gemini 1.5 Flash", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ────────────────────────────────────────────────  Gemini LLM init  ───────────────────────────────────────
def init_gemini_llm():
    global gemini_model
    if gemini_model: return gemini_model
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=genai.types.GenerationConfig(
            temperature=0.1, top_p=0.8, top_k=40,
            max_output_tokens=1000, response_mime_type="text/plain"
        )
    )
    initialization_status["gemini"] = True
    logger.info("✅ Gemini 1.5 Flash LLM ready")
    return gemini_model

# ─────────────────────────────────────────  EMBEDDING HELPERS  ────────────────────────────────────────────
@lru_cache(maxsize=2)
def _check_embed_ready():
    """Ping once so later calls don’t re-configure."""
    genai.configure(api_key=GEMINI_API_KEY)
    return True

async def get_embedding_gemini(text: str, task_type="RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Async wrapper around Google Embedding API (768-dim).
    task_type: 'RETRIEVAL_DOCUMENT'  or  'RETRIEVAL_QUERY'
    """
    _check_embed_ready()
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model = EMBED_MODEL,
                content = text,
                task_type = task_type
            )
        )
        vec = resp["embedding"]
        if not vec or len(vec) != 768:
            raise ValueError("Bad embedding length")
        return vec
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return [0.0]*768

async def embed_batch(texts: List[str], task_type="RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed a list of strings concurrently (fan-out)."""
    coros = [get_embedding_gemini(t, task_type) for t in texts]
    return await asyncio.gather(*coros)

# ────────────────────────────────────────────────  Pinecone init  ─────────────────────────────────────────
def init_pinecone():
    global pc, index
    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("✅ Pinecone client connected")
    if index is None:
        index_name = "policy-docs-gemini-1-5-flash"
        names = [i.name for i in pc.list_indexes()]
        if index_name not in names:
            logger.info(f"Creating index '{index_name}' (768-dim)")
            pc.create_index(index_name, dimension=768, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            time.sleep(8)
        index = pc.Index(index_name)
        initialization_status["pinecone"] = True
        logger.info("✅ Pinecone index ready")
    return pc, index

# ────────────────────────────────────────────────  Extraction / Chunking  ────────────────────────────────
def extract_text(file_path:str)->str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext==".pdf":
        with open(file_path,"rb")as f:
            r,txt=PyPDF2.PdfReader(f),""
            for p in r.pages:
                if (t:=p.extract_text()): txt+=t+"\n"
            return txt
    if ext==".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported doc format")

def chunk_text(text:str, size=800, overlap=100)->List[str]:
    out=[]; i=0; L=len(text)
    while i<L:
        end=min(i+size,L); out.append(text[i:end])
        i = end-overlap if end-overlap>i else end
    return out

# ─────────────────────────────────────────────  DOCUMENT INGESTION  ──────────────────────────────────────
async def upsert_chunks_async(chunks: List[str]):
    batch = 10
    for i in range(0,len(chunks),batch):
        batch_text = chunks[i:i+batch]
        vectors  = await embed_batch(batch_text, task_type="RETRIEVAL_DOCUMENT")
        payloads = [{
            "id"   : f"doc-{i+j}-{int(time.time())}",
            "values": vec,
            "metadata": {"text": batch_text[j]}
        } for j,vec in enumerate(vectors)]
        index.upsert(vectors=payloads)
        logger.info(f"Upserted batch {i//batch+1}")

async def process_documents_background():
    try:
        initialization_status["document_processing"]=True
        file="policy.pdf"
        if not os.path.exists(file):
            logger.error("policy.pdf not found – skipping ingestion")
            return
        txt = extract_text(file)
        chunks = chunk_text(txt)
        logger.info(f"📄 {len(chunks)} chunks generated")
        # optional: clear previous
        try: index.delete(delete_all=True); time.sleep(3)
        except Exception: pass
        await upsert_chunks_async(chunks)
        initialization_status["document"]=True
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        initialization_status["error"]=str(e)
    finally:
        initialization_status["document_processing"]=False

# ────────────────────────────────────────────────  Retrieval  ────────────────────────────────────────────
def query_chunks(query:str, top_k:int=3)->List[str]:
    try:
        vec = asyncio.run(get_embedding_gemini(query,"RETRIEVAL_QUERY"))
        res = index.query(vector=vec, top_k=top_k, include_metadata=True)
        return [m.metadata["text"] for m in res.matches]
    except Exception as e:
        logger.error(f"Query err: {e}")
        return []

def answer_with_llm(question:str, clauses:List[str])->str:
    ctx="\n\n".join(f"CLAUSE {i+1}:\n{c}" for i,c in enumerate(clauses))
    prompt=f"""You are an insurance-policy assistant.
Use only the clauses to answer in ≤3 sentences, quote if helpful.

QUESTION: {question}
CLAUSES:
{ctx}

ANSWER:"""
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        logger.error(e); return "Unable to answer."

# ───────────────────────────────────────────────  Security  ─────────────────────────────────────────────
security = HTTPBearer()
def verify(credentials:HTTPAuthorizationCredentials=Depends(security)):
    if credentials.credentials!=API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

# ─────────────────────────────────────────────  Pydantic models  ─────────────────────────────────────────
class QueryRequest(BaseModel):
    questions: List[str]
class QueryResponse(BaseModel):
    answers  : List[str]

# ───────────────────────────────────────────────  Startup  ──────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    try:
        init_gemini_llm()
        init_pinecone()
        asyncio.create_task(process_documents_background())
    except Exception as e:
        initialization_status["error"]=str(e)

@app.on_event("shutdown")
async def _shutdown():
    pass  # nothing async to close now

# ───────────────────────────────────────────  End-user endpoints  ───────────────────────────────────────
@app.post("/hackrx/run", response_model=QueryResponse)
async def run(req:QueryRequest, _:bool=Depends(verify)):
    answers=[]
    for q in req.questions:
        clauses = query_chunks(q)
        answers.append(answer_with_llm(q,clauses) if clauses else "No data yet.")
    return QueryResponse(answers=answers)

@app.get("/health")
async def health():
    return {"status":"ok","init":initialization_status}

@app.get("/")
async def root():
    return {"message":"Insurance Policy RAG – Gemini 1.5 Flash + embeddings-001"}

# ─────────────────────────────────────────────  Railway entrypoint  ─────────────────────────────────────
if __name__=="__main__":
    import uvicorn, sys
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT",8000)))
