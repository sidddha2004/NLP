import os
import time
import logging
import asyncio
import re
from typing import List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import PyPDF2
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
gemini_model = None
pc = None
index = None
index_name = "ultra-opt-policy"  # Global index name consistent across the app
embedding_model = None
status = {"ready": False, "error": None}


# FastAPI app with CORS middleware
app = FastAPI(title="Ultra-Optimized Insurance RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


def initialize_services():
    """Initialize embedding model, Gemini LLM, Pinecone client and index"""
    global gemini_model, pc, index, embedding_model, status

    try:
        # Initialize sentence-transformers embedding model (384-dimensional)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model (all-MiniLM-L6-v2) initialized.")

        # Initialize Gemini model for generation (not embedding)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1200,
            )
        )
        logger.info("✅ Gemini 1.5 Flash model initialized.")

        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("✅ Pinecone client connected.")

        # Create Pinecone index if it does not exist
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created Pinecone index '{index_name}'. Waiting 15 seconds for readiness...")
            time.sleep(15)  # wait for the index to be ready

        index = pc.Index(index_name)
        logger.info(f"✅ Pinecone index '{index_name}' ready.")

        status["ready"] = True

    except Exception as e:
        status["error"] = str(e)
        logger.error(f"Service initialization failed: {e}")
        raise e


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def optimal_chunk_text(text: str, size=900, overlap=100) -> List[str]:
    """Chunk text smartly while preserving sentence boundaries and overlap"""
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()[-overlap // 15 :]
            overlapped.append(" ".join(prev_words) + " " + chunks[i])
        chunks = overlapped

    return chunks


def get_high_quality_embedding(text: str) -> List[float]:
    """Embed a given text using sentence-transformers model"""
    try:
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Optional: fallback or return zero vector
        return [0.0] * 384


def query_relevant_chunks(question: str, top_k=3) -> List[Dict[str, Any]]:
    """Query Pinecone index for most relevant document chunks"""
    try:
        query_embedding = get_high_quality_embedding(question)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = results.matches
        if not matches:
            return []

        max_score = max(m.score for m in matches) if matches else 0.0
        threshold = 0.75 if max_score > 0.85 else 0.60

        relevant = [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "id": match.id,
            }
            for match in matches if match.score > threshold
        ]

        return relevant if relevant else [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "id": match.id,
            }
            for match in matches[:2]
        ]

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []


def generate_precise_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer from Gemini using relevant chunks as context"""
    if not chunks:
        return "I couldn't find relevant information in the policy document for this question."

    context = "\n\n".join(
        f"POLICY SECTION {i+1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
        for i, chunk in enumerate(chunks[:3])
    )

    prompt = f"""You are an expert insurance policy analyst. Use ONLY the provided policy sections to answer the user's question precisely and comprehensively.

REQUIREMENTS:
- Provide specific, accurate info from the policy.
- Include exact amounts, percentages, limits when mentioned.
- List exclusions with bullet points if found.
- Quote relevant policy language where helpful.
- If info isn't found in sections, state this clearly.
- Be comprehensive but focused.

QUESTION: {question}

POLICY SECTIONS:
{context}

COMPREHENSIVE ANSWER:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()

        # Clean and optimize answer text
        answer = re.sub(r'^(Based on.*?policy.*?[,:])\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s+', ' ', answer)

        return answer
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "I encountered an error while analyzing the policy. Please try again."


async def process_and_store_document():
    """Process the policy PDF, chunk, embed, and upsert to Pinecone"""
    try:
        policy_file = "policy.pdf"
        if not os.path.exists(policy_file):
            logger.error("Policy file not found")
            return

        text = extract_pdf_text(policy_file)
        chunks = optimal_chunk_text(text, size=900, overlap=100)

        # Clear existing index vectors if any
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                index.delete(delete_all=True)
                await asyncio.sleep(2)
        except Exception:
            pass

        batch_size = 6
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            vectors = []
            for j, chunk in enumerate(batch_chunks):
                try:
                    embedding = get_high_quality_embedding(chunk)
                    vectors.append(
                        {
                            "id": f"chunk-{i+j}-{int(time.time())}",
                            "values": embedding,
                            "metadata": {"text": chunk, "index": i + j},
                        }
                    )
                except Exception as e:
                    logger.warning(f"Chunk {i+j} embedding failed: {e}")

            if vectors:
                index.upsert(vectors=vectors)
                await asyncio.sleep(0.5)  # avoid rate limits

        logger.info(f"Successfully processed and ingested {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")


# Security dependency - API token verification
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected = os.getenv("API_BEARER_TOKEN")
    if not expected or credentials.credentials != expected:
        raise HTTPException(status_code=403, detail="Invalid API token")
    return True


# Request and response models
class QueryRequest(BaseModel):
    questions: List[str]


class QueryResponse(BaseModel):
    answers: List[str]


@app.on_event("startup")
async def startup_event():
    initialize_services()
    if status["ready"]:
        asyncio.create_task(process_and_store_document())


@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(req: QueryRequest, verified: bool = Depends(verify_token)):
    if not status["ready"]:
        raise HTTPException(status_code=503, detail="Service not ready")

    answers = []
    for question in req.questions:
        try:
            chunks = query_relevant_chunks(question, top_k=3)
            answer = generate_precise_answer(question, chunks)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")

    return QueryResponse(answers=answers)


@app.get("/health")
async def health():
    return {"status": "healthy" if status["ready"] else "initializing", "index": index_name}


@app.get("/debug")
async def debug_info():
    """Provide debug info about index and status"""
    try:
        if not index:
            return {"error": "Index not ready"}

        stats = index.describe_index_stats()
        return {
            "vectors_stored": stats.total_vector_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "status": status,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {
        "name": "Ultra-Optimized Insurance RAG API",
        "embedding_quality": "High (all-MiniLM-L6-v2)",
        "size_optimized": True,
        "status": status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
