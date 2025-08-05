from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
import requests
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Configuration from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "insurance-rag")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = FastAPI()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
index = pc.Index(PINECONE_INDEX)

# Extract text from PDF

def extract_text_from_pdf(file_path: str):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Chunk sentences with optional overlap

def chunk_sentences(text: str, max_tokens=150, overlap=1):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        current_chunk.append(sentence)
        joined = " ".join(current_chunk)
        if len(joined.split()) >= max_tokens:
            chunks.append(joined)
            current_chunk = current_chunk[-overlap:] if overlap else []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Embedding function

def get_embedding(text):
    return model.encode(text).tolist()

# Answer generation using Together Gemini

def generate_answer_from_gemini(context: str, query: str):
    prompt = f"""You are an insurance assistant. Based on the policy below, answer the question.

Policy:
"""
{context}
"""

Question: {query}
Answer:"""
    response = requests.post(
        "https://api.together.xyz/inference",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        json={
            "model": "gemini-1.5-flash",
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.2
        }
    )
    return response.json().get("output", "").strip()

# Upload endpoint
@app.post("/upload/")
async def upload_policy(file: UploadFile):
    try:
        contents = await file.read()
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(file_path)
        chunks = chunk_sentences(text)

        vectors = []
        for chunk in chunks:
            embedding = get_embedding(chunk)
            chunk_id = str(uuid.uuid4())
            vectors.append({"id": chunk_id, "values": embedding, "metadata": {"text": chunk}})

        index.upsert(vectors)
        return {"status": "indexed", "chunks": len(chunks)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Query endpoint
@app.get("/query/")
async def query_policy(q: str):
    try:
        query_embedding = get_embedding(q)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        contexts = [match["metadata"]["text"] for match in results["matches"]]
        combined_context = "\n\n".join(contexts)
        answer = generate_answer_from_gemini(combined_context, q)
        return {"answer": answer, "context_used": contexts}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
