import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import Config

class DocumentProcessor:
    def __init__(self):
        Config.validate()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self._initialize_index()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _initialize_index(self):
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        # Delete existing index with wrong dimension
        if Config.PINECONE_INDEX_NAME in existing_indexes:
            try:
                self.pc.delete_index(Config.PINECONE_INDEX_NAME)
                print(f"Deleted old index with wrong dimension")
            except Exception as e:
                print(f"Warning: {e}")

        # Create fresh index with correct dimension
        self.pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new Pinecone index: {Config.PINECONE_INDEX_NAME} (dimension: 384)")
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)

    def load_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str, source: str = "unknown") -> List[dict]:
        chunks = self.text_splitter.split_text(text)
        return [{"text": chunk, "source": source, "chunk_id": i}
                for i, chunk in enumerate(chunks)]

    def embed_and_upload(self, chunks: List[dict]):
        vectors = []
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": f"{chunk['source']}_{chunk['chunk_id']}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"]
                }
            })

        self.index.upsert(vectors=vectors)
        print(f"Uploaded {len(vectors)} chunks to Pinecone")

    def process_pdf(self, pdf_path: str):
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")
        text = self.load_pdf(pdf_path)
        chunks = self.chunk_text(text, source=filename)
        self.embed_and_upload(chunks)
        return len(chunks)

    def process_directory(self, dir_path: str):
        total_chunks = 0
        for file in os.listdir(dir_path):
            if file.endswith(".pdf"):
                total_chunks += self.process_pdf(os.path.join(dir_path, file))
        print(f"Total chunks uploaded: {total_chunks}")
