import os
import sys
from document_processor import DocumentProcessor
from rag_chain import RAGChain
from config import Config

def create_test_data_dir():
    os.makedirs("test_data", exist_ok=True)
    print("Created 'test_data' directory. Add PDF files here.")

def index_documents():
    processor = DocumentProcessor()

    if not os.path.exists("test_data"):
        print("Error: test_data directory not found. Run 'setup' first.")
        return

    pdfs = [f for f in os.listdir("test_data") if f.endswith(".pdf")]
    if not pdfs:
        print("No PDF files found in test_data/")
        return

    print(f"Found {len(pdfs)} PDF file(s)")
    processor.process_directory("test_data")

def query_mode():
    rag = RAGChain()

    print("\n🤖 Self-Verifying RAG System with 4 Checkpoints")
    print("=" * 60)
    print("Checkpoints: Retrieve → IsRel → IsSup → IsUse")
    print("=" * 60)

    while True:
        query = input("\n❓ Your question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        result = rag.query(query)

        # Display final answer summary
        print("\n" + "="*60)
        print("📋 FINAL ANSWER")
        print("="*60)
        print(f"{result['answer']}")

        # Display checkpoint summary
        checkpoints = result.get("checkpoints", {})
        print("\n" + "="*60)
        print("📊 CHECKPOINT SUMMARY")
        print("="*60)

        retrieve = checkpoints.get("retrieve", {})
        isrel = checkpoints.get("isrel", {})
        issup = checkpoints.get("issup", {})
        isuse = checkpoints.get("isuse", {})

        print(f"Retrieve:   {retrieve.get('decision', 'N/A').upper()}")
        print(f"IsRel:      {isrel.get('decision', 'N/A').upper()} "
              f"({isrel.get('chunks_used', 0)}/{isrel.get('chunks_retrieved', 0)} chunks)")
        print(f"IsSup:      {issup.get('decision', 'N/A').upper()}")
        print(f"IsUse:      {isuse.get('score', 'N/A')}/5 - {isuse.get('rating', 'N/A')}")

        print("="*60)

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "setup":
            create_test_data_dir()
        elif command == "index":
            index_documents()
        elif command == "query":
            query_mode()
        else:
            print("Unknown command. Available: setup, index, query")
    else:
        print("""
Self-Verifying RAG System
=========================

Commands:
  python main.py setup   - Create test_data directory
  python main.py index   - Index PDFs from test_data/
  python main.py query   - Start interactive query mode

Novelty: Self-verification layer grades document relevance
before generation, filtering out irrelevant chunks to prevent
hallucinations.
        """)

if __name__ == "__main__":
    main()
