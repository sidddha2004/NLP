# Self-Verifying RAG System with 4-Checkoint Architecture

## Novelty
Unlike standard RAG that blindly trusts retrieved data, this system implements a **4-checkpoint self-verification pipeline** using LLM-as-judge at each stage. This multi-stage filtering significantly reduces hallucinations and ensures answer quality.

## The 4 Checkpoints

| Checkpoint | Purpose | Output |
|------------|---------|--------|
| **Retrieve** | Evaluates if external documents are needed | yes, no, continue |
| **IsRel** | Evaluates if retrieved documents are relevant | relevant, irrelevant |
| **IsSup** | Evaluates if answer is factually supported | fully supported, partially supported, no support |
| **IsUse** | Evaluates overall utility and completeness | score 1-5 |

### Key Innovations:
1. **Retrieve Checkpoint** - Avoids unnecessary retrieval for general knowledge questions
2. **IsRel Checkpoint** - Filters irrelevant chunks BEFORE generation (prevents contamination)
3. **IsSup Checkpoint** - Detects hallucinations AFTER generation (quality assurance)
4. **IsUse Checkpoint** - Provides actionable feedback on response quality (improvement loop)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Create directory for PDFs
python main.py setup

# Add your PDFs to test_data/ then index
python main.py index

# Start interactive Q&A
python main.py query
```

## Architecture

```
Query
  ↓
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT 1: Retrieve (Need documents?)                    │
│   Output: yes → continue | no → answer from knowledge       │
└─────────────────────────────────────────────────────────────┘
  ↓ (if yes)
┌─────────────────────────────────────────────────────────────┐
│ Retrieve from Pinecone (vector similarity search)           │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT 2: IsRel (Documents relevant?)                   │
│   Grades chunks 0-1, filters by threshold                   │
│   Output: relevant → continue | irrelevant → stop           │
└─────────────────────────────────────────────────────────────┘
  ↓ (if relevant)
┌─────────────────────────────────────────────────────────────┐
│ Generate Answer (LLM with filtered context)                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT 3: IsSup (Answer factually supported?)           │
│   Verifies claims against retrieved documents               │
│   Output: fully/partially/no support                        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT 4: IsUse (Response utility & completeness)       │
│   Scores overall quality on 1-5 scale                       │
└─────────────────────────────────────────────────────────────┘
  ↓
Final Answer + Full Checkpoint Report
```

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration |
| `document_processor.py` | PDF → Chunks → Embeddings → Pinecone |
| `retrieve_evaluator.py` | **Checkpoint 1:** Pre-retrieval decision |
| `relevance_grader.py` | **Checkpoint 2:** Document relevance grader |
| `factuality_checker.py` | **Checkpoint 3:** Post-generation factuality check |
| `utility_evaluator.py` | **Checkpoint 4:** Utility and completeness scorer |
| `rag_chain.py` | Main RAG pipeline with all checkpoints |
| `main.py` | CLI interface |
