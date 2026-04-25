from typing import List, Dict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import Config
from relevance_grader import RelevanceGrader
from retrieve_evaluator import RetrieveEvaluator
from factuality_checker import FactualityChecker
from utility_evaluator import UtilityEvaluator

class RAGChain:
    def __init__(self):
        Config.validate()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3
        )
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)

        # All four checkpoint evaluators
        self.retrieve_eval = RetrieveEvaluator()
        self.grader = RelevanceGrader()
        self.factuality_checker = FactualityChecker()
        self.utility_eval = UtilityEvaluator()

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question using ONLY the provided context. If the answer is not in the context, say "I cannot answer this from the available documents."

Context:
{context}

Question: {question}

Answer:"""
        )

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL

        query_embedding = self.model.encode(query).tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [{"id": match["id"], "metadata": match["metadata"]}
                for match in results["matches"]]

    def generate(self, query: str, verified_chunks: List[Dict]) -> str:
        if not verified_chunks:
            return "No relevant documents found to answer this question."

        context = "\n\n---\n\n".join([
            chunk["metadata"]["text"] for chunk in verified_chunks
        ])

        response = self.llm.invoke(
            self.qa_prompt.format(context=context, question=query)
        )
        return response.content

    def query(self, query: str, top_k: int = None) -> Dict:
        """Execute full RAG pipeline with all four checkpoints."""

        # ========== CHECKPOINT 1: Retrieve ==========
        # Evaluates if external documents are needed
        print("\n" + "="*60)
        print("🔍 CHECKPOINT 1: Retrieve - Do we need documents?")
        print("="*60)

        retrieve_decision = self.retrieve_eval.evaluate(query)
        retrieve_mapping = {
            "yes": "✅ YES - Documents needed, proceeding with retrieval",
            "no": "❌ NO - Can answer from general knowledge",
            "continue": "⏭️ CONTINUE - Proceeding with retrieval to be safe"
        }
        print(f"Decision: {retrieve_mapping.get(retrieve_decision, retrieve_decision)}")

        if retrieve_decision == "no":
            # Answer without retrieval
            direct_response = self.llm.invoke(f"Answer this question concisely: {query}")
            return {
                "question": query,
                "answer": direct_response.content,
                "checkpoints": {
                    "retrieve": {"decision": "no", "retrieval_used": False},
                    "isrel": {"decision": "skipped", "chunks_used": 0},
                    "issup": {"decision": "skipped"},
                    "isuse": {"score": None, "rating": "skipped"}
                }
            }

        # Retrieve documents
        raw_chunks = self.retrieve(query, top_k)
        print(f"\n📚 Retrieved {len(raw_chunks)} chunks from vector store")

        if not raw_chunks:
            return {
                "question": query,
                "answer": "No relevant documents found in the knowledge base.",
                "checkpoints": {
                    "retrieve": {"decision": retrieve_decision, "retrieval_used": True},
                    "isrel": {"decision": "no_chunks", "chunks_used": 0},
                    "issup": {"decision": "skipped"},
                    "isuse": {"score": None, "rating": "skipped"}
                }
            }

        # ========== CHECKPOINT 2: IsRel ==========
        # Evaluates if retrieved documents are relevant
        print("\n" + "="*60)
        print("📋 CHECKPOINT 2: IsRel - Are documents relevant?")
        print("="*60)

        verified_chunks = self.grader.grade_and_filter(query, raw_chunks)

        isrel_decision: Literal["relevant", "irrelevant"] = (
            "relevant" if verified_chunks else "irrelevant"
        )
        print(f"\nIsRel Decision: {isrel_decision.upper()}")

        if not verified_chunks:
            return {
                "question": query,
                "answer": "Retrieved documents were not relevant enough to answer this question.",
                "checkpoints": {
                    "retrieve": {"decision": retrieve_decision, "retrieval_used": True},
                    "isrel": {
                        "decision": "irrelevant",
                        "chunks_retrieved": len(raw_chunks),
                        "chunks_used": 0
                    },
                    "issup": {"decision": "skipped"},
                    "isuse": {"score": None, "rating": "skipped"}
                }
            }

        # Generate answer
        answer = self.generate(query, verified_chunks)

        # ========== CHECKPOINT 3: IsSup ==========
        # Evaluates if generated answer is supported by documents
        print("\n" + "="*60)
        print("🛡️ CHECKPOINT 3: IsSup - Is answer factually supported?")
        print("="*60)

        issup_decision = self.factuality_checker.check(query, answer, verified_chunks)
        issup_mapping = {
            "fully supported": "✅ FULLY SUPPORTED - All claims backed by documents",
            "partially supported": "⚠️ PARTIALLY SUPPORTED - Some claims not in documents",
            "no support": "❌ NO SUPPORT - Claims not found or contradicted"
        }
        print(f"Decision: {issup_mapping.get(issup_decision, issup_decision)}")

        # ========== CHECKPOINT 4: IsUse ==========
        # Evaluates overall utility and completeness
        print("\n" + "="*60)
        print("⭐ CHECKPOINT 4: IsUse - How useful is this response?")
        print("="*60)

        utility_score = self.utility_eval.evaluate(query, answer)
        utility_rating = self.utility_eval.get_rating(query, answer)
        print(f"Score: {utility_score}/5")
        print(f"Rating: {utility_rating}")

        # Return comprehensive results
        return {
            "question": query,
            "answer": answer,
            "checkpoints": {
                "retrieve": {
                    "decision": retrieve_decision,
                    "retrieval_used": True
                },
                "isrel": {
                    "decision": "relevant",
                    "chunks_retrieved": len(raw_chunks),
                    "chunks_used": len(verified_chunks),
                    "chunks": verified_chunks
                },
                "issup": {
                    "decision": issup_decision
                },
                "isuse": {
                    "score": utility_score,
                    "rating": utility_rating
                }
            }
        }
