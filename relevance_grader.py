from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config

class RelevanceGrader:
    def __init__(self):
        Config.validate()
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0
        )

        self.grading_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""You are a relevance evaluator. Rate how well the document answers the query.

Query: {query}

Document: {document}

Provide a relevance score from 0.0 to 1.0:
- 0.9-1.0: Directly answers the query with specific information
- 0.7-0.8: Contains relevant information but not complete
- 0.5-0.6: Somewhat related tangentially
- 0.0-0.4: Not relevant or contains no useful information

Respond with ONLY the numeric score (e.g., 0.75)."""
        )

    def grade_chunk(self, query: str, chunk: Dict) -> float:
        response = self.llm.invoke(
            self.grading_prompt.format(
                query=query,
                document=chunk["metadata"]["text"]
            )
        )
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.0

    def grade_and_filter(self, query: str, chunks: List[Dict]) -> List[Dict]:
        graded_chunks = []
        for chunk in chunks:
            score = self.grade_chunk(query, chunk)
            chunk["relevance_score"] = score
            if score >= Config.RELEVANCE_THRESHOLD:
                graded_chunks.append(chunk)
            print(f"Chunk {chunk['id']}: {score:.2f} {'✓' if score >= Config.RELEVANCE_THRESHOLD else '✗'}")

        print(f"\nFiltered from {len(chunks)} → {len(graded_chunks)} relevant chunks")
        return sorted(graded_chunks, key=lambda x: x["relevance_score"], reverse=True)
