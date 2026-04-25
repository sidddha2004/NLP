from typing import Literal, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config


class FactualityChecker:
    """Evaluates if the generated segment is factually supported by retrieved documents."""

    def __init__(self):
        Config.validate()
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0
        )

        self.check_prompt = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""You are a factuality evaluator. Assess if the generated answer is supported by the retrieved documents.

Question: {question}

Generated Answer: {answer}

Retrieved Context:
{context}

Evaluate the answer and respond with ONE of:
- fully supported: All claims in the answer are directly supported by the context
- partially supported: Some claims are supported, but others are not or are speculative
- no support: Answer contains claims not found in or contradicted by the context

Respond with ONLY the decision (fully supported, partially supported, or no support)."""
        )

    def check(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> Literal["fully supported", "partially supported", "no support"]:
        """Check if the answer is factually supported by retrieved chunks."""
        if not chunks:
            # No chunks to support anything
            return "no support"

        context = "\n\n---\n\n".join([
            chunk["metadata"]["text"] for chunk in chunks
        ])

        response = self.llm.invoke(
            self.check_prompt.format(
                question=question,
                answer=answer,
                context=context
            )
        )

        decision = response.content.strip().lower()
        valid_decisions = {"fully supported", "partially supported", "no support"}

        if decision not in valid_decisions:
            return "partially supported"  # Default to middle ground

        return decision

    def is_factually_sound(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> bool:
        """Return True if answer is at least partially supported."""
        decision = self.check(question, answer, chunks)
        return decision in {"fully supported", "partially supported"}
