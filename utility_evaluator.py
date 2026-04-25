from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config


class UtilityEvaluator:
    """Evaluates the overall utility and completeness of the final response."""

    def __init__(self):
        Config.validate()
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0
        )

        self.eval_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""You are a response quality evaluator. Assess the overall utility and completeness of the answer.

Question: {question}

Answer: {answer}

Rate the answer on a scale of 1 to 5:
- 1: Does not address the question at all
- 2: Minimally relevant, mostly unhelpful
- 3: Partially answers the question, may be vague or incomplete
- 4: Good answer that addresses the question well with minor gaps
- 5: Excellent, complete, and directly addresses the question

Respond with ONLY the numeric score (1, 2, 3, 4, or 5)."""
        )

    def evaluate(self, question: str, answer: str) -> int:
        """Evaluate the utility of the answer on a scale of 1-5."""
        response = self.llm.invoke(
            self.eval_prompt.format(
                question=question,
                answer=answer
            )
        )

        try:
            score = int(response.content.strip())
            return max(1, min(5, score))
        except ValueError:
            return 3  # Default to middle score

    def get_rating(self, question: str, answer: str) -> str:
        """Get a descriptive rating for the answer."""
        score = self.evaluate(question, answer)
        ratings = {
            1: "Poor - Does not address the question",
            2: "Weak - Minimally relevant",
            3: "Fair - Partially answers",
            4: "Good - Addresses well with minor gaps",
            5: "Excellent - Complete and comprehensive"
        }
        return ratings.get(score, "Unknown")
