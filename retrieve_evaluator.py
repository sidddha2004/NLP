from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config


class RetrieveEvaluator:
    """Evaluates if external documents are needed to answer the prompt."""

    def __init__(self):
        Config.validate()
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0
        )

        self.eval_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query analyzer. Determine if external documents are needed to answer the query.

Query: {query}

Analyze the query and respond with ONE of:
- yes: External documents ARE needed (requires specific facts, data, or context from sources)
- no: External documents are NOT needed (can be answered from general knowledge)
- continue: Uncertain - proceed with retrieval to be safe

Respond with ONLY the decision word (yes, no, or continue)."""
        )

    def evaluate(self, query: str) -> Literal["yes", "no", "continue"]:
        """Evaluate if retrieval is needed for the query."""
        response = self.llm.invoke(
            self.eval_prompt.format(query=query)
        )

        decision = response.content.strip().lower()
        valid_decisions = {"yes", "no", "continue"}

        if decision not in valid_decisions:
            return "continue"  # Default to safe behavior

        return decision

    def should_retrieve(self, query: str) -> bool:
        """Return True if retrieval should proceed."""
        decision = self.evaluate(query)
        return decision in {"yes", "continue"}
