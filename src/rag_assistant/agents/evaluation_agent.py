from rag_assistant.llm.base_provider import LLMProvider


class EvaluationAgent:
    """Evaluates whether the answer is grounded in retrieved context."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, question: str, context: str, answer: str) -> str:
        prompt = f"""
You are an evaluation agent.

Evaluate the answer using ONLY the provided context.

Return a concise evaluation with:
- Groundedness: High, Medium, or Low
- Completeness: High, Medium, or Low
- Confidence: High, Medium, or Low
- Missing information: list anything important missing from the context

Context:
{context}

Question:
{question}

Answer:
{answer}

Evaluation:
"""
        return self.llm.invoke(prompt)