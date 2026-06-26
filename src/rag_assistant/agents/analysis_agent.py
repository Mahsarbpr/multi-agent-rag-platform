from rag_assistant.llm.base_provider import LLMProvider


class AnalysisAgent:
    """Generates the final answer from retrieved context."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, question: str, context: str) -> str:
        prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
        return self.llm.invoke(prompt)