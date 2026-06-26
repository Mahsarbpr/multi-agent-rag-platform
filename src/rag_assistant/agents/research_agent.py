from langchain_core.documents import Document


class ResearchAgent:
    """Prepares retrieved document context for downstream reasoning."""

    def run(
        self,
        question: str,
        context: str,
        documents: list[Document],
    ) -> dict:
        return {
            "question": question,
            "context": context,
            "documents": documents,
        }