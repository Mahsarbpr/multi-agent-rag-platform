from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rag_assistant.llm.base_provider import LLMProvider
from rag_assistant.config import LLM_MODEL, TOP_K

# RAG Service: Retrieve relevant documents, build context, and generate answer

# Retrieve similar documents from vectorstore
def retrieve_similar_documents(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
) -> list[Document]:
    return vectorstore.similarity_search(question, k=k)

# Build context from retrieved documents to provide to LLM for answer generation 
def build_context(documents: list[Document]) -> str:
    context_parts = []

    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown source")
        page = doc.metadata.get("page")

        source = file_name
        if page is not None:
            source += f", page {page}"

        context_parts.append(
            f"Source: {source}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)

# Build prompt from question and context
def build_prompt(question: str, context: str) -> str:
    return f"""
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

# Generate answer using LLM and the built prompt 
def generate_answer(
    llm: LLMProvider,
    question: str,
    context: str,
) -> str:
    prompt = build_prompt(question, context)
    return llm.invoke(prompt)

# Main function to find answer to question using RAG approach 
def find_answer_to_question(
    vectorstore: FAISS,
    llm: LLMProvider,
    question: str,
    k: int = TOP_K,
) -> dict:
    retrieved_docs = retrieve_similar_documents(vectorstore, question, k)
    context = build_context(retrieved_docs)
    answer = generate_answer(llm, question, context)

    sources = [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
        }
        for doc in retrieved_docs
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_documents": retrieved_docs,
    }
