
from rag_assistant.document_loader import load_documents
from rag_assistant.text_splitter import split_documents
from rag_assistant.vector_store import create_vectorstore, save_vectorstore, load_vectorstore
from rag_assistant.rag_service import find_answer_to_question
from rag_assistant.config import DATA_FOLDER, LLM_MODEL, FAISS_INDEX_PATH, INDEX_METADATA_PATH, OLLAMA_BASE_URL
from langchain_ollama import OllamaLLM
from rag_assistant.index_metadata import (
    calculate_data_fingerprint,
    save_fingerprint,
    data_changed,
)
from pathlib import Path

# RAG Pipeline: Main class to orchestrate the entire Retrieval-Augmented Generation process, including loading documents, 
# creating/loading vectorstore, and answering questions using the RAG approach. 
class RAGPipeline:
    def __init__(self, data_folder: str = DATA_FOLDER, model_name: str = LLM_MODEL, ollama_base_url: str = OLLAMA_BASE_URL):
        self.data_folder = data_folder
        self.vectorstore = None
        self.model_name = model_name
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_base_url,
            )
    def ask_question(self, question: str) -> dict:
        if not self.vectorstore:
            raise ValueError("Vectorstore not created. Please load and process documents first.")
        
        return find_answer_to_question(self.vectorstore, self.llm, question)
    
    # Load documents, split into chunks, create vectorstore, and save it to disk if it doesn't exist or if data has changed
    # since last index creation
    def load_or_build_vectorstore(self) -> None:
        index_exists = Path(FAISS_INDEX_PATH).exists()

        if index_exists and not data_changed(self.data_folder, INDEX_METADATA_PATH):
            print("Loading existing FAISS index...")
            self.vectorstore = load_vectorstore(FAISS_INDEX_PATH)
            return

        print("Building new FAISS index...")

        documents = load_documents(self.data_folder)
        chunks = split_documents(documents)

        self.vectorstore = create_vectorstore(chunks)

        save_vectorstore(self.vectorstore, FAISS_INDEX_PATH)

        fingerprint = calculate_data_fingerprint(self.data_folder)
        save_fingerprint(INDEX_METADATA_PATH, fingerprint)