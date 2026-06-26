import os


DATA_FOLDER = "data/sample_docs"
ALLOWED_FILE_EXTENSIONS = {".pdf", ".txt"}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

FAISS_INDEX_PATH = "faiss_index"
INDEX_METADATA_PATH = "faiss_index/index_metadata.json"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-3-5-haiku-latest",
)