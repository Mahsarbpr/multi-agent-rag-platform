DATA_FOLDER = "data/sample_docs"
ALLOWED_FILE_EXTENSIONS = {".pdf", ".txt"}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

FAISS_INDEX_PATH = "faiss_index"
INDEX_METADATA_PATH = "faiss_index/index_metadata.json"
