from pathlib import Path
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from rag_assistant.config import DATA_FOLDER, ALLOWED_FILE_EXTENSIONS
from rag_assistant.rag_pipeline import RAGPipeline

# Initialize FastAPI app and RAG pipeline 

app = FastAPI()

rag = RAGPipeline()
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading vectorstore...")
    rag.load_or_build_vectorstore()
    print("Vectorstore ready.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

def rebuild_vectorstore() -> None:
    rag.load_or_build_vectorstore()

# Define request model for asking questions
class QuestionRequest(BaseModel):
    question: str

# Define response model for answers with sources
class QuestionResponse(BaseModel):
    answer: str
    sources: list

class UploadResponse(BaseModel):
    file_name: str
    message: str

# Define API endpoints
@app.get("/")
def root():
    return {"message": "RAG API is running"}

# Endpoint to ask a question and get an answer with sources
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    result = rag.ask_question(request.question)

    return QuestionResponse(
        answer=result["answer"],
        sources=result["sources"],
    )

@app.post("/upload", response_model=UploadResponse)
def upload_document(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...)):
    file_name = file.filename

    if file_name is None:
        raise HTTPException(status_code=400, detail="File name is missing.")

    file_extension = Path(file_name).suffix.lower()

    if file_extension not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    data_folder = Path(DATA_FOLDER)
    data_folder.mkdir(parents=True, exist_ok=True)

    save_path = data_folder / file_name

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
    rebuild_vectorstore
    )

    return UploadResponse(
        file_name=file_name,
        message="File uploaded successfully and vectorstore updated.",
    )