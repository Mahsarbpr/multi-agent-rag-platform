from fastapi import FastAPI
from pydantic import BaseModel

from rag_assistant.rag_pipeline import RAGPipeline

# Initialize FastAPI app and RAG pipeline 

app = FastAPI()

rag = RAGPipeline()
rag.load_or_build_vectorstore()

# Define request model for asking questions
class QuestionRequest(BaseModel):
    question: str

# Define response model for answers with sources
class QuestionResponse(BaseModel):
    answer: str
    sources: list

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