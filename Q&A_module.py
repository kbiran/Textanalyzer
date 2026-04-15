# qa_module.py

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline

# Create the FastAPI app instance.
app = FastAPI()

# The loaded Hugging Face pipeline is stored here after startup.
qa_pipeline: Optional[Pipeline] = None

# Request schema for the /qa endpoint.
class QARequest(BaseModel):
    question: str
    context: str

# Response schema for the /qa endpoint.
class QAResponse(BaseModel):
    answer: str
    score: float

# Load the model once when the FastAPI app starts.
# This avoids reloading the model on each request.
@app.on_event("startup")
def load_model() -> None:
    global qa_pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

# Core business logic separated from the endpoint.
# It validates inputs and formats the model output.
def get_answer(question: str, context: str) -> QAResponse:
    if not question.strip() or not context.strip():
        raise HTTPException(status_code=400, detail="Both question and context must be provided.")

    result = qa_pipeline(question=question, context=context)
    return QAResponse(answer=result["answer"], score=result["score"])

# API endpoint for question-answering.
# Accepts JSON payload matching QARequest and returns QAResponse.
@app.post("/qa", response_model=QAResponse)
def answer_question(payload: QARequest) -> QAResponse:
    return get_answer(payload.question, payload.context)
