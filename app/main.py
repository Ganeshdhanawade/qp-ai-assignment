from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
from .utils import parse_document, query_document
from .models import load_model
from .vector_store import VectorStore

app = FastAPI()

# Initialize model and vector store
model = load_model()
vector_store = VectorStore()

class Query(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Parse and chunk document
    text = parse_document(file_path)
    chunks = vector_store.chunk_document(text)

    # Add chunks to vector store
    vector_store.add_chunks(chunks)

    return {"message": "Document uploaded and processed successfully."}

@app.post("/query")
async def query_document_endpoint(query: Query):
    answer = query_document(query.question, vector_store, model)
    if not answer:
        return {"answer": "I don't know the answer"}
    return {"answer": answer}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}