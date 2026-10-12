from fastapi import FastAPI

from app import models
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.database import Base, engine


app = FastAPI(
    title="Conversational RAG Backend",
    description="Document ingestion and conversational RAG API",
    version="1.0.0",
)

Base.metadata.create_all(bind=engine)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


app.include_router(documents_router)
app.include_router(chat_router)