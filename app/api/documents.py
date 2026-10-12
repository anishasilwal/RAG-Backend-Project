from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.database import SessionLocal
from app.models import Document
from app.services.chunking import fixed_size_chunks, recursive_chunks
from app.services.document_loader import extract_text
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

embedding_service = EmbeddingService()
vector_store = VectorStore()


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    if chunking_strategy not in {"recursive", "fixed"}:
        raise HTTPException(
            status_code=400,
            detail="chunking_strategy must be 'recursive' or 'fixed'.",
        )

    if chunk_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="chunk_size must be positive.",
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    document_id = str(uuid4())

    try:
        file_bytes = await file.read()

        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = Path(temporary_file.name)

        text = extract_text(temporary_path)

        if chunking_strategy == "recursive":
            chunks = recursive_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            chunks = fixed_size_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks were generated from the document.",
            )

        embeddings = embedding_service.embed_texts(chunks)

        vector_store.create_collection(
            vector_size=len(embeddings[0])
        )

        vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=file.filename,
        )

        # Save document metadata in SQL database
        db = SessionLocal()

        try:
            document = Document(
                document_id=document_id,
                filename=file.filename,
                file_type=suffix,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_count=len(chunks),
            )

            db.add(document)
            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunking_strategy": chunking_strategy,
            "chunk_count": len(chunks),
            "status": "indexed",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc

    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)