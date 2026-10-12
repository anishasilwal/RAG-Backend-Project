# Conversational RAG Backend

A modular **Conversational Retrieval-Augmented Generation (RAG) backend** built with FastAPI. The system allows users to upload PDF/TXT documents, process and index their content, ask questions using semantic search and an LLM, maintain multi-turn conversations using Redis, and store interview booking information in a SQL database.


## Features

### Document Ingestion

- Upload PDF and TXT documents
- Extract text from uploaded documents
- Support two chunking strategies:
  - Recursive chunking
  - Fixed-size chunking
- Configurable chunk size and overlap
- Generate embeddings using Sentence Transformers
- Store document vectors in Qdrant
- Store document metadata in SQLite

### Conversational RAG

- Custom RAG pipeline
- Semantic similarity search using Qdrant
- Retrieve relevant document chunks
- Generate answers using Groq LLM
- Multi-turn conversations
- Session-based conversation history
- Redis-based conversational memory
- Source attribution in responses
- Prevents unsupported answers by instructing the LLM to use available document context

### Interview Booking

- Interview booking through REST API
- Collect:
  - Name
  - Email
  - Interview date
  - Interview time
- Store booking information in SQLite
- Return booking confirmation and booking ID

### API Documentation

- Automatic Swagger/OpenAPI documentation using FastAPI
- Interactive API testing through `/docs`

---

# System Architecture

## Overall Architecture

                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         └────────┬────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
        Document Ingestion   Conversational   Interview
                              RAG API          Booking
                 │                │                │
                 ▼                ▼                ▼
          Text Extraction    Redis Memory      SQLite
                 │                │
                 ▼                │
             Chunking             │
                 │                │
                 ▼                │
            Embeddings            │
                 │                │
                 ▼                ▼
              Qdrant ◄──── Semantic Search
                 │
                 ▼
          Relevant Chunks
                 │
                 ▼
             Groq LLM
                 │
                 ▼
              Answer

## Tech Stack

- Python
- FastAPI
- Qdrant
- Redis
- Sentence Transformers
- Groq
- Pydantic
- Uvicorn
- SQLite

## Running the Project

1. Create virtual environment
2. Install dependencies
3. Configure `.env`
4. Start Redis
5. Run FastAPI
6. Open `/docs`