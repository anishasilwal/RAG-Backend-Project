# Conversational RAG Backend

A modular FastAPI-based Conversational RAG backend that supports document ingestion, semantic search, multi-turn conversations, and AI-assisted interview booking.

The system uses LangGraph to orchestrate the conversational workflow. An LLM-based router determines whether a user request should be handled by the RAG workflow or the interview booking workflow.


## Features

- PDF and TXT document ingestion
- Text extraction from uploaded documents
- Two chunking strategies:
  - Recursive chunking
  - Fixed-size chunking
- Sentence Transformer embeddings
- Qdrant vector database for semantic search
- Custom RAG implementation
- Groq LLM integration
- LangGraph-based conversational workflow
- LLM-based intent routing
- Multi-turn conversational memory using Redis
- LLM-based extraction of interview booking information
- Interview booking using:
  - Name
  - Email
  - Interview date
  - Interview time
- SQLite database for document metadata and interview bookings
- Source attribution for retrieved document chunks
- FastAPI REST APIs
- Swagger/OpenAPI documentation
- Modular service-based architecture


# Architecture

                         User
                           |
                           v
                    FastAPI /chat
                           |
                           v
                    +-------------+
                    | LangGraph   |
                    | Router Node |
                    +------+------+
                           |
                +----------+----------+
                |                     |
             RAG Route            Booking Route
                |                     |
                v                     v
         +-------------+       +---------------+
         | RAG Node    |       | Booking Node  |
         +------+------+       +-------+-------+
                |                      |
                v                      v
        Sentence Transformer      Groq LLM
                |                      |
                v                 Extract fields
             Qdrant              name/email/date/time
                |                      |
                v                      v
             Context             Missing fields?
                |                 /           \
                |               Yes            No
                |                |              |
                |                v              v
                |           Ask user         SQLite
                |                              |
                +-------------+----------------+
                              |
                              v
                           Response