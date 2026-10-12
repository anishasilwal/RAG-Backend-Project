from openai import OpenAI

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.memory import ConversationMemory
from app.services.vector_store import VectorStore


class RAGService:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise ValueError(
                "Configure LLM_API_KEY in .env"
            )

        if not settings.llm_model:
            raise ValueError(
                "Configure LLM_MODEL in .env"
            )

        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()

    def chat(
        self,
        session_id: str,
        question: str,
    ) -> dict:

        # 1. Get previous conversation
        history = self.memory.get_history(
            session_id
        )

        # 2. Convert question into embedding
        query_vector = self.embedding_service.embed_text(
            question
        )

        # 3. Search Qdrant
        retrieved_chunks = (
            self.vector_store.search_similar_chunks(
                query_vector=query_vector,
                limit=5,
            )
        )

        # 4. Build document context
        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"Source: {chunk.get('filename', 'unknown')}\n"
                f"{chunk.get('text', '')}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # 5. Build conversation messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful conversational RAG assistant. "
                    "Answer questions using the provided document context. "
                    "If the answer cannot be found in the context, "
                    "say that the information is not available in the "
                    "provided documents. Do not invent information."
                ),
            }
        ]

        # Add previous conversation
        messages.extend(history)

        # Add current question + retrieved context
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Document context:\n\n"
                    f"{context}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        # 6. Ask LLM
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = "I could not generate an answer."

        # 7. Save conversation
        self.memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        self.memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # 8. Return answer + sources
        sources = [
            {
                "filename": chunk.get("filename"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            }
            for chunk in retrieved_chunks
        ]

        return {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": sources,
        }