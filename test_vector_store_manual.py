
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


def main() -> None:
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    chunks = [
        "Employees receive 20 days of annual leave.",
        "Interview hours are from 10 AM to 4 PM.",
        "Candidates must provide a valid email address.",
    ]

    embeddings = embedding_service.embed_texts(chunks)

    vector_size = len(embeddings[0])
    vector_store.create_collection(vector_size)

    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test-document-001",
        filename="sample.txt",
    )

    question = "How many days of leave do employees get?"
    query_vector = embedding_service.embed_text(question)

    results = vector_store.search_similar_chunks(
        query_vector=query_vector,
        limit=3,
    )

    for result in results:
        print(result["score"], result["text"])


if __name__ == "__main__":
    main()
