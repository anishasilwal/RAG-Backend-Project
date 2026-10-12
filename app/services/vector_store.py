
import uuid

from qdrant_client import QdrantClient, models

from app.config import settings


class VectorStore:
    def __init__(self) -> None:
        if not settings.qdrant_url or not settings.qdrant_api_key:
            raise ValueError(
                "Configure QDRANT_URL and QDRANT_API_KEY in .env"
            )

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection

    def create_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Each chunk must have one embedding."
            )

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search_similar_chunks(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "score": point.score,
                **(point.payload or {}),
            }
            for point in results.points
        ]
