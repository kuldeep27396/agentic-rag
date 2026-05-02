from __future__ import annotations
import hashlib
import math

from app.core.config import get_settings
from app.schemas.state import ChunkRecord, SearchHit, StoredVector
from app.services.redis_state import redis_state


def vector_key(document_id: str) -> str:
    return f"vectors:{document_id}"


def build_document_filter(document_id: str) -> str:
    escaped = document_id.replace('"', '\\"')
    return f'document_id == "{escaped}"'


def local_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class VectorStore:
    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        vectors = [
            StoredVector(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                parent_id=chunk.parent_id,
                embedding=local_embedding(chunk.text),
            )
            for chunk in chunks
        ]
        ttl = get_settings().retention_days * 24 * 60 * 60
        await redis_state.set_json(
            vector_key(chunks[0].document_id),
            [vector.model_dump(mode="json") for vector in vectors],
            ttl,
        )

    async def search(self, document_id: str, query: str, top_k: int | None = None) -> list[SearchHit]:
        settings = get_settings()
        query_vector = local_embedding(query)
        limit = top_k or settings.retrieval_child_top_k
        stored = await redis_state.get_json(vector_key(document_id)) or []
        hits = []
        for item in stored:
            vector = StoredVector.model_validate(item)
            if vector.document_id != document_id:
                continue
            hits.append(
                SearchHit(
                    chunk_id=vector.chunk_id,
                    parent_id=vector.parent_id,
                    score=cosine(query_vector, vector.embedding),
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


vector_store = VectorStore()
