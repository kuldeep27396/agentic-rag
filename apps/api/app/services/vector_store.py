import hashlib
import math
from typing import List, Optional

import httpx
from pymilvus import DataType, MilvusClient

from app.core.config import get_settings
from app.schemas.state import ChunkRecord, SearchHit
from app.services.redis_state import redis_state

_SHARED_COLLECTION = "documents"


def vector_key(document_id: str) -> str:
    return f"vectors:{document_id}"


def build_document_filter(document_id: str) -> str:
    escaped = document_id.replace('"', '\\"')
    return f'document_id == "{escaped}"'


def local_embedding(text: str, dimensions: int = 256) -> List[float]:
    vector = [0.0] * dimensions
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: List[float], right: List[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


async def openrouter_embedding(texts: List[str], model: str) -> List[List[float]]:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for embeddings")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def _get_milvus_client() -> Optional[MilvusClient]:
    settings = get_settings()
    if not settings.zilliz_uri:
        return None
    return MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)


def _ensure_shared_collection(client: MilvusClient, dimensions: int) -> None:
    if client.has_collection(_SHARED_COLLECTION):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimensions)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="document_id", index_type="Trie")
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=_SHARED_COLLECTION, schema=schema, index_params=index_params)


class VectorStore:
    MAX_DOCUMENTS = 5

    async def _cleanup_if_needed(self, current_document_id: str) -> None:
        client = _get_milvus_client()
        if not client or not client.has_collection(_SHARED_COLLECTION):
            return
        try:
            from pymilvus import Collection
            col = Collection(_SHARED_COLLECTION, using=client)
            stats = col.num_entities
        except Exception:
            stats = 0
        if stats <= self.MAX_DOCUMENTS:
            return
        existing = client.query(
            collection_name=_SHARED_COLLECTION,
            filter="document_id != '{}'".format(current_document_id.replace('"', '\\"')),
            output_fields=["document_id"],
            limit=1,
        )
        if existing and existing[0]:
            orphan_ids = {hit["entity"]["document_id"] for hit in existing[0]}
            if orphan_ids:
                for doc_id in orphan_ids:
                    client.delete(collection_name=_SHARED_COLLECTION, filter=build_document_filter(doc_id))

    async def upsert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        settings = get_settings()
        await self._cleanup_if_needed(chunks[0].document_id)
        texts = [chunk.text for chunk in chunks]
        dimensions = settings.embedding_dimensions

        client = _get_milvus_client()
        if client and settings.openrouter_api_key:
            embeddings = await openrouter_embedding([chunk.text for chunk in chunks], settings.openrouter_embedding_model)
            dimensions = len(embeddings[0])
            _ensure_shared_collection(client, dimensions)
            data = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "parent_id": chunk.parent_id,
                    "embedding": emb,
                }
                for chunk, emb in zip(chunks, embeddings)
            ]
            client.upsert(collection_name=_SHARED_COLLECTION, data=data)
        else:
            vectors = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "parent_id": chunk.parent_id,
                    "embedding": local_embedding(chunk.text),
                }
                for chunk in chunks
            ]
            ttl = settings.retention_days * 24 * 60 * 60
            await redis_state.set_json(
                vector_key(chunks[0].document_id),
                vectors,
                ttl,
            )

    async def search(self, document_id: str, query: str, top_k: Optional[int] = None) -> List[SearchHit]:
        settings = get_settings()
        limit = top_k or settings.retrieval_child_top_k

        client = _get_milvus_client()
        if client and client.has_collection(_SHARED_COLLECTION) and settings.openrouter_api_key:
            query_embeddings = await openrouter_embedding([query], settings.openrouter_embedding_model)
            results = client.search(
                collection_name=_SHARED_COLLECTION,
                data=query_embeddings,
                limit=limit,
                filter=build_document_filter(document_id),
                output_fields=["chunk_id", "parent_id"],
            )
            hits: List[SearchHit] = []
            for hit in results[0]:
                entity = hit["entity"]
                hits.append(
                    SearchHit(
                        chunk_id=entity["chunk_id"],
                        parent_id=entity["parent_id"],
                        score=hit["distance"],
                    )
                )
            return hits

        query_vector = local_embedding(query)
        stored = await redis_state.get_json(vector_key(document_id)) or []
        fallback_hits: List[SearchHit] = []
        for item in stored:
            emb = item.get("embedding", [])
            if not emb:
                continue
            fallback_hits.append(
                SearchHit(
                    chunk_id=item["chunk_id"],
                    parent_id=item["parent_id"],
                    score=cosine(query_vector, emb),
                )
            )
        return sorted(fallback_hits, key=lambda h: h.score, reverse=True)[:limit]

    async def delete_collection(self, document_id: str) -> None:
        client = _get_milvus_client()
        if client and client.has_collection(_SHARED_COLLECTION):
            client.delete(collection_name=_SHARED_COLLECTION, filter=build_document_filter(document_id))


vector_store = VectorStore()
