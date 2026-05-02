from typing import Dict, List, Set, Tuple

from app.core.config import get_settings
from app.db.repository import SessionRepository
from app.schemas.models import Citation
from app.services.llm import OpenRouterClient
from app.services.vector_store import VectorStore


class PdfRagAgent:
    def __init__(self, repository: SessionRepository, vector_store: VectorStore, llm: OpenRouterClient) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.llm = llm

    async def run(self, document_id: str, filename: str, question: str, *, hybrid: bool = True) -> Tuple[str, List[Citation]]:
        settings = get_settings()
        hits = await self.vector_store.search(document_id, question, settings.retrieval_child_top_k)
        parent_ids: List[str] = []
        scores_by_parent: Dict[str, float] = {}
        for hit in hits:
            if hit.parent_id not in parent_ids:
                parent_ids.append(hit.parent_id)
                scores_by_parent[hit.parent_id] = hit.score
        parent_ids = parent_ids[: settings.retrieval_parent_top_k]
        parents = await self.repository.get_parent_chunks(document_id, parent_ids)
        contexts: List[str] = []
        citations: List[Citation] = []
        seen_parent_ids: Set[str] = set()
        for chunk in parents:
            if chunk.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(chunk.parent_id)
            contexts.append(f"[{chunk.parent_id}] pages {chunk.page_start}-{chunk.page_end}\n{chunk.text}")
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    filename=filename,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=scores_by_parent.get(chunk.parent_id),
                )
            )
        use_web = hybrid and self._needs_web(question, citations)
        answer = await self.llm.answer(question, contexts, citations, use_web=use_web)
        return answer, citations

    def _needs_web(self, question: str, citations: List[Citation]) -> bool:
        if not citations:
            return True
        terms = ("current", "latest", "today", "recent", "news", "web", "internet")
        return any(term in question.lower() for term in terms)
