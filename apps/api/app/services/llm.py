from typing import List

import httpx

from app.core.config import get_settings
from app.schemas.models import Citation


class OpenRouterClient:
    async def answer(self, question: str, contexts: List[str], citations: List[Citation], *, use_web: bool = False) -> str:
        settings = get_settings()
        if not settings.openrouter_api_key:
            joined = "\n\n".join(contexts[:2]).strip()
            if not joined:
                return "I could not find enough information in this PDF to answer that."
            return f"Based on the uploaded PDF: {joined[:900]}"

        prompt = (
            "Answer using only the supplied PDF context unless web context is explicitly present. "
            "Cite the provided citation labels in the answer.\n\n"
            f"Question: {question}\n\nContext:\n" + "\n\n".join(contexts)
        )
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                json={
                    "model": settings.openrouter_chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "plugins": [{"id": "web", "engine": "firecrawl", "max_results": 3}] if use_web else [],
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


llm_client = OpenRouterClient()
