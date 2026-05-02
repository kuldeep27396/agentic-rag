import re
from typing import List, Tuple

import httpx

from app.core.config import get_settings
from app.schemas.models import Citation

FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-3-27b-it:free",
    "openai/gpt-oss-120b:free",
    "minimax/minimax-m2.5:free",
]

_SUGGESTION_BLOCK = re.compile(r"<suggestions>(.*?)</suggestions>", re.DOTALL)


class OpenRouterClient:
    async def answer(self, question: str, contexts: List[str], citations: List[Citation], *, use_web: bool = False) -> Tuple[str, List[str]]:
        settings = get_settings()
        if not settings.openrouter_api_key:
            joined = "\n\n".join(contexts[:2]).strip()
            if not joined:
                return "I could not find enough information in this PDF to answer that.", []
            return f"Based on the uploaded PDF: {joined[:900]}", []

        prompt = (
            "Answer using only the supplied PDF context unless web context is explicitly present. "
            "Cite the provided citation labels in the answer. "
            "Use markdown formatting (headers, bold, bullet points, numbered lists) to make the answer well-structured and readable.\n\n"
            "After your answer, suggest 3 short follow-up questions the user might ask about this document. "
            "Put them inside a <suggestions> tag, one per line, each prefixed with '- '.\n\n"
            f"Question: {question}\n\nContext:\n" + "\n\n".join(contexts)
        )

        models = [settings.openrouter_chat_model] + FALLBACK_MODELS
        last_error = ""

        for model in models:
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "plugins": [{"id": "web", "engine": "firecrawl", "max_results": 3}] if use_web else [],
                            "temperature": 0.2,
                        },
                    )
                    if response.status_code == 429:
                        last_error = f"Rate limited by {model}"
                        continue
                    response.raise_for_status()
                    data = response.json()
                    if "choices" in data and data["choices"]:
                        raw = data["choices"][0]["message"]["content"]
                        answer, suggestions = self._parse_suggestions(raw)
                        return answer, suggestions
                    error_msg = data.get("error", str(data))
                    last_error = error_msg if isinstance(error_msg, str) else error_msg.get("message", str(data))
                    continue
                except httpx.HTTPStatusError as exc:
                    last_error = f"{model}: {exc.response.text[:200]}"
                    continue

        return f"All LLM providers failed. Last error: {last_error}", []

    @staticmethod
    def _parse_suggestions(raw: str) -> Tuple[str, List[str]]:
        match = _SUGGESTION_BLOCK.search(raw)
        suggestions: List[str] = []
        clean = raw
        if match:
            clean = raw[: match.start()].strip()
            for line in match.group(1).strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:]
                line = line.strip().strip("\"'")
                if line:
                    suggestions.append(line)
        suggestions = suggestions[:3]
        return clean, suggestions


llm_client = OpenRouterClient()
