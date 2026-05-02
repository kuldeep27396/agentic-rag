import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import get_settings


class RedisState:
    def __init__(self) -> None:
        self._memory: Dict[str, Tuple[Any, Optional[float]]] = {}

    def _memory_get(self, key: str) -> Any:
        item = self._memory.get(key)
        if not item:
            return None
        value, expires_at = item
        if expires_at and expires_at <= datetime.now(timezone.utc).timestamp():
            self._memory.pop(key, None)
            return None
        return value

    async def get_json(self, key: str) -> Any:
        settings = get_settings()
        if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
            return self._memory_get(key)

        result = await self.pipeline([["GET", key]])
        value = result[0]["result"]
        return json.loads(value) if value else None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        settings = get_settings()
        if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
            expires_at = datetime.now(timezone.utc).timestamp() + ttl_seconds
            self._memory[key] = (value, expires_at)
            return

        await self.pipeline([["SET", key, json.dumps(value), "EX", ttl_seconds]])

    async def delete(self, *keys: str) -> None:
        if not keys:
            return
        settings = get_settings()
        if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
            for key in keys:
                self._memory.pop(key, None)
            return
        await self.pipeline([["DEL", *keys]])

    async def keys(self, pattern: str) -> List[str]:
        settings = get_settings()
        if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
            import fnmatch
            return [k for k in self._memory if fnmatch.fnmatch(k, pattern)]
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.upstash_redis_rest_url}/pipeline",
                headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                json=[["KEYS", pattern]],
            )
            response.raise_for_status()
            return response.json()[0]["result"] or []

    async def pipeline(self, commands: List[List[Any]]) -> List[Dict[str, Any]]:
        settings = get_settings()
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.upstash_redis_rest_url}/pipeline",
                headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                json=commands,
            )
            response.raise_for_status()
            return response.json()


redis_state = RedisState()
