from collections import defaultdict

import httpx
from fastapi import HTTPException, Request

from app.core.config import get_settings


class RateLimiter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)

    async def check(self, request: Request, *, limit: int = 30, window_seconds: int = 60) -> None:
        settings = get_settings()
        forwarded_for = request.headers.get("x-forwarded-for", "")
        ip = forwarded_for.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        key = f"rate:{ip}:{window_seconds}"

        if settings.upstash_redis_rest_url and settings.upstash_redis_rest_token:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{settings.upstash_redis_rest_url}/pipeline",
                    headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                    json=[["INCR", key], ["EXPIRE", key, window_seconds]],
                )
                response.raise_for_status()
            count = int(response.json()[0]["result"])
        else:
            self._counts[key] += 1
            count = self._counts[key]

        if count > limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")


rate_limiter = RateLimiter()

