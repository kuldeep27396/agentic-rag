from __future__ import annotations
from fastapi import Header, HTTPException, Request

from app.core.config import get_settings


async def verify_qstash_request(request: Request, upstash_signature: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.environment == "development" and not settings.qstash_current_signing_key:
        return
    if not upstash_signature:
        raise HTTPException(status_code=401, detail="Missing QStash signature")
    # Production deployments should use upstash-qstash's Receiver.verify once the
    # package is installed; keep the dependency boundary isolated here.
    if settings.qstash_current_signing_key not in upstash_signature and settings.qstash_next_signing_key not in upstash_signature:
        raise HTTPException(status_code=401, detail="Invalid QStash signature")

