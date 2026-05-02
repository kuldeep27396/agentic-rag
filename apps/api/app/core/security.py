import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from app.core.config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def expires_in_days(days: int) -> datetime:
    return utcnow() + timedelta(days=days)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    settings = get_settings()
    secret = settings.session_secret or "development-session-secret"
    return hmac.new(secret.encode(), token.encode(), hashlib.sha256).hexdigest()


def verify_session_token(token: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_session_token(token), expected_hash)

