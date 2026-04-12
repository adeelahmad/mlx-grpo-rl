"""JWT authentication and user management for the MLX-GRPO server.

Uses only Python stdlib (hashlib, hmac, base64, json, os, time, pathlib).
Tokens are signed HS256 JWTs; passwords use PBKDF2-HMAC-SHA256.
User store lives at ~/.mlx-grpo/users.json.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_SECRET = os.environ.get("MLX_GRPO_JWT_SECRET") or os.urandom(32).hex()
_EXPIRY = 86_400 * 7  # 7-day tokens
_USER_FILE = Path.home() / ".mlx-grpo" / "users.json"
_lock = threading.Lock()
_users: dict[str, str] = {}  # username → "salt$pbkdf2-hash"


def _load() -> None:
    global _users
    if _USER_FILE.exists():
        try:
            _users = json.loads(_USER_FILE.read_text())
        except Exception:
            _users = {}


def _save() -> None:
    _USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USER_FILE.write_text(json.dumps(_users))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _hash_pwd(password: str, salt: str = "") -> str:
    if not salt:
        salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${h.hex()}"


def _verify_pwd(password: str, stored: str) -> bool:
    try:
        salt = stored.split("$", 1)[0]
        return hmac.compare_digest(stored, _hash_pwd(password, salt))
    except Exception:
        return False


def make_token(username: str) -> str:
    """Return a signed HS256 JWT for username."""
    hdr = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    pay = _b64(json.dumps({"sub": username, "exp": int(time.time()) + _EXPIRY}).encode())
    sig = _b64(hmac.digest(_SECRET.encode(), f"{hdr}.{pay}".encode(), "sha256"))
    return f"{hdr}.{pay}.{sig}"


def verify_token(token: str) -> str | None:
    """Validate signature + expiry. Returns username or None."""
    try:
        hdr, pay, sig = token.split(".")
        expected = _b64(hmac.digest(_SECRET.encode(), f"{hdr}.{pay}".encode(), "sha256"))
        if not hmac.compare_digest(sig, expected):
            return None
        data: dict[str, Any] = json.loads(_b64d(pay))
        if data.get("exp", 0) < time.time():
            return None
        return str(data.get("sub", ""))
    except Exception:
        return None


def register(username: str, password: str) -> tuple[bool, str]:
    """Create a new user. Returns (ok, message)."""
    if not username or not password or len(username) > 64 or len(password) < 6:
        return False, "Username required; password >= 6 chars"
    with _lock:
        if username in _users:
            return False, "Username already exists"
        _users[username] = _hash_pwd(password)
        _save()
    return True, "OK"


def login(username: str, password: str) -> str | None:
    """Verify credentials. Returns a JWT token or None."""
    with _lock:
        stored = _users.get(username)
    if stored and _verify_pwd(password, stored):
        return make_token(username)
    return None


_load()
