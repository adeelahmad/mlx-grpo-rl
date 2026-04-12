"""Tests for server_auth: stdlib HS256 JWT + PBKDF2 user management."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import mlx_grpo.server_auth as auth


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect user store to a temp file and reset in-memory dict."""
    monkeypatch.setattr(auth, "_USER_FILE", tmp_path / "users.json")
    monkeypatch.setattr(auth, "_users", {})


# ── base64 helpers ────────────────────────────────────────────────────────────


def test_b64_roundtrip():
    data = b"hello\x00\xff\xfe world"
    assert auth._b64d(auth._b64(data)) == data


def test_b64_no_padding_chars():
    # urlsafe b64 must not contain '=' padding
    assert "=" not in auth._b64(b"test")


# ── password hashing ──────────────────────────────────────────────────────────


def test_hash_pwd_contains_salt():
    h = auth._hash_pwd("secret")
    assert "$" in h
    salt, hval = h.split("$", 1)
    assert len(salt) == 32  # 16 bytes → 32 hex chars
    assert len(hval) == 64  # SHA-256 → 32 bytes → 64 hex chars


def test_hash_pwd_deterministic_with_explicit_salt():
    h1 = auth._hash_pwd("secret", "deadbeef")
    h2 = auth._hash_pwd("secret", "deadbeef")
    assert h1 == h2


def test_hash_pwd_different_salts_differ():
    h1 = auth._hash_pwd("secret")
    h2 = auth._hash_pwd("secret")
    # Random salts → almost certainly different
    assert h1.split("$")[0] != h2.split("$")[0]


def test_verify_pwd_correct():
    h = auth._hash_pwd("mypassword")
    assert auth._verify_pwd("mypassword", h)


def test_verify_pwd_wrong():
    h = auth._hash_pwd("mypassword")
    assert not auth._verify_pwd("wrongpassword", h)


def test_verify_pwd_malformed_stored():
    # No "$" separator → exception → returns False
    assert not auth._verify_pwd("anything", "noseparatorhere")


# ── JWT ───────────────────────────────────────────────────────────────────────


def test_make_token_three_parts():
    tok = auth.make_token("alice")
    assert tok.count(".") == 2


def test_verify_token_returns_username():
    tok = auth.make_token("alice")
    assert auth.verify_token(tok) == "alice"


def test_verify_token_bad_signature():
    tok = auth.make_token("alice")
    hdr, pay, _ = tok.split(".")
    tampered = f"{hdr}.{pay}.invalidsignature"
    assert auth.verify_token(tampered) is None


def test_verify_token_expired():
    with patch.object(auth, "_EXPIRY", -100):
        tok = auth.make_token("bob")
    assert auth.verify_token(tok) is None


def test_verify_token_only_two_parts():
    assert auth.verify_token("a.b") is None


def test_verify_token_garbage():
    assert auth.verify_token("not-a-token-at-all") is None


def test_verify_token_bad_base64_payload():
    # Header and sig valid but payload undecodable as JSON
    assert auth.verify_token("!!!.!!!.!!!") is None


# ── register ──────────────────────────────────────────────────────────────────


def test_register_success():
    ok, msg = auth.register("alice", "password123")
    assert ok is True
    assert msg == "OK"
    assert "alice" in auth._users


def test_register_saves_to_file(tmp_path, monkeypatch):
    nested = tmp_path / "sub" / "dir" / "users.json"
    monkeypatch.setattr(auth, "_USER_FILE", nested)
    monkeypatch.setattr(auth, "_users", {})
    auth.register("carol", "pass1234")
    assert nested.exists()
    data = json.loads(nested.read_text())
    assert "carol" in data


def test_register_duplicate():
    auth.register("dave", "pass1234")
    ok, msg = auth.register("dave", "pass1234")
    assert ok is False
    assert "already exists" in msg


def test_register_short_password():
    ok, msg = auth.register("eve", "hi")
    assert ok is False
    assert ">= 6" in msg


def test_register_empty_username():
    ok, _ = auth.register("", "password123")
    assert ok is False


def test_register_long_username():
    ok, _ = auth.register("a" * 65, "password123")
    assert ok is False


def test_register_empty_password():
    ok, _ = auth.register("frank", "")
    assert ok is False


# ── login ─────────────────────────────────────────────────────────────────────


def test_login_success():
    auth.register("grace", "correctpass")
    tok = auth.login("grace", "correctpass")
    assert tok is not None
    assert auth.verify_token(tok) == "grace"


def test_login_wrong_password():
    auth.register("harry", "realpass")
    assert auth.login("harry", "wrongpass") is None


def test_login_nonexistent_user():
    assert auth.login("nobody", "pass") is None


# ── _load / _save ─────────────────────────────────────────────────────────────


def test_load_valid_file(tmp_path, monkeypatch):
    user_file = tmp_path / "users.json"
    user_file.write_text(json.dumps({"stored_user": "salt$hash"}))
    monkeypatch.setattr(auth, "_USER_FILE", user_file)
    monkeypatch.setattr(auth, "_users", {})
    auth._load()
    assert "stored_user" in auth._users


def test_load_missing_file_leaves_users_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "_USER_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(auth, "_users", {"preexisting": "val"})
    auth._load()
    assert auth._users == {"preexisting": "val"}


def test_load_corrupt_file_resets_to_empty(tmp_path, monkeypatch):
    bad = tmp_path / "users.json"
    bad.write_text("{{{{ NOT JSON }}}}")
    monkeypatch.setattr(auth, "_USER_FILE", bad)
    monkeypatch.setattr(auth, "_users", {"preexisting": "val"})
    auth._load()
    assert auth._users == {}


def test_save_creates_parent_dirs(tmp_path, monkeypatch):
    deep = tmp_path / "a" / "b" / "c" / "users.json"
    monkeypatch.setattr(auth, "_USER_FILE", deep)
    monkeypatch.setattr(auth, "_users", {"u": "h"})
    auth._save()
    assert deep.exists()
    assert json.loads(deep.read_text()) == {"u": "h"}
