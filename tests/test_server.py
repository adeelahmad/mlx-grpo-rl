"""Tests for server, server_auth, and server_jobs modules.

Tests exercise functions directly (no real HTTP server started).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import mlx_grpo.server_auth as auth
import mlx_grpo.server_jobs as jobs
from mlx_grpo.server import _CORS, _H, run

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_auth(tmp_path, monkeypatch):
    """Redirect user store to tmp and reset in-memory state."""
    monkeypatch.setattr(auth, "_USER_FILE", tmp_path / "users.json")
    monkeypatch.setattr(auth, "_users", {})


@pytest.fixture(autouse=True)
def _isolate_jobs(monkeypatch):
    """Reset job/log state between tests."""
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_logs", {})


# ── server_auth: register ──────────────────────────────────────────────────────


def test_register_success():
    ok, msg = auth.register("alice", "password123")
    assert ok is True
    assert msg == "OK"


def test_register_duplicate_rejected():
    auth.register("bob", "password123")
    ok, msg = auth.register("bob", "password123")
    assert ok is False
    assert "already exists" in msg


def test_register_short_password_rejected():
    ok, _ = auth.register("carol", "hi")
    assert ok is False


def test_register_empty_username_rejected():
    ok, _ = auth.register("", "password123")
    assert ok is False


def test_register_long_username_rejected():
    ok, _ = auth.register("x" * 65, "password123")
    assert ok is False


# ── server_auth: login ─────────────────────────────────────────────────────────


def test_login_returns_token():
    auth.register("dave", "correctpass")
    tok = auth.login("dave", "correctpass")
    assert tok is not None
    assert tok.count(".") == 2


def test_login_wrong_password_returns_none():
    auth.register("eve", "realpass")
    assert auth.login("eve", "wrongpass") is None


def test_login_unknown_user_returns_none():
    assert auth.login("ghost", "password") is None


# ── server_auth: verify_token ──────────────────────────────────────────────────


def test_verify_token_valid():
    auth.register("frank", "password123")
    tok = auth.login("frank", "password123")
    assert tok is not None
    assert auth.verify_token(tok) == "frank"


def test_verify_token_tampered_signature():
    tok = auth.make_token("grace")
    hdr, pay, _ = tok.split(".")
    assert auth.verify_token(f"{hdr}.{pay}.badsig") is None


def test_verify_token_garbage_string():
    assert auth.verify_token("not-a-token") is None


def test_verify_token_expired():
    with patch.object(auth, "_EXPIRY", -100):
        tok = auth.make_token("harry")
    assert auth.verify_token(tok) is None


# ── server_jobs: submit / get_jobs ────────────────────────────────────────────


def test_submit_returns_job_with_fields():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "test-run", "model": "m", "config": "cfg"}, "alice")
    assert job["id"]
    assert job["status"] in ("pending", "running")
    assert job["name"] == "test-run"
    assert job["user_id"] == "alice"


def test_submit_job_appears_in_get_jobs():
    with patch("mlx_grpo.server_jobs.launch"):
        jobs.submit({"name": "myjob", "config": ""}, "bob")
    result = jobs.get_jobs("bob")
    assert len(result) == 1
    assert result[0]["name"] == "myjob"


def test_get_jobs_filters_by_user():
    with patch("mlx_grpo.server_jobs.launch"):
        jobs.submit({"name": "job-a", "config": ""}, "user1")
        jobs.submit({"name": "job-b", "config": ""}, "user2")
    assert len(jobs.get_jobs("user1")) == 1
    assert len(jobs.get_jobs("user2")) == 1
    assert len(jobs.get_jobs("user3")) == 0


def test_get_jobs_no_private_keys():
    with patch("mlx_grpo.server_jobs.launch"):
        jobs.submit({"name": "j", "config": "yaml"}, "carol")
    result = jobs.get_jobs("carol")
    for job in result:
        for key in job:
            assert not key.startswith("_"), f"Private key exposed: {key}"


# ── server_jobs: stop ─────────────────────────────────────────────────────────


def test_stop_nonexistent_job():
    ok, msg = jobs.stop("nonexistent-id", "alice")
    assert ok is False
    assert msg == "not found"


def test_stop_forbidden_for_other_user():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    ok, msg = jobs.stop(job["id"], "intruder")
    assert ok is False
    assert msg == "forbidden"


def test_stop_own_job_succeeds():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    with jobs._lock:
        jobs._jobs[job["id"]]["_proc"] = None
    ok, msg = jobs.stop(job["id"], "owner")
    assert ok is True
    assert jobs._jobs[job["id"]]["status"] == "stopped"


# ── server_jobs: pause / resume ───────────────────────────────────────────────


def test_pause_nonexistent_job():
    ok, msg = jobs.pause("bad-id", "alice")
    assert ok is False
    assert msg == "not found"


def test_pause_forbidden():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    ok, msg = jobs.pause(job["id"], "hacker")
    assert ok is False
    assert msg == "forbidden"


def test_pause_pending_job_returns_not_running():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    # Force pending state
    with jobs._lock:
        jobs._jobs[job["id"]]["status"] = "pending"
    ok, msg = jobs.pause(job["id"], "owner")
    assert ok is False
    assert msg == "not running"


def test_resume_nonexistent_job():
    ok, msg = jobs.resume("bad-id", "alice")
    assert ok is False
    assert msg == "not found"


def test_resume_forbidden():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    ok, msg = jobs.resume(job["id"], "intruder")
    assert ok is False
    assert msg == "forbidden"


def test_resume_non_paused_job():
    with patch("mlx_grpo.server_jobs.launch"):
        job = jobs.submit({"name": "j", "config": ""}, "owner")
    with jobs._lock:
        jobs._jobs[job["id"]]["status"] = "running"
    ok, msg = jobs.resume(job["id"], "owner")
    assert ok is False
    assert msg == "not paused"


# ── server_jobs: pub ──────────────────────────────────────────────────────────


def test_pub_strips_private_keys():
    d = {"id": "abc", "name": "n", "_yaml": "secret", "_proc": None}
    result = jobs.pub(d)
    assert "id" in result
    assert "_yaml" not in result
    assert "_proc" not in result


# ── server.py: run() callable ─────────────────────────────────────────────────


def test_run_function_exists_and_callable():
    assert callable(run)


def test_run_accepts_port_and_host():
    import inspect

    sig = inspect.signature(run)
    assert "port" in sig.parameters
    assert "host" in sig.parameters
    assert sig.parameters["port"].default == 7820
    assert sig.parameters["host"].default == "localhost"


# ── server.py: CORS ───────────────────────────────────────────────────────────


def test_cors_set_includes_localhost_3000():
    assert "http://localhost:3000" in _CORS


def test_cors_set_includes_127_0_0_1():
    assert "http://127.0.0.1:3000" in _CORS


# ── server.py: _H handler (mocked) ────────────────────────────────────────────


def _make_handler(method: str, path: str, headers: dict, body: bytes = b"") -> _H:
    """Create an _H handler instance with a mocked socket."""
    request = MagicMock()
    request.makefile = MagicMock(return_value=MagicMock())
    client_address = ("127.0.0.1", 9999)
    server = MagicMock()

    handler = _H.__new__(_H)
    handler.request = request
    handler.client_address = client_address
    handler.server = server
    handler.command = method
    handler.path = path
    handler.headers = MagicMock()
    handler.headers.get = lambda k, d="": headers.get(k, d)

    handler.headers.__getitem__ = lambda self, k: headers.get(k, "")
    handler.rfile = MagicMock()
    handler.rfile.read = MagicMock(return_value=body)

    # Capture response
    response_parts = []
    handler.wfile = MagicMock()
    handler.wfile.write = lambda data: response_parts.append(data)
    handler._response_parts = response_parts

    sent_headers: dict[str, str] = {}
    sent_status: list[int] = []

    def send_response(code):
        sent_status.append(code)

    def send_header(k, v):
        sent_headers[k] = v

    def end_headers():
        pass

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    handler._sent_status = sent_status
    handler._sent_headers = sent_headers

    return handler


def test_handler_health_endpoint_returns_ok():
    h = _make_handler("GET", "/health", {})
    h.do_GET()
    assert h._sent_status == [200]
    payload = json.loads(b"".join(h._response_parts))
    assert payload["status"] == "ok"
    assert "running" in payload
    assert "pending" in payload


def test_handler_get_jobs_without_token_returns_401():
    h = _make_handler("GET", "/api/jobs", {})
    h.do_GET()
    assert h._sent_status == [401]


def test_handler_options_returns_204():
    h = _make_handler("OPTIONS", "/", {})
    h.do_OPTIONS()
    assert h._sent_status == [204]


def test_handler_register_route_no_auth_needed():
    body = json.dumps({"username": "newuser", "password": "pass123"}).encode()
    h = _make_handler("POST", "/api/auth/register", {"Content-Length": str(len(body))}, body)
    h.do_POST()
    assert h._sent_status == [200]
    payload = json.loads(b"".join(h._response_parts))
    assert payload["ok"] is True


def test_handler_login_route_bad_creds_returns_401():
    body = json.dumps({"username": "nobody", "password": "wrong"}).encode()
    h = _make_handler("POST", "/api/auth/login", {"Content-Length": str(len(body))}, body)
    h.do_POST()
    assert h._sent_status == [401]


def test_handler_protected_post_without_token_returns_401():
    body = json.dumps({"name": "run", "config": ""}).encode()
    h = _make_handler("POST", "/api/train", {"Content-Length": str(len(body))}, body)
    h.do_POST()
    assert h._sent_status == [401]


def test_handler_train_submit_with_valid_token():
    auth.register("testuser", "password123")
    tok = auth.login("testuser", "password123")
    assert tok is not None

    body = json.dumps({"name": "my-run", "model": "m", "config": "cfg"}).encode()
    h = _make_handler(
        "POST",
        "/api/train",
        {"Content-Length": str(len(body)), "Authorization": f"Bearer {tok}"},
        body,
    )
    with patch("mlx_grpo.server_jobs.launch"):
        h.do_POST()
    assert h._sent_status == [200]
    payload = json.loads(b"".join(h._response_parts))
    assert "id" in payload
    assert "status" in payload
    assert "name" in payload


def test_handler_cors_header_set_on_response():
    h = _make_handler("GET", "/health", {"Origin": "http://localhost:3000"})
    h.do_GET()
    assert "Access-Control-Allow-Origin" in h._sent_headers


def test_handler_404_on_unknown_get():
    tok = auth.make_token("someone")
    h = _make_handler("GET", "/api/unknown", {"Authorization": f"Bearer {tok}"})
    h.do_GET()
    assert h._sent_status == [404]
