"""Local HTTP server for the MLX-GRPO fine-tune UI.

Start: mlx-grpo serve [--port 7820]
Multiple jobs are queued; at most one runs at a time.
Auth: JWT Bearer tokens — POST /api/auth/register then /api/auth/login.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .server_auth import login as auth_login
from .server_auth import register as auth_register
from .server_auth import verify_token
from .server_jobs import _jobs, _lock, _logs, get_jobs, pause, pub, resume, stop, submit

_CORS = {"http://localhost:3000", "http://127.0.0.1:3000", "https://adeelahmad.github.io"}


class _H(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def _cors(self) -> None:
        o = self.headers.get("Origin", "")
        self.send_header("Access-Control-Allow-Origin", o if o in _CORS else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _auth(self) -> str | None:
        """Extract username from Authorization: Bearer <token> header."""
        raw = self.headers.get("Authorization", "")
        token = raw.removeprefix("Bearer ").strip()
        return verify_token(token) if token else None

    def _unauth(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="mlx-grpo"')
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        p = urlparse(self.path)
        parts = [x for x in p.path.split("/") if x]

        if p.path == "/health":
            with _lock:
                running = sum(1 for j in _jobs.values() if j["status"] == "running")
                pending = sum(1 for j in _jobs.values() if j["status"] == "pending")
            self._json({"status": "ok", "running": running, "pending": pending})
            return

        user = self._auth()
        if user is None:
            self._unauth()
            return

        if p.path == "/api/jobs":
            self._json(get_jobs(user))

        elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            jid = parts[2]
            with _lock:
                job = _jobs.get(jid)
            if job is None:
                self._json({"error": "not found"}, 404)
            elif job.get("user_id") != user:
                self._json({"error": "forbidden"}, 403)
            else:
                self._json(pub(job))

        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "logs":
            jid = parts[2]
            qs = parse_qs(p.query)
            since = int(qs.get("since", ["0"])[0])
            with _lock:
                job = _jobs.get(jid)
            if job is None:
                self._json({"error": "not found"}, 404)
            elif job.get("user_id") != user:
                self._json({"error": "forbidden"}, 403)
            else:
                with _lock:
                    lines = _logs.get(jid, [])[since:]
                    total = len(_logs.get(jid, []))
                self._json({"lines": lines, "total": total})

        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = json.loads(self.rfile.read(n)) if n else {}
        parts = [x for x in self.path.split("/") if x]

        # ── Public auth routes ──────────────────────────────────────────────────
        if self.path == "/api/auth/register":
            ok, msg = auth_register(body.get("username", ""), body.get("password", ""))
            self._json({"ok": ok, "message": msg}, 200 if ok else 400)
            return

        if self.path == "/api/auth/login":
            tok = auth_login(body.get("username", ""), body.get("password", ""))
            if tok:
                self._json({"token": tok, "username": body.get("username", "")})
            else:
                self._json({"error": "Invalid credentials"}, 401)
            return

        # ── Protected routes ────────────────────────────────────────────────────
        user = self._auth()
        if user is None:
            self._unauth()
            return

        if self.path == "/api/train":
            job = submit(body, user)
            self._json({"id": job["id"], "status": job["status"], "name": job["name"]})

        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            ok2, msg2 = stop(parts[2], user)
            code = 200 if ok2 else (403 if msg2 == "forbidden" else 404)
            self._json({"stopped": True} if ok2 else {"error": msg2}, code)

        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "pause":
            ok2, msg2 = pause(parts[2], user)
            code = 200 if ok2 else (403 if msg2 == "forbidden" else 400)
            self._json({"paused": True} if ok2 else {"error": msg2}, code)

        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "resume":
            ok2, msg2 = resume(parts[2], user)
            code = 200 if ok2 else (403 if msg2 == "forbidden" else 400)
            self._json({"resumed": True} if ok2 else {"error": msg2}, code)

        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def _json(self, data: object, code: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


def run(port: int = 7820, host: str = "localhost") -> None:
    srv = HTTPServer((host, port), _H)
    print(f"  MLX-GRPO UI server  →  http://{host}:{port}/health")
    print("  Fine-tune UI        →  http://localhost:3000/mlx-grpo-framework/finetune")
    print("  Ctrl-C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
