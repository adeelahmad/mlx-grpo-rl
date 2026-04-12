"""Job queue management for the MLX-GRPO UI server.

Handles subprocess launching, log streaming, metric extraction,
and the background queue runner (one job at a time).
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_logs: dict[str, list[str]] = {}
MAX_LOGS, MAX_METRICS, MAX_JOBS = 2000, 200, 50

_IT_RE = re.compile(r"\b(?:step|it(?:eration)?)[:\s]+(\d+)\s*/\s*(\d+)", re.I)
_LOSS_RE = re.compile(r"\bloss[=:\s]+([0-9]+(?:\.[0-9]+)?)", re.I)
_REW_RE = re.compile(r"\breward[=:\s]+([0-9]+(?:\.[0-9]+)?)", re.I)


def ts() -> str:
    return datetime.now(UTC).isoformat()


def pub(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal _ keys for API responses."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _parse_line(job: dict[str, Any], line: str) -> None:
    """Extract metrics; caller holds _lock."""
    if m := _IT_RE.search(line):
        job["iteration"] = int(m.group(1))
        if not job["total"]:
            job["total"] = int(m.group(2))
    if m := _LOSS_RE.search(line):
        job["loss"].append(float(m.group(1)))
        if len(job["loss"]) > MAX_METRICS:
            job["loss"] = job["loss"][-MAX_METRICS:]
    if m := _REW_RE.search(line):
        job["reward"].append(float(m.group(1)))
        if len(job["reward"]) > MAX_METRICS:
            job["reward"] = job["reward"][-MAX_METRICS:]


def _stream(jid: str, proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        with _lock:
            _logs[jid].append(line)
            if len(_logs[jid]) > MAX_LOGS:
                _logs[jid] = _logs[jid][-MAX_LOGS:]
            _parse_line(_jobs[jid], line)
    rc = proc.wait()
    with _lock:
        job = _jobs.get(jid, {})
        if job.get("status") == "running":
            job["status"] = "done" if rc == 0 else "error"
        job["finished_at"] = ts()


def launch(job: dict[str, Any]) -> None:
    """Start subprocess for a job. Caller must hold _lock."""
    cfg = job.pop("_yaml")
    fd, path = tempfile.mkstemp(suffix=".yml", prefix="mlxgrpo_")
    with os.fdopen(fd, "w") as fh:
        fh.write(cfg)
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_grpo", "train", "--config", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job.update(status="running", started_at=ts(), pid=proc.pid, _proc=proc)
    threading.Thread(target=_stream, args=(job["id"], proc), daemon=True).start()

    def _cleanup() -> None:
        proc.wait()
        with contextlib.suppress(OSError):
            os.unlink(path)

    threading.Thread(target=_cleanup, daemon=True).start()


def submit(body: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Create and enqueue a job. Returns the job dict."""
    job: dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "name": body.get("name", "run"),
        "model": body.get("model", ""),
        "user_id": user_id,
        "status": "pending",
        "iteration": 0,
        "total": body.get("iterations", 0),
        "loss": [],
        "reward": [],
        "pid": None,
        "created_at": ts(),
        "started_at": None,
        "finished_at": None,
        "_yaml": body.get("config", ""),
    }
    with _lock:
        if len(_jobs) >= MAX_JOBS:
            done = sorted(
                (j for j in _jobs.values() if j["status"] in ("done", "error", "stopped")),
                key=lambda x: x.get("finished_at") or "",
            )
            for dj in done[: max(1, len(done) // 2)]:
                del _jobs[dj["id"]]
                _logs.pop(dj["id"], None)
        _jobs[job["id"]] = job
        _logs[job["id"]] = []
        if not any(j["status"] in ("running", "paused") for j in _jobs.values()):
            launch(job)
    return job


def get_jobs(user_id: str) -> list[dict[str, Any]]:
    """Return all jobs for a user, newest first."""
    with _lock:
        return [
            pub(j)
            for j in sorted(
                (j for j in _jobs.values() if j.get("user_id") == user_id),
                key=lambda x: x["created_at"],
                reverse=True,
            )
        ]


def stop(jid: str, user_id: str) -> tuple[bool, str]:
    """Stop/cancel a job. Returns (ok, message)."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return False, "not found"
        if job.get("user_id") != user_id:
            return False, "forbidden"
        proc = job.get("_proc")
        if proc and proc.poll() is None:
            if job["status"] == "paused":
                proc.send_signal(signal.SIGCONT)  # unblock so SIGTERM is handled
            proc.terminate()
        job.update(status="stopped", finished_at=ts())
    return True, "stopped"


def pause(jid: str, user_id: str) -> tuple[bool, str]:
    """Suspend a running job via SIGSTOP. Returns (ok, message)."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return False, "not found"
        if job.get("user_id") != user_id:
            return False, "forbidden"
        if job["status"] != "running":
            return False, "not running"
        proc = job.get("_proc")
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGSTOP)
        job["status"] = "paused"
    return True, "paused"


def resume(jid: str, user_id: str) -> tuple[bool, str]:
    """Continue a paused job via SIGCONT. Returns (ok, message)."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return False, "not found"
        if job.get("user_id") != user_id:
            return False, "forbidden"
        if job["status"] != "paused":
            return False, "not paused"
        proc = job.get("_proc")
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGCONT)
        job["status"] = "running"
    return True, "resumed"


def _queue_runner() -> None:
    while True:
        time.sleep(1)
        with _lock:
            if any(j["status"] in ("running", "paused") for j in _jobs.values()):
                continue
            oldest = next(
                iter(
                    sorted(
                        (j for j in _jobs.values() if j["status"] == "pending"),
                        key=lambda x: x["created_at"],
                    )
                ),
                None,
            )
            if oldest:
                launch(oldest)


threading.Thread(target=_queue_runner, daemon=True).start()
