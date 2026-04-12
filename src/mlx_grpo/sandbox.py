"""Sandbox for isolated Python code execution.

Prevents Metal GPU crashes from fork() and restricts model-generated
code to a venv with resource limits and command validation.

Key design: NO preexec_fn — it forces fork() over the fast C-level
fork+exec path, duplicating Metal GPU buffer handles and causing
kIOGPUCommandBufferCallbackErrorImpactingInteractivity crashes.
Resource limits are injected as a Python preamble instead.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

_BLOCKED = re.compile(
    r"\b(rm|rmdir|mv|cp|chmod|chown|chgrp|mkfs|dd|curl|wget|nc|ncat"
    r"|ssh|scp|rsync|ftp|telnet|kill|killall|pkill|shutdown|reboot"
    r"|mount|umount|fdisk|sudo|su|doas|open|osascript|xdg-open)\b",
)

_PYTHON_CMDS = frozenset(("python", "python3", "python3.11", "python3.12", "python3.13"))

# Preamble injected into solution.py — sets resource limits from within
# the child process AFTER exec, so we don't need preexec_fn.
_LIMITS_PREAMBLE = """\
import resource as _R, contextlib as _C
for _w, _s, _h in [
    (_R.RLIMIT_CPU, 30, 30),
    (_R.RLIMIT_FSIZE, 10485760, 10485760),
    (_R.RLIMIT_NOFILE, 64, 64),
    (_R.RLIMIT_DATA, {mem}, {mem}),
]:
    with _C.suppress(ValueError, OSError):
        _R.setrlimit(_w, (_s, _h))
del _R, _C, _w, _s, _h
"""


def validate_command(cmd: str) -> str | None:
    """Return violation description or None if safe."""
    m = _BLOCKED.search(cmd)
    if m:
        return f"blocked command: {m.group(1)}"
    stripped = cmd.strip().split()[0] if cmd.strip() else ""
    if stripped not in _PYTHON_CMDS:
        return f"only python commands allowed, got: {stripped}"
    return None


class Sandbox:
    """Isolated Python execution environment for model-generated code."""

    def __init__(self, timeout: int = 10, max_memory_mb: int = 256) -> None:
        self._timeout = timeout
        self._max_memory_mb = max_memory_mb
        self._venv_dir: Path | None = None
        self._python: str | None = None
        self._preamble = _LIMITS_PREAMBLE.format(
            mem=max_memory_mb * 1024 * 1024,
        )

    def _ensure_venv(self) -> str:
        """Create venv on first call, return python path."""
        if self._python and self._venv_dir and self._venv_dir.exists():
            return self._python
        self._venv_dir = Path(tempfile.mkdtemp(prefix="mlx_sandbox_"))
        venv.create(self._venv_dir, with_pip=False, clear=True)
        self._python = str(self._venv_dir / "bin" / "python")
        return self._python

    def execute(
        self,
        code: str,
        run_cmd: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """Run code safely. Returns (returncode, stdout, stderr)."""
        violation = validate_command(run_cmd)
        if violation:
            return -1, "", f"blocked: {violation}"

        _drain_metal()

        python_path = self._ensure_venv()
        safe_cmd = _rewrite_python_cmd(run_cmd, python_path)

        with tempfile.TemporaryDirectory(prefix="pyv_") as tmpdir:
            # Prepend resource-limit preamble to the code
            (Path(tmpdir) / "solution.py").write_text(
                self._preamble + code,
            )
            env = _clean_env(python_path)
            try:
                r = subprocess.run(
                    safe_cmd,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout or self._timeout,
                    cwd=tmpdir,
                    env=env,
                    close_fds=True,
                    start_new_session=True,
                )
                return r.returncode, r.stdout.strip(), r.stderr.strip()
            except subprocess.TimeoutExpired:
                return -1, "", "timeout"
            except OSError:
                return -1, "", "exec_error"

    def cleanup(self) -> None:
        """Remove venv directory."""
        if self._venv_dir and self._venv_dir.exists():
            shutil.rmtree(self._venv_dir, ignore_errors=True)
            self._venv_dir = None
            self._python = None


def _rewrite_python_cmd(run_cmd: str, python_path: str) -> list[str]:
    """Convert 'python solution.py [args]' to [venv_python, ...]."""
    parts = shlex.split(run_cmd)
    if parts and parts[0] in _PYTHON_CMDS:
        parts[0] = python_path
    return parts


def _drain_metal() -> None:
    """Force-complete pending Metal GPU work before forking."""
    try:
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _clean_env(python_path: str) -> dict[str, str]:
    """Minimal environment — no inherited GPU/Metal state."""
    venv_dir = str(Path(python_path).parent.parent)
    return {
        "PATH": str(Path(python_path).parent),
        "HOME": "/tmp",
        "LANG": "en_US.UTF-8",
        "VIRTUAL_ENV": venv_dir,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
