"""Tests for sandbox execution environment."""

import os

import pytest

from mlx_grpo.sandbox import Sandbox, _clean_env, validate_command


@pytest.fixture
def sandbox():
    sb = Sandbox(timeout=5, max_memory_mb=256)
    yield sb
    sb.cleanup()


class TestValidateCommand:
    def test_python_allowed(self):
        assert validate_command("python solution.py") is None

    def test_python3_allowed(self):
        assert validate_command("python3 solution.py --arg") is None

    def test_rm_blocked(self):
        v = validate_command("rm -rf /")
        assert v and "blocked" in v

    def test_curl_blocked(self):
        v = validate_command("curl http://example.com")
        assert v and "blocked" in v

    def test_wget_blocked(self):
        assert validate_command("wget http://x") is not None

    def test_sudo_blocked(self):
        assert validate_command("sudo python solution.py") is not None

    def test_bash_rejected(self):
        v = validate_command("bash script.sh")
        assert v and "only python" in v

    def test_node_rejected(self):
        v = validate_command("node app.js")
        assert v and "only python" in v

    def test_empty_rejected(self):
        assert validate_command("") is not None

    def test_embedded_rm_blocked(self):
        assert validate_command("python solution.py && rm -rf /") is not None


class TestSandboxExecution:
    def test_valid_python(self, sandbox):
        rc, stdout, stderr = sandbox.execute('print("hello")', "python solution.py")
        assert rc == 0
        assert stdout == "hello"

    def test_syntax_error(self, sandbox):
        rc, stdout, stderr = sandbox.execute("def f(\n", "python solution.py")
        assert rc != 0

    def test_timeout(self, sandbox):
        code = "import time\nwhile True: time.sleep(0.1)"
        rc, stdout, stderr = sandbox.execute(code, "python solution.py", timeout=1)
        assert rc == -1
        assert stderr == "timeout"

    def test_blocked_command(self, sandbox):
        rc, stdout, stderr = sandbox.execute("x", "rm -rf /")
        assert rc == -1
        assert "blocked" in stderr

    def test_non_python_command(self, sandbox):
        rc, stdout, stderr = sandbox.execute("x", "bash script.sh")
        assert rc == -1
        assert "blocked" in stderr

    def test_venv_reused(self, sandbox):
        sandbox.execute("print(1)", "python solution.py")
        venv1 = sandbox._venv_dir
        sandbox.execute("print(2)", "python solution.py")
        venv2 = sandbox._venv_dir
        assert venv1 == venv2
        assert venv1 is not None and venv1.exists()

    def test_cleanup(self):
        sb = Sandbox(timeout=5)
        sb.execute("print(1)", "python solution.py")
        venv_dir = sb._venv_dir
        assert venv_dir and venv_dir.exists()
        sb.cleanup()
        assert not venv_dir.exists()
        assert sb._venv_dir is None

    def test_preamble_sets_limits(self, sandbox):
        """Verify resource-limit preamble is injected and runs."""
        code = "import resource\ncpu = resource.getrlimit(resource.RLIMIT_CPU)\nprint(cpu[0])\n"
        rc, stdout, _ = sandbox.execute(code, "python solution.py")
        assert rc == 0
        assert stdout == "30"

    def test_no_preexec_fn(self, sandbox):
        """Verify subprocess doesn't use preexec_fn (no fork penalty)."""
        # If preexec_fn were used, RLIMIT_NOFILE=64 would be set
        # BEFORE exec. With preamble, it's set AFTER exec by our code.
        # This test just confirms execution works (preexec_fn=None path).
        rc, stdout, _ = sandbox.execute(
            "import os; print(os.getpid())",
            "python solution.py",
        )
        assert rc == 0
        assert int(stdout) > 0

    def test_start_new_session(self, sandbox):
        """Child runs in a new session, isolated from parent signals."""
        code = "import os; print(os.getsid(0))"
        rc, stdout, _ = sandbox.execute(code, "python solution.py")
        assert rc == 0
        child_sid = int(stdout)
        assert child_sid != os.getsid(0)


class TestCleanEnv:
    def test_no_metal_vars(self):
        env = _clean_env("/tmp/venv/bin/python")
        for k in env:
            assert not k.startswith(("DYLD_", "METAL_", "MTL_", "GPU_"))

    def test_has_path(self):
        env = _clean_env("/tmp/venv/bin/python")
        assert "PATH" in env
        assert "/tmp/venv/bin" in env["PATH"]

    def test_has_virtual_env(self):
        env = _clean_env("/tmp/venv/bin/python")
        assert env["VIRTUAL_ENV"] == "/tmp/venv"
