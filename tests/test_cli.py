"""Tests for CLI commands."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mlx_grpo.cli import main


class TestCLI:
    def test_list_types(self, capsys):
        rc = main(["list-types"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mcq" in out
        assert "math" in out

    def test_list_rewards(self, capsys):
        rc = main(["list-rewards"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "accuracy" in out

    def test_list_backends(self, capsys):
        rc = main(["list-backends"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mlx" in out

    def test_dry_run(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "training": {"iterations": 10, "group_size": 2, "model_path": "test"},
                "generation": {"backend": "mlx"},
            }
            cfg_path = Path(d) / "config.yml"
            cfg_path.write_text(yaml.dump(cfg))

            rc = main(["train", "--config", str(cfg_path), "--dry-run"])
            assert rc == 0
            out = capsys.readouterr().out
            assert "Dry Run" in out
            assert "10" in out

    def test_no_command(self, capsys):
        rc = main([])
        assert rc == 1

    def test_help(self):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
