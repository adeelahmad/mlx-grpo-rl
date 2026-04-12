"""Logging configuration with file + console handlers and distributed-training awareness."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class LoggingConfig:
    """Logging configuration for training."""

    console_level: str = "INFO"  # Console/stdout log level
    file_level: str = "DEBUG"  # File log level (more verbose)
    log_file: str | None = None  # Path to log file; None = no file logging
    log_dir: str | None = None  # Directory for log files; auto-created if needed


def setup_logging(config: LoggingConfig, rank: int = 0) -> None:
    """Configure logging with console and optional file handlers.

    In distributed training, only rank 0 writes to file (prevents multi-process conflicts).
    Console output from all ranks (users can filter if needed).

    Args:
        config: LoggingConfig with level and file settings
        rank: Process rank in distributed training (0 = main process)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (all ranks)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, config.console_level.upper(), logging.INFO))
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (rank 0 only in distributed, or always if rank=0 passed)
    if config.log_file and rank == 0:
        log_path = Path(config.log_file)

        # Create directory if needed
        if config.log_dir:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / log_path.name

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(getattr(logging, config.file_level.upper(), logging.DEBUG))
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Log startup info
        root_logger.debug(f"Logging to file: {log_path} (rank {rank})")


def get_rank() -> int:
    """Get process rank in distributed training context.

    Checks for common distributed training environment variables:
    - RANK (pytorch distributed)
    - OMPI_COMM_WORLD_RANK (OpenMPI)
    - PMI_RANK (Intel MPI)
    - SLURM_PROCID (SLURM)
    """
    for env_var in ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
        rank_str = os.environ.get(env_var)
        if rank_str is not None:
            try:
                return int(rank_str)
            except ValueError:
                pass
    return 0  # Default to rank 0 (no distributed training)
