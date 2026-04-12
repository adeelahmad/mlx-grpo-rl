"""Command-line interface — thin entry point for training.

python -m mlx_grpo train --config config.yml
python -m mlx_grpo list-types
python -m mlx_grpo list-rewards
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
from pathlib import Path
from typing import Any


def _combine_datasets_balanced(datasets: list, seed: int) -> Any:
    """Combine multiple datasets with hierarchical balancing.

    Hierarchy:
    1. Balance datasets equally (equal samples from each)
    2. Balance types within each dataset
    3. Shuffle within balanced groups

    Args:
        datasets: list of DatasetLoader objects
        seed: random seed for reproducibility

    Returns:
        New DatasetLoader with combined and balanced samples
    """
    from .data import DatasetLoader, Sample

    # Collect samples by (dataset_idx, type_name)
    # This ensures we can balance both datasets and types
    samples_by_dataset_type: dict[tuple[int, str], list[Sample]] = {}
    for ds_idx, ds in enumerate(datasets):
        for sample in ds:
            key = (ds_idx, sample.type_name)
            samples_by_dataset_type.setdefault(key, []).append(sample)

    # Get unique datasets and types
    dataset_indices = sorted(set(k[0] for k in samples_by_dataset_type))
    type_names = sorted(set(k[1] for k in samples_by_dataset_type))

    # Find minimum count for balancing
    # Level 1: equal samples per dataset
    samples_per_ds = {}
    for ds_idx in dataset_indices:
        count = sum(len(samples_by_dataset_type.get((ds_idx, t), [])) for t in type_names)
        samples_per_ds[ds_idx] = count

    min_ds_samples = min(samples_per_ds.values())

    # Level 2: equal samples per type within each dataset
    min_type_samples_per_ds: dict[tuple[int, str], int] = {}
    for ds_idx in dataset_indices:
        type_counts = {}
        for t in type_names:
            count = len(samples_by_dataset_type.get((ds_idx, t), []))
            if count > 0:
                type_counts[t] = count
        if type_counts:
            min_type = min(type_counts.values())
            for t in type_counts:
                min_type_samples_per_ds[(ds_idx, t)] = min_type

    # Shuffle each bucket and balance
    rng = random.Random(seed)
    balanced: list[Sample] = []

    # Round-robin: datasets → types → shuffle within
    ds_idx = 0
    while len(balanced) < min_ds_samples * len(dataset_indices):
        for ds in dataset_indices:
            for t in type_names:
                key = (ds, t)
                bucket = samples_by_dataset_type.get(key, [])
                if not bucket:
                    continue
                # Take one sample (implicitly shuffles via pop)
                if bucket:
                    idx = rng.randint(0, len(bucket) - 1)
                    sample = bucket.pop(idx)
                    balanced.append(sample)
                    # Stop when we reach target balance
                    if len(balanced) >= min_ds_samples * len(dataset_indices):
                        break
            if len(balanced) >= min_ds_samples * len(dataset_indices):
                break

    # Create a synthetic DatasetLoader with combined balanced samples
    # Use first dataset's config as template
    loader = DatasetLoader(datasets[0]._config)
    loader._samples = balanced
    loader._index = 0
    loader._epoch = 0
    return loader


def _run_prechecks(config: Any, model_path: Path, backend: str) -> bool:
    """Run startup prechecks. Return True if all pass, False if any fail."""

    issues = []

    # 1. CUDA checks
    if backend == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                # Test CUDA synchronization
                try:
                    torch.cuda.synchronize()
                    gpu_count = torch.cuda.device_count()
                    print(f"✓ CUDA OK ({gpu_count} GPU{'s' if gpu_count > 1 else ''})")
                except RuntimeError as e:
                    issues.append(f"CUDA sync failed: {e}")

                # Check GPU memory
                try:
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        total = props.total_memory / 1e9
                        allocated = torch.cuda.memory_allocated(i) / 1e9
                        free = total - allocated
                        if free < 0.5:
                            issues.append(f"GPU {i} low memory: {free:.1f}GB free of {total:.1f}GB")
                        else:
                            print(f"✓ GPU {i} memory OK ({free:.1f}GB free)")
                except Exception as e:
                    issues.append(f"GPU memory check failed: {e}")
        except ImportError:
            pass

    # 2. Model path check
    if not Path(model_path).exists():
        issues.append(f"Model path not found: {model_path}")
    else:
        print(f"✓ Model path OK: {model_path}")

    # 3. Adapter directory check
    adapter_path_cfg = config.training.adapter_path
    if adapter_path_cfg:
        _ap = Path(adapter_path_cfg).expanduser()
        ckpt_dir = _ap if _ap.is_absolute() else Path(model_path).parent / _ap
    else:
        ckpt_dir = Path(model_path).parent / "checkpoints"

    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        test_file = ckpt_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Checkpoint dir writable: {ckpt_dir}")
    except (OSError, PermissionError) as e:
        issues.append(f"Checkpoint dir not writable: {ckpt_dir} ({e})")

    # 4. Dataset paths check
    ds_configs = config.datasets or {}
    for name, ds_cfg in ds_configs.items():
        if isinstance(ds_cfg, dict) and ds_cfg.get("path"):
            ds_path = ds_cfg.get("path")
            if not Path(ds_path).exists():
                issues.append(f"Dataset '{name}' path not found: {ds_path}")
            else:
                print(f"✓ Dataset '{name}' path OK")

    # 5. Config sanity checks
    if config.training.batch_size < 1:
        issues.append("batch_size must be >= 1")
    if config.training.learning_rate <= 0:
        issues.append("learning_rate must be > 0")
    if config.training.iterations < 1:
        issues.append("iterations must be >= 1")
    if config.training.group_size < 1:
        issues.append("group_size must be >= 1")

    # Print results
    if issues:
        print("\n⚠️  PRECHECKS FAILED:\n")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print("✓ All prechecks passed\n")
        return True


def _print_env_info(backend: str) -> None:
    """Print environment and backend info at startup."""
    import sys

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_info = f"{platform.system()} {platform.release()}"
    sep = "=" * 60

    print(f"\n{sep}")
    print("🚀 MLX-GRPO Training Framework")
    print(sep)
    print(f"Python:  {py_version}")
    print(f"OS:      {os_info}")

    # Backend-specific info
    if backend == "mlx":
        try:
            import pkg_resources

            mlx_ver = pkg_resources.get_distribution("mlx").version
            print(f"MLX:     {mlx_ver}")
            device = "Metal"
            print(f"Device:  {device}")
        except (ImportError, Exception):
            print("MLX:     (version check unavailable)")
    elif backend == "pytorch":
        try:
            import torch

            print(f"PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                device = "CUDA"
                print(f"Device:  {device}")
                print(f"CUDA:    {torch.version.cuda}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "MPS (Metal)"
                print(f"Device:  {device}")
            else:
                device = "CPU"
                print(f"Device:  {device}")
        except ImportError:
            print("PyTorch: NOT AVAILABLE")
    elif backend == "vllm":
        try:
            import vllm

            print(f"vLLM:    {vllm.__version__}")
        except (ImportError, AttributeError):
            print("vLLM:    (version check unavailable)")

    print(f"Backend: {backend.upper()}")
    print(f"{sep}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlx-grpo",
        description="Template-driven GRPO training framework",
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ──
    train_p = sub.add_parser("train", help="Run GRPO training")
    train_p.add_argument("--config", type=Path, required=True, help="YAML config path")
    train_p.add_argument("--model", type=str, help="Model path (overrides config)")
    train_p.add_argument("--iterations", type=int, help="Training iterations (overrides config)")
    train_p.add_argument("--dry-run", action="store_true", help="Validate config without training")
    train_p.add_argument("--resume", type=Path, help="Resume from checkpoint directory")
    train_p.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="WandB project name (overrides config)",
    )
    train_p.add_argument("--tui", action="store_true", help="Launch Textual TUI dashboard")
    train_p.add_argument(
        "--merge-adapters",
        nargs="+",
        type=Path,
        default=[],
        metavar="DIR",
        help="Fuse these adapter checkpoint dirs into the base model (in order) before training",
    )

    # ── eval ──
    eval_p = sub.add_parser("eval", help="Run evaluation pass")
    eval_p.add_argument("--config", type=Path, required=True, help="YAML config path")
    eval_p.add_argument("--checkpoint", type=Path, help="Checkpoint to evaluate")
    eval_p.add_argument("--max-samples", type=int, default=50, help="Max evaluation samples")

    # ── list-types ──
    sub.add_parser("list-types", help="List registered task types")

    # ── list-rewards ──
    sub.add_parser("list-rewards", help="List registered reward functions")

    # ── list-backends ──
    sub.add_parser("list-backends", help="List available generation backends")

    # ── validate-data ──
    val_p = sub.add_parser("validate-data", help="Validate dataset rows against types")
    val_p.add_argument("--config", type=Path, required=True, help="YAML config path")
    val_p.add_argument(
        "--output-dir", type=Path, default=None, help="Save valid rows as per-type JSONL files"
    )
    val_p.add_argument("--verbose", action="store_true", help="Show per-row issue details")

    # ── serve ──
    serve_p = sub.add_parser("serve", help="Start the MLX-GRPO HTTP API server")
    serve_p.add_argument("--port", type=int, default=7820, help="Port (default 7820)")
    serve_p.add_argument("--host", type=str, default="localhost", help="Host (default localhost)")

    # ── inspect ──
    insp = sub.add_parser("inspect", help="Mechanistic interpretability analysis")
    insp.add_argument("model_path", type=str)
    insp.add_argument("--model-b", type=str, default=None)
    insp.add_argument("--prompt", type=str, default="")
    insp.add_argument("--prompts-file", type=str, default=None)
    insp.add_argument("--control-prompt", type=str, default=None)
    insp.add_argument(
        "--feature",
        type=str,
        default="compare",
        choices=[
            "compare",
            "checkpoint",
            "reward_strat",
            "lora",
            "fingerprint",
            "scaffold",
            "probe_save",
        ],
    )
    insp.add_argument("--save-dir", type=str, default="./inspect_out")
    insp.add_argument("--capture-sublayers", action="store_true")
    insp.add_argument("--trace-generation", action="store_true")
    insp.add_argument("--max-new-tokens", type=int, default=30)
    insp.add_argument("--top-k", type=int, default=5)
    insp.add_argument("--layers", type=str, default="")
    insp.add_argument("--circuit-layers", type=str, default="")
    insp.add_argument("--checkpoints-dir", type=str, default=None)
    insp.add_argument("--rollout-log", type=str, default=None)
    insp.add_argument("--lora-path", type=str, default=None)
    insp.add_argument("--probe-base", type=str, default=None)

    # ── torch-convert ──
    torch_conv = sub.add_parser(
        "torch-convert",
        help="Download and convert HuggingFace models (mlx_lm.convert equivalent)",
    )
    torch_conv.add_argument("model_id", type=str, help="HuggingFace model ID")
    torch_conv.add_argument("output_dir", type=str, help="Local output directory")
    torch_conv.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote code execution",
    )
    torch_conv.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize to 4-bit (QLoRA) or 8-bit during conversion",
    )
    torch_conv.add_argument(
        "--merge-adapters",
        type=str,
        nargs="+",
        default=None,
        help="Adapter paths to merge into base weights",
    )

    # ── torch-quantize ──
    torch_quant = sub.add_parser(
        "torch-quantize",
        help="Quantize a local PyTorch model",
    )
    torch_quant.add_argument("model_dir", type=str, help="Local model directory")
    torch_quant.add_argument("output_dir", type=str, help="Quantized output directory")
    torch_quant.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (default 4 for QLoRA)",
    )

    # ── torch-inspect ──
    torch_inspect = sub.add_parser(
        "torch-inspect",
        help="Inspect PyTorch model config and parameters",
    )
    torch_inspect.add_argument("model_dir", type=str, help="Local model directory")

    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)
    elif args.command == "eval":
        return _cmd_eval(args)
    elif args.command == "list-types":
        return _cmd_list_types()
    elif args.command == "list-rewards":
        return _cmd_list_rewards()
    elif args.command == "list-backends":
        return _cmd_list_backends()
    elif args.command == "validate-data":
        return _cmd_validate_data(args)
    elif args.command == "serve":
        from .server import run as _serve

        _serve(args.port, args.host)
        return 0
    elif args.command == "inspect":
        return _cmd_inspect(args)
    elif args.command == "torch-convert":
        return _cmd_torch_convert(args)
    elif args.command == "torch-quantize":
        return _cmd_torch_quantize(args)
    elif args.command == "torch-inspect":
        return _cmd_torch_inspect(args)
    else:
        parser.print_help()
        return 1


def _cmd_train(args) -> int:
    from .config.loader import load_config

    config = load_config(args.config)

    # ── Setup logging ──
    from .config.logging import get_rank, setup_logging

    setup_logging(config.training.logging, rank=get_rank())

    from dataclasses import replace as _replace

    model_path = args.model or config.training.model_path
    if args.iterations:
        config.training = _replace(config.training, iterations=args.iterations)
    iterations = config.training.iterations

    # ── Seed management ──
    import random as _random

    seed = config.training.seed
    if seed == 0:
        seed = _random.randint(1, 2**31 - 1)
        print(f"Generated random seed: {seed}")
    _random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # ── Print environment info ──
    _print_env_info(config.generation.backend)

    # ── Run prechecks ──
    print("\n🔍 Running prechecks...")
    if not _run_prechecks(config, model_path, config.generation.backend):
        print("❌ Prechecks failed. Fix issues above and retry.")
        return 1

    if args.dry_run:
        print("╭─ Dry Run ─────────────────────╮")
        print(f"│  Model:      {model_path}")
        _ep = config.training.epochs
        if _ep > 0:
            print(f"│  Epochs:     {_ep}")
        print(f"│  Iterations: {iterations}")
        print(f"│  Seed:       {seed}")
        print(f"│  Group size: {config.training.group_size}")
        print(f"│  Loss type:  {config.training.loss_type}")
        print(f"│  Beta:       {config.training.beta}")
        print(f"│  LR:         {config.training.learning_rate}")
        print(f"│  Types:      {', '.join(_get_types())}")
        print(f"│  Rewards:    {', '.join(_get_rewards())}")
        print(f"│  Backend:    {config.generation.backend}")

        # Load dataset to validate
        from .data import DatasetConfig, DatasetLoader

        ds_configs = config.datasets or {}
        for name, ds_cfg in ds_configs.items():
            if isinstance(ds_cfg, dict) and ds_cfg.get("path"):
                if "type" in ds_cfg and "type_name" not in ds_cfg:
                    ds_cfg["type_name"] = ds_cfg.pop("type")
                loader = DatasetLoader(DatasetConfig(**ds_cfg)).load()
                print(f"│  Dataset '{name}': {len(loader)} samples")
                if len(loader) > 0:
                    s = loader[0]
                    print(f'│    Sample 0: type={s.type_name}, prompt="{s.prompt[:50]}..."')

        print("╰───────────────────────────────╯")
        return 0

    # ── Checkpoint directory: use adapter_path from config when set,
    #    otherwise fall back to <config_dir>/checkpoints ──
    adapter_path_cfg = config.training.adapter_path
    if adapter_path_cfg:
        _ap = Path(adapter_path_cfg).expanduser()
        ckpt_dir = _ap if _ap.is_absolute() else args.config.parent / _ap
    else:
        ckpt_dir = args.config.parent / "checkpoints"

    # ── WandB (initialize BEFORE GPU operations — background sync threads
    #    can trigger Metal interactivity timeouts if started mid-training) ──
    wandb_project = args.wandb or config.training.wandb_project
    wandb_logger = None
    if wandb_project:
        import json as _json

        from .training.wandb_logger import WandbLogger

        # Only resume an existing WandB run if there's a valid checkpoint
        # to continue from. After a crash (SIGABRT), wandb_run.json may
        # exist but training_state has iteration=0 → step conflict.
        wandb_run_id = None
        _ts_path = ckpt_dir / "training_state.json"
        if _ts_path.exists():
            try:
                _ts = _json.loads(_ts_path.read_text())
                if _ts.get("iteration", 0) > 0:
                    wandb_run_id = WandbLogger.load_run_id(ckpt_dir / "wandb_run.json")
            except (ValueError, OSError):
                pass
        wandb_logger = WandbLogger(
            project=wandb_project,
            config=config.raw,
            run_id=wandb_run_id,
        )
        wandb_logger.log_artifact("config", "config", [args.config])

    # ── Load dataset ──
    from .data import DatasetConfig, DatasetLoader

    ds_configs = config.datasets or {"default": {"path": "", "type_name": "mcq"}}
    datasets = []
    # Propagate training-level shuffle settings to per-dataset configs
    _shuffle = config.training.shuffle_data
    _balanced = config.training.balanced_shuffle
    _seed = config.training.shuffle_seed or random.randint(0, 2**31)
    for name, ds_cfg in ds_configs.items():
        if isinstance(ds_cfg, dict) and ds_cfg.get("path"):
            # Normalize 'type' → 'type_name' (YAML uses 'type', dataclass uses 'type_name')
            if "type" in ds_cfg and "type_name" not in ds_cfg:
                ds_cfg["type_name"] = ds_cfg.pop("type")
            ds_cfg.setdefault("shuffle", _shuffle)
            ds_cfg.setdefault("balanced_shuffle", _balanced)
            ds_cfg.setdefault("seed", _seed)
            loader = DatasetLoader(DatasetConfig(**ds_cfg)).load()
            datasets.append(loader)
            print(f"📊 Dataset '{name}': {len(loader)} samples")

    if not datasets:
        print("❌ No datasets configured. Add 'datasets:' section to config.")
        return 1

    # Combine multiple datasets with hierarchical balancing
    if len(datasets) > 1 and _balanced:
        dataset = _combine_datasets_balanced(datasets, _seed)
    else:
        dataset = datasets[0]  # Primary dataset

    # ── Epochs → iterations conversion ──
    epochs = config.training.epochs
    if epochs > 0 and not args.iterations:
        import math as _math

        bs = config.training.batch_size
        iterations = epochs * _math.ceil(len(dataset) / max(bs, 1))
        config.training = _replace(config.training, iterations=iterations)
        print(f"   {epochs} epoch(s) x {len(dataset)} samples / {bs} batch = {iterations} iters")

    # ── Load backend ──
    from .backends.base import BackendRegistry

    _ensure_builtins()

    backend_name = config.generation.backend
    _nw = getattr(config.training, "rollout_workers", 1)
    _wmode = getattr(config.training, "rollout_worker_mode", "thread")
    _use_process_pool = backend_name in ("mlx", "pytorch") and _wmode == "process" and _nw > 1

    model = None
    ref_model = None
    tokenizer = None
    optimizer = None
    backend = None

    if _use_process_pool:
        print(f"🔧 Backend: {backend_name} [process-pool, {_nw} workers — model lives in workers]")
        _ensure_builtins()
    else:
        print(f"🔧 Backend: {backend_name}")
        try:
            backend = BackendRegistry.get(
                backend_name,
                model_path=model_path,
            )
        except (KeyError, ImportError) as e:
            print(f"❌ Backend '{backend_name}' not available: {e}")
            return 1

    if not _use_process_pool and backend_name == "mlx":
        try:
            from .training.model_loader import (
                AdapterConfig,
                AdapterType,
                ModelConfig,
                ModelLoader,
                count_params,
            )

            adapter_type_str = getattr(config.training, "adapter_type", "lora")
            adapter_type = AdapterType.from_str(adapter_type_str)
            print(f"🎯 Training mode: {adapter_type.value}")

            adapter_cfg = AdapterConfig(
                rank=getattr(config.training, "lora_rank", 8),
                alpha=getattr(config.training, "lora_alpha", 16.0),
                target_modules=getattr(
                    config.training,
                    "lora_targets",
                    [
                        "self_attn.q_proj",
                        "self_attn.v_proj",
                    ],
                ),
                layers=getattr(config.training, "lora_layers", None),
            )
            model_cfg = ModelConfig(
                model_path=model_path,
                adapter_path=str(args.resume) if args.resume else None,
                adapter_type=adapter_type_str,
                adapter=adapter_cfg,
                freeze_base=(adapter_type != AdapterType.FULL),
                grad_checkpoint=getattr(config.training, "grad_checkpoint", True),
                moe_cfg=config.moe,
            )
            loader = ModelLoader(model_cfg)
            model, ref_model, tokenizer = loader.load()

            # Ensure eval mode: LoRA layers default to training=True (dropout
            # active) even when the base model is in eval mode. GRPO needs
            # deterministic forward passes for consistent ref/policy KL.
            model.eval()

            total, trainable = count_params(model, adapter_type)
            print(f"🧠 Model loaded: {total:,} params ({trainable:,} trainable)")

            # Validate weights for NaN/Inf corruption (skip for quantized models —
            # the per-parameter isnan/isinf checks can trigger Metal GPU timeout)
            # from .training.model_loader import validate_weights, CorruptionError
            # try:
            #     validate_weights(model, label="loaded model")
            # except CorruptionError as ce:
            #     print(f"❌ {ce}")
            #     return CorruptionError.EXIT_CODE

            # Create optimizer
            from .training.optimizer import build_optimizer, describe_optimizer

            optimizer = build_optimizer(config.training)
            print(f"   {describe_optimizer(optimizer, config.training)}")

            # Load draft model for speculative decoding (if configured)
            draft_model = None
            if config.generation.draft_model_path:
                from mlx_lm import load as _mlx_load

                draft_model, _ = _mlx_load(config.generation.draft_model_path)
                print(f"   Draft model: {config.generation.draft_model_path}")

            # Force-materialize all model weights NOW so the first generate
            # call doesn't hit Metal with model-load + KV-prefill + generation
            # in one command buffer (which triggers the watchdog timeout).
            # Chunk the eval to avoid a single huge Metal command buffer.
            from mlx.utils import tree_flatten as _tf

            _all_params = [v for _, v in _tf(model.parameters())]
            for _ci in range(0, len(_all_params), 50):
                mx.eval(*_all_params[_ci : _ci + 50])
            mx.synchronize()

            # Wire the cache AFTER materialization so Metal knows the true
            # working set size. Calling this at startup (cache=0) wires
            # nothing, starving all subsequent operations.
            mx.clear_cache()

            # Re-create backend with loaded model/tokenizer + KV/draft options
            from .backends.base import BackendRegistry

            backend = BackendRegistry.get(
                "mlx",
                model=model,
                tokenizer=tokenizer,
                kv_bits=config.generation.kv_bits,
                kv_group_size=config.generation.kv_group_size,
                max_kv_size=config.generation.max_kv_size,
                draft_model=draft_model,
                prefill_step_size=config.generation.prefill_step_size,
                quantized_kv_start=config.generation.quantized_kv_start,
            )

        except ImportError as e:
            print(f"⚠️  MLX not available, running rollout-only mode: {e}")

    elif not _use_process_pool and backend_name == "mlx_batch_generate":
        try:
            from .training.model_loader import (
                AdapterConfig,
                AdapterType,
                ModelConfig,
                ModelLoader,
                count_params,
            )

            adapter_type_str = getattr(config.training, "adapter_type", "lora")
            adapter_type = AdapterType.from_str(adapter_type_str)
            print(f"🎯 Training mode: {adapter_type.value} (MLX batch_generate)")

            adapter_cfg = AdapterConfig(
                rank=getattr(config.training, "lora_rank", 8),
                alpha=getattr(config.training, "lora_alpha", 16.0),
                target_modules=getattr(
                    config.training,
                    "lora_targets",
                    [
                        "self_attn.q_proj",
                        "self_attn.v_proj",
                    ],
                ),
                layers=getattr(config.training, "lora_layers", None),
            )
            model_cfg = ModelConfig(
                model_path=model_path,
                adapter_path=str(args.resume) if args.resume else None,
                adapter_type=adapter_type_str,
                adapter=adapter_cfg,
                freeze_base=(adapter_type != AdapterType.FULL),
                grad_checkpoint=getattr(config.training, "grad_checkpoint", True),
                moe_cfg=config.moe,
            )
            loader = ModelLoader(model_cfg)
            model, ref_model, tokenizer = loader.load()

            model.eval()

            total, trainable = count_params(model, adapter_type)
            print(f"🧠 Model loaded: {total:,} params ({trainable:,} trainable)")

            from .training.optimizer import build_optimizer, describe_optimizer

            optimizer = build_optimizer(config.training)
            print(f"   {describe_optimizer(optimizer, config.training)}")

            draft_model = None
            if config.generation.draft_model_path:
                from mlx_lm import load as _mlx_load

                draft_model, _ = _mlx_load(config.generation.draft_model_path)
                print(f"   Draft model: {config.generation.draft_model_path}")

            from mlx.utils import tree_flatten as _tf

            _all_params = [v for _, v in _tf(model.parameters())]
            for _ci in range(0, len(_all_params), 50):
                mx.eval(*_all_params[_ci : _ci + 50])
            mx.synchronize()

            mx.clear_cache()

            from .backends.base import BackendRegistry

            backend = BackendRegistry.get(
                "mlx_batch_generate",
                model=model,
                tokenizer=tokenizer,
            )

        except ImportError as e:
            print(f"⚠️  MLX not available, running rollout-only mode: {e}")

    elif not _use_process_pool and backend_name == "pytorch":
        try:
            import torch as _torch

            from .training.model_loader import AdapterConfig, AdapterType, ModelConfig
            from .training.torch_model_loader import TorchModelLoader, count_params

            adapter_type_str = getattr(config.training, "adapter_type", "lora")
            adapter_type = AdapterType.from_str(adapter_type_str)
            print(f"🎯 Training mode: {adapter_type.value} (PyTorch)")

            adapter_cfg = AdapterConfig(
                rank=getattr(config.training, "lora_rank", 8),
                alpha=getattr(config.training, "lora_alpha", 16.0),
                target_modules=getattr(
                    config.training,
                    "lora_targets",
                    [
                        "self_attn.q_proj",
                        "self_attn.v_proj",
                    ],
                ),
                layers=getattr(config.training, "lora_layers", None),
            )
            model_cfg = ModelConfig(
                model_path=model_path,
                adapter_path=str(args.resume) if args.resume else None,
                adapter_type=adapter_type_str,
                adapter=adapter_cfg,
                freeze_base=(adapter_type != AdapterType.FULL),
                grad_checkpoint=getattr(config.training, "grad_checkpoint", True),
                vision=config.training.vision,
            )
            loader = TorchModelLoader(model_cfg)
            model, ref_model, tokenizer, processor = loader.load()

            total, trainable = count_params(model)
            print(f"🧠 Model loaded: {total:,} params ({trainable:,} trainable)")

            lr = config.training.learning_rate
            optimizer = _torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=0.01,
            )

            # Re-create backend with loaded model/tokenizer and optional processor
            from .backends.base import BackendRegistry

            backend = BackendRegistry.get(
                "pytorch",
                model=model,
                tokenizer=tokenizer,
                processor=processor,
            )

        except ImportError as e:
            print(f"⚠️  PyTorch/PEFT not available, running rollout-only mode: {e}")

    # ── Create trainer ──
    from .training import Trainer
    from .training.checkpoint import CheckpointManager
    from .training.monitor import Monitor, MonitorConfig
    from .training.recovery import RecoveryManager

    trainer = Trainer(
        config=config.training,
        dataset=dataset,
        backend=backend,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_completion_tokens=config.generation.max_tokens,
        scaffold=config.scaffold,
        max_grad_norm=config.gradient.max_grad_norm,
        used_tokens=config.generation.used_tokens,
        truncate_from=config.generation.truncate_from,
        truncate_marker=config.generation.truncate_marker,
        truncate_marker_masked=config.generation.truncate_marker_masked,
        snap_to_boundary=config.generation.snap_to_boundary,
        type_overrides=config.type_overrides,
        system_prompt=config.generation.system_prompt,
        is_cfg=config.importance_sampling,
        dv_cfg=config.dynamic_views,
        esc_cfg=config.escalation,
        generation_config=config.generation,
        gen_defaults={
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "min_p": config.generation.min_p,
            "repetition_penalty": config.generation.repetition_penalty,
            "repetition_context_size": config.generation.repetition_context_size,
            "xtc_probability": config.generation.xtc_probability,
            "xtc_threshold": config.generation.xtc_threshold,
            "min_tokens_to_keep": config.generation.min_tokens_to_keep,
            # Per-call Metal-safe chunk size. ctx.generate() loops in chunks of this
            # size until the caller's max_tokens budget is exhausted or a stop
            # sequence fires. Prevents any single Metal command from exceeding the
            # watchdog limit regardless of how large a thinking_budget is requested.
            "max_tokens": config.generation.max_tokens,
        },
    )

    # ── Process pool (N-copy mode: workers own model, main is coordinator) ──
    if _use_process_pool:
        from .training.parallel import ProcessWorkerPool
        from .training.process_worker import _build_trainer_kwargs

        trainer._process_pool = ProcessWorkerPool(
            n_workers=_nw,
            model_path=model_path,
            full_config=config,
            yaml_path=str(args.config),
            trainer_kwargs=_build_trainer_kwargs(trainer),
            ckpt_dir=str(ckpt_dir),
        )

    # ── Attach monitor ──
    monitor = Monitor(
        MonitorConfig(
            enabled=config.monitor.enabled,
            kl_warning=config.monitor.kl_warning,
            kl_critical=config.monitor.kl_critical,
            stop_on_critical=config.monitor.stop_on_critical,
        )
    )
    trainer.on("after_step", monitor.step_hook)

    # ── Attach spike guard (skipped in process mode — worker 0 manages its own) ──
    if config.spike_guard is not None and not _use_process_pool:
        from .training.spike_guard import SpikeGuard

        _sg = SpikeGuard(
            config=config.spike_guard,
            adapter_path=config.training.adapter_path,
        )
        trainer.set_spike_guard(_sg)
        print(
            f"[spike-guard] enabled | kl_threshold={config.spike_guard.kl_threshold}"
            f" | action={config.spike_guard.action}"
            f" | max_consecutive={config.spike_guard.max_consecutive}"
        )

    # ── Attach checkpoint manager ──
    _save_optimizer = getattr(config.training, "save_optimizer", True)
    if backend_name == "pytorch":
        from .training.torch_checkpoint import TorchCheckpointManager

        ckpt_mgr = TorchCheckpointManager(
            ckpt_dir,
            keep_recent=config.training.keep_last_n_checkpoints,
            keep_best=config.training.keep_best_n_checkpoints,
        )
    elif backend_name == "mlx" and getattr(config.training, "adapter_type", "lora") == "full":
        from .training.full_checkpoint import FullFTCheckpointManager

        ckpt_mgr = FullFTCheckpointManager(
            ckpt_dir,
            keep_recent=config.training.keep_last_n_checkpoints,
            keep_best=config.training.keep_best_n_checkpoints,
            save_optimizer=_save_optimizer,
            model_source_path=model_path,
        )
    else:
        ckpt_mgr = CheckpointManager(
            ckpt_dir,
            keep_recent=config.training.keep_last_n_checkpoints,
            keep_best=config.training.keep_best_n_checkpoints,
            save_optimizer=_save_optimizer,
        )

    # Snapshot old adapter_config.json BEFORE save_config overwrites it.
    # Used below to fuse the old adapter into base weights when the adapter
    # config changes mid-run (rank change, DoRA→LoRA, etc.).
    _old_adapter_cfg: dict = {}
    _old_cfg_path = ckpt_dir / "adapter_config.json"
    if _old_cfg_path.exists():
        import json as _json_cfg

        with open(_old_cfg_path) as _f_old:
            _old_adapter_cfg = _json_cfg.load(_f_old)

    # Build adapter_config.json (mlx_lm-compatible flat format for MLX,
    # informational for PyTorch — PEFT saves its own config).
    _adapter_type_str = getattr(config.training, "adapter_type", "lora")
    if _adapter_type_str == "lora_moe" and config.moe is not None:
        _moe = config.moe
        adapter_config = {
            "fine_tune_type": "lora_moe",
            "model": model_path,
            "num_experts": _moe.num_experts,
            "lora_rank": _moe.lora_rank,
            "lora_alpha": _moe.lora_alpha,
            "lora_layers": _moe.lora_layers or -1,
            "lora_targets": list(_moe.lora_targets),
        }
    else:
        lora_rank = getattr(config.training, "lora_rank", 8)
        lora_alpha = getattr(config.training, "lora_alpha", 16.0)
        adapter_config = {
            "fine_tune_type": _adapter_type_str,
            "model": model_path,
            "num_layers": getattr(config.training, "lora_layers", None) or -1,
            "lora_parameters": {
                "rank": lora_rank,
                "scale": lora_alpha / max(lora_rank, 1),
                "dropout": getattr(config.training, "lora_dropout", 0.0),
                "keys": getattr(
                    config.training,
                    "lora_targets",
                    [
                        "self_attn.q_proj",
                        "self_attn.v_proj",
                    ],
                ),
            },
        }
    ckpt_mgr.save_config(adapter_config)

    def _ckpt_hook(t, m):
        ckpt_mgr.save(
            m.iteration,
            t.model,
            optimizer=t.optimizer,
            learning_rate=config.training.learning_rate,
            wandb_run_id=wandb_logger.run_id if wandb_logger else None,
            metrics={
                "mean_reward": m.mean_reward,
                "loss": m.loss,
                "kl_divergence": m.kl_divergence,
            },
        )
        if wandb_logger:
            _is_full = getattr(config.training, "adapter_type", "lora") == "full"
            if _is_full:
                # Full FT model weights can be multi-GB — skip unless opted in.
                if getattr(config.training, "wandb_log_full_model", False):
                    _art_files = [ckpt_dir / "latest" / "adapters.safetensors"]
                    wandb_logger.log_artifact(
                        "checkpoint",
                        "model",
                        _art_files,
                        metadata={"iteration": m.iteration},
                    )
            else:
                wandb_logger.log_artifact(
                    "adapter",
                    "model",
                    [ckpt_dir / "adapters.safetensors", ckpt_dir / "adapter_config.json"],
                    metadata={"iteration": m.iteration},
                )

    # In process mode, worker 0 handles checkpointing autonomously
    if not _use_process_pool:
        trainer.on("on_checkpoint", _ckpt_hook)

    # ── Attach evaluator (skipped in process mode — no executor in main) ──
    eval_ds_for_probe = datasets[1] if len(datasets) > 1 else None
    if eval_ds_for_probe is not None and not _use_process_pool:
        from .training.evaluation import Evaluator

        evaluator = Evaluator(
            trainer.executor,
            eval_dataset=eval_ds_for_probe,
            type_overrides=config.type_overrides,
            system_prompt=config.generation.system_prompt,
        )

        def _eval_hook(t, m):
            result = evaluator.evaluate(iteration=m.iteration)
            print(f"  📈 Eval: reward={result.mean_reward:.3f} ({result.total_samples} samples)")

        trainer.on("on_eval", _eval_hook)

    # ── Scaffold probe (skipped in process mode — no executor in main) ──
    if not _use_process_pool:
        from .training.scaffold_probe import ScaffoldProbe

        scaffold_probe = ScaffoldProbe(
            executor=trainer.executor,
            eval_dataset=eval_ds_for_probe,
            train_dataset=dataset,
            ratios=config.training.scaffold_probe_ratios,
            samples_per_ratio=config.training.scaffold_probe_samples,
            scaffold_target=config.scaffold.target,
        )

        def _scaffold_probe_hook(t, m):
            probe_result = scaffold_probe.run(iteration=m.iteration)
            m.extras["scaffold_probe"] = probe_result
            parts = []
            for r in sorted(probe_result.per_ratio_accuracy):
                mean = probe_result.per_ratio_accuracy[r]
                std = probe_result.per_ratio_reward_std.get(r, 0.0)
                parts.append(f"{int(r * 100)}%={mean:.3f}\u00b1{std:.3f}")
            print(f"  Scaffold probe: {' | '.join(parts)}")
            if wandb_logger:
                wb: dict[str, float] = {}
                for r in probe_result.per_ratio_accuracy:
                    tag = f"{int(r * 100)}pct"
                    wb[f"scaffold/mean_{tag}"] = probe_result.per_ratio_accuracy[r]
                    wb[f"scaffold/std_{tag}"] = probe_result.per_ratio_reward_std.get(r, 0.0)
                wb["scaffold/unassisted"] = probe_result.unassisted_accuracy
                wandb_logger.log_step(wb, step=m.iteration)
            stats_logger.log_scaffold(probe_result, m.iteration)

        trainer.on("on_eval", _scaffold_probe_hook)

    # ── WandB hook (logger already initialized before GPU operations) ──
    if wandb_logger:

        def _wandb_hook(t, m):
            wb = {
                "loss": m.loss,
                "reward": m.mean_reward,
                "reward_std": m.reward_std,
                "kl": m.kl_divergence,
                "grad_norm": m.grad_norm,
                "lr": m.learning_rate,
                "tokens_per_sec": m.tokens_per_sec,
                "generated_tokens": m.generated_tokens,
                "trained_tokens": m.extras.get("trained_tokens", m.generated_tokens),
                "total_views": m.extras.get("total_views", 0),
                "peak_memory_mb": m.peak_memory_mb,
                "rss_mb": m.rss_mb,
            }
            type_stats = m.extras.get("type_stats", {})
            for tn, ts in type_stats.items():
                rews = ts["rewards"]
                advs = ts["advantages"]
                wb[f"type/{tn}/reward"] = sum(rews) / len(rews) if rews else 0
                wb[f"type/{tn}/advantage"] = sum(advs) / len(advs) if advs else 0
                wb[f"type/{tn}/correct"] = ts["correct"]
                wb[f"type/{tn}/incorrect"] = ts["incorrect"]
                wb[f"type/{tn}/gen_tokens"] = ts["gen_tokens"]
                wb[f"type/{tn}/trained_tokens"] = ts["trained_tokens"]
                wb[f"type/{tn}/count"] = ts["count"]
            view_stats = m.extras.get("view_stats", {})
            for vn, vs in view_stats.items():
                rews, advs = vs["rewards"], vs["advantages"]
                wb[f"view/{vn}/reward"] = sum(rews) / len(rews) if rews else 0.0
                wb[f"view/{vn}/advantage"] = sum(advs) / len(advs) if advs else 0.0
                wb[f"view/{vn}/scaffold_ratio"] = vs.get("scaffold_ratio", 0.0)
            naked_rews = [
                r for vs in view_stats.values() if vs.get("is_naked") for r in vs["rewards"]
            ]
            scaffolded_rews = [
                r for vs in view_stats.values() if not vs.get("is_naked") for r in vs["rewards"]
            ]
            if naked_rews and scaffolded_rews:
                wb["scaffold_gap"] = sum(scaffolded_rews) / len(scaffolded_rews) - sum(
                    naked_rews
                ) / len(naked_rews)
            is_stats = m.extras.get("is_weight_stats", [])
            if is_stats:
                all_means = [s["mean"] for s in is_stats]
                wb["is_weights/mean"] = sum(all_means) / len(all_means)
                wb["is_weights/min"] = min(s["min"] for s in is_stats)
                wb["is_weights/max"] = max(s["max"] for s in is_stats)
                for i, s in enumerate(is_stats):
                    wb[f"is_weights/r{i}/mean"] = s["mean"]
            # sample_repeat metrics
            _rep_idx = m.extras.get("repeat_idx", 0)
            if _rep_idx:
                wb["repeat/idx"] = _rep_idx
                wb["repeat/dataset_row"] = m.extras.get("dataset_row", -1)
                _reason = m.extras.get("repeat_done_reason", "")
                if _reason:  # only on final rep
                    # Encode reason as int for WandB charting (0=max_reached, 1=fixed,
                    # 2=non_forced_correct, 3=naked_correct)
                    _reason_map = {
                        "max_reached": 0,
                        "fixed": 1,
                        "non_forced_correct": 2,
                        "naked_correct": 3,
                    }
                    wb["repeat/done_reason"] = _reason_map.get(_reason, -1)
                    _rt = m.extras.get("repeat_total")
                    if _rt is not None:
                        wb["repeat/total_reps"] = _rt
            for lk, deg in m.extras.get("weight_angles", {}).items():
                wb[f"weight_angle/{lk}"] = deg
            wandb_logger.log_step(wb, step=int(m.iteration))

        trainer.on("after_step", _wandb_hook)

    # ── Rollout logger ──
    from .training.rollout_logger import RolloutLogger, StepContext

    rollout_log_path = ckpt_dir / "rollouts.jsonl"
    rollout_logger = RolloutLogger(
        rollout_log_path,
        log_tokens=config.training.log_token_ids,
        decode_fn=trainer.backend.decode if trainer.backend else None,
    )

    def _rollout_log_hook(t, m):
        rollouts = m.extras.get("rollouts", [])
        advantages = m.extras.get("advantages", [])
        if not rollouts:
            return
        prompts = m.extras.get("prompts")
        answers = m.extras.get("answers")
        targets = m.extras.get("targets")
        # Extract gradient multipliers for JSONL
        grad_mults = m.extras.get("gradient_multipliers", [])
        grad_mult_dict = None
        grad_mult_max = None
        grad_mult_mean = None
        if grad_mults:
            layer_names = [f"L{20 + i}" for i in range(16)]
            grad_mult_dict = dict(zip(layer_names, grad_mults, strict=False))
            grad_mult_max = max(grad_mults)
            grad_mult_mean = sum(grad_mults) / len(grad_mults) if grad_mults else None

        ctx = StepContext(
            iteration=m.iteration,
            total_iterations=config.training.iterations,
            loss=m.loss,
            kl_divergence=m.kl_divergence,
            grad_norm=m.grad_norm,
            tokens_per_sec=m.tokens_per_sec,
            generation_sec=m.generation_sec,
            elapsed_sec=m.elapsed_sec,
            mean_reward=m.mean_reward,
            reward_std=m.reward_std,
            dataset_row=m.extras.get("dataset_row", -1),
            repeat_idx=m.extras.get("repeat_idx", 0),
            naked_phase=m.extras.get("naked_phase", ""),
            weight_angle_avg=(
                sum(m.extras["weight_angles"].values()) / len(m.extras["weight_angles"])
                if m.extras.get("weight_angles")
                else None
            ),
            gradient_multipliers=grad_mult_dict,
            gradient_multiplier_max=grad_mult_max,
            gradient_multiplier_mean=grad_mult_mean,
            accumulation_step=m.extras.get("accum_count", 0),
            accumulation_steps_total=m.extras.get("accum_steps", 0),
            accumulation_over_repeats=m.extras.get("accum_over_repeats", 0),
        )
        rollout_logger.log(
            m.iteration,
            rollouts,
            m.rewards,
            advantages,
            prompts=prompts,
            answers=answers,
            targets=targets,
            step_ctx=ctx,
            group_size=config.training.group_size,
        )

    trainer.on("after_step", _rollout_log_hook)

    # ── CSV stats logger (per-iteration metrics + memory) ──
    from .training.stats_logger import StatsLogger

    stats_csv_path = ckpt_dir / "stats.csv"
    stats_logger = StatsLogger(
        stats_csv_path,
        seed=seed,
        epochs=config.training.epochs,
        samples_per_step=config.training.samples_per_step,
        dataset_size=len(dataset),
        batch_size=config.training.batch_size,
    )
    trainer.on("after_step", stats_logger.step_hook)

    # ── CKA drift probe (optional — auto-wired when cka_probe_base is set) ──
    _probe_base = getattr(config.training, "cka_probe_base", None)
    if _probe_base and model is not None:
        from pathlib import Path as _ProbePath  # noqa: PLC0415

        if _ProbePath(_probe_base).is_file():
            from .inspect import CKAProbe, build_inspect_config  # noqa: PLC0415

            _probe_cfg = build_inspect_config(
                {
                    "prompt": getattr(config.training, "cka_probe_prompt", ""),
                    "probe_wandb_key": "inspect/cka_drift",
                }
            )
            _cka_probe = CKAProbe(model, tokenizer, _probe_cfg, _probe_base, wandb_logger)
            trainer.on("after_step", _cka_probe.step_hook)
            trainer.on("on_eval", _cka_probe.on_eval_hook)
            print(f"[cka-probe] enabled | base={_probe_base}")
        else:
            print(f"[cka-probe] WARNING: probe_base not found: {_probe_base}")

    # ── Merge adapters into base ──
    # CLI --merge-adapters takes precedence; YAML merge_adapters is the fallback.
    _merge_paths = list(args.merge_adapters or [])
    if not _merge_paths:
        _merge_paths = [Path(p) for p in (getattr(config.training, "merge_adapters", None) or [])]
    if _merge_paths and model is not None:
        from mlx_grpo.training.adapter_migration import merge_adapter_into_base

        print(f"   Merging {len(_merge_paths)} adapter(s) into base weights (in order):")
        for _mp in _merge_paths:
            _n = merge_adapter_into_base(model, _mp)
            print(f"     {_mp}: fused {_n} layers")

    # ── Resume from checkpoint ──
    training_state = ckpt_mgr.load_training_state()
    resume_from = int(training_state.get("iteration", 0)) if training_state else 0
    if resume_from > 0 and model is not None and not args.resume and backend_name == "mlx":
        weights = ckpt_mgr.load_weights()
        if not weights:
            print(f"   ⚠️  training_state says iter {resume_from} but no adapter weights found")
            print("   Starting fresh (iteration counter reset to 0)")
            resume_from = 0
        else:
            from mlx.utils import tree_flatten as _tf_resume

            _cur = {k: v for k, v in _tf_resume(model.trainable_parameters())}
            _compat = {k: v for k, v in weights.items() if k in _cur and v.shape == _cur[k].shape}
            _shape_skip = sum(1 for k in weights if k in _cur and weights[k].shape != _cur[k].shape)
            _new_params = sum(1 for k in _cur if k not in weights)
            _dropped_keys = {k for k in weights if k not in _cur}
            _config_changed = _shape_skip > 0 or len(_dropped_keys) > 0

            if _config_changed and _old_adapter_cfg:
                # Adapter config changed (rank, type, etc.): fuse the old adapter
                # delta into the frozen base weights in-memory so nothing learned
                # is discarded. New adapter (already applied) starts from zero.
                from mlx_grpo.training.adapter_migration import fuse_old_adapter

                n_fused = fuse_old_adapter(model, weights, _old_adapter_cfg, _dropped_keys)
                _msg = f"   Resuming iter {resume_from}: fused {n_fused} layers into base"
                if _shape_skip:
                    _msg += f" ({_shape_skip} rank-changed)"
                if _dropped_keys:
                    _msg += f" ({len(_dropped_keys)} absorbed, e.g. DoRA magnitude)"
                _msg += " — new adapter starts from zero delta"
                print(_msg)
                print("   Optimizer state reset (adapter config changed)")
            else:
                # Same config (or no old config to migrate from): load matching
                # weights directly. LoRA→DoRA falls here: lora_a/lora_b load,
                # magnitude initialises fresh from base weight column norms.
                if _compat:
                    model.load_weights(list(_compat.items()), strict=False)
                _msg = (
                    f"   Resuming iter {resume_from}: "
                    f"{len(_compat)}/{len(_cur)} adapter params loaded"
                )
                if _new_params:
                    _msg += f", {_new_params} new (initialized fresh)"
                print(_msg)

                if optimizer is not None:
                    if _shape_skip == 0 and _new_params == 0:
                        # Skip optimizer state if LR changed — step counter inside
                        # the saved state would place the schedule mid-decay, negating
                        # the LR bump. Resetting gives full init LR from step 0.
                        _saved_lr = (training_state or {}).get("learning_rate")
                        _cfg_lr = config.training.learning_rate
                        _lr_changed = (
                            _saved_lr is not None and abs(float(_saved_lr) - float(_cfg_lr)) > 1e-12
                        )
                        if _lr_changed:
                            print(
                                f"   LR changed ({_saved_lr} → {_cfg_lr})"
                                " — optimizer state reset, schedule restarts from 0"
                            )
                        else:
                            _opt_loaded = ckpt_mgr.load_optimizer_state(optimizer)
                            if _opt_loaded:
                                print("   Loaded optimizer state for resume")
                            else:
                                print("   No optimizer state found — starting fresh optimizer")
                    else:
                        print("   Adapter config changed — optimizer state reset")

        # Materialise all weights after any load/fuse operation (MLX only)
        if backend_name == "mlx":
            try:
                import mlx.core as _mx
                from mlx.utils import tree_flatten as _tf2

                _arrays = [v for _, v in _tf2(model.parameters()) if hasattr(v, "shape")]
                if optimizer is not None and hasattr(optimizer, "state"):
                    _arrays.extend(v for _, v in _tf2(optimizer.state) if hasattr(v, "shape"))
                for _ci in range(0, len(_arrays), 50):
                    _mx.eval(*_arrays[_ci : _ci + 50])
                _mx.synchronize()
            except (ImportError, AttributeError):
                pass

    # ── Crash recovery ──
    recovery = RecoveryManager(str(ckpt_dir))
    recovery_state = recovery.load_or_create()

    # ── Train ──
    print(f"\n🚀 Training: {iterations} iterations, group_size={config.training.group_size}")
    print(f"   Resume from: {resume_from}\n")

    # ── TUI mode: wrap training in Textual app ──
    if getattr(args, "tui", False):
        return _run_with_tui(
            trainer,
            resume_from,
            iterations,
            monitor,
            rollout_logger,
            stats_logger,
            wandb_logger,
            ckpt_dir,
            recovery,
            recovery_state,
            backend,
        )

    interrupted = False
    try:
        trainer.train(resume_from=resume_from)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  Training interrupted — progress saved")
        recovery_state.iteration = trainer.iteration
        recovery.save_state(recovery_state)
    except Exception as e:
        print(f"\n  Training error: {e}")
        recovery.record_crash(recovery_state, str(e))
        raise
    finally:
        # Always close loggers and save wandb run ID, even on interrupt
        monitor.close()
        rollout_logger.close()
        stats_logger.close()
        if wandb_logger:
            wandb_logger.save_run_id(ckpt_dir / "wandb_run.json")
            wandb_logger.finish()

    # ── Summary ──
    if not interrupted:
        summary = monitor.summary()
        print("\n  Training complete")
        if "reward" in summary:
            print(f"   Final reward: {summary['reward']['latest']:.3f}")
        if "loss" in summary:
            print(f"   Final loss:   {summary['loss']['latest']:.4f}")

    return 0


def _run_with_tui(
    trainer,
    resume_from,
    total_iters,
    monitor,
    rollout_logger,
    stats_logger,
    wandb_logger,
    ckpt_dir,
    recovery,
    recovery_state,
    backend=None,
) -> int:
    """Launch training inside a Textual TUI dashboard."""
    try:
        from .tui import TrainingApp
    except ImportError as e:
        print(f"TUI dependencies not installed: {e}")
        print("Install with: pip install -e '.[tui]'")
        return 1

    # Suppress plain-text step reporting when TUI is active
    from dataclasses import replace as _replace

    trainer.config = _replace(trainer.config, steps_per_report=0)

    app = TrainingApp(None, total_iterations=total_iters)
    bridge = app.bridge

    # Redirect monitor logs to TUI log panel — prevents print() from
    # corrupting the terminal while Textual owns it in raw mode.
    monitor._log_fn = bridge.post_log

    # Training function runs in a worker thread — checks stop_event
    def train_fn() -> None:
        try:
            trainer.train(resume_from=resume_from)
        except KeyboardInterrupt:
            pass
        finally:
            recovery_state.iteration = trainer.iteration
            recovery.save_state(recovery_state)

    app._train_fn = train_fn

    # Wire bridge hooks (metrics, phase progress, streaming, checkpoint/eval log)
    trainer.on("after_step", bridge.step_hook)
    trainer.on("on_group", bridge.group_hook)
    trainer.on("on_gradient", bridge.gradient_hook)
    trainer.on("on_rollout_start", bridge.rollout_start_hook)
    trainer.on("after_rollout", bridge.pause_hook)
    trainer._tui_bridge = bridge  # escalation.py reads this for segment callbacks

    # Enable real-time token streaming if backend supports it
    if hasattr(backend, "_stream_cb"):
        backend._stream_cb = bridge.make_stream_cb()

    # Forward checkpoint/eval events to TUI log panel
    def _ckpt_log(t, m):
        bridge.post_log(f"[ckpt] Saved iter {m.iteration}")

    def _eval_log(t, m):
        bridge.post_log(f"[eval] reward={m.mean_reward:.3f}")

    trainer.on("on_checkpoint", _ckpt_log)
    trainer.on("on_eval", _eval_log)

    # Cooperative stop: bridge.stop_event → trainer.stop_requested
    def _stop_watcher(t, m):
        if bridge.stop_event.is_set():
            trainer.stop_requested = True

    trainer.on("after_step", _stop_watcher)
    trainer.on("on_group", _stop_watcher)

    try:
        app.run()
    except KeyboardInterrupt:
        trainer.stop_requested = True
        bridge.stop_event.set()
    finally:
        # Signal trainer to stop if still running
        trainer.stop_requested = True
        bridge.stop_event.set()
        # The Textual training worker runs in a daemon thread.  When app.run()
        # returns the daemon may not have reached its stop_requested check yet
        # and will be killed at process exit without writing a checkpoint.
        # Explicitly fire the checkpoint hook here so the adapter weights and
        # optimizer state are always flushed on any exit path.
        if trainer._last_metrics is not None:
            try:
                trainer._save_interrupt_checkpoint()
            except Exception as e:
                print(f"  Warning: interrupt checkpoint save failed: {e}")
        # Restore monitor to console output before closing
        monitor._log_fn = None
        # Clean up loggers
        monitor.close()
        rollout_logger.close()
        stats_logger.close()
        if wandb_logger:
            wandb_logger.save_run_id(ckpt_dir / "wandb_run.json")
            wandb_logger.finish()
        # Disable streaming callback to prevent messages to dead app
        if hasattr(backend, "_stream_cb"):
            backend._stream_cb = None


def _cmd_eval(args) -> int:
    from .backends.base import BackendRegistry
    from .config.loader import load_config
    from .data import DatasetConfig, DatasetLoader
    from .engine.executor import RolloutExecutor
    from .training.evaluation import Evaluator

    config = load_config(args.config)
    _ensure_builtins()

    # Load eval dataset
    ds_configs = config.datasets or {}
    eval_ds = None
    for _name, ds_cfg in ds_configs.items():
        if isinstance(ds_cfg, dict) and ds_cfg.get("path"):
            if "type" in ds_cfg and "type_name" not in ds_cfg:
                ds_cfg["type_name"] = ds_cfg.pop("type")
            eval_ds = DatasetLoader(DatasetConfig(**ds_cfg)).load()
            break

    if eval_ds is None:
        print("❌ No dataset configured for evaluation.")
        return 1

    # Load backend
    backend = BackendRegistry.get(config.generation.backend, model_path=config.training.model_path)
    executor = RolloutExecutor(backend)
    evaluator = Evaluator(
        executor,
        eval_dataset=eval_ds,
        type_overrides=config.type_overrides,
        system_prompt=config.generation.system_prompt,
    )

    print(f"📊 Evaluating on {min(args.max_samples, len(eval_ds))} samples...")
    result = evaluator.evaluate(max_samples=args.max_samples)

    print(f"\n{'─' * 40}")
    print(f"Mean reward:   {result.mean_reward:.4f}")
    print(f"Total samples: {result.total_samples}")
    print(f"Total tokens:  {result.total_tokens}")
    for t, r in result.per_type_reward.items():
        print(f"  {t}: {r:.4f} ({result.per_type_count[t]} samples)")
    print(f"{'─' * 40}")
    return 0


def _cmd_list_types() -> int:
    _ensure_builtins()
    for name in _get_types():
        t = _get_type_class(name)
        doc = (t.__doc__ or "").split("\n")[0].strip()
        print(f"  {name:20s} {doc}")
    return 0


def _cmd_list_rewards() -> int:
    _ensure_builtins()
    for name in _get_rewards():
        print(f"  {name}")
    return 0


def _cmd_list_backends() -> int:
    from .backends.base import BackendRegistry

    for name in BackendRegistry.available():
        print(f"  {name}")
    return 0


def _cmd_validate_data(args) -> int:
    from .config.loader import load_config
    from .data import DatasetConfig, DatasetLoader
    from .data.validator import format_report, save_valid_jsonl, validate_dataset

    config = load_config(args.config)
    _ensure_builtins()

    ds_configs = config.datasets or {}
    if not ds_configs:
        print("No datasets configured.")
        return 1

    exit_code = 0
    for name, ds_cfg in ds_configs.items():
        if not isinstance(ds_cfg, dict) or not ds_cfg.get("path"):
            continue
        if "type" in ds_cfg and "type_name" not in ds_cfg:
            ds_cfg["type_name"] = ds_cfg.pop("type")
        loader = DatasetLoader(DatasetConfig(**ds_cfg)).load()
        report = validate_dataset(name, list(loader))
        print(format_report(report, verbose=args.verbose))
        if report.invalid_rows > 0:
            exit_code = 1
        if args.output_dir:
            counts = save_valid_jsonl(report, args.output_dir)
            for tname, cnt in sorted(counts.items()):
                print(f"  Saved {cnt} valid rows -> {args.output_dir / f'{tname}.jsonl'}")

    return exit_code


def _ensure_builtins() -> None:
    import mlx_grpo.rewards.accuracy  # noqa: F401
    import mlx_grpo.types.builtins  # noqa: F401 — triggers registration


def _get_types() -> list[str]:
    from .types.base import TaskType

    _ensure_builtins()
    return TaskType.list_types()


def _get_rewards() -> list[str]:
    from .rewards.base import BaseReward

    _ensure_builtins()
    return BaseReward.list_rewards()


def _get_type_class(name: str):
    from .types.base import TaskType

    return TaskType._registry.get(name)


def _cmd_inspect(args) -> int:
    from .inspect import build_inspect_config, run_inspect  # noqa: PLC0415

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    circuit_layers = tuple(int(x) for x in args.circuit_layers.split(",") if x.strip())
    cfg = build_inspect_config(
        {
            "model_path": args.model_path,
            "model_b_path": args.model_b,
            "prompt": args.prompt,
            "prompts_file": args.prompts_file,
            "control_prompt": args.control_prompt,
            "save_dir": args.save_dir,
            "capture_sublayers": args.capture_sublayers,
            "trace_generation": args.trace_generation,
            "max_new_tokens": args.max_new_tokens,
            "top_k_logit_lens": args.top_k,
            "layers_to_plot": layers,
            "circuit_layers": circuit_layers,
            "feature": args.feature,
            "checkpoints_dir": args.checkpoints_dir,
            "rollout_log": args.rollout_log,
            "lora_path": args.lora_path,
            "probe_base": args.probe_base,
        }
    )
    return run_inspect(cfg)


def _cmd_torch_convert(args) -> int:
    """Convert HuggingFace model to local directory with optional quantization."""
    from .backends.torch_convert import convert_model

    return convert_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        quantize_bits=args.quantize,
        merge_adapters=args.merge_adapters,
    )


def _cmd_torch_quantize(args) -> int:
    """Quantize a local PyTorch model."""
    from .backends.torch_convert import quantize_model

    return quantize_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        bits=args.bits,
    )


def _cmd_torch_inspect(args) -> int:
    """Inspect PyTorch model configuration."""
    from .backends.torch_convert import inspect_model

    return inspect_model(args.model_dir)


if __name__ == "__main__":
    sys.exit(main())

# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: pyyaml (core), mlx + mlx-lm (MLX backend), torch + transformers (PyTorch)
# Run: python -m mlx_grpo train --config config.yml [--dry-run] [--resume DIR]
#      python -m mlx_grpo eval --config config.yml [--max-samples 50]
#      python -m mlx_grpo list-types | list-rewards | list-backends
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
