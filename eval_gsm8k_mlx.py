#!/usr/bin/env python3
"""Multi-backend evaluator for GSM8K, MMLU, and other datasets.

Supports:
- Multiple backends: MLX (Apple Silicon), HuggingFace/PyTorch, vLLM
- YAML configuration (config_eval_example.yml)
- CLI argument override (--dataset gsm8k --num-samples 100 --seed 42)
- lm-eval compatible parameters (tasks, num_fewshot, batch_size, device, etc.)
- Rich TUI with progress, TPS tracking, streaming
- Graceful Ctrl+C with partial result saving

Usage:
    python eval_gsm8k_mlx.py --config config.yml --backend mlx --num-samples 50
    python eval_gsm8k_mlx.py --dataset mmlu --backend huggingface --seed 42 --verbose
    python eval_gsm8k_mlx.py --backend vllm --model-path "meta-llama/Llama-2-7b"
"""


import logging
import argparse
import json
import os
import random
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from rich.console import Console
from rich.columns import Columns
from rich.layout import Layout
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.live import Live

from eval_config import EvalConfig
from eval_backends import get_backend

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Global state for Ctrl+C handling
_global_state = {
    "results": [],
    "correct": 0,
    "total": 0,
    "current_idx": 0,
    "config": None,
    "interrupted": False,
}

console = Console()


def setup_logging(log_level: str):
    """Configure logging with rich."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger()


def save_intermediate_results(config: EvalConfig):
    """Save partial results on Ctrl+C."""
    if not _global_state["results"]:
        return

    total = _global_state["total"]
    correct = _global_state["correct"]
    current_idx = _global_state["current_idx"]
    accuracy = (correct / total * 100) if total > 0 else 0.0
    progress_pct = (current_idx / total * 100) if total > 0 else 0.0

    os.makedirs(config.output_dir, exist_ok=True)
    filename = Path(config.output_dir) / f"partial_{config.dataset}_{int(time.time())}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "config": {
                    "dataset": config.dataset,
                    "seed": config.seed,
                    "batch_size": config.batch_size,
                },
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "completed": len(_global_state["results"]),
                "progress_pct": progress_pct,
                "interrupted": True,
                "results": _global_state["results"],
            },
            f,
            indent=2,
        )

    console.print(f"\n[yellow]⚠ Partial results saved to {filename}[/yellow]")
    console.print(f"[yellow]Progress: {current_idx}/{total} ({progress_pct:.1f}%)[/yellow]")
    console.print(f"[yellow]Accuracy: {correct}/{current_idx} ({accuracy:.1f}%)[/yellow]")


def interrupt_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    _global_state["interrupted"] = True
    console.print("\n[red]⏸ Interrupt received...[/red]")
    if _global_state["config"]:
        save_intermediate_results(_global_state["config"])
    sys.exit(0)


def normalize_answer(s: str) -> str:
    """Normalize answer for flexible matching (GSM8K style).

    Removes punctuation, spaces, converts to lowercase.
    """
    import re
    # Remove spaces
    s = re.sub(r'\s+', '', s)
    # Remove commas in numbers (e.g., "1,000" -> "1000")
    s = re.sub(r',', '', s)
    # Convert to lowercase
    s = s.lower()
    return s


def strict_match(extracted: str, expected: str) -> bool:
    """Exact string match (strict evaluation)."""
    if not extracted:
        return False
    return str(extracted).strip() == str(expected).strip()


def flexible_match(extracted: str, expected: str) -> bool:
    """Flexible match with normalization (GSM8K style).

    - Normalizes both strings
    - Checks numeric equivalence
    - Handles fractions and decimals
    """
    if not extracted or not expected:
        return False

    extracted_norm = normalize_answer(str(extracted))
    expected_norm = normalize_answer(str(expected))

    # Direct match after normalization
    if extracted_norm == expected_norm:
        return True

    # Try numeric comparison
    try:
        # Handle fractions (1/2)
        if '/' in extracted_norm or '/' in expected_norm:
            ex_parts = extracted_norm.split('/')
            exp_parts = expected_norm.split('/')
            if len(ex_parts) == 2 and len(exp_parts) == 2:
                ex_val = float(ex_parts[0]) / float(ex_parts[1])
                exp_val = float(exp_parts[0]) / float(exp_parts[1])
                return abs(ex_val - exp_val) < 1e-6

        # Numeric values
        ex_val = float(extracted_norm)
        exp_val = float(expected_norm)
        return abs(ex_val - exp_val) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


def extract_answer(text: str, dataset: str = "gsm8k", verbose: bool = False) -> Optional[str]:
    """Extract answer from multiple formats (dataset-aware priority).

    For numeric datasets (gsm8k): prioritize numbers before multiple choice
    For choice datasets (mmlu): prioritize multiple choice letters

    Supports:
    - \\boxed{answer}
    - Answer: X, The answer is X, etc.
    - Numbers, fractions, decimals (numeric datasets)
    - Multiple choice: A, B, C, D (choice datasets)
    """
    if not text:
        return None

    # 1. Try \\boxed{} format (highest priority always)
    boxed_match = re.search(r"\\boxed\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if verbose:
            logging.getLogger().info(f"  [green]✓ Boxed: {answer}[/green]")
        return answer

    # 2. Try "Answer: X" or "The answer is X" patterns
    answer_patterns = [
        r"(?:the\s+)?(?:answer|result|solution|final\s+answer)\s*[:=]\s*([^\n.;,!?]+)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if answer and answer not in [".", ",", ")", " "]:
                numeric_part = re.search(r"^([A-D]|\d+\.?\d*(?:/\d+)?)", answer)
                if numeric_part:
                    result = numeric_part.group(1)
                    if verbose:
                        logging.getLogger().info(f"  [green]✓ Answer keyword: {result}[/green]")
                    return result

    # 3. Dataset-aware extraction: numeric vs choice
    if dataset in ["gsm8k", "custom"]:  # Numeric datasets
        # 3a. Try numbers from text (before choice letters which are noise)
        # Strategy: prefer numbers appearing after "=" or before end of first paragraph
        number_matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if number_matches:
            # Look for pattern like "X = Y" where Y is the answer
            equals_match = re.search(r"=\s*(\d+(?:\.\d+)?)\b", text)
            if equals_match:
                answer = equals_match.group(1)
                if verbose:
                    logging.getLogger().info(f"  [green]✓ Number (from =): {answer}[/green]")
                return answer

            # Split by double-newline to handle rambling after the answer
            blocks = re.split(r"\n\s*\n", text)
            if len(blocks) > 1:
                # Take numbers from before rambling starts
                all_earlier = " ".join(blocks[:-1])
                earlier_nums = re.findall(r"\b\d+(?:\.\d+)?\b", all_earlier)
                if earlier_nums:
                    answer = earlier_nums[-1]
                else:
                    answer = number_matches[-1]
            else:
                # Single block: prefer last number (most likely the answer)
                answer = number_matches[-1]

            if verbose:
                logging.getLogger().info(f"  [green]✓ Number: {answer}[/green]")
            return answer

        # 3b. Try fractions (only after exhausting numbers)
        fraction_matches = re.findall(r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", text)
        if fraction_matches:
            answer = re.sub(r"\s+", "", fraction_matches[-1])
            if verbose:
                logging.getLogger().info(f"  [green]✓ Fraction: {answer}[/green]")
            return answer

        # 3c. Last resort for numeric: multiple choice letter
        choice_matches = list(re.finditer(r"\b([A-D])\b", text))
        if choice_matches:
            answer = choice_matches[-1].group(1).strip()
            if verbose:
                logging.getLogger().info(f"  [yellow]⚠ Choice fallback: {answer}[/yellow]")
            return answer

    elif dataset == "mmlu":  # Multiple choice dataset
        # 4a. Multiple choice letter (primary for mmlu)
        choice_matches = list(re.finditer(r"\b([A-D])\b", text))
        if choice_matches:
            answer = choice_matches[-1].group(1).strip()
            if verbose:
                logging.getLogger().info(f"  [green]✓ Multiple choice: {answer}[/green]")
            return answer

        # 4b. Fallback to numbers if no choice found
        number_matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if number_matches:
            answer = number_matches[-1]
            if verbose:
                logging.getLogger().info(f"  [yellow]⚠ Number fallback: {answer}[/yellow]")
            return answer

    # 5. Last resort: extract from last line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        cleaned = re.sub(r"[.,;:!?\)\]}\"]$", "", last_line).strip()
        if cleaned:
            first_token = re.search(r"^(\S+)", cleaned)
            if first_token:
                answer = first_token.group(1)
                if len(answer) < 50:
                    if verbose:
                        logging.getLogger().info(f"  [yellow]✓ Fallback: {answer}[/yellow]")
                    return answer

    if verbose:
        logging.getLogger().info(f"  [yellow]⚠ No pattern matched[/yellow]")
    return None


def build_sample(sample: dict, dataset: str) -> tuple[str, str]:
    """Build prompt and expected answer for dataset."""
    if dataset == "gsm8k":
        question = sample["question"]
        prompt = f"Question: {question}\nAnswer:"
        expected = sample["answer"].split("#### ")[-1].strip()
    elif dataset == "mmlu":
        question = sample["question"]
        choices = sample["choices"]
        choice_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Question: {question}\n{choice_text}\nAnswer:"
        expected = chr(65 + sample["answer"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return prompt, expected


def build_custom_sample(sample: dict, system_prompt_field: str = "system_prompt",
                       user_prompt_field: str = "user_prompt",
                       answer_field: str = "answer") -> tuple[str, str]:
    """Build prompt from custom JSONL sample with system/user prompts."""
    # Get system prompt if available
    system_text = sample.get(system_prompt_field, "").strip()
    user_text = sample.get(user_prompt_field, "").strip()

    # Build full prompt
    if system_text and user_text:
        prompt = f"{system_text}\n\n{user_text}"
    elif user_text:
        prompt = user_text
    elif system_text:
        prompt = system_text
    else:
        raise ValueError(f"Sample missing both {system_prompt_field} and {user_prompt_field}")

    # Get expected answer
    expected = str(sample.get(answer_field, "")).strip()
    if not expected:
        raise ValueError(f"Sample missing {answer_field}")

    return prompt, expected


def load_jsonl_dataset(path: str, limit: int = None) -> list[dict]:
    """Load JSONL file or directory with test.jsonl."""
    from pathlib import Path
    import json

    file_path = Path(path)

    # If directory, look for test.jsonl
    if file_path.is_dir():
        test_file = file_path / "test.jsonl"
        if not test_file.exists():
            # Try other common names
            for name in ["test.jsonl", "eval.jsonl", "validation.jsonl", "data.jsonl"]:
                alt = file_path / name
                if alt.exists():
                    test_file = alt
                    break
        if not test_file.exists():
            raise FileNotFoundError(f"No test.jsonl found in {path}")
        file_path = test_file

    # Load JSONL
    samples = []
    try:
        with open(file_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                if line.strip():
                    samples.append(json.loads(line))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSONL at line {i}: {e}")

    if not samples:
        raise ValueError(f"No samples found in {file_path}")

    return samples


def auto_detect_fields(sample: dict) -> tuple[str, str, str]:
    """Auto-detect system_prompt, user_prompt, and answer fields."""
    # Common field name patterns
    system_patterns = ["system_prompt", "system", "system_instruction", "instructions"]
    user_patterns = ["user_prompt", "user", "prompt", "question", "input", "text", "query"]
    answer_patterns = ["answer", "output", "target", "label", "expected", "ground_truth"]

    system_field = next((k for k in sample.keys() if k.lower() in system_patterns), None)
    user_field = next((k for k in sample.keys() if k.lower() in user_patterns), None)
    answer_field = next((k for k in sample.keys() if k.lower() in answer_patterns), None)

    if not user_field:
        raise ValueError(f"Could not auto-detect user_prompt field. Available: {list(sample.keys())}")
    if not answer_field:
        raise ValueError(f"Could not auto-detect answer field. Available: {list(sample.keys())}")

    return system_field or "", user_field, answer_field


def generate_detailed_analytics(results: list[dict]) -> dict:
    """Generate detailed analytics for failures and successes."""
    if not results:
        return {}

    correct_samples = [r for r in results if r["correct"]]
    failure_samples = [r for r in results if not r["correct"]]

    # Failure analysis
    failure_stats = {
        "count": len(failure_samples),
        "percentage": len(failure_samples) / len(results) * 100,
        "examples": failure_samples[:5],  # First 5 failures
        "patterns": [],
    }

    # Identify failure patterns
    if failure_samples:
        # Pattern: No answer extracted
        no_extraction = [r for r in failure_samples if not r["extracted"]]
        if no_extraction:
            failure_stats["patterns"].append({
                "type": "no_extraction",
                "count": len(no_extraction),
                "percentage": len(no_extraction) / len(failure_samples) * 100,
                "description": "Model generated output but answer couldn't be extracted",
                "examples": [{"q": r["question"], "out": r["generated"][:100]} for r in no_extraction[:2]],
            })

        # Pattern: Wrong number/answer
        wrong_value = [r for r in failure_samples if r["extracted"]]
        if wrong_value:
            failure_stats["patterns"].append({
                "type": "wrong_value",
                "count": len(wrong_value),
                "percentage": len(wrong_value) / len(failure_samples) * 100,
                "description": "Extracted correct format but wrong value",
                "examples": [
                    {
                        "q": r["question"][:50],
                        "expected": str(r["expected"])[:20],
                        "extracted": str(r["extracted"])[:20],
                    }
                    for r in wrong_value[:3]
                ],
            })

    # Success analysis
    success_stats = {
        "count": len(correct_samples),
        "percentage": len(correct_samples) / len(results) * 100,
        "avg_tokens": sum(r["tokens"] for r in correct_samples) / len(correct_samples) if correct_samples else 0,
        "avg_tps": sum(r["tps"] for r in correct_samples) / len(correct_samples) if correct_samples else 0,
        "examples": correct_samples[:3],  # First 3 successes
    }

    # Overall stats
    all_tokens = [r["tokens"] for r in results]
    all_tps = [r["tps"] for r in results if r["tps"] > 0]

    return {
        "failures": failure_stats,
        "successes": success_stats,
        "overall": {
            "total": len(results),
            "correct": len(correct_samples),
            "accuracy": len(correct_samples) / len(results) * 100,
            "token_stats": {
                "mean": sum(all_tokens) / len(all_tokens) if all_tokens else 0,
                "min": min(all_tokens) if all_tokens else 0,
                "max": max(all_tokens) if all_tokens else 0,
            },
            "tps_stats": {
                "mean": sum(all_tps) / len(all_tps) if all_tps else 0,
                "min": min(all_tps) if all_tps else 0,
                "max": max(all_tps) if all_tps else 0,
            },
        },
    }


def log_to_wandb(run_name: str, config: EvalConfig, metrics: dict, results: list[dict],
                analytics: dict, output_path: Path):
    """Log evaluation to Weights & Biases."""
    if not WANDB_AVAILABLE:
        return

    try:
        wandb.init(
            project="mlx-evaluation",
            name=run_name,
            config={
                "dataset": config.dataset,
                "model": config.model.path,
                "batch_size": config.batch_size,
                "max_tokens": config.generation.max_tokens,
                "seed": config.seed,
            },
        )

        # Log main metrics
        wandb.log({
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "avg_tps": metrics["avg_tps"],
            "total_time": metrics["total_time_seconds"],
        })

        # Log analytics
        if analytics:
            wandb.log({
                "failure_count": analytics["failures"]["count"],
                "failure_percentage": analytics["failures"]["percentage"],
                "success_count": analytics["successes"]["count"],
                "success_percentage": analytics["successes"]["percentage"],
                "avg_tokens_success": analytics["successes"]["avg_tokens"],
                "avg_tps_success": analytics["successes"]["avg_tps"],
            })

            # Log failure patterns
            for pattern in analytics["failures"]["patterns"]:
                wandb.log({
                    f"failure_pattern_{pattern['type']}": pattern["count"]
                })

        # Log failure examples as table
        if analytics["failures"]["examples"]:
            failure_table = wandb.Table(columns=["Index", "Question", "Expected", "Extracted", "Generated"])
            for ex in analytics["failures"]["examples"][:10]:
                failure_table.add_data(
                    ex["index"],
                    ex["question"][:50],
                    ex["expected"][:50],
                    ex["extracted"] or "N/A",
                    ex["generated"][:100],
                )
            wandb.log({"failure_samples": failure_table})

        # Log success examples as table
        if analytics["successes"]["examples"]:
            success_table = wandb.Table(columns=["Index", "Question", "Expected", "Extracted", "Tokens"])
            for ex in analytics["successes"]["examples"][:10]:
                success_table.add_data(
                    ex["index"],
                    ex["question"][:50],
                    ex["expected"][:50],
                    ex["extracted"] or "N/A",
                    ex["tokens"],
                )
            wandb.log({"success_samples": success_table})

        # Log artifacts
        if output_path.exists():
            wandb.save(str(output_path))

        wandb.finish()
        print(f"✓ Logged to WandB: https://wandb.ai")

    except Exception as e:
        print(f"⚠ WandB logging failed: {e}")


def parse_models_from_args(models_list: list) -> dict:
    """Parse model definitions from --model arguments.

    Args:
        models_list: ["path=model1", "path=model2,title=M2", ...]

    Returns:
        {model_path: {"path": ..., "title": ...}}
    """
    if not models_list:
        return {}

    models = {}
    for model_def in models_list:
        # Parse "path=x,title=y" format
        parts = {}
        for pair in model_def.split(","):
            if "=" in pair:
                key, val = pair.split("=", 1)
                parts[key.strip()] = val.strip()
            else:
                raise ValueError(f"Invalid model format: {model_def}. Use: path=<path> or path=<path>,title=<title>")

        if "path" not in parts:
            raise ValueError(f"Model must have 'path' key: {model_def}")

        path = parts["path"]
        title = parts.get("title", path)  # Default title to path if not specified
        models[path] = {"path": path, "title": title}

    return models


def compare_model_results(results_dict: dict, model_labels: dict = None) -> dict:
    """Compare evaluation results across multiple models.

    Args:
        results_dict: {model_name: {metrics_dict, results_list}}

    Returns:
        Comparison analytics
    """
    if not results_dict:
        return {}

    comparison = {
        "models": {},
        "comparison": {
            "accuracy_ranking": [],
            "speed_ranking": [],
            "efficiency_ranking": [],  # accuracy per token
        }
    }

    # Collect per-model metrics
    for model_name, data in results_dict.items():
        metrics = data["metrics"]
        comparison["models"][model_name] = {
            "accuracy": metrics["accuracy"],
            "total_time": metrics["total_time_seconds"],
            "avg_tps": metrics["avg_tps"],
            "total_tokens": metrics["total_tokens_generated"],
            "correct": metrics["correct"],
            "total": metrics["total"],
        }
        # Efficiency: accuracy % per 1000 tokens
        efficiency = (metrics["accuracy"] * metrics["total"]) / max(metrics["total_tokens_generated"], 1)
        comparison["models"][model_name]["efficiency_score"] = efficiency

    # Generate rankings
    models_by_accuracy = sorted(
        comparison["models"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )
    models_by_speed = sorted(
        comparison["models"].items(),
        key=lambda x: x[1]["avg_tps"],
        reverse=True
    )
    models_by_efficiency = sorted(
        comparison["models"].items(),
        key=lambda x: x[1]["efficiency_score"],
        reverse=True
    )

    comparison["comparison"]["accuracy_ranking"] = [
        {"rank": i+1, "model": m, "accuracy": d["accuracy"]}
        for i, (m, d) in enumerate(models_by_accuracy)
    ]
    comparison["comparison"]["speed_ranking"] = [
        {"rank": i+1, "model": m, "tps": d["avg_tps"]}
        for i, (m, d) in enumerate(models_by_speed)
    ]
    comparison["comparison"]["efficiency_ranking"] = [
        {"rank": i+1, "model": m, "score": d["efficiency_score"]}
        for i, (m, d) in enumerate(models_by_efficiency)
    ]

    # Summary stats
    if models_by_accuracy:
        best_model = models_by_accuracy[0][0]
        best_accuracy = models_by_accuracy[0][1]["accuracy"]
        worst_accuracy = models_by_accuracy[-1][1]["accuracy"]
        comparison["comparison"]["summary"] = {
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "worst_accuracy": worst_accuracy,
            "accuracy_spread": best_accuracy - worst_accuracy,
            "avg_accuracy": sum(d["accuracy"] for d in comparison["models"].values()) / len(comparison["models"]),
        }

    return comparison


def parse_args() -> tuple[argparse.Namespace, EvalConfig]:
    """Parse CLI arguments and load config."""
    parser = argparse.ArgumentParser(
        description="Evaluate MLX model on GSM8K, MMLU, or other datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_gsm8k_mlx.py --config config.yml
  python eval_gsm8k_mlx.py --dataset mmlu --num-samples 50 --seed 42
  python eval_gsm8k_mlx.py --dataset gsm8k --verbose --batch-size 1
        """,
    )

    # Config file
    parser.add_argument(
        "--config", "-c", type=str, help="YAML config file (overridden by CLI args)"
    )

    # lm-eval compatible args
    parser.add_argument(
        "--dataset", "-d", type=str,
        help="Dataset: gsm8k, mmlu, custom, or comma-separated"
    )
    parser.add_argument(
        "--dataset-path", "--dataset_path", type=str,
        help="Path to custom JSONL dataset (file or directory with test.jsonl)"
    )
    parser.add_argument(
        "--system-prompt-field", "--system_prompt_field", type=str,
        default="system_prompt", help="JSONL field name for system prompt"
    )
    parser.add_argument(
        "--user-prompt-field", "--user_prompt_field", type=str,
        default="user_prompt", help="JSONL field name for user prompt"
    )
    parser.add_argument(
        "--answer-field", "--answer_field", type=str,
        default="answer", help="JSONL field name for ground truth answer"
    )
    parser.add_argument("--tasks", type=str, help="Tasks (comma-separated, lm-eval style)")
    parser.add_argument("--num-fewshot", "--num_fewshot", type=int, help="Few-shot examples")
    parser.add_argument("--num-samples", "-n", type=int, help="Total samples to evaluate")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    parser.add_argument("--seed", "-s", type=int, help="Random seed")
    parser.add_argument("--split", type=str, help="Data split (train/val/test)")

    # Model args
    parser.add_argument("--model-path", "--model_path", type=str,
                       help="[Deprecated] Use --model instead. Model path or HF ID")
    parser.add_argument("--model", action="append", dest="models",
                       help="Model definition: path=<path> or path=<path>,title=<title>. Use multiple times for comparison")
    parser.add_argument("--adapter-path", "--adapter_path", type=str, help="LoRA adapter path")
    parser.add_argument(
        "--backend", type=str, choices=["mlx", "huggingface", "hf", "torch", "vllm"],
        default="mlx",
        help="Generation backend: mlx (default), huggingface, vllm"
    )

    # Generation args - presets
    parser.add_argument(
        "--gen-preset", "--gen_preset", type=str,
        choices=["deterministic", "balanced", "creative", "fast"],
        help="Preset generation config (overrides individual params)"
    )

    # Generation args - individual control
    parser.add_argument("--max-tokens", "--max_tokens", type=int,default=2048, help="Max generation tokens (default: 2048)")
    parser.add_argument(
        "--temperature", type=float,
        help="Sampling temperature: 0=deterministic, 1=balanced, >1=random (default: 0.0)"
    )
    parser.add_argument(
        "--top-p", "--top_p", default=1.0, type=float,
        help="Nucleus sampling: 1.0=all, 0.9=top 90%% (default: 1.0)"
    )
    parser.add_argument(
        "--top-k", "--top_k", type=int,
        help="Top-k sampling: 0=disabled, 50=top 50 tokens (default: 0)"
    )
    parser.add_argument(
        "--num-beams", "--num_beams", type=int,
        help="Beam search width: 1=greedy, >1=beam (default: 1)"
    )
    parser.add_argument(
        "--repetition-penalty", "--repetition_penalty", type=float,
        help="Penalize repeated tokens: 1.0=none, >1=penalty (default: 1.0)"
    )
    parser.add_argument(
        "--length-penalty", "--length_penalty", type=float,
        help="Length penalty: >1=longer, <1=shorter (default: 1.0)"
    )

    # Output/display
    parser.add_argument("--output-dir", "--output_dir", type=str, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, help="Device: cpu, cuda, mps")
    parser.add_argument("--log-level", "--log_level", type=str, help="Log level")
    parser.add_argument("--no-cache", "--no_cache", action="store_true", help="Disable cache")
    parser.add_argument(
        "--show-stream", "--show_stream", action="store_true", help="Show streaming output"
    )

    # Analytics & logging
    parser.add_argument(
        "--wandb", action="store_true", help="Log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project", "--wandb_project", type=str, default="mlx-evaluation",
        help="WandB project name"
    )
    parser.add_argument(
        "--detailed-analytics", "--detailed_analytics", action="store_true",
        help="Generate detailed failure/success analysis"
    )

    # Parallel evaluation
    parser.add_argument(
        "--parallel", action="store_true",
        help="Evaluate multiple models in parallel (side-by-side) instead of sequentially"
    )

    args = parser.parse_args()

    # Load base config from YAML or use defaults
    if args.config:
        config = EvalConfig.from_yaml(args.config)
        console.print(f"[cyan]Loaded config from {args.config}[/cyan]")
    else:
        config = EvalConfig()

    # Override with CLI args
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.tasks:
        overrides["tasks"] = [t.strip() for t in args.tasks.split(",")]
    if args.num_fewshot is not None:
        overrides["num_fewshot"] = args.num_fewshot
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.split:
        overrides["split"] = args.split

    # Store custom dataset fields for later use
    custom_fields = {
        "dataset_path": args.dataset_path,
        "system_prompt_field": args.system_prompt_field,
        "user_prompt_field": args.user_prompt_field,
        "answer_field": args.answer_field,
    }
    if args.model_path:
        # Update nested model config
        model_cfg = config.model
        overrides["model"] = model_cfg.from_dict({"path": args.model_path})

    # Handle generation config
    gen_cfg = config.generation
    gen_overrides = {}

    logging.getLogger().info(f"Args: {args}")

    # Apply preset if specified
    if args.gen_preset:
        from eval_config import GenerationConfig
        preset_config = GenerationConfig.get_preset(args.gen_preset)
        gen_cfg = preset_config
        logging.getLogger().info(f"Using generation preset: {args.gen_preset}")
    else:
        # Build from individual arguments
        if args.max_tokens is not None:
            gen_overrides["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            gen_overrides["temperature"] = args.temperature
        if args.top_p is not None:
            gen_overrides["top_p"] = args.top_p
        if args.top_k is not None:
            gen_overrides["top_k"] = args.top_k
        if args.num_beams is not None:
            gen_overrides["num_beams"] = args.num_beams
        if args.repetition_penalty is not None:
            gen_overrides["repetition_penalty"] = args.repetition_penalty
        if args.length_penalty is not None:
            gen_overrides["length_penalty"] = args.length_penalty

        if gen_overrides:
            gen_cfg = gen_cfg.from_dict(gen_overrides)

    if gen_overrides or args.gen_preset:
        overrides["generation"] = gen_cfg
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.verbose:
        overrides["verbose"] = True
    if args.device:
        overrides["device"] = args.device
    if args.log_level:
        overrides["log_level"] = args.log_level
    if args.no_cache:
        overrides["no_cache"] = True
    if args.show_stream:
        overrides["show_stream"] = True

    if overrides:
        config = config.update_from_args(**overrides)

    return args, config


def display_side_by_side_results(results_dict: dict):
    """Display results from multiple models side-by-side in TUI.

    Args:
        results_dict: {model_title: {metrics, results}}
    """
    if not results_dict or len(results_dict) < 2:
        return

    console.print("\n" + "="*150)
    console.print("[bold cyan]SIDE-BY-SIDE RESULTS[/bold cyan]")
    console.print("="*150)

    # Create side-by-side tables
    model_titles = list(results_dict.keys())
    tables = []

    for title in model_titles:
        data = results_dict[title]
        results = data["results"][:10]  # Show first 10

        table = Table(title=f"[bold]{title}[/bold]", show_header=True, header_style="bold cyan")
        table.add_column("Q", style="dim", width=15)
        table.add_column("Result", width=8)
        table.add_column("Extracted", width=12)
        table.add_column("Expected", width=12)
        table.add_column("TPS", width=6)

        for r in results:
            status = "[green]✓[/green]" if r["correct"] else "[red]✗[/red]"
            table.add_row(
                r["question"][:13] + "...",
                status,
                str(r["extracted"])[:12] if r["extracted"] else "N/A",
                str(r["expected"])[:12],
                f"{r['tps']:.1f}",
            )
        tables.append(table)

    # Display side-by-side
    if len(tables) == 2:
        console.print(Columns(tables, equal=True, expand=True))
    else:
        for table in tables:
            console.print(table)

    console.print("="*150)


def _display_model_comparison(comparison: dict):
    """Display model comparison results."""
    if not comparison:
        return

    console.print("\n" + "="*70)
    console.print("[bold cyan]MODEL COMPARISON[/bold cyan]")
    console.print("="*70)

    # Accuracy ranking
    if comparison["comparison"]["accuracy_ranking"]:
        console.print("\n[bold]Accuracy Ranking:[/bold]")
        for item in comparison["comparison"]["accuracy_ranking"]:
            console.print(f"  {item['rank']}. {item['model']}: {item['accuracy']:.1f}%")

    # Speed ranking
    if comparison["comparison"]["speed_ranking"]:
        console.print("\n[bold]Speed Ranking (tokens/sec):[/bold]")
        for item in comparison["comparison"]["speed_ranking"]:
            console.print(f"  {item['rank']}. {item['model']}: {item['tps']:.1f} TPS")

    # Efficiency ranking
    if comparison["comparison"]["efficiency_ranking"]:
        console.print("\n[bold]Efficiency Ranking:[/bold]")
        for item in comparison["comparison"]["efficiency_ranking"]:
            console.print(f"  {item['rank']}. {item['model']}: {item['score']:.4f}")

    # Summary
    if "summary" in comparison["comparison"]:
        summary = comparison["comparison"]["summary"]
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Best model: {summary['best_model']} ({summary['best_accuracy']:.1f}%)")
        console.print(f"  Accuracy spread: {summary['accuracy_spread']:.1f}%")
        console.print(f"  Average accuracy: {summary['avg_accuracy']:.1f}%")

    console.print("="*70)


def main():
    """Main evaluation loop (supports single or multiple models)."""
    args, config = parse_args()
    _global_state["config"] = config

    # Setup logging
    logger = setup_logging(config.log_level)

    # Register interrupt handler
    signal.signal(signal.SIGINT, interrupt_handler)

    # Handle model arguments
    models_dict = {}

    # New format: --model path=...,title=...
    if hasattr(args, 'models') and args.models:
        try:
            models_dict = parse_models_from_args(args.models)
        except ValueError as e:
            logger.error(f"Invalid model format: {e}")
            sys.exit(1)

    # Fallback: old format --model-path (deprecated but supported)
    elif args.model_path:
        path = args.model_path
        models_dict = {path: {"path": path, "title": path}}

    # Last resort: from config
    elif config.model.path:
        path = config.model.path
        models_dict = {path: {"path": path, "title": path}}

    if not models_dict:
        logger.error("No model specified. Use: --model path=<model> [--model path=<model>,title=<label>]")
        sys.exit(1)

    # Multi-model evaluation
    if len(models_dict) > 1:
        use_parallel = args.parallel if hasattr(args, 'parallel') else False
        backend_name = args.backend if hasattr(args, 'backend') else 'mlx'

        # MLX backend doesn't support parallel (Metal GPU is not thread-safe)
        if use_parallel and backend_name == 'mlx':
            console.print("[yellow]⚠ MLX backend doesn't support parallel (Metal GPU limitation)[/yellow]")
            console.print("[yellow]→ Using sequential mode instead[/yellow]")
            console.print("[yellow]→ For parallel: use --backend huggingface or --backend vllm[/yellow]")
            use_parallel = False

        eval_mode = "[cyan]Parallel[/cyan]" if use_parallel else "[cyan]Sequential[/cyan]"
        console.print(f"{eval_mode} evaluation of {len(models_dict)} models")
        results_dict = {}

        if use_parallel:
            # Parallel evaluation using threads
            with ThreadPoolExecutor(max_workers=len(models_dict)) as executor:
                # Submit all tasks
                futures = {}
                for model_path, model_info in models_dict.items():
                    title = model_info["title"]
                    config_copy = config.update_from_args(model=config.model.from_dict({"path": model_path}))
                    future = executor.submit(_evaluate_single_model, args, config_copy, logger, model_title=title)
                    futures[future] = title

                # Collect results as they complete
                for future in as_completed(futures):
                    title = futures[future]
                    try:
                        result = future.result()
                        results_dict[title] = result
                    except Exception as e:
                        logger.error(f"Model {title} evaluation failed: {e}")
        else:
            # Sequential evaluation (original behavior)
            for model_path, model_info in models_dict.items():
                title = model_info["title"]
                console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
                console.print(f"[bold cyan]Model: {title}[/bold cyan]")
                console.print(f"[bold cyan]{'='*70}[/bold cyan]")
                config_copy = config.update_from_args(model=config.model.from_dict({"path": model_path}))
                result = _evaluate_single_model(args, config_copy, logger, model_title=title)
                results_dict[title] = result

        # Display side-by-side results (if parallel)
        if use_parallel and len(results_dict) > 1:
            display_side_by_side_results(results_dict)

        # Generate comparison
        comparison = compare_model_results(results_dict)
        _display_model_comparison(comparison)

        # Save comparison
        if config.write_out:
            os.makedirs(config.output_dir, exist_ok=True)
            comparison_file = Path(config.output_dir) / f"comparison_{int(time.time())}.json"
            with open(comparison_file, "w") as f:
                json.dump(comparison, f, indent=2)
            console.print(f"\n[green]✓ Comparison saved to {comparison_file}[/green]")
        return

    # Single model evaluation
    model_path = list(models_dict.keys())[0]
    config = config.update_from_args(model=config.model.from_dict({"path": model_path}))
    _evaluate_single_model(args, config, logger)


def _evaluate_single_model(args, config: EvalConfig, logger, model_title: str = None):
    """Evaluate a single model (extracted from main for reuse)."""
    # Use model path as title if not provided
    if not model_title:
        model_title = config.model.path

    # Set seed
    if config.seed is None:
        config = config.update_from_args(seed=random.randint(0, 2**31 - 1))
    random.seed(config.seed)
    logger.info(f"Seed: {config.seed}")

    # Get backend
    backend_name = args.backend if hasattr(args, 'backend') else 'mlx'
    console.print(f"[cyan]Loading backend: {backend_name}[/cyan]")
    try:
        backend = get_backend(backend_name, config.model.path, device=config.device)
    except Exception as e:
        logger.error(f"Failed to load backend: {e}")
        sys.exit(1)
    console.print(f"[green]✓ Backend loaded: {backend_name}[/green]")

    # Load dataset
    dataset_name = config.dataset.upper()
    console.print(f"[cyan]Loading {dataset_name} ({config.num_samples} samples)...[/cyan]")

    # Determine dataset type and load
    if config.dataset == "custom":
        if not args.dataset_path:
            logger.error("--dataset-path required for custom dataset")
            sys.exit(1)

        try:
            ds = load_jsonl_dataset(args.dataset_path, limit=config.num_samples)
            console.print(f"[cyan]Loaded {len(ds)} samples from {args.dataset_path}[/cyan]")

            # Auto-detect fields if needed
            if ds:
                first_sample = ds[0]
                sys_field, user_field, ans_field = auto_detect_fields(first_sample)
                if sys_field:
                    args.system_prompt_field = sys_field
                args.user_prompt_field = user_field
                args.answer_field = ans_field
                console.print(f"[dim]Auto-detected fields: system={sys_field or 'None'}, "
                            f"user={user_field}, answer={ans_field}[/dim]")
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            sys.exit(1)

        # Create a wrapper to match HF dataset interface
        class CustomDataset:
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]
            def select(self, indices):
                return CustomDataset([self.samples[i] for i in indices])

        ds = CustomDataset(ds)
    elif config.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=config.split)
        ds = ds.select(range(min(config.num_samples, len(ds))))
    elif config.dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split=config.split)
        ds = ds.select(range(min(config.num_samples, len(ds))))
    else:
        logger.error(f"Unknown dataset: {config.dataset}")
        sys.exit(1)

    _global_state["total"] = len(ds)

    results = []
    correct = 0
    total_tokens = 0
    total_time = 0.0

    os.makedirs(config.output_dir, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        model_label = f" [{model_title}]" if model_title else ""
        task = progress.add_task(f"[cyan]Evaluating {config.dataset.upper()}{model_label}...[/cyan]", total=len(ds))

        # Build prompts
        prompts = []
        sample_data = []
        for i, sample in enumerate(ds):
            try:
                if config.dataset == "custom":
                    prompt, expected = build_custom_sample(
                        sample,
                        system_prompt_field=args.system_prompt_field,
                        user_prompt_field=args.user_prompt_field,
                        answer_field=args.answer_field,
                    )
                else:
                    prompt, expected = build_sample(sample, config.dataset)
                prompts.append(prompt)
                sample_data.append((i + 1, sample, expected))
            except Exception as e:
                logger.warning(f"Skipping sample {i + 1}: {e}")
                continue

        # Generate in batches
        for batch_start in range(0, len(prompts), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_samples = sample_data[batch_start:batch_end]

            batch_start_time = time.time()
            try:
                logger.debug(f"Generating {len(batch_prompts)} samples (max_tokens={config.generation.max_tokens})")
                generated_batch = backend.batch_generate(
                    batch_prompts,
                    max_tokens=config.generation.max_tokens,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                    top_k=config.generation.top_k,
                )
                logger.debug(f"✓ Generated {len(generated_batch)} texts")

            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                generated_batch = [""] * len(batch_prompts)

            batch_elapsed = time.time() - batch_start_time

            # Process results
            for (idx, sample, expected), prompt, generated in zip(
                batch_samples, batch_prompts, generated_batch
            ):
                _global_state["current_idx"] = idx

                # Clean up generated text
                if isinstance(generated, str):
                    generated = (
                        generated[len(prompt) :].strip()
                        if generated.startswith(prompt)
                        else generated
                    )
                else:
                    generated = ""

                # Count tokens (use backend tokenizer if available)
                if hasattr(backend, 'tokenizer'):
                    gen_tokens = len(backend.tokenizer.encode(generated)) if generated else 0
                else:
                    # Rough estimate for backends without tokenizer access
                    gen_tokens = len(generated.split())
                total_tokens += gen_tokens

                # Extract answer (dataset-aware priority)
                extracted = extract_answer(generated, dataset=config.dataset, verbose=config.verbose)

                # Check correctness (both strict and flexible)
                strict_correct = strict_match(extracted, expected)
                flexible_correct = flexible_match(extracted, expected)

                # For progress, use flexible (more fair)
                is_correct = flexible_correct
                correct += is_correct
                _global_state["correct"] = correct

                # TPS
                sample_time = batch_elapsed / len(batch_samples)
                tps = gen_tokens / sample_time if sample_time > 0 else 0.0

                # Log result with model label
                status = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
                q = sample.get("question", "")[:40]
                model_badge = f"[cyan]【{model_title}】[/cyan]" if model_title else ""
                console.print(f"{model_badge} {status} {q}...")
                console.print(f"  Expected: {str(expected)[:50]}")
                console.print(f"  Extracted: {extracted or '[dim]N/A[/dim]'}")
                if config.show_tps:
                    console.print(f"  [dim]TPS: {tps:.1f} | Tokens: {gen_tokens}[/dim]")

                results.append(
                    {
                        "index": idx,
                        "question": q,
                        "expected": expected,
                        "extracted": extracted,
                        "generated": generated[:500],
                        "correct_strict": strict_correct,  # Exact match
                        "correct_flexible": flexible_correct,  # Flexible match
                        "correct": is_correct,  # Backward compat (flexible)
                        "tokens": gen_tokens,
                        "tps": tps,
                    }
                )
                _global_state["results"] = results
                progress.advance(task)

            total_time += batch_elapsed
            avg_tps = total_tokens / total_time if total_time > 0 else 0.0
            progress.update(
                task,
                description=f"[cyan]{config.dataset.upper()} | Avg TPS: {avg_tps:.1f}[/cyan]",
            )

    # Final summary (compute both metrics)
    strict_correct = sum(1 for r in results if r.get("correct_strict", False))
    flexible_correct = sum(1 for r in results if r.get("correct_flexible", False))

    strict_accuracy = strict_correct / len(ds) * 100 if len(ds) > 0 else 0.0
    flexible_accuracy = flexible_correct / len(ds) * 100 if len(ds) > 0 else 0.0

    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    avg_time_per_sample = total_time / len(results) if results else 0.0

    console.print("\n" + "=" * 70)
    summary = Table(title="Evaluation Summary", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Dataset", config.dataset.upper())
    summary.add_row("Samples", str(len(ds)))
    summary.add_row("", "")  # Separator
    summary.add_row("[bold]Accuracy (Strict)[/bold]", f"{strict_correct}/{len(ds)} = {strict_accuracy:.1f}%")
    summary.add_row("[bold]Accuracy (Flexible)[/bold]", f"{flexible_correct}/{len(ds)} = {flexible_accuracy:.1f}%")
    summary.add_row("", "")  # Separator
    summary.add_row("Seed", str(config.seed))
    summary.add_row("Batch Size", str(config.batch_size))
    summary.add_row("Total Time", f"{total_time:.2f}s")
    summary.add_row("Avg Time/Sample", f"{avg_time_per_sample:.2f}s")
    summary.add_row("Total Tokens", str(total_tokens))
    summary.add_row("Avg TPS", f"{avg_tps:.1f}")
    if config.dataset == "custom" and args.dataset_path:
        summary.add_row("Dataset Path", args.dataset_path)
    console.print(summary)
    console.print("=" * 70)

    # Generate detailed analytics
    analytics = {}
    if args.detailed_analytics or args.wandb:
        analytics = generate_detailed_analytics(results)

        if args.detailed_analytics:
            # Display failure analysis
            console.print("\n" + "=" * 70)
            console.print("[bold red]FAILURE ANALYSIS[/bold red]")
            console.print("=" * 70)

            if analytics["failures"]["count"] > 0:
                console.print(f"\n[red]{analytics['failures']['count']} failures "
                            f"({analytics['failures']['percentage']:.1f}%)[/red]")

                for pattern in analytics["failures"]["patterns"]:
                    console.print(f"\n[yellow]{pattern['type'].upper()}:[/yellow] "
                                f"{pattern['count']} ({pattern['percentage']:.1f}%)")
                    console.print(f"  {pattern['description']}")
                    if pattern["examples"]:
                        for ex in pattern["examples"][:2]:
                            console.print(f"    - {ex}")
            else:
                console.print("[green]✓ No failures![/green]")

            # Display success analysis
            console.print("\n" + "=" * 70)
            console.print("[bold green]SUCCESS ANALYSIS[/bold green]")
            console.print("=" * 70)

            if analytics["successes"]["count"] > 0:
                console.print(f"\n[green]{analytics['successes']['count']} successes "
                            f"({analytics['successes']['percentage']:.1f}%)[/green]")
                console.print(f"  Avg tokens: {analytics['successes']['avg_tokens']:.1f}")
                console.print(f"  Avg TPS: {analytics['successes']['avg_tps']:.1f}")

    # Build comprehensive stats (always for multi-model comparison)
    result_data = {
        "metadata": {
            "dataset": config.dataset,
            "model": config.model.path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": config.seed,
        },
        "config": {
            "batch_size": config.batch_size,
            "max_tokens": config.generation.max_tokens,
            "temperature": config.generation.temperature,
        },
        "metrics": {
            "accuracy_strict": strict_accuracy,  # Exact match
            "accuracy_flexible": flexible_accuracy,  # Flexible match
            "accuracy": flexible_accuracy,  # Default (backward compat)
            "correct_strict": strict_correct,
            "correct_flexible": flexible_correct,
            "correct": flexible_correct,  # Default (backward compat)
            "total": len(ds),
            "total_time_seconds": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "total_tokens_generated": total_tokens,
            "avg_tps": avg_tps,
        },
        "results": results,
    }

    # Add analytics if generated
    if analytics:
        result_data["analytics"] = analytics

    # Add custom dataset info if applicable
    if config.dataset == "custom" and args.dataset_path:
        result_data["dataset_info"] = {
            "path": args.dataset_path,
            "system_prompt_field": args.system_prompt_field,
            "user_prompt_field": args.user_prompt_field,
            "answer_field": args.answer_field,
        }

    # Save results with comprehensive stats
    if config.write_out:
        output_file = config.output_file or f"results_{config.dataset}_{int(time.time())}.json"
        output_path = Path(config.output_dir) / output_file

        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Also save a CSV for easy analysis
        csv_path = output_path.with_suffix(".csv")
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                if results:
                    fieldnames = ["index", "correct", "question", "expected", "extracted", "tokens", "tps"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in results:
                        writer.writerow({
                            "index": r["index"],
                            "correct": int(r["correct"]),
                            "question": r["question"],
                            "expected": r["expected"],
                            "extracted": r["extracted"] or "",
                            "tokens": r["tokens"],
                            "tps": f"{r['tps']:.1f}",
                        })
            logger.info(f"CSV saved to {csv_path}")
        except Exception as e:
            logger.debug(f"Could not save CSV: {e}")

        # Log to WandB if requested
        if args.wandb:
            run_name = f"{config.dataset}_{config.seed}_{int(time.time())}"
            log_to_wandb(
                run_name,
                config,
                result_data["metrics"],
                results,
                analytics,
                output_path,
            )

    # Return metrics and results for multi-model comparison
    return {
        "metrics": result_data["metrics"],
        "results": results,
    }


if __name__ == "__main__":
    main()
