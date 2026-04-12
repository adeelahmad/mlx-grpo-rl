#!/usr/bin/env python3
"""
PyTorch/CUDA inference — drop-in replacement for mlx_lm.generate.

Usage:
  pytorch_infer.py --model <model_id> --prompt <text> [options]
  pytorch_infer.py --model <model_id> --adapter-path <path> --prompt <text> [options]

Examples:
  pytorch_infer.py --model Qwen/Qwen2.5-4B-Instruct --prompt "What is 2+2?"
  pytorch_infer.py --model Qwen/Qwen2.5-4B-Instruct \
    --adapter-path configs/adapters/ylora-4B-qwen3 --prompt "What is 2+2?"
"""

import argparse
import sys
from pathlib import Path

import torch


def infer_pytorch(
    model_id: str,
    adapter_path: str | None,
    prompt: str,
    max_tokens: int = 32000,
    temp: float = 0.8,
    top_p: float = 0.95,
    verbose: bool = False,
    seed: int | None = None,
) -> str:
    """Infer using PyTorch + transformers + PEFT."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if seed is not None:
        torch.manual_seed(seed)

    if verbose:
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
        if torch.cuda.is_available():
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[INFO] CUDA memory: {mem_gb:.1f} GB", file=sys.stderr)
        print(f"[INFO] Loading model: {model_id}", file=sys.stderr)

    # Load base model on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    # Ensure model is on correct device (explicit after loading)
    model = model.to(device)

    # Load adapter if provided
    if adapter_path:
        adapter_path = Path(adapter_path)
        if adapter_path.exists():
            try:
                # For PyTorch format: load from numbered dir or latest symlink
                if adapter_path.is_dir():
                    # It's a directory with PEFT checkpoint
                    latest = adapter_path / "latest"
                    if latest.is_symlink():
                        latest = latest.resolve()

                    if latest.exists() and (latest / "adapter_config.json").exists():
                        if verbose:
                            print(f"[INFO] Loading adapter from: {latest}", file=sys.stderr)
                        model = PeftModel.from_pretrained(model, str(latest))
                        # Move adapter to device
                        model = model.to(device)
                    else:
                        if verbose:
                            print(
                                f"[WARN] No valid adapter found in {adapter_path}", file=sys.stderr
                            )
                else:
                    if verbose:
                        print(
                            f"[WARN] Adapter path is not a directory: {adapter_path}",
                            file=sys.stderr,
                        )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to load adapter: {e}", file=sys.stderr)
        else:
            if verbose:
                print(f"[WARN] Adapter path does not exist: {adapter_path}", file=sys.stderr)

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if verbose:
        print(f"[INFO] Device: {device}", file=sys.stderr)
        print(f"[INFO] Model dtype: {model.dtype}", file=sys.stderr)
        print(f"[INFO] Generating {max_tokens} tokens with temperature {temp}...", file=sys.stderr)

    # Apply chat template (prompt is treated as user message)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_formatted = prompt

    # Prepare input
    if verbose:
        print("[INFO] Tokenizing prompt...", file=sys.stderr)
    inputs = tokenizer(prompt_formatted, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    if verbose:
        print("[INFO] Starting generation (streaming)...", file=sys.stderr)

    # Stream tokens in real-time
    from threading import Thread

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, skip_prompt=False)

    # Build stop token list: EOS + common chat end tokens
    stop_tokens = [tokenizer.eos_token_id]
    for marker in ("<|im_end|>", "<|end|>", "<|eot_id|>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if ids:
            stop_tokens.append(ids[-1])
    # Deduplicate while preserving order, drop None
    stop_tokens = list(dict.fromkeys(t for t in stop_tokens if t is not None))

    # Run generation in thread so we can stream output
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        eos_token_id=stop_tokens,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream output as it's generated
    response = ""
    try:
        for text in streamer:
            response += text
            print(text, end="", flush=True)
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Generation failed: {e}", file=sys.stderr)
        raise
    finally:
        thread.join()

    return response


def main():
    import time

    parser = argparse.ArgumentParser(
        description="PyTorch/CUDA inference (mlx_lm.generate replacement)"
    )
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--adapter-path",
        "-ap",
        default=None,
        help="Path to PEFT adapter",
    )
    parser.add_argument("--max-tokens", type=int, default=32000, help="Max new tokens")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose logging")

    args = parser.parse_args()

    # Print prompt
    print(args.prompt)
    print("=" * 70)

    # Time inference
    start_time = time.time()
    response = infer_pytorch(
        model_id=args.model,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        verbose=args.verbose,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    # Count tokens in response
    response_lines = response.split("\n")
    completion = "\n".join(response_lines)

    # Print completion
    print(completion)
    print("=" * 70)

    # Estimate tokens (rough: split by spaces)
    response_tokens = len(response.split())
    tok_per_sec = response_tokens / elapsed if elapsed > 0 else 0

    print("\nStats:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens (est): {response_tokens}")
    print(f"  Tokens/sec: {tok_per_sec:.1f}")


if __name__ == "__main__":
    main()
