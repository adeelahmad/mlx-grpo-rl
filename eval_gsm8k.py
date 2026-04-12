#!/usr/bin/env python3
import json
import logging
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
eval_dir = "./eval_gsm8k"
os.makedirs(eval_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(eval_dir, "eval.log")), logging.StreamHandler()],
)
logger = logging.getLogger()

model_name = "Qwen/Qwen3-4B-Instruct-2507"
logger.info(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

logger.info("Loading GSM8K (max 10 samples)...")
ds = load_dataset("openai/gsm8k", "main", split="test")
ds = ds.select(range(min(10, len(ds))))


def extract_answer(text):
    """Extract answer from \\boxed{} format"""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


results = []
correct = 0

for i, sample in enumerate(ds, 1):
    question = sample["question"]
    prompt = f"Question: {question}\nAnswer:"
    expected = sample["answer"].split("#### ")[-1].strip()

    logger.info(f"{i}. Generating (max 1024 tokens)...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.0, top_p=1.0, do_sample=False
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated[len(prompt) :].strip()

    # Extract answer from \boxed{} format
    extracted = extract_answer(generated)

    # Check if extracted answer matches expected
    is_correct = False
    if extracted:
        is_correct = str(extracted) == str(expected) or extracted in generated
    else:
        # Fallback: check if expected answer appears in text
        is_correct = expected in generated

    correct += is_correct

    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    logger.info(
        f"{status} | Expected: {expected[:30]} | Extracted: {extracted if extracted else 'N/A'} | Got: {generated[:100]}..."
    )

    results.append(
        {
            "index": i,
            "question": question[:100],
            "expected": expected,
            "extracted": extracted,
            "generated": generated[:300],
            "correct": is_correct,
        }
    )

accuracy = correct / len(ds) * 100
logger.info(f"\n{'=' * 60}")
logger.info(f"FINAL ACCURACY: {correct}/{len(ds)} = {accuracy:.1f}%")
logger.info(f"{'=' * 60}")

with open(f"{eval_dir}/results.json", "w") as f:
    json.dump(
        {
            "model": model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ds),
            "results": results,
        },
        f,
        indent=2,
    )

logger.info(f"Full results saved to {eval_dir}/results.json")
