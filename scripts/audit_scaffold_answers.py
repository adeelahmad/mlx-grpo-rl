#!/usr/bin/env python3
"""Audit scaffold answer consistency — flag rows where scaffold hints contradict ground truth.

Checks three potential mismatch sources:
  1. [ANSWER INTUITION: X] letter vs ground_truth
  2. \\boxed{X} letter vs ground_truth
  3. Last explicit "answer is X" / "correct answer is X" in reasoning vs ground_truth

Usage:
    python scripts/audit_scaffold_answers.py /path/to/train.jsonl
    python scripts/audit_scaffold_answers.py /path/to/dataset_dir/
    python scripts/audit_scaffold_answers.py /path/to/train.jsonl --fix-out clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns to extract answer letters from scaffold text
_INTUITION = re.compile(r"\[ANSWER INTUITION:\s*([A-Ha-h])\b")
_BOXED = re.compile(r"\\boxed\{([A-Ha-h])\}")
_ARGUES_FOR = re.compile(
    r"(?:(?:correct|right|best)\s+answer\s+is|answer\s*[:=])\s*\(?([A-Ha-h])\)?",
    re.IGNORECASE,
)


def extract_scaffold_letters(text: str) -> dict[str, str]:
    """Extract answer letters from scaffold/answer text."""
    found: dict[str, str] = {}
    m = _INTUITION.search(text)
    if m:
        found["intuition"] = m.group(1).upper()
    m = _BOXED.search(text)
    if m:
        found["boxed"] = m.group(1).upper()
    # Last "answer is X" in text (closest to conclusion)
    matches = _ARGUES_FOR.findall(text)
    if matches:
        found["argues"] = matches[-1].upper()
    return found


def audit_file(path: Path, verbose: bool = False) -> list[dict]:
    """Audit a single JSONL file. Returns list of mismatch dicts."""
    mismatches: list[dict] = []
    total = 0
    checked = 0

    with open(path) as f:
        for lineno, line in enumerate(f):
            row = json.loads(line)
            gt = (row.get("ground_truth") or row.get("expected_answer") or "").strip()
            if not gt or not re.match(r"^[A-Ha-h]$", gt):
                total += 1
                continue  # Skip non-letter ground truths (math answers, etc.)
            gt = gt.upper()
            total += 1
            checked += 1

            answer_text = row.get("answer", "")
            letters = extract_scaffold_letters(answer_text)
            if not letters:
                continue

            for source, letter in letters.items():
                if letter != gt:
                    entry = {
                        "file": str(path),
                        "line": lineno,
                        "source": source,
                        "scaffold_letter": letter,
                        "ground_truth": gt,
                        "type": row.get("type", "?"),
                        "prompt": row.get("prompt", "")[:120],
                    }
                    mismatches.append(entry)
                    if verbose:
                        print(
                            f"  MISMATCH line {lineno}: {source}={letter} vs gt={gt} "
                            f"[{row.get('type', '?')}]  {entry['prompt']}..."
                        )

    print(f"  {path.name}: {total} rows, {checked} letter-GT checked, {len(mismatches)} mismatches")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit scaffold answer consistency")
    parser.add_argument("path", type=Path, help="JSONL file or dataset directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each mismatch")
    parser.add_argument(
        "--fix-out",
        type=Path,
        default=None,
        help="Write clean JSONL (mismatched rows removed) to this path",
    )
    parser.add_argument("--json", action="store_true", help="Output mismatches as JSON")
    args = parser.parse_args(argv)

    # Collect JSONL files
    p = args.path
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    elif p.is_file():
        files = [p]
    else:
        print(f"Not found: {p}", file=sys.stderr)
        return 1

    all_mismatches: list[dict] = []
    for f in files:
        all_mismatches.extend(audit_file(f, verbose=args.verbose))

    print(f"\nTotal mismatches: {len(all_mismatches)}")

    if args.json and all_mismatches:
        print(json.dumps(all_mismatches, indent=2))

    # Optionally write cleaned file (remove rows with mismatches)
    if args.fix_out and all_mismatches:
        bad_lines: set[tuple[str, int]] = {(m["file"], m["line"]) for m in all_mismatches}
        kept = 0
        removed = 0
        with open(args.fix_out, "w") as out:
            for f in files:
                with open(f) as inp:
                    for lineno, line in enumerate(inp):
                        if (str(f), lineno) in bad_lines:
                            removed += 1
                        else:
                            out.write(line)
                            kept += 1
        print(f"Wrote {args.fix_out}: {kept} kept, {removed} removed")

    return 1 if all_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
