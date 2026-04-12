#!/usr/bin/env python3
"""
Validate ConditionalRulesEngine token-level suppression fix impact.

Usage:
  python scripts/validate_conditional_rules_fix.py \
    --rollouts rollouts.jsonl \
    --baseline-mean 0.323 \
    --output validation_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Any
import numpy as np


def load_rollouts(path: str) -> list[dict[str, Any]]:
    """Load rollouts from JSONL file."""
    rollouts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))
    return rollouts


def analyze_token_changes(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze token_changes metadata from rollouts."""
    all_changes = []
    by_rule = {}

    for rollout in rollouts:
        changes = rollout.get("metadata", {}).get("token_changes", [])
        all_changes.extend(changes)

        for change in changes:
            rule_idx = change["rule_idx"]
            if rule_idx not in by_rule:
                by_rule[rule_idx] = []
            by_rule[rule_idx].append(change)

    if not all_changes:
        return {"status": "No token_changes found in rollouts"}

    effective_mults = [c["effective_multiplier"] for c in all_changes]
    tokens_affected = [c.get("tokens_affected", 0) for c in all_changes]
    segment_lengths = [c.get("segment_length", 1) for c in all_changes]

    return {
        "total_changes": len(all_changes),
        "effective_multiplier": {
            "mean": float(np.mean(effective_mults)),
            "median": float(np.median(effective_mults)),
            "std": float(np.std(effective_mults)),
            "min": float(np.min(effective_mults)),
            "max": float(np.max(effective_mults)),
        },
        "affected_tokens": {
            "mean": float(np.mean(tokens_affected)),
            "median": float(np.median(tokens_affected)),
        },
        "segment_lengths": {
            "mean": float(np.mean(segment_lengths)),
            "median": float(np.median(segment_lengths)),
        },
        "by_rule": {
            rule_idx: {
                "count": len(changes),
                "mean_effective_mult": float(np.mean([c["effective_multiplier"] for c in changes])),
                "mean_tokens_affected": float(np.mean([c.get("tokens_affected", 0) for c in changes])),
            }
            for rule_idx, changes in sorted(by_rule.items())
        },
    }


def analyze_rewards(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze reward metrics."""
    if not rollouts:
        return {}

    rewards = []
    correct_advantages = []
    wrong_advantages = []

    for rollout in rollouts:
        if "reward" in rollout:
            rewards.append(rollout["reward"])

        adv = rollout.get("advantage", 0)
        if rollout.get("answer_correct"):
            correct_advantages.append(adv)
        else:
            wrong_advantages.append(adv)

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0,
        "std_reward": float(np.std(rewards)) if rewards else 0,
        "correct_advantage": {
            "mean": float(np.mean(correct_advantages)) if correct_advantages else 0,
            "count": len(correct_advantages),
        },
        "wrong_advantage": {
            "mean": float(np.mean(wrong_advantages)) if wrong_advantages else 0,
            "count": len(wrong_advantages),
        },
    }


def generate_report(
    rollouts: list[dict[str, Any]],
    token_analysis: dict[str, Any],
    reward_analysis: dict[str, Any],
    baseline_mean: float = 0.323,
) -> str:
    """Generate markdown validation report."""
    lines = [
        "# ConditionalRulesEngine Fix Validation Report",
        "",
        "## Summary",
        f"- **Rollouts analyzed**: {len(rollouts)}",
        f"- **Token changes recorded**: {token_analysis.get('total_changes', 0)}",
        "",
    ]

    # Reward metrics
    lines.extend([
        "## Reward Metrics",
        "",
        "### Current Performance",
        f"- **Mean reward**: {reward_analysis.get('mean_reward', 0):.4f}",
        f"- **Std dev**: {reward_analysis.get('std_reward', 0):.4f}",
        "",
        "### Advantage Distribution",
        f"- **Correct answers**: {reward_analysis['correct_advantage']['mean']:.4f} "
        f"({reward_analysis['correct_advantage']['count']} samples)",
        f"- **Wrong answers**: {reward_analysis['wrong_advantage']['mean']:.4f} "
        f"({reward_analysis['wrong_advantage']['count']} samples)",
        "",
    ])

    # Baseline comparison
    current = reward_analysis.get("mean_reward", 0)
    improvement = (current - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
    lines.extend([
        "### Comparison to Baseline",
        f"- **Baseline mean**: {baseline_mean:.4f}",
        f"- **Current mean**: {current:.4f}",
        f"- **Improvement**: {improvement:+.1f}% {'✅' if improvement >= 5 else '⚠️'}",
        "",
    ])

    # Token-level analysis
    lines.extend([
        "## Token-Level Suppression Analysis",
        "",
        "### Effective Multiplier Distribution",
        f"- **Mean**: {token_analysis['effective_multiplier']['mean']:.4f}",
        f"- **Median**: {token_analysis['effective_multiplier']['median']:.4f}",
        f"- **Std**: {token_analysis['effective_multiplier']['std']:.4f}",
        f"- **Range**: [{token_analysis['effective_multiplier']['min']:.4f}, "
        f"{token_analysis['effective_multiplier']['max']:.4f}]",
        "",
    ])

    # Check for segment-level behavior
    mean_mult = token_analysis["effective_multiplier"]["mean"]
    if mean_mult < 0.5:
        lines.append("⚠️ **WARNING**: Mean effective_mult < 0.5. May still be segment-level!")
    elif mean_mult > 0.95:
        lines.append("✅ **GOOD**: Mean effective_mult ~1.0. Token-level proportional working!")
    else:
        lines.append("⚠️ **CHECK**: Mean effective_mult moderate. Verify by rule below.")

    lines.extend([
        "",
        "### Affected Tokens per Change",
        f"- **Mean**: {token_analysis['affected_tokens']['mean']:.1f} tokens",
        f"- **Median**: {token_analysis['affected_tokens']['median']:.1f} tokens",
        "",
        "### Segment Sizes",
        f"- **Mean**: {token_analysis['segment_lengths']['mean']:.1f} tokens",
        f"- **Median**: {token_analysis['segment_lengths']['median']:.1f} tokens",
        "",
    ])

    # Per-rule analysis
    lines.extend([
        "## Per-Rule Analysis",
        "",
    ])

    for rule_idx, stats in token_analysis["by_rule"].items():
        lines.append(f"### Rule {rule_idx}")
        lines.append(f"- **Changes**: {stats['count']}")
        lines.append(f"- **Mean effective_mult**: {stats['mean_effective_mult']:.4f}")
        lines.append(f"- **Mean tokens affected**: {stats['mean_tokens_affected']:.1f}")
        lines.append("")

    # Validation checks
    lines.extend([
        "## Validation Checks",
        "",
    ])

    checks = [
        ("Token-level working", mean_mult > 0.95, "effective_mult > 0.95"),
        ("Improvement vs baseline", improvement >= 5, "reward change > +5%"),
        ("Advantage fairness", abs(reward_analysis['correct_advantage']['mean']
         - reward_analysis['wrong_advantage']['mean']) > 1.0, "correct - wrong > 1.0"),
    ]

    for check_name, passed, criteria in checks:
        status = "✅" if passed else "❌"
        lines.append(f"{status} {check_name}: {criteria}")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "If all checks pass:",
        "1. Merge `feature/conditional-rules` to master",
        "2. Deploy to production training",
        "3. Monitor for continued improvement",
        "",
        "If checks fail:",
        "1. Check `token_changes` metadata for segment-level behavior",
        "2. Review logs for ConditionalRulesEngine warnings",
        "3. File issue with token_changes samples",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ConditionalRulesEngine token-level suppression fix"
    )
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl")
    parser.add_argument("--baseline-mean", type=float, default=0.323, help="Baseline mean reward")
    parser.add_argument("--output", type=str, help="Output report path (stdout if not set)")

    args = parser.parse_args()

    print("Loading rollouts...", flush=True)
    rollouts = load_rollouts(args.rollouts)
    print(f"Loaded {len(rollouts)} rollouts", flush=True)

    print("Analyzing token changes...", flush=True)
    token_analysis = analyze_token_changes(rollouts)

    print("Analyzing rewards...", flush=True)
    reward_analysis = analyze_rewards(rollouts)

    print("Generating report...", flush=True)
    report = generate_report(rollouts, token_analysis, reward_analysis, args.baseline_mean)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
