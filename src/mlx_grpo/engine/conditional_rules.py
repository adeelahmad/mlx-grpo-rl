"""ConditionalRulesEngine — apply lambda/pattern-based rules to control gradient/reward multipliers.

Evaluates rules against rollout segments and applies per-segment gradient multipliers
and per-result reward multipliers. Fully resilient: all errors warn but never crash.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config.loader import ConditionalRuleConfig
from .result import RolloutResult, Segment, SegmentSource

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuleContext:
    """Unified context passed to all rule conditions and multiplier evaluations."""

    segment: Segment
    reward: float
    answer_correct: bool
    result: RolloutResult | None = None
    is_naked: bool = False
    attempt_count: int = 0
    kl_divergence: float = 0.0
    group_index: int = 0
    sample_index: int = 0
    sequence_tokens: list[int] = field(default_factory=list)
    sequence_proximity_weights: npt.NDArray[Any] | None = None
    segment_token_range: tuple[int, int] = (0, 0)
    prompt_text: str = ""


class ConditionalRulesEngine:
    """Apply conditional gradient and reward multipliers to rollout results.

    Supports:
    - Pattern matching: regex (with case-insensitive via (?i)), contains
    - Conditions: lambda expressions with full RuleContext access
    - Methods: static method invocation ("ClassName.method_name")
    - Multipliers: float literals, lambdas, or methods for both gradient and reward
    - Resilience: all errors logged as warnings, rules skipped on failure

    Example usage::

        rules = [
            ConditionalRuleConfig(
                pattern="(?i)error",
                check="regex",
                gradient="0.0",      # Mask (zero gradient)
                reward="-0.5",       # Reduce reward
            ),
            ConditionalRuleConfig(
                when="lambda ctx: not ctx.answer_correct",
                gradient="0.5",
                reward="-1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        engine.apply_to_result(result, reward=0.8, answer_correct=True)
    """

    def __init__(
        self,
        rules: tuple[ConditionalRuleConfig, ...],
        method_registry: dict[str, Any] | None = None,
    ) -> None:
        """Initialize engine with rules and optional method registry.

        Args:
            rules: Tuple of ConditionalRuleConfig rules (normally from config).
            method_registry: Optional dict of {class_name: class_obj} for static method
                resolution. Used when rule gradient/reward is "ClassName.method".
        """
        self._rules = rules
        self._registry = method_registry or {}

    def apply_to_result(
        self,
        result: RolloutResult,
        reward: float,
        answer_correct: bool = False,
        is_naked: bool = False,
        attempt_count: int = 0,
        kl_divergence: float = 0.0,
        group_index: int = 0,
        sample_index: int = 0,
        sequence_proximity_weights: npt.NDArray[Any] | None = None,
        segment_token_range: tuple[int, int] | None = None,
        prompt_text: str = "",
    ) -> None:
        """Apply all rules to segments in result, modifying in-place.

        Skips already-masked segments. Tracks compounding reward multipliers in
        result.metadata["reward_multiplier"]. Records token-level changes in
        result.metadata["token_changes"] for explainability.

        Args:
            result: RolloutResult containing segments to modify.
            reward: Baseline reward for this result.
            answer_correct: Whether the answer was correct.
            is_naked: Whether this is a naked (unscaffolded) rollout.
            attempt_count: Attempt number (for adaptive rules).
            kl_divergence: KL divergence from reference model.
            group_index: Index within the current sample group.
            sample_index: Overall sample index.
            sequence_proximity_weights: Token-level proximity weights (by reference).
            segment_token_range: (start, end) token indices for the first segment.
            prompt_text: The input prompt text.
        """
        if not self._rules:
            return

        # Ensure segment_token_range is valid
        if segment_token_range is None:
            segment_token_range = (0, 0)

        # Initialize token changes tracking
        token_changes: list[dict[str, Any]] = []
        result.metadata["token_changes"] = token_changes

        rules_applied_count = 0
        for segment_idx, segment in enumerate(result.segments):
            if segment.masked:
                continue

            # Build context for this segment
            ctx = RuleContext(
                segment=segment,
                reward=reward,
                answer_correct=answer_correct,
                result=result,
                is_naked=is_naked,
                attempt_count=attempt_count,
                kl_divergence=kl_divergence,
                group_index=group_index,
                sample_index=sample_index,
                sequence_tokens=result.prompt_tokens,
                sequence_proximity_weights=sequence_proximity_weights,
                segment_token_range=segment_token_range,
                prompt_text=prompt_text,
            )

            # Apply each rule to this segment
            for rule_idx, rule in enumerate(self._rules):
                if not self._should_target(segment, rule):
                    continue

                # Check condition (when or pattern)
                match_reason = ""
                try:
                    if rule.when:
                        if not self._when_matches(ctx, rule):
                            continue
                        match_reason = f"when: {rule.when[:60]}"
                    elif rule.pattern:
                        if not self._pattern_matches(segment, rule):
                            continue
                        match_reason = f"pattern: {rule.pattern[:60]}"
                except (SyntaxError, ValueError, AttributeError, KeyError, re.error) as e:
                    logger.warning(f"Rule condition raised {type(e).__name__}: {e}. Skipping rule.")
                    continue

                # Resolve and apply gradient multiplier
                try:
                    grad_mult = self._resolve_value(rule.gradient, ctx)

                    # FIX: Find matched tokens, don't suppress entire segment!
                    # Prevents one word match ("possibly" in 620 tokens) from nuking trace
                    affected_token_indices = self._find_matched_token_spans(segment, rule)

                    if affected_token_indices:
                        # Apply suppression only to matched tokens ± 2-token window
                        old_weight = segment.loss_weight
                        # Track how many tokens are affected
                        total_affected = len(affected_token_indices)
                        total_tokens = len(segment.tokens)

                        # Apply fractional suppression proportional to affected tokens
                        # E.g., if 3 tokens match in 620-token segment:
                        # Effective multiplier = 1 - (0.7 * 3/620) = 0.997 (almost no effect)
                        # Instead of: 0.3 (70% destroyed)
                        fraction_affected = min(1.0, total_affected / max(1, total_tokens))
                        effective_mult = 1.0 - (1.0 - grad_mult) * fraction_affected

                        segment.loss_weight *= effective_mult
                        new_weight = segment.loss_weight

                        # Record the change
                        for tok_idx in affected_token_indices:
                            token_changes.append(
                                {
                                    "token_id": int(segment.tokens[tok_idx]),
                                    "token_index": tok_idx,
                                    "segment_index": segment_idx,
                                    "before_weight": float(old_weight),
                                    "after_weight": float(new_weight),
                                    "multiplier": float(grad_mult),
                                    "rule_idx": rule_idx,
                                    "change_type": "gradient_token_level",
                                    "match_reason": match_reason,
                                    "tokens_affected": total_affected,
                                    "segment_length": total_tokens,
                                    "effective_multiplier": float(effective_mult),
                                }
                            )

                        rules_applied_count += 1
                        logger.info(
                            f"[Rule {rule_idx}] Token-level suppression: "
                            f"{total_affected}/{total_tokens} tokens matched. "
                            f"Segment weight: {old_weight:.4f}→{new_weight:.4f} "
                            f"(rule={grad_mult:.3f}, effective={effective_mult:.3f}) "
                            f"— {match_reason}"
                        )
                except (TypeError, ValueError, SyntaxError, AttributeError, KeyError) as e:
                    logger.warning(
                        f"Failed to resolve gradient {rule.gradient!r}: {e}. Skipping rule."
                    )
                    continue

                # Resolve and apply reward multiplier (stored in metadata)
                try:
                    reward_mult = self._resolve_value(rule.reward, ctx)
                    prev = result.metadata.get("reward_multiplier", 1.0)
                    # Rewards are ADDITIVE: sum deltas, don't compound them
                    # prev is accumulated delta; reward_mult is this rule's delta
                    new = prev + reward_mult
                    result.metadata["reward_multiplier"] = new
                    logger.info(
                        f"[Rule {rule_idx}] Reward mult: {prev:.4f}→{new:.4f} "
                        f"(delta {reward_mult:+.3f}) — {match_reason}"
                    )
                except (TypeError, ValueError, SyntaxError, AttributeError, KeyError) as e:
                    logger.warning(f"Failed to resolve reward {rule.reward!r}: {e}. Skipping rule.")
                    continue

        # Summary log
        if rules_applied_count > 0:
            final_reward_mult = result.metadata.get("reward_multiplier", 1.0)
            token_summary = self._summarize_token_changes(token_changes)
            logger.info(
                f"Applied {rules_applied_count} conditional rule(s) to {len(result.segments)} "
                f"segment(s). Final reward_multiplier: {final_reward_mult:.4f}.\n"
                f"Token-level changes:\n{token_summary}"
            )

    def _should_target(self, segment: Segment, rule: ConditionalRuleConfig) -> bool:
        """Check if rule should apply to this segment based on target_segments."""
        target = rule.target_segments
        source = segment.source

        if target == "both":  # noqa: SIM116
            return True
        elif target == "generated":
            return source == SegmentSource.GENERATED
        elif target == "scaffolded":
            return source == SegmentSource.INJECTED
        return False

    def _apply_gradient_multiplier(self, segment: Segment, multiplier: float) -> None:
        """Apply gradient multiplier to segment's loss_weight."""
        segment.loss_weight *= multiplier

    def _pattern_matches(self, segment: Segment, rule: ConditionalRuleConfig) -> bool:
        """Check if pattern (regex or contains) matches segment text.

        Raises re.error on bad regex.
        """
        text = segment.text
        if rule.check == "regex":
            return bool(re.search(rule.pattern, text))
        elif rule.check == "contains":
            return rule.pattern in text
        return False

    def _when_matches(self, ctx: RuleContext, rule: ConditionalRuleConfig) -> bool:
        """Evaluate when condition (lambda or static method).

        Raises SyntaxError on bad lambda, AttributeError/KeyError on missing method.
        """
        when_str = rule.when.strip()

        # Try static method resolution first
        if "." in when_str and not when_str.startswith("lambda"):
            result = self._invoke_method(when_str, ctx)
            return bool(result) if result is not None else False

        # Try lambda evaluation
        if when_str.startswith("lambda"):
            result = self._eval_lambda(when_str, ctx)
            # eval() returns the lambda function itself, need to call it
            if callable(result):
                # Inspect lambda parameter to decide what to pass
                import inspect

                sig = inspect.signature(result)
                param_name = list(sig.parameters.keys())[0] if sig.parameters else "ctx"
                # If parameter is "prompt", pass prompt_text; otherwise pass full context
                arg = ctx.prompt_text if param_name == "prompt" else ctx
                result = result(arg)
            return bool(result) if result is not None else False

        # Fallback: assume it's a method path anyway
        result = self._invoke_method(when_str, ctx)
        return bool(result) if result is not None else False

    def _pattern_matches_str(self, text: str, rule: ConditionalRuleConfig) -> bool:
        """Check if pattern (regex or contains) matches text string."""
        if rule.check == "regex":
            try:
                return bool(re.search(rule.pattern, text))
            except re.error:
                return False
        elif rule.check == "contains":
            return rule.pattern in text
        return False

    def _find_matched_token_spans(
        self, segment: Segment, rule: ConditionalRuleConfig, window: int = 2
    ) -> list[int]:
        """Find token indices that match the pattern, with optional context window.

        Args:
            segment: Segment to search.
            rule: Rule with pattern to match.
            window: Context tokens on each side (default 2).

        Returns:
            List of token indices to suppress (matched tokens ± window).
        """
        text = segment.text
        if not text:
            return []

        matched_indices = set()
        pattern_found = False

        # Find all pattern matches in the text
        try:
            if rule.check == "regex":
                for match in re.finditer(rule.pattern, text):
                    pattern_found = True
                    start_char = match.start()
                    end_char = match.end()
                    # Find tokens that overlap with this match
                    char_pos = 0
                    for tok_idx, _token_id in enumerate(segment.tokens):
                        # Approximate token span (very rough)
                        token_text_len = 4  # Average token ≈ 4 chars
                        char_end = char_pos + token_text_len

                        # Check if token overlaps with match
                        if char_pos < end_char and char_end > start_char:
                            # Add this token and ±window neighbors
                            for offset in range(-window, window + 1):
                                neighbor_idx = tok_idx + offset
                                if 0 <= neighbor_idx < len(segment.tokens):
                                    matched_indices.add(neighbor_idx)

                        char_pos = char_end

            elif rule.check == "contains":
                # Find the pattern substring
                idx = text.find(rule.pattern)
                if idx >= 0:
                    pattern_found = True
                    # Rough token mapping: assume 4 chars per token
                    start_token = max(0, idx // 4)
                    pattern_end = idx + len(rule.pattern)
                    end_token = min(len(segment.tokens), pattern_end // 4 + 1)

                    # Add matched tokens and window
                    for tok_idx in range(start_token, end_token):
                        for offset in range(-window, window + 1):
                            neighbor_idx = tok_idx + offset
                            if 0 <= neighbor_idx < len(segment.tokens):
                                matched_indices.add(neighbor_idx)

        except (re.error, ValueError, AttributeError):
            # If anything fails, return empty (don't crash)
            return []

        # Fallback for small segments: if pattern matched but no tokens found via
        # character mapping, include all tokens. Character mapping is unreliable
        # for very small segments where token boundaries don't align with 4-char assumption.
        if pattern_found and not matched_indices and len(segment.tokens) <= 10:
            matched_indices = set(range(len(segment.tokens)))

        return sorted(matched_indices)

    def _resolve_value(self, value_str: str, ctx: RuleContext) -> float:
        """Resolve gradient/reward value: float literal, lambda, or method.

        Raises TypeError/ValueError on non-numeric, NaN, or Inf.
        """
        value_str = value_str.strip()

        # Try float literal first
        try:
            val = float(value_str)
            if np.isnan(val):
                raise ValueError(f"Multiplier {value_str!r} resolved to NaN.")
            if np.isinf(val):
                raise ValueError(f"Multiplier {value_str!r} resolved to Inf.")
            return val
        except ValueError:
            pass

        # Try lambda
        if value_str.startswith("lambda"):
            lambda_func = self._eval_lambda(value_str, ctx)
            # eval() returns the lambda function itself, need to call it
            if callable(lambda_func):  # noqa: SIM108
                val = lambda_func(ctx)
            else:
                val = lambda_func
            if not isinstance(val, int | float | bool):
                raise TypeError(f"Lambda {value_str!r} returned non-numeric value {val!r}.")
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Lambda {value_str!r} returned NaN or Inf.")
            return val

        # Try method
        if "." in value_str:
            val = self._invoke_method(value_str, ctx)
            if not isinstance(val, int | float | bool):
                raise TypeError(f"Method {value_str!r} returned non-numeric value {val!r}.")
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Method {value_str!r} returned NaN or Inf.")
            return val

        raise ValueError(f"Could not resolve multiplier {value_str!r}.")

    def _eval_lambda(self, lambda_str: str, ctx: RuleContext) -> Any:
        """Safely evaluate lambda string in restricted context.

        Raises SyntaxError on bad syntax, other exceptions on runtime errors.
        """
        # Safe globals: only numpy and basic builtins
        safe_globals = {
            "np": np,
            "len": len,
            "max": max,
            "min": min,
            "abs": abs,
            "sum": sum,
            "any": any,
            "all": all,
            "str": str,
            "int": int,
            "float": float,
        }
        safe_locals = {"ctx": ctx}
        result = eval(lambda_str, safe_globals, safe_locals)
        return result

    def _invoke_method(self, method_path: str, ctx: RuleContext) -> Any:
        """Invoke static method via "ClassName.method_name" string.

        Raises AttributeError/KeyError if class or method not found.
        """
        parts = method_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid method path {method_path!r}. Expected 'ClassName.method_name'."
            )

        class_name, method_name = parts

        # Look up class in registry
        if class_name not in self._registry:
            raise KeyError(f"Class {class_name!r} not found in method_registry.")

        cls = self._registry[class_name]

        # Get method
        if not hasattr(cls, method_name):
            raise AttributeError(f"Method {method_name!r} not found on class {class_name!r}.")

        method = getattr(cls, method_name)

        # Invoke with context
        return method(ctx)

    def _summarize_token_changes(self, token_changes: list[dict[str, Any]]) -> str:
        """Generate a human-readable summary of token-level changes.

        Returns a concise string showing which rules changed which tokens,
        including the effective suppression ratio (tokens_affected/segment_length).
        """
        if not token_changes:
            return "No token changes."

        # Group changes by rule
        by_rule: dict[int, list[dict[str, Any]]] = {}
        for change in token_changes:
            rule_idx = change["rule_idx"]
            if rule_idx not in by_rule:
                by_rule[rule_idx] = []
            by_rule[rule_idx].append(change)

        lines = []
        for rule_idx in sorted(by_rule.keys()):
            changes = by_rule[rule_idx]
            unique_tokens = len(set(c["token_id"] for c in changes))
            sample_change = changes[0]

            # Include token-level details
            tokens_affected = sample_change.get("tokens_affected", 0)
            segment_length = sample_change.get("segment_length", 1)
            effective_mult = sample_change.get("effective_multiplier", 1.0)
            rule_mult = sample_change.get("multiplier", 1.0)

            if tokens_affected and segment_length:
                affected_pct = 100 * tokens_affected / segment_length
                lines.append(
                    f"  Rule {rule_idx}: {tokens_affected}/{segment_length} tokens matched "
                    f"({affected_pct:.1f}%). "
                    f"Rule mult={rule_mult:.3f}, effective={effective_mult:.3f} "
                    f"({sample_change['match_reason']})"
                )
            else:
                lines.append(
                    f"  Rule {rule_idx}: {unique_tokens} token(s) "
                    f"{sample_change['before_weight']:.4f}→"
                    f"{sample_change['after_weight']:.4f} "
                    f"({sample_change['match_reason']})"
                )

        return "\n".join(lines)
