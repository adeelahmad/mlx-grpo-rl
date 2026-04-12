"""Apply masking rules to rollout results — zero gradient on matched sequences."""

import re
from contextlib import suppress
from typing import Any


class MaskingRuleEngine:
    """Apply masking rules to RolloutResult segments based on text and token content."""

    def __init__(self, rules: tuple[Any, ...], tokenizer: Any | None = None):
        """Initialize with list of MaskingRuleConfig.

        Args:
            rules: tuple of MaskingRuleConfig objects
            tokenizer: tokenizer for decoding tokens in regex patterns
        """
        self.rules = rules
        self.tokenizer = tokenizer
        self._compiled_regex: dict[int, re.Pattern] = {}  # rule index -> compiled pattern

    def apply_to_result(self, result: Any) -> None:
        """Apply all masking rules to RolloutResult.segments in-place.

        Modifies segment.loss_mask where matched sequences appear.
        """
        if not self.rules:
            return

        # Process each segment — filter by target_segments in rule
        for segment in result.segments:
            if not segment.tokens:
                continue

            # Skip segment if rule doesn't target it
            segment_source = getattr(segment, "source", None)
            if segment_source is None:
                continue

            # Decode segment to text for string/regex matching
            segment_text = ""
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                with suppress(Exception):
                    segment_text = self.tokenizer.decode(segment.tokens, skip_special_tokens=False)

            # Apply each rule that targets this segment type
            for rule_idx, rule in enumerate(self.rules):
                if self._should_apply_rule_to_segment(segment_source, rule):
                    self._apply_rule(segment, rule, rule_idx, segment_text)

    @staticmethod
    def _should_apply_rule_to_segment(segment_source: Any, rule: Any) -> bool:
        """Check if rule should apply to this segment based on target_segments."""
        from mlx_grpo.engine.result import SegmentSource

        if rule.target_segments == "both":
            return True
        elif rule.target_segments == "generated":
            return segment_source == SegmentSource.GENERATED
        elif rule.target_segments == "scaffolded":
            return segment_source == SegmentSource.INJECTED
        return False

    def _apply_rule(self, segment: Any, rule: Any, rule_idx: int, segment_text: str) -> None:
        """Apply a single rule to a segment — modify loss_weight if pattern matches."""
        # Determine which tokens match the pattern
        token_indices_to_match = set()

        if rule.check == "regex":
            token_indices_to_match = self._find_regex_matches(segment_text, segment.tokens, rule)
        elif rule.check == "contains":
            token_indices_to_match = self._find_string_matches(segment_text, segment.tokens, rule)

        # If pattern matches, apply the weight multiplier to the segment's loss_weight
        # (0.0 = mask out entirely, >1.0 = amplify gradient)
        if token_indices_to_match:
            segment.loss_weight *= rule.weight_multiplier

    def _find_string_matches(self, text: str, tokens: list[int], rule: Any) -> set[int]:
        """Find token indices matching a string literal (contains check)."""
        if not text:
            return set()

        pattern = rule.pattern
        matches = set()
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            # Map character position to token index (approximate via cumsum)
            matches.add(self._char_pos_to_token_idx(pos, text, tokens))
            start = pos + 1
        return matches

    def _find_regex_matches(self, text: str, tokens: list[int], rule: Any) -> set[int]:
        """Find token indices matching a regex pattern (use (?i) for case-insensitive)."""
        if not text:
            return set()

        matches = set()
        try:
            # Compile and cache regex (flags handled in pattern via (?i))
            if id(rule) not in self._compiled_regex:
                self._compiled_regex[id(rule)] = re.compile(rule.pattern)
            pattern = self._compiled_regex[id(rule)]

            for match in pattern.finditer(text):
                start_pos = match.start()
                matches.add(self._char_pos_to_token_idx(start_pos, text, tokens))
        except re.error as e:
            print(f"Invalid regex pattern: {rule.pattern} — {e}")
        return matches

    @staticmethod
    def _char_pos_to_token_idx(char_pos: int, text: str, tokens: list[int]) -> int:
        """Map character position in text to approximate token index."""
        # Rough estimate: assume avg ~4 chars per token
        if not tokens:
            return 0
        avg_chars_per_token = max(1, len(text) // len(tokens))
        return min(char_pos // avg_chars_per_token, len(tokens) - 1)
