"""Pure Python region planner for diverse scaffold repetition across repeats.

Divides expert thinking blocks into non-overlapping regions. Each repeat
scaffolds a DIFFERENT region, ensuring cumulative coverage grows monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RegionPlan:
    """The scaffold region assigned to one repeat step."""

    offset_chars: int  # char start position in thinking block
    length_chars: int  # char length (0 = rest of block from offset)
    region_index: int  # 0-based region number
    total_regions: int  # how many regions block is divided into
    covered_frac: float  # cumulative coverage BEFORE this repeat
    is_fallback: bool  # True = coverage exhausted, inject full block


@dataclass(slots=True)
class CoverageState:
    """Mutable coverage state for one (iteration, dataset_row) pair."""

    thinking_text_len: int
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)  # sorted, non-overlapping
    repeat_count: int = 0


# ── Core Functions ───────────────────────────────────────────────────────────


def extract_thinking_text(answer: str) -> str:
    """Strip <think> prefix and </think> suffix from sample.answer.

    Returns '' when no thinking block found.
    """
    text = answer.lstrip()
    if not text.startswith("<think>"):
        return ""
    text = text[len("<think>") :]
    end = text.find("</think>")
    if end != -1:
        text = text[:end]
    return text


def plan_thinking_regions(thinking_len: int, num_regions: int) -> list[tuple[int, int]]:
    """Divide [0, thinking_len) into num_regions equal-ish (start, end) pairs.

    Returns sorted list of non-overlapping char ranges covering the full block.
    """
    if thinking_len == 0 or num_regions <= 0:
        return []
    if num_regions == 1:
        return [(0, thinking_len)]
    region_size = thinking_len // num_regions
    regions = []
    for i in range(num_regions):
        start = i * region_size
        end = (i + 1) * region_size if i < num_regions - 1 else thinking_len
        regions.append((start, end))
    return regions


def compute_region_plan(
    thinking_len: int,
    repeat_index: int,
    num_regions: int,
    density_decay: float,
    coverage_threshold: float,
    state: CoverageState,
) -> RegionPlan:
    """Compute which region of the thinking block this repeat should scaffold.

    repeat_index 0 → first region
    repeat_index 1 → second region, etc.
    Clamps if repeat_index >= num_regions.

    is_fallback=True when all regions are covered AND cumulative coverage
    exceeds coverage_threshold.
    """
    regions = plan_thinking_regions(thinking_len, num_regions)
    clamped_idx = min(repeat_index, len(regions) - 1) if regions else 0
    is_fb = False
    offset_chars = 0
    length_chars = 0

    if regions and clamped_idx < len(regions):
        start, end = regions[clamped_idx]
        offset_chars = start
        length_chars = end - start
        # Check if all regions covered AND coverage above threshold
        covered_frac = coverage_fraction(state.covered_ranges, thinking_len)
        is_fb = covered_frac >= coverage_threshold and repeat_index >= len(regions)

    return RegionPlan(
        offset_chars=offset_chars,
        length_chars=length_chars,
        region_index=clamped_idx,
        total_regions=num_regions,
        covered_frac=coverage_fraction(state.covered_ranges, thinking_len),
        is_fallback=is_fb,
    )


def scale_chunks_for_density(
    base_chunks: int, repeat_index: int, decay: float, min_chunks: int = 1
) -> int:
    """Compute diminishing chunk count for this repeat.

    Earlier repeats get more chunks; later ones get fewer.
    Returns max(min_chunks, round(base_chunks * decay**repeat_index)).
    """
    if base_chunks <= 0:
        return min_chunks
    scaled = base_chunks * (decay**repeat_index)
    return max(min_chunks, round(scaled))


def update_coverage(state: CoverageState, offset: int, length: int) -> float:
    """Add (offset, offset+length) to state.covered_ranges.

    Merges overlapping ranges. Returns updated coverage fraction.
    """
    if length <= 0:
        return coverage_fraction(state.covered_ranges, state.thinking_text_len)
    new_range = (offset, offset + length)
    # Merge new range with all existing ranges
    all_ranges = state.covered_ranges + [new_range]
    merged = []
    for r in sorted(all_ranges):
        if merged and r[0] <= merged[-1][1]:
            # Overlapping or adjacent: merge
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            # Non-overlapping: add new range
            merged.append(r)
    state.covered_ranges = merged
    return coverage_fraction(state.covered_ranges, state.thinking_text_len)


def coverage_fraction(ranges: list[tuple[int, int]], total_len: int) -> float:
    """Compute what fraction of [0, total_len) is covered by ranges.

    Assumes ranges are sorted and non-overlapping.
    """
    if total_len <= 0:
        return 0.0
    covered = sum(e - s for s, e in ranges)
    return min(1.0, covered / total_len)
