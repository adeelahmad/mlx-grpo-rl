"""run_inspect() orchestrator — routes to feature-specific runners.

MLX-dependent via collector/model loading — excluded from coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import InspectConfig


def run_inspect(cfg: InspectConfig) -> int:  # pragma: no cover
    """Main entry point. Returns 0 on success, 1 on error."""
    feature = cfg.feature
    if feature == "compare":
        return _run_compare(cfg)
    if feature == "checkpoint":
        return _run_checkpoint_trajectory(cfg)
    if feature == "reward_strat":
        return _run_reward_stratified(cfg)
    if feature == "lora":
        return _run_lora_isolation(cfg)
    if feature == "fingerprint":
        return _run_type_fingerprints(cfg)
    if feature == "scaffold":
        return _run_scaffold_naked(cfg)
    if feature == "probe_save":
        return _run_save_base_activations(cfg)
    print(f"Unknown feature: {feature}", file=sys.stderr)
    return 1


# ── Private Helpers ───────────────────────────────────────────────────────────


def _load_collector(  # pragma: no cover
    cfg: InspectConfig, model_path: str
) -> tuple[Any, Any, Any]:
    """Load model + collector. Returns (model, tokenizer, collector)."""
    from .collector import ActivationCollector, load_model_for_inspect  # noqa: PLC0415

    model, tokenizer = load_model_for_inspect(model_path)
    collector = ActivationCollector(model, tokenizer, cfg)
    return model, tokenizer, collector


def _ensure_dir(cfg: InspectConfig) -> Path:  # pragma: no cover
    out = Path(cfg.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_compare(cfg: InspectConfig) -> int:  # pragma: no cover
    """Compare two model checkpoints via SHAMAN analysis."""
    from .circuit import trace_model_circuits  # noqa: PLC0415
    from .hypotheses import evaluate_hypotheses, format_shaman_report  # noqa: PLC0415
    from .lens import logit_lens, save_logit_lens_csv  # noqa: PLC0415
    from .metrics import (  # noqa: PLC0415
        compute_layerwise_similarity,
        compute_neuron_metrics,
        compute_rsa_layerwise,
        compute_sublayer_similarity,
    )
    from .plots import plot_layer_comparison, plot_shaman_validation  # noqa: PLC0415

    out = _ensure_dir(cfg)

    model_a, tok_a, coll_a = _load_collector(cfg, cfg.model_path)
    result_a = coll_a.collect(cfg.prompt)

    if cfg.model_b_path is None:
        print("No --model-b specified; comparing model against itself.", file=sys.stderr)
        result_b = result_a
        name_b = "self"
    else:
        _, _, coll_b = _load_collector(cfg, cfg.model_b_path)
        result_b = coll_b.collect(cfg.prompt)
        name_b = Path(cfg.model_b_path).name

    name_a = Path(cfg.model_path).name
    similarity = compute_layerwise_similarity(result_a.sketches, result_b.sketches)
    neuron_metrics = compute_neuron_metrics(result_a.raw_np, result_b.raw_np)
    rsa_scores = compute_rsa_layerwise(
        {k: v.reshape(1, -1) for k, v in result_a.raw_np.items()},
        {k: v.reshape(1, -1) for k, v in result_b.raw_np.items()},
    )
    sublayer_sim = compute_sublayer_similarity(result_a.layer_acts, result_b.layer_acts)

    control_sim: dict[str, dict[str, float]] | None = None
    if cfg.control_prompt:
        ctrl_a = coll_a.collect(cfg.control_prompt)
        ctrl_b = coll_b.collect(cfg.control_prompt) if cfg.model_b_path else ctrl_a
        control_sim = compute_layerwise_similarity(ctrl_a.sketches, ctrl_b.sketches)

    n_layers = len(similarity)
    hyp_results = evaluate_hypotheses(
        similarity, neuron_metrics, sublayer_sim, control_sim, n_layers
    )

    report = format_shaman_report(
        hyp_results, rsa_scores, neuron_metrics, sublayer_sim, name_a, name_b
    )
    print(report)
    (out / "shaman_report.txt").write_text(report)

    plot_layer_comparison(similarity, name_a, name_b, out / "layer_comparison.png")
    if cfg.capture_sublayers:
        plot_shaman_validation(
            similarity,
            rsa_scores,
            sublayer_sim,
            neuron_metrics,
            name_a,
            name_b,
            out / "shaman_validation.png",
        )

    # Logit lens
    rows = logit_lens(
        model_a, None, tok_a, result_a.layer_acts, result_a.track_pos, cfg.top_k_logit_lens
    )
    save_logit_lens_csv(rows, out / "logit_lens.csv")

    # Circuit
    if cfg.trace_generation:
        trace = trace_model_circuits(model_a, tok_a, cfg.prompt, name_a, cfg)
        print(f"Generated: {trace.generated_text[:80]}...")

    print(f"Results saved to {out}")
    return 0


def _run_checkpoint_trajectory(cfg: InspectConfig) -> int:  # pragma: no cover
    from .features import run_checkpoint_trajectory  # noqa: PLC0415
    from .plots import plot_cka_trajectory  # noqa: PLC0415

    _, _, coll = _load_collector(cfg, cfg.model_path)
    out = _ensure_dir(cfg)
    result = run_checkpoint_trajectory(cfg, coll)
    if result.cka_matrix.shape[0] > 0:
        plot_cka_trajectory(
            result.checkpoint_labels, result.cka_matrix, out / "checkpoint_cka_trajectory.png"
        )
        import numpy as np  # noqa: PLC0415

        np.save(out / "cka_matrix.npy", result.cka_matrix)
    print(f"Checkpoint trajectory saved to {out}")
    return 0


def _run_reward_stratified(cfg: InspectConfig) -> int:  # pragma: no cover
    from .features import run_reward_stratified  # noqa: PLC0415
    from .plots import plot_reward_stratified  # noqa: PLC0415

    _, _, coll = _load_collector(cfg, cfg.model_path)
    out = _ensure_dir(cfg)
    result = run_reward_stratified(cfg, coll)
    plot_reward_stratified(
        result.high_similarity, result.low_similarity, out / "reward_stratified.png"
    )
    print(f"Stratified analysis: n_high={result.n_high}, n_low={result.n_low}")
    print(f"Results saved to {out}")
    return 0


def _run_lora_isolation(cfg: InspectConfig) -> int:  # pragma: no cover
    from .features import run_lora_isolation  # noqa: PLC0415
    from .plots import plot_layer_comparison  # noqa: PLC0415

    model, tokenizer, _ = _load_collector(cfg, cfg.model_path)
    out = _ensure_dir(cfg)
    result = run_lora_isolation(cfg, model, tokenizer)
    plot_layer_comparison(result.lora_similarity, "base", "lora", out / "lora_isolation.png")
    print(f"LoRA isolation analysis saved to {out}")
    return 0


def _run_type_fingerprints(cfg: InspectConfig) -> int:  # pragma: no cover
    import numpy as np  # noqa: PLC0415

    from .features import run_type_fingerprints  # noqa: PLC0415

    _, _, coll = _load_collector(cfg, cfg.model_path)
    out = _ensure_dir(cfg)
    result = run_type_fingerprints(cfg, coll)
    np.save(out / "type_fingerprints_cka.npy", result.cross_cka)
    print(f"Type fingerprints for {result.labels} saved to {out}")
    return 0


def _run_scaffold_naked(cfg: InspectConfig) -> int:  # pragma: no cover
    from .features import run_scaffold_naked_comparison  # noqa: PLC0415
    from .plots import plot_layer_comparison  # noqa: PLC0415

    _, _, coll = _load_collector(cfg, cfg.model_path)
    out = _ensure_dir(cfg)
    scaffold_prefix = cfg.control_prompt or "<scaffold> "
    result = run_scaffold_naked_comparison(cfg, coll, scaffold_prefix)
    plot_layer_comparison(
        result.similarity, "naked", "scaffold", out / "scaffold_naked_comparison.png"
    )
    print(
        f"Scaffold vs naked: mean_cka={result.mean_cka:.4f}, mean_cosine={result.mean_cosine:.4f}"
    )
    return 0


def _run_save_base_activations(cfg: InspectConfig) -> int:  # pragma: no cover
    from .cka_probe import CKAProbe  # noqa: PLC0415

    if cfg.probe_base is None:
        print("--probe-base path required for feature=probe_save", file=sys.stderr)
        return 1

    model, tokenizer, _ = _load_collector(cfg, cfg.model_path)
    probe = CKAProbe(model, tokenizer, cfg, cfg.probe_base)
    probe.save_base_activations(cfg.prompt, cfg.probe_base)
    print(f"Base activations saved to {cfg.probe_base}")
    return 0
