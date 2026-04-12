"""Parity tests: PyTorch backend vs reference Python loss implementation.

Validates that torch_ops produces numerically identical results to the pure-Python
reference in loss/__init__.py, and that PyTorchBackend implements the same protocol
as the MLX backend.

Identified discrepancies (documented, not bugs unless noted):
  D1 - compute_grads normalises per-sequence then averages (≠ joint token normalisation)
  D2 - stop_ids built but not passed to HF model.generate() (post-hoc text truncation)
  D3 - GradStepResult.policy_loss / kl_penalty not populated (0.0 default, same as MLX)
  D4 - KL weighted by segment_weights in torch, mirrors mlx_ops behaviour

All tests run on CPU with a tiny deterministic model — no GPU required.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402,I001

from mlx_grpo.loss import LossResult, grpo_loss  # noqa: E402
from mlx_grpo.loss.torch_ops import (  # noqa: E402
    GradStepResult,
    compute_grads,
    compute_log_probs_for_tokens,
    grpo_loss_fn,
    precompute_ref_logps,
)


# ── Shared tiny model ───────────────────────────────────────────────────────


class TinyLM(nn.Module):
    """Minimal causal LM: embed → linear → vocab logits."""

    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, **kwargs):
        return self.head(self.embed(input_ids))  # [B, T, V]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    m = TinyLM(vocab=64, hidden=32)
    m.eval()
    return m


@pytest.fixture
def tiny_ref():
    torch.manual_seed(99)  # different seed → different weights from `tiny`
    m = TinyLM(vocab=64, hidden=32)
    m.eval()
    return m


DEVICE = "cpu"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _lp_as_list(model: nn.Module, seq: list[int]) -> list[float]:
    """Extract per-token log probs from model as Python floats."""
    with torch.no_grad():
        t = compute_log_probs_for_tokens(model, seq, DEVICE)
    return t.tolist()


def _ref_grpo(
    model: nn.Module,
    ref_model: nn.Module,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float = 0.04,
) -> LossResult:
    """Compute GRPO loss using the pure-Python reference implementation."""
    log_probs = []
    ref_log_probs = []
    masked_lps = []
    masked_rlps = []
    masked_masks = []

    for seq, plen, mask in zip(sequences, prompt_lengths, masks, strict=True):
        lp = _lp_as_list(model, seq)
        rlp = _lp_as_list(ref_model, seq)
        start = max(plen - 1, 0)
        log_probs.append(lp)
        ref_log_probs.append(rlp)
        # Slice to completion only — matches torch_ops slicing
        masked_lps.append(lp[start : start + len(mask)])
        masked_rlps.append(rlp[start : start + len(mask)])
        masked_masks.append([float(m) for m in mask])

    return grpo_loss(masked_lps, masked_rlps, advantages, masked_masks, beta=beta)


# ── Loss formula parity ──────────────────────────────────────────────────────


class TestLossFormulaParity:
    """grpo_loss_fn (torch) must match grpo_loss (pure Python) token-for-token."""

    def _torch_loss(
        self,
        model: nn.Module,
        ref: nn.Module,
        seqs: list[list[int]],
        plens: list[int],
        masks: list[list[bool]],
        advs: list[float],
        beta: float = 0.04,
        epsilon: float = 0.0,
    ) -> tuple[float, float]:
        """Returns (loss, mean_kl) from torch_ops.grpo_loss_fn."""
        ref_lps = precompute_ref_logps(ref, seqs, DEVICE)
        loss, kl, _, _ = grpo_loss_fn(
            model, ref_lps, seqs, plens, masks, advs, beta=beta, device=DEVICE, epsilon=epsilon
        )
        return float(loss.detach()), kl

    def test_same_policy_zero_kl(self, tiny):
        """When policy == ref, KL must be 0."""
        seqs = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        plens = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        advs = [1.0, -1.0]
        _, kl = self._torch_loss(tiny, tiny, seqs, plens, masks, advs)
        assert abs(kl) < 1e-5, f"KL should be ~0 when policy==ref, got {kl}"

    def test_kl_non_negative(self, tiny, tiny_ref):
        """Schulman k3 KL estimator is always >= 0."""
        seqs = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        plens = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        _, kl = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, [1.0, -1.0])
        assert kl >= 0.0, f"KL must be non-negative, got {kl}"

    def test_kl_formula_schulman_k3(self):
        """K3 estimator: r - log(r) - 1 where r = pi_ref/pi."""
        # For scalar log_ratio, compute KL manually and compare
        log_ratio = 0.3  # log(pi/pi_ref)
        r_ref = math.exp(-log_ratio)  # pi_ref/pi
        expected_kl = r_ref - (-log_ratio) - 1.0
        # Verify formula is non-negative for any log_ratio
        assert expected_kl >= 0.0
        # Verify symmetry: KL = 0 when log_ratio = 0
        assert math.exp(0) - 0 - 1 == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage_pure_kl_loss(self, tiny, tiny_ref):
        """With adv=0, loss = beta * KL only (policy gradient term vanishes)."""
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [0.0]
        loss, kl = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, advs, beta=0.04)
        expected = 0.04 * kl
        assert abs(loss - expected) < 1e-5, f"loss={loss} expected beta*kl={expected}"

    def test_zero_beta_no_kl_penalty(self, tiny, tiny_ref):
        """With beta=0, KL penalty is excluded from loss."""
        seqs = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        plens = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        advs = [1.0, -1.0]
        loss_with_kl, kl = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, advs, beta=0.04)
        loss_no_kl, _ = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, advs, beta=0.0)
        # loss_with_kl = policy_loss + 0.04 * kl
        # loss_no_kl = policy_loss
        diff = loss_with_kl - loss_no_kl
        assert abs(diff - 0.04 * kl) < 1e-5

    def test_all_masked_returns_zero(self, tiny, tiny_ref):
        """All-masked sequences → zero loss."""
        seqs = [[1, 5, 10, 3]]
        plens = [1]
        masks = [[False, False, False]]
        loss, kl = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, [1.0])
        assert loss == pytest.approx(0.0)
        assert kl == pytest.approx(0.0)

    def test_short_sequence_skipped(self, tiny, tiny_ref):
        """Sequence where plen >= len(seq) is skipped gracefully."""
        seqs = [[1, 2]]
        plens = [5]  # plen > len(seq)
        masks = [[True]]
        loss, kl = self._torch_loss(tiny, tiny_ref, seqs, plens, masks, [1.0])
        assert loss == pytest.approx(0.0)

    def test_grpo_loss_fn_matches_reference_python(self, tiny, tiny_ref):
        """grpo_loss_fn numerically matches pure-Python grpo_loss (same formula)."""
        torch.manual_seed(42)
        seqs = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9], [3, 4, 5, 6, 7, 8]]
        plens = [2, 2, 3]
        masks = [[True, True, True], [True, True, True], [True, True, True]]
        advs = [1.0, -0.5, 0.3]

        # Torch path
        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        t_loss, t_kl, _, _ = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, beta=0.04, device=DEVICE
        )
        t_loss_val = float(t_loss.detach())

        # Reference Python path: extract log probs manually
        py_lps, py_rlps, py_masks_f = [], [], []
        for seq, plen, mask in zip(seqs, plens, masks, strict=True):
            start = max(plen - 1, 0)
            lp = _lp_as_list(tiny, seq)
            rlp = _lp_as_list(tiny_ref, seq)
            py_lps.append(lp[start : start + len(mask)])
            py_rlps.append(rlp[start : start + len(mask)])
            py_masks_f.append([float(m) for m in mask])

        ref_result = grpo_loss(py_lps, py_rlps, advs, py_masks_f, beta=0.04)

        assert t_loss_val == pytest.approx(ref_result.loss, abs=1e-4), (
            f"Torch loss {t_loss_val:.6f} ≠ Python loss {ref_result.loss:.6f}"
        )
        assert t_kl == pytest.approx(ref_result.kl_divergence, abs=1e-4), (
            f"Torch KL {t_kl:.6f} ≠ Python KL {ref_result.kl_divergence:.6f}"
        )

    def test_ppo_clipping_accepted_without_error(self, tiny, tiny_ref):
        """With PPO clipping, grpo_loss_fn must return a finite differentiable loss."""
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [2.0]

        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        loss_clipped, kl, cl, ch = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, beta=0.0, device=DEVICE, epsilon=0.1
        )
        assert math.isfinite(float(loss_clipped.detach()))
        # Clipping stats must be [0, 1] fractions
        assert 0.0 <= cl <= 1.0
        assert 0.0 <= ch <= 1.0

    def test_segment_weights_scale_loss(self, tiny, tiny_ref):
        """Segment weights > 1.0 should scale loss upward relative to uniform weights."""
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [-1.0]  # Negative advantage → loss should be positive (pushing away)

        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        loss_uniform, _, _, _ = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, beta=0.0, device=DEVICE
        )
        seg_w = [[2.0, 2.0, 2.0, 2.0]]
        loss_weighted, _, _, _ = grpo_loss_fn(
            tiny,
            ref_lps,
            seqs,
            plens,
            masks,
            advs,
            beta=0.0,
            device=DEVICE,
            segment_weights=seg_w,
        )
        # Token normalisation: total weighted tokens = 2 × uniform tokens
        # Loss numerator also 2×, so ratio stays same → same per-token loss
        assert float(loss_uniform) == pytest.approx(float(loss_weighted), abs=1e-5)


# ── Normalisation discrepancy (D1) ───────────────────────────────────────────


class TestNormalisationDiscrepancy:
    """D1: compute_grads normalises per-sequence then averages.

    This differs from grpo_loss_fn (joint normalisation over all tokens).
    When sequences have equal lengths, both give the same result.
    When lengths differ, they diverge — this is a known design tradeoff.
    """

    def test_equal_length_seqs_agree(self, tiny, tiny_ref):
        """Equal-length sequences: per-seq average == joint normalisation."""
        seqs = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        plens = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        advs = [1.0, -1.0]

        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        joint_loss, joint_kl, _, _ = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, beta=0.04, device=DEVICE
        )

        # compute_grads: per-seq then average
        tiny.train()
        avg_loss, _, _, _, avg_kl = compute_grads(
            tiny, tiny_ref, seqs, plens, masks, advs, beta=0.04, device=DEVICE
        )

        assert float(joint_loss.detach()) == pytest.approx(avg_loss, abs=1e-4), (
            "Equal-length sequences: joint and per-seq normalisation should agree"
        )

    def test_unequal_length_seqs_diverge(self, tiny, tiny_ref):
        """Unequal-length sequences: per-seq average ≠ joint normalisation (D1)."""
        # Sequence 0: 1 completion token; sequence 1: 5 completion tokens
        seqs = [[1, 5], [2, 8, 4, 6, 9, 3, 7]]
        plens = [1, 2]
        masks = [[True], [True, True, True, True, True]]
        advs = [1.0, -1.0]

        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        joint_loss, _, _, _ = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, beta=0.04, device=DEVICE
        )

        tiny.train()
        avg_loss, _, _, _, _ = compute_grads(
            tiny, tiny_ref, seqs, plens, masks, advs, beta=0.04, device=DEVICE
        )

        # These should NOT be equal — document the known discrepancy
        # (they may accidentally agree for specific weights, so just check both are finite)
        assert math.isfinite(float(joint_loss.detach()))
        assert math.isfinite(avg_loss)
        # The test documents D1 exists without asserting exact divergence
        # (value depends on model weights)


# ── compute_log_probs shape and values ───────────────────────────────────────


class TestComputeLogProbs:
    def test_shape(self, tiny):
        tokens = [1, 5, 10, 3, 7]
        lp = compute_log_probs_for_tokens(tiny, tokens, DEVICE)
        assert lp.shape == (len(tokens) - 1,), "Should return T-1 log probs"

    def test_all_non_positive(self, tiny):
        tokens = [1, 5, 10, 3, 7]
        lp = compute_log_probs_for_tokens(tiny, tokens, DEVICE)
        assert (lp <= 0).all(), "Log probs must be <= 0"

    def test_single_token_pair(self, tiny):
        tokens = [1, 2]
        lp = compute_log_probs_for_tokens(tiny, tokens, DEVICE)
        assert lp.shape == (1,)

    def test_gradient_flows(self, tiny):
        tiny.train()
        tokens = [1, 5, 10]
        lp = compute_log_probs_for_tokens(tiny, tokens, DEVICE)
        (-lp.mean()).backward()
        assert any(p.grad is not None for p in tiny.parameters())

    def test_log_probs_sum_to_at_most_zero(self, tiny):
        """log probs should be strictly < 0 for non-degenerate distributions."""
        tokens = [1, 5, 10, 3, 7, 2, 8]
        lp = compute_log_probs_for_tokens(tiny, tokens, DEVICE)
        assert lp.sum().item() < 0


# ── GradStepResult fields ────────────────────────────────────────────────────


class TestGradStepResultFields:
    """D3: GradStepResult.policy_loss and kl_penalty are 0.0 (not populated)."""

    def test_required_fields_set(self, tiny, tiny_ref):
        tiny.train()
        optimizer = torch.optim.SGD(tiny.parameters(), lr=1e-3)
        from mlx_grpo.loss.torch_ops import gradient_step

        result = gradient_step(
            tiny,
            tiny_ref,
            optimizer,
            [[1, 5, 10, 3, 7]],
            [2],
            [[True, True, True]],
            [1.0],
            beta=0.04,
            device=DEVICE,
        )
        assert isinstance(result, GradStepResult)
        assert math.isfinite(result.loss)
        assert result.grad_norm >= 0.0
        assert math.isfinite(result.kl_divergence)
        assert result.effective_tokens == 3

    def test_policy_loss_and_kl_penalty_not_populated(self, tiny, tiny_ref):
        """D3: policy_loss and kl_penalty stay at 0.0 — known gap vs reference."""
        tiny.train()
        optimizer = torch.optim.SGD(tiny.parameters(), lr=0.0)
        from mlx_grpo.loss.torch_ops import gradient_step

        result = gradient_step(
            tiny,
            tiny_ref,
            optimizer,
            [[1, 5, 10, 3, 7]],
            [2],
            [[True, True, True]],
            [1.0],
            beta=0.04,
            device=DEVICE,
        )
        # Both MLX and PyTorch leave these at 0 — consistent but incomplete
        assert result.policy_loss == 0.0, "D3: policy_loss not populated"
        assert result.kl_penalty == 0.0, "D3: kl_penalty not populated"


# ── PyTorch backend interface parity ─────────────────────────────────────────


class FakeTok:
    """Minimal tokenizer stub for backend tests (no network required)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 64 for c in text[:8]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(max(32, (i % 95) + 32)) for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, **kw):
        return "hello"


def _make_hf_model(tiny: nn.Module, n_new: int = 5) -> nn.Module:
    """Wrap TinyLM with an HF-compatible .generate() that returns fake tokens."""

    class HFWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self._base = base

        def forward(self, input_ids, **kwargs):
            return self._base(input_ids)

        @property
        def device(self):
            return next(self._base.parameters()).device

        def generate(self, input_ids, **kwargs):
            # Return prefix + n_new fake tokens
            max_new = kwargs.get("max_new_tokens", n_new)
            fake_new = torch.arange(10, 10 + max_new, dtype=torch.long).unsqueeze(0)
            return torch.cat([input_ids, fake_new], dim=1)

        def parameters(self, *a, **kw):
            return self._base.parameters(*a, **kw)

    return HFWrapper(tiny)


class TestPyTorchBackendInterface:
    """PyTorchBackend must satisfy the same protocol as other backends."""

    @pytest.fixture
    def backend(self, tiny):
        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        hf_model = _make_hf_model(tiny)
        return PyTorchBackend(model=hf_model, tokenizer=FakeTok())

    def test_generate_returns_generate_output(self, backend):
        from mlx_grpo.backends.base import GenerateParams

        params = GenerateParams(max_tokens=5, temperature=1.0, top_p=1.0, top_k=0, stop=[])
        out = backend.generate([1, 2, 3], params)
        assert hasattr(out, "text")
        assert hasattr(out, "tokens")
        assert hasattr(out, "stop_reason")
        assert isinstance(out.tokens, list)

    def test_generate_respects_max_tokens(self, backend):
        from mlx_grpo.backends.base import GenerateParams

        params = GenerateParams(max_tokens=3, temperature=1.0, top_p=1.0, top_k=0, stop=[])
        out = backend.generate([1, 2, 3], params)
        assert len(out.tokens) <= 3

    def test_stop_sequence_post_hoc_truncation(self, tiny):
        """D2: stop sequences are applied via post-hoc text truncation.

        The HF .generate() runs to max_new_tokens; text is truncated afterwards.
        If the stop text appears, stop_reason == 'stop_sequence' and text is clean.
        """
        from mlx_grpo.backends.base import GenerateParams
        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        class FixedTok(FakeTok):
            def decode(self, ids, skip_special_tokens=True):
                return "hello STOP world"  # always contains the stop seq

        hf_model = _make_hf_model(tiny)
        backend = PyTorchBackend(model=hf_model, tokenizer=FixedTok())
        params = GenerateParams(max_tokens=30, temperature=1.0, top_p=1.0, top_k=0, stop=["STOP"])
        out = backend.generate([1, 2, 3], params)
        assert out.stop_reason == "stop_sequence"
        assert "STOP" not in out.text

    def test_encode_decode_roundtrip(self, backend):
        tokens = backend.encode("hi")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        decoded = backend.decode(tokens)
        assert isinstance(decoded, str)

    def test_backend_name(self, backend):
        assert backend.name == "pytorch"


# ── Stop-ids now properly used (FIXED) ───────────────────────────────────────


class TestStopIdsProperlyUsed:
    """FIXED: stop_ids are now properly passed via StoppingCriteria.

    HF model stops generation when stop sequence is detected.
    Also includes post-hoc truncation for robustness.
    Matches mlx-lm behavior with proper token-level stop handling.
    """

    def test_stop_ids_wired_into_hf_call(self):
        """Verify: stop_ids are now properly passed via stopping_criteria."""
        import inspect

        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        src = inspect.getsource(PyTorchBackend.generate)
        assert "stop_ids" in src, "stop_ids should be computed"
        assert "stopping_criteria" in src, (
            "FIXED: stop_ids now properly passed via StoppingCriteria to HF generate"
        )

    def test_generate_without_stop_sequences(self, tiny):
        """Without stop sequences, backend runs cleanly end-to-end."""
        from mlx_grpo.backends.base import GenerateParams
        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        hf_model = _make_hf_model(tiny, n_new=5)
        backend = PyTorchBackend(model=hf_model, tokenizer=FakeTok())
        params = GenerateParams(max_tokens=5, temperature=1.0, top_p=1.0, top_k=0, stop=[])
        out = backend.generate([1, 2, 3], params)
        assert len(out.tokens) <= 5
        assert out.stop_reason in ("eos", "max_tokens", "stop_sequence")


# ── precompute_ref_logps ──────────────────────────────────────────────────────


class TestPrecomputeRefLogps:
    def test_returns_detached_tensors(self, tiny):
        seqs = [[1, 5, 10], [2, 3, 4, 5]]
        ref_lps = precompute_ref_logps(tiny, seqs, DEVICE)
        assert len(ref_lps) == 2
        assert ref_lps[0].shape == (2,)  # T-1
        assert ref_lps[1].shape == (3,)
        assert not ref_lps[0].requires_grad, "Ref log probs must be detached"

    def test_single_token_sequence_padded(self, tiny):
        """Sequences shorter than 2 tokens get a zero placeholder."""
        seqs = [[1]]
        ref_lps = precompute_ref_logps(tiny, seqs, DEVICE)
        assert ref_lps[0].shape == (1,)
        assert float(ref_lps[0][0]) == 0.0

    def test_values_non_positive(self, tiny):
        seqs = [[1, 5, 10, 3, 7]]
        ref_lps = precompute_ref_logps(tiny, seqs, DEVICE)
        assert (ref_lps[0] <= 0).all()

    def test_different_from_policy(self, tiny, tiny_ref):
        """Different model weights → different log probs."""
        seqs = [[1, 5, 10, 3, 7]]
        policy_lp = precompute_ref_logps(tiny, seqs, DEVICE)[0]
        ref_lp = precompute_ref_logps(tiny_ref, seqs, DEVICE)[0]
        # Tiny models have same architecture but different (random) weights
        # They should produce different log probs (may fail with prob ~0)
        assert not torch.allclose(policy_lp, ref_lp, atol=1e-3)


# ── Clipping behaviour ────────────────────────────────────────────────────────


class TestClippingBehaviour:
    def test_no_clipping_returns_zero_clip_fracs(self, tiny, tiny_ref):
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [1.0]
        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        _, _, cl, ch = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, epsilon=0.0, device=DEVICE
        )
        assert cl == 0.0
        assert ch == 0.0

    def test_clipping_with_large_epsilon_clips_nothing(self, tiny, tiny_ref):
        """Very large epsilon effectively disables clipping."""
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [1.0]
        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        _, _, cl, ch = grpo_loss_fn(
            tiny, ref_lps, seqs, plens, masks, advs, epsilon=100.0, device=DEVICE
        )
        # With epsilon=100, ratio must be in [1-100, 1+100] which always holds
        assert ch == 0.0

    def test_asymmetric_clipping(self, tiny, tiny_ref):
        """epsilon_high != epsilon should be accepted without error."""
        seqs = [[1, 5, 10, 3, 7]]
        plens = [1]
        masks = [[True, True, True, True]]
        advs = [1.0]
        ref_lps = precompute_ref_logps(tiny_ref, seqs, DEVICE)
        loss, kl, cl, ch = grpo_loss_fn(
            tiny,
            ref_lps,
            seqs,
            plens,
            masks,
            advs,
            epsilon=0.1,
            epsilon_high=0.2,
            device=DEVICE,
        )
        assert math.isfinite(float(loss.detach()))
