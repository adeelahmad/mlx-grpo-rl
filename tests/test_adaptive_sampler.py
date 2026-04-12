"""Tests for AdaptiveSampler."""

from mlx_grpo.data.adaptive_sampler import AdaptiveSampler


class FakeDataset:
    def __init__(self, n=20):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"prompt": f"q{idx}", "answer": f"a{idx}"}


def test_initial_all_active():
    ds = FakeDataset(10)
    s = AdaptiveSampler(ds)
    assert s.pool_sizes == {"active": 10, "hard": 0, "easy": 0}


def test_demote_to_hard_after_zeros():
    ds = FakeDataset(10)
    s = AdaptiveSampler(ds, zero_threshold=3)
    for _ in range(3):
        s.update(0, 0.0)
    assert s.get_stats(0).pool == "hard"
    assert 0 not in s._active
    assert 0 in s._hard


def test_demote_to_easy_after_perfects():
    ds = FakeDataset(10)
    s = AdaptiveSampler(ds, perfect_threshold=3)
    for _ in range(3):
        s.update(0, 1.0)
    assert s.get_stats(0).pool == "easy"


def test_promote_from_hard_on_success():
    ds = FakeDataset(10)
    s = AdaptiveSampler(ds, zero_threshold=2)
    s.update(0, 0.0)
    s.update(0, 0.0)  # Demoted to hard
    assert s.get_stats(0).pool == "hard"
    s.update(0, 0.5)  # Non-zero -> promote back
    assert s.get_stats(0).pool == "active"


def test_next_returns_valid_sample():
    ds = FakeDataset(5)
    s = AdaptiveSampler(ds, seed=42)
    idx, sample = s.next()
    assert 0 <= idx < 5
    assert "prompt" in sample


def test_summary():
    ds = FakeDataset(10)
    s = AdaptiveSampler(ds)
    summary = s.summary()
    assert summary["active"] == 10
    assert summary["hard"] == 0
