"""Tests for MLX batch_generate backend."""

from unittest.mock import MagicMock

import pytest

from mlx_grpo.backends import BackendRegistry, GenerateParams
from mlx_grpo.backends.mlx_batch_generate_backend import MLXBatchGenerateBackend


class TestRegistration:
    """Test that backend registers properly."""

    def test_registered_name(self):
        assert "mlx_batch_generate" in BackendRegistry.available()

    def test_instantiate_via_registry(self):
        backend = BackendRegistry.get(
            "mlx_batch_generate", model=MagicMock(), tokenizer=MagicMock()
        )
        assert isinstance(backend, MLXBatchGenerateBackend)

    def test_tokenizer_required_at_usage(self):
        backend = BackendRegistry.get("mlx_batch_generate", model=MagicMock())
        with pytest.raises(RuntimeError, match="requires tokenizer"):
            backend.encode("test")

    def test_existing_mlx_backend_still_available(self):
        assert "mlx" in BackendRegistry.available()


class TestInterface:
    """Test protocol compliance (encode/decode/chat_template)."""

    def setup_method(self):
        self.tok = MagicMock()
        self.tok.decode = MagicMock(return_value="hello")
        self.tok.encode = MagicMock(return_value=[1, 2, 3])
        self.tok.apply_chat_template = MagicMock(return_value="[CHAT]")
        self.backend = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=self.tok)

    def test_name(self):
        assert self.backend.name == "mlx_batch_generate"

    def test_eos_token_id(self):
        self.tok.eos_token_id = 0
        assert self.backend.eos_token_id == 0

    def test_eos_token_id_missing(self):
        del self.tok.eos_token_id
        assert self.backend.eos_token_id is None

    def test_encode(self):
        assert self.backend.encode("hi") == [1, 2, 3]
        self.tok.encode.assert_called_once_with("hi", add_special_tokens=False)

    def test_decode(self):
        assert self.backend.decode([1, 2]) == "hello"
        self.tok.decode.assert_called_once_with([1, 2])

    def test_apply_chat_template(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.backend.apply_chat_template(msgs)
        assert result == [1, 2, 3]

    def test_verbose_default(self):
        assert self.backend._verbose is False

    def test_verbose_flag(self):
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=self.tok, verbose=True)
        assert b._verbose is True


class TestTokenizeStops:
    """Test _tokenize_stops helper."""

    def test_eos_tokens_included(self):
        tok = MagicMock()
        tok.eos_token_ids = [0, 2]
        tok.encode = MagicMock(return_value=[])
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        stops = b._tokenize_stops(GenerateParams(stop=[]))
        assert [0] in stops
        assert [2] in stops

    def test_eos_token_id_fallback(self):
        tok = MagicMock(spec=["eos_token_id", "encode"])
        tok.eos_token_id = 42
        tok.encode = MagicMock(return_value=[])
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        stops = b._tokenize_stops(GenerateParams(stop=[]))
        assert [42] in stops

    def test_string_stops_tokenized(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(side_effect=lambda s, **kw: [10, 11] if s == "</think>" else [])
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        stops = b._tokenize_stops(GenerateParams(stop=["</think>"]))
        assert [10, 11] in stops

    def test_empty_tokenization_skipped(self):
        tok = MagicMock(spec=["eos_token_ids", "encode"])
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        stops = b._tokenize_stops(GenerateParams(stop=["<|"]))
        assert stops == []

    def test_tokenize_error_skipped(self):
        tok = MagicMock(spec=["eos_token_ids", "encode"])
        tok.eos_token_ids = []
        tok.encode = MagicMock(side_effect=Exception("bad"))
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        stops = b._tokenize_stops(GenerateParams(stop=["broken"]))
        assert stops == []


def _make_backend_with_mock_gen(tok, responses):
    """Create backend with mocked _make_generator and _clear_cache."""
    b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)

    mock_gen = MagicMock()
    uids = list(range(len({r.uid for r in responses})))
    mock_gen.insert.return_value = uids
    mock_gen.next_generated.side_effect = [responses, []]

    b._make_generator = MagicMock(return_value=mock_gen)
    b._clear_cache = MagicMock()
    return b


class TestGenerateBatch:
    """Test generate_batch with mocked BatchGenerator."""

    def test_empty_batch(self):
        tok = MagicMock()
        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        assert b.generate_batch([], GenerateParams()) == []

    def test_basic_generation(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(return_value="hello")

        r1 = MagicMock(uid=0, token=10, finish_reason=None, match_sequence=None)
        r2 = MagicMock(uid=0, token=11, finish_reason="length", match_sequence=None)

        b = _make_backend_with_mock_gen(tok, [r1, r2])
        results = b.generate_batch([[1, 2]], GenerateParams())

        assert len(results) == 1
        assert results[0].stop_reason == "max_tokens"
        assert results[0].tokens == [10, 11]

    def test_stop_sequence_detected(self):
        tok = MagicMock()
        tok.eos_token_ids = [0]
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(
            side_effect=lambda t: "</think>" if tuple(t) == (20, 21) else "thinking"
        )

        r1 = MagicMock(uid=0, token=10, finish_reason=None, match_sequence=None)
        r2 = MagicMock(uid=0, token=20, finish_reason="stop", match_sequence=(20, 21))

        b = _make_backend_with_mock_gen(tok, [r1, r2])
        results = b.generate_batch([[1]], GenerateParams(stop=["</think>"]))

        assert results[0].stop_reason == "stop_sequence"

    def test_eos_distinguished_from_custom_stop(self):
        tok = MagicMock()
        tok.eos_token_ids = [0]
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(return_value="text")

        r1 = MagicMock(uid=0, token=5, finish_reason=None, match_sequence=None)
        r2 = MagicMock(uid=0, token=0, finish_reason="stop", match_sequence=(0,))

        b = _make_backend_with_mock_gen(tok, [r1, r2])
        results = b.generate_batch([[1]], GenerateParams())

        assert results[0].stop_reason == "eos"

    def test_string_fallback_catches_untokenizable_stop(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(return_value="hello<|im_end|>world")

        r1 = MagicMock(uid=0, token=5, finish_reason="length", match_sequence=None)

        b = _make_backend_with_mock_gen(tok, [r1])
        results = b.generate_batch([[1]], GenerateParams(stop=["<|"]))

        assert results[0].stop_reason == "stop_sequence"
        assert results[0].stop_sequence == "<|"
        assert results[0].text == "hello"

    def test_generate_delegates_to_batch(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(return_value="answer")

        r1 = MagicMock(uid=0, token=10, finish_reason="length", match_sequence=None)

        b = _make_backend_with_mock_gen(tok, [r1])
        result = b.generate([1, 2, 3], GenerateParams())

        assert result.text == "answer"
        assert result.tokens == [10]

    def test_clear_cache_called(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        tok.decode = MagicMock(return_value="text")

        r1 = MagicMock(uid=0, token=5, finish_reason="length", match_sequence=None)

        b = _make_backend_with_mock_gen(tok, [r1])
        b.generate_batch([[1]], GenerateParams())

        b._clear_cache.assert_called_once()

    def test_multi_sequence_batch(self):
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.encode = MagicMock(return_value=[])
        decode_map = {0: "ans_0", 1: "ans_1"}
        call_count = [0]

        def mock_decode(tokens):
            idx = call_count[0]
            call_count[0] += 1
            return decode_map.get(idx, "")

        tok.decode = MagicMock(side_effect=mock_decode)

        r0 = MagicMock(uid=0, token=10, finish_reason="length", match_sequence=None)
        r1 = MagicMock(uid=1, token=20, finish_reason="length", match_sequence=None)

        b = MLXBatchGenerateBackend(model=MagicMock(), tokenizer=tok)
        mock_gen = MagicMock()
        mock_gen.insert.return_value = [0, 1]
        mock_gen.next_generated.side_effect = [[r0, r1], []]
        b._make_generator = MagicMock(return_value=mock_gen)
        b._clear_cache = MagicMock()

        results = b.generate_batch([[1], [2]], GenerateParams())

        assert len(results) == 2
        assert results[0].text == "ans_0"
        assert results[1].text == "ans_1"
