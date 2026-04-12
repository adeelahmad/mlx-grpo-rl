"""Tests for TemplateCompiler DSL — parse_template_string, TemplateExecutor, TemplateCompiler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_grpo.engine.compiler import (
    OpCode,
    TemplateCompiler,
    TemplateExecutor,
    TemplateOp,
    _parse_args,
    parse_template_string,
)

# ── _parse_args ────────────────────────────────────────────────────────────────


class TestParseArgs:
    def test_empty(self):
        assert _parse_args("") == {}

    def test_int(self):
        assert _parse_args("max_tokens=512")["max_tokens"] == 512

    def test_float(self):
        assert _parse_args("temperature=0.7")["temperature"] == 0.7

    def test_bool_true(self):
        assert _parse_args("masked=true")["masked"] is True

    def test_bool_false(self):
        assert _parse_args("masked=false")["masked"] is False

    def test_string(self):
        assert _parse_args("tag=thinking")["tag"] == "thinking"

    def test_quoted_string(self):
        assert _parse_args('text="hello world"')["text"] == "hello world"

    def test_multiple_params(self):
        p = _parse_args("max_tokens=256 temperature=0.5 tag=answer")
        assert p["max_tokens"] == 256
        assert p["temperature"] == 0.5
        assert p["tag"] == "answer"

    def test_stop_sequence(self):
        p = _parse_args("stop=</think>")
        assert p["stop"] == "</think>"

    def test_negative_int(self):
        p = _parse_args("count=-1")
        assert p["count"] == -1


# ── parse_template_string ──────────────────────────────────────────────────────


class TestParseTemplateString:
    def test_empty(self):
        assert parse_template_string("") == []

    def test_comment_skipped(self):
        assert parse_template_string("# comment\n# another") == []

    def test_generate_directive(self):
        ops = parse_template_string("@generate max_tokens=128 tag=thinking")
        assert len(ops) == 1
        assert ops[0].op == OpCode.GENERATE
        assert ops[0].params["max_tokens"] == 128
        assert ops[0].params["tag"] == "thinking"

    def test_inject_directive(self):
        ops = parse_template_string('@inject text="hello" masked=true')
        assert ops[0].op == OpCode.INJECT
        assert ops[0].params["text"] == "hello"
        assert ops[0].params["masked"] is True

    def test_scaffold_directive(self):
        ops = parse_template_string("@scaffold ratio=0.5 mode=prefix")
        assert ops[0].op == OpCode.SCAFFOLD

    def test_set_meta_directive(self):
        ops = parse_template_string("@set_meta key=val")
        assert ops[0].op == OpCode.SET_META

    def test_abort_directive(self):
        ops = parse_template_string("@abort")
        assert ops[0].op == OpCode.ABORT

    def test_plain_text_becomes_inject(self):
        ops = parse_template_string("plain text line")
        assert len(ops) == 1
        assert ops[0].op == OpCode.INJECT
        assert ops[0].params["text"] == "plain text line"
        assert ops[0].params["masked"] is False

    def test_if_endif(self):
        template = "@if condition=true\n@generate\n@endif"
        ops = parse_template_string(template)
        assert len(ops) == 1
        assert ops[0].op == OpCode.CONDITIONAL
        assert len(ops[0].children) == 1

    def test_if_else_endif(self):
        template = "@if condition=x>0\n@generate\n@else\n@inject text=fallback\n@endif"
        ops = parse_template_string(template)
        cond = ops[0]
        assert cond.op == OpCode.CONDITIONAL
        assert len(cond.children) == 1
        assert len(cond.else_children) == 1

    def test_for_endfor(self):
        template = "@for count=3\n@generate\n@endfor"
        ops = parse_template_string(template)
        assert len(ops) == 1
        assert ops[0].op == OpCode.LOOP
        assert ops[0].params["count"] == 3
        assert len(ops[0].children) == 1

    def test_generate_until(self):
        ops = parse_template_string("@generate_until endswith=X tag=out")
        assert ops[0].op == OpCode.GENERATE_UNTIL

    def test_generate_if(self):
        ops = parse_template_string("@generate_if max_tokens=128")
        assert ops[0].op == OpCode.GENERATE_IF

    def test_capture_directive(self):
        ops = parse_template_string("@capture name=thinking")
        assert ops[0].op == OpCode.CAPTURE

    def test_multiple_directives_in_sequence(self):
        template = "@generate tag=thinking\n@inject text=sep\n@generate tag=answer"
        ops = parse_template_string(template)
        assert len(ops) == 3

    def test_elif_in_if_block(self):
        template = "@if condition=a>0\n@generate\n@elif condition=b>0\n@inject text=x\n@endif"
        ops = parse_template_string(template)
        assert ops[0].op == OpCode.CONDITIONAL


# ── TemplateExecutor ──────────────────────────────────────────────────────────


class _FakeGenerateResult:
    def __init__(self, text="gen", stop_reason="eos"):
        self.text = text
        self.stop_reason = stop_reason
        self.tokens = list(text.encode())
        self.token_count = len(self.tokens)


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.answer = "answer text"
    ctx.prompt = "prompt"
    ctx._variables = {}
    ctx.batch.progress = 0.5
    ctx.batch.group_index = 0
    ctx.batch.scaffold.mode = "prefix"
    ctx.batch.scaffold.effective_ratio.return_value = 0.5
    ctx.batch.scaffold.max_scaffolded_tokens = None
    gen_result = _FakeGenerateResult()
    ctx.generate.return_value = gen_result
    ctx.generate_until.return_value = gen_result
    ctx.generate_if.return_value = gen_result
    ctx.encode.return_value = [1, 2, 3, 4, 5]
    ctx.decode.return_value = "decoded"
    ctx.callbacks.slice_answer_tokens.return_value = [1, 2]
    return ctx


class TestTemplateExecutorBasic:
    def _exec(self):
        return TemplateExecutor()

    def test_generate_calls_ctx(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.GENERATE, params={"max_tokens": 128, "tag": "t"})]
        e.execute(ops, ctx)
        ctx.generate.assert_called_once()

    def test_inject_unescape_newline(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.INJECT, params={"text": "a\\nb", "masked": True})]
        e.execute(ops, ctx)
        ctx.inject.assert_called_once_with("a\nb", masked=True, tag="")

    def test_inject_unescape_tab(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.INJECT, params={"text": "a\\tb", "masked": False})]
        e.execute(ops, ctx)
        ctx.inject.assert_called_once_with("a\tb", masked=False, tag="")

    def test_set_meta_calls_ctx(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.SET_META, params={"key": "value", "flag": True})]
        e.execute(ops, ctx)
        assert ctx.set_meta.call_count == 2

    def test_abort_stops_execution(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [
            TemplateOp(op=OpCode.ABORT),
            TemplateOp(op=OpCode.GENERATE, params={}),
        ]
        e.execute(ops, ctx)
        ctx.generate.assert_not_called()

    def test_conditional_true(self):
        e = self._exec()
        ctx = _make_ctx()
        child = TemplateOp(op=OpCode.INJECT, params={"text": "yes", "masked": False})
        ops = [TemplateOp(op=OpCode.CONDITIONAL, params={"condition": "true"}, children=[child])]
        e.execute(ops, ctx)
        ctx.inject.assert_called_once()

    def test_conditional_false_runs_else(self):
        e = self._exec()
        ctx = _make_ctx()
        else_child = TemplateOp(op=OpCode.INJECT, params={"text": "else_branch", "masked": False})
        ops = [
            TemplateOp(
                op=OpCode.CONDITIONAL,
                params={"condition": "false"},
                else_children=[else_child],
            )
        ]
        e.execute(ops, ctx)
        ctx.inject.assert_called_once_with("else_branch", masked=False, tag="")

    def test_loop_runs_count_times(self):
        e = self._exec()
        ctx = _make_ctx()
        child = TemplateOp(op=OpCode.INJECT, params={"text": "x", "masked": False})
        ops = [TemplateOp(op=OpCode.LOOP, params={"count": 3}, children=[child])]
        e.execute(ops, ctx)
        assert ctx.inject.call_count == 3

    def test_capture_noop(self):
        """CAPTURE op is a no-op by itself (handled via 'as' param in generate)."""
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.CAPTURE, params={"name": "thinking"})]
        e.execute(ops, ctx)  # Should not raise

    def test_generate_with_as_calls_capture(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.GENERATE, params={"max_tokens": 128, "as": "thinking"})]
        e.execute(ops, ctx)
        ctx.capture.assert_called_once()


class TestEvalCondition:
    def _exec(self):
        return TemplateExecutor()

    def test_true_literal(self):
        e = self._exec()
        ctx = _make_ctx()
        assert e._eval_condition("true", ctx) is True

    def test_false_literal(self):
        e = self._exec()
        ctx = _make_ctx()
        assert e._eval_condition("false", ctx) is False

    def test_yes_literal(self):
        assert self._exec()._eval_condition("yes", _make_ctx()) is True

    def test_no_literal(self):
        assert self._exec()._eval_condition("no", _make_ctx()) is False

    def test_numeric_one(self):
        assert self._exec()._eval_condition("1", _make_ctx()) is True

    def test_eval_expr(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.batch.progress = 0.5
        assert e._eval_condition("progress > 0.3", ctx) is True

    def test_eval_expr_false(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.batch.progress = 0.5
        assert e._eval_condition("progress > 0.9", ctx) is False

    def test_eval_exception_returns_false(self):
        e = self._exec()
        ctx = _make_ctx()
        assert e._eval_condition("undefined_var == 'x'", ctx) is False


# ── TemplateCompiler ──────────────────────────────────────────────────────────


class TestTemplateCompiler:
    def test_compile_returns_callable(self):
        c = TemplateCompiler()
        fn = c.compile("@generate max_tokens=128")
        assert callable(fn)

    def test_compile_caches(self):
        c = TemplateCompiler()
        fn1 = c.compile("@generate")
        fn2 = c.compile("@generate")
        assert fn1 is not fn2  # different closures but same ops list
        assert len(c._cache) == 1

    def test_clear_cache(self):
        c = TemplateCompiler()
        c.compile("@generate")
        c.clear_cache()
        assert len(c._cache) == 0

    def test_parse_only(self):
        c = TemplateCompiler()
        ops = c.parse_only("@generate tag=t")
        assert len(ops) == 1
        assert ops[0].op == OpCode.GENERATE

    def test_compile_executes(self):
        c = TemplateCompiler()
        fn = c.compile('@inject text="test" masked=true')
        ctx = _make_ctx()
        fn(ctx)
        ctx.inject.assert_called_once_with("test", masked=True, tag="")

    def test_compile_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write("@generate max_tokens=100 tag=thinking\n")
            fname = f.name
        c = TemplateCompiler()
        fn = c.compile_file(fname)
        assert callable(fn)
        ctx = _make_ctx()
        fn(ctx)
        ctx.generate.assert_called_once()


# ── TemplateExecutor: generate_until, generate_if, scaffold ──────────────────


class TestTemplateExecutorExtended:
    def _exec(self):
        return TemplateExecutor()

    def test_execute_generate_until(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [
            TemplateOp(
                op=OpCode.GENERATE_UNTIL,
                params={"endswith": "</think>", "max_tokens": 256, "tag": "thinking"},
            )
        ]
        e.execute(ops, ctx)
        ctx.generate_until.assert_called_once()
        call_kw = ctx.generate_until.call_args[1]
        assert call_kw["endswith"] == "</think>"
        assert call_kw["max_tokens"] == 256

    def test_execute_generate_until_with_as(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.GENERATE_UNTIL, params={"endswith": "X", "as": "thinking"})]
        e.execute(ops, ctx)
        ctx.generate_until.assert_called_once()
        ctx.capture.assert_called_once()

    def test_execute_generate_if(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [
            TemplateOp(
                op=OpCode.GENERATE_IF, params={"max_retries": 3, "max_tokens": 128, "tag": "answer"}
            )
        ]
        e.execute(ops, ctx)
        ctx.generate_if.assert_called_once()
        call_kw = ctx.generate_if.call_args[1]
        assert call_kw["max_tokens"] == 128
        assert call_kw["max_retries"] == 3

    def test_execute_generate_if_with_as(self):
        e = self._exec()
        ctx = _make_ctx()
        ops = [TemplateOp(op=OpCode.GENERATE_IF, params={"max_tokens": 64, "as": "my_answer"})]
        e.execute(ops, ctx)
        ctx.generate_if.assert_called_once()
        ctx.capture.assert_called_once()

    def test_execute_generate_if_with_validator(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.callbacks.is_valid_xml = lambda text: True
        ops = [
            TemplateOp(
                op=OpCode.GENERATE_IF, params={"validator": "is_valid_xml", "max_tokens": 64}
            )
        ]
        e.execute(ops, ctx)
        ctx.generate_if.assert_called_once()
        # validator was passed
        validator = ctx.generate_if.call_args[1]["validator"]
        assert callable(validator)

    def test_execute_scaffold_no_answer(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.answer = None
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"ratio": 0.5})]
        e.execute(ops, ctx)
        ctx.inject.assert_not_called()

    def test_execute_scaffold_with_answer(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.encode.return_value = [1, 2, 3, 4, 5]
        ctx.callbacks.slice_answer_tokens.return_value = [1, 2]
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"ratio": 0.5, "mode": "prefix"})]
        e.execute(ops, ctx)
        ctx.callbacks.slice_answer_tokens.assert_called()
        ctx.inject.assert_called_once()

    def test_execute_scaffold_zero_budget(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.encode.return_value = [1, 2, 3]
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"ratio": 0.0})]
        e.execute(ops, ctx)
        ctx.inject.assert_not_called()

    def test_execute_scaffold_max_tokens_override(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.encode.return_value = [1, 2, 3, 4, 5]
        ctx.batch.scaffold.effective_ratio.return_value = 0.5
        ctx.callbacks.slice_answer_tokens.return_value = [1, 2]
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"max_tokens": 200, "tag": "scaffold"})]
        e.execute(ops, ctx)
        ctx.batch.scaffold.effective_ratio.assert_called()

    def test_execute_scaffold_max_scaffolded_tokens(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.batch.scaffold.max_scaffolded_tokens = 100
        ctx.encode.return_value = list(range(20))
        ctx.callbacks.slice_answer_tokens.return_value = [1, 2, 3]
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"ratio": 0.5})]
        e.execute(ops, ctx)
        ctx.callbacks.slice_answer_tokens.assert_called()

    def test_execute_scaffold_string_ratio(self):
        """ratio can be a string expression evaluated at runtime."""
        e = self._exec()
        ctx = _make_ctx()
        ctx.encode.return_value = list(range(10))
        ctx.callbacks.slice_answer_tokens.return_value = [1]
        ops = [TemplateOp(op=OpCode.SCAFFOLD, params={"ratio": "0.5"})]
        e.execute(ops, ctx)
        # ratio was evaluated as float 0.5

    def test_eval_ratio(self):
        e = self._exec()
        ctx = _make_ctx()
        ctx.batch.progress = 0.5
        result = e._eval_ratio("progress", ctx)
        assert result == pytest.approx(0.5)

    def test_eval_ratio_bad_expr(self):
        e = self._exec()
        ctx = _make_ctx()
        result = e._eval_ratio("@#$invalid", ctx)
        assert result == pytest.approx(0.0)

    def test_build_namespace_includes_captured_vars(self):
        e = self._exec()
        ctx = _make_ctx()
        fake_result = MagicMock()
        fake_result.stop_reason = "stop_sequence"
        ctx._variables = {"thinking": fake_result}
        ns = e._build_namespace(ctx)
        assert "thinking" in ns
        assert ns["thinking"].stop_reason == "stop_sequence"

    def test_condition_uses_captured_var(self):
        e = self._exec()
        ctx = _make_ctx()
        fake_result = MagicMock()
        fake_result.stop_reason = "max_tokens"
        ctx._variables = {"thinking": fake_result}
        result = e._eval_condition('thinking.stop_reason == "max_tokens"', ctx)
        assert result is True
