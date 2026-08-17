"""One result-envelope parser, over the nine envelope shapes (§9.5).

``cursor_agent.parse_output`` and ``claude_code.parse_output`` were 34-line
copies differing in **one comment**, with 0/9 behavioural differences measured
over the envelope shapes below. Both now delegate to
``utils.providers.result_envelope.parse_result_envelope``; this file is the
table that made the extraction safe and keeps the two adapters from diverging
again.

The legacy third copy — ``llm_client.parse_subagent_response``, which uses a
weaker extractor and a different return shape — is deliberately **left alone**:
it is live behind the shipped ``--internal-validation`` flag
(``review_orchestrator_base`` → ``validation.validate_groups`` →
``llm_client.invoke_for_json`` → ``invoke_subagent`` +
``parse_subagent_response``). ``check_cursor_agent_available`` had no caller at
all and is gone; the last test here pins both facts.
"""
import json

import pytest

from utils.providers.result_envelope import parse_result_envelope

# --- The nine envelope shapes ---------------------------------------------
#
# Every branch of the envelope parser, keyed by what the runner put in the
# `{"type": "result", "result": ...}` wrapper both CLIs emit under
# `--output-format json`.
ENVELOPE_SHAPES = [
    (
        "result-json-array",
        json.dumps({"type": "result", "result": '[{"id": 1}]'}),
        {"success": True, "data": [{"id": 1}]},
    ),
    (
        "result-json-object",
        json.dumps({"type": "result", "result": '{"a": 1}'}),
        {"success": True, "data": {"a": 1}},
    ),
    (
        "result-fenced-json",
        json.dumps(
            {
                "type": "result",
                "result": 'Here you go:\n```json\n[{"id": 2}]\n```\ndone',
            }
        ),
        {"success": True, "data": [{"id": 2}]},
    ),
    (
        "result-prose-only",
        json.dumps({"type": "result", "result": "no json here at all"}),
        {
            "success": False,
            "error": "No valid JSON found in text",
            "raw": "no json here at all",
            "data": None,
        },
    ),
    (
        "result-empty-string",
        json.dumps({"type": "result", "result": "   "}),
        {
            "success": False,
            "error": "No valid JSON found in text",
            "raw": "",
            "data": None,
        },
    ),
    (
        "result-already-a-mapping",
        json.dumps({"type": "result", "result": {"already": "parsed"}}),
        {"success": True, "data": {"already": "parsed"}},
    ),
    (
        "result-null",
        json.dumps({"type": "result", "result": None}),
        {"success": False, "error": "Empty result", "data": None},
    ),
    (
        "no-result-key",
        json.dumps({"type": "other", "payload": [1, 2]}),
        {"success": True, "data": {"type": "other", "payload": [1, 2]}},
    ),
    (
        "bare-array",
        '[{"id": 9}]',
        {"success": True, "data": [{"id": 9}]},
    ),
]

# Shapes that are NOT an envelope at all and must fall through to the shared
# text extractor rather than raising.
FALLTHROUGH_SHAPES = [
    (
        "truncated-object",
        '{"type": "result", ',
        {
            "success": False,
            "error": "No valid JSON found in text",
            "raw": '{"type": "result",',
            "data": None,
        },
    ),
    (
        "prose-stdout",
        "plain prose, no json",
        {
            "success": False,
            "error": "No valid JSON found in text",
            "raw": "plain prose, no json",
            "data": None,
        },
    ),
    (
        "result-truncated-array",
        json.dumps({"type": "result", "result": '[{"id": 3}'}),
        {"success": True, "data": {"id": 3}},
    ),
]


def _adapters():
    from utils.provider_registry import _PROVIDERS

    return {name: _PROVIDERS[name] for name in ("cursor-agent", "claude-code")}


class TestSharedEnvelopeParser:
    def test_the_table_really_has_nine_shapes(self):
        assert len(ENVELOPE_SHAPES) == 9
        assert len({shape for shape, _, _ in ENVELOPE_SHAPES}) == 9

    @pytest.mark.parametrize(
        "shape,stdout,expected",
        ENVELOPE_SHAPES,
        ids=[shape for shape, _, _ in ENVELOPE_SHAPES],
    )
    def test_shape(self, shape, stdout, expected):
        assert parse_result_envelope(stdout) == expected

    @pytest.mark.parametrize(
        "shape,stdout,expected",
        FALLTHROUGH_SHAPES,
        ids=[shape for shape, _, _ in FALLTHROUGH_SHAPES],
    )
    def test_non_envelope_stdout_falls_through(self, shape, stdout, expected):
        assert parse_result_envelope(stdout) == expected

    def test_leading_and_trailing_whitespace_is_stripped(self):
        payload = json.dumps({"type": "result", "result": '[{"id": 1}]'})
        assert parse_result_envelope(f"\n  {payload}  \n") == {
            "success": True,
            "data": [{"id": 1}],
        }


class TestBothAdaptersDelegate:
    """0/9 behavioural differences — asserted, not assumed."""

    @pytest.mark.parametrize(
        "shape,stdout,expected",
        ENVELOPE_SHAPES + FALLTHROUGH_SHAPES,
        ids=[shape for shape, _, _ in ENVELOPE_SHAPES + FALLTHROUGH_SHAPES],
    )
    def test_cursor_agent_and_claude_code_agree(self, shape, stdout, expected):
        results = {
            name: provider.parse_output(stdout, "")
            for name, provider in _adapters().items()
        }
        assert results["cursor-agent"] == expected
        assert results["claude-code"] == expected

    def test_neither_adapter_keeps_its_own_copy(self):
        """Both `parse_output` bodies are the shared call, not 34 lines each."""
        import inspect

        from utils.providers.claude_code import ClaudeCodeProvider
        from utils.providers.cursor_agent import CursorAgentProvider

        for cls in (CursorAgentProvider, ClaudeCodeProvider):
            source = inspect.getsource(cls.parse_output)
            assert "parse_result_envelope" in source, cls.__name__
            assert "json.loads" not in source, (
                f"{cls.__name__}.parse_output still carries its own envelope copy"
            )


class TestLegacyChainSurvivesTheCleanup:
    """§9.5's split verdict: delete the dead references, keep the live chain."""

    def test_dead_availability_helper_is_gone(self):
        import utils
        import utils.llm_client as llm_client

        assert not hasattr(llm_client, "check_cursor_agent_available")
        assert not hasattr(utils, "check_cursor_agent_available")
        assert "check_cursor_agent_available" not in utils.__all__

    def test_internal_validation_chain_is_intact(self):
        """`--internal-validation` still resolves end to end."""
        import utils
        import utils.validation as validation
        from utils.llm_client import (
            invoke_for_json,
            invoke_subagent,
            parse_subagent_response,
        )

        assert callable(invoke_for_json)
        assert callable(invoke_subagent)
        assert callable(parse_subagent_response)
        for name in ("invoke_subagent", "parse_subagent_response", "invoke_for_json"):
            assert name in utils.__all__
        # The flag's own leg: validation dispatches through invoke_for_json.
        assert "invoke_for_json" in validation.__dict__

    def test_legacy_parser_still_reads_the_wrapper(self):
        """Weaker extractor, different return shape — unchanged on purpose."""
        from utils.llm_client import parse_subagent_response

        payload = json.dumps({"type": "result", "result": '[{"id": 1}]'})
        assert parse_subagent_response(payload) == [{"id": 1}]

    def test_code_review_orchestrator_dropped_its_dead_import(self):
        import code_review_orchestrator

        assert not hasattr(code_review_orchestrator, "parse_subagent_response")
