"""One result-envelope parser, over the nine envelope shapes (§9.5).

``cursor_agent.parse_output`` and ``claude_code.parse_output`` were 34-line
copies differing in **one comment**, with 0/9 behavioural differences measured
over the envelope shapes below. Both now delegate to
``utils.providers.result_envelope.parse_result_envelope``; this file is the
table that made the extraction safe and keeps the two adapters from diverging
again.

The legacy third copy, ``llm_client.parse_subagent_response``, served only the
in-process validation lane and was retired with it, as was
``check_cursor_agent_available``; the last class here pins that both stay gone.
"""
import json

import pytest

from utils.providers.result_envelope import parse_result_envelope
from tests.source_cache import source_tree

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
        """Both `parse_output` bodies are the shared call, not 34 lines each.

        **Stated over the parsed body, not over the source text** (C5). The
        earlier form asserted that the token ``json.loads`` appeared nowhere in
        the function *as text*, docstring included — so a docstring explaining
        what the shared parser does with the envelope failed a function whose
        body is one delegating call. Reading a function's source by text also
        means reading it through ``inspect.getsource``, which slices the file by
        the loaded code object's line numbers: edit the module while a session
        has it imported and the assertion reports on whatever block now sits at
        those lines. The property is structural, so assert it structurally.

        The property is unchanged and unweakened: each ``parse_output`` body is
        exactly ``return parse_result_envelope(stdout)``, which is stricter than
        "contains the call and no ``json.loads``" — a second copy of the unwrap
        cannot hide in it in any spelling.

        Scope is ``parse_output`` and nothing else, deliberately. C5 gave both
        adapters a ``classify_failure`` override reading the same envelope's
        ``api_error_status``, and ``claude_code.parse_usage`` has always read its
        ``usage`` block. Those are different functions reading different fields
        of the same bytes; neither is a second copy of the unwrap this test
        exists to forbid.
        """
        import ast
        import inspect

        from utils.providers.claude_code import ClaudeCodeProvider
        from utils.providers.cursor_agent import CursorAgentProvider

        for cls in (CursorAgentProvider, ClaudeCodeProvider):
            module = source_tree(inspect.getfile(cls))
            functions = [
                node for node in ast.walk(module)
                if isinstance(node, ast.FunctionDef) and node.name == "parse_output"
            ]
            assert len(functions) == 1, f"{cls.__name__}: expected one parse_output"

            body = [
                node for node in functions[0].body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            assert len(body) == 1 and isinstance(body[0], ast.Return), (
                f"{cls.__name__}.parse_output is no longer a single delegating "
                f"return — it has {len(body)} statements, which is where a "
                f"second copy of the envelope unwrap would live"
            )
            call = body[0].value
            assert (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "parse_result_envelope"
            ), f"{cls.__name__}.parse_output does not delegate to the shared parser"


class TestTheLegacyLaneIsGone:
    """The in-process validation lane and its helpers are retired (PROV-10)."""

    #: Every name that lane owned, and the module that used to bind it.
    RETIRED = {
        "utils.llm_client": (
            "check_cursor_agent_available", "invoke_subagent",
            "parse_subagent_response", "_extract_json_from_text",
            "invoke_for_json", "LLMClientError", "SubagentTimeoutError",
        ),
        "utils.validation": (
            "validate_groups", "revalidate_failed_items",
            "_classify_validation_error", "VALIDATION_PROMPT_TEMPLATE",
        ),
        "utils.review_orchestrator_base": (
            "internal_validation_with_scope", "validate_groups",
        ),
    }

    @pytest.mark.parametrize("module_name", sorted(RETIRED))
    def test_no_module_binds_a_retired_name(self, module_name):
        import importlib

        module = importlib.import_module(module_name)
        bound = [n for n in self.RETIRED[module_name] if hasattr(module, n)]
        assert not bound, f"{module_name} still binds {bound}"

    def test_the_package_exports_none_of_them(self):
        import utils

        names = {n for names in self.RETIRED.values() for n in names}
        assert not names & set(utils.__all__)
        assert not [n for n in names if hasattr(utils, n)]

    def test_code_review_orchestrator_dropped_its_dead_imports(self):
        import code_review_orchestrator

        assert not hasattr(code_review_orchestrator, "parse_subagent_response")
        assert not hasattr(code_review_orchestrator, "validate_groups")
