"""M7 — native structured output (§4.2 / §4.2.1 / §4.2.2).

Four groups, one per clause of §14.2's M7 row:

* :class:`TestAcceptanceCases` — §4.2.1's **six** cases, including the one that
  makes the whole item worth shipping: ``exit 0`` with no ``structured_output``
  is **not** a success. The claude envelopes in ``ENVELOPE_*`` below are not
  hand-written; they were captured on 2026-08-16 from the real ``claude``
  2.1.233 binary driven against a localhost Anthropic stub (zero paid tokens),
  which is why the ``structured_output`` key is *absent* rather than ``null`` on
  the two failure shapes.
* :class:`TestClaudeJsonSchemaGate` — §5.4.1's optional-flag gate: a help text
  without ``--json-schema`` emits no flag, warns **once**, and leaves the run on
  §4.2.1's recovery branch (``recovered``), never ``native``.
* :class:`TestSchemaKeywordIsInertWhenUnset` — the off switch (§14.4).
  ``schema=None`` renders the same bytes ``timeout_s=None`` does, over all 13
  adapters, reusing L5's frozen ``PRE_L5_ARGV`` table so "byte-identical to
  today" is asserted against a capture rather than against itself.
* :class:`TestSplicePosition` — §4.2.2's normative splice point and the §9.8/L8
  trap: the flag sits in the policy region, never adjacent to the prompt and
  never adjacent to a variadic tool flag.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import structured_output as so  # noqa: E402
from utils.providers.base import (  # noqa: E402
    LLMProvider,
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    StructuredOutputUnavailable,
    StructuredSchema,
)
from utils.providers.claude_code import (  # noqa: E402
    JSON_SCHEMA_FLAG,
    ClaudeCodeProvider,
    reset_flag_gate_cache,
)
from utils.providers.codex import CodexProvider, OUTPUT_SCHEMA_FLAG  # noqa: E402


# --------------------------------------------------------------------------- #
# Captured claude 2.1.233 envelopes (localhost stub, 0 paid tokens, 2026-08-16)
# --------------------------------------------------------------------------- #

#: Case 1. The tool was called with conforming input: `structured_output` holds
#: the object and `result` holds it serialized (so `parse_output` is unchanged).
ENVELOPE_NATIVE = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "num_turns": 2,
    "stop_reason": "tool_use",
    "result": "{\"issues\":[{\"title\":\"a\"}]}",
    "structured_output": {"issues": [{"title": "a"}]},
})

#: Case 3, the trap. Exit 0, `subtype: success`, `is_error: false` — and **no**
#: `structured_output` key at all, with `result` holding raw prose.
ENVELOPE_PROSE = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "num_turns": 2,
    "stop_reason": "end_turn",
    "result": "The plan looks fine to me.",
})

#: Case 3's recoverable half: prose, but the JSON document is in there.
ENVELOPE_PROSE_RECOVERABLE = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Sure! Here you go:\n```json\n{\"issues\": [{\"title\": \"a\"}]}\n```\n",
})

#: Case 3's unrecoverable-but-parseable half: JSON comes back, and it is the
#: wrong document. Recovery must not launder it.
ENVELOPE_PROSE_WRONG_SHAPE = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "{\"verdict\": \"looks good\"}",
})

#: Case 2. The model called the tool with non-conforming input; the CLI spent
#: its (pinned) budget repairing and gave up.
ENVELOPE_MAX_RETRIES = json.dumps({
    "type": "result",
    "subtype": "error_max_structured_output_retries",
    "is_error": True,
    "num_turns": 2,
    "stop_reason": "tool_use",
    "errors": ["Failed to provide valid structured output after 1 attempts"],
})

#: Case 4. Both malformed-schema refusals, verbatim from claude 2.1.233. Neither
#: reaches the API: 0 requests, nothing billed.
STDERR_SCHEMA_NOT_JSON = (
    "Error: --json-schema is not valid JSON: JSON Parse error: Expected '}'"
)
STDERR_SCHEMA_INVALID = (
    "Error: --json-schema is not a valid JSON Schema: data/type must be equal "
    "to one of the allowed values, data/type must be array, data/type must "
    "match a schema in anyOf"
)

#: The schema those envelopes were produced against.
ISSUES_SCHEMA = {
    "type": "object",
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title"],
                "properties": {"title": {"type": "string"}},
            },
        }
    },
    "additionalProperties": False,
}


@pytest.fixture
def schema():
    return StructuredSchema(document=ISSUES_SCHEMA, name="issues.schema.json")


# --------------------------------------------------------------------------- #
# §4.2.1 — the six acceptance cases
# --------------------------------------------------------------------------- #


class TestAcceptanceCases:
    """The table in §4.2.1, one test per row."""

    def test_case_1_valid_structured_output_is_native_with_no_warning(self, schema):
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_NATIVE, stderr="", exit_code=0
        )

        assert outcome.marker == so.MARKER_NATIVE
        assert outcome.failure_reason is None
        assert outcome.data == {"issues": [{"title": "a"}]}
        assert outcome.warning is None

    def test_case_2_max_structured_output_retries_fails_and_is_not_retried(
        self, schema
    ):
        """The CLI already spent its budget; our ladder must not re-run it."""
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_MAX_RETRIES, stderr="", exit_code=1
        )

        assert outcome.marker is None
        assert outcome.failure_reason == so.FAILURE_SCHEMA_VIOLATION
        assert outcome.retryable is False

    def test_case_3_prose_only_exit_zero_is_never_a_success(self, schema):
        """Success **is** the field being present — a prose answer is not one."""
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_PROSE, stderr="", exit_code=0
        )

        assert outcome.marker != so.MARKER_NATIVE
        assert outcome.marker is None
        assert outcome.failure_reason == so.FAILURE_SCHEMA_VIOLATION
        assert outcome.data is None

    def test_case_3_recovery_that_validates_is_accepted_as_recovered(self, schema):
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_PROSE_RECOVERABLE, stderr="", exit_code=0
        )

        assert outcome.marker == so.MARKER_RECOVERED
        assert outcome.marker != so.MARKER_NATIVE
        assert outcome.failure_reason is None
        assert outcome.data == {"issues": [{"title": "a"}]}
        assert outcome.warning and "issues.schema.json" in outcome.warning

    def test_case_3_recovery_that_parses_but_fails_validation_is_a_violation(
        self, schema
    ):
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_PROSE_WRONG_SHAPE, stderr="", exit_code=0
        )

        assert outcome.marker is None
        assert outcome.failure_reason == so.FAILURE_SCHEMA_VIOLATION
        assert outcome.data is None

    @pytest.mark.parametrize(
        "stderr", [STDERR_SCHEMA_NOT_JSON, STDERR_SCHEMA_INVALID]
    )
    def test_case_4_malformed_schema_is_a_configuration_error(self, schema, stderr):
        """`exit 1` with **0 requests**: never retried, never billed."""
        outcome = so.classify_attempt(
            schema, stdout="", stderr=stderr, exit_code=1
        )

        assert outcome.failure_reason == so.FAILURE_CONFIG_ERROR
        assert outcome.failure_reason != so.FAILURE_SCHEMA_VIOLATION
        assert outcome.retryable is False
        assert "issues.schema.json" in (outcome.warning or "")

    def test_case_5_timeout_is_a_timeout_not_a_schema_violation(self, schema):
        outcome = so.classify_attempt(
            schema, stdout="", stderr="", exit_code=None, timed_out=True
        )

        assert outcome.failure_reason == so.FAILURE_TIMEOUT
        assert outcome.failure_reason != so.FAILURE_SCHEMA_VIOLATION

    def test_case_5_codex_sidecar_directory_is_removed_on_an_exception(self, schema):
        """The `finally` covers timeout, kill-path teardown and exception alike."""
        seen = {}
        with pytest.raises(RuntimeError):
            with so.schema_sidecar(schema) as sidecar:
                seen["path"] = sidecar.path
                assert os.path.exists(sidecar.path)
                raise RuntimeError("timeout mid-turn")

        assert not os.path.exists(seen["path"])
        assert not os.path.exists(os.path.dirname(seen["path"]))

    def test_case_6_codex_without_a_written_sidecar_fails_before_spawn(self, schema):
        """No partial argv reaches the CLI."""
        provider = CodexProvider()

        with pytest.raises(StructuredOutputUnavailable) as excinfo:
            provider.build_command("p", "gpt-5.5", schema=schema)

        assert "issues.schema.json" in str(excinfo.value)

    def test_case_6_codex_with_a_vanished_sidecar_fails_before_spawn(
        self, schema, tmp_path
    ):
        provider = CodexProvider()
        missing = str(tmp_path / "gone" / "issues.schema.json")

        with pytest.raises(StructuredOutputUnavailable):
            provider.build_command(
                "p", "gpt-5.5", schema=schema.with_path(missing)
            )

    # -- the sidecar's happy path, so case 6 is not vacuous ------------------

    def test_the_sidecar_is_written_read_back_and_then_removed(self, schema):
        with so.schema_sidecar(schema) as sidecar:
            assert json.loads(Path(sidecar.path).read_text()) == ISSUES_SCHEMA
            argv = CodexProvider().build_command("p", "gpt-5.5", schema=sidecar)
            assert OUTPUT_SCHEMA_FLAG in argv
            assert argv[argv.index(OUTPUT_SCHEMA_FLAG) + 1] == sidecar.path
        assert not os.path.exists(sidecar.path)


class TestRetryBudgetIsPinned:
    """§4.2.1: `MAX_STRUCTURED_OUTPUT_RETRIES=1`, set explicitly, never defaulted."""

    def test_the_pin_is_one(self):
        assert so.MAX_STRUCTURED_OUTPUT_RETRIES == 1

    def test_claude_get_env_carries_the_pin(self):
        env = ClaudeCodeProvider().get_env("sonnet")
        assert env["MAX_STRUCTURED_OUTPUT_RETRIES"] == "1"


# --------------------------------------------------------------------------- #
# §5.4.1 — the in-task optional-flag gate
# --------------------------------------------------------------------------- #


class TestClaudeJsonSchemaGate:
    PROMPT = "Analyze this plan carefully"

    @pytest.fixture(autouse=True)
    def _clean_gate(self):
        reset_flag_gate_cache()
        yield
        reset_flag_gate_cache()

    @staticmethod
    def _help(text):
        """Patch the help read so the gate sees exactly ``text``."""
        return patch(
            "utils.providers.base.read_help_tokens",
            lambda binary, **_kwargs: frozenset(
                token for token in text.split() if token.startswith("--")
            ),
        )

    def test_help_carrying_the_flag_emits_it(self, schema):
        provider = ClaudeCodeProvider()
        with patch("shutil.which", return_value="/usr/bin/claude"), self._help(
            "--output-format --model --json-schema --effort"
        ):
            cmd = provider.build_command(self.PROMPT, "sonnet", schema=schema)

        assert JSON_SCHEMA_FLAG in cmd

    def test_help_lacking_the_flag_emits_no_schema_flag(self, schema, capsys):
        provider = ClaudeCodeProvider()
        with patch("shutil.which", return_value="/usr/bin/claude"), self._help(
            "--output-format --model --effort"
        ):
            cmd = provider.build_command(self.PROMPT, "sonnet", schema=schema)
            provider.build_command(self.PROMPT, "opus", schema=schema)

        assert JSON_SCHEMA_FLAG not in cmd
        assert cmd == provider.build_command(self.PROMPT, "sonnet")
        # One WARN per declined capability per process, not one per dispatch.
        warnings = [
            line
            for line in capsys.readouterr().err.splitlines()
            if JSON_SCHEMA_FLAG in line
        ]
        assert len(warnings) == 1

    def test_a_declined_gate_takes_the_recovery_branch_never_native(self, schema):
        """The degraded run is `recovered`, never silently upgraded to `native`."""
        provider = ClaudeCodeProvider()
        with patch("shutil.which", return_value="/usr/bin/claude"), self._help(
            "--output-format --model"
        ):
            cmd = provider.build_command(self.PROMPT, "sonnet", schema=schema)
        assert JSON_SCHEMA_FLAG not in cmd

        # With no schema flag on argv the CLI legitimately emits no
        # `structured_output` key — §4.2.1 case 2 applies unchanged.
        outcome = so.classify_attempt(
            schema, stdout=ENVELOPE_PROSE_RECOVERABLE, stderr="", exit_code=0
        )
        assert outcome.marker == so.MARKER_RECOVERED
        assert outcome.marker != so.MARKER_NATIVE

    def test_an_undecidable_help_read_declines_the_flag_not_the_run(self, schema):
        provider = ClaudeCodeProvider()
        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "utils.providers.base.read_help_tokens",
            lambda binary, **_kwargs: frozenset(),
        ):
            cmd = provider.build_command(self.PROMPT, "sonnet", schema=schema)

        assert JSON_SCHEMA_FLAG not in cmd
        assert cmd[0] == "claude"
        assert cmd[-1] == self.PROMPT

    def test_codex_carries_no_gate_at_its_declared_floor(self, schema, tmp_path):
        """§5.4.1's exemption: verified functionally at the declared min_version."""
        path = tmp_path / "issues.schema.json"
        path.write_text(json.dumps(ISSUES_SCHEMA))
        provider = CodexProvider()

        with patch("shutil.which", return_value=None):
            cmd = provider.build_command(
                "p", "gpt-5.5", schema=schema.with_path(str(path))
            )

        assert OUTPUT_SCHEMA_FLAG in cmd


# --------------------------------------------------------------------------- #
# §14.4's off switch — `schema=None` is byte-identical to today
# --------------------------------------------------------------------------- #


class TestSchemaKeywordIsInertWhenUnset:
    """Every adapter that never sees a schema must not move one byte.

    Counted rather than spelled: the number was "eleven" when it was written and
    is fifteen now, and a docstring that names a count is a docstring that goes
    stale on the next registry entry while the assertion beside it stays right.

    Asserted against L5's frozen ``PRE_L5_ARGV`` capture rather than against a
    freshly-built argv, so "identical to today" cannot drift into "identical to
    itself".
    """

    @pytest.fixture(autouse=True)
    def ungated(self):
        """Neutralize the flag gates, for the reason the L5 sweep does.

        omp's §2.11 probe and cursor's `--trust`/`--mode` one read the
        *installed* binary's `--help` before `build_command` emits anything, so
        ungated this sweep raises `OmpFlagsUnconfirmed` on a machine carrying an
        older omp rather than comparing argv -- machine-dependence in the sweep
        that exists to prove these bytes are machine-independent. Both gates
        resolve their binary through `LLMProvider.gate_binary`; with nothing to
        read, `missing_flags` reports nothing missing and the argv is the
        adapter's own. `TestClaudeJsonSchemaGate` above grades a gate the way a
        gate is graded -- help text stubbed, lookup real.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    @staticmethod
    def _table():
        from tests.test_providers import (
            PRE_L5_ARGV,
            PRE_L5_ARGV_MODELS,
            PRE_L5_ARGV_PROMPT,
        )

        return PRE_L5_ARGV, PRE_L5_ARGV_MODELS, PRE_L5_ARGV_PROMPT

    def test_the_table_covers_every_registered_adapter(self):
        from utils.provider_registry import _PROVIDERS

        table, _models, _prompt = self._table()
        assert set(table) == set(_PROVIDERS)
        assert len(table) == 19

    @pytest.mark.parametrize("name", sorted(so.__dict__.get("_ADAPTERS", ())) or [
        "agy", "aider", "claude-code", "cline", "codex", "copilot", "cursor-agent", "fx",
        "gemini",
        "goose", "grok", "kilocode", "kimi", "muse", "omp", "opencode", "pi",
        "qwen-code",
        "vibe",
    ])
    def test_explicit_none_renders_the_pre_m7_argv(self, name):
        from utils.provider_registry import _PROVIDERS

        table, models, prompt = self._table()
        provider = _PROVIDERS[name]

        assert provider.build_command(prompt, models[name], schema=None) == table[name]

    @pytest.mark.parametrize("name", [
        "agy", "aider", "claude-code", "cline", "codex", "copilot", "cursor-agent", "fx",
        "gemini",
        "goose", "grok", "kilocode", "kimi", "muse", "omp", "opencode", "pi",
        "qwen-code",
        "vibe",
    ])
    def test_omitting_the_keyword_is_the_same_as_passing_none(self, name):
        from utils.provider_registry import _PROVIDERS

        _table, models, prompt = self._table()
        provider = _PROVIDERS[name]

        assert provider.build_command(prompt, models[name]) == provider.build_command(
            prompt, models[name], schema=None
        )

    @pytest.mark.parametrize("name", [
        "agy", "aider", "claude-code", "cline", "codex", "copilot", "cursor-agent", "fx",
        "gemini",
        "goose", "grok", "kilocode", "kimi", "muse", "omp", "opencode", "pi",
        "qwen-code",
        "vibe",
    ])
    def test_the_other_three_builders_are_inert_too(self, name):
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_GENERATE

        _table, models, prompt = self._table()
        provider = _PROVIDERS[name]
        model = models[name]

        for transport in ("argv", "stdin"):
            if transport != "argv" and not provider.supports_prompt_transport(transport):
                continue
            assert provider.build_command_for_transport(
                prompt, model, transport, schema=None
            ) == provider.build_command_for_transport(prompt, model, transport)

        for profile in provider.agent_profiles:
            if profile == PROFILE_GENERATE:
                continue
            assert provider.build_agent_command(
                prompt, model, profile, schema=None
            ) == provider.build_agent_command(prompt, model, profile)
            for transport in ("argv", "stdin"):
                if transport != "argv" and not provider.supports_prompt_transport(
                    transport
                ):
                    continue
                assert provider.build_agent_command_for_transport(
                    prompt, model, profile, transport, schema=None
                ) == provider.build_agent_command_for_transport(
                    prompt, model, profile, transport
                )

    def test_the_structured_output_roster_is_exactly_the_measured_three(self):
        """Renamed from `..._only_claude_code_and_codex_...`: qwen-code is the
        third, measured at 0.21.13 (OQ2). `--json-schema` really constrains the
        output and fails closed on plain text, which is what licenses a `True` —
        a flag that appears in `--help` without constraining would have been a
        `False`, because the declaration is read as permission to mark a phase
        `native`."""
        from utils.provider_registry import _PROVIDERS

        declared = {
            name
            for name, provider in _PROVIDERS.items()
            if provider.structured_output_support
        }
        assert declared == {"claude-code", "codex", "qwen-code"}

    def test_an_adapter_without_support_ignores_the_schema_entirely(self, schema):
        from utils.provider_registry import _PROVIDERS

        _table, models, prompt = self._table()
        for name, provider in _PROVIDERS.items():
            if provider.structured_output_support:
                continue
            assert provider.build_command(
                prompt, models[name], schema=schema
            ) == provider.build_command(prompt, models[name])


# --------------------------------------------------------------------------- #
# §4.2.2 — the splice point, and the §9.8/L8 trap
# --------------------------------------------------------------------------- #


VARIADIC_TOOL_FLAGS = ("--allowed-tools", "--disallowed-tools")


class TestSplicePosition:
    PROMPT = "Analyze this plan carefully"

    @pytest.fixture(autouse=True)
    def _gate_open(self):
        reset_flag_gate_cache()
        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "utils.providers.base.read_help_tokens",
            lambda binary, **_kwargs: frozenset(
                {"--json-schema", "--output-format", "--model", "--effort"}
            ),
        ):
            yield
        reset_flag_gate_cache()

    @staticmethod
    def _assert_policy_region(argv, flag, prompt=None):
        assert flag in argv, argv
        at = argv.index(flag)
        # A flag with a value: the value must follow it and must not be the prompt.
        assert at + 1 < len(argv)
        assert argv[at + 1] != prompt
        # Never adjacent to the prompt on either side.
        if prompt is not None and prompt in argv:
            prompt_at = argv.index(prompt)
            assert prompt_at not in (at - 1, at + 2)
        # Never adjacent to a variadic tool flag: a token after `--allowed-tools`
        # is part of its VALUE (§9.8), so the schema flag must precede both.
        for variadic in VARIADIC_TOOL_FLAGS:
            if variadic in argv:
                assert at < argv.index(variadic), argv

    @pytest.mark.parametrize("model", ["sonnet", "fable:max"])
    def test_claude_generate_argv_splices_into_the_policy_region(self, schema, model):
        argv = ClaudeCodeProvider().build_command(self.PROMPT, model, schema=schema)

        self._assert_policy_region(argv, JSON_SCHEMA_FLAG, self.PROMPT)
        assert argv[-1] == self.PROMPT
        # §4.2.2's fixed index: immediately after `--output-format json`, never an
        # append before `cmd[-1]`.
        assert argv[:4] == ["claude", "-p", "--output-format", "json"]
        assert argv[4] == JSON_SCHEMA_FLAG
        assert json.loads(argv[5]) == ISSUES_SCHEMA

    def test_claude_stdin_transport_differs_by_exactly_the_prompt(self, schema):
        provider = ClaudeCodeProvider()
        argv_transport, _ = provider.build_command_for_transport(
            self.PROMPT, "sonnet", "argv", schema=schema
        )
        stdin_argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, "sonnet", "stdin", schema=schema
        )

        assert stdin_text == self.PROMPT
        assert stdin_argv == argv_transport[:-1]
        self._assert_policy_region(stdin_argv, JSON_SCHEMA_FLAG)

    @pytest.mark.parametrize(
        "profile", [PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE]
    )
    def test_claude_agent_profiles_keep_the_flag_ahead_of_the_variadics(
        self, schema, profile
    ):
        argv = ClaudeCodeProvider().build_agent_command(
            self.PROMPT, "sonnet", profile, schema=schema
        )

        self._assert_policy_region(argv, JSON_SCHEMA_FLAG, self.PROMPT)
        assert argv[-1] == self.PROMPT
        assert argv[-2] == "--"

    @pytest.mark.parametrize(
        "profile", [PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE]
    )
    def test_claude_agent_stdin_transport_keeps_the_flag_in_place(
        self, schema, profile
    ):
        argv, stdin_text = ClaudeCodeProvider().build_agent_command_for_transport(
            self.PROMPT, "sonnet", profile, "stdin", schema=schema
        )

        assert stdin_text == self.PROMPT
        assert self.PROMPT not in argv
        self._assert_policy_region(argv, JSON_SCHEMA_FLAG)

    def test_codex_splices_into_the_same_policy_region(self, schema, tmp_path):
        path = tmp_path / "issues.schema.json"
        path.write_text(json.dumps(ISSUES_SCHEMA))
        argv = CodexProvider().build_command(
            "p", "gpt-5.5:high", schema=schema.with_path(str(path))
        )

        self._assert_policy_region(argv, OUTPUT_SCHEMA_FLAG, "p")
        assert argv[-1] == "p"
        assert argv[argv.index(OUTPUT_SCHEMA_FLAG) + 1] == str(path)

    def test_codex_agent_readonly_keeps_the_sandbox_rewrite_by_position(
        self, schema, tmp_path
    ):
        path = tmp_path / "issues.schema.json"
        path.write_text(json.dumps(ISSUES_SCHEMA))
        argv = CodexProvider().build_agent_command(
            "p", "gpt-5.5", PROFILE_AGENT_READONLY, schema=schema.with_path(str(path))
        )

        assert argv[argv.index("--sandbox") + 1] == "read-only"
        self._assert_policy_region(argv, OUTPUT_SCHEMA_FLAG, "p")
        assert argv[-1] == "p"


# --------------------------------------------------------------------------- #
# §4.2.2 — the phase → schema-file table
# --------------------------------------------------------------------------- #


class TestPhaseSchemaTable:
    def test_the_table_is_closed_over_every_declared_phase(self):
        import modes

        assert set(so.PHASE_SCHEMA_FILE) >= set(modes.phase_names())

    def test_the_free_text_phases_are_explicitly_out_of_scope(self):
        for phase in ("ask", "implement", "apply-suggestions",
                      "apply-task-suggestions", "apply-code-fixes"):
            assert so.PHASE_SCHEMA_FILE[phase] is None
            assert so.load_phase_schema(phase) is None

    @pytest.mark.parametrize(
        "phase",
        ["review-plan", "review-tasks", "code-review", "generate-tasks",
         "consolidation"],
    )
    def test_every_affected_phase_resolves_to_a_file_that_exists(self, phase):
        loaded = so.load_phase_schema(phase)
        assert loaded is not None
        assert loaded.name == so.PHASE_SCHEMA_FILE[phase]
        assert isinstance(loaded.document, dict)
        assert loaded.document.get("$schema", "").endswith("schema#")

    def test_no_call_site_may_pass_an_inline_schema(self):
        """§4.2.2: the schema is a file, always. `load_phase_schema` is the door."""
        with pytest.raises(so.SchemaConfigError):
            so.load_phase_schema("review-plan", schema_dir=Path("/nonexistent"))

    def test_the_new_review_schema_validates_a_real_prompt_shaped_document(self):
        loaded = so.load_phase_schema("review-plan")
        from utils.schema_validator import validate_document

        document = [
            {
                "title": "Missing error handling for API timeouts",
                "desc": "Add a section specifying timeout values.",
                "importance": "high",
                "reference": "### Task 3: API Integration",
                "type": "addition",
            }
        ]
        assert validate_document(document, dict(loaded.document)) == []
        assert validate_document([], dict(loaded.document)) == []
        assert validate_document(
            [{"title": "x", "desc": "y", "importance": "critical",
              "reference": "r", "type": "addition"}],
            dict(loaded.document),
        ) != []


class TestCallerLeg:
    """§4.2.2's "the caller reads the flag before dispatch", end to end.

    Drives ``invoke_with_provider`` with a mocked ``subprocess.run`` so the six
    cases are asserted where they are really decided, not only in the policy
    function. ``schema_phase=None`` — every existing caller — is asserted to
    change nothing.
    """

    #: A phase whose schema is **strict-clean** (object root,
    #: `additionalProperties: false`, every property required), so both
    #: transports are exercised. No schema shipped today is strict-clean — see
    #: `TestNativeTransportPreconditions` — which is exactly why the caller leg
    #: needs a fixture rather than a shipped file to prove it works.
    STRICT_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "format_version": 1,
        "type": "object",
        "required": ["issues"],
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title"],
                    "properties": {"title": {"type": "string"}},
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }

    @pytest.fixture(autouse=True)
    def _gate_open(self, tmp_path):
        reset_flag_gate_cache()
        so.reset_declined_cache()
        (tmp_path / "probe.schema.json").write_text(json.dumps(self.STRICT_SCHEMA))
        table = dict(so.PHASE_SCHEMA_FILE)
        table["generate-tasks"] = "probe.schema.json"
        table["task-decomposition"] = "probe.schema.json"
        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "utils.providers.base.read_help_tokens",
            lambda binary, **_kwargs: frozenset({"--json-schema"}),
        ), patch.object(so, "SCHEMA_DIR", tmp_path), patch.object(
            so, "PHASE_SCHEMA_FILE", table
        ):
            yield
        reset_flag_gate_cache()
        so.reset_declined_cache()

    @staticmethod
    def _invoke(stdout, *, returncode=0, phase="generate-tasks", side_effect=None):
        import subprocess as sp

        from utils.llm_client import invoke_with_provider

        completed = sp.CompletedProcess(
            args=["claude"], returncode=returncode, stdout=stdout, stderr=""
        )
        with patch("utils.llm_client.subprocess.run") as run, patch(
            "utils.llm_client._resolve_executable",
            return_value=(["/usr/bin/claude"], False),
        ):
            if side_effect is not None:
                run.side_effect = side_effect
            else:
                run.return_value = completed
            result = invoke_with_provider(
                prompt="p",
                model_spec="claude-code:sonnet",
                timeout=5,
                schema_phase=phase,
            )
        return result, run

    def test_schema_phase_none_leaves_the_argv_and_the_result_untouched(self):
        import subprocess as sp

        from utils.llm_client import invoke_with_provider

        completed = sp.CompletedProcess(
            args=["claude"], returncode=0, stdout=ENVELOPE_NATIVE, stderr=""
        )
        with patch("utils.llm_client.subprocess.run", return_value=completed) as run, \
                patch(
                    "utils.llm_client._resolve_executable",
                    return_value=(["/usr/bin/claude"], False),
                ):
            result = invoke_with_provider(
                prompt="p", model_spec="claude-code:sonnet", timeout=5
            )

        assert JSON_SCHEMA_FLAG not in run.call_args.args[0]
        assert "structured_output" not in result.get("details", {})
        assert result["success"] is True

    def test_the_flag_reaches_argv_for_a_supported_phase(self):
        _result, run = self._invoke(ENVELOPE_NATIVE)
        argv = run.call_args.args[0]
        assert JSON_SCHEMA_FLAG in argv
        schema_at = argv.index(JSON_SCHEMA_FLAG)
        assert json.loads(argv[schema_at + 1])["type"] == "object"

    def test_a_native_answer_is_marked_native(self):
        result, _run = self._invoke(ENVELOPE_NATIVE)
        assert result["success"] is True
        assert result["details"]["structured_output"] == so.MARKER_NATIVE

    def test_exit_zero_without_structured_output_fails_as_schema_violation(self):
        result, _run = self._invoke(ENVELOPE_PROSE)

        assert result["success"] is False
        assert result["details"]["failure_reason"] == so.FAILURE_SCHEMA_VIOLATION
        assert result["details"]["structured_output"] is None

    def test_the_max_retries_exit_is_a_schema_violation_that_is_not_retried(self):
        result, _run = self._invoke(ENVELOPE_MAX_RETRIES, returncode=1)

        assert result["success"] is False
        assert result["details"]["failure_reason"] == so.FAILURE_SCHEMA_VIOLATION
        assert result["details"]["retryable"] is False

    def test_a_malformed_schema_is_a_config_error_not_a_schema_violation(self):
        import subprocess as sp

        from utils.llm_client import invoke_with_provider

        completed = sp.CompletedProcess(
            args=["claude"], returncode=1, stdout="", stderr=STDERR_SCHEMA_NOT_JSON
        )
        with patch("utils.llm_client.subprocess.run", return_value=completed), patch(
            "utils.llm_client._resolve_executable",
            return_value=(["/usr/bin/claude"], False),
        ):
            result = invoke_with_provider(
                prompt="p", model_spec="claude-code:sonnet", timeout=5,
                schema_phase="generate-tasks",
            )

        assert result["details"]["failure_reason"] == so.FAILURE_CONFIG_ERROR
        assert result["details"]["retryable"] is False

    def test_a_timeout_stays_a_timeout(self):
        import subprocess as sp

        from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider

        with patch(
            "utils.llm_client.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="claude", timeout=5),
        ), patch(
            "utils.llm_client._resolve_executable",
            return_value=(["/usr/bin/claude"], False),
        ):
            result = invoke_with_provider(
                prompt="p", model_spec="claude-code:sonnet", timeout=5,
                schema_phase="generate-tasks",
            )

        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"].get("failure_reason") != so.FAILURE_SCHEMA_VIOLATION

    def test_codexs_sidecar_is_written_for_the_dispatch_and_removed_after(self):
        import subprocess as sp

        from utils.llm_client import invoke_with_provider

        seen = {}

        def capture(argv, **kwargs):
            at = argv.index(OUTPUT_SCHEMA_FLAG)
            seen["path"] = argv[at + 1]
            assert os.path.isfile(seen["path"])
            return sp.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        with patch("utils.llm_client.subprocess.run", side_effect=capture), patch(
            "utils.llm_client._resolve_executable",
            return_value=(["/usr/bin/codex"], False),
        ):
            invoke_with_provider(
                prompt="p", model_spec="codex:gpt-5.5", timeout=5,
                schema_phase="task-decomposition",
            )

        assert seen["path"]
        assert not os.path.exists(seen["path"])
        assert not os.path.exists(os.path.dirname(seen["path"]))

    def test_an_out_of_scope_phase_never_attaches_a_schema(self):
        _result, run = self._invoke(ENVELOPE_PROSE, phase="ask")
        argv = run.call_args.args[0]
        assert JSON_SCHEMA_FLAG not in argv


class TestNativeTransportPreconditions:
    """Measured 2026-08-16 — a schema the CLI cannot carry is not a native run.

    Both are real, both were measured, and both resolve the same way as a
    declined §5.4.1 gate: the phase drops to §4.2.1's prompt-level path marked
    ``recovered``/``unvalidated``, never ``native``.
    """

    def test_claude_refuses_an_array_rooted_schema(self):
        provider = ClaudeCodeProvider()
        array_schema = StructuredSchema(
            document={"type": "array", "items": {"type": "object"}},
            name="code_review_issues.schema.json",
        )
        reason = so.native_transport_error(provider, array_schema)
        assert reason and "object" in reason

    def test_codex_refuses_a_non_strict_schema(self):
        provider = CodexProvider()
        loose = StructuredSchema(
            document={"type": "object", "properties": {"a": {"type": "string"}}},
            name="loose.schema.json",
        )
        reason = so.native_transport_error(provider, loose)
        assert reason and "additionalProperties" in reason

    def test_an_object_rooted_schema_is_transportable_on_claude(self):
        assert (
            so.native_transport_error(
                ClaudeCodeProvider(),
                StructuredSchema(document=ISSUES_SCHEMA, name="issues.schema.json"),
            )
            is None
        )

    def test_the_shipped_census_is_pinned(self):
        """Which shipped schema each CLI can really carry, today, as a table.

        Pinned rather than derived so the day a document's shape changes — or
        somebody authors a strict-clean schema — the change is announced here
        instead of quietly switching a phase between the native and the
        recovery path.

        Read it as the honest state of §4.2: the *mechanism* ships complete, and
        three of the six schema-carrying phases can use it on claude while
        only generate-tasks can on codex (strict-clean since 2026-10-02), because every other shipped schema predates OpenAI's
        strict subset. The other three are array-rooted documents, which no
        tool-input transport can express; changing that is a contract change for
        every consumer of those files, not something M7 may do at a call site.
        """
        from utils.provider_registry import _PROVIDERS

        claude = _PROVIDERS["claude-code"]
        codex = _PROVIDERS["codex"]
        census = {}
        for phase, name in so.PHASE_SCHEMA_FILE.items():
            if not name:
                continue
            loaded = so.load_phase_schema(phase)
            census[phase] = (
                so.native_transport_error(claude, loaded) is None,
                so.native_transport_error(codex, loaded) is None,
            )

        assert census == {
            "review-plan": (False, False),
            "review-tasks": (False, False),
            "code-review": (False, False),
            # 2026-10-02: `subagent_type` became required-but-nullable, which
            # made generated_tasks.schema.json the first strict-clean document.
            "generate-tasks": (True, True),
            "consolidation": (True, False),
            "task-decomposition": (True, False),
        }

    def test_schema_for_dispatch_declines_rather_than_emitting_a_dead_flag(self):
        provider = ClaudeCodeProvider()
        resolved, marker = so.schema_for_dispatch(provider, "code-review")
        assert resolved is None
        assert marker != so.MARKER_NATIVE

    def test_schema_for_dispatch_returns_nothing_for_an_unsupporting_adapter(self):
        from utils.provider_registry import _PROVIDERS

        resolved, marker = so.schema_for_dispatch(_PROVIDERS["gemini"], "generate-tasks")
        assert resolved is None
        assert marker != so.MARKER_NATIVE

    def test_an_adapter_declaring_no_support_is_not_warned_about(self, capsys):
        """Run 6 (#11c): "[WARN] structured output: opencode declares no
        structured_output_support" opened every generate-tasks log. It guards
        no choice: such an adapter always takes the prompt-level path, and
        the result says `recovered`. A schema an adapter that does carry one
        cannot take is still warned about, once."""
        from utils.provider_registry import _PROVIDERS

        so.reset_declined_cache()
        try:
            resolved, marker = so.schema_for_dispatch(
                _PROVIDERS["opencode"], "generate-tasks")
            assert (resolved, marker) == (None, so.MARKER_RECOVERED)
            assert "structured output" not in capsys.readouterr().err

            so.schema_for_dispatch(ClaudeCodeProvider(), "code-review")
            so.schema_for_dispatch(ClaudeCodeProvider(), "code-review")
            warned = [line for line in capsys.readouterr().err.splitlines()
                      if "[WARN] structured output" in line]
            assert len(warned) == 1 and "root type" in warned[0]
        finally:
            so.reset_declined_cache()


#: codex-cli 0.160.0 `exec --json --output-schema <file>`, captured 2026-10-05
#: verbatim from the real binary driven at a localhost Responses stub (zero
#: spend, throwaway CODEX_HOME). The request the stub received carried
#: `text.format = {"type": "json_schema", "strict": true, "name":
#: "codex_output_schema", "schema": <the file>}`; the document came back as the
#: final agent_message's text. No event carries a `structured_output` key.
CODEX_0160_OUTPUT_SCHEMA_STREAM = (
    '{"type":"thread.started","thread_id":"01a10c51-c9bb-7430-ac35-57cafd990b0c"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"{\\"answer\\": \\"hello\\"}"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,'
    '"cache_write_input_tokens":0,"output_tokens":1,"reasoning_output_tokens":0}}\n'
)

ANSWER_SCHEMA = StructuredSchema(
    document={
        "type": "object", "additionalProperties": False,
        "required": ["answer"], "properties": {"answer": {"type": "string"}},
    },
    name="answer.schema.json",
)


class TestCodexCarriesItsAnswerInTheFinalMessage:
    """Every codex task generation logged "[WARN] structured output: no
    `structured_output` key … recovered" (2026-10-05, run 3): the policy looked
    for claude's key, which codex never prints."""

    def test_the_adapter_reads_the_document_off_the_final_message(self):
        from utils.providers.codex import CodexProvider

        assert CodexProvider().native_structured_output(
            CODEX_0160_OUTPUT_SCHEMA_STREAM
        ) == {"answer": "hello"}

    def test_a_conforming_final_message_is_native_with_no_warning(self):
        from utils.providers.codex import CodexProvider

        outcome = so.classify_attempt(
            ANSWER_SCHEMA,
            stdout=CODEX_0160_OUTPUT_SCHEMA_STREAM,
            native_document=so.native_document_for(
                CodexProvider(), CODEX_0160_OUTPUT_SCHEMA_STREAM
            ),
        )

        assert outcome.marker == so.MARKER_NATIVE
        assert outcome.data == {"answer": "hello"}
        assert outcome.warning is None

    def test_a_non_conforming_one_is_still_a_violation(self):
        outcome = so.classify_attempt(
            ANSWER_SCHEMA,
            stdout=CODEX_0160_OUTPUT_SCHEMA_STREAM,
            native_document={"wrong": 1},
        )

        assert outcome.marker is None
        assert outcome.failure_reason == so.FAILURE_SCHEMA_VIOLATION

    def test_prose_around_the_document_is_not_native(self):
        from utils.providers.codex import CodexProvider

        stream = CODEX_0160_OUTPUT_SCHEMA_STREAM.replace(
            '"text":"{\\"answer\\": \\"hello\\"}"',
            '"text":"Here you go: {\\"answer\\": \\"hello\\"}"',
        )
        assert CodexProvider().native_structured_output(stream) is None

    def test_an_adapter_without_the_hook_or_a_raising_one_has_no_opinion(self):
        class _Raising:
            def native_structured_output(self, stdout):
                raise RuntimeError("boom")

        assert so.native_document_for(object(), "x") is None
        assert so.native_document_for(_Raising(), "x") is None
