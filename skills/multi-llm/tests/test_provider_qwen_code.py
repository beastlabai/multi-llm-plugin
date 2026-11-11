"""Tests for the Qwen Code adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import qwen_code as qwen_code_module
from utils.providers.qwen_code import QwenCodeProvider


class TestQwenCodeProvider:
    """Tests for QwenCodeProvider (harness-expansion.md §3.3, H4·qwen).

    Every assertion is anchored to a capture in `tests/fixtures/qwen-code/`,
    taken from the real binary at 0.25.0 in the pinned container image
    (re-captured 2026-10-08; first captured at 0.21.13 in the §6 probe) — the
    `tests/fixtures/kilocode/` precedent. The two `*.0.21.13.stdout.json` files
    keep the floor release's exit-0 API-error shape graded.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "qwen-code"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "qwen/qwen3.8-max"

    @pytest.fixture
    def provider(self):
        return QwenCodeProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- identity ---------------------------------------------------------

    def test_name_is_the_harness_not_the_model_family(self, provider):
        """§8.1: the key names the HARNESS, the binary is `qwen`."""
        assert provider.name == "qwen-code"

    def test_is_available_probes_the_binary_not_the_key(self, provider, monkeypatch):
        monkeypatch.setattr(
            qwen_code_module.shutil, "which", lambda n: "/usr/bin/qwen" if n == "qwen" else None
        )
        assert provider.is_available() is True
        monkeypatch.setattr(qwen_code_module.shutil, "which", lambda n: None)
        assert provider.is_available() is False

    # -- argv -------------------------------------------------------------

    def test_prompt_is_a_bare_trailing_positional(self, provider):
        """`-p` is DEPRECATED at 0.21.13; the positional is the shipped form."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "qwen"
        assert argv[-1] == self.PROMPT
        assert "-p" not in argv and "--prompt" not in argv

    def test_declared_prompt_index_matches_the_rendered_argv(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)
        assert argv[provider.prompt_argv_index] == self.PROMPT

    def test_the_shipped_tool_tier_is_on_every_invocation(self, provider):
        """§6.5: `--max-tool-calls 0` is a rung BELOW tier 1 and always emitted."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--max-tool-calls") + 1] == "0"
        # and none of the rungs ABOVE tier 1 ship, because tier 1 answered
        assert "--yolo" not in argv
        assert "--approval-mode" not in argv
        assert "--add-dir" not in argv
        assert "--include-directories" not in argv

    def test_json_output_mode_and_no_chat_recording(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("-o") + 1] == "json"
        assert argv[argv.index("--chat-recording") + 1] == "false"
        assert "--bare" in argv

    def test_model_travels_on_the_m_flag(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)
        assert argv[argv.index("-m") + 1] == self.MODEL

    def test_an_effort_suffix_is_not_split_off(self, provider):
        """qwen has NO effort surface, so the suffix stays part of the model id.

        Measured: `-m qwen3-coder-plus:high` reached the backend verbatim. The
        adapter declining to parse it is what this pins — a silent split would
        send a different model than the user asked for.
        """
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:high")
        assert argv[argv.index("-m") + 1] == f"{self.MODEL}:high"

    # -- §1's guard: timeout_s / schema are inert when unset ---------------

    def test_timeout_none_renders_no_wall_time_flag(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=None)
        assert "--max-wall-time" not in argv

    def test_timeout_none_is_byte_identical_to_omitting_it(self, provider):
        assert provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=None
        ) == provider.build_command(self.PROMPT, self.MODEL)

    def test_schema_none_is_byte_identical_to_omitting_it(self, provider):
        assert provider.build_command(
            self.PROMPT, self.MODEL, schema=None
        ) == provider.build_command(self.PROMPT, self.MODEL)

    # -- §3.3's wall-time table -------------------------------------------

    @pytest.mark.parametrize("outer,expected", [
        (1200, "1170"),
        (600, "570"),
        (300, "270"),
        (91, "61"),
        (90, "60"),
    ])
    def test_wall_time_derivation_table(self, provider, outer, expected):
        """Bare integer seconds, outer minus the 30 s shutdown margin."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=outer)
        assert argv[argv.index("--max-wall-time") + 1] == expected

    @pytest.mark.parametrize("outer", [1200, 600, 300, 91])
    def test_the_inner_bound_is_strictly_under_the_outer(self, provider, outer):
        """agy's INNER_SHUTDOWN_MARGIN_S invariant: qwen must exit first."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=outer)
        assert int(argv[argv.index("--max-wall-time") + 1]) < outer

    def test_a_too_small_timeout_clamps_and_warns(self, provider, capsys):
        """qwen REJECTS sub-1s values, so the clamp is a guard, not a taste."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=40)

        assert argv[argv.index("--max-wall-time") + 1] == "60"
        assert "WARNING" in capsys.readouterr().err

    # -- structured output (OQ2) ------------------------------------------

    def test_schema_renders_inline_like_claude_code(self, provider):
        from utils.providers.base import StructuredSchema

        schema = StructuredSchema(document={"b": 2, "a": 1}, name="s.json")
        argv = provider.build_command(self.PROMPT, self.MODEL, schema=schema)

        value = argv[argv.index("--json-schema") + 1]
        # sorted keys + compact separators, so one schema is one argv byte string
        assert value == '{"a":1,"b":2}'

    def test_schema_is_declared_inline_not_file(self, provider):
        assert provider.structured_output_support is True
        assert provider.structured_output_transport == "inline"

    def test_the_schema_never_lands_next_to_the_prompt(self, provider):
        from utils.providers.base import StructuredSchema

        schema = StructuredSchema(document={"a": 1}, name="s.json")
        argv = provider.build_command(self.PROMPT, self.MODEL, schema=schema)

        assert argv[-1] == self.PROMPT
        assert argv[-2] != "--json-schema"

    # -- stdin transport (§6.4) -------------------------------------------

    def test_stdin_transport_is_declared(self, provider):
        assert provider.supports_prompt_transport("stdin") is True

    def test_stdin_form_differs_by_exactly_the_prompt(self, provider):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )
        full = provider.build_command(self.PROMPT, self.MODEL)

        assert stdin_text == self.PROMPT
        assert argv == full[:-1]

    def test_a_long_prompt_stays_under_the_windows_cap_on_stdin(self, provider):
        """§6.4's portability test — measured off the rendered argv, so it runs
        everywhere and needs no Windows host."""
        import subprocess

        long_prompt = "x" * 20000
        assert len(long_prompt) > 8191

        argv, stdin_text = provider.build_command_for_transport(
            long_prompt, self.MODEL, "stdin"
        )
        rendered = subprocess.list2cmdline(argv)

        assert stdin_text == long_prompt
        assert len(rendered.encode("utf-16-le")) // 2 < 8191 - 256

    def test_the_argv_form_of_that_prompt_would_not_fit(self, provider):
        """The other half: this is why the stdin transport is load-bearing."""
        import subprocess

        argv = provider.build_command("x" * 20000, self.MODEL)
        rendered = subprocess.list2cmdline(argv)

        assert len(rendered.encode("utf-16-le")) // 2 > 8191

    # -- parse_output: the real document ----------------------------------

    def test_success_returns_the_models_document_not_the_envelope(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data."""
        result = provider.parse_output(self._fixture("success-document.stdout.json"), "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert [s["id"] for s in result["data"]] == ["S1", "S2"]
        assert result["data"][0]["title"] == "Bound the retry loop"

    def test_structured_output_capture_unwraps_to_the_document(self, provider):
        result = provider.parse_output(
            self._fixture("structured-output-success.stdout.json"), ""
        )

        assert result["success"] is True
        assert result["data"] == {"suggestions": [{"id": 1, "title": "probe"}]}

    def test_a_bare_string_answer_is_not_a_document(self, provider):
        """`success.stdout.json` answers "OK" — no JSON, so the unwrap says so
        rather than inventing one."""
        result = provider.parse_output(self._fixture("success.stdout.json"), "")

        assert result["success"] is False
        assert result["data"] is None

    # -- parse_output: the exit-0 defect ----------------------------------

    def test_exit_zero_failure_is_classified_despite_three_green_signals(self, provider):
        """rc=0, subtype=success, is_error=false — and it is still a failure.

        Captured on 0.21.13, the adapter's `min_version`: 0.25.0 no longer
        writes this shape (see the next test), but the floor still does."""
        raw = self._fixture("exit-zero-failure.0.21.13.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["subtype"] == "success" and record["is_error"] is False

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert "[API Error: 500" in result["error"]
        assert result["data"] is None

    def test_since_0_25_0_an_api_error_is_an_error_record(self, provider):
        """0.25.0 marks the same 500-forever run as an error: exit 1,
        `is_error: true`, no `result` key, the text under `error.message`, and
        the same text in an `AlreadyReportedError` envelope on stderr. The
        failure must read as that text, not as the error object's repr."""
        raw = self._fixture("exit-zero-failure.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["is_error"] is True and "result" not in record
        assert record["error"] == {"message": "[API Error: 500 internal server error]"}
        stderr = self._fixture("exit-zero-failure.stderr.txt")
        assert json.loads(stderr)["error"]["type"] == "AlreadyReportedError"

        result = provider.parse_output(raw, stderr)

        assert result["success"] is False
        assert result["error"] == "[API Error: 500 internal server error]"
        assert result["data"] is None

    @pytest.mark.parametrize("row,expected", [
        ("auth-error", "[API Error: 401 Incorrect API key provided.]"),
        ("invalid-model", "[API Error: 404 The model `stub-model` does not exist]"),
        ("truncated", "[API Error: Model stream ended without a finish reason.]"),
        ("empty-stream", "[API Error: Model stream ended with empty response text.]"),
    ])
    def test_the_error_text_is_the_message_not_the_object(self, provider, row, expected):
        result = provider.parse_output(
            self._fixture(f"{row}.stdout.json"), self._fixture(f"{row}.stderr.txt")
        )
        assert result["success"] is False
        assert result["error"] == expected

    @pytest.mark.parametrize("row,expected", [
        ("auth-error", "auth_failed"),
        ("invalid-model", "invalid_model"),
        ("context-overflow", "context_length_exceeded"),
    ])
    def test_the_0_25_0_streams_classify_with_their_stderr(self, provider, row, expected):
        """Both channels as captured: stdout's error record and stderr's
        `AlreadyReportedError` envelope carry the same upstream sentence."""
        assert provider.classify_failure(
            self._fixture(f"{row}.stdout.json"), self._fixture(f"{row}.stderr.txt"), 1
        ) == expected

    @pytest.mark.parametrize("message", [
        "[API Error: 401 Incorrect API key provided.]",  # the captured stub wording
        "[API Error: 401 No auth credentials found]",    # OpenRouter's missing-key wording
        "[API Error: 401 User not found.]",              # OpenRouter's revoked-key wording
    ])
    def test_a_401_is_an_auth_failure_whatever_the_upstream_wording(self, provider, message):
        """qwen renders every upstream failure as `[API Error: <status> <text>]`.
        The base auth patterns match none of these sentences, so a 401 used to
        classify `server_error` (the `api error` rung) and enter the retry
        ladder as retryable — on 0.21.13 and 0.25.0 alike. The status in qwen's
        own prefix is what names it."""
        assert provider.classify_failure(message, "", 1) == "auth_failed"

    def test_the_ripgrep_advisory_does_not_decide_the_class(self, provider):
        """ADAPTER CHANGE (0.25.0). A root-owned install cannot restore the exec
        bit on qwen's bundled `rg`, so every run's stderr opens with
        `Ripgrep not available: spawn …/rg EACCES. Falling back to built-in
        grep.` Captured that way in the pinned image: a 429-forever run ended by
        `--max-wall-time`. The `EACCES` classified it `permission_denied`,
        hiding the rate limit the next line reports."""
        stderr = self._fixture("rate-limit-wall-time.root-owned-install.stderr.txt")
        first, rest = stderr.split("\n", 1)
        assert first.startswith("Ripgrep not available: ") and "EACCES" in first
        assert "[API Error: 429 rate limit exceeded]" in rest

        assert provider.vendor_text("", stderr) == ("", rest)
        assert provider.classify_failure("", stderr, 55) == "rate_limited"
        assert provider.classify_failure("", stderr, 55) == provider.classify_failure(
            "", self._fixture("rate-limit-wall-time.stderr.txt"), 55
        )

    def test_the_ripgrep_advisory_alone_classifies_nothing(self, provider):
        """The same line ahead of a successful run's empty stderr."""
        line = ("Ripgrep not available: spawn /opt/qwen/vendor/ripgrep/x64-linux/rg "
                "EACCES. Falling back to built-in grep.\n")
        success = self._fixture("success.stdout.json")
        assert provider.classify_failure(success, line, 0) is None

    def test_the_budget_envelope_is_found_after_retry_lines(self, provider):
        """ADAPTER CHANGE (0.25.0). The exit-55 envelope used to be the whole
        of stderr; 0.25.0 prints a `Retrying in …` line per backoff first, so
        a parse from byte 0 failed and the report quoted the retry line instead
        of the abort reason."""
        stderr = self._fixture("rate-limit-wall-time.stderr.txt")
        assert stderr.startswith("Retrying in 60s")

        result = provider.parse_output(self._fixture("rate-limit-wall-time.stdout.txt"), stderr)

        assert result["success"] is False
        assert result["error"] == (
            "qwen aborted with no output — FatalBudgetExceededError: Run aborted: "
            "wall-clock budget of 100s exceeded (--max-wall-time)."
        )
        rooted = provider.parse_output(
            "", self._fixture("rate-limit-wall-time.root-owned-install.stderr.txt")
        )
        assert "FatalBudgetExceededError" in rooted["error"]

    @pytest.mark.parametrize("fixture,needle", [
        ("auth-error.stdout.json", "401"),
        ("invalid-model.stdout.json", "404"),
        ("context-overflow.stdout.json", "400"),
        ("empty-stream.stdout.json", "[API Error:"),
    ])
    def test_every_api_error_class_is_a_failure(self, provider, fixture, needle):
        result = provider.parse_output(self._fixture(fixture), "")

        assert result["success"] is False
        assert needle in result["error"]

    def test_invalid_model_failure_names_the_model(self, provider):
        """§6.2: an unnamed model makes a stray `:effort` suffix unfindable."""
        result = provider.parse_output(self._fixture("invalid-model.stdout.json"), "")
        assert "stub-model" in result["error"]

    def test_concatenated_partial_text_across_retries_is_still_a_failure(self, provider):
        """The muse trap by a different route: the marker is a SUFFIX here, so a
        prefix test would serve `OKOKOKOKOK` as the answer. Captured on 0.21.13
        (`min_version`); 0.25.0 discards each retry's partial text — see
        `test_0_25_0_keeps_only_the_last_attempts_partial_text`."""
        raw = self._fixture("truncated.0.21.13.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["result"].startswith("OK")
        assert record["stats"]["models"]["stub-model"]["api"]["totalErrors"] == 0

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert "OKOKOKOKOK" not in str(result.get("data"))

    def test_0_25_0_keeps_only_the_last_attempts_partial_text(self, provider):
        """0.25.0 writes one `system`/`retry` record per retry (with
        `preserveText: false`), the assistant record holds only the last
        attempt's `OK` ahead of the marker, and the `result` record is an error
        record — so the parser reads the verdict, not the text."""
        raw = self._fixture("truncated.stdout.json")
        records = json.loads(raw)
        retries = [r for r in records if r["type"] == "system" and r.get("subtype") == "retry"]
        assert retries and all(r["data"]["preserveText"] is False for r in retries)
        texts = [c["text"] for r in records if r["type"] == "assistant"
                 for c in r["message"]["content"] if c["type"] == "text"]
        assert texts == ["OK[API Error: Model stream ended without a finish reason.]"]
        assert records[-1]["is_error"] is True and "result" not in records[-1]

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert result["data"] is None

    def test_no_finish_reason_takes_the_same_path(self, provider):
        result = provider.parse_output(self._fixture("no-finish-reason.stdout.json"), "")
        assert result["success"] is False

    def test_the_marker_is_searched_not_prefix_matched(self, provider):
        """The marker arrives as a SUFFIX on the truncated shape, so the
        mechanism has to search. `classify_streams` consumes these as compiled
        regexes via `_matches_marker`, which is `.search` — asserted here
        against the real helper rather than against a re-implementation."""
        from utils.providers.base import _matches_marker

        assert _matches_marker(
            provider.exit_zero_failure_markers,
            "OKOKOKOKOK[API Error: 500 internal server error]",
        )
        assert not _matches_marker(
            provider.exit_zero_failure_markers, "a perfectly ordinary answer"
        )

    # -- parse_output: the error-shaped envelope --------------------------

    def test_an_error_shaped_record_is_not_served_as_the_answer(self, provider):
        """The trap the probe found, and the half of it that is now closed.

        An error record has no `result` key, so `parse_result_envelope`'s
        `if "result" in wrapper` guard missed it and the helper returned the
        record itself as the answer. C5 closed the half this fixture happens to
        exercise: the helper now refuses any envelope whose own `is_error` /
        `terminal_reason` says the run failed, and qwen's error record carries
        `is_error: true` like claude-code's does, so the helper alone would now
        refuse these bytes.

        **The ordering this test exists for is unaffected, and the second half
        below is why it still has to be an ordering.** The helper reads two
        fields; qwen's rule 1 reads three — it also refuses a `subtype` other
        than `"success"`, and a record with no `result` key at all. Strip the
        one field C5 reads and the trap is still there, which is what the
        `is_error`-less derivation pins.
        """
        from utils.providers.result_envelope import parse_result_envelope

        raw = self._fixture("stdout-error.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["is_error"] is True and record["subtype"] != "success"

        # the shared helper, on the record as qwen really wrote it
        refused = parse_result_envelope(json.dumps(record))
        assert refused["success"] is False
        assert "is_error: true" in refused["error"]

        # the same record minus that one field: no `result` key and no verdict
        # the helper reads; it refuses only because a result envelope without
        # a `result` field carries no answer
        blind = dict(record)
        del blind["is_error"]
        naive = parse_result_envelope(json.dumps(blind))
        assert naive["success"] is False and naive["data"] is None
        assert "no result field" in naive["error"]

        # what still needs qwen's rule first: a `subtype` the helper never reads
        answered = dict(blind, result='[{"id": 1}]')
        assert parse_result_envelope(json.dumps(answered))["success"] is True
        assert provider.parse_output(json.dumps([answered]), "")["success"] is False

        # what the adapter does, because it classifies FIRST
        result = provider.parse_output(raw, "")
        assert result["success"] is False
        assert result["data"] is None

    def test_multi_assistant_is_last_wins_by_construction(self, provider):
        """qwen avoids the muse trap here: step 1 reads the `result` record
        rather than scanning assistant messages."""
        raw = self._fixture("multi-assistant.stdout.json")
        assert len([r for r in json.loads(raw) if r["type"] == "assistant"]) >= 2

        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["result"] == "LAST-ASSISTANT-TEXT"

    # -- parse_output: degenerate shapes ----------------------------------

    def test_empty_stdout_is_a_classified_failure(self, provider):
        """The exit-55 shape writes zero bytes — distinguishable from an empty
        answer, and never a traceback."""
        result = provider.parse_output(self._fixture("wall-time-exceeded.stdout.txt"), "")

        assert result["success"] is False
        assert result["data"] is None

    def test_an_array_with_no_result_record_is_a_failure(self, provider):
        """Defensive: no run of the real binary produces this, which is why it is
        a unit test rather than a fixture (see ephemeral-only.NOT-APPLICABLE.md)."""
        result = provider.parse_output(
            json.dumps([{"type": "system", "subtype": "init"}]), ""
        )

        assert result["success"] is False
        assert "no result record" in result["error"]

    def test_non_json_stdout_is_a_classified_failure(self, provider):
        result = provider.parse_output("this is not json at all", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_a_json_object_instead_of_an_array_is_a_failure(self, provider):
        result = provider.parse_output(json.dumps({"type": "result"}), "")

        assert result["success"] is False
        assert "array" in result["error"]

    # -- env (§2.2 / §8.2) ------------------------------------------------

    def test_get_env_carries_the_whole_b1_selector_set_by_name(self, provider, monkeypatch):
        """§2.2 rule 6(a): asserted BY NAME, including the base URL when the key
        is empty — a half-set is the bug, not a degraded run."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = provider.get_env(self.MODEL)

        assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
        assert env["OPENAI_API_KEY"] == ""
        assert env["OPENAI_MODEL"] == self.MODEL

    def test_the_base_url_carries_v1(self, provider):
        """Measured: qwen appends /chat/completions verbatim and adds no /v1."""
        assert provider.get_env(self.MODEL)["OPENAI_BASE_URL"].endswith("/v1")

    def test_the_key_comes_from_openrouter_only(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert provider.get_env(self.MODEL)["OPENAI_API_KEY"] == "sk-or-test"

    def test_get_env_never_mutates_os_environ(self, provider, monkeypatch):
        """§3.3 rule 1: a process-wide write would put qwen's base URL under
        codex's next spawn in the same fan-out."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        before = dict(os.environ)

        provider.get_env(self.MODEL)

        assert dict(os.environ) == before

    def test_telemetry_and_updater_are_opted_out(self, provider):
        env = provider.get_env(self.MODEL)
        assert env["QWEN_TELEMETRY_ENABLED"] == "false"
        assert env["QWEN_USAGE_STATISTICS_ENABLED"] == "false"
        assert env["QWEN_CODE_SKIP_UPDATE_CHECK_ONCE"] == "1"

    def test_get_remove_env_strips_every_b3_name(self, provider):
        """§2.2 rule 6(b), asserted by name."""
        stripped = set(provider.get_remove_env())

        assert {"QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_CODE_MODEL",
                "QWEN_DEFAULT_AUTH_TYPE", "DASHSCOPE_API_KEY",
                "DASHSCOPE_BASE_URL", "MODELSCOPE_API_KEY"} <= stripped

    def test_the_legacy_openai_alias_is_stripped(self, provider):
        """A strip is not a substitute for injecting the base URL — both."""
        assert "OPENAI_API_BASE" in provider.get_remove_env()
        assert "OPENAI_BASE_URL" in provider.get_env(self.MODEL)

    def test_it_never_strips_a_name_it_also_sets(self, provider):
        """A value we strip must never also be a value we branch on."""
        assert set(provider.get_remove_env()).isdisjoint(provider.get_env(self.MODEL))

    def test_unattended_retry_is_not_set(self, provider):
        """Measured inert in both states, so setting it would advertise a knob
        that moves nothing — and would drag §1's guard in for free."""
        assert "QWEN_CODE_UNATTENDED_RETRY" not in provider.get_env(self.MODEL)

    # -- internal retries (§6.1.1) ----------------------------------------

    def test_the_envelope_is_declared_from_the_wire_not_the_self_report(self, provider):
        """qwen's own stats said 7 requests where the stub logged 28."""
        assert provider.internal_retries.max_requests == 28

    def test_max_wall_s_is_none_because_two_paths_are_unbounded(self, provider):
        assert provider.internal_retries.max_wall_s is None

    def test_it_declares_that_it_retries_rate_limits(self, provider):
        """Makes §7.5 rule 6(b) suppress a duplicate harness-side backoff."""
        assert provider.internal_retries.retries_rate_limit is True

    def test_configurable_by_names_only_a_knob_that_moves_a_number(self, provider):
        assert provider.internal_retries.configurable_by == ("QWEN_CODE_API_TIMEOUT_MS",)

    # -- version / usage ---------------------------------------------------

    def test_version_is_a_bare_semver(self, provider):
        """The draft lock pattern guessed a product-name prefix; there is none."""
        assert provider.parse_version("0.21.13\n", "") == "0.21.13"

    def test_usage_stays_none_until_the_live_smoke_earns_it(self, provider):
        assert provider.usage_support == "none"
        assert provider.parse_usage("anything", "") is None
