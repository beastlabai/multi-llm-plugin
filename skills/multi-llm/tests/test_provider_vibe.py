"""Tests for the Vibe adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import vibe as vibe_module
from utils.providers.vibe import VibeProvider


class TestVibeProvider:
    """Tests for VibeProvider (harness-expansion.md §4.2, H6·vibe).

    Every assertion is anchored to a capture in `tests/fixtures/vibe/`, taken
    from the real binary at 2.24.1 inside the §6 probe container and re-taken
    at 2.26.0 in the pinned image on 2026-10-08 and at 2.26.1 on 2026-10-09
    (manifest.yaml) — the `tests/fixtures/kilocode/` precedent.

    vibe is the row where the plan was most wrong, so several tests here exist
    specifically to pin a **corrected** contract rather than the researched one:
    the permission ladder is inverted at the bottom (§4.2 called the bare form
    tier 1; measured it is DESTRUCTIVE), the model surface is env rather than a
    templated per-run file, and the `emits_finish_reason` retry §4.2 built an
    abort criterion around does not happen at all.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "vibe"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "vibe-default"

    @pytest.fixture
    def provider(self):
        return VibeProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- identity ---------------------------------------------------------

    def test_name_is_the_binary_not_the_pypi_package(self, provider):
        """§8.1: `mistral-vibe` is a package name, never an identifier."""
        assert provider.name == "vibe"

    def test_is_available_probes_the_binary(self, provider, monkeypatch):
        monkeypatch.setattr(
            vibe_module.shutil, "which",
            lambda n: "/usr/bin/vibe" if n == "vibe" else None,
        )
        assert provider.is_available() is True
        monkeypatch.setattr(vibe_module.shutil, "which", lambda n: None)
        assert provider.is_available() is False

    # -- argv -------------------------------------------------------------

    def test_the_prompt_is_the_last_element(self, provider):
        """§8.3: flags first, the `-p` pair trailing, so `prompt_argv_index` is -1."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "vibe"
        assert argv[-2:] == ["-p", self.PROMPT]
        assert provider.prompt_argv_index == -1

    def test_the_model_never_reaches_argv(self, provider):
        """vibe has no `--model` flag; the alias rides VIBE_ACTIVE_MODEL."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert self.MODEL not in argv
        assert "--model" not in argv

    def test_the_safe_agent_is_emitted_and_the_wide_ones_are_not(self, provider):
        """§6.5's rung, and the whole reason it is not the bare invocation.

        `default_agent` is `accept-edits`, whose `AgentSafety` is DESTRUCTIVE and
        whose write_file/edit permission is `always` — so omitting `--agent`
        would be the second-widest rung on the ladder, not tier 1.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--agent") + 1] == "plan"
        for wide in ("--auto-approve", "--yolo", "--trust", "--dangerously-skip"):
            assert wide not in argv

    def test_the_spend_and_turn_caps_are_policy_constants(self, provider):
        """Both are verbatim in the pinned argv table, never placeholders.

        `--max-price` is the roster's only USD cap but is checked BETWEEN turns —
        a single completion under a 0.0000001 cap ran to completion with the full
        answer — so `--max-turns` is the bound that actually stops a runaway.
        2.26.1 still runs that completion to its end, then reports the run
        `Stopped: price_limit` and exits 3 (the `price-limit` fixture row).
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--max-price") + 1] == vibe_module.MAX_PRICE_USD
        assert argv[argv.index("--max-turns") + 1] == vibe_module.MAX_TURNS

    def test_timeout_none_renders_byte_identical_argv(self, provider):
        """§1's guard: no default seconds value sneaks in.

        vibe has no wall-clock FLAG at all — its inner bound is the pinned
        `VIBE_API_TIMEOUT` — so a threaded timeout must change nothing here.
        """
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=None) == bare
        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=600) == bare

    def test_schema_none_renders_byte_identical_argv(self, provider):
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(self.PROMPT, self.MODEL, schema=None) == bare

    def test_the_effort_suffix_changes_no_argv(self, provider):
        """Effort is a CONFIG key, so it rides env — argv must be untouched."""
        assert provider.build_command(
            self.PROMPT, f"{self.MODEL}:high"
        ) == provider.build_command(self.PROMPT, self.MODEL)

    # -- transport --------------------------------------------------------

    def test_stdin_keeps_the_flag_and_drops_only_the_value(self, provider):
        """vibe's stdin form is the odd one out, and the difference is fatal.

        Every other stdin adapter drops the `-p <prompt>` PAIR. vibe must keep
        `-p` and drop only its value: a bare positional prompt — or no `-p` at
        all — launches the interactive TUI and hangs forever behind an ANSI
        alt-screen, which burns the whole slot rather than degrading.
        """
        argv, stdin = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin == self.PROMPT
        assert argv[-1] == "-p"
        assert self.PROMPT not in argv

    def test_the_transport_form_differs_only_by_the_prompt(self, provider):
        """§1's guard-shaped test for a second transport."""
        argv_form = provider.build_command(self.PROMPT, self.MODEL)
        stdin_form, _ = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin_form == argv_form[:-1]

    def test_a_multi_kb_prompt_fits_the_windows_cap_on_stdin(self, provider):
        """§6.4's long-prompt portability test — a unit test on every platform.

        A review-plan prompt is multi-KB by construction, and
        `_prompt_length_error` refuses to spawn above 8,191 UTF-16 units for the
        `.cmd` shim `npm install -g`/`pip install` leaves on Windows. The cap is
        measured off the rendered command line, so this needs no Windows host.
        """
        import subprocess as _sp

        big = "x" * 20000
        assert len(big) > 8191

        argv_form = provider.build_command(big, self.MODEL)
        stdin_form, payload = provider.build_command_for_transport(
            big, self.MODEL, "stdin"
        )

        assert len(_sp.list2cmdline(argv_form)) > 8191
        assert len(_sp.list2cmdline(stdin_form)) < 8191 - 256
        assert payload == big

    # -- parse_output: the §3.4 two-step unwrap ---------------------------

    def test_the_success_fixture_unwraps_to_the_models_document(self, provider):
        """Wave-1 acceptance 1: assert the unwrapped `data`, not `success` alone.

        The answer must be the model's own JSON document — not vibe's transcript
        envelope, and not the answer as a raw string. All three type-check and
        all three break reaggregation.
        """
        result = provider.parse_output(self._fixture("success-json.json"), "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"][0]["id"] == "S001"
        assert len(result["data"]) == 3

    def test_the_streaming_capture_carries_the_same_entries(self, provider):
        """`--output streaming` is the same entry objects as NDJSON.

        The adapter ships `json`, so this pins the *reason* rather than a second
        code path: one parse of one document beats N line-parses when the schema
        is identical.
        """
        raw = self._fixture("success-streaming.jsonl")
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]

        assert any(
            e.get("role") == "assistant" and e.get("type") == "message"
            for e in entries
        )

    def test_multi_assistant_takes_the_last_entry(self, provider):
        """The muse trap: last-wins, never first-wins and never concatenation."""
        result = provider.parse_output(self._fixture("multi-assistant.json"), "")

        assert result["success"] is True
        assert result["data"][0]["id"] == "S001"

    def test_a_transcript_with_no_assistant_entry_is_a_failure(self, provider):
        """An all-noise run is a failure, not an empty-string success."""
        result = provider.parse_output(self._fixture("ephemeral-only.json"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert "no assistant message" in result["error"]

    def test_a_config_error_on_stdout_is_not_read_as_an_answer(self, provider):
        """§6.2's `stdout-error` trap, in vibe's 2.24.1-2.26.0 shape.

        Config-validation errors print on **stdout with stderr empty** and still
        exit 1 — the opencode shape. The reason exists only in that text, so it
        has to be reported rather than handed to the extractor, which would find
        nothing and report an empty answer instead of the cause. The fixture is
        2.26.0's bytes; 2.26.1 moved the error to stderr (the next test).
        """
        result = provider.parse_output(self._fixture("stdout-error.stdout.txt"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert "Invalid configuration" in result["error"]

    def test_a_config_error_on_stderr_is_quoted(self, provider):
        """2.26.1 (2026-10-09): the same validation error, on stderr, stdout empty."""
        result = provider.parse_output("", self._fixture("config-error.stderr.txt"))

        assert result["success"] is False
        assert result["data"] is None
        assert "stderr said: Error: Invalid configuration" in result["error"]

    @pytest.mark.parametrize("stdout,stderr,expected", [
        # Both measured shapes: opening stdout (2.26.0), opening a stderr line
        # (2.26.1).
        ("Invalid configuration (1 error(s)):\n  - models.x: bad\n", "", "spawn_failed"),
        ("", "warn: x\nError: Invalid configuration (1 error(s)):\n", "spawn_failed"),
        # A model's answer quoting it, on a line of its own after the run
        # started: not the batch-fatal refusal.
        ('{"role": "assistant"}\nInvalid configuration (1 error(s)):\n', "", None),
        ("The review found that\nInvalid configuration (2 error(s)): is printed.", "", None),
    ])
    def test_the_config_refusal_is_read_only_where_vibe_prints_it(
        self, provider, stdout, stderr, expected
    ):
        """``spawn_failed`` fails the executor's whole batch, so on stdout the
        refusal counts only where it opens stdout, printed before any request."""
        assert provider.classify_failure(stdout, stderr, 1) == expected

    @pytest.mark.parametrize(
        "row,outcome",
        [
            ("turn-limit", "turn_limit"),
            ("price-limit", "price_limit"),
            ("length", "length"),
            ("terminated", "terminated"),
        ],
    )
    def test_a_run_stopped_by_a_limit_is_not_read_as_an_answer(
        self, provider, row, outcome
    ):
        """2.26.1's exit 3 leaves a whole transcript on stdout (2026-10-09).

        `--max-turns 4`, `--max-price 0.25`, a `length` finish and the harness's
        SIGTERM each print every entry `completed`, the last assistant message
        holding half the array or all of it, and say `Stopped: <outcome>` on
        stderr and nowhere else. Without that line the same bytes parse as a
        success — which is what the 2.26.0 parser did with all four — so the
        refusal has to read stderr.
        """
        stdout = self._fixture(f"{row}.json")
        stderr = self._fixture(f"{row}.stderr.txt")

        assert provider.parse_output(stdout, "")["success"] is True
        result = provider.parse_output(stdout, stderr)

        assert result["success"] is False
        assert result["data"] is None
        assert f"(Stopped: {outcome})" in result["error"]
        assert result["raw"] == stdout

    def test_only_vibe_s_own_stopped_line_refuses(self):
        """A whole stderr line in vibe's spelling, never a fragment of another."""
        assert vibe_module.stopped_by_a_limit("Stopped: turn_limit\n") == "turn_limit"
        assert vibe_module.stopped_by_a_limit(
            "Warning: /w is not trusted\nStopped: deadline\n"
        ) == "deadline"
        assert vibe_module.stopped_by_a_limit("Error: Stopped: turn_limit\n") is None
        assert vibe_module.stopped_by_a_limit("Stopped: the model gave up\n") is None
        assert vibe_module.stopped_by_a_limit("") is None
        assert vibe_module.stopped_by_a_limit(None) is None

    def test_zero_bytes_on_both_streams_is_classified(self, provider):
        """Distinguishable from a valid empty answer — the silent-hang shape."""
        result = provider.parse_output(
            self._fixture("empty.stdout.txt"), self._fixture("empty.stderr.txt")
        )

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"] == ""

    def test_the_failure_branch_carries_raw_and_a_null_data(self, provider):
        """`kilocode.py`'s empty-output branch is the shape §6.2 reads."""
        result = provider.parse_output("", "")

        assert set(result) >= {"success", "error", "raw", "data"}
        assert result["success"] is False
        assert result["data"] is None

    # -- failure classification -------------------------------------------

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("auth-error", "auth_failed"),
            ("rate-limit", "rate_limited"),
            ("context-overflow", "context_length_exceeded"),
        ],
    )
    def test_the_measured_stderr_shapes_classify(self, provider, fixture, expected):
        """Each pins the vendor wording the declared pattern set was measured on."""
        stderr = self._fixture(f"{fixture}.stderr.txt")

        assert provider.classify_failure("", stderr, 1) == expected

    def test_the_context_overflow_shape_is_terminal(self, provider):
        """It must reach the terminal status, not the bounded-retry path.

        vibe's wording puts the noun first and calls the bound a "limit"
        ("The conversation context exceeds the model's maximum limit"), which the
        pre-existing gateway pattern does not reach — so this is the assertion
        behind the one shared-table marker this adapter added. That sentence is
        2.24.1's (the floor) and is quoted here from the first capture; 2.26.0
        forwards the upstream sentence instead, which the committed fixture
        carries.
        """
        floor_stderr = (
            "Error: The conversation context exceeds the model's maximum limit. "
            "Use /rewind to undo recent actions, then /compact to summarize the "
            "conversation.\n"
        )
        assert provider.classify_failure("", floor_stderr, 1) == "context_length_exceeded"

        stderr = self._fixture("context-overflow.stderr.txt")

        assert "maximum context length" in stderr.lower()
        assert provider.classify_failure("", stderr, 1) == "context_length_exceeded"

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """The shape a stray `:effort` suffix produces; an unnamed model is unfindable."""
        stderr = self._fixture("invalid-model.stderr.txt")

        assert "stub-model" in stderr

    def test_no_exit_zero_failure_markers_are_declared(self, provider):
        """Empty BY MEASUREMENT: every failure mode probed exits nonzero.

        1 at 2.24.1 and 2.26.0; 1, 2 or 3 at 2.26.1 (manifest.yaml).

        §6.2's `exit-zero-failure` row is recorded N/A for this CLI rather than
        left untested, and that is what this asserts.
        """
        assert provider.exit_zero_failure_markers == ()

    # -- declarations -----------------------------------------------------

    def test_the_retry_envelope_is_measured_and_pinned(self, provider):
        """§6.1.1, including the experiment no other adapter runs.

        §4.2 predicted a 200 with no `finish_reason` makes vibe retry, and built
        abort criterion 1 on the hazard. Measured, it is one POST and a hard
        error — so the envelope's 3 attempts come from the retryable HTTP set,
        and `max_wall_s` is real (rather than `None`, as copilot's and
        qwen-code's are) precisely because `Retry-After` was NOT honoured at
        2.24.1, so no server-chosen sleep could extend it.

        2.26.0 (2026-10-08) replaced the three tries with a wall-clock retry
        budget and honours `Retry-After` up to a 30 s cap, so the declaration
        now covers both supported releases from the two pins: measured at the
        pinned 150 s budget, a 500-forever chain is 11 requests over 155 s and
        six `Retry-After: 29` 429s followed by a stall end at 267 s, under
        budget + 30 + timeout + 2 = 272 s, the same bound 2.24.1's 3 x 90 + 2
        gives.

        2.26.1 (2026-10-09) backs off 0.5 s x 1.5^n plus up to 1 s of jitter and
        makes a retry only when its wait ends inside the budget: 13 requests
        over 124.5 s on every fail-forever row (14 when the jitter runs low),
        and five `Retry-After: 29` 429s then a stall end at 239 s — so the
        request count moved up and the 272 s wall still holds.
        """
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 14
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == (
            "VIBE_API_TIMEOUT", "VIBE_API_RETRY_MAX_ELAPSED_TIME",
        )
        assert envelope.max_wall_s == max(
            3 * vibe_module.API_TIMEOUT_S + 2,
            vibe_module.API_RETRY_BUDGET_S + 30 + vibe_module.API_TIMEOUT_S + 2,
        ) == 272

    def test_the_declared_bound_sits_under_the_slot_timeout(self, provider):
        """§3.3's invariant for any inner bound."""
        assert provider.internal_retries.max_wall_s < provider.default_timeout

    def test_effort_levels_are_the_validators_own_set(self, provider):
        """Read off the installed pydantic validator, not from documentation."""
        assert vibe_module.REASONING_EFFORTS == frozenset(
            {"off", "low", "medium", "high", "max"}
        )

    def test_usage_stays_none(self, provider):
        """The transcript entries carry no token fields — §3.4's default."""
        assert provider.usage_support == "none"
        assert provider.parse_usage("anything", "") is None

    def test_structured_output_is_not_declared(self, provider):
        assert provider.structured_output_support is False

    def test_the_verified_version_is_the_capture_version(self, provider):
        # Re-captured on 2.26.0 on 2026-10-08 and on 2.26.1 on 2026-10-09;
        # 2.24.1 (the first capture) stays the floor.
        assert provider.verified_version == "2.26.1"
        assert provider.min_version == "2.24.1"
