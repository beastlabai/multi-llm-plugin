"""Tests for the Copilot adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.copilot import CopilotProvider


class TestCopilotProvider:
    """Tests for CopilotProvider (harness-expansion.md §3.1, H4·copilot).

    Every assertion is anchored to a capture in `tests/fixtures/copilot/`, taken
    from the real binary at 1.0.80 inside the §6 probe container — the
    `tests/fixtures/kilocode/` precedent — and re-captured at 1.0.93 on
    2026-10-08, except `exit-zero-failure.jsonl`, which only 1.0.80 produces.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "openai/gpt-4.1-mini"

    #: The document the captured answer really carries, so the success test can
    #: assert the UNWRAPPED data rather than `success is True` alone
    #: (Wave-1 acceptance criterion 1).
    ANSWER = [
        {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
        {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
    ]

    @pytest.fixture
    def provider(self):
        return CopilotProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- argv -------------------------------------------------------------

    def test_build_command_puts_the_prompt_pair_last(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "copilot"
        assert argv[-2:] == ["-p", self.PROMPT]
        assert argv[provider.prompt_argv_index] == self.PROMPT

    def test_build_command_emits_no_permission_flag(self, provider):
        """§6.5 tier 1, and the help text's own claim is what this refutes.

        `--allow-all-tools` is documented as "required for non-interactive
        mode". Measured, all four rungs produced byte-identical output with a
        clean filesystem canary, so rule 1's ascending stop applies at rung 1 and
        §3.1's bracketed line is struck. Widening this later must be a visible
        test edit, which is the whole point of asserting absence.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        for flag in (
            "--allow-all-tools", "--allow-tool", "--deny-tool", "--add-dir",
            "--yolo", "--allow-all", "--allow-all-paths", "--allow-all-urls",
        ):
            assert flag not in argv, f"{flag} is above the measured tier"

    def test_build_command_carries_the_determinism_flags(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        for pair in (
            ["--output-format", "json"],
            ["--stream", "off"],
            ["--log-level", "none"],
            ["--model", self.MODEL],
        ):
            assert pair[0] in argv and argv[argv.index(pair[0]) + 1] == pair[1]
        for flag in ("--no-custom-instructions", "--no-ask-user",
                     "--no-auto-update", "--disable-builtin-mcps"):
            assert flag in argv

    def test_timeout_and_schema_keywords_are_inert(self, provider):
        """§1's guard: copilot has no wall-clock flag and no schema transport."""
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=None, schema=None
        ) == bare
        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=600) == bare

    def test_declares_no_structured_output_support(self, provider):
        assert provider.structured_output_support is False

    # -- the :effort suffix ----------------------------------------------

    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_every_measured_effort_level_renders(self, provider, effort):
        argv = provider.build_command(f"{self.MODEL}:{effort}", self.MODEL)
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:{effort}")

        assert argv[argv.index("--model") + 1] == self.MODEL
        assert argv[argv.index("--effort") + 1] == effort
        assert argv[-1] == self.PROMPT

    def test_an_unknown_suffix_passes_through_as_part_of_the_model_id(self, provider):
        argv = provider.build_command(self.PROMPT, "openai/gpt-4.1-mini:free")

        assert argv[argv.index("--model") + 1] == "openai/gpt-4.1-mini:free"
        assert "--effort" not in argv

    def test_the_effort_suffix_keeps_the_prompt_last(self, provider):
        """grok's regression, re-pinned here: the suffix must not move argv[-1]."""
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:high")

        assert argv[provider.prompt_argv_index] == self.PROMPT

    # -- prompt transport (§6.4) ------------------------------------------

    def test_stdin_transport_drops_the_whole_pair(self, provider):
        """Measured: piped stdin with `-p` OMITTED carries the prompt.

        `-p -` sends the literal one-character prompt `-`, and a bare `-` exits
        1 — so dropping only `argv[-1]` would leave a dangling `-p` and change
        what the CLI is asked to do.
        """
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin_text == self.PROMPT
        assert "-p" not in argv
        assert self.PROMPT not in argv
        assert argv == provider.build_command(self.PROMPT, self.MODEL)[:-2]

    def test_argv_transport_matches_build_command(self, provider):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv == provider.build_command(self.PROMPT, self.MODEL)

    def test_a_long_prompt_stays_under_the_windows_cap_on_stdin(self, provider):
        """§6.4's portability test — measured off the rendered argv, so it needs
        no Windows host.

        A review-plan prompt is multi-KB by construction, which is above
        `CMDLINE_CAP_UTF16_BATCH` (8191) and would be refused before spawn on the
        `.cmd` shim `npm install -g` leaves on Windows.
        """
        import subprocess

        long_prompt = "x" * 20000
        argv, stdin_text = provider.build_command_for_transport(
            long_prompt, self.MODEL, "stdin"
        )

        assert stdin_text == long_prompt
        rendered = subprocess.list2cmdline(argv)
        assert len(rendered.encode("utf-16-le")) // 2 < 8191 - 256

    # -- parse_output: the two-step unwrap (§3.4) -------------------------

    def test_success_returns_the_models_own_document(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data, not `success` alone."""
        result = provider.parse_output(self._fixture("success.jsonl"), "")

        assert result["success"] is True
        assert result["data"] == self.ANSWER

    def test_the_streamed_shape_parses_identically(self, provider):
        """One recipe covers both stream modes, so `--stream off` is a
        determinism choice rather than a parsing requirement."""
        result = provider.parse_output(self._fixture("success-streamed.jsonl"), "")

        assert result["success"] is True
        assert result["data"] == self.ANSWER

    def test_the_user_message_is_never_served_as_the_answer(self, provider):
        """THE trap §3.4's recipe walks into.

        `user.message` is non-ephemeral and carries `data.content` — our own
        prompt. Filtering on `ephemeral` alone and taking the first survivor
        returns the prompt as the model's answer, which type-checks and is wrong.
        """
        raw = self._fixture("success.jsonl")
        prompt_record = [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]
        prompt_text = next(
            r["data"]["content"] for r in prompt_record if r.get("type") == "user.message"
        )

        result = provider.parse_output(raw, "")

        assert len(prompt_text) > 30000, "the fixture's prompt echo is the trap"
        assert result["data"] == self.ANSWER
        assert prompt_text not in json.dumps(result["data"])

    def test_multi_assistant_takes_the_last_message(self, provider):
        """last-wins, not first-wins and not concatenation — the muse trap."""
        raw = self._fixture("multi-assistant.jsonl")

        result = provider.parse_output(raw, "")

        assert "FIRST ANSWER" in raw
        assert result["success"] is True
        assert result["data"] == self.ANSWER

    # -- parse_output: the §6.2 failure set --------------------------------

    def test_an_all_noise_run_is_a_failure(self, provider):
        result = provider.parse_output(self._fixture("ephemeral-only.jsonl"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"]

    def test_a_truncated_stream_has_no_result_record(self, provider):
        """The capture is a real kill mid-flight: it ends at `model.call_start`,
        so a parser that assumes a terminal `result` record raises instead of
        classifying."""
        raw = self._fixture("truncated.jsonl")

        result = provider.parse_output(raw, "")

        assert '"type":"result"' not in raw
        assert result["success"] is False
        assert result["data"] is None

    def test_the_exit_zero_failure_is_caught_structurally(self, provider):
        """exit 0, `result.exitCode: 0`, no error record, empty answer.

        There is no text for a marker to match, which is why
        `exit_zero_failure_markers` is empty for this adapter and the detection
        lives here instead.
        """
        raw = self._fixture("exit-zero-failure.jsonl")

        result = provider.parse_output(raw, "")

        assert '"exitCode":0' in raw
        assert result["success"] is False
        assert "empty assistant message" in result["error"]

    def test_empty_output_is_classified_not_crashed(self, provider):
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_empty_stdout_quotes_stderr_when_there_is_any(self, provider):
        """stderr is 0 bytes on every model-path failure, so text here is an
        argument-parse or launch error — exactly the case worth quoting."""
        result = provider.parse_output("", "error: unknown option '--nope'")

        assert result["success"] is False
        assert "--nope" in result["error"]

    @pytest.mark.parametrize(
        "fixture,needle",
        [
            ("auth-error.jsonl", "authentication"),
            ("rate-limit.jsonl", "rate_limit"),
            ("invalid-model.jsonl", "query"),
            ("context-overflow.jsonl", "query"),
        ],
    )
    def test_session_error_becomes_the_reported_failure(
        self, provider, fixture, needle
    ):
        """`session.error` on stdout with stderr empty is §6.2's `stdout-error`
        trap in its exact shape, and it is copilot's ONLY error channel."""
        result = provider.parse_output(self._fixture(fixture), "")

        assert result["success"] is False
        assert needle in result["error"]
        assert result["data"] is None

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """Which is what makes a stray `:effort` suffix findable rather than an
        unattributable 404."""
        result = provider.parse_output(self._fixture("invalid-model.jsonl"), "")

        assert "gpt-5.5" in result["error"]

    # -- classification ----------------------------------------------------

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("auth-error.jsonl", "auth_failed"),
            ("rate-limit.jsonl", "rate_limited"),
            ("invalid-model.jsonl", "invalid_model"),
            ("context-overflow.jsonl", "context_length_exceeded"),
        ],
    )
    def test_classify_failure_reads_the_error_record(
        self, provider, fixture, expected
    ):
        """The classifier reads STDOUT here, because that is where copilot's
        errors are and stderr is empty."""
        assert provider.classify_failure(
            self._fixture(fixture), "", 1
        ) == expected

    def test_auth_wording_needed_its_own_pattern(self, provider):
        """copilot says "Authentication failed with provider", which matches none
        of the base set's ten patterns."""
        from utils.providers.base import LLMProvider as Base

        assert "authentication failed" in provider.auth_error_patterns
        assert "authentication failed" not in Base.auth_error_patterns

    def test_the_exit_zero_failure_stays_transport_shaped(self, provider):
        """Deliberate: it reaches the ladder as `None` and takes the bounded
        retry path, which is the right treatment for an empty answer."""
        assert provider.classify_failure(
            self._fixture("exit-zero-failure.jsonl"), "", 0
        ) is None

    def test_exit_zero_markers_are_empty_by_measurement(self, provider):
        assert provider.exit_zero_failure_markers == ()

    # -- the measured internal-retry envelope (§6.1.1) --------------------

    def test_the_envelope_is_the_measured_one_not_the_plans_placeholders(
        self, provider
    ):
        """§3.1 guessed 5 / ~30 s / codex's False. All three were wrong."""
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 6
        assert envelope.max_wall_s is None
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == ()

    def test_the_rate_limit_fixture_corroborates_the_request_count(self, provider):
        """The CLI's own error text is the independent second source."""
        raw = self._fixture("rate-limit.jsonl")

        assert "retried 5 times" in raw  # 1 + 5 == max_requests

    # -- env (§2.2, §8.2) --------------------------------------------------

    def test_get_env_emits_the_complete_selector_set(self, provider):
        env = provider.get_env(self.MODEL)

        for name in (
            "COPILOT_PROVIDER_BASE_URL", "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_API_KEY", "COPILOT_OFFLINE",
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS",
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS",
        ):
            assert name in env, f"{name} missing from the B1 selector set"
        assert env["COPILOT_PROVIDER_BASE_URL"].endswith("/v1")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"
        assert env["COPILOT_OFFLINE"] == "true"

    def test_the_base_url_is_emitted_even_with_no_key(self, provider, monkeypatch):
        """§2.2 rule 3: never omit the base URL to "let the CLI decide".

        The base URL is the BYOK activation switch, so omitting it does not
        degrade the run — it selects the GitHub seat path instead.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        env = provider.get_env(self.MODEL)

        assert env["COPILOT_PROVIDER_API_KEY"] == ""
        assert env["COPILOT_PROVIDER_BASE_URL"].startswith("https://openrouter.ai")

    def test_does_not_set_xdg_cache_home(self, provider):
        """A per-slot value would re-extract copilot's 209 MB package cache on
        EVERY spawn, and a hardcoded path would be wrong on a host. The image
        owns this one."""
        assert "XDG_CACHE_HOME" not in provider.get_env(self.MODEL)

    def test_never_mutates_the_parent_environment(self, provider):
        before = dict(os.environ)

        provider.get_env(self.MODEL)

        assert os.environ == before

    @pytest.mark.parametrize(
        "name",
        [
            "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
            "COPILOT_ALLOW_ALL", "COPILOT_PROVIDER_BEARER_TOKEN",
            "OTEL_EXPORTER_OTLP_ENDPOINT", "COPILOT_OTEL_ENABLED",
            "COPILOT_MODEL",
        ],
    )
    def test_get_remove_env_strips_every_measured_name(self, provider, name):
        assert name in provider.get_remove_env()

    def test_the_tier_widening_env_twin_is_stripped(self, provider):
        """Without this, §6.5's recorded tier is unenforceable: COPILOT_ALLOW_ALL
        promotes the run to tier 4 with no argv change at all."""
        assert "COPILOT_ALLOW_ALL" in provider.get_remove_env()

    # -- the token-limit pair (§3.1): the catalog and the floor, full stop.
    # The override rung above them was REMOVED rather than deferred (C9);
    # `TestCopilotHasNoConfigurableTokenLimits` below pins its absence.

    def test_a_catalogued_slug_gets_its_own_pair(self, provider):
        from utils.providers.copilot import OUTPUT_RESERVE_TOKENS, token_limits_for

        prompt_budget, output_budget = token_limits_for("openai/gpt-4o-mini")

        assert prompt_budget == 128000 - OUTPUT_RESERVE_TOKENS
        assert output_budget == OUTPUT_RESERVE_TOKENS

    def test_an_uncatalogued_slug_falls_to_the_floor(self, provider):
        from utils.providers.copilot import (
            FLOOR_MAX_OUTPUT_TOKENS, FLOOR_MAX_PROMPT_TOKENS, token_limits_for,
        )

        assert token_limits_for("some/unknown-model") == (
            FLOOR_MAX_PROMPT_TOKENS, FLOOR_MAX_OUTPUT_TOKENS,
        )

    def test_the_pair_is_resolved_for_the_model_handed_in(self, provider):
        """Never a process-wide constant, and never a pair resolved for a
        different slug."""
        small = provider.get_env("openai/gpt-4o-mini")
        large = provider.get_env("openai/gpt-4.1-mini")

        assert (small["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"]
                != large["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"])

    def test_the_effort_suffix_does_not_break_the_limit_lookup(self, provider):
        """The catalog is keyed by slug, so the suffix has to come off first."""
        plain = provider.get_env("openai/gpt-4o-mini")
        suffixed = provider.get_env("openai/gpt-4o-mini:high")

        assert (suffixed["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"]
                == plain["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"])

    def test_every_shipped_catalog_entry_is_known_to_the_limit_table(self):
        """A slug in `providers.yaml` with no catalog row silently takes the
        floor, which is safe but is not what the block claims."""
        import utils.provider_registry as reg
        from utils.providers.copilot import _MODEL_CATALOG

        config = reg.load_config()
        for slug in config["providers"]["copilot"]["models"]:
            assert slug in _MODEL_CATALOG, f"{slug} has no measured token pair"

    # -- version + usage ---------------------------------------------------

    def test_usage_stays_none_because_no_input_count_exists(self, provider):
        """`result.usage` carries premiumRequests (0 on BYOK by construction),
        durations and codeChanges — no prompt-token field at all, and
        `data.outputTokens` vanishes under `--stream on`. Absent must mean "not
        reported", never 0."""
        assert provider.usage_support == "none"
        assert provider.parse_usage(self._fixture("success.jsonl"), "") is None

    def test_version_parsing_tolerates_the_trailing_update_line(self, provider):
        """`copilot --version` prints "GitHub Copilot CLI 1.0.80." AND a second
        "Run 'copilot update'…" line, so nothing may anchor to end-of-output."""
        raw = "GitHub Copilot CLI 1.0.80.\nRun 'copilot update' to check for updates.\n"

        assert provider.parse_version(raw, "") == "1.0.80"


@pytest.mark.config_override
class TestCopilotHasNoConfigurableTokenLimits:
    """C9: the `providers.yaml` override rung is removed, not deferred.

    `token_limits_for` derives the pair from the committed per-slug catalog and
    the conservative floor, and from nothing else. What is pinned here is the
    **absence** of the affordance, because an absence is exactly what a later
    reader re-adds half-way: the config layer merges an undeclared
    `max_prompt_tokens:` into the copilot block and then nothing reads it, so a
    documented key would have been a silent no-op rather than a feature.

    copilot is the only adapter in the registry that declares a token budget at
    all, which is why the typed-config alternative was rejected instead of held
    open, and why the one condition that reopens it is a *second* such adapter.
    """

    MODEL = "openai/gpt-4o-mini"

    #: Deliberately far from both the catalog pair (128000 - 8000 / 8000) and
    #: the floor (32000 / 8000), so a half-implemented read of either key shows
    #: up as an equality failure rather than hiding behind a coincidence.
    CONFIGURED_PROMPT = 190000
    CONFIGURED_OUTPUT = 32000

    def _with_the_key_configured(self, monkeypatch, tmp_path):
        """A providers.yaml carrying both keys, as the removed rung described."""
        import utils.provider_registry as registry

        override = tmp_path / "providers.yaml"
        override.write_text(
            "providers:\n"
            "  copilot:\n"
            f"    max_prompt_tokens: {self.CONFIGURED_PROMPT}\n"
            f"    max_output_tokens: {self.CONFIGURED_OUTPUT}\n",
            encoding="utf-8",
        )
        # Deterministic layering: project discovery off, env layer on, cache dropped.
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        registry._config = None
        registry._config_key = None
        return registry

    def test_the_key_really_does_reach_the_merged_provider_block(
        self, monkeypatch, tmp_path
    ):
        """The premise of the removal, measured rather than argued.

        `provider_registry` merges an undeclared key and then ignores it, so the
        key is *present* in the loaded block. That is what makes documenting it
        worse than not having it, and it is the half a reader must see before
        the two assertions below mean anything.
        """
        registry = self._with_the_key_configured(monkeypatch, tmp_path)

        block = registry.load_config()["providers"]["copilot"]

        assert block["max_prompt_tokens"] == self.CONFIGURED_PROMPT
        assert block["max_output_tokens"] == self.CONFIGURED_OUTPUT

    def test_the_same_pair_comes_back_with_and_without_the_block(
        self, monkeypatch, tmp_path
    ):
        from utils.providers.copilot import token_limits_for

        without = token_limits_for(self.MODEL)
        self._with_the_key_configured(monkeypatch, tmp_path)

        assert token_limits_for(self.MODEL) == without

    def test_get_env_emits_the_derived_pair_and_not_the_configured_one(
        self, monkeypatch, tmp_path
    ):
        """The end of the wire: what the CLI is actually told."""
        from utils.providers.copilot import token_limits_for

        derived_prompt, derived_output = token_limits_for(self.MODEL)
        self._with_the_key_configured(monkeypatch, tmp_path)

        env = CopilotProvider().get_env(self.MODEL)

        assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] == str(derived_prompt)
        assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] == str(derived_output)
        assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != str(self.CONFIGURED_PROMPT)
        assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != str(self.CONFIGURED_OUTPUT)

    def test_the_decision_is_written_where_the_next_reader_will_look(self):
        """A rejected alternative with no reopening condition is just a gap.

        Both halves live in `token_limits_for`'s own docstring, next to the
        behaviour they explain, rather than only in a plan file.
        """
        from utils.providers.copilot import token_limits_for

        doc = (token_limits_for.__doc__ or "").lower()

        assert "second adapter" in doc, (
            "the one reopening condition is not recorded"
        )
        assert "typing the key is" in doc, (
            "the docstring no longer says that a user typing the key does NOT "
            "reopen the decision"
        )
        assert "no config override" in doc, (
            "the docstring no longer states the absence as a decision"
        )
