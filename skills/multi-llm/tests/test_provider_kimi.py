"""Tests for the Kimi adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import CATEGORY_INVALID_MODEL
from utils.retry_ladder import (
    STATUS_AUTH_FAILED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_RATE_LIMITED,
)
from utils.providers import kimi as kimi_module
from utils.providers.kimi import KimiProvider


class TestKimiProvider:
    """Tests for the KimiProvider class (harness-expansion.md §3.2, H4·kimi).

    Every assertion here is anchored to a capture in `tests/fixtures/kimi/`,
    taken from the real binary at 0.36.1 inside the §6 probe container against
    the committed stub — the `tests/fixtures/kilocode/` precedent — and
    re-captured on 2.1.1 (2026-10-08) in the pinned image, through the wrapper,
    by the same probe servers.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "kimi"

    @pytest.fixture
    def provider(self):
        return KimiProvider()

    def _capture(self, stem):
        """One captured run as (stdout, stderr)."""
        out = self.FIXTURES / f"{stem}.stdout.txt"
        err = self.FIXTURES / f"{stem}.stderr.txt"
        return (
            out.read_text(encoding="utf-8") if out.exists() else "",
            err.read_text(encoding="utf-8") if err.exists() else "",
        )

    # -- identity + argv ---------------------------------------------------

    def test_provider_name(self, provider):
        assert provider.name == "kimi"

    def test_is_available_checks_the_kimi_binary(self, provider):
        with patch("shutil.which", return_value="/usr/bin/kimi") as which:
            assert provider.is_available() is True
        which.assert_called_once_with("kimi")

    def test_is_available_false_when_absent(self, provider):
        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_build_command_renders_the_measured_argv(self, provider):
        assert provider.build_command("Review this plan", "moonshotai/kimi-k2-thinking") == [
            "kimi", "--output-format", "stream-json", "-p", "Review this plan",
        ]

    def test_the_model_never_reaches_argv(self, provider):
        """The measured contract: `-m <slug>` is a HARD FAILURE at 0.36.1.

        `kimi -m moonshotai/kimi-k2-thinking` exits 1 with `Model "..." is not
        configured in config.toml`, because the KIMI_MODEL_* family synthesizes
        its model under the fixed literal id `__kimi_env_model__` and
        KIMI_MODEL_NAME carries the upstream wire slug. This is the plan's §8
        item 3 rule inverted, so it is pinned rather than left to a comment.
        """
        argv = provider.build_command("p", "moonshotai/kimi-k2-thinking")

        assert "-m" not in argv
        assert "--model" not in argv
        assert "moonshotai/kimi-k2-thinking" not in argv

    def test_the_prompt_is_last(self, provider):
        """`prompt_argv_index = -1`, and the -p pair is what trails."""
        argv = provider.build_command("the prompt", "moonshotai/kimi-k2-thinking")

        assert provider.prompt_argv_index == -1
        assert argv[-2:] == ["-p", "the prompt"]

    def test_an_effort_suffix_is_part_of_the_model_id(self, provider):
        """kimi DECLINES effort: KIMI_MODEL_THINKING_EFFORT is inert on the wire.

        So a `:high` suffix is not stripped — it stays part of the slug and
        travels to KIMI_MODEL_NAME verbatim, which is what makes the failure a
        legible "model not found" rather than a silently downgraded run.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking:high")

        assert env["KIMI_MODEL_NAME"] == "moonshotai/kimi-k2-thinking:high"

    def test_the_keywords_are_inert(self, provider):
        """§1's guard: timeout_s/schema render argv byte-identical when unset."""
        base = provider.build_command("p", "m")

        assert provider.build_command("p", "m", timeout_s=None, schema=None) == base
        assert provider.build_command("p", "m", timeout_s=600) == base

    # -- parse_output: §3.4's two-step unwrap ------------------------------

    def test_success_returns_the_models_own_json_document(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data, not `success is True`.

        The capture is a real review-plan-shaped run whose answer is a suggestion
        list, so this pins the whole chain — JSONL envelope, last assistant
        record, then `extract_json_from_text` — rather than merely that the
        parser did not raise.
        """
        result = provider.parse_output(*self._capture("success"))

        assert result["success"] is True
        assert isinstance(result["data"], list) and len(result["data"]) == 4
        assert result["data"][0]["title"] == "No rollback path for the connection pool"
        assert {s["severity"] for s in result["data"]} == {"high", "medium"}

    def test_the_resume_hint_is_never_mistaken_for_the_answer(self, provider):
        """The trap the plan did not know about (§3.4).

        The trailing `session.resume_hint` meta record ALSO carries a `content`
        key. A parser taking "the last record with a content field" returns
        "To resume this session: kimi -r ..." as the model's answer on every
        successful run, so the `role == "assistant"` filter is load-bearing.
        """
        stdout, stderr = self._capture("success")

        assert "session.resume_hint" in stdout, "fixture no longer covers the trap"
        result = provider.parse_output(stdout, stderr)

        assert "To resume this session" not in json.dumps(result["data"])

    def test_multi_assistant_takes_the_last_message(self, provider):
        """§6.2's `multi-assistant` row: last-wins, never first-wins or concat.

        The muse trap. The capture interleaves an assistant preamble, a tool
        record and the real answer; first-wins would return the preamble and
        concatenation would return prose glued to JSON.
        """
        result = provider.parse_output(*self._capture("multi-assistant"))

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"][0]["title"] == "No rollback path"
        assert "Let me look at the plan first." not in json.dumps(result["data"])

    def test_an_all_noise_run_is_a_failure_not_an_empty_success(self, provider):
        """§6.2's `ephemeral-only` row.

        5xx-forever emits `system.version` plus nine `turn.step.retrying` meta
        records and no assistant record at all. Reporting that as an empty
        success would hand reaggregation a run that never happened.
        """
        result = provider.parse_output(*self._capture("ephemeral-only"))

        assert result["success"] is False
        assert result["data"] is None
        assert "no assistant record" in result["error"]

    def test_a_truncated_stream_fails_cleanly(self, provider):
        """§6.2's `truncated` row: never a traceback, never a half-parsed object."""
        result = provider.parse_output(*self._capture("truncated"))

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"]

    def test_an_unparseable_trailing_line_does_not_lose_the_answer(self, provider):
        """A stream cut mid-record keeps every complete record before the cut."""
        stdout, _ = self._capture("success")
        result = provider.parse_output(stdout + '{"role":"assis', "")

        assert result["success"] is True
        assert isinstance(result["data"], list)

    def test_the_failure_branch_has_the_documented_shape(self, provider):
        """`kilocode.py`'s empty-output branch is the shape §3.4 names."""
        result = provider.parse_output("", "")

        assert set(result) == {"success", "error", "raw", "data"}
        assert result["success"] is False and result["data"] is None

    # -- failure classification (§6.2) -------------------------------------

    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("auth-error", STATUS_AUTH_FAILED),
            ("rate-limit", STATUS_RATE_LIMITED),
            ("context-overflow", STATUS_CONTEXT_LENGTH_EXCEEDED),
            ("invalid-model", CATEGORY_INVALID_MODEL),
        ],
    )
    def test_captured_failures_classify(self, provider, stem, expected):
        """Each §6.2 fixture pins the classification it was captured for."""
        stdout, stderr = self._capture(stem)

        assert provider.classify_failure(stdout, stderr, exit_code=1) == expected

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """§6.2: an unnamed model makes the failure unfindable."""
        _stdout, stderr = self._capture("invalid-model")

        assert "no-such-model-xyz" in stderr

    def test_a_masked_error_envelope_is_a_recorded_limitation(self, provider):
        """§6.2's `error-envelope` row — a FINDING, pinned so it cannot regress
        into a silent assumption.

        kimi MASKS a provider error delivered inside a 200 success envelope: the
        vendor's text is replaced by a generic "empty response" message, so it
        never reaches `auth_error_patterns`/`rate_limit_patterns`. The run still
        fails loudly (exit 1 plus a classified line), which is what makes this
        acceptable — but a future reader must not conclude the vendor's own error
        text is available here, because it is not.
        """
        stdout, stderr = self._capture("error-envelope")

        assert "empty response" in stderr
        assert provider.classify_failure(stdout, stderr, exit_code=1) is None
        assert provider.parse_output(stdout, stderr)["success"] is False

    def test_no_exit_zero_markers_are_declared(self, provider):
        """Measured, not omitted: across 27 probe runs the exit code was
        authoritative in every shape (0 success, 1 failure). Declaring a marker
        would be guessing at a failure mode this CLI does not have."""
        assert provider.exit_zero_failure_markers == ()

    # -- §2.2 rule 6: the three guard tests --------------------------------

    B1_SELECTOR_SET = (
        "KIMI_MODEL_NAME",
        "KIMI_MODEL_API_KEY",
        "KIMI_MODEL_BASE_URL",
        "KIMI_MODEL_PROVIDER_TYPE",
    )

    def test_get_env_carries_the_whole_b1_selector_set_by_name(self, provider):
        """§2.2 rule 2 — selection is total, never partial.

        KIMI_MODEL_NAME is the *enable switch*: verified against the binary,
        with no member of the family set, `kimi provider list --json` returns
        empty and the CLI silently reverts to its config-file route. So a
        half-set is not a degraded run, it is a run on somebody else's
        credential.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        for key in self.B1_SELECTOR_SET:
            assert key in env, key
        assert env["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
        assert env["KIMI_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"

    def test_the_base_url_is_emitted_even_when_the_key_is_empty(
        self, provider, monkeypatch
    ):
        """§2.2 rule 3: never omit the base URL to "let the CLI decide".

        With no key the run must fail as a legible OpenRouter auth error, not
        quietly resolve to the Moonshot subscription endpoint.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        assert env["KIMI_MODEL_API_KEY"] == ""
        assert env["KIMI_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"

    def test_the_base_url_carries_the_version_segment(self, provider):
        """Measured: kimi appends `/chat/completions` VERBATIM and inserts no
        `/v1` of its own, so a base URL without it 404s."""
        assert provider.get_env("m")["KIMI_MODEL_BASE_URL"].endswith("/v1")

    def test_get_remove_env_strips_every_unset_family_member(self, provider):
        """§8.2's rule verbatim, over the measured 18-name family.

        A `KIMI_MODEL_TEMPERATURE` or `KIMI_MODEL_MAX_CONTEXT_SIZE` left over
        from the user's own kimi session would silently reshape a fan-out request
        nobody configured.
        """
        stripped = set(provider.get_remove_env())
        # The adapter's REAL injected set, not the B1 selector set: the output
        # ceiling is a KIMI_MODEL_* name that is injected without being a
        # backend selector, and reading get_env keeps the two in step.
        injected = set(provider.get_env("moonshotai/kimi-k2-thinking"))

        for name in kimi_module._KIMI_MODEL_VARS:
            if name in injected:
                assert name not in stripped, f"{name} is injected, not stripped"
            else:
                assert name in stripped, f"unset family member not stripped: {name}"

    def test_get_remove_env_strips_the_b3_names(self, provider):
        """The Moonshot subscription route and the legacy aliases (§2.2 rule 3)."""
        stripped = set(provider.get_remove_env())

        for name in kimi_module._STRIPPED_VENDOR_VARS:
            assert name in stripped, name
        assert "KIMI_API_KEY" in stripped
        assert "KIMI_LOOP_MAX_RETRIES_PER_STEP" in stripped

    def test_every_catalog_entry_is_in_the_b1_namespace(self):
        """§2.2 rule 6(c): every shipped `models:` entry is an OpenRouter slug,
        passed verbatim to the model surface (here, KIMI_MODEL_NAME)."""
        from utils.provider_registry import load_base_config

        models = load_base_config()["providers"]["kimi"]["models"]

        assert models, "kimi ships no models: it would be invisible to the picker"
        for slug in models:
            assert "/" in slug, f"not an OpenRouter slug: {slug}"
            assert KimiProvider().get_env(slug)["KIMI_MODEL_NAME"] == slug

    def test_the_default_model_is_models_zero(self):
        """`--init` injects models[0]; the current Kimi slug must reach its env."""
        from utils.provider_registry import load_base_config

        models = load_base_config()["providers"]["kimi"]["models"]

        assert models[0] == "moonshotai/kimi-k3.1-pro"
        assert KimiProvider().get_env(models[0])["KIMI_MODEL_NAME"] == models[0]

    def test_defaults_models_gains_no_uncommented_row(self):
        """§8 item 3's rule: the shipped example line is COMMENTED.

        `is_available()` is a bare `shutil.which`, so an active row would put a
        provider the user has no binary for into the default fan-out panel and
        into `--quick` — a dead slot in every review.
        """
        from utils.provider_registry import load_base_config

        defaults = load_base_config()["defaults"]["models"]

        assert not [m for m in defaults if str(m).startswith("kimi:")]

    # -- the pinned retry knob (§6.1.1) ------------------------------------

    def test_the_output_ceiling_is_pinned(self, provider):
        """Measured against the REAL endpoint, which is the only place it shows.

        kimi derives `max_tokens` from its synthesized model's context size and
        sends 131072 on every call. Several OpenRouter upstreams cap output below
        that and REJECT the request rather than clamping it -- the first live
        smoke died on `400 Provider returned error`, raw body "Requested maximum
        tokens of 131072 exceeds the maximum output tokens limit: 102400". So the
        pin is what makes kimi answer at all on those providers, not a spend
        preference. It is also §7.4's per-CLI output ceiling.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        assert env["KIMI_MODEL_MAX_TOKENS"] == str(kimi_module.KIMI_MODEL_MAX_TOKENS)
        assert kimi_module.KIMI_MODEL_MAX_TOKENS < 102_400

    def test_the_other_token_knobs_stay_stripped(self, provider):
        """Three other names move the same wire field. Only the one that names it
        is injected; the rest are stripped so a stale export cannot quietly raise
        the ceiling back to something an upstream rejects."""
        stripped = set(provider.get_remove_env())

        for name in ("KIMI_MODEL_MAX_OUTPUT_SIZE",
                     "KIMI_MODEL_MAX_COMPLETION_TOKENS",
                     "KIMI_MODEL_MAX_CONTEXT_SIZE"):
            assert name in stripped, name

    def test_the_envelope_declares_the_pinned_number(self, provider):
        """`claude_code.py`'s precedent: pin the knob, declare the PINNED number.

        The bare CLI measured 10 attempts; `get_env` pins
        KIMI_LOOP_MAX_ATTEMPTS_PER_STEP=1, so declaring 10 would budget for a
        configuration this harness never runs.
        """
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == kimi_module.KIMI_LOOP_MAX_ATTEMPTS_PER_STEP
        assert str(envelope.max_requests) == provider.get_env("m")[
            "KIMI_LOOP_MAX_ATTEMPTS_PER_STEP"
        ]

    def test_the_wall_clock_is_honestly_unbounded(self, provider):
        """`max_wall_s=None` is forced twice over, not unmeasured: a stalled
        request has no client-side timeout (no output within 420 s, and no
        KIMI_* request-timeout knob exists) and a honoured `Retry-After` is
        server-chosen. Rule 4 then runs no admission test, which is honest."""
        assert provider.internal_retries.max_wall_s is None

    def test_rate_limits_are_retried_internally(self, provider):
        """Measured against a 429-forever stub. `True` is what makes §7.5 rule
        6(b) suppress an identical harness-side re-dispatch."""
        assert provider.internal_retries.retries_rate_limit is True
