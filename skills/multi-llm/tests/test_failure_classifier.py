"""M19 (§7.2/§7.1.a) — one shared failure classifier, and no duplicates left.

Every raw-substring classifier this repo grew independently decides something
expensive: a re-dispatch at a fresh full timeout, a batch-fatal ``auth_failed``,
or a persisted ``recoverable`` flag that buys a whole revalidation pass. The
point of M19 is that there is exactly **one** implementation of "what is this
failure", living in ``utils/providers/base.py`` beside the boundary-aware
matcher, and that Lane A stops carrying its own copies of it.

The tests are grouped by the thing that can regress:

* the shared category vocabulary and the retryable allowlist (§7.2)
* ``context_length_exceeded`` as a **terminal** status — the assertion is on the
  number of dispatches, never on the label, because the label is cheap and the
  re-dispatch is what costs a slot (§7.6(a))
* launch failures ordered root-cause-before-auth, with ``exit_code == 127``
* the exit-0 CLIs, whose exit code carries no verdict at all
* the six §7.1.a duplicate sites, asserted gone by a source-level guard
* ``validation.py``'s delegation, and the persisted contract it may not move
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.providers import base as pbase  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402
from utils.retry_ladder import (  # noqa: E402
    BACKOFF_STATUSES,
    FATAL_STATUSES,
    RETRYABLE_STATUSES,
    STATUS_AUTH_FAILED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_RATE_LIMITED,
    STATUS_SPAWN_FAILED,
    STATUS_TIMEOUT,
)
from utils.subagent_tasks import build_inputs, make_task  # noqa: E402
from tests import opencode_fixtures as oc_pin  # noqa: E402


SKILL_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Measured provider streams. Every constant below was captured from the real
# binary on the date in its comment, or lifted from a fixture already in this
# repo — none is invented, because a classifier tested against invented text
# pins the test author's imagination rather than the CLI.
# ---------------------------------------------------------------------------

#: goose 1.41.0, exit **0**, OPENROUTER_API_KEY=<invalid>, 2026-08-16.
GOOSE_EXIT_ZERO_AUTH = (
    '{"messages":[{"role":"assistant","content":[{"type":"text","text":'
    '"Ran into this error: Authentication error: Authentication failed for '
    "https://openrouter.ai/api/v1/chat/completions. Status: 401 Unauthorized. "
    'Response: User not found..\\n\\nPlease retry if you think this is a '
    'transient or recoverable error."}]}]}'
)

#: goose, same marker, a failure that names no auth or rate-limit word at all.
GOOSE_EXIT_ZERO_OPAQUE = (
    '{"messages":[{"role":"assistant","content":[{"type":"text","text":'
    '"Ran into this error: the tool call could not be completed."}]}]}'
)

#: aider 0.86.2, exit **0**, --api-key openrouter=sk-invalid, 2026-08-16.
AIDER_EXIT_ZERO_AUTH = (
    "litellm.AuthenticationError: AuthenticationError: OpenrouterException - \n"
    '{"error":{"message":"Missing Authentication header","code":401}}\n'
    "The API provider is not able to authenticate you. Check your API key.\n"
)

#: cursor-agent, exit **0** (all six auth states exit 0, §5.3), scratch HOME,
#: 2026-08-16. Not JSON at all — the envelope never gets written.
CURSOR_EXIT_ZERO_AUTH = (
    "Error: Authentication required. Please run 'agent login' first, or set "
    "CURSOR_API_KEY environment variable.\n"
)

#: cursor-agent's JSON envelope carries the verdict in `is_error`, never in the
#: exit code.
CURSOR_ENVELOPE_IS_ERROR = (
    '{"type":"result","subtype":"error","is_error":true,"duration_ms":812,'
    '"result":"The request could not be completed.","session_id":"s1"}'
)

#: cline 3.0.39, exit **0**, OPENROUTER_API_KEY=<invalid>, 2026-08-16. The
#: message names no auth word whatsoever — only the structured field does.
CLINE_EXIT_ZERO_AUTH = (
    '{"type":"agent_event","event":{"type":"error","error":{"name":"Error",'
    '"message":"User not found."},"errorClass":"unknown","recoverable":false}}\n'
    '{"type":"run_result","finishReason":"error","iterations":1,'
    '"durationMs":419,"text":"User not found."}\n'
    '{"type":"error","message":"User not found."}\n'
)

#: codex-cli 0.146.0, exit 1, OPENAI_API_KEY=<invalid>, 2026-08-16.
CODEX_TURN_FAILED_401 = (
    '{"type":"error","message":"unexpected status 401 Unauthorized: Missing '
    'bearer or basic authentication in header, url: '
    'https://api.openai.com/v1/responses"}\n'
    '{"type":"turn.failed","error":{"message":"unexpected status 401 '
    "Unauthorized: Missing bearer or basic authentication in header, url: "
    'https://api.openai.com/v1/responses"}}\n'
)

#: codex 0.146.0, the measured rate-limit terminal (§7.2's measured note): a 429
#: is never retried internally, so this is the whole story the stream tells.
CODEX_RETRY_LIMIT_429 = (
    "stream error: exceeded retry limit, last status: 429 Too Many Requests"
)

#: The capture rig's `auth_error` case on the pin: opencode's auth failure, an
#: `error` event on **stdout** (stderr carries only `--print-logs` lines).
OPENCODE_PROVIDER_AUTH_ERROR = oc_pin.read("auth_error")

#: `opencode auth list`, exit **0**, scratch HOME/XDG dirs, 2026-08-16. A
#: logged-out opencode prints this and exits 0; today the probe reads that as
#: READY.
OPENCODE_AUTH_LIST_EMPTY = (
    "┌  Credentials ~/data/opencode/auth.json\n"
    "│\n"
    "└  0 credentials\n"
)

#: The same command with one stored credential (measured on this box).
OPENCODE_AUTH_LIST_POPULATED = (
    "┌  Credentials ~/.local/share/opencode/auth.json\n"
    "│\n"
    "●  OpenRouter api\n"
    "│\n"
    "└  1 credentials\n"
)

#: The same command with no stored credential but a usable environment key.
OPENCODE_AUTH_LIST_ENV_ONLY = (
    "┌  Credentials ~/data/opencode/auth.json\n"
    "│\n"
    "└  0 credentials\n"
    "\n"
    "┌  Environment\n"
    "│\n"
    "●  OpenAI OPENAI_API_KEY\n"
    "│\n"
    "└  1 environment variable\n"
)

#: §7.2's three named vendors, verbatim from their APIs.
CONTEXT_OVERFLOW_STREAMS = {
    "openai": (
        "Error: This model's maximum context length is 128000 tokens. However, "
        "your messages resulted in 214537 tokens. Please reduce the length of "
        "the messages."
    ),
    "anthropic": (
        '{"type":"error","error":{"type":"invalid_request_error","message":'
        '"prompt is too long: 248113 tokens > 200000 maximum"}}'
    ),
    "gemini": (
        "google.api_core.exceptions.InvalidArgument: 400 The input token count "
        "(1093122) exceeds the maximum number of tokens allowed (1048576)."
    ),
    "openai_code": (
        '{"error":{"code":"context_length_exceeded","message":"too many tokens"}}'
    ),
    # goose 1.53.0's own compaction report (tests/fixtures/goose/
    # context-overflow): it swallows the upstream 400 and says "effective".
    "goose_compaction": (
        "Ran into this error trying to compact: Failed to compact: the base "
        "prompt (system prompt, tool schemas, and conversation) exceeds the "
        "model's effective context window, and there are no tool responses to "
        "remove."
    ),
}

#: The prose §7.1.a:653 misread as a context overflow. `'context' in error`
#: matches every one of these.
CONTEXT_FALSE_POSITIVES = (
    "the task declares context_mode: workspace-read, which this runner lacks",
    "in this context the reviewer means the caller, not the callee",
    "Failed to parse JSON: the plan's Context section is 40 lines long",
    "this step exceeds the effective context of the plan's Phase 2 section",
)


# ---------------------------------------------------------------------------
# §7.2 — the shared vocabulary
# ---------------------------------------------------------------------------

class TestSharedRetryableAllowlist:
    """One allowlist, shared across harnesses, with two deliberate absences."""

    def test_the_allowlist_is_exactly_the_four_declared_codes(self):
        assert pbase.RETRYABLE_FAILURE_CATEGORIES == frozenset({
            STATUS_RATE_LIMITED,
            STATUS_TIMEOUT,
            pbase.CATEGORY_CONNECTION_ERROR,
            pbase.CATEGORY_SERVER_ERROR,
        })

    def test_auth_is_deliberately_absent(self):
        """Retrying a logged-out CLI spends the budget to fail identically."""
        assert STATUS_AUTH_FAILED not in pbase.RETRYABLE_FAILURE_CATEGORIES

    def test_context_length_exceeded_is_deliberately_absent(self):
        """§7.2: out of the allowlist until §7.6(a) ships a splitter."""
        assert (
            STATUS_CONTEXT_LENGTH_EXCEEDED
            not in pbase.RETRYABLE_FAILURE_CATEGORIES
        )

    def test_the_status_strings_are_the_ladder_s_own(self):
        """No second copy of a status string (the ladder owns the vocabulary)."""
        import utils.retry_ladder as ladder

        assert pbase.STATUS_RATE_LIMITED is ladder.STATUS_RATE_LIMITED
        assert pbase.STATUS_AUTH_FAILED is ladder.STATUS_AUTH_FAILED
        assert (
            pbase.STATUS_CONTEXT_LENGTH_EXCEEDED
            is ladder.STATUS_CONTEXT_LENGTH_EXCEEDED
        )


# ---------------------------------------------------------------------------
# §7.6(a) — context overflow is terminal, and the proof is a dispatch count
# ---------------------------------------------------------------------------

class TestContextOverflowIsTerminal:
    """Classify it, name it, fail the slot. **No** re-dispatch of any kind."""

    @pytest.mark.parametrize("vendor", sorted(CONTEXT_OVERFLOW_STREAMS))
    def test_each_vendor_s_overflow_classifies(self, vendor):
        assert pbase.classify_streams(
            CONTEXT_OVERFLOW_STREAMS[vendor], ""
        ) == STATUS_CONTEXT_LENGTH_EXCEEDED

    @pytest.mark.parametrize("prose", CONTEXT_FALSE_POSITIVES)
    def test_prose_mentioning_context_is_not_an_overflow(self, prose):
        """`:653`'s `'context' in error` fired on every one of these."""
        assert pbase.classify_streams(prose, "") != STATUS_CONTEXT_LENGTH_EXCEEDED

    def test_the_ladder_never_retries_it(self):
        assert STATUS_CONTEXT_LENGTH_EXCEEDED not in RETRYABLE_STATUSES
        assert STATUS_CONTEXT_LENGTH_EXCEEDED not in BACKOFF_STATUSES
        assert STATUS_CONTEXT_LENGTH_EXCEEDED not in FATAL_STATUSES

    def test_the_message_names_the_model_and_both_remedies(self):
        message = pbase.context_length_message("codex:gpt-5.5")
        assert "codex:gpt-5.5" in message
        assert "narrow the plan/diff scope" in message
        assert "larger-context model" in message


class TestContextOverflowSpawnsNothingTwice:
    """The assertion is the *count*, not the label (§14.2)."""

    def _lane_b(self, tmp_path, monkeypatch, stream):
        """Drive ``dispatch_task`` against a runner whose stream overflows.

        Returns ``(outcome, spawn_count)``.
        """
        spawns = []

        def fake_spawn(**kwargs):
            spawns.append(kwargs)
            return executor._Capture(
                stdout=stream, stderr="", exit_code=1, duration=0.01,
                argv=["codex", "exec"],
            )

        monkeypatch.setattr(executor, "_spawn", fake_spawn)
        task = make_task(
            kind="validation",
            task_id="validation-batch-0",
            prompt="You are validating review suggestions. Print ONE JSON object.",
            output_name="validation_batch_0.json",
            inputs=build_inputs(grouped_file=str(tmp_path / "grouped.json")),
            phase_dir=str(tmp_path),
            extra={
                "batch_index": 0,
                "group_indices": [0],
                "group_ids": ["aaaa000000000000"],
                "groups_count": 1,
            },
        )
        outcome = executor.dispatch_task(
            task,
            phase_dir=str(tmp_path),
            runner="codex:gpt-5.5",
            retries=3,
            timeout=30,
            sleep=lambda _s: None,
        )
        return outcome, len(spawns)

    def test_lane_b_dispatches_exactly_once(self, tmp_path, monkeypatch):
        """retries=3 would allow four attempts if the status were retryable."""
        outcome, spawns = self._lane_b(
            tmp_path, monkeypatch, CONTEXT_OVERFLOW_STREAMS["openai"]
        )
        assert outcome.status == STATUS_CONTEXT_LENGTH_EXCEEDED
        assert spawns == 1
        assert len(outcome.attempts) == 1

    def test_lane_b_records_the_actionable_message(self, tmp_path, monkeypatch):
        outcome, _ = self._lane_b(
            tmp_path, monkeypatch, CONTEXT_OVERFLOW_STREAMS["anthropic"]
        )
        recorded = json.dumps(outcome.to_dict())
        assert "narrow the plan/diff scope" in recorded
        assert "larger-context model" in recorded

    def test_the_batch_report_carries_the_status_verbatim(
        self, tmp_path, monkeypatch
    ):
        outcome, _ = self._lane_b(
            tmp_path, monkeypatch, CONTEXT_OVERFLOW_STREAMS["gemini"]
        )
        report = executor.BatchReport(
            phase="review-plan", kind="validation", phase_dir=str(tmp_path),
            outcomes=[outcome],
        )
        rendered = report.to_dict()
        assert rendered["counts"] == {STATUS_CONTEXT_LENGTH_EXCEEDED: 1}

    def test_lane_a_dispatches_exactly_once(self):
        """``run_single_model``'s `:653` returned early; `:664` could re-enter."""
        assert _lane_a_dispatches(CONTEXT_OVERFLOW_STREAMS["openai"]) == 1
        assert _lane_a_dispatches(CONTEXT_OVERFLOW_STREAMS["anthropic"]) == 1

    def test_lane_a_reports_the_actionable_message(self):
        result = _lane_a_result(CONTEXT_OVERFLOW_STREAMS["openai"])
        assert "narrow the plan/diff scope" in str(result.get("error"))
        assert "larger-context model" in str(result.get("error"))


# ---------------------------------------------------------------------------
# §7.2 — launch failures, ordered root-cause-before-auth
# ---------------------------------------------------------------------------

class TestLaunchFailureMarkers:

    def test_exit_127_is_a_missing_binary(self):
        assert pbase.classify_streams(
            "", "sh: 1: cursor-agent: not found", exit_code=127
        ) == STATUS_SPAWN_FAILED

    def test_exit_127_wins_even_with_an_empty_stream(self):
        """127 is the shell's own verdict; no text is needed to read it."""
        assert pbase.classify_streams("", "", exit_code=127) == STATUS_SPAWN_FAILED

    @pytest.mark.parametrize("stream", [
        "bash: codex: command not found",
        "exec: \"opencode\": executable file not found in $PATH",
        "'goose' is not recognized as an internal or external command",
        "bash: /usr/local/bin/aider: cannot execute binary file",
    ])
    def test_the_launch_markers_are_recognized(self, stream):
        assert pbase.classify_streams("", stream, exit_code=1) == STATUS_SPAWN_FAILED

    @pytest.mark.parametrize("stream", [
        # omp 18.2.2: a Bun trace naming the error class.
        "CliUsageError: Unknown tool in --tools: bogus. Valid tools: read, write.",
        # omp 18.8.3 (tests/fixtures/omp/invalid-tool, exit 2): no class name,
        # no trace, and the plural for two names.
        "Error: Unknown tool in --tools: bogus.\n"
        "Built-in tools: read, bash, edit, grep, glob, write.\n"
        "Run `omp --help` for available flags.\n",
        "Error: Unknown tools in --tools: bogus1, bogus2.\n",
        # Both releases, unchanged wording (exit 1).
        "error: Config overlay not found: /work/assets/omp/overlay.yml",
    ])
    def test_an_omp_argv_refusal_is_a_launch_failure(self, stream):
        """A rejected ``--tools`` name or a missing ``--config`` overlay is a
        property of the install, identical for every sibling slot: rung 1."""
        assert pbase.classify_streams("", stream, exit_code=2) == STATUS_SPAWN_FAILED

    @pytest.mark.parametrize("quoted", [
        # A reviewer reading this repo's omp fixtures, quoting the refusal in
        # its JSON answer (the newline escaped inside a string) ...
        '[{"title": "omp prints `Error: Unknown tool in --tools: bogus.` '
        'on stderr", "severity": "low"}]',
        # ... and in plain text, the line on its own in a fenced block.
        "omp rejects an unknown name before it resolves a model:\n```\n"
        "Error: Unknown tool in --tools: bogus.\n```\nso the run is dead at "
        "launch.",
    ])
    def test_a_reviewer_quoting_the_omp_refusal_on_stdout_is_no_launch_failure(
        self, quoted
    ):
        """Review 24, pins #5: rung 1 fails the whole batch, so the omp row is
        read where omp writes it — stderr, at a line start — and never off a
        model's answer on stdout that quotes the same words."""
        assert pbase.classify_streams(quoted, "", exit_code=1) != (
            STATUS_SPAWN_FAILED
        )
        match = pbase.match_streams(quoted, "", 1)
        assert match is None or match.status != STATUS_SPAWN_FAILED

    def test_omps_refusal_mid_line_on_stderr_is_no_launch_failure_either(self):
        """The measured shape opens its line; prose quoting it on stderr does not."""
        stream = "warning: the plan says Error: Unknown tool in --tools: bogus."
        assert pbase.classify_streams("", stream, exit_code=1) != (
            STATUS_SPAWN_FAILED
        )

    def test_lane_as_composed_sentence_still_reads_the_omp_refusal(self):
        """Lane A classifies its own ``"<cli> exited with code N: <describe_failure>"``
        string, handed over as stdout (``review_orchestrator_base``); omp's
        ``describe_failure`` quotes the stderr line into it, so the refusal is
        read there too, right after that prefix and nowhere else."""
        from utils.providers.omp import OmpProvider

        fixtures = Path(__file__).parent / "fixtures" / "omp"
        stdout = (fixtures / "invalid-tool.stdout.txt").read_text(encoding="utf-8")
        stderr = (fixtures / "invalid-tool.stderr.txt").read_text(encoding="utf-8")
        reason = OmpProvider().describe_failure(stdout, stderr)
        composed = f"omp exited with code 2: {reason}"
        assert pbase.classify_streams(composed, "") == STATUS_SPAWN_FAILED
        assert pbase.match_streams(composed, "").status == STATUS_SPAWN_FAILED
        # The prefix is llm_client's own composition, pinned so the two move
        # together.
        client = (
            Path(__file__).parent.parent / "utils" / "llm_client.py"
        ).read_text(encoding="utf-8")
        assert (
            'error_message = f"{provider_name} exited with code '
            '{result.returncode}"' in client
        )
        # Another runner's sentence that only quotes it later is not omp's.
        quoted = (
            "claude-code exited with code 1: the reviewer wrote that omp says "
            "Error: Unknown tool in --tools: bogus."
        )
        assert pbase.classify_streams(quoted, "") != STATUS_SPAWN_FAILED

    def test_a_root_refusal_is_read_as_the_root_cause_not_as_auth(self):
        """§7.2's ordering rule: a root refusal reads like a permission problem.

        The launch tuple is consulted **before** the auth patterns, so a sandbox
        that refused the exec is not laundered into a batch-fatal auth failure
        just because the runner also printed a login hint afterwards.
        """
        stream = (
            "docker: permission denied while trying to connect to the Docker "
            "daemon socket\nhint: run `login` first"
        )
        assert pbase.classify_streams("", stream, exit_code=1) == (
            pbase.CATEGORY_PERMISSION_DENIED
        )

    def test_a_model_not_found_is_not_a_missing_binary(self):
        """`:434`'s bare `'not found' in stderr` called this a missing CLI."""
        assert pbase.classify_streams(
            "", "Error: model not found: gpt-5.5-turbo", exit_code=1
        ) == pbase.CATEGORY_INVALID_MODEL


# ---------------------------------------------------------------------------
# §7.2 — the CLIs whose exit code carries no verdict
# ---------------------------------------------------------------------------

class TestExitZeroFailuresClassifyAsFailures:
    """goose, aider, cursor-agent and cline all exit 0 on a provider failure."""

    @pytest.mark.parametrize("name,stream,expected", [
        ("goose", GOOSE_EXIT_ZERO_AUTH, STATUS_AUTH_FAILED),
        ("aider", AIDER_EXIT_ZERO_AUTH, STATUS_AUTH_FAILED),
        ("cursor-agent", CURSOR_EXIT_ZERO_AUTH, STATUS_AUTH_FAILED),
        ("cline", CLINE_EXIT_ZERO_AUTH, STATUS_AUTH_FAILED),
    ])
    def test_a_measured_exit_zero_auth_failure_is_auth_failed(
        self, name, stream, expected
    ):
        provider = get_provider(name)
        assert provider.classify_failure(stream, "", 0) == expected

    @pytest.mark.parametrize("name,stream", [
        ("goose", GOOSE_EXIT_ZERO_OPAQUE),
        ("cursor-agent", CURSOR_ENVELOPE_IS_ERROR),
    ])
    def test_an_opaque_exit_zero_failure_still_classifies_as_a_failure(
        self, name, stream
    ):
        """No auth or rate-limit word at all — only the CLI's own marker."""
        provider = get_provider(name)
        assert provider.classify_failure(stream, "", 0) is not None

    def test_a_healthy_exit_zero_run_still_classifies_as_nothing(self):
        """The markers may not fire on the answer a working run produced."""
        provider = get_provider("goose")
        healthy = (
            '{"messages":[{"role":"assistant","content":[{"type":"text",'
            '"text":"[{\\"title\\":\\"Add a test\\"}]"}]}]}'
        )
        assert provider.classify_failure(healthy, "", 0) is None


class TestOpencodeAuthListExitsZeroWhenLoggedOut:
    """§5.3: `opencode auth list` exits 0 with no credentials at all."""

    def test_zero_credentials_is_not_authenticated(self):
        provider = get_provider("opencode")
        assert provider.classify_auth_probe(OPENCODE_AUTH_LIST_EMPTY, "", 0) is False

    def test_a_stored_credential_is_authenticated(self):
        provider = get_provider("opencode")
        assert provider.classify_auth_probe(
            OPENCODE_AUTH_LIST_POPULATED, "", 0
        ) is True

    def test_an_environment_key_alone_is_not_refused(self):
        """Refusing a run is expensive; an env key really does authenticate."""
        provider = get_provider("opencode")
        assert provider.classify_auth_probe(
            OPENCODE_AUTH_LIST_ENV_ONLY, "", 0
        ) is not False


#: `opencode auth list` on the pin (`auth_list_capture.py`: no TTY, stdin
#: /dev/null, scratch HOME). The strings above are the 1.18.12-era hand
#: transcriptions; this is the byte-exact envelope of the supported release,
#: ANSI and all, so the verdict is graded on both.
OPENCODE_AUTH_LIST = json.loads(oc_pin.AUTH_LIST.read_text(encoding="utf-8"))


class TestOpencodeAuthListOnThePin:
    """The pinned envelope: same count lines, with colour escapes around them.

    Two things it changes and one it does not. stdout now carries ``\\x1b[90m``
    before the path and each entry's type; stderr is exactly ``\\x1b[0m\\n`` on
    every run, logged in or not (NO_COLOR/TERM=dumb do not suppress it). The
    count line itself — the only verdict — is unchanged, so the classifier needs
    no new reading; these tests hold that true.
    """

    CASES = OPENCODE_AUTH_LIST["cases"]

    def _classify(self, case):
        run = self.CASES[case]
        return get_provider("opencode").classify_auth_probe(
            run["stdout"], run["stderr"], run["exit"]
        )

    def test_the_fixture_is_the_pinned_build(self):
        assert OPENCODE_AUTH_LIST["opencode_version"] == oc_pin.PIN
        for case, run in self.CASES.items():
            assert run["exit"] == 0, case
            assert run["stderr"] == "\x1b[0m\n", case

    def test_empty_home_is_not_authenticated(self):
        assert self._classify("empty") is False
        assert self._classify("empty_no_color") is False

    def test_a_stored_credential_is_authenticated(self):
        assert self._classify("stored") is True

    def test_an_environment_key_alone_is_not_refused(self):
        assert self._classify("env_only") is not False

    def test_a_config_declared_provider_is_invisible_to_auth_list(self):
        """Measured, and pinned so a change is noticed rather than inherited.

        A custom provider declared in ``opencode.json`` — even one carrying its
        own ``options.apiKey`` — is not listed by ``auth list``: the envelope is
        byte-identical to the empty home's, and this probe still reads it as
        logged out. What now **admits** such a provider is the next leg (plan
        §8, T8.3/T8.4): on this ``False`` verdict the executor asks
        ``OpenCodeProvider.config_credential_source``, which reads the
        dispatch's own resolved config through ``opencode debug config`` and
        records the auth state ``config`` — see
        ``TestConfigDeclaredProvidersOnThePin`` below. The container lane's
        stub runner still passes on its seeded stored credential in stage 1.
        """
        assert (
            self.CASES["config_provider_only"]["stdout"]
            == self.CASES["empty"]["stdout"]
        )
        assert self._classify("config_provider_only") is False


#: `opencode debug config` on the pin, one case per §8.2 verdict-table row
#: plus jsonc, project layer, `OPENCODE_CONFIG`, and the hermetic
#: content/`--pure` cases (plan §8 T8.1). Every `apiKey` is a canary.
OPENCODE_DEBUG_CONFIG = json.loads(oc_pin.DEBUG_CONFIG.read_text(encoding="utf-8"))
CANARY = "sk-canary-not-a-key-"


class TestConfigDeclaredProvidersOnThePin:
    """Plan §8 T8.7: `auth list` says `0 credentials`; the config leg decides.

    Every case drives ``subagent_executor._detect_auth`` end to end with the
    committed ``auth list`` envelope (``empty``) and the committed ``debug
    config`` capture served through the patched spawn seam. The non-hermetic
    ``agent-write`` profile is used for the verdict-table rows, because that is
    where the user's own resolved config is the dispatch's config.
    """

    DEBUG = OPENCODE_DEBUG_CONFIG["cases"]
    AUTH = OPENCODE_AUTH_LIST["cases"]

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        from utils.providers import opencode as oc

        self.oc = oc
        home = tmp_path / "user-config"
        (home / "opencode").mkdir(parents=True)
        self.user_config = home / "opencode"
        self.monkeypatch = monkeypatch
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
        monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
        monkeypatch.setenv("OPENCODE_TEST_MANAGED_CONFIG_DIR", str(tmp_path / "no-managed"))
        monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
        monkeypatch.delenv("OPENCODE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            oc, "_read_help_tokens", lambda binary: frozenset({oc.PURE_FLAG})
        )
        oc.reset_hermetic_state()
        executor.reset_auth_cache()
        self.probes = []
        self.reads = []
        self.auth_case = "empty"
        self.debug_stdout = None
        self.debug_code = 0

        def run_probe(argv, env=None):
            self.probes.append(list(argv))
            run = self.AUTH[self.auth_case]
            return subprocess.CompletedProcess(
                list(argv), run["exit"], run["stdout"], run["stderr"]
            )

        def spawn(argv, env, cwd, timeout):
            self.reads.append(list(argv))
            if self.debug_stdout is None:
                return None
            return subprocess.CompletedProcess(
                list(argv), self.debug_code, self.debug_stdout, ""
            )

        monkeypatch.setattr(executor, "_run_probe", run_probe)
        monkeypatch.setattr(oc, "_spawn_debug_config", spawn)
        yield
        oc.reset_hermetic_state()
        executor.reset_auth_cache()

    def _detect(self, model, profile="agent-write"):
        return executor._detect_auth(
            "opencode", model, key="driver.subagent_runner",
            spec=f"opencode:{model}", profile=profile, cwd="/repo",
        )

    def _use(self, case):
        """Serve the capture's stdout over the global files it was captured
        from: the pin masks every `apiKey` as `"***"`, so a built-in
        override's key is resolved from the file, in the child's env."""
        run = self.DEBUG[case]
        self.debug_stdout = run["stdout"]
        self.debug_code = run["exit"]
        for name, body in run["global_files"].items():
            text = body if isinstance(body, str) else json.dumps(body)
            (self.user_config / name).write_text(text, encoding="utf-8")
        for key, value in run["env"].items():
            if key.startswith("MY_TEST_"):
                self.monkeypatch.setenv(key, value)

    @pytest.mark.parametrize(
        "case, model",
        [
            ("custom_inline_key", "stub/stub-model"),
            ("custom_keyless", "keyless/m1"),
            ("custom_envkey_unset", "envkey/m2"),
            ("custom_envkey_set", "envkey/m2"),
            ("builtin_override_keyed", "openrouter/some/model"),
            ("builtin_override_envkey_set", "openrouter/some/model"),
            ("builtin_override_filekey", "openrouter/some/model"),
            ("jsonc_commented", "stub/stub-model"),
            ("project_layer", "stub/stub-model"),
            ("opencode_config_file", "stub/stub-model"),
        ],
    )
    def test_a_declared_row_is_admitted_as_config(self, case, model):
        self._use(case)
        assert self._detect(model) == "config"

    @pytest.mark.parametrize(
        "case", [
            "builtin_override_keyless", "nothing_declared", "project_layer_disabled",
            # T8.3 on the pin: declared (`"***"`), but the key does not resolve.
            "builtin_override_envkey_unset", "builtin_override_empty_key",
            "builtin_override_filekey_missing",
        ]
    )
    def test_a_keyless_or_absent_builtin_is_still_refused(self, case):
        self._use(case)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            self._detect("openrouter/some/model")
        assert excinfo.value.reason_code == "not_authenticated"

    @pytest.mark.parametrize(
        "stdout, code",
        [
            (None, 0),
            ("{not json " + CANARY + "1", 0),
            ('{"provider": ' + '{"stub": {"npm": "x"}}' + "}", 1),
            ('{"agent": {}}', 0),
        ],
        ids=["spawn-failure-or-timeout", "non-json", "exit-1", "missing-provider"],
    )
    def test_a_reader_failure_keeps_todays_refusal(self, stdout, code):
        self.debug_stdout, self.debug_code = stdout, code
        with pytest.raises(executor.RunnerRejected) as excinfo:
            self._detect("stub/stub-model")
        assert excinfo.value.reason_code == "not_authenticated"
        assert CANARY not in str(excinfo.value)

    def test_the_lazy_path_never_spawns_debug_config(self):
        self.auth_case = "stored"
        self._use("custom_inline_key")
        assert self._detect("stub/stub-model") == "ok"
        assert self._detect("stub/stub-model") == "ok"
        assert len(self.probes) == 1
        assert self.reads == []

    def test_a_config_providers_prefix_is_admitted_in_hermetic_mode(
        self, tmp_path, monkeypatch
    ):
        from utils import provider_registry as registry

        path = tmp_path / "cfg.yaml"
        path.write_text(
            "providers:\n  opencode:\n    config_providers:\n      stub:\n"
            "        npm: '@ai-sdk/openai-compatible'\n"
            "        options: {baseURL: 'http://127.0.0.1:9/v1'}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        registry._config = None
        registry._config_key = None
        self._use("hermetic_pure_config_content")
        assert self._detect("stub/stub-model", "agent-readonly") == "config"
        assert ["opencode", "--pure", "debug", "config"] in self.reads

    def test_the_hermetic_user_config_only_prefix_is_unresolvable_not_unauthenticated(self):
        from utils.provider_registry import get_provider as _get

        self._use("custom_inline_key")  # the user's own config declares `stub`
        # ...in a commented `opencode.jsonc`: the file read cannot answer, so
        # `_user_defines_provider` asks `debug config` (a plain `opencode.json`
        # would be answered from the file alone, with no spawn).
        jsonc = os.path.join(os.environ["XDG_CONFIG_HOME"], "opencode", "opencode.jsonc")
        with open(jsonc, "w", encoding="utf-8") as handle:
            handle.write("// served by debug config\n{}\n")
        provider = _get("opencode")
        with patch.object(provider, "is_available", lambda: True):
            with pytest.raises(executor.RunnerRejected) as excinfo:
                executor.preflight_runner(
                    "opencode:stub/stub-model", profile="agent-readonly",
                    key="driver.subagent_runner",
                )
        assert excinfo.value.reason_code == "unresolvable_model"
        assert "config_providers" in str(excinfo.value)
        assert self.probes == []

    @pytest.mark.parametrize("profile", ["agent-write", "hermetic-off"])
    def test_the_same_prefix_under_write_or_hermetic_off_is_admitted(
        self, profile, tmp_path, monkeypatch
    ):
        self._use("custom_inline_key")
        if profile == "hermetic-off":
            from utils import provider_registry as registry

            path = tmp_path / "cfg.yaml"
            path.write_text("providers:\n  opencode:\n    hermetic: off\n", encoding="utf-8")
            monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
            registry._config = None
            registry._config_key = None
            profile = "agent-readonly"
        assert self._detect("stub/stub-model", profile) == "config"

    def test_secret_hygiene_across_every_surface(self, capsys, tmp_path, monkeypatch):
        """No canary in a refusal, a doctor detail, a status row or a log line."""
        from utils import doctor
        from utils.provider_registry import get_provider as _get

        surfaces = []
        for case in self.DEBUG:
            executor.reset_auth_cache()
            self.oc.reset_hermetic_state()
            self._use(case)
            for model in ("stub/stub-model", "openrouter/some/model", "envkey/m2"):
                for profile in ("agent-write", "agent-readonly"):
                    try:
                        surfaces.append(self._detect(model, profile))
                    except executor.RunnerRejected as exc:
                        surfaces.append(str(exc))
            provider = _get("opencode")
            monkeypatch.setattr(
                doctor, "_run",
                lambda argv, **kw: (0, self.AUTH["empty"]["stdout"], self.AUTH["empty"]["stderr"]),
            )
            surfaces.append(json.dumps(doctor._probe_auth(provider, 5.0, str(tmp_path), None)))
            surfaces.append(repr(dict(executor._AUTH_CACHE)))
        captured = capsys.readouterr()
        surfaces += [captured.out, captured.err]
        for text in surfaces:
            assert CANARY not in text


# ---------------------------------------------------------------------------
# §7.2 — per-adapter overrides for the four structured-error adapters
# ---------------------------------------------------------------------------

class TestPerAdapterPatternOverrides:
    """"Currently **zero** adapters override the base English lists" — no longer."""

    @pytest.mark.parametrize("name", ["codex", "opencode", "cline", "muse"])
    def test_the_structured_error_adapters_override_something(self, name):
        provider = get_provider(name)
        base_auth = set(pbase.LLMProvider.auth_error_patterns)
        base_rate = set(pbase.LLMProvider.rate_limit_patterns)
        extra = (set(provider.auth_error_patterns) - base_auth) | (
            set(provider.rate_limit_patterns) - base_rate
        )
        assert extra, f"{name} declares no override of its own"

    @pytest.mark.parametrize("name", ["codex", "opencode", "cline", "muse"])
    def test_an_override_extends_the_base_set_and_never_replaces_it(self, name):
        provider = get_provider(name)
        assert set(pbase.LLMProvider.auth_error_patterns) <= set(
            provider.auth_error_patterns
        )
        assert set(pbase.LLMProvider.rate_limit_patterns) <= set(
            provider.rate_limit_patterns
        )

    def test_codex_reads_its_own_401_shape(self):
        assert get_provider("codex").classify_failure(
            CODEX_TURN_FAILED_401, "", 1
        ) == STATUS_AUTH_FAILED

    def test_codex_reads_its_own_retry_limit_terminal(self):
        assert get_provider("codex").classify_failure(
            CODEX_RETRY_LIMIT_429, "", 1
        ) == STATUS_RATE_LIMITED

    def test_opencode_reads_its_own_error_name(self):
        assert get_provider("opencode").classify_failure(
            OPENCODE_PROVIDER_AUTH_ERROR, "", 1
        ) == STATUS_AUTH_FAILED

    def test_cline_reads_its_own_run_result(self):
        assert get_provider("cline").classify_failure(
            CLINE_EXIT_ZERO_AUTH, "", 0
        ) == STATUS_AUTH_FAILED

    def test_the_base_pattern_sets_were_not_edited(self):
        """§7.1.a: the adapter list is measured clean and stays byte-identical."""
        assert pbase.LLMProvider.auth_error_patterns == (
            "not logged in",
            "please log in",
            "please login",
            "authentication required",
            "unauthorized",
            "invalid api key",
            "api key is missing",
            "no credentials",
            "credentials not found",
            "run `login`",
        )
        assert pbase.LLMProvider.rate_limit_patterns == (
            "429",
            "rate limit",
            "rate_limit",
            "ratelimit",
            "too many requests",
            "quota exceeded",
            "quota_exceeded",
            "insufficient_quota",
            "overloaded",
        )


# ---------------------------------------------------------------------------
# §7.1.a — `_format_failure_details`' elif chain becomes a category lookup
# ---------------------------------------------------------------------------

class TestReasonLineIsACategoryLookup:
    """The `Reason:` line and the retry decision read the same classifier."""

    def _reason(self, returncode, stdout="", stderr=""):
        from utils.review_orchestrator_base import _format_failure_details

        return _format_failure_details(returncode, stdout, stderr)

    def test_a_missing_binary_no_longer_hard_codes_cursor_agent(self):
        """`:434` named "cursor-agent" as the missing CLI for every provider."""
        reason = self._reason(127, stderr="sh: 1: goose: not found")
        assert "cursor-agent" not in reason

    @pytest.mark.parametrize("stderr,expected", [
        ("bash: codex: command not found", "could not be started"),
        ("Error: permission denied", "Permission denied"),
        ("HTTP status 429 returned", "Rate limited"),
        ("Error: model not found: gpt-5.5", "Model not found or invalid"),
        ("HTTP status 503 service unavailable", "API/Server error"),
        ("the request timed out after 600s", "Timed out"),
        ("Error: connection refused by the API host", "Connection"),
    ])
    def test_each_category_renders_its_own_reason(self, stderr, expected):
        assert expected in self._reason(1, stderr=stderr)

    def test_a_context_overflow_gets_the_actionable_reason(self):
        reason = self._reason(1, stderr=CONTEXT_OVERFLOW_STREAMS["openai"])
        assert "context window" in reason

    def test_an_unrecognized_failure_is_still_unknown(self):
        assert "Unknown" in self._reason(1, stderr="something went sideways")

    def test_the_reason_line_is_still_boundary_aware(self):
        """L2's guarantee, re-asserted through the shared classifier."""
        prose = (
            "Failed to parse JSON output from cursor-agent. The module under "
            "review spans 429 lines and imports nothing unusual.\n"
        )
        assert "Rate limited" not in self._reason(1, stdout=prose)


# ---------------------------------------------------------------------------
# §14.2 — the grep-style guard. Red today at six sites.
# ---------------------------------------------------------------------------

#: The four tokens §14.2 names. A classifier spelled as a raw Python substring
#: test on any of these is a duplicate of ``providers/base.py``'s declared
#: pattern sets, and duplicates drift: the whole L2 defect was ``'429' in
#: str(error)`` disagreeing with a list that had already been made safe.
BANNED_CLASSIFIER_PREFIXES = ("429", "rate limit", "rate-limit", "ratelimit",
                              "context", "connection")

#: The one module allowed to declare them: it owns the boundary-aware matcher
#: (``stream_pattern``/``matches_any``) that makes them safe.
GUARD_EXEMPT = {"utils/providers/base.py"}


def _runtime_modules():
    for path in sorted(SKILL_DIR.rglob("*.py")):
        rel = path.relative_to(SKILL_DIR).as_posix()
        if rel.startswith(("tests/", "legacy/", ".venv/", "build/")):
            continue
        if rel in GUARD_EXEMPT:
            continue
        yield rel, path


def _banned(text):
    lowered = (text or "").lower()
    return any(lowered.startswith(prefix) for prefix in BANNED_CLASSIFIER_PREFIXES)


def _substring_classifier_sites(rel, tree):
    """Every raw ``<banned literal> in <haystack>`` test in one module.

    Two spellings count, and only two:

    * a literal on the left of an ``in`` comparison — ``'rate limit' in out``
    * a list/tuple of literals consumed by a raw ``any(p in hay for p in NAME)``
      comprehension, which is the same test wearing a loop

    A tuple handed to ``matches_any`` is **not** a substring test — that is the
    shared boundary-aware matcher, and routing through it is the fix, not the
    defect. So ``subagent_executor``'s ``NETWORK_DENIAL_PATTERNS`` and Lane A's
    ``_LANE_A_RATE_LIMIT_MARKERS`` are correctly invisible here.
    """
    literal_lists = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        items = [
            e.value for e in node.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if not items:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                literal_lists[target.id] = (node.lineno, items)

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            continue
        left = node.left
        if isinstance(left, ast.Constant) and isinstance(left.value, str):
            if _banned(left.value):
                sites.append((rel, node.lineno, repr(left.value)))
        elif isinstance(left, ast.Name):
            # `any(pattern in error_lower for pattern in <NAME>)`
            for comp in ast.walk(tree):
                if not isinstance(comp, (ast.GeneratorExp, ast.ListComp)):
                    continue
                if comp.elt is not node:
                    continue
                for generator in comp.generators:
                    iterated = generator.iter
                    if isinstance(iterated, ast.Name):
                        found = literal_lists.get(iterated.id)
                        if found and any(_banned(v) for v in found[1]):
                            sites.append((rel, found[0], iterated.id))
    return sites


class TestNoDuplicateSubstringClassifiers:
    """§14.2's grep-style guard, over the AST rather than over grep output."""

    def _sites(self):
        found = []
        for rel, path in _runtime_modules():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            found.extend(_substring_classifier_sites(rel, tree))
        return sorted(set(found))

    def test_no_raw_substring_classifier_survives_outside_base_py(self):
        sites = self._sites()
        rendered = "\n".join(f"  {rel}:{line}  {what}" for rel, line, what in sites)
        assert not sites, (
            "raw substring classifiers still live outside "
            "utils/providers/base.py's declared pattern sets:\n" + rendered
        )

    def test_the_exempt_module_really_does_declare_them(self):
        """The guard must be exempting a real owner, not an empty file."""
        assert "429" in pbase.LLMProvider.rate_limit_patterns
        assert "rate limit" in pbase.LLMProvider.rate_limit_patterns


# ---------------------------------------------------------------------------
# §7.1.a — validation.py delegates, and its persisted contract does not move
# ---------------------------------------------------------------------------

class TestValidationErrorDelegation:
    """``_classify_validation_error`` keeps its vocabulary and loses its body."""

    def _classify(self, error, http_status=None):
        from utils.validation import _classify_validation_error

        return _classify_validation_error(error, http_status)

    def test_the_http_fast_path_is_still_first(self):
        from utils.validation import ERROR_TYPE_RATE_LIMITED

        assert self._classify("Request timeout", 429) == ERROR_TYPE_RATE_LIMITED

    @pytest.mark.parametrize("error,expected", [
        ("Rate limit exceeded", "rate_limited"),
        ("Request was throttled", "rate_limited"),
        ("Connection timed out", "timeout"),
        ("deadline exceeded", "timeout"),
        ("Connection refused by server", "model_failure"),
        ("Service unavailable", "model_failure"),
        ("Binary not found in PATH", "model_failure"),
        ("Error: unauthorized", "model_failure"),
    ])
    def test_the_four_transport_categories_map_onto_the_kept_vocabulary(
        self, error, expected
    ):
        assert self._classify(error) == expected

    def test_the_parse_half_stays_local(self):
        from utils.validation import ERROR_TYPE_PARSING

        assert self._classify("No valid JSON in response") == ERROR_TYPE_PARSING
        assert self._classify("Expecting value: line 1 column 1") == ERROR_TYPE_PARSING

    def test_an_error_naming_a_json_path_is_not_a_parsing_error(self):
        """The bare ``"json"`` pattern bought a wasted revalidation pass.

        An error that merely *names* the output file classified as
        ``parsing_error`` — hence ``recoverable`` — so the phase spent a whole
        revalidation dispatch on a condition no re-ask can fix.
        """
        from utils.validation import ERROR_TYPE_PARSING, RECOVERABLE_ERROR_TYPES

        for error in (
            "Output file /tmp/plan/review-plan/plan_review_codex.json was never written",
            "Permission denied writing validation_batch_0.json",
        ):
            classified = self._classify(error)
            assert classified != ERROR_TYPE_PARSING, error
            assert classified not in RECOVERABLE_ERROR_TYPES, error

    def test_the_persisted_contract_did_not_move(self):
        """`error_type`/`recoverable` are written into every group result."""
        from utils.validation import (
            ERROR_TYPE_AMBIGUOUS,
            ERROR_TYPE_MODEL_FAILURE,
            ERROR_TYPE_PARSING,
            ERROR_TYPE_RATE_LIMITED,
            ERROR_TYPE_TIMEOUT,
            ERROR_TYPE_UNKNOWN,
            RECOVERABLE_ERROR_TYPES,
        )

        assert (ERROR_TYPE_PARSING, ERROR_TYPE_TIMEOUT, ERROR_TYPE_RATE_LIMITED,
                ERROR_TYPE_AMBIGUOUS, ERROR_TYPE_MODEL_FAILURE,
                ERROR_TYPE_UNKNOWN) == (
            "parsing_error", "timeout", "rate_limited", "real_ambiguity",
            "model_failure", "unknown")
        assert RECOVERABLE_ERROR_TYPES == frozenset(
            {"parsing_error", "timeout", "rate_limited"}
        )


# ---------------------------------------------------------------------------
# Lane A harness
# ---------------------------------------------------------------------------

def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


def _lane_a_run(error_text):
    """Run ``run_single_model`` against a provider that always fails."""
    from utils import review_orchestrator_base as rob

    calls = []

    def _fake_invoke(**kwargs):
        calls.append(kwargs)
        return {
            "success": False,
            "error": error_text,
            "details": {"stderr": error_text},
        }

    with tempfile.TemporaryDirectory() as tmp:
        plan_path = str(Path(tmp) / "plan.md")
        Path(plan_path).write_text("plan", encoding="utf-8")
        with patch.object(rob, "invoke_with_file_output", _fake_invoke), \
             patch.object(rob, "get_project_root", lambda _p: tmp), \
             patch.object(rob, "get_provider_timeout", lambda *a, **k: 30), \
             patch.object(rob.asyncio, "sleep", _no_sleep):
            result = asyncio.run(
                rob.run_single_model(
                    model_spec="cursor-agent:auto",
                    plan_path=plan_path,
                    prompt_template="{prefix}{output_json_path}",
                    prompt_context={},
                    phase_name="review-plan",
                    invoke_phase="plan_review",
                    timeout=30,
                    out_dir=tmp,
                    prefix="plan",
                )
            )
    return result, len(calls)


def _lane_a_dispatches(error_text):
    return _lane_a_run(error_text)[1]


def _lane_a_result(error_text):
    return _lane_a_run(error_text)[0]


class TestLaneANetworkRetryReadsTheSharedClassifier:
    """`:664`'s `'connection' in error or 'network' in error` was a second rule."""

    def test_a_real_connection_error_still_gets_its_one_retry(self):
        assert _lane_a_dispatches("Error: connection refused by api.openai.com") == 2

    def test_prose_naming_a_connection_is_not_a_transport_failure(self):
        """A review answer discussing a connection pool is not a network error."""
        assert _lane_a_dispatches(
            "Failed to parse JSON. The reviewer noted that the database "
            "connection pool in db.py is never closed on the error path."
        ) == 1


# ---------------------------------------------------------------------------
# D6 — the classifier has a word for a sandbox that cannot start
# ---------------------------------------------------------------------------

class TestSandboxMarkers:
    """`bwrap:` lines are the machine's fault, fatal, and never `permission_denied`."""

    BWRAP_LINES = [
        "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
        "bwrap: setting up uid map: Permission denied",
        "error: bubblewrap failed to start the sandbox",
    ]

    @pytest.mark.parametrize("stream", BWRAP_LINES)
    def test_every_measured_line_is_the_sandbox_category(self, stream):
        from utils.providers import base as pbase

        assert pbase.classify_streams("", stream, exit_code=1) == (
            pbase.CATEGORY_SANDBOX_UNAVAILABLE
        )
        # Wherever the line sits — this haystack is `stdout\nstderr`.
        assert pbase.classify_streams("event: started\n", stream + "\n", exit_code=1) == (
            pbase.CATEGORY_SANDBOX_UNAVAILABLE
        )

    def test_the_category_is_the_ladder_status_at_rung_1(self):
        from utils import review_orchestrator_base as rob
        from utils.providers import base as pbase
        from utils.retry_ladder import (
            FATAL_STATUSES, RUNG_BY_STATUS, STATUS_SANDBOX_UNAVAILABLE,
        )

        category = pbase.CATEGORY_SANDBOX_UNAVAILABLE
        assert category == STATUS_SANDBOX_UNAVAILABLE
        assert rob._LADDER_STATUS_BY_CATEGORY.get(category, category) in FATAL_STATUSES
        assert RUNG_BY_STATUS[STATUS_SANDBOX_UNAVAILABLE] == 1
        assert category not in pbase.RETRYABLE_FAILURE_CATEGORIES

    def test_the_reason_line_names_doctor(self):
        from utils.review_orchestrator_base import _format_failure_details

        reason = _format_failure_details(1, "", self.BWRAP_LINES[0])
        assert "sandbox" in reason
        assert "multi_llm.py doctor" in reason
        assert "Permission denied" not in reason.split("\n")[0]

    def test_a_bwrap_word_inside_prose_is_not_a_sandbox_failure(self):
        """`^bwrap: ` is anchored to a line start; a review quoting the word
        `bwrap` mid-sentence stays whatever the rest of the stream says."""
        from utils.providers import base as pbase

        assert pbase.classify_streams(
            "", "the plan mentions bwrap: as a sandbox option", exit_code=1
        ) is None
