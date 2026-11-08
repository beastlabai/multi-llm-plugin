"""review-code's fan-out slot obeys the same retry ladder as review-plan's.

``code_review_orchestrator.run_single_review`` dispatches through
``review_orchestrator_base.dispatch_with_ladder``, the loop
``run_single_model`` uses, so a transient failure is retried, a fatal one is
not, a jail denial is never re-dispatched and a context overflow names its
remedy on both lanes. The parity table at the bottom drives each row through
both lanes and fails on any drift between them.

A slot that raises is recorded as that exception, never as a deadline loss.

See docs/development/delegation.md#read-and-analyze-executor.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
from utils import llm_client, progress_tracker, status_file  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CODEX_HOME_CHECK_MARKER,
    MODE_CHECK_MARKER,
    SANDBOX_BYPASSED_MARKER,
)
from utils.providers.codex import CodexProvider  # noqa: E402
from tests.support.gitrepo import git  # noqa: E402

#: Hermetic git (tests/support/gitrepo.py), returning the CompletedProcess.
_git = partial(git, text_out=False)

#: "No ``failure_category`` key at all": the error-line fallback an exit-0
#: failure takes. ``None`` is a verdict, so it cannot mark absence.
_ABSENT = object()

#: A secret-shaped value the redactor replaces (`sk-family`).
_SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"


def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


def _failing_result(error, category=_ABSENT, details=None):
    fake_details = {"stderr": error, **(details or {})}
    if category is not _ABSENT:
        fake_details["failure_category"] = category
    return {"success": False, "error": error, "details": fake_details}


def _scripted(results, calls):
    """A fake ``invoke_with_file_output`` playing one entry per attempt (the
    last repeats): a result dict is returned, an exception raised, and a
    callable called with the attempt's kwargs for its side effect first."""
    def _fake_invoke(**kwargs):
        calls.append(kwargs)
        entry = results[min(len(calls), len(results)) - 1]
        if isinstance(entry, BaseException):
            raise entry
        if callable(entry):
            return entry(kwargs)
        return entry
    return _fake_invoke


def _run_review_code(
    tmp_path,
    *,
    results,
    model_spec="cursor-agent:auto",
    timeout=30,
    usage=None,
):
    """Drive ``run_single_review`` against a scripted sequence of attempts.

    ``results`` is one fake ``invoke_with_file_output`` return per attempt;
    the last one repeats once the list runs out (see :func:`_scripted`).
    """
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# plan\n", encoding="utf-8")
    calls = []
    _fake_invoke = _scripted(results, calls)

    with patch.object(cro, "invoke_with_file_output", _fake_invoke), \
         patch.object(cro, "get_project_root", lambda _p: str(tmp_path)), \
         patch.object(rob.asyncio, "sleep", _no_sleep):
        outcome = asyncio.run(cro.run_single_review(
            model_spec, plan_path, tmp_path / "out", timeout=timeout,
            usage=usage, run_model_specs=[model_spec],
        ))
    return outcome, calls


def _run_review_plan(tmp_path, *, results, model_spec="cursor-agent:auto", timeout=30):
    """The same scripted attempts through ``run_single_model`` (review-plan)."""
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# plan\n", encoding="utf-8")
    calls = []
    _fake_invoke = _scripted(results, calls)

    with patch.object(rob, "invoke_with_file_output", _fake_invoke), \
         patch.object(rob, "get_project_root", lambda _p: str(tmp_path)), \
         patch.object(rob, "get_provider_timeout", lambda *a, **k: timeout), \
         patch.object(rob.asyncio, "sleep", _no_sleep):
        outcome = asyncio.run(rob.run_single_model(
            model_spec=model_spec,
            plan_path=str(plan_path),
            prompt_template="{prefix}{output_json_path}",
            prompt_context={},
            phase_name="review-plan",
            invoke_phase="plan_review",
            timeout=timeout,
            out_dir=str(tmp_path),
            prefix="plan",
        ))
    return outcome, calls


# ---------------------------------------------------------------------------
# The ladder on review-code
# ---------------------------------------------------------------------------

class TestReviewCodeRetries:

    def test_a_connection_error_gets_one_transport_retry(self, tmp_path):
        (success, issues, error), calls = _run_review_code(
            tmp_path,
            results=[_failing_result(
                "cli exited with code 1: connection refused [connection_error]",
                category="connection_error",
            )],
        )
        assert len(calls) == 2, "rung 4: driver.subagent_retries=1"
        assert success is False and issues is None
        assert "connection refused" in error

    def test_a_retried_slot_returns_the_answer_of_its_second_attempt(self, tmp_path):
        answer = [{"title": "t", "importance": "low"}]
        usage = {}
        (success, issues, error), calls = _run_review_code(
            tmp_path,
            results=[
                _failing_result(
                    "cli exited with code 1 [server_error]", category="server_error",
                    details={"usage": {"input_tokens": 10, "cost_usd": 0.5}},
                ),
                {
                    "success": True, "data": answer, "source": "file",
                    "output_file": str(tmp_path / "out.json"),
                    "details": {"usage": {"input_tokens": 5, "cost_usd": 0.25}},
                },
            ],
            usage=usage,
        )
        assert len(calls) == 2
        assert (success, issues, error) == (True, answer, None)
        # What the slot spent is every attempt's, not the winning one's.
        assert usage["cursor-agent:auto"] == {"input_tokens": 15, "cost_usd": 0.75}
        # The second attempt is spawned with what is left of the one deadline.
        assert calls[1]["timeout"] <= calls[0]["timeout"]

    def test_the_result_baselines_are_taken_once_per_slot(self, tmp_path):
        _, calls = _run_review_code(
            tmp_path,
            results=[_failing_result("cli timed out after 30s", category="timeout")],
        )
        assert len(calls) == 2
        assert calls[0]["requested_baseline"] is calls[1]["requested_baseline"]
        assert calls[0]["misnamed_baseline"] is calls[1]["misnamed_baseline"]

    def test_a_rate_limit_backs_off_three_times(self, tmp_path):
        (_, _, error), calls = _run_review_code(
            tmp_path,
            results=[_failing_result(
                "cli exited with code 1: HTTP 429 [rate_limited]", category="rate_limited",
            )],
        )
        assert len(calls) == 4, "one dispatch plus MAX_RATE_LIMIT_BACKOFFS"
        # The attempt's own error leads; the ladder's reason follows it.
        assert error.startswith("cli exited with code 1: HTTP 429 [rate_limited]")
        assert "(Rate limited (429) - rate_limited after 3 backoffs)" in error

    def test_an_auth_failure_is_not_retried(self, tmp_path):
        (success, _, error), calls = _run_review_code(
            tmp_path,
            results=[_failing_result(
                "cli exited with code 1: HTTP 429 Too Many Requests [auth_failed]",
                category="auth_failed",
            )],
        )
        assert len(calls) == 1, "auth_failed is rung 1: never retried"
        assert success is False
        assert "[auth_failed]" in error

    def test_an_auth_failure_after_a_rate_limit_ends_the_slot(self, tmp_path):
        """The backoff a rate limit earns is granted, and the auth failure the
        retried attempt then meets stops the slot: no further backoff."""
        (_, _, error), calls = _run_review_code(
            tmp_path,
            results=[
                _failing_result("cli exited with code 1 [rate_limited]", category="rate_limited"),
                _failing_result("cli exited with code 1 [auth_failed]", category="auth_failed"),
            ],
        )
        assert len(calls) == 2
        assert "[auth_failed]" in error

    def test_a_jail_denial_wins_over_a_retryable_verdict(self, tmp_path):
        for category in ("rate_limited", None, "timeout"):
            _, calls = _run_review_code(
                tmp_path,
                results=[_failing_result(
                    "codex exited with code 1: [JAIL_DENIED] write outside the jail",
                    category=category,
                    details={"failure_reason": "jail-denied"},
                )],
            )
            assert len(calls) == 1, category

    def test_a_jail_denial_on_a_retried_attempt_ends_the_slot(self, tmp_path):
        """The first attempt's rate limit is retried; the retry the jail denies
        is not, although its verdict is a rate limit too."""
        _, calls = _run_review_code(
            tmp_path,
            results=[
                _failing_result("cli exited with code 1 [rate_limited]", category="rate_limited"),
                _failing_result(
                    "codex exited with code 1: [JAIL_DENIED] write outside the jail",
                    category="rate_limited",
                    details={"failure_reason": "jail-denied"},
                ),
            ],
        )
        assert len(calls) == 2

    def test_a_context_overflow_names_the_remedy(self, tmp_path):
        (_, _, error), calls = _run_review_code(
            tmp_path,
            results=[_failing_result(
                "cli exited with code 1 [context_length_exceeded]",
                category="context_length_exceeded",
            )],
        )
        assert len(calls) == 1
        assert "[context_length_exceeded]" not in error
        assert "context" in error.lower()

    def test_rule_6b_holds_for_claude_code(self, tmp_path):
        _, calls = _run_review_code(
            tmp_path,
            model_spec="claude-code:sonnet",
            timeout=1800,
            results=[_failing_result(
                "claude-code exited with code 1 [rate_limited]", category="rate_limited",
            )],
        )
        assert len(calls) == 1

    def test_a_rate_limit_keeps_the_adapters_own_sentence_on_both_lanes(
        self, tmp_path
    ):
        """claude-code tags an account usage cap `[USAGE_LIMIT]` so the reader
        is told the fix is another model, not a wait; the ladder's reason is
        added after it, never in its place."""
        line = (
            "claude-code exited with code 1: [USAGE_LIMIT] HTTP 429: You've "
            "reached your Fable limit. [rate_limited]"
        )
        failing = [_failing_result(line, category="rate_limited")]
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        plan_outcome, _ = _run_review_plan(
            tmp_path / "a", results=failing, model_spec="claude-code:sonnet",
            timeout=1800,
        )
        (_, _, code_error), _ = _run_review_code(
            tmp_path / "b", results=failing, model_spec="claude-code:sonnet",
            timeout=1800,
        )
        for error in (plan_outcome["error"], code_error):
            assert error.startswith(line)
            assert "an identical re-dispatch would repeat a chain" in error

    def test_a_slot_too_short_for_the_declared_envelope_is_not_dispatched(
        self, tmp_path
    ):
        (success, _, error), calls = _run_review_code(
            tmp_path, model_spec="claude-code:sonnet", timeout=5,
            results=[_failing_result("never sent")],
        )
        assert calls == [], "§7.5 rule 4 is a pre-dispatch admission test"
        assert success is False
        assert error.startswith("budget_exhausted")


# ---------------------------------------------------------------------------
# A slot that raises
# ---------------------------------------------------------------------------

class TestASlotExceptionIsAnError:

    def _fanout(self, tmp_path, invoke):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto"]
        with patch.object(cro, "invoke_with_file_output", invoke), \
             patch.object(cro, "get_project_root", lambda _p: str(tmp_path)), \
             patch.object(rob, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(
                cro.run_all_reviews(specs, plan_path, tmp_path / "out", 30, 2)
            )
        return plan_path, specs, results

    def test_failed_models_names_the_exception_not_the_deadline(self, tmp_path):
        def explode(**_kwargs):
            raise RuntimeError(f"provider exploded with {_SECRET}")

        plan_path, specs, results = self._fanout(tmp_path, explode)

        success, issues, error = results[specs[0]]
        assert (success, issues) == (False, None)
        assert error.startswith("Exception: RuntimeError: provider exploded")
        assert _SECRET not in error, "the exception text is redacted"
        assert "<redacted:" in error

        # main()'s two writers of `failed_models`, in main()'s order.
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        failed_models = {}
        for spec, (ok, found, err) in results.items():
            if not cro.save_model_result("plan", spec, ok, found, err, phase_dir):
                failed_models[spec] = err
        rob.backfill_deadline_losses(specs, results, failed_models)
        assert failed_models == {specs[0]: error}
        assert progress_tracker.DEADLINE_ERROR not in failed_models.values()

        # The table names the same cause as the report.
        models = status_file.read_status(str(phase_dir))["progress"]["models"]
        assert models[specs[0]]["state"] == "failed"
        assert models[specs[0]]["source"] == "exception"
        assert models[specs[0]]["error"] == error


    def test_a_write_before_a_raising_retry_reaches_the_status_file(
        self, tmp_path, monkeypatch
    ):
        """The fan-out's record of a bypassed slot whose first attempt wrote
        and whose retry raised: failed by path, with the audit verdict."""
        repo = _repo(tmp_path)
        monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: _BYPASSED)
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        calls = []
        invoke = _scripted([_write_then_fail(repo), ValueError("parser exploded")], calls)

        plan_path, specs, results = self._fanout(repo, invoke)

        assert len(calls) == 2
        success, _issues, error = results[specs[0]]
        assert success is False
        assert error.startswith(SANDBOX_BYPASSED_MARKER) and "src/app.py" in error
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        entry = status_file.read_status(str(phase_dir))["progress"]["models"][specs[0]]
        assert entry["state"] == "failed"
        assert entry["error"] == error
        assert entry["readonly_audit"]["clean"] is False
        assert "attempt 2" in entry["readonly_audit"]["reason"]


# ---------------------------------------------------------------------------
# A retry that raises keeps what the attempts before it left
# ---------------------------------------------------------------------------

_LANES = ["plan", "code"]

#: What the first, failed attempt reports it spent.
_SPENT = {"cost_usd": 1.25, "output_tokens": 100}

_BYPASSED = rob.SandboxVerdict(
    rob.SANDBOX_DISPATCH_UNSANDBOXED, f"{SANDBOX_BYPASSED_MARKER} simulated",
    "unsandboxed",
)
_JAILED = rob.SandboxVerdict(
    rob.SANDBOX_DISPATCH_UNSANDBOXED, "[SANDBOX_INCIDENTAL] simulated",
    "plugin-landlock", incidental=True,
)


def _repo(tmp_path):
    """A committed repository holding the plan, a source file and a script."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# plan\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _connection_failure(details=None):
    """A retryable failure (rung 4) that reports :data:`_SPENT`."""
    return _failing_result(
        "cli exited with code 1: connection refused [connection_error]",
        category="connection_error",
        details={"usage": dict(_SPENT), **(details or {})},
    )


def _write_then_fail(repo):
    """An attempt that edits a tracked file, reports a clean stream audit and
    fails retryably."""
    def _attempt(_kwargs):
        (repo / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
        return _connection_failure(
            {"readonly_audit": {"clean": True, "reason": "", "actions": 1}}
        )
    return _attempt


def _run_lane(
    lane, tmp_path, monkeypatch, *, results, verdict=None,
    model_spec="cursor-agent:auto", timeout=30,
):
    """``(success, error, spent, calls)`` of one slot through review-plan's
    (``"plan"``) or review-code's (``"code"``) lane, the sandbox gate pinned
    to ``verdict`` when one is given."""
    if verdict is not None:
        monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: verdict)
    if lane == "plan":
        outcome, calls = _run_review_plan(
            tmp_path, results=results, model_spec=model_spec, timeout=timeout,
        )
        return outcome["success"], outcome["error"], outcome.get("usage"), calls
    usage = {}
    (success, _issues, error), calls = _run_review_code(
        tmp_path, results=results, usage=usage, model_spec=model_spec,
        timeout=timeout,
    )
    return success, error, usage.get(model_spec), calls


class TestARaisingRetryKeepsWhatTheAttemptsBeforeItLeft:
    """A dispatch that raises ends the slot as that exception, after the same
    write, mode and home checks a returned attempt gets, and with what every
    attempt before it spent."""

    @pytest.mark.parametrize("lane", _LANES)
    def test_the_spend_before_it_is_recorded(self, lane, tmp_path, monkeypatch):
        success, error, spent, calls = _run_lane(
            lane, tmp_path, monkeypatch,
            results=[_connection_failure(), ValueError("Exceeds the limit")],
        )
        assert len(calls) == 2, "the first failure was retried"
        assert success is False
        assert error.startswith("Exception: ValueError: Exceeds the limit"), error
        assert spent == _SPENT

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_write_before_it_fails_the_slot_by_path(
        self, lane, tmp_path, monkeypatch
    ):
        """The retry that raised read nothing back, so its audit is missing and
        the first attempt's clean audit cannot clear the window."""
        repo = _repo(tmp_path)
        success, error, spent, calls = _run_lane(
            lane, repo, monkeypatch, verdict=_BYPASSED,
            results=[_write_then_fail(repo), ValueError("parser exploded")],
        )
        assert len(calls) == 2
        assert success is False
        assert error.startswith(SANDBOX_BYPASSED_MARKER), error
        assert "src/app.py" in error
        assert spent == _SPENT

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    @pytest.mark.parametrize("lane", _LANES)
    def test_a_mode_flip_before_it_fails_the_jailed_slot(
        self, lane, tmp_path, monkeypatch
    ):
        repo = _repo(tmp_path)
        script = repo / "tests" / "run.sh"

        def chmod_then_fail(_kwargs):
            script.chmod(script.stat().st_mode | 0o111)
            return _connection_failure()

        success, error, spent, calls = _run_lane(
            lane, repo, monkeypatch, verdict=_JAILED,
            results=[chmod_then_fail, ValueError("parser exploded")],
        )
        assert len(calls) == 2
        assert success is False
        assert MODE_CHECK_MARKER in error and "tests/run.sh" in error, error
        assert spent == _SPENT

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_home_change_before_it_fails_the_jailed_slot(
        self, lane, tmp_path, monkeypatch
    ):
        repo = _repo(tmp_path)
        home = {
            "home": "$CODEX_HOME", "changed": ["config.toml"],
            "config_keys": ["model"], "other_live_children": 0,
        }
        success, error, spent, calls = _run_lane(
            lane, repo, monkeypatch, verdict=_JAILED,
            results=[_connection_failure({"home_check": home}), ValueError("x")],
        )
        assert len(calls) == 2
        assert success is False
        assert CODEX_HOME_CHECK_MARKER in error, error
        assert spent == _SPENT

    @pytest.mark.parametrize("raised", [asyncio.CancelledError, KeyboardInterrupt])
    def test_cancellation_propagates_and_records_nothing(self, raised):
        """`CancelledError` and `KeyboardInterrupt` are not a slot's ending:
        they leave the shared loop unchanged, so the fan-out's teardown, not
        the lane, retires the slot."""
        attempts = []

        async def dispatch(attempt_timeout):
            attempts.append(attempt_timeout)
            if len(attempts) == 1:
                return _connection_failure()
            raise raised()

        async def main():
            try:
                await rob.dispatch_with_ladder(
                    dispatch,
                    ladder=rob.slot_ladder("cursor-agent", 30, anchor=None),
                    model_spec="cursor-agent:auto",
                    display_name="cursor-agent:auto",
                    timeout=30,
                    start_line=lambda attempt, slice_s: f"attempt {attempt}",
                    jail_summary_key=None,
                )
            except raised:
                return "propagated"
            return "swallowed"

        with patch.object(rob.asyncio, "sleep", _no_sleep):
            assert asyncio.run(main()) == "propagated"
        assert len(attempts) == 2


class TestTheWriteRecordOfARaisingFirstAttemptNamesTheException:
    """A bypassed slot whose first attempt wrote and raised is failed by path
    with ``source: exception``, not ``unknown``: no attempt returned one."""

    def test_review_plan(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: _BYPASSED)

        def write_then_raise(_kwargs):
            (repo / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
            raise ValueError("parser exploded")

        outcome, _calls = _run_review_plan(repo, results=[write_then_raise])
        assert outcome["error"].startswith(SANDBOX_BYPASSED_MARKER)
        assert outcome["source"] == "exception"

    def test_review_code(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: _BYPASSED)

        def write_then_raise(**_kwargs):
            (repo / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
            raise ValueError("parser exploded")

        plan_path, specs, results = TestASlotExceptionIsAnError()._fanout(
            repo, write_then_raise
        )
        assert results[specs[0]][2].startswith(SANDBOX_BYPASSED_MARKER)
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        entry = status_file.read_status(str(phase_dir))["progress"]["models"][specs[0]]
        assert entry["source"] == "exception"


# ---------------------------------------------------------------------------
# A raising attempt on the plugin's jail still gets its home check
# ---------------------------------------------------------------------------

_CODEX = "codex:gpt-5.5"


def _codex_stream():
    """A finished codex turn (the real invoker parses it, unless faked)."""
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {
            "id": "item_1", "type": "agent_message", "text": "[]"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )) + "\n"


@pytest.fixture
def jailed_codex(tmp_path, tmp_path_factory, monkeypatch):
    """A committed repo, and outside it a ``$CODEX_HOME`` and the attempts'
    scratch base; the gate pinned to the plugin's jail; the codex spawn faked
    by ``child(cmd)`` (set per test, for its side effect on the home), and the
    real ``invoke_with_provider`` underneath each attempt, whose parse of the
    codex stream raises, as the parser in the Codex review's probe did."""
    repo = _repo(tmp_path)
    outside = tmp_path_factory.mktemp("outside")
    home = outside / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    slot_base = outside / "slot-tmp"
    slot_base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(slot_base))
    monkeypatch.setenv("HOME", str(outside))
    monkeypatch.setenv("CODEX_HOME", str(home))
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: "/usr/bin/codex" if name == "codex" else real_which(name, *a, **k),
    )
    state = {"repo": repo, "home": home, "child": lambda _cmd: None, "spawned": 0}

    def fake_cli(cmd, **_kwargs):
        state["spawned"] += 1
        state["child"](cmd)
        return subprocess.CompletedProcess(cmd, 0, _codex_stream(), "")

    def exploding_parse(self, stdout, stderr):
        raise ValueError("parser exploded")

    monkeypatch.setattr(llm_client, "_run_provider_cli", fake_cli)
    monkeypatch.setattr(CodexProvider, "parse_output", exploding_parse)

    def attempt(kwargs):
        return llm_client.invoke_with_provider(
            "prompt", kwargs["model_spec"], kwargs["timeout"], cwd=kwargs["cwd"],
            anchor=kwargs["anchor"], sandbox_fallback=kwargs.get("sandbox_fallback"),
        )

    state["attempt"] = attempt
    return state


def _run_jailed(lane, jailed_codex, monkeypatch):
    return _run_lane(
        lane, jailed_codex["repo"], monkeypatch, verdict=_JAILED,
        model_spec=_CODEX, timeout=3600, results=[jailed_codex["attempt"]],
    )


class TestARaisingJailedAttemptKeepsItsHomeCheck:
    """``$CODEX_HOME`` outlives the jail (see ``home_check_applies``): an
    attempt that raises returns no ``details["home_check"]``, so the ladder
    loop compares the home of every jailed child it armed itself, against
    the invoker's own before-snapshot, and fails closed when it cannot."""

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_home_write_then_a_raise_fails_with_the_home_line(
        self, lane, jailed_codex, monkeypatch, capsys
    ):
        def child(_cmd):
            (jailed_codex["home"] / "AGENTS.md").write_text("injected\n", encoding="utf-8")

        jailed_codex["child"] = child
        success, error, _spent, calls = _run_jailed(lane, jailed_codex, monkeypatch)
        assert len(calls) == 1 and jailed_codex["spawned"] == 1
        assert success is False
        assert error.startswith(CODEX_HOME_CHECK_MARKER), error
        assert "AGENTS.md" in error and "fails closed" in error
        assert "0 other codex children of this run were live" in error
        assert (jailed_codex["home"] / "AGENTS.md").exists(), "never reverted"

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_raise_that_left_the_home_alone_is_only_the_exception(
        self, lane, jailed_codex, monkeypatch, capsys
    ):
        success, error, _spent, calls = _run_jailed(lane, jailed_codex, monkeypatch)
        assert len(calls) == 1 and jailed_codex["spawned"] == 1
        assert success is False
        assert error.startswith("Exception: ValueError: parser exploded"), error
        assert CODEX_HOME_CHECK_MARKER not in error
        assert CODEX_HOME_CHECK_MARKER not in capsys.readouterr().out

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_home_that_cannot_be_compared_fails_closed(
        self, lane, jailed_codex, monkeypatch, capsys
    ):
        """The home gone after the child: nothing to compare against, so the
        home is unaudited and takes a change's verdict."""
        jailed_codex["child"] = lambda _cmd: shutil.rmtree(jailed_codex["home"])
        success, error, _spent, _calls = _run_jailed(lane, jailed_codex, monkeypatch)
        assert success is False
        assert error.startswith(CODEX_HOME_CHECK_MARKER), error
        assert "was not checked" in error and "fails closed" in error

    def test_a_returned_attempt_reads_nothing_back_in_the_loop(
        self, jailed_codex, monkeypatch
    ):
        """The normal path is the invoker's own check: the loop's read-back
        runs only for an attempt that raised."""
        monkeypatch.setattr(
            rob, "raised_attempt_home_checks",
            lambda _armed: pytest.fail("read back after a returned attempt"),
        )
        monkeypatch.setattr(CodexProvider, "parse_output", lambda self, out, err: {
            "success": True, "data": [],
        })
        success, error, _spent, calls = _run_jailed("code", jailed_codex, monkeypatch)
        assert len(calls) == 1 and jailed_codex["spawned"] == 1
        assert CODEX_HOME_CHECK_MARKER not in str(error)


# ---------------------------------------------------------------------------
# The slot's budget starts before its baselines are collected
# ---------------------------------------------------------------------------

class TestTheSlotBudgetStartsBeforeTheBaselines:
    """§7.5 rule 1: the deadline is taken before the slot's result baselines
    are collected, so time spent collecting them is charged to the slot."""

    @staticmethod
    def _slow_baselines(monkeypatch, lane, seconds):
        """An injected ladder clock, advanced by ``seconds`` while the lane
        collects its requested-result baseline."""
        clock = {"t": 0.0}
        real_ladder = rob.RetryLadder

        class Ladder(real_ladder):
            def __init__(self, *args, **kwargs):
                kwargs["now"] = lambda: clock["t"]
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(rob, "RetryLadder", Ladder)
        module = rob if lane == "plan" else cro
        real_baseline = module.requested_result_baseline

        def slow(*args, **kwargs):
            clock["t"] += seconds
            return real_baseline(*args, **kwargs)

        monkeypatch.setattr(module, "requested_result_baseline", slow)

    @pytest.mark.parametrize("lane", _LANES)
    def test_the_first_attempt_gets_what_the_baselines_left(
        self, lane, tmp_path, monkeypatch
    ):
        self._slow_baselines(monkeypatch, lane, 10.0)
        _, _, _, calls = _run_lane(
            lane, tmp_path, monkeypatch,
            results=[_failing_result("cli exited with code 1 [auth_failed]",
                                     category="auth_failed")],
        )
        assert [call["timeout"] for call in calls] == [20]

    @pytest.mark.parametrize("lane", _LANES)
    def test_a_slot_the_baselines_used_up_dispatches_nothing(
        self, lane, tmp_path, monkeypatch
    ):
        self._slow_baselines(monkeypatch, lane, 29.5)
        success, error, _, calls = _run_lane(
            lane, tmp_path, monkeypatch, results=[_failing_result("never sent")],
        )
        assert calls == [], "§7.5 rule 4 refuses before any spawn"
        assert success is False
        assert error.startswith("budget_exhausted"), error


# ---------------------------------------------------------------------------
# Parity: one scripted failure, the same number of dispatches on both lanes
# ---------------------------------------------------------------------------

_PARITY_ROWS = [
    ("connection_error", "cli exited with code 1 [connection_error]", {}),
    ("server_error", "cli exited with code 1 [server_error]", {}),
    ("runner_error", "cli exited with code 1 [runner_error]", {}),
    (None, "HTTP 429 Too Many Requests", {}),
    ("timeout", "cli timed out after 30s", {}),
    ("rate_limited", "cli exited with code 1 [rate_limited]", {}),
    ("auth_failed", "cli exited with code 1 [auth_failed]", {}),
    ("invalid_model", "cli exited with code 1 [invalid_model]", {}),
    ("spawn_failed", "cli exited with code 1 [spawn_failed]", {}),
    ("permission_denied", "cli exited with code 1 [permission_denied]", {}),
    ("context_length_exceeded", "cli exited with code 1 [context_length_exceeded]", {}),
    ("rate_limited", "[JAIL_DENIED] write outside the jail", {"failure_reason": "jail-denied"}),
    (_ABSENT, "HTTP 429 Too Many Requests", {}),
    (_ABSENT, "JSON parsing failed from both file and stdout", {}),
]


@pytest.mark.parametrize(
    "category,error,details", _PARITY_ROWS,
    ids=[f"{'absent' if c is _ABSENT else c}:{e[:24]}" for c, e, _d in _PARITY_ROWS],
)
def test_both_review_lanes_dispatch_a_failure_the_same_number_of_times(
    tmp_path, category, error, details
):
    failing = [_failing_result(error, category=category, details=details)]
    plan_dir, code_dir = tmp_path / "plan", tmp_path / "code"
    plan_dir.mkdir()
    code_dir.mkdir()
    plan_outcome, plan_calls = _run_review_plan(plan_dir, results=failing)
    (code_ok, _, code_error), code_calls = _run_review_code(code_dir, results=failing)
    assert len(code_calls) == len(plan_calls)
    assert code_ok is plan_outcome["success"] is False
    assert code_error.split(" (spent ")[0] == plan_outcome["error"]
