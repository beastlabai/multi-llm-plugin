"""§5 — rate-limit backpressure narrows on the *attempt*, not on the task.

Before this, ``_gather``'s task wrapper narrowed a throttled provider only once
``dispatch_task`` had returned, retries and backoff sleeps included: on run
``b38de8bb`` the narrowing landed about fifteen minutes after the attempt that
earned it, and every sibling that acquired a slot in between ran at the old
width. Two halves are pinned here:

* **T5.1** — ``dispatch_task(after_attempt=…)`` hands each recorded
  :class:`Attempt` to the caller the moment it is appended, before the ladder
  decides and before any backoff sleep; a raising observer never fails the
  attempt it observes.
* **T5.2** — the wrapper's hook fires on the ``asyncio.to_thread`` worker, so it
  submits ``narrow()`` to the loop with ``run_coroutine_threadsafe`` and never
  blocks on it: the narrowing is visible to a task acquiring during the
  throttled task's *next* attempt, it happens at most once per task, never on a
  non-rate-limited attempt, and a ``narrow()`` that raises is logged and leaves
  the task's attempts and outcome exactly as they were. The hook is passed only
  when it can matter and only to a ``dispatch_task`` that accepts it (§F6).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.subagent_tasks import build_inputs, make_task  # noqa: E402


RUNNER = "codex:gpt-5.5"
PROVIDER = "codex"


# ---------------------------------------------------------------------------
# T5.1 — dispatch_task's after_attempt seam
# ---------------------------------------------------------------------------

def _validation_task(phase_dir: Path):
    return make_task(
        kind="validation",
        task_id="validation-batch-0",
        prompt="You are validating review suggestions. Print ONE JSON object.",
        output_name="validation_batch_0.json",
        inputs=build_inputs(grouped_file=str(phase_dir / "grouped.json")),
        phase_dir=str(phase_dir),
        extra={
            "batch_index": 0,
            "group_indices": [0],
            "group_ids": ["aaaa000000000000"],
            "groups_count": 1,
        },
    )


@pytest.fixture
def rate_limited_runner(monkeypatch):
    """A runner whose every attempt is classified ``rate_limited``.

    The classifier is stubbed rather than fed a 429 stream: what is under test
    is *when* the attempt reaches the observer, not which bytes earn the
    verdict (§2/§3 own that).
    """
    spawns = []

    def fake_spawn(**kwargs):
        spawns.append(kwargs)
        return executor._Capture(
            stdout="", stderr="", exit_code=1, duration=0.01,
            argv=["codex", "exec"],
        )

    monkeypatch.setattr(executor, "_spawn", fake_spawn)
    monkeypatch.setattr(
        executor, "_classify_failed_run",
        lambda provider, capture, model: (
            executor.STATUS_RATE_LIMITED, "HTTP 429 Too Many Requests"
        ),
    )
    return spawns


class TestAfterAttemptSeam:

    def test_each_attempt_reaches_the_observer_before_its_backoff_sleep(
        self, tmp_path, rate_limited_runner
    ):
        events = []
        outcome = executor.dispatch_task(
            _validation_task(tmp_path),
            phase_dir=str(tmp_path),
            runner=RUNNER,
            retries=0,
            timeout=600,
            sleep=lambda delay: events.append("sleep"),
            after_attempt=lambda attempt: events.append(
                (attempt.index, attempt.status)
            ),
        )
        assert outcome.status == executor.STATUS_RATE_LIMITED
        attempts = len(outcome.attempts)
        assert attempts > 1, "rate limiting backs off and retries"
        # Every attempt is announced, in order, and each backoff sleep comes
        # AFTER the attempt that caused it — never before.
        expected = []
        for record in outcome.attempts:
            expected.append((record.index, executor.STATUS_RATE_LIMITED))
            expected.append("sleep")
        assert events == expected[:-1], "the last attempt is not followed by a sleep"

    def test_the_observer_sees_the_recorded_attempt_object(
        self, tmp_path, rate_limited_runner
    ):
        seen = []
        outcome = executor.dispatch_task(
            _validation_task(tmp_path),
            phase_dir=str(tmp_path),
            runner=RUNNER,
            retries=0,
            timeout=600,
            sleep=lambda _s: None,
            after_attempt=seen.append,
        )
        assert seen == outcome.attempts
        assert all(a is b for a, b in zip(seen, outcome.attempts))

    def test_a_successful_attempt_is_announced_too(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            executor, "_spawn",
            lambda **k: executor._Capture(
                stdout="{}", stderr="", exit_code=0, duration=0.01,
                argv=["codex", "exec"],
            ),
        )
        monkeypatch.setattr(
            executor, "_parse_with_schema_policy",
            lambda *a, **k: (executor.STATUS_OK, None, {"ok": True}, None),
        )
        monkeypatch.setattr(
            executor, "persist_result",
            lambda task, phase_dir, data: str(tmp_path / "validation_batch_0.json"),
        )
        seen = []
        outcome = executor.dispatch_task(
            _validation_task(tmp_path),
            phase_dir=str(tmp_path),
            runner=RUNNER,
            retries=0,
            timeout=600,
            sleep=lambda _s: None,
            after_attempt=seen.append,
        )
        assert outcome.status == executor.STATUS_OK
        assert [a.status for a in seen] == [executor.STATUS_OK]

    def test_a_raising_observer_never_fails_the_dispatch(
        self, tmp_path, rate_limited_runner, capsys
    ):
        def boom(_attempt):
            raise RuntimeError("observer exploded")

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        baseline = executor.dispatch_task(
            _validation_task(tmp_path / "a"),
            phase_dir=str(tmp_path / "a"),
            runner=RUNNER, retries=0, timeout=600, sleep=lambda _s: None,
        )
        observed = executor.dispatch_task(
            _validation_task(tmp_path / "b"),
            phase_dir=str(tmp_path / "b"),
            runner=RUNNER, retries=0, timeout=600, sleep=lambda _s: None,
            after_attempt=boom,
        )
        assert baseline.status == executor.STATUS_RATE_LIMITED
        assert observed.status == baseline.status
        assert observed.reason == baseline.reason
        assert [a.status for a in observed.attempts] == [
            a.status for a in baseline.attempts
        ]
        err = capsys.readouterr().err
        assert "after_attempt callback failed" in err
        assert "observer exploded" in err


# ---------------------------------------------------------------------------
# T5.2 — the task wrapper narrows from the worker thread
# ---------------------------------------------------------------------------

def _attempt(index, status):
    return executor.Attempt(index=index, status=status, duration_seconds=0.0)


def _outcome(task, runner, attempts, status):
    return executor.TaskOutcome(
        task_id=task["task_id"], kind=task["kind"], runner=runner,
        status=status, attempts=list(attempts),
    )


def _gather(tmp_path, monkeypatch, tasks, *, limit):
    """Run ``_gather`` over ``tasks`` on one provider seeded at ``limit`` slots."""
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    monkeypatch.setattr(
        executor, "get_provider_max_concurrent", lambda *a, **k: limit
    )
    return asyncio.run(executor._gather(
        tasks,
        runners={task["task_id"]: RUNNER for task in tasks},
        phase_dir=str(tmp_path),
        anchor=None,
        cwd=None,
        max_parallel=limit,
        resume=False,
    ))


@pytest.fixture
def narrow_calls(monkeypatch):
    """Record every real ``narrow()`` call (and the limit it produced)."""
    calls = []
    original = executor._ProviderBackpressure.narrow

    async def recording(self, provider):
        result = await original(self, provider)
        calls.append((provider, result))
        return result

    monkeypatch.setattr(executor._ProviderBackpressure, "narrow", recording)
    return calls


class TestTheWrapperNarrowsOnTheAttempt:

    def test_a_task_acquiring_during_the_retry_sees_the_narrower_gate(
        self, tmp_path, monkeypatch, capsys
    ):
        """Seeded at 2 slots; A is throttled on attempt 1, then "retries".

        B holds the second slot until A's narrowing has landed, then leaves.
        C, queued behind both, may now take B's slot only if the gate is still
        2 wide. It is 1 wide — the narrowing already happened, while A is still
        mid-retry — so C starts only after A releases. Before §5 the narrowing
        waited for A to *return*, and C ran beside A's second attempt.
        """
        narrowed = threading.Event()
        a_done = threading.Event()
        c_saw_a_running = []
        original = executor._ProviderBackpressure.narrow

        async def signalling(self, provider):
            result = await original(self, provider)
            narrowed.set()
            return result

        monkeypatch.setattr(executor._ProviderBackpressure, "narrow", signalling)

        def fake_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None,
                          after_attempt=None):
            name = task["task_id"]
            if name == "A":
                first = _attempt(1, executor.STATUS_RATE_LIMITED)
                after_attempt(first)
                time.sleep(0.5)  # attempt 2: the backoff and the retry
                a_done.set()
                return _outcome(
                    task, runner,
                    [first, _attempt(2, executor.STATUS_OK)], executor.STATUS_OK,
                )
            if name == "B":
                assert narrowed.wait(5), "A's narrowing never landed"
                return _outcome(task, runner, [_attempt(1, executor.STATUS_OK)],
                                executor.STATUS_OK)
            c_saw_a_running.append(not a_done.is_set())
            return _outcome(task, runner, [_attempt(1, executor.STATUS_OK)],
                            executor.STATUS_OK)

        monkeypatch.setattr(executor, "dispatch_task", fake_dispatch)
        tasks = [{"task_id": t, "kind": "validation"} for t in ("A", "B", "C")]
        results = _gather(tmp_path, monkeypatch, tasks, limit=2)

        assert {tid: o.status for tid, o in results.items()} == {
            "A": executor.STATUS_OK, "B": executor.STATUS_OK, "C": executor.STATUS_OK,
        }
        assert c_saw_a_running == [False], (
            "C acquired a slot while A was still retrying — the gate was not "
            "narrowed until A returned"
        )
        out = capsys.readouterr().out
        assert (
            f"[exec-subagents] {PROVIDER} reported a rate limit (A, attempt 1) — "
            f"narrowing it to 1 in-flight slot(s) for the rest of the batch"
        ) in out
        assert out.count("reported a rate limit") == 1, (
            "the post-return check must skip a task that already narrowed"
        )

    def test_one_narrowing_per_task(self, tmp_path, monkeypatch, narrow_calls):
        def fake_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None,
                          after_attempt=None):
            attempts = [
                _attempt(i, executor.STATUS_RATE_LIMITED) for i in (1, 2, 3)
            ]
            for attempt in attempts:
                after_attempt(attempt)
            return _outcome(task, runner, attempts, executor.STATUS_RATE_LIMITED)

        monkeypatch.setattr(executor, "dispatch_task", fake_dispatch)
        _gather(tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
                limit=8)
        # 8 → 4 once; three throttled attempts plus a rate_limited verdict would
        # have taken it to 1 if each of them narrowed.
        assert narrow_calls == [(PROVIDER, 4)]

    def test_no_narrowing_on_a_non_rate_limited_attempt(
        self, tmp_path, monkeypatch, narrow_calls
    ):
        def fake_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None,
                          after_attempt=None):
            attempts = [
                _attempt(1, executor.STATUS_TIMEOUT),
                _attempt(2, executor.STATUS_RUNNER_ERROR),
            ]
            for attempt in attempts:
                after_attempt(attempt)
            return _outcome(task, runner, attempts, executor.STATUS_RUNNER_ERROR)

        monkeypatch.setattr(executor, "dispatch_task", fake_dispatch)
        _gather(tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
                limit=4)
        assert narrow_calls == []

    def test_a_raising_narrow_is_logged_and_changes_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        async def broken(self, provider):
            raise RuntimeError("narrow blew up")

        monkeypatch.setattr(executor._ProviderBackpressure, "narrow", broken)
        attempts = [
            _attempt(1, executor.STATUS_RATE_LIMITED),
            _attempt(2, executor.STATUS_OK),
        ]

        def fake_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None,
                          after_attempt=None):
            after_attempt(attempts[0])
            time.sleep(0.05)  # let the loop run the failing narrow()
            after_attempt(attempts[1])
            return _outcome(task, runner, attempts, executor.STATUS_OK)

        monkeypatch.setattr(executor, "dispatch_task", fake_dispatch)
        results = _gather(
            tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
            limit=4,
        )
        outcome = results["A"]
        assert outcome.status == executor.STATUS_OK
        assert [(a.index, a.status) for a in outcome.attempts] == [
            (1, executor.STATUS_RATE_LIMITED), (2, executor.STATUS_OK),
        ]
        captured = capsys.readouterr()
        assert "narrow blew up" in captured.err
        assert "narrowing codex after a rate limit on A attempt 1 failed" in (
            captured.err
        )
        # The flag was set before submitting, so the post-return check did not
        # try again (a second failure would have raised out of the wrapper).
        assert "reported a rate limit" not in captured.out


class TestTheHookIsPassedOnlyWhenNeeded:
    """§F6: a stand-in ``dispatch_task`` is never handed a keyword it rejects."""

    def test_a_double_without_the_keyword_still_narrows_after_return(
        self, tmp_path, monkeypatch, narrow_calls
    ):
        def legacy_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None):
            return _outcome(
                task, runner, [_attempt(1, executor.STATUS_RATE_LIMITED)],
                executor.STATUS_RATE_LIMITED,
            )

        monkeypatch.setattr(executor, "dispatch_task", legacy_dispatch)
        _gather(tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
                limit=4)
        assert narrow_calls == [(PROVIDER, 2)]

    def test_a_provider_already_at_the_floor_gets_no_hook(
        self, tmp_path, monkeypatch
    ):
        received = []

        def dispatch(task, *, phase_dir, runner, anchor=None, cwd=None, **kwargs):
            received.append(sorted(kwargs))
            return _outcome(task, runner, [_attempt(1, executor.STATUS_OK)],
                            executor.STATUS_OK)

        monkeypatch.setattr(executor, "dispatch_task", dispatch)
        _gather(tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
                limit=1)
        assert received == [[]]

    def test_a_narrowable_provider_gets_the_hook(self, tmp_path, monkeypatch):
        received = []

        def dispatch(task, *, phase_dir, runner, anchor=None, cwd=None, **kwargs):
            received.append(sorted(kwargs))
            return _outcome(task, runner, [_attempt(1, executor.STATUS_OK)],
                            executor.STATUS_OK)

        monkeypatch.setattr(executor, "dispatch_task", dispatch)
        _gather(tmp_path, monkeypatch, [{"task_id": "A", "kind": "validation"}],
                limit=2)
        assert received == [["after_attempt"]]

    def test_the_real_dispatch_task_accepts_the_keyword(self):
        assert executor._accepts_keyword(executor.dispatch_task, "after_attempt")
