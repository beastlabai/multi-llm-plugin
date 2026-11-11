"""``dispatch_task``'s typed steps, one at a time (EXEC-9).

Each step of the attempt loop in ``utils/task_dispatch.py`` takes the plan,
the hooks, the carried state, the ladder and the outcome it reads or
changes, so one rung is reached here without driving the whole loop: a
hand-built ``_Capture`` stands in for a process, a provider stand-in for an
adapter, and a ladder on a fake clock for the deadline.

What the whole loop does with them is pinned by the executor's own tests
(``test_subagent_executor.py``, ``test_write_executor.py`` and the rest);
these pin the contracts the steps share: the plan's refusals, the state's
resets, each hook's key set, and when the observer sees an attempt's usage.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import dispatch_spawn, task_dispatch, task_outcome, task_result  # noqa: E402
from utils.retry_ladder import (  # noqa: E402
    REPAIR_INCOMPLETE_REPORT,
    REPAIR_SCHEMA_INVALID,
    RESEND_NOTHING_CHANGED,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_REJECTED,
    STATUS_RUNNER_ERROR,
    STATUS_SCHEMA_INVALID,
    STATUS_SPAWN_FAILED,
    STATUS_TIMEOUT,
    STATUS_UNPARSEABLE,
    STOP_BUDGET,
    RetryLadder,
)
from utils.subagent_prompts import build_schema_repair_prompt  # noqa: E402
from utils.subagent_tasks import build_inputs, make_task  # noqa: E402

RUNNER = "codex:gpt-5.5"
JAIL_KEYS = ("sandbox_rung", "jail_denials", "jail_evidence")


def _validation_task(phase_dir: Path, prompt: str = "Validate. Print ONE JSON object."):
    return make_task(
        kind="validation",
        task_id="validation-batch-0",
        prompt=prompt,
        output_name="validation_batch_0.json",
        inputs=build_inputs(grouped_file=str(phase_dir / "grouped.json")),
        phase_dir=str(phase_dir),
        extra={"batch_index": 0, "group_indices": [0],
               "group_ids": ["aaaa000000000000"], "groups_count": 1},
    )


class _Provider:
    """An adapter stand-in: usage from a fixed dict, its stream as the reply."""

    name = "stand-in"

    def __init__(self, usage=None):
        self.usage = usage

    def parse_usage(self, stdout, stderr):
        return self.usage

    def salvage_text(self, stdout):
        return stdout


def _plan(tmp_path, *, kind="task_generation", provider=None, **overrides):
    """A plan built by hand: no adapter lookup, no preflight, a stdin
    transport (so no argv budget applies)."""
    fields = dict(
        task={"task_id": "t-1", "kind": kind, "expected_ids": []},
        task_id="t-1", kind=kind,
        profile=("agent-write" if kind in task_dispatch.WRITE_KINDS else "agent-readonly"),
        runner="stand-in:m", anchor=None, cwd=str(tmp_path), phase_dir=str(tmp_path),
        provider=provider or _Provider(), provider_name="stand-in", model="m",
        transport="stdin", fallback=None, wrapping=None, jail_request=None,
        write_consent=None, read_dirs=[], target=tmp_path / "out.json",
        prompt="THE TASK PROMPT", prefer_arrays=False, salvage_failure=None,
        effective_timeout=600, retries=1, schema=None, schema_phase=None,
        declined_marker=None,
    )
    fields.update(overrides)
    return task_dispatch.DispatchPlan(**fields)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _ladder(clock=None, **kwargs):
    kwargs.setdefault("timeout", 600)
    kwargs.setdefault("retries", 1)
    return RetryLadder(now=clock or _Clock(), jitter=lambda: 0.0, **kwargs)


def _hooks(**kwargs):
    kwargs.setdefault("sleep", lambda delay: None)
    return task_dispatch.DispatchHooks(**kwargs)


def _outcome(plan):
    return task_outcome.TaskOutcome(
        task_id=plan.task_id, kind=plan.kind, runner=plan.runner,
        status=STATUS_REJECTED, profile=plan.profile,
    )


def _reading(status, *, error="what went wrong", data=None, usage=None, stdout="", **capture):
    capture.setdefault("exit_code", 1)
    capture.setdefault("argv", ["stand-in", "run"])
    return task_dispatch.AttemptReading(
        index=1, capture=dispatch_spawn._Capture(stdout=stdout, **capture),
        status=status, error=error, data=data, usage=usage,
    )


# ---------------------------------------------------------------------------
# plan_dispatch: a plan, or a refusal with nothing spawned
# ---------------------------------------------------------------------------


class TestPlanDispatch:
    def test_a_runner_with_no_adapter_is_spawn_failed_with_no_target(self, tmp_path):
        refused = task_dispatch.plan_dispatch(
            _validation_task(tmp_path), phase_dir=str(tmp_path), runner="no-such-cli:m",
        )
        assert refused == task_dispatch.Refusal(
            "runner 'no-such-cli:m' has no built-in adapter", status=STATUS_SPAWN_FAILED,
        )

    def test_a_refusal_after_the_target_keeps_it(self, tmp_path):
        refused = task_dispatch.plan_dispatch(
            _validation_task(tmp_path), phase_dir=str(tmp_path), runner=RUNNER, prompt="",
        )
        assert isinstance(refused, task_dispatch.Refusal)
        assert refused.status == STATUS_REJECTED
        assert refused.reason == "Task validation-batch-0: carries no prompt"
        assert refused.output_path == str(tmp_path / "validation_batch_0.json")

    def test_dispatch_task_records_a_refusal_as_the_plan_said(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dispatch_spawn, "_spawn", _must_not_spawn)
        outcome = task_dispatch.dispatch_task(
            _validation_task(tmp_path, prompt="   "), phase_dir=str(tmp_path), runner=RUNNER,
        )
        assert (outcome.status, outcome.reason, outcome.attempts) == (
            STATUS_REJECTED, "Task validation-batch-0: carries no prompt", [])
        assert outcome.output_path == str(tmp_path / "validation_batch_0.json")

    def test_a_plan_carries_what_every_attempt_sends(self, tmp_path):
        plan = task_dispatch.plan_dispatch(
            _validation_task(tmp_path), phase_dir=str(tmp_path), runner=RUNNER,
            cwd=str(tmp_path), retries=2, timeout=300,
        )
        assert isinstance(plan, task_dispatch.DispatchPlan)
        assert (plan.kind, plan.profile, plan.provider_name, plan.model) == (
            "validation", "agent-readonly", "codex", "gpt-5.5")
        assert (plan.retries, plan.effective_timeout, plan.schema_phase) == (2, 300, None)
        assert plan.prompt == "Validate. Print ONE JSON object."
        assert str(plan.target) == str(tmp_path / "validation_batch_0.json")
        assert plan.budget_kwargs == {}


def _must_not_spawn(**_kwargs):
    raise AssertionError("nothing may be spawned")


# ---------------------------------------------------------------------------
# AttemptState: each rule's resets in one place
# ---------------------------------------------------------------------------


class TestAttemptState:
    def _dirty(self):
        state = task_dispatch.AttemptState(prompt="first")
        state.start_report_reask(
            "restate", {"a.py": "1"}, repair_check={"passed": True},
            restate_check=({"findings": []}, ["x"]), fallback={"k": 1},
        )
        return state

    def test_a_resend_is_a_first_dispatch_again(self):
        state = self._dirty()
        state.start_resend("again")
        assert (state.prompt, state.resend, state.repair) == (
            "again", RESEND_NOTHING_CHANGED, None)
        assert (state.repair_baseline, state.repair_check, state.restate_check) == (
            None, None, None)
        assert state.reask_fallback == {"k": 1}, "the fallback outlives the re-send"
        assert state.repair_tag == RESEND_NOTHING_CHANGED

    def test_a_report_reask_drops_the_resend_mark(self):
        state = task_dispatch.AttemptState(prompt="first")
        state.start_resend("again")
        state.start_report_reask(
            "restate", "baseline", repair_check=None, restate_check=None, fallback={},
        )
        assert (state.prompt, state.repair, state.resend) == (
            "restate", REPAIR_INCOMPLETE_REPORT, None)
        assert state.repair_baseline == "baseline"

    def test_a_schema_repair_keeps_the_resend_and_a_read_kind_takes_no_guard(self):
        state = task_dispatch.AttemptState(prompt="first")
        state.start_resend("again")
        state.start_schema_repair("repair")
        assert (state.prompt, state.repair, state.resend) == (
            "repair", REPAIR_SCHEMA_INVALID, RESEND_NOTHING_CHANGED)
        assert state.repair_tag == REPAIR_SCHEMA_INVALID
        assert (state.repair_baseline, state.repair_check) == (None, None)

    def test_a_write_kinds_schema_repair_records_its_guard(self):
        state = task_dispatch.AttemptState(prompt="first")
        state.start_schema_repair("repair", ({"a.py": "1"}, {"passed": False}))
        assert (state.repair_baseline, state.repair_check) == ({"a.py": "1"}, {"passed": False})

    def test_a_refused_resend_keeps_its_note(self):
        state = task_dispatch.AttemptState(prompt="first")
        state.refuse_resend("no re-send: the budget")
        assert state.resend_note == "no re-send: the budget"
        assert state.resend is None


# ---------------------------------------------------------------------------
# hook_context: every hook's key set, exactly as before the steps
# ---------------------------------------------------------------------------


#: Each hook's keys as the six dict literals in ``dispatch_task`` had them.
EXPECTED_KEYS = {
    "tree_fingerprint": ("task", "task_id", "kind", "cwd", "outcome"),
    "before_attempt": ("task", "task_id", "kind", "profile", "runner", "provider",
                       "model", "cwd", "attempt", "prompt", "repair", "transport",
                       "outcome"),
    "reply": ("task", "task_id", "kind", "profile", "runner", "cwd", "attempt",
              "prompt", "result", "exit_code", *JAIL_KEYS),
    "result": ("task", "task_id", "kind", "profile", "runner", "cwd", "attempt",
               "prompt", "result", "exit_code", "stderr_tail", "usage",
               "duration_seconds", "outcome", "resent", "resend_note", *JAIL_KEYS),
    "changed_files": ("task", "task_id", "kind", "attempt", "cwd", "outcome", *JAIL_KEYS),
    "reask_fallback": ("task", "task_id", "kind", "profile", "runner", "cwd", "attempt",
                       "prompt", "result", "exit_code", "stderr_tail", "usage",
                       "resent", "resend_note", *JAIL_KEYS),
}

#: The keys ``after_result`` receives for the report a re-ask could not
#: restate, in order: the remembered context, then the two ``close_outcome``
#: appends, as before the dispatch was split into steps.
FALLBACK_AFTER_RESULT_KEYS = (*EXPECTED_KEYS["reask_fallback"], "duration_seconds", "outcome")


class TestHookContext:
    @pytest.mark.parametrize("shape", list(EXPECTED_KEYS))
    def test_each_shape_has_its_keys_in_order(self, tmp_path, shape):
        plan = _plan(tmp_path)
        state = task_dispatch.AttemptState(prompt="sent")
        context = task_dispatch.hook_context(
            shape, plan, _outcome(plan), state=state,
            reading=_reading(STATUS_OK, data={"tasks": []}, usage={"input_tokens": 3}),
            attempt=None if shape != "before_attempt" else 2,
        )
        assert tuple(context) == EXPECTED_KEYS[shape]

    def test_the_values_are_read_now(self, tmp_path):
        plan = _plan(tmp_path)
        outcome = _outcome(plan)
        outcome.duration_seconds = 4.5
        state = task_dispatch.AttemptState(prompt="sent")
        state.start_resend("resent prompt")
        state.refuse_resend("no re-send: x")
        reading = _reading(STATUS_OK, data={"tasks": []}, usage={"output_tokens": 2},
                           stderr="tail", exit_code=0)
        context = task_dispatch.hook_context("result", plan, outcome, state=state, reading=reading)
        assert context["prompt"] == "resent prompt"
        assert (context["resent"], context["resend_note"]) == (True, "no re-send: x")
        assert (context["attempt"], context["exit_code"], context["result"]) == (1, 0, {"tasks": []})
        assert context["usage"] == {"output_tokens": 2}
        assert context["duration_seconds"] == 4.5 and context["outcome"] is outcome
        assert context["stderr_tail"] == "tail"
        assert context["sandbox_rung"] is None

    def test_the_stderr_tail_is_redacted_at_the_source(self, tmp_path, monkeypatch):
        from utils import failure_capture

        seen = []
        monkeypatch.setattr(failure_capture, "_redact_probe_tail",
                            lambda text, limit: seen.append((text, limit)) or "<redacted>")
        plan = _plan(tmp_path)
        context = task_dispatch.hook_context(
            "result", plan, _outcome(plan), state=task_dispatch.AttemptState(prompt="p"),
            reading=_reading(STATUS_OK, stderr="token=raw"),
        )
        assert context["stderr_tail"] == "<redacted>"
        assert seen == [("token=raw", 2000)]


# ---------------------------------------------------------------------------
# admit_attempt and spawn_attempt: the stops before a spawn
# ---------------------------------------------------------------------------


class _FakeLadder:
    def __init__(self, *, admitted, stop_reason=None, attempt_index=0, remaining=100.0):
        self.admitted = admitted
        self.stop_reason = stop_reason
        self.attempt_index = attempt_index
        self._remaining = remaining

    def remaining(self):
        return self._remaining

    def next_attempt(self):
        return self._remaining if self.admitted else None

    def elapsed(self):
        return 0.0


class TestAdmission:
    def test_an_admitted_attempt_is_none(self, tmp_path):
        assert task_dispatch.admit_attempt(_plan(tmp_path), _FakeLadder(admitted=True), None) is None

    def test_a_first_attempt_the_envelope_refuses_is_budget_exhausted(self, tmp_path):
        ladder = _ladder(timeout=60, envelope=SimpleNamespace(max_wall_s=120))
        stop = task_dispatch.admit_attempt(_plan(tmp_path, effective_timeout=60), ladder, None)
        assert (stop.action, stop.status) == (task_dispatch.NEXT_STOP, STATUS_BUDGET_EXHAUSTED)

    def test_a_grant_taken_back_keeps_the_attempt_that_ran(self, tmp_path):
        clock = _Clock()
        ladder = _ladder(clock, timeout=600, envelope=SimpleNamespace(max_wall_s=120))
        assert ladder.next_attempt() is not None
        clock.t += 500  # an overslept backoff: 100 s left, under the declared 120 s
        previous = task_dispatch.Next.retry(STATUS_RATE_LIMITED, "HTTP 429")
        stop = task_dispatch.admit_attempt(_plan(tmp_path), ladder, previous)
        assert ladder.stop_reason == STOP_BUDGET
        assert stop.status == STATUS_RATE_LIMITED
        assert stop.error.startswith("HTTP 429 (no further attempt: ")
        assert "attempt 2 was not started" in stop.error

    def test_a_deadline_stop_names_the_attempt_never_started(self, tmp_path):
        ladder = _FakeLadder(admitted=False, stop_reason="deadline", attempt_index=2)
        stop = task_dispatch.admit_attempt(_plan(tmp_path, effective_timeout=90), ladder, None)
        assert stop == task_dispatch.Next.stop(STATUS_TIMEOUT, (
            "the per-task wall-clock budget of 90s was exhausted after 2 attempt(s); "
            "attempt 3 was never started"))

    def test_a_budget_the_before_hook_drained_spawns_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dispatch_spawn, "_spawn", _must_not_spawn)
        plan = _plan(tmp_path, effective_timeout=30)
        stop = task_dispatch.spawn_attempt(
            plan, _hooks(), task_dispatch.AttemptState(prompt="p"),
            _FakeLadder(admitted=True, remaining=0.0), _outcome(plan), 1,
        )
        assert stop == task_dispatch.Next.stop(STATUS_TIMEOUT, (
            "the per-task wall-clock budget of 30s was exhausted after 0 attempt(s); "
            "attempt 1 was never started"))

    def test_a_before_hook_refusal_ends_the_dispatch_with_nothing_spawned(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(dispatch_spawn, "_spawn", _must_not_spawn)
        seen = []

        def before(context):
            seen.append(dict(context))
            raise task_outcome.ResultRejected("policy says no", status=STATUS_REJECTED)

        outcome = task_dispatch.dispatch_task(
            _validation_task(tmp_path), phase_dir=str(tmp_path), runner=RUNNER,
            before_attempt=before,
        )
        assert (outcome.status, outcome.reason, outcome.attempts) == (
            STATUS_REJECTED, "policy says no", [])
        assert tuple(seen[0]) == EXPECTED_KEYS["before_attempt"]
        assert (seen[0]["attempt"], seen[0]["repair"]) == (1, None)


# ---------------------------------------------------------------------------
# read_attempt: a hand-built capture onto its rung, no process
# ---------------------------------------------------------------------------


class TestReadAttempt:
    def _read(self, tmp_path, capture, *, state=None, hooks=None, usage=None):
        plan = _plan(tmp_path, provider=_Provider(usage))
        return task_dispatch.read_attempt(
            plan, hooks or _hooks(), state or task_dispatch.AttemptState(prompt="p"),
            _outcome(plan), capture, 3,
        )

    def test_a_launch_error_is_spawn_failed(self, tmp_path):
        reading = self._read(tmp_path, dispatch_spawn._Capture(launch_error="ENOENT"))
        assert (reading.index, reading.status) == (3, STATUS_SPAWN_FAILED)
        assert "could not start the runner: ENOENT" in reading.error

    def test_a_timeout_is_rung_2_with_its_usage(self, tmp_path):
        reading = self._read(tmp_path, dispatch_spawn._Capture(timed_out=True, exit_code=-9),
                             usage={"input_tokens": 7})
        assert reading.status == STATUS_TIMEOUT
        assert reading.usage == {"input_tokens": 7}

    def test_a_jail_denial_is_rejected_on_its_own_line(self, tmp_path):
        line = "[JAIL_DENIED] codex: a write the jail stopped"
        reading = self._read(tmp_path, dispatch_spawn._Capture(exit_code=1, jail_denied_line=line))
        assert (reading.status, reading.error) == (STATUS_REJECTED, line)

    def test_a_report_reask_that_moved_a_file_is_a_partial_application(self, tmp_path):
        state = task_dispatch.AttemptState(prompt="restate")
        state.start_report_reask("restate", {"a.py": "old"}, repair_check=None,
                                 restate_check=None, fallback={})
        hooks = _hooks(tree_fingerprint=lambda context: {"a.py": "new"})
        reading = self._read(tmp_path, dispatch_spawn._Capture(exit_code=0, stdout="{}"),
                             state=state, hooks=hooks)
        assert reading.status == task_dispatch.STATUS_PARTIAL
        assert "the report re-ask (attempt 3) was asked only to restate" in reading.error
        assert "files changed while it ran (a.py)" in reading.error


# ---------------------------------------------------------------------------
# decide_next: the ladder's decision on a table of readings
# ---------------------------------------------------------------------------


class TestDecideNext:
    def _decide(self, tmp_path, reading, *, kind="task_generation", guard=None,
                hooks=None, **ladder):
        plan = _plan(tmp_path, kind=kind)
        state = task_dispatch.AttemptState(prompt=plan.prompt)
        ladder_obj = _ladder(**ladder)
        ladder_obj.next_attempt()
        step = task_dispatch.decide_next(
            plan, hooks or _hooks(), state, ladder_obj, _outcome(plan), reading,
            guard or task_dispatch.RetryGuard(),
        )
        return step, state, plan

    def test_a_schema_invalid_document_gets_the_repair_prompt(self, tmp_path, capsys):
        data = {"tasks": "not a list"}
        step, state, plan = self._decide(
            tmp_path, _reading(STATUS_SCHEMA_INVALID, data=data), schema_repairs=1,
        )
        assert step == task_dispatch.Next.retry(STATUS_SCHEMA_INVALID, "what went wrong")
        problems = task_result.validate_result("task_generation", data)
        assert state.prompt == build_schema_repair_prompt(
            "task_generation", plan.prompt, problems or ["what went wrong"], syntax_error=None,
        )
        assert state.repair == REPAIR_SCHEMA_INVALID
        assert "(schema repair, attempt 2)" in capsys.readouterr().out

    def test_without_a_document_rung_6_stays_terminal(self, tmp_path):
        step, state, _ = self._decide(
            tmp_path, _reading(STATUS_SCHEMA_INVALID, data=None), schema_repairs=1,
        )
        assert step == task_dispatch.Next.stop(STATUS_SCHEMA_INVALID, "what went wrong")
        assert state.repair is None

    def test_a_transport_failure_spends_a_retry(self, tmp_path):
        slept = []
        step, _, _ = self._decide(tmp_path, _reading(STATUS_RUNNER_ERROR), retries=1,
                                  hooks=_hooks(sleep=slept.append))
        assert step == task_dispatch.Next.retry(STATUS_RUNNER_ERROR, "what went wrong")
        assert slept == []

    def test_a_rate_limit_backs_off_once_before_the_retry(self, tmp_path):
        slept = []
        step, _, _ = self._decide(tmp_path, _reading(STATUS_RATE_LIMITED),
                                  hooks=_hooks(sleep=slept.append))
        assert step.action == task_dispatch.NEXT_RETRY
        assert len(slept) == 1 and slept[0] > 0

    def test_a_refused_report_reask_ends_as_the_partial_application(self, tmp_path):
        guard = task_dispatch.RetryGuard(repeatable=False, partial_if_refused="PARTIAL: a.py")
        step, _, _ = self._decide(tmp_path, _reading(STATUS_UNPARSEABLE), kind="implement",
                                  guard=guard, schema_repairs=0)
        assert (step.action, step.status) == (task_dispatch.NEXT_STOP, task_dispatch.STATUS_PARTIAL)
        assert step.error.startswith("PARTIAL: a.py")

    def test_a_write_reask_whose_baseline_cannot_be_read_is_not_sent(self, tmp_path):
        def broken(context):
            raise RuntimeError("no snapshot")

        guard = task_dispatch.RetryGuard(repeatable=False, partial_if_refused="PARTIAL: a.py")
        reading = replace(_reading(STATUS_UNPARSEABLE), reply_text="I fixed it.")
        step, state, _ = self._decide(
            tmp_path, reading, kind="implement",
            guard=guard, schema_repairs=1, hooks=_hooks(tree_fingerprint=broken),
        )
        assert step == task_dispatch.Next.stop(task_dispatch.STATUS_PARTIAL, (
            "PARTIAL: a.py (no report re-ask: the files' state before it could not be "
            "recorded, RuntimeError)"))
        assert state.repair is None


# ---------------------------------------------------------------------------
# The attempt rows: the observer and the usage count
# ---------------------------------------------------------------------------


class TestTheAttemptRecord:
    """The accepted attempt reaches the observer before its usage is counted,
    every other attempt after: the order the four copies had, kept by the one
    recording sequence (``_record_attempt``)."""

    def _observer(self, outcome, seen):
        return lambda attempt: seen.append((attempt.status, dict(outcome.usage or {})))

    def test_the_accepted_attempt_is_seen_before_its_usage_counts(self, tmp_path):
        plan = _plan(tmp_path)
        outcome = _outcome(plan)
        seen = []
        result = task_dispatch.finish_ok(
            plan, _hooks(after_attempt=self._observer(outcome, seen)),
            task_dispatch.AttemptState(prompt="p"), _ladder(), outcome,
            _reading(STATUS_OK, exit_code=0, data={"tasks": []}, usage={"input_tokens": 5}),
        )
        assert result is outcome and outcome.status == STATUS_OK
        assert seen == [(STATUS_OK, {})]
        assert outcome.usage == {"input_tokens": 5}

    def test_a_failed_attempt_is_seen_with_its_usage_counted(self, tmp_path):
        plan = _plan(tmp_path)
        outcome = _outcome(plan)
        seen = []
        reading = task_dispatch.record_failed_attempt(
            plan, _hooks(after_attempt=self._observer(outcome, seen)),
            task_dispatch.AttemptState(prompt="p"), outcome,
            _reading(STATUS_RUNNER_ERROR, usage={"input_tokens": 5}),
        )
        assert seen == [(STATUS_RUNNER_ERROR, {"input_tokens": 5})]
        row = outcome.attempts[0]
        assert (row.index, row.status, row.error) == (1, STATUS_RUNNER_ERROR, reading.error)
        assert row.capture_path and Path(row.capture_path).is_file()

    def test_an_ordinary_permission_error_says_it_is_not_the_jails(self, tmp_path):
        plan = _plan(tmp_path)
        reading = task_dispatch.record_failed_attempt(
            plan, _hooks(), task_dispatch.AttemptState(prompt="p"), _outcome(plan),
            _reading(STATUS_RUNNER_ERROR, jail_evidence={
                "class": task_dispatch.landlock_dispatch.JAIL_CLASS_ORDINARY,
                "reason": "inside an allow entry"}),
        )
        assert reading.error == "what went wrong (not a jail denial: inside an allow entry)"


# ---------------------------------------------------------------------------
# close_outcome: the report a re-ask could not restate
# ---------------------------------------------------------------------------


class TestCloseOutcome:
    def test_the_first_report_is_classified_with_the_result_keys(self, tmp_path):
        plan = _plan(tmp_path, kind="apply_suggestions")
        outcome = _outcome(plan)
        state = task_dispatch.AttemptState(prompt="the batch")
        fallback = task_dispatch.hook_context(
            "reask_fallback", plan, outcome, state=state,
            reading=_reading(STATUS_OK, exit_code=0, data={"ids": ["1"]}),
        )
        state.start_report_reask("restate", {}, repair_check=None, restate_check=None,
                                 fallback=fallback)
        seen = []
        closed = task_dispatch.close_outcome(
            plan, _hooks(after_result=seen.append), state, _ladder(), outcome,
            task_dispatch.Next.stop(STATUS_TIMEOUT, "the re-ask timed out"),
        )
        assert closed.status == STATUS_OK
        assert closed.reason == "(its report re-ask did not restate the report: the re-ask timed out)"
        assert tuple(seen[0]) == FALLBACK_AFTER_RESULT_KEYS
        assert set(seen[0]) == set(EXPECTED_KEYS["result"])
        assert seen[0]["prompt"] == "the batch" and seen[0]["outcome"] is outcome

    def test_a_partial_application_is_never_classified_from_the_fallback(self, tmp_path):
        plan = _plan(tmp_path, kind="apply_suggestions")
        state = task_dispatch.AttemptState(prompt="the batch")
        state.start_report_reask("restate", {}, repair_check=None, restate_check=None,
                                 fallback={"result": {}})
        seen = []
        closed = task_dispatch.close_outcome(
            plan, _hooks(after_result=seen.append), state, _ladder(), _outcome(plan),
            task_dispatch.Next.stop(task_dispatch.STATUS_PARTIAL, "PARTIAL: moved"),
        )
        assert (closed.status, closed.reason, seen) == (task_dispatch.STATUS_PARTIAL,
                                                        "PARTIAL: moved", [])

    def test_a_failed_repair_says_it_came_after_the_reask(self, tmp_path):
        plan = _plan(tmp_path)
        state = task_dispatch.AttemptState(prompt="p")
        state.start_schema_repair("repair")
        closed = task_dispatch.close_outcome(
            plan, _hooks(), state, _ladder(), _outcome(plan),
            task_dispatch.Next.stop(STATUS_SCHEMA_INVALID, "bad shape"),
        )
        assert closed.reason == "bad shape (after the one schema-repair re-ask)"
