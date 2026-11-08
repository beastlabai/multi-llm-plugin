"""M9 phase 2 (§7.1.b, §7.5) — Lane A on the ladder, with the vendor envelope on.

Phase 1 extracted :class:`utils.retry_ladder.RetryLadder` out of Lane B with
Lane B's tests as the fixed point and deliberately passed ``envelope=None``:
its contract was "introduce no new status". ``tests/test_retry_ladder.py`` pins
the pure policy object against an injected clock. **This file pins the two
lanes** — the four §14.2 M9 acceptance checks, each red before the change:

* the per-slot budget is **monotonic** — Lane A's second attempt is spawned with
  ``remaining()``, never with a fresh full ``timeout`` (the recursion at
  ``review_orchestrator_base.py:753`` / ``:798`` reset it every time);
* §7.5 rule 4 — a slot whose remaining budget is below the adapter's *declared*
  ``max_wall_s`` is recorded ``budget_exhausted`` with **no** spawn, on both
  lanes, and the status travels into ``subagent_batch_report.json``;
* §7.5 rule 6(b) — a ``retries_rate_limit: true`` adapter (claude-code) that
  returns ``rate_limited`` gets **no** identical re-dispatch, while an adapter
  that declares no envelope keeps the harness backoff unchanged;
* a fake provider that 429s forever ends the slot inside ``resolved_timeout``
  **without** ``run_models_concurrent``'s outer ``asyncio.wait_for`` ever firing
  — the constraint of §7.1.b: that ``wait_for`` is a backstop, and when it fires
  it cancels the task without killing the child (§8), so it may never be M9's
  enforcement point.

It also pins what Lane A hands the ladder: on a non-zero exit, an exit 0
that left no answer or a deadline kill, the adapter's own verdict
(``details["failure_category"]``, set by ``invoke_with_provider``), with
``None`` taking one transport retry as on Lane B; a failure with no verdict (a
launch failure, an adapter whose ``classify_failure`` raised) falls back to
classifying the error line. Every committed failing fixture row, and every row
as a deadline kill, goes through both lanes' real dispatch loops and must be
dispatched the same number of times, apart from the one verdict the fan-out
deliberately stops on (``_LANE_A_STOPS_LANE_B_RETRIES``).
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.retry_ladder import MIN_ATTEMPT_BUDGET_S  # noqa: E402
from utils.subagent_tasks import build_inputs, make_task  # noqa: E402
from tests.source_cache import source_tree

#: ``_run_lane_a``'s "no ``failure_category`` key at all": the error-line
#: fallback a launch failure takes. ``None`` is a verdict, so it cannot mark
#: absence.
_ABSENT = object()

#: Error lines as ``invoke_with_provider`` composes them for a non-zero exit,
#: category tag included.
RATE_LIMIT_LINE = "cli exited with code 1: HTTP 429 Too Many Requests [rate_limited]"
CONNECTION_REFUSED_LINE = (
    "cli exited with code 1: Error: connection refused by api.openai.com "
    "[connection_error]"
)


# ---------------------------------------------------------------------------
# Lane A harness
# ---------------------------------------------------------------------------

def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


def _run_lane_a(
    *,
    error: str = "",
    category=_ABSENT,
    details=None,
    result=None,
    model_spec: str = "cursor-agent:auto",
    timeout: float = 30,
    burn: float = 0.0,
    real_sleep: bool = False,
):
    """Drive ``run_single_model`` against a provider that always fails.

    ``category`` is the adapter verdict ``invoke_with_provider`` puts on
    ``details["failure_category"]`` for a failed child; leaving it out is the
    error-line fallback. ``result`` replaces the whole fake return.
    ``burn`` makes each fake dispatch spend real wall clock, which is the only
    way to observe the monotonic budget: an instantaneous fake leaves
    ``remaining()`` indistinguishable from a fresh ``timeout``.
    """
    from utils import review_orchestrator_base as rob

    calls = []
    fake_result = result
    if fake_result is None:
        fake_details = {"stderr": error, **(details or {})}
        if category is not _ABSENT:
            fake_details["failure_category"] = category
        fake_result = {"success": False, "error": error, "details": fake_details}

    def _fake_invoke(**kwargs):
        calls.append(kwargs)
        if burn:
            time.sleep(burn)
        return fake_result

    with tempfile.TemporaryDirectory() as tmp:
        plan_path = str(Path(tmp) / "plan.md")
        Path(plan_path).write_text("plan", encoding="utf-8")
        patches = [
            patch.object(rob, "invoke_with_file_output", _fake_invoke),
            patch.object(rob, "get_project_root", lambda _p: tmp),
            patch.object(rob, "get_provider_timeout", lambda *a, **k: timeout),
        ]
        if not real_sleep:
            patches.append(patch.object(rob.asyncio, "sleep", _no_sleep))
        for entered in patches:
            entered.__enter__()
        try:
            result = asyncio.run(
                rob.run_single_model(
                    model_spec=model_spec,
                    plan_path=plan_path,
                    prompt_template="{prefix}{output_json_path}",
                    prompt_context={},
                    phase_name="review-plan",
                    invoke_phase="plan_review",
                    timeout=timeout,
                    out_dir=tmp,
                    prefix="plan",
                )
            )
        finally:
            for entered in reversed(patches):
                entered.__exit__(None, None, None)
    return result, calls


def _validation_batch(phase_dir: Path):
    """One materialized validation task, as ``utils/validation.py`` writes it."""
    records = [{
        "index": 0, "group_hash": "aaaa000000000000", "group_id": "aaaa000000000000",
        "theme": "Theme 0", "category": "reliability", "models": ["model-a"],
        "suggestions": [{"title": "S0", "desc": "d", "importance": "MEDIUM",
                         "type": "addition"}],
    }]
    return make_task(
        kind="validation",
        task_id="validation-batch-0",
        prompt=(
            "You are validating review suggestions.\n\n"
            + json.dumps(records, indent=2)
            + "\n\nPrint ONE JSON object and nothing else."
        ),
        output_name="validation_batch_0.json",
        inputs=build_inputs(grouped_file=str(Path(phase_dir) / "grouped.json")),
        phase_dir=str(phase_dir),
        extra={
            "batch_index": 0,
            "group_indices": [0],
            "group_ids": ["aaaa000000000000"],
            "groups_count": 1,
        },
    )


# ---------------------------------------------------------------------------
# 0 — the structural half: the recursion is gone, and one vocabulary governs
# ---------------------------------------------------------------------------

class TestTheRecursionIsDeleted:
    """§7.1.b: ``run_single_model`` re-entered itself at ``:753`` and ``:798``.

    A tail call inside an ``async def`` keeps its retry state in a place no
    budget can see, which is *why* the two recursions had to go rather than be
    bounded. An AST check, so a future edit cannot quietly reintroduce one.
    """

    def test_run_single_model_never_calls_itself(self):
        import ast

        tree = source_tree(
            Path(__file__).parent.parent / "utils" / "review_orchestrator_base.py"
        )
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "run_single_model"
        )
        self_calls = [
            node for node in ast.walk(target)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_single_model"
        ]
        assert self_calls == [], (
            "run_single_model recursed instead of looping; every self-call handed "
            "the retry a fresh full `timeout` (§7.5 rule 1)"
        )

    def test_the_classifier_and_the_ladder_agree_about_retryable(self):
        """One table, not two.

        Every category §7.2 calls retryable must reach the ladder as a status the
        ladder retries — either because it already *is* one, or through Lane A's
        single translation table. A category in neither set would be a silent
        downgrade to "never retried".
        """
        from utils import review_orchestrator_base as rob
        from utils.providers.base import RETRYABLE_FAILURE_CATEGORIES
        from utils.retry_ladder import BACKOFF_STATUSES, RETRYABLE_STATUSES

        for category in RETRYABLE_FAILURE_CATEGORIES:
            status = rob._LADDER_STATUS_BY_CATEGORY.get(category, category)
            assert status in (RETRYABLE_STATUSES | BACKOFF_STATUSES), (
                f"{category!r} is declared retryable but reaches the ladder as "
                f"{status!r}, which it never retries"
            )


# ---------------------------------------------------------------------------
# 1 — §7.5 rule 1: one deadline per slot, taken once
# ---------------------------------------------------------------------------

class TestLaneABudgetIsMonotonic:
    """The recursion handed every attempt a *fresh full* ``timeout``.

    ``run_single_model`` re-entered itself with the same ``timeout`` argument,
    so a claude-code slot at ``default_timeout: 1800`` could run 4 × 1800 s +
    35 s of backoff ≈ 2 h against a configured ceiling of 1800 s. The retry now
    runs inside what is *left* of the one deadline.
    """

    def test_the_second_attempt_is_spawned_with_what_is_left(self):
        result, calls = _run_lane_a(
            error=CONNECTION_REFUSED_LINE,
            category="connection_error",
            timeout=3,
            burn=0.55,
        )
        assert len(calls) == 2, "one transport retry (driver.subagent_retries=1)"
        handed = [call["timeout"] for call in calls]
        # Red before M9 phase 2: [3, 3] — a fresh full copy of the ceiling.
        assert handed[1] < handed[0], (
            f"the retry was spawned with {handed[1]}s of a {handed[0]}s slot's "
            f"budget — the deadline is not monotonic"
        )
        assert result["success"] is False
        assert result["duration_seconds"] <= 3 + MIN_ATTEMPT_BUDGET_S


# ---------------------------------------------------------------------------
# 2 — §7.5 rule 4: the pre-dispatch admission test
# ---------------------------------------------------------------------------

class TestBudgetExhaustedRefusesTheSpawn:
    """Vendor-internal retries are **declared**, not discovered.

    A slot whose remaining budget cannot hold one worst-case attempt of the
    adapter's declared envelope is failed with nothing spawned, rather than a
    child being killed part-way through a retry chain the harness cannot see.
    """

    def test_lane_a_refuses_a_claude_code_slot_shorter_than_its_declared_wall(self):
        # claude-code declares max_wall_s = 4 × API_TIMEOUT_MS = 1200 s at the
        # configured 1800 s slot. A 60 s slot cannot hold one worst-case attempt.
        result, calls = _run_lane_a(
            error="anything at all",
            model_spec="claude-code:sonnet",
            timeout=60,
        )
        assert calls == [], "rule 4 is a pre-dispatch admission test"
        assert result["success"] is False
        assert "budget_exhausted" in str(result["error"])

    def test_lane_a_refuses_a_muse_slot_shorter_than_its_declared_985_s_wall(self):
        """The same rule against muse's **real** envelope (C12).

        muse declares ``max_wall_s = 985.0`` — four rounds of a ten-attempt
        provider ladder (40 provider requests) measured at 985 s on 1.4.3 in
        ``tests/fixtures/muse/``'s ``truncated`` row. A 600 s slot cannot hold
        one worst-case attempt, so nothing is spawned at all rather than a
        child being killed part-way through the chain.
        """
        result, calls = _run_lane_a(
            error="anything at all",
            model_spec="muse:muse-spark-1.2",
            timeout=600,
        )
        assert calls == [], "rule 4 is a pre-dispatch admission test"
        assert result["success"] is False
        assert "budget_exhausted" in str(result["error"])

    def test_lane_b_refuses_muse_before_rule_4_ever_runs(self, tmp_path, monkeypatch):
        """Lane B never reaches rule 4 for muse, and the reason is *stronger*.

        muse declares no ``agent-readonly`` profile, and ``dispatch_task``
        refuses an undeclared profile before it builds the ladder at all — a
        read/analyze task is never downgraded to a write-capable invocation. So
        muse's 985 s envelope cannot produce ``budget_exhausted`` on this lane:
        the slot is ``rejected`` first, with nothing spawned either way. Pinned
        rather than assumed, because "the Lane B rule-4 test uses codex" would
        otherwise read like an oversight instead of the capability fact it is.
        """
        monkeypatch.setattr(
            executor, "_spawn",
            lambda **k: (_ for _ in ()).throw(AssertionError("no spawn")),
        )
        outcome = executor.dispatch_task(
            _validation_batch(tmp_path),
            phase_dir=str(tmp_path),
            runner="muse:muse-spark-1.2",
            timeout=600,
            retries=0,
            sleep=lambda _s: None,
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert outcome.attempts == []
        assert "agent-readonly" in str(outcome.reason)

    def test_lane_bs_rule_4_refusal_names_muses_declared_985_s(self):
        """So the real envelope still meets **this lane's** admission test.

        ``dispatch_task`` builds its ladder as
        ``RetryLadder(..., envelope=getattr(provider, "internal_retries", None))``
        and renders a refusal with ``budget_exhausted_reason``; both are
        reproduced here over muse's shipped envelope, since the dispatch path
        above stops earlier for a capability reason that has nothing to do with
        budgets. Zero spawns, and the declared number in the sentence a human
        reads — the slot that refused is the one failure with no captured
        streams to explain itself.
        """
        from utils.providers.muse import MuseProvider
        from utils.retry_ladder import (
            STATUS_BUDGET_EXHAUSTED,
            STOP_BUDGET,
            RetryLadder,
            budget_exhausted_reason,
        )

        provider = MuseProvider()
        ladder = RetryLadder(
            timeout=600.0,
            retries=0,
            envelope=getattr(provider, "internal_retries", None),
            now=time.monotonic,
        )

        assert ladder.next_attempt() is None
        assert ladder.stop_reason == STOP_BUDGET
        assert ladder.spawns == 0
        reason = budget_exhausted_reason(
            ladder, target="muse:muse-spark-1.2", timeout=600.0
        )
        assert reason.startswith(STATUS_BUDGET_EXHAUSTED)
        assert "985" in reason, "the declared envelope is named"

    def test_lane_a_names_the_envelope_when_it_refuses_a_retry(self, monkeypatch):
        """Fix round 6: the ladder refuses a retry the envelope cannot hold at
        the grant, so Lane A's "granted, then refused" note no longer fires
        for it; the refusal's own note has to reach the slot's error instead,
        naming the declared wall rather than an exhausted budget."""
        from types import SimpleNamespace

        from utils import review_orchestrator_base as rob

        clock = {"t": 0.0}
        real_ladder = rob.RetryLadder

        class Ladder(real_ladder):
            def __init__(self, *args, **kwargs):
                kwargs["now"] = lambda: clock["t"]
                super().__init__(*args, **kwargs)
                self.envelope = SimpleNamespace(
                    max_requests=4, max_wall_s=self.timeout - 10.0,
                    retries_rate_limit=False,
                )

            def record(self, status, **kwargs):
                clock["t"] += 40.0  # the attempt took 40 s
                return super().record(status, **kwargs)

        monkeypatch.setattr(rob, "RetryLadder", Ladder)
        result, calls = _run_lane_a(
            error=CONNECTION_REFUSED_LINE, category="connection_error", timeout=300,
        )
        assert len(calls) == 1
        assert result["success"] is False
        error = str(result["error"])
        assert "(no retry: only 260.0s of the slot's 300.0s budget is left" in error
        assert "§7.5 rule 4" in error

    def test_lane_b_records_budget_exhausted_and_spawns_nothing(
        self, tmp_path, monkeypatch
    ):
        def _never(**kwargs):
            raise AssertionError("rule 4 must refuse before any spawn")

        monkeypatch.setattr(executor, "_spawn", _never)
        outcome = executor.dispatch_task(
            _validation_batch(tmp_path),
            phase_dir=str(tmp_path),
            runner="codex:gpt-5.5",  # declares max_wall_s = 25.0
            timeout=5,
            retries=0,
            sleep=lambda _s: None,
        )
        assert outcome.status == executor.STATUS_BUDGET_EXHAUSTED
        assert outcome.attempts == []
        # Not a dispatch outcome, so it carries no D3k rung — the same shape as
        # the D3g verification statuses.
        assert outcome.rung is None
        assert not outcome.fatal, "one slot's budget is not the batch's problem"
        assert "25" in str(outcome.reason), "the declared envelope is named"

    def test_the_batch_report_carries_budget_exhausted_verbatim(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            executor, "_spawn",
            lambda **k: (_ for _ in ()).throw(AssertionError("no spawn")),
        )
        outcome = executor.dispatch_task(
            _validation_batch(tmp_path), phase_dir=str(tmp_path),
            runner="codex:gpt-5.5", timeout=5, retries=0, sleep=lambda _s: None,
        )
        report = executor.BatchReport(
            phase="review-plan", kind="validation", phase_dir=str(tmp_path),
            outcomes=[outcome],
        )
        assert report.ok is False, "a refused slot never reaggregates"
        rendered = report.to_dict()
        assert rendered["counts"] == {executor.STATUS_BUDGET_EXHAUSTED: 1}
        assert rendered["tasks"][0]["status"] == executor.STATUS_BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# 3 — §7.5 rule 6(b): the two mechanisms compose; they must not multiply
# ---------------------------------------------------------------------------

class TestRule6bNoIdenticalRateLimitRedispatch:

    def test_a_claude_code_slot_is_not_re_dispatched_after_a_rate_limit(self):
        """claude-code declares ``retries_rate_limit: true``.

        An identical same-rung re-dispatch repeats a chain that already ran
        ``max_requests`` times against the same upstream, at the same price, for
        the same verdict — ≤ 3 harness backoffs would turn 4 requests into 20.
        """
        result, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="claude-code:sonnet",
            timeout=1800,  # comfortably above the declared 1200 s wall
        )
        assert len(calls) == 1, "red before M9 phase 2: four identical dispatches"
        assert result["success"] is False
        assert "429" in str(result["error"])

    def test_a_muse_slot_is_not_re_dispatched_after_a_rate_limit(self):
        """muse declares ``retries_rate_limit: true`` (C12).

        Measured: a fail-forever 429 costs ten provider attempts, 30 POSTs and
        248 s **inside one dispatch** — the 5xx row's envelope to two decimal
        places, because muse's ladder does not distinguish the two. So the
        harness's own backoff is removed for this adapter: a smaller retry
        budget, not a larger one. Before the declaration this slot took up to
        ``MAX_RATE_LIMIT_BACKOFFS`` further identical dispatches of a chain muse
        had already run for four minutes.
        """
        result, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="muse:muse-spark-1.2",
            timeout=1800,  # above the declared 985 s wall
        )
        assert len(calls) == 1, "rule 6(b): no identical rate-limit re-dispatch"
        assert result["success"] is False
        assert "429" in str(result["error"])

    def test_a_gemini_slot_is_not_re_dispatched_after_a_rate_limit(self):
        """gemini declares ``retries_rate_limit: true`` (C12).

        Measured: a fail-forever 429 costs ten requests over 243 s **inside one
        dispatch** — nine ``Attempt N failed with status 429. Retrying with
        backoff...`` lines on stderr and a tenth attempt that gives up
        (``tests/fixtures/gemini/``'s ``rate-limit`` row; ``server-error`` is
        the same ladder at 10 / 232 s). The wall is gemini's own sleeping, so
        rule 4 admits this slot and rule 6(b) is what stops it: ≤ 3 harness
        backoffs would have re-run a four-minute chain the CLI had already run,
        turning 10 requests into 40.
        """
        result, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="gemini:gemini-2.5-flash",
            timeout=1800,  # comfortably above the declared 243 s wall
        )
        assert len(calls) == 1, "rule 6(b): no identical rate-limit re-dispatch"
        assert result["success"] is False
        assert "429" in str(result["error"])

    def test_a_goose_slot_is_not_re_dispatched_after_a_rate_limit(self):
        """goose declares ``retries_rate_limit: true`` with **no** wall (C12).

        8 requests over 7.2 s against a fail-forever 429, four attempts at the
        real turn — but the seconds are transport against a localhost
        failserver, so ``max_wall_s`` stays ``None`` and rule 4 admits every
        slot. Rule 6(b) still fires: ≤ 3 harness backoffs would have turned 8
        requests into 32.
        """
        result, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="goose:openrouter/z-ai/glm-5.2",
            timeout=30,  # no declared wall, so a short slot is still admitted
        )
        assert len(calls) == 1, "rule 6(b): no identical rate-limit re-dispatch"
        assert result["success"] is False

    def test_a_pi_slot_is_not_re_dispatched_after_a_rate_limit(self):
        """pi declares ``retries_rate_limit: true`` with **no** wall (T7.14).

        Measured at 0.87.1 (``tests/fixtures/pi/``'s ``rate-limit`` row): a
        fail-forever 429 is 4 requests over 14.4 s inside one dispatch — three
        ``auto_retry_start`` events at 2/4/8 s and a give-up. ≤ 3 harness
        backoffs would have turned 4 requests into 16.
        """
        result, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="pi:openrouter/deepseek/deepseek-v4.1-flash",
            timeout=30,  # no declared wall, so a short slot is still admitted
        )
        assert len(calls) == 1, "rule 6(b): no identical rate-limit re-dispatch"
        assert result["success"] is False
        assert "429" in str(result["error"])

    def test_an_adapter_with_no_declared_envelope_keeps_the_harness_backoff(self):
        """Rule 6(c) — ``None`` means "assume one request", never "the worst".

        cursor-agent measures nothing, so §7.5's ≤ 3 backoffs apply unchanged.
        This is the control that keeps rule 6(b) from disabling the real path.
        """
        _, calls = _run_lane_a(
            error=RATE_LIMIT_LINE,
            category="rate_limited",
            model_spec="cursor-agent:auto",
            timeout=30,
        )
        assert len(calls) == 4, "one initial dispatch plus MAX_RATE_LIMIT_BACKOFFS"


# ---------------------------------------------------------------------------
# 4 — the forever-429 slot, inside resolved_timeout, outer wait_for untouched
# ---------------------------------------------------------------------------

class TestForever429EndsInsideResolvedTimeout:
    """§7.1.b's constraint, stated as a test.

    ``run_models_concurrent``'s ``asyncio.wait_for`` is a **backstop**: when it
    fires it cancels the task and does *not* kill the child (§8). So the slot
    has to end on its own budget, and this test fails if the fan-out timeout is
    what stopped it.
    """

    def test_the_slot_ends_on_its_own_budget(self, tmp_path, monkeypatch):
        from utils import review_orchestrator_base as rob

        resolved_timeout = 4.0
        # Each fake dispatch burns 40 % of the budget it was *handed*. Under the
        # recursion every attempt is handed the full ceiling, so four attempts
        # cost 4 × 1.6 s = 6.4 s against a 4 s slot. Under the ladder attempt N
        # is handed what is left, so the slot is bounded by its own deadline.
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        calls = []

        def _fake_invoke(**kwargs):
            calls.append(kwargs)
            time.sleep(kwargs["timeout"] * 0.4)
            return {
                "success": False,
                "error": RATE_LIMIT_LINE,
                "details": {"stderr": "", "failure_category": "rate_limited"},
            }

        fired = {"outer": False}
        real_wait_for = asyncio.wait_for

        async def _spy_wait_for(awaitable, timeout=None):
            try:
                return await real_wait_for(awaitable, timeout)
            except asyncio.TimeoutError:
                fired["outer"] = True
                raise

        plan = tmp_path / "plan.md"
        plan.write_text("plan", encoding="utf-8")

        async def _drive():
            async def per_model(spec, index):
                return await rob.run_single_model(
                    model_spec=spec,
                    plan_path=str(plan),
                    prompt_template="{prefix}{output_json_path}",
                    prompt_context={},
                    phase_name="review-plan",
                    invoke_phase="plan_review",
                    timeout=resolved_timeout,
                    out_dir=str(tmp_path),
                    prefix="plan",
                )

            return await rob.run_models_concurrent(
                ["cursor-agent:auto"], per_model, resolved_timeout, 4,
                anchor=str(plan),
            )

        started = time.monotonic()
        with patch.object(rob, "invoke_with_file_output", _fake_invoke), \
             patch.object(rob, "get_project_root", lambda _p: str(tmp_path)), \
             patch.object(rob, "get_provider_timeout",
                          lambda *a, **k: resolved_timeout), \
             patch.object(rob.asyncio, "wait_for", _spy_wait_for):
            results = asyncio.run(_drive())
        elapsed = time.monotonic() - started

        assert fired["outer"] is False, (
            "the outer wait_for is a backstop, never the enforcement point — "
            "when it fires it cancels the task without killing the child (§8)"
        )
        assert "cursor-agent:auto" in results, "the slot produced a real result"
        assert results["cursor-agent:auto"]["success"] is False
        # Red before M9 phase 2: 4 × 1.6 s = 6.4 s against a 4 s ceiling.
        assert elapsed <= resolved_timeout + MIN_ATTEMPT_BUDGET_S, (
            f"the slot ran {elapsed:.2f}s against a {resolved_timeout}s budget"
        )
        assert len(calls) <= 5, "§7.5 rule 2's ceiling across the two counters"
        # …and the fan-out as a whole stayed inside base_timeout × 2 + stagger.
        assert elapsed <= resolved_timeout * 2 + 2.0


# ---------------------------------------------------------------------------
# 5 — what the ladder is handed: the adapter's verdict, else the error line
# ---------------------------------------------------------------------------

class TestTheAdapterVerdictDecides:
    """A failed child carries ``details["failure_category"]``.

    ``run_single_model`` obeys it and does not re-read the error line, which
    is composed for a human and can say less (a bare ``exited with code 1``)
    or something else (``API Error`` on an auth failure) than the streams did.
    """

    def test_the_category_wins_over_a_rate_limit_shaped_line(self):
        _, calls = _run_lane_a(
            error="cli exited with code 1: HTTP 429 Too Many Requests [auth_failed]",
            category="auth_failed",
        )
        assert len(calls) == 1, "auth_failed is rung 1: never retried"

    def test_a_bare_exit_line_still_gets_the_transport_retry_it_was_owed(self):
        _, calls = _run_lane_a(
            error="cline exited with code 1 [runner_error]", category="runner_error",
        )
        assert len(calls) == 2, "one transport retry (driver.subagent_retries=1)"

    def test_none_is_a_verdict_not_a_missing_key(self):
        """The adapter recognised nothing: one transport retry, as the executor
        gives a run with no usable result, and the rate-limit-shaped line is
        not read (that would be four dispatches)."""
        _, calls = _run_lane_a(error="HTTP 429 Too Many Requests", category=None)
        assert len(calls) == 2, "rung 4: one transport retry (driver.subagent_retries=1)"

    def test_spawn_failed_and_permission_denied_are_never_retried(self):
        """An install or argv refusal, and a 403 or a denied tool, fail the
        same way on every attempt. The executor stops on the first too (it
        fails the batch); the second it retries once, on purpose."""
        for category in ("spawn_failed", "permission_denied"):
            _, calls = _run_lane_a(
                error=f"cli exited with code 1 [{category}]", category=category,
            )
            assert len(calls) == 1, category

    def test_a_timeout_category_is_rung_2(self):
        _, calls = _run_lane_a(
            error="cli timed out after 30s", category="timeout",
        )
        assert len(calls) == 2

    def test_a_jail_denial_still_wins(self):
        """F1 item 6: non-retryable and never moved to another rung, whether
        the adapter named a retryable category or recognised nothing."""
        for category in ("rate_limited", None):
            _, calls = _run_lane_a(
                error="codex exited with code 1: [JAIL_DENIED] write outside the jail",
                category=category,
                details={"failure_reason": "jail-denied"},
            )
            assert len(calls) == 1, category

    def test_a_context_overflow_category_names_the_remedy(self):
        result, calls = _run_lane_a(
            error="cli exited with code 1 [context_length_exceeded]",
            category="context_length_exceeded",
        )
        assert len(calls) == 1
        assert "context" in str(result["error"]).lower()
        assert "[context_length_exceeded]" not in str(result["error"])

    def test_a_claude_code_rate_limit_category_is_not_re_dispatched(self):
        """Rule 6(b) holds on the category path too (``retries_rate_limit``)."""
        _, calls = _run_lane_a(
            error="claude-code exited with code 1 [rate_limited]",
            category="rate_limited",
            model_spec="claude-code:sonnet",
            timeout=1800,
        )
        assert len(calls) == 1


class TestTheErrorLineFallback:
    """No ``failure_category`` key (a launch failure, an adapter whose
    ``classify_failure`` raised): the shared classifier reads the error line
    alone."""

    def test_a_rate_limit_line_backs_off(self):
        _, calls = _run_lane_a(error="HTTP 429 Too Many Requests")
        assert len(calls) == 4, "one initial dispatch plus MAX_RATE_LIMIT_BACKOFFS"

    def test_rule_6b_holds_on_the_fallback_too(self):
        _, calls = _run_lane_a(
            error="HTTP 429 Too Many Requests",
            model_spec="claude-code:sonnet",
            timeout=1800,
        )
        assert len(calls) == 1

    def test_a_connection_line_gets_one_transport_retry(self):
        _, calls = _run_lane_a(error="Error: connection refused by api.openai.com")
        assert len(calls) == 2

    def test_an_unrecognised_line_is_never_retried(self):
        _, calls = _run_lane_a(error="JSON parsing failed from both file and stdout")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6 — both lanes' real dispatch loops make the same call on every failing child
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
#: Far above every adapter's declared ``max_wall_s``, so rule 4 admits every
#: slot and the ladder decides on the status alone.
_ROW_SLOT_S = 100000
_ROW_RETRIES = 1


def _manifest_rows():
    import yaml

    rows = []
    for manifest_path in sorted(_FIXTURES.glob("*/manifest.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        for name, row in (manifest.get("rows") or {}).items():
            if "not_applicable" in row:
                continue
            rows.append((manifest["provider"], name, row))
    return rows


_ROWS = _manifest_rows()


def _assert_the_tag_only_names_itself(untagged, category):
    """A string reader of the tagged line (``classify_streams`` over the error
    line, the fallback's rule) reaches the untagged line's verdict or the
    adapter's own category, never a third one."""
    from utils.providers.base import classify_streams

    tagged = classify_streams(f"{untagged} [{category}]", "")
    assert tagged in (classify_streams(untagged, ""), category), (
        f"the [{category}] tag moved {untagged!r} to {tagged!r}"
    )


def _category_vocabulary():
    from utils import retry_ladder
    from utils.providers import base

    names = {
        value for name, value in vars(base).items()
        if name.startswith("CATEGORY_") and isinstance(value, str)
    }
    names |= {
        retry_ladder.STATUS_AUTH_FAILED, retry_ladder.STATUS_BUDGET_EXHAUSTED,
        retry_ladder.STATUS_CONTEXT_LENGTH_EXCEEDED, retry_ladder.STATUS_RATE_LIMITED,
        retry_ladder.STATUS_RUNNER_ERROR, retry_ladder.STATUS_SPAWN_FAILED,
        retry_ladder.STATUS_TIMEOUT,
    }
    return sorted(names)


@pytest.mark.parametrize("category", _category_vocabulary())
@pytest.mark.parametrize("untagged", [
    "cli exited with code 1",
    "cli exited with code 1: Error: connection refused by the API host",
    "cli exited with code 1: HTTP 401 Unauthorized: invalid api key",
    "cli exited with code 2: the run took too long and timed out",
    "cli exited with code 1: the module under review spans 429 lines",
])
def test_the_category_tag_only_ever_names_itself(untagged, category):
    _assert_the_tag_only_names_itself(untagged, category)


_NONZERO_ROWS = [r for r in _ROWS if r[2].get("exit_code", 0) != 0]

#: The adapter verdict the lanes answer differently on purpose: a 403 or a
#: denied tool. The executor's ``_classify_failed_run`` does not forward it, so
#: it keeps its parse failure's rung-4/5 status and retries once; the fan-out
#: stops. Pinned both ways, so a change on either lane shows up here.
_LANE_A_STOPS_LANE_B_RETRIES = frozenset({"permission_denied"})


def _row_streams(provider, row):
    def read(key):
        name = row.get(key)
        if not name:
            return ""
        return (_FIXTURES / provider / name).read_bytes().decode("utf-8", errors="replace")

    return read("stdout"), read("stderr")


def _patch_adapter(monkeypatch, adapter, name, value):
    """Shadow one method of a registry adapter for this test.

    Through the instance ``__dict__``, whose undo deletes the key: the
    registry's adapters are process-wide, and undoing an instance ``setattr``
    leaves the original bound method behind on the instance, where it
    shadows a later test's class-level patch of the same method
    (``test_opencode_config_providers`` counts ``resolve_env`` calls that way).
    """
    monkeypatch.setitem(vars(adapter), name, value)


def _replay(monkeypatch, tmp_path, provider_name, stdout, stderr, exit_code, *, kill=False):
    """The real ``invoke_with_file_output`` over a captured row, spawn stubbed."""
    import contextlib
    import subprocess

    from utils import llm_client
    from utils.provider_registry import get_provider

    adapter = get_provider(provider_name)

    def fake_run(cmd, **kwargs):
        if kill:
            raise subprocess.TimeoutExpired(cmd, 30, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, exit_code, stdout, stderr)

    @contextlib.contextmanager
    def no_dispatch_context(*args, **kwargs):
        class _Dispatch:
            refusal = None
        yield _Dispatch()

    monkeypatch.setattr(llm_client, "_run_provider_cli", fake_run)
    monkeypatch.setattr(llm_client, "_resolve_executable", lambda name: ([name], False))
    monkeypatch.setattr(llm_client, "provider_dispatch_context", no_dispatch_context)
    _patch_adapter(monkeypatch, adapter, "is_available", lambda: True)
    _patch_adapter(
        monkeypatch,
        adapter, "build_command_for_transport", lambda *a, **k: (["/bin/true"], None)
    )
    _patch_adapter(monkeypatch, adapter, "resolve_env", lambda *a, **k: {})
    return adapter, llm_client.invoke_with_file_output(
        prompt_template="hi {output_json_path}",
        model_spec=f"{provider_name}:m",
        prompt_context={},
        output_dir=tmp_path,
        phase="review-plan",
        timeout=30,
        prefer_arrays=True,
        cwd=str(tmp_path),
    )


def _lane_b_dispatches(monkeypatch, phase_dir, provider_name, **capture):
    """The executor's real call: ``dispatch_task`` on a read task, every attempt
    answered by one ``_Capture`` built from ``capture`` (spawn stubbed). Returns
    how many attempts it spawned.

    Most adapters declare no ``agent-readonly`` profile and the executor refuses
    them before any spawn, so the profile gate is lifted here: what is compared
    is how the executor classifies and retries a capture once it has one, and
    that path reads the profile only for a write task.
    """
    from utils.provider_registry import get_provider

    spawned = []

    def fake_spawn(**kwargs):
        spawned.append(kwargs)
        return executor._Capture(argv=["/bin/true"], **capture)

    monkeypatch.setattr(executor, "_spawn", fake_spawn)
    _patch_adapter(
        monkeypatch, get_provider(provider_name), "supports_profile", lambda _p: True
    )
    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)
    executor.dispatch_task(
        _validation_batch(phase_dir),
        phase_dir=str(phase_dir),
        runner=f"{provider_name}:m",
        timeout=_ROW_SLOT_S,
        retries=_ROW_RETRIES,
        sleep=lambda _s: None,
    )
    return len(spawned)


def _lane_a_dispatches(monkeypatch, result, provider_name):
    """``run_single_model`` handed this result on every dispatch: how many?"""
    from utils import review_orchestrator_base as rob

    monkeypatch.setattr(rob, "get_subagent_retries", lambda *a, **k: _ROW_RETRIES)
    monkeypatch.setattr(
        rob, "sandbox_refusal",
        lambda *a, **k: rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None),
    )
    _, calls = _run_lane_a(
        result=result, model_spec=f"{provider_name}:m", timeout=_ROW_SLOT_S,
    )
    return len(calls)


def _assert_lanes_agree(verdict, lane_a, lane_b):
    if verdict in _LANE_A_STOPS_LANE_B_RETRIES:
        assert (lane_a, lane_b) == (1, 1 + _ROW_RETRIES), (
            f"{verdict}: the fan-out stops, the executor retries once"
        )
    else:
        assert lane_a == lane_b, (
            f"verdict {verdict!r}: the fan-out dispatched {lane_a} times, "
            f"the executor {lane_b}"
        )


@pytest.mark.parametrize("verdict", [None, *_category_vocabulary()])
def test_every_verdict_gets_the_executors_call(monkeypatch, tmp_path, verdict):
    """Every verdict an adapter can give a non-zero exit, through both lanes'
    dispatch loops. opencode declares the read profile and no
    ``internal_retries`` envelope, so a rate limit takes the full backoff on
    both lanes."""
    from utils.provider_registry import get_provider

    _patch_adapter(
        monkeypatch, get_provider("opencode"), "classify_failure", lambda *a, **k: verdict
    )
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path, "opencode", stdout="", stderr="boom", exit_code=1,
    )
    tag = f" [{verdict}]" if verdict else ""
    lane_a = _lane_a_dispatches(monkeypatch, {
        "success": False,
        "error": f"opencode exited with code 1: boom{tag}",
        "details": {"stderr": "boom", "failure_category": verdict},
    }, "opencode")
    _assert_lanes_agree(verdict, lane_a, lane_b)


def test_an_empty_codex_exit_1_is_retried_on_both_lanes(monkeypatch, tmp_path):
    """codex recognises nothing in an exit-1 run with empty streams; both lanes
    send the slot once more."""
    _, result = _replay(monkeypatch, tmp_path, "codex", "", "", 1)
    assert result["details"]["failure_category"] is None
    lane_a = _lane_a_dispatches(monkeypatch, result, "codex")
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path / "lane-b", "codex", stdout="", stderr="", exit_code=1,
    )
    assert lane_a == lane_b == 1 + _ROW_RETRIES


@pytest.mark.parametrize(
    "provider_name,row_name,row",
    _NONZERO_ROWS,
    ids=[f"{p}/{n}" for p, n, _ in _NONZERO_ROWS],
)
def test_a_non_zero_exit_row_gets_the_executors_call(
    monkeypatch, tmp_path, provider_name, row_name, row
):
    """The same captured child through both lanes' real paths: Lane A is
    ``invoke_with_file_output``'s real result handed to ``run_single_model``,
    Lane B is ``dispatch_task`` over the capture."""
    stdout, stderr = _row_streams(provider_name, row)
    exit_code = row["exit_code"]
    adapter, result = _replay(
        monkeypatch, tmp_path, provider_name, stdout, stderr, exit_code
    )
    assert result["success"] is False
    verdict = adapter.classify_failure(stdout, stderr, exit_code)
    assert result["details"]["failure_category"] == verdict
    if verdict:
        line = str(result["error"])
        assert line.endswith(f" [{verdict}]"), "the error line names the category"
        _assert_the_tag_only_names_itself(line[: -len(f" [{verdict}]")], verdict)

    lane_a = _lane_a_dispatches(monkeypatch, result, provider_name)
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path / "lane-b", provider_name,
        stdout=stdout, stderr=stderr, exit_code=exit_code,
    )
    _assert_lanes_agree(verdict, lane_a, lane_b)


_EXIT_ZERO_FAILING_ROWS = [
    r for r in _ROWS
    if r[2].get("exit_code", 0) == 0 and r[2]["expect"]["success"] is False
]


@pytest.mark.parametrize(
    "provider_name,row_name,row",
    _EXIT_ZERO_FAILING_ROWS,
    ids=[f"{p}/{n}" for p, n, _ in _EXIT_ZERO_FAILING_ROWS],
)
def test_an_exit_zero_failure_row_gets_the_executors_call(
    monkeypatch, tmp_path, provider_name, row_name, row
):
    """An exit 0 that left no answer, and no result file: the fan-out gets
    the verdict of the gate both lanes read (``exit_zero_failure_category``),
    and both lanes' real paths dispatch it the same number of times. An
    adapter that is not ``EXIT0_CLASSIFIABLE`` gives no verdict: the fan-out
    stops on its error line and the executor keeps its parse failure's one
    retry, the permanent difference its rows mark ``by_design``."""
    from utils.providers.base import exit_zero_failure_category

    stdout, stderr = _row_streams(provider_name, row)
    adapter, result = _replay(monkeypatch, tmp_path, provider_name, stdout, stderr, 0)
    assert result["success"] is False
    classified, verdict = exit_zero_failure_category(adapter, stdout, stderr)
    if classified:
        assert result["details"]["failure_category"] == verdict
    else:
        assert "failure_category" not in result["details"]

    lane_a = _lane_a_dispatches(monkeypatch, result, provider_name)
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path / "lane-b", provider_name,
        stdout=stdout, stderr=stderr, exit_code=0,
    )
    if classified:
        _assert_lanes_agree(verdict, lane_a, lane_b)
    else:
        assert "by_design" in row["expect"]["lane_status"]
        assert (lane_a, lane_b) == (1, 1 + _ROW_RETRIES)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_spawn_failed_stops_both_lanes_only_on_a_non_zero_exit(
    monkeypatch, tmp_path, exit_code
):
    """A refused invocation never exits 0: on an exit 0 the launch-failure
    words came from the model or a tool, and both lanes take the parse
    failure's one transport retry; on a non-zero exit both stop."""
    from utils.provider_registry import get_provider

    _patch_adapter(
        monkeypatch, get_provider("opencode"), "classify_failure",
        lambda *a, **k: "spawn_failed",
    )
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path, "opencode", stdout="", stderr="boom",
        exit_code=exit_code,
    )
    _, result = _replay(monkeypatch, tmp_path / "a", "opencode", "", "boom", exit_code)
    lane_a = _lane_a_dispatches(monkeypatch, result, "opencode")
    expected = 1 if exit_code else 1 + _ROW_RETRIES
    assert (lane_a, lane_b) == (expected, expected)


@pytest.mark.parametrize(
    "provider_name,row_name,row",
    _ROWS,
    ids=[f"{p}/{n}" for p, n, _ in _ROWS],
)
def test_a_deadline_kill_is_a_timeout_whatever_the_row_streamed(
    monkeypatch, tmp_path, provider_name, row_name, row
):
    """Lane B decides a killed run before it reads a byte (``_classify_pre_parse``),
    so a login notice or a 429 in the partial streams never moves it off rung 2.
    Lane A mirrors that."""
    from utils.retry_ladder import STATUS_TIMEOUT

    stdout, stderr = _row_streams(provider_name, row)
    _, result = _replay(
        monkeypatch, tmp_path, provider_name, stdout, stderr, None, kill=True
    )
    assert result["details"]["failure_category"] == STATUS_TIMEOUT
    lane_b = _lane_b_dispatches(
        monkeypatch, tmp_path / "lane-b", provider_name,
        stdout=stdout, stderr=stderr, timed_out=True,
    )
    assert _lane_a_dispatches(monkeypatch, result, provider_name) == lane_b
