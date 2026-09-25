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
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.retry_ladder import MIN_ATTEMPT_BUDGET_S  # noqa: E402
from utils.subagent_tasks import build_inputs, make_task  # noqa: E402


# ---------------------------------------------------------------------------
# Lane A harness
# ---------------------------------------------------------------------------

def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


def _run_lane_a(
    *,
    error: str,
    model_spec: str = "cursor-agent:auto",
    timeout: float = 30,
    burn: float = 0.0,
    real_sleep: bool = False,
):
    """Drive ``run_single_model`` against a provider that always fails.

    ``burn`` makes each fake dispatch spend real wall clock, which is the only
    way to observe the monotonic budget: an instantaneous fake leaves
    ``remaining()`` indistinguishable from a fresh ``timeout``.
    """
    from utils import review_orchestrator_base as rob

    calls = []

    def _fake_invoke(**kwargs):
        calls.append(kwargs)
        if burn:
            time.sleep(burn)
        return {
            "success": False,
            "error": error,
            "details": {"stderr": error},
        }

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

        source = (
            Path(__file__).parent.parent / "utils" / "review_orchestrator_base.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
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
            error="Error: connection refused by api.openai.com",
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

    def test_lane_a_refuses_a_muse_slot_shorter_than_its_declared_736_s_wall(self):
        """The same rule against muse's **real** envelope (C12).

        muse declares ``max_wall_s = 736.0`` — its own 12-minute wall-clock
        ceiling, measured at 12m16s across four rounds of a ten-attempt provider
        ladder (40 requests) in ``tests/fixtures/muse/``'s ``truncated`` row.
        A 600 s slot — muse's own ``default_timeout`` — cannot hold one
        worst-case attempt, so nothing is spawned at all rather than a child
        being killed part-way through the chain.
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
        muse's 736 s envelope cannot produce ``budget_exhausted`` on this lane:
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

    def test_lane_bs_rule_4_refusal_names_muses_declared_736_s(self):
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
        assert "736" in reason, "the declared envelope is named"

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
            error="HTTP 429 Too Many Requests",
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
            error="HTTP 429 Too Many Requests",
            model_spec="muse:muse-spark-1.2",
            timeout=1800,  # comfortably above the declared 736 s wall
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
            error="HTTP 429 Too Many Requests",
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
            error="HTTP 429 Too Many Requests",
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
            error="HTTP 429 Too Many Requests",
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
            error="HTTP 429 Too Many Requests",
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
                "error": "HTTP 429 Too Many Requests",
                "details": {"stderr": "HTTP 429 Too Many Requests"},
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
