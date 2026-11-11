"""A write batch that declines an item: replies, task builders and an apply rig over a pair of items.

``_two_item_task`` and ``_respond_with`` script a batch over the write
executor's ``env`` fixture (``tests/test_write_executor.py``); ``_PairRig`` runs
the real apply-suggestions orchestrator over two items per batch.
``CODEX_08_REPLY`` and the ``_codex_08_*`` helpers replay the Codex reply that
applied only part of a batch. No fixtures.
"""

from __future__ import annotations

import json

from tests.support.write_orchestrators import APPLY_PLAN, _ApplyRig, _validation


REASON = "the implementation record says T011 removed the clamp on purpose"


def _two_item_task(env, index=0, ids=("s0", "s1"), **over):
    return env.apply_task(index, expected_ids=list(ids), **over)


def _respond_with(env, reply, *, edit=True):
    seen = []

    def respond(task, prompt):
        seen.append(prompt)
        if edit:
            env.plan.write_text(env.plan.read_text() + f"edit {len(seen)}\n")
        return dict(reply)

    respond.seen = seen  # type: ignore[attr-defined]
    return respond


def _alpha_pair_groups():
    """Two suggestions on one section (one batch) and one on another."""
    def group(n, section, step):
        return {"theme": f"theme {n}", "category": "c", "models": ["m"],
                "suggestions": [{
                    "title": f"suggestion {n}", "desc": f"Do {n}.",
                    "type": "addition", "reference": f"## Step {step}: {section}",
                    "importance": "MEDIUM", "source_model": "m",
                }]}
    return [group("A1", "Alpha", 1), group("A2", "Alpha", 1), group("B", "Beta", 2)]


class _PairRig(_ApplyRig):
    """The apply-suggestions orchestrator over a review whose first batch
    holds two items."""

    def _review_pair(self, repo):
        from utils.state_manager import get_or_create_state

        plan = repo.root / "plans" / "apply.md"
        plan.write_text(APPLY_PLAN, encoding="utf-8")
        review = repo.root / "plans" / "apply" / "review-plan"
        review.mkdir(parents=True, exist_ok=True)
        (review / "grouped.json").write_text(
            json.dumps(_alpha_pair_groups()), encoding="utf-8")
        (review / "validation.json").write_text(
            json.dumps(_validation(["valid"] * 3)), encoding="utf-8")
        state = get_or_create_state(plan)
        state.mark_phase_completed("review-plan")
        state.save()
        return plan


CODEX_08_REPLY = {
    # Codex racing run 7, apply-code-fixes 08: fix 2 applied and attested,
    # fix 1 left alone on purpose and said so only in `notes`.
    "status": "applied", "ids": ["s1"], "files_touched": ["plans/p.md"],
    "summary": "applied suggestion 2; kept the deliberate T011 decision",
    "notes": "Suggestion 1: Not applied; T011 records it as deliberate. "
             "Suggestion 2: applied.",
}
SKIP_FLAGS = ["--resume", "--human-task", "apply-batch-0=skip"]


def _codex_08_respond(env, calls):
    def respond(task, prompt):
        calls.append(task["task_id"])
        env.plan.write_text(env.plan.read_text() + f"{task['task_id']}\n")
        if task["task_id"] == "apply-batch-0":
            return dict(CODEX_08_REPLY)
        return {"status": "applied", "ids": ["s2"],
                "files_touched": ["plans/p.md"], "summary": "applied s2"}
    return respond


def _codex_08_tasks(env):
    return [_two_item_task(env), env.apply_task(1, expected_ids=["s2"])]
