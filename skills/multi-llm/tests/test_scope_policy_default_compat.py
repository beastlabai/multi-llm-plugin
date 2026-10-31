"""Default compatibility of the scope-policy apply gate (plans/scope-policy-and-report-tags.md §5, §7).

With identical recorded validation results (scope verdicts included) and
identical user choices, an apply run with no ``--scope-policy`` and one with
an explicit ``expand`` produce exactly what the code produced before the
feature existed: the same selected/applied ids, priorities, judge decisions,
skip lists, confirmation payload, batches and write prompts.

``tests/golden/scope_policy_default_apply.json`` was captured from the
baseline code (c91e8d29, before any apply-side scope change) with
``MULTI_LLM_UPDATE_GOLDEN=1``; re-capture it only for an intentional change
that is not about scope.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import apply_code_fixes_orchestrator as acf  # noqa: E402
import apply_suggestions_orchestrator as aso  # noqa: E402

from tests.scope_apply_fixtures import (  # noqa: E402
    judge_record, native_snapshot, normalize, run_apply, write_code_review,
    write_review,
)
from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401

GOLDEN = Path(__file__).parent / "golden" / "scope_policy_default_apply.json"


def _roots(repo):
    return [str(repo.root), str(repo.tmp_path)]


#: The one intended stdout addition on a default ``--yes`` run: D6 row 16's
#: recorded resolution (plan §4 asks for it; like row 2's ``defaults.models``
#: line it is a record, not a decision). Dropped before comparing with the
#: pre-feature golden so every other byte is still held to it.
_ROW_16_LINE = "[AUTO_RESOLVED] scope-policy: "


def _without_row_16(stdout):
    return "".join(
        line for line in stdout.splitlines(keepends=True)
        if not line.startswith(_ROW_16_LINE)
    )


def _capture_plan(repo, capsys, policy, review_policy):
    """apply-suggestions on the native leg: confirmation, then a --yes run."""
    review = write_review(repo.root, selections=False, verdicts=False,
                          review_policy=review_policy)
    plan = review["plan"]
    capsys.readouterr()
    assert run_apply(aso, plan, scope_policy=policy) == 0
    confirmation = json.loads(capsys.readouterr().out)

    write_review(repo.root, review_policy=review_policy)  # choices and verdicts
    capsys.readouterr()
    assert run_apply(aso, plan, "--yes", scope_policy=policy) == 0
    stdout = capsys.readouterr().out
    return {
        "confirmation": normalize(confirmation, _roots(repo)),
        "native": native_snapshot(plan, "apply-suggestions", _roots(repo)),
        "judge": judge_record(plan, "apply-suggestions", _roots(repo)),
        "stdout": normalize(_without_row_16(stdout), _roots(repo)),
    }


def _capture_code(repo, capsys, policy, review_policy):
    review = write_code_review(repo.root, plan_name="code", review_policy=review_policy)
    plan = review["plan"]
    capsys.readouterr()
    assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD", scope_policy=policy) == 0
    stdout = capsys.readouterr().out
    return {
        "native": native_snapshot(plan, "apply-fixes", _roots(repo)),
        "stdout": normalize(_without_row_16(stdout), _roots(repo)),
    }


def _capture_judge_tasks(repo, capsys, policy, review_policy):
    """No verdict on disk yet: the judging tasks the run materializes (the
    model calls it asks for) and the work set it hands out meanwhile."""
    review = write_review(repo.root, verdicts=False, plan_name="judge",
                          review_policy=review_policy)
    plan = review["plan"]
    capsys.readouterr()
    assert run_apply(aso, plan, "--yes", scope_policy=policy) == 0
    capsys.readouterr()
    phase_dir = plan.parent / plan.stem / "apply-suggestions"
    document = json.loads((phase_dir / "human_decision_tasks.json").read_text(encoding="utf-8"))
    tasks = document.get("tasks") or document.get("batches") or []
    return {
        "judge_tasks": normalize(
            [{k: t.get(k) for k in ("task_id", "prompt", "group_ids", "output_name")}
             for t in tasks], _roots(repo)),
        "native": native_snapshot(plan, "apply-suggestions", _roots(repo)),
    }


def _capture(repo, capsys, policy, review_policy):
    return {"apply-suggestions": _capture_plan(repo, capsys, policy, review_policy),
            "apply-code-fixes": _capture_code(repo, capsys, policy, review_policy),
            "judge-tasks": _capture_judge_tasks(repo, capsys, policy, review_policy)}


@pytest.mark.parametrize("policy,review_policy", [
    # A new run's default: a round reviewed under expand, and a legacy round
    # with no scope record at all.
    (None, "expand"),
    (None, None),
    # An explicit expand override of a preserve review: the stamped selected
    # remedy and every deferral-worthy verdict are ignored.
    ("expand", "preserve"),
    ("expand", "expand"),
])
def test_default_and_explicit_expand_match_the_pre_feature_golden(
        repo, capsys, policy, review_policy):
    captured = _capture(repo, capsys, policy, review_policy)
    if (os.environ.get("MULTI_LLM_UPDATE_GOLDEN") == "1" and policy is None
            and review_policy is None):
        GOLDEN.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for phase in ("apply-suggestions", "apply-code-fixes", "judge-tasks"):
        for key in golden[phase]:
            assert captured[phase][key] == golden[phase][key], (phase, key)


def test_the_golden_covers_every_scope_shape(repo, capsys):
    """The fixture is not vacuous: extensions, unclear, legacy, malformed and
    stale-bound groups, a LOW style finding and an optional improvement are
    all applied under the default, as before."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    titles = {item["title"] for item in golden["apply-suggestions"]["native"]["to_apply"]}
    assert {"Also export PDF", "Clarify the week boundary", "Handle empty weeks",
            "Tidy wording", "Quote fields", "Fix a typo", "Add caching",
            "Add an API endpoint"} <= titles
    assert golden["apply-suggestions"]["native"]["write_prompts"]
    assert "Scope Policy" not in json.dumps(golden)
