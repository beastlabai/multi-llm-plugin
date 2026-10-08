"""A report-only re-ask cannot retract the checks of the turn that edited files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import test_write_executor as harness  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.subagent_prompts import build_schema_repair_prompt  # noqa: E402


FAILED_CHECK = {
    "passed": False,
    "findings": ["npm test failed: regression X"],
    "listed": {"npm test": "failed", "node --check src/a.py": "passed"},
}
MISSING = object()
env = harness.env


def _repair(
    env, kind, original_check=MISSING, repaired_check=MISSING, *,
    incomplete=False, verification=False,
):
    env.use("opencode:opencode/big-pickle")
    if kind == "implement":
        task = harness._code_task(env, **({"declared_files": []} if verification else {}))
    else:
        task = env.apply_task(
            0, kind=kind, declared_files=[str(env.root / "src/a.py")],
            expected_ids=["s0", "s1"],
        )
    original = {
        "status": "no-op" if verification else "applied", "ids": task["expected_ids"],
        "files_touched": [] if verification else ["src/a.py"],
    }
    repaired = {**original, "summary": "Changed src/a.py"}
    if incomplete:
        original.update(ids=["s0"], summary="Changed s0; s1 intentionally left alone")
        repaired.update(ids=["s0"], declined=[{"id": "s1", "reason": "intentional"}])
    # Otherwise the missing summary draws the existing schema-repair re-ask.
    if original_check is not MISSING:
        original["check"] = original_check
    if repaired_check is not MISSING:
        repaired["check"] = repaired_check
    env.edits(
        [] if verification else [{"path": "src/a.py", "mode": "patch", "content": "a = 2\n"}],
        result=original,
    )
    env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(repaired))
    result = env.scheduler([task], kind=kind).run()
    outcome = result.outcomes[0]
    assert len(outcome.attempts) == 2
    assert outcome.ok
    return outcome, json.loads(Path(outcome.output_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("replacement", [MISSING, {"passed": True, "findings": []}])
def test_incomplete_apply_report_cannot_drop_or_reverse_failed_checks(env, replacement):
    outcome, saved = _repair(
        env, "apply_code_fixes", FAILED_CHECK, replacement, incomplete=True,
    )
    assert outcome.attempts[0].status == executor.STATUS_INCOMPLETE_REPORT
    assert saved["check"] == FAILED_CHECK
    assert outcome.batch_check == FAILED_CHECK["findings"]
    assert saved["verdict"]["batch_check"]["findings"] == outcome.batch_check


@pytest.mark.parametrize("kind", ["apply_code_fixes", "implement"])
@pytest.mark.parametrize("check", [
    FAILED_CHECK,
    {"passed": "false", "findings": "npm test failed: regression X"},
    {"passed": 0},
    {"could_not_run": "1", "findings": ["npm test is unavailable"]},
    {"findings": ["npm test failed: regression X"]},
])
def test_schema_repair_preserves_the_original_readable_check(env, kind, check):
    outcome, saved = _repair(env, kind, check, {"passed": True, "findings": []})
    assert outcome.attempts[0].status == executor.STATUS_SCHEMA_INVALID
    assert saved["check"] == check
    assert outcome.batch_check == executor.batch_check_findings({"check": check})


def test_listed_failure_survives_without_a_top_level_check_verdict(env):
    # `listed` can be useful even when `passed` and `findings` were omitted.
    (env.root / "package.json").write_text(
        json.dumps({"scripts": {"test": "node tests.mjs"}}), encoding="utf-8",
    )
    check = {"listed": {"npm test": "failed"}}
    outcome, saved = _repair(env, "apply_code_fixes", check, incomplete=True)
    assert saved["check"] == check
    assert outcome.batch_check and any("npm test" in f for f in outcome.batch_check)


def test_a_check_tasks_schema_repair_cannot_turn_its_findings_into_a_pass(env):
    outcome, saved = _repair(
        env, "implement", FAILED_CHECK, {"passed": True}, verification=True,
    )
    assert saved["check"] == FAILED_CHECK
    assert outcome.check_findings == FAILED_CHECK["findings"]
    assert outcome.outcome_label == executor.WRITE_LABEL_CHECK_FINDINGS


@pytest.mark.parametrize("original", [MISSING, None, {}, "malformed", {"passed": "maybe"}])
@pytest.mark.parametrize("kind", ["apply_code_fixes", "implement"])
def test_unknown_original_check_can_be_repaired(env, kind, original):
    outcome, saved = _repair(env, kind, original, FAILED_CHECK)
    assert saved["check"] == FAILED_CHECK
    assert outcome.batch_check == FAILED_CHECK["findings"]


def test_a_report_with_no_check_does_not_gain_one(env):
    outcome, saved = _repair(env, "apply_code_fixes", incomplete=True)
    assert "check" not in saved
    assert outcome.batch_check is None


@pytest.mark.parametrize("kind", [
    "apply_suggestions", "apply_task_suggestions", "apply_code_fixes", "implement",
])
def test_every_write_repair_instructs_preserving_the_whole_check(kind):
    prompt = build_schema_repair_prompt(
        kind, "edit", ["missing summary"], previous_reply=json.dumps({"check": FAILED_CHECK}),
    )
    assert "include that entire object unchanged" in " ".join(prompt.split())
    assert "every `listed` command" in prompt
    assert "turn a failure into a pass" in prompt
