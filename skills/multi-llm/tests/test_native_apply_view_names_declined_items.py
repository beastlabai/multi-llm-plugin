"""`status` names a fix the batch declined on the native apply view too
(fix round 22, live run 9 #2).

The decline fixture on Claude Code: the host ran `util batch-done --task-id
apply-fix-batch-0 --status completed --declined "c8fe719158c8d2be=<reason>"`,
the plugin printed `[BATCH_DECLINED]` and recorded the fix `declined` in
`processed_apply-fixes`, and `status` (plain and `--view run`) still read
only "apply-fixes  done on the host (marked completed) — no run record …",
while on Codex and OpenCode the executor's progress line said "not applied,
declined by its batch: c8fe7191 in apply-fix-batch-0 (<reason>)". The native
block now reads the recorded declines (and the items skipped at the user's
request) and prints the executor's own words.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import multi_llm  # noqa: E402
from utils import run_status  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.text_clip import clip, without_closing_mark  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)

TYPED = "plans/p.md"
REASON = ("Not clamping speed to MAX_SPEED is a deliberate plan decision recorded "
          "in the implementation record")


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "plans" / "p.md"
    path.parent.mkdir()
    path.write_text("# Plan\n\n## Step 1\n\nBody.\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                  "commit", "-q", "-m", "plan"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    return path


def _native_apply(plan: Path, review=("cycle 1",)) -> Path:
    """The native emit's batches for apply-suggestions: three items in
    batch 0, one in batch 1, bound to a review-plan `grouped.json`."""
    out = plan.parent / plan.stem
    phase_dir = out / "apply-suggestions"
    phase_dir.mkdir(parents=True, exist_ok=True)
    grouped = out / "review-plan" / "grouped.json"
    grouped.parent.mkdir(parents=True, exist_ok=True)
    grouped.write_text(json.dumps([{"theme": t} for t in review]), encoding="utf-8")
    batches = [
        {"section_key": "s1", "prompt": "Apply.", "suggestions": [
            {"group_id": "g1", "title": "One"},
            {"group_id": "g1b", "title": "One b"},
            {"group_id": "g1c", "title": "One c"}]},
        {"section_key": "s2", "prompt": "Apply.", "suggestions": [
            {"group_id": "g2", "title": "Two"}]},
    ]
    tasks = materialize_write_tasks(
        kind="apply_suggestions", batches=batches, phase_dir=str(phase_dir),
        plan_file=str(plan), grouped_file=str(grouped))
    write_task_file_for_phase(
        kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
        plan_file=str(plan), grouped_file=str(grouped))
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.mark_phase_started("apply-suggestions")
    state.save()
    return phase_dir


def _record(phase_dir, task_id, *flags):
    assert multi_llm.main(["util", "batch-done", "--phase-dir", str(phase_dir),
                           "--task-id", task_id, "--status", "completed",
                           *flags]) == 0


def _finish(plan):
    state = StateManager(plan)
    state.mark_phase_completed("apply-suggestions")
    state.save()


def _status(capsys, *flags) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", TYPED, "--host", "claude-code", *flags])
    return capsys.readouterr().out


DECLINED_CLAUSE = f"1 declined by its batch (not applied): g1 in apply-batch-0 ({REASON})"
#: Fix round 28 (live run 12 #4): after the items count, as the executor's
#: progress line puts it.
DECLINED_LINE = f"      3/4 items applied to plans/p.md; {DECLINED_CLAUSE}"


class TestTheNativeApplyBlockNamesWhatItsBatchesDidNotApply:

    def test_a_declined_item_is_named_in_the_executor_views_words(self, plan, capsys):
        phase_dir = _native_apply(plan)
        _record(phase_dir, "apply-batch-0", "--declined", f"g1={REASON}.")
        _record(phase_dir, "apply-batch-1")
        _finish(plan)

        for flags in ((), ("--view", "run")):
            out = _status(capsys, *flags)
            lines = out.splitlines()
            head = next(i for i, line in enumerate(lines)
                        if line.startswith("  apply-suggestions  done on the host"))
            assert lines[head + 1] == DECLINED_LINE, out
        # The executor view's own clause, for the same record.
        assert run_status.not_applied_item_clauses({"declined_items": [
            {"task_id": "apply-batch-0", "id": "g1", "reason": REASON}]}) == [
            DECLINED_CLAUSE]

    def test_an_item_skipped_at_your_request_too(self, plan, capsys):
        phase_dir = _native_apply(plan)
        _record(phase_dir, "apply-batch-0", "--declined", f"g1={REASON}",
                "--skipped", "g1b")
        out = _status(capsys, "--view", "run")
        # Fix round 26: the clause says "skipped at your request" once.
        # Fix round 28: both after the count, on one line.
        assert any(line.startswith(
            f"      1/4 items applied to plans/p.md; {DECLINED_CLAUSE}; "
            "1 skipped at your request (not applied): g1b in apply-batch-0 "
            "(the batch did not report it applied") for line in out.splitlines()), out

    def test_json_carries_them(self, plan, capsys):
        phase_dir = _native_apply(plan)
        _record(phase_dir, "apply-batch-0", "--declined", f"g1={REASON}",
                "--skipped", "g1b")
        out = _status(capsys, "--view", "run", "--json")
        entry = json.loads(out)["apply_phases"][0]
        assert entry["declined_items"] == [
            {"task_id": "apply-batch-0", "id": "g1", "reason": REASON}]
        assert [e["id"] for e in entry["skipped_items"]] == ["g1b"]
        # Fix round 28.
        assert (entry["items_applied"], entry["items_total"], entry["applied_to"]) == (
            1, 4, "plans/p.md")

    def test_nothing_declined_names_nothing(self, plan, capsys):
        phase_dir = _native_apply(plan)
        _record(phase_dir, "apply-batch-0")
        out = _status(capsys, "--view", "run")
        assert "not applied" not in out, out
        # Fix round 28: the count still reads as the executor's does.
        assert "      3/4 items applied to plans/p.md" in out.splitlines(), out

    def test_no_record_yet_counts_nothing(self, plan, capsys):
        """A phase whose host recorded no batch yet has no count to give:
        "0/4 items applied" would read as four items that failed."""
        _native_apply(plan)
        out = _status(capsys, "--view", "run")
        assert any(line.startswith("  apply-suggestions  ") for line in out.splitlines()), out
        assert "items applied" not in out, out

    def test_an_earlier_reviews_decline_is_not_this_runs(self, plan, capsys):
        """A record of the review before (another `grouped.json`) is not
        named on this cycle's block, as the executor's mirror names only its
        own run's."""
        phase_dir = _native_apply(plan)
        _record(phase_dir, "apply-batch-0", "--declined", f"g1={REASON}")
        _native_apply(plan, review=("cycle 1", "cycle 2"))
        out = _status(capsys, "--view", "run")
        assert any(line.startswith("  apply-suggestions  ")
                   for line in out.splitlines()), out
        assert "declined by its batch" not in out, out


class TestTheNativeApplyCodeFixesBlockCountsItsItems:
    """Fix round 28 (live run 12 #4): Claude Code's native apply-code-fixes
    view, both `status` views, read "1 declined by its batch (not applied):
    c8fe7191 in apply-fix-batch-0 (…)" with no "2/3 items applied;" before
    it, where Codex's executor line read "2/3 items applied; 1 declined by
    its batch (not applied): …". The decline fixture's shape: batch 0 holds
    fixes A and B and declines B, batch 1 holds C."""

    REASON = ("The implementation record and the plan's Decisions say opponent "
              "speed is deliberately NOT clamped to MAX_SPEED, and a test pins it.")

    def _native_fixes(self, plan: Path) -> Path:
        out = plan.parent / plan.stem
        phase_dir = out / "apply-fixes"
        phase_dir.mkdir(parents=True, exist_ok=True)
        grouped = out / "code-review" / "grouped.json"
        grouped.parent.mkdir(parents=True, exist_ok=True)
        grouped.write_text(json.dumps([{"theme": "Opponent speed", "suggestions": [
            {"title": "Opponent speed exceeds MAX_SPEED"}]}]), encoding="utf-8")
        batches = [
            {"file_key": "src/opponent.js", "prompt": "Fix.", "fixes": [
                {"group_id": "c8fe719158c8d2be"}, {"group_id": "c2e33cbae17e32bc"}]},
            {"file_key": "src/race.js", "prompt": "Fix.", "fixes": [
                {"group_id": "5ee23cdd775e8db0"}]},
        ]
        tasks = materialize_write_tasks(
            kind="apply_code_fixes", batches=batches, phase_dir=str(phase_dir),
            plan_file=str(plan), grouped_file=str(grouped))
        write_task_file_for_phase(
            kind="apply_code_fixes", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan), grouped_file=str(grouped))
        state = StateManager(plan)
        state.mark_phase_completed("code-review")
        state.mark_phase_started("apply-code-fixes")
        state.save()
        return phase_dir

    def test_both_views_put_the_count_before_the_declined_clause(self, plan, capsys):
        phase_dir = self._native_fixes(plan)
        _record(phase_dir, "apply-fix-batch-0", "--declined",
                f"c8fe719158c8d2be={self.REASON}")
        _record(phase_dir, "apply-fix-batch-1")
        state = StateManager(plan)
        state.mark_phase_completed("apply-code-fixes")
        state.save()
        expected = (
            "      2/3 items applied; 1 declined by its batch (not applied): "
            "c8fe7191 in apply-fix-batch-0 ("
            + clip(without_closing_mark(self.REASON), 120) + ")")
        for flags in ((), ("--view", "run")):
            lines = _status(capsys, *flags).splitlines()
            head = next(i for i, line in enumerate(lines)
                        if line.startswith("  apply-fixes  done on the host"))
            assert lines[head + 1] == expected, lines
        # The executor's progress line says the same after its batch count.
        mirror = {
            "kind": "apply_code_fixes", "total": 2, "ok": 2,
            "items_total": 3, "items_applied": 2,
            "tasks": [{"task_id": "apply-fix-batch-0", "status": "ok"},
                      {"task_id": "apply-fix-batch-1", "status": "ok"}],
            "declined_items": [{"task_id": "apply-fix-batch-0",
                                "id": "c8fe719158c8d2be", "reason": self.REASON}],
        }
        executor = " ".join(run_status._mirror_text("write_subagents", mirror))
        assert executor.startswith("apply_code_fixes 2/2 batches ok, " + expected.strip()), executor

    def test_a_phase_without_one_target_names_no_file(self, plan, capsys):
        phase_dir = self._native_fixes(plan)
        _record(phase_dir, "apply-fix-batch-0")
        out = _status(capsys, "--view", "run", "--json")
        entry = json.loads(out)["apply_phases"][0]
        assert (entry["items_applied"], entry["items_total"]) == (2, 3)
        assert "applied_to" not in entry
