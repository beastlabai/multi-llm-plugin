"""A stopped apply run resumed under another batch layout.

Six MEDIUM additions to one plan section: the default layout (four per batch)
makes two batches, `--no-batch` six. Resume evidence is keyed by the batch
(`ResumeEvidence.task_fingerprint`), so under another layout the finished
batch's items read as never done and are dispatched again. This is why
`--max-batch-size` is refused rather than honoured
(`apply_orchestrator_base.refuse_unsupported_batch_size`); `--no-batch` still
changes the layout, and the strict xfail below records what it does today.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.support.write_orchestrators import (  # noqa: E402
    _ApplyRig,
    _validation,
    orchestrator_repo,  # noqa: F401 - the repo fixture
)


def _one_section_groups(count):
    return [
        {"theme": f"theme {n}", "category": "c", "models": ["m"],
         "suggestions": [{
             "title": f"suggestion {n}", "desc": f"Do {n}.",
             "type": "addition", "reference": "## Step 1: Alpha",
             "importance": "MEDIUM", "source_model": "m",
         }]}
        for n in range(count)
    ]


class _LayoutRig(_ApplyRig):

    def _review(self, repo, prefix, statuses=("valid",) * 6):
        plan = super()._review(repo, prefix, statuses)
        review = repo.root / "plans" / "apply" / "review-plan"
        (review / "grouped.json").write_text(
            json.dumps(_one_section_groups(len(statuses))), encoding="utf-8")
        (review / "validation.json").write_text(
            json.dumps(_validation(list(statuses))), encoding="utf-8")
        return plan

    def _ids(self, plan, *flags):
        import apply_suggestions_orchestrator as orchestrator

        assert orchestrator.main(
            ["--plan-file", str(plan), "--yes", "--dry-run-subagents", *flags]) == 0
        tasks = json.loads((self._phase_dir(plan) / "write_tasks.json").read_text())
        return [task["expected_ids"] for task in tasks["batches"]]


class TestNoBatchOnResume(_LayoutRig):

    def test_default_layout_has_two_batches(self, repo):
        plan = self._review(repo, "x")
        assert [len(b) for b in self._ids(plan)] == [4, 2]

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "known gap: a stopped default-layout run resumed with --no-batch "
        "re-sends the four landed items (measured 2026-11-18)"))
    def test_a_no_batch_resume_never_resends_the_landed_batch(self, repo):
        def setup(ok, what):
            # Not an AssertionError: a broken setup must fail the test, not
            # pass as the expected failure.
            if not ok:
                raise RuntimeError(what)

        plan = self._review(repo, "x")
        ids = self._ids(plan)
        setup([len(b) for b in ids] == [4, 2], f"default layout {ids}")
        # Batch 0 (four items) lands, batch 1 attests to nothing: the loop stops.
        setup(self._run(repo, plan, ids[0], content="run1\n") == 3, "first run did not stop")
        setup(self._run(repo, plan, sum(ids, []), "--resume", "--no-batch",
                        content="run2\n") == 0, "the resume did not finish")
        phase_dir = self._phase_dir(plan)
        tasks = {t["task_id"]: t["expected_ids"] for t in json.loads(
            (phase_dir / "write_tasks.json").read_text())["batches"]}
        dispatched = {
            item for row in json.loads((phase_dir / "write_batch_report.json").read_text())["tasks"]
            if not row.get("from_earlier_run") for item in tasks[row["task_id"]]
        }
        # Only the two items of the stopped batch are dispatched again.
        assert dispatched == set(ids[1])


class TestMaxBatchSizeOnResume(_LayoutRig):

    def test_a_resume_with_another_max_batch_size_is_refused_before_anything_runs(self, repo):
        plan = self._review(repo, "x")
        ids = self._ids(plan)
        assert self._run(repo, plan, ids[0], content="run1\n") == 3
        stopped = plan.read_text(encoding="utf-8")
        log = repo.tmp_path / "resume.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        with pytest.raises(SystemExit) as exc:
            self._run(repo, plan, sum(ids, []), "--resume", "--max-batch-size", "1",
                      content="run2\n")
        assert exc.value.code == 2
        assert not log.exists()
        assert plan.read_text(encoding="utf-8") == stopped


def test_the_public_command_refuses_it_as_a_usage_error(tmp_path):
    """Through `multi_llm.py` a usage error is exit 3 (exit 2 means "nothing
    to do"), and nothing is written."""
    skill = Path(__file__).resolve().parent.parent
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans" / "p.md").write_text("# P\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if ".git" not in p.parts)
    result = subprocess.run(
        [sys.executable, str(skill / "multi_llm.py"), "apply-suggestions", "plans/p.md",
         "--max-batch-size", "2"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=120,
        env={**os.environ, "MULTI_LLM_TEST_MODE": "1"},
    )
    assert result.returncode == 3, result.stderr
    assert "--max-batch-size 2 is refused" in result.stderr
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if ".git" not in p.parts)
    assert after == before


def test_a_stored_retry_drops_the_cap_it_was_recorded_with():
    """A run recorded before the refusal may carry a cap that had no effect;
    replayed, it would turn the resume into a usage error."""
    from utils import cli_common

    stored = ["--yes", "--max-batch-size", "2", "--exec-subagents", "--max-batch-size=5",
              "--max-batch", "3", "--max-b=6", "--no-batch"]
    assert cli_common.current_replay_flags(stored, mode="apply-suggestions") == [
        "--yes", "--exec-subagents", "--no-batch"]


def test_the_commit_front_door_refuses_another_cap_before_it_writes(capsys):
    from utils import commit_front_door

    with pytest.raises(SystemExit) as exc:
        commit_front_door._parse_door(
            "apply-commit-fixes", ["--max-batch-size", "5"], door="--apply-commit-fixes")
    assert exc.value.code == 2
    assert "--max-batch-size 5 is refused" in capsys.readouterr().err
    typed = commit_front_door._parse_door(
        "apply-commit-fixes", ["--max-batch-size", "3"], door="--apply-commit-fixes")
    assert typed.namespace.max_batch_size == 3
