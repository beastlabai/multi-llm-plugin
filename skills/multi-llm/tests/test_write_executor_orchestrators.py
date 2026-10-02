"""The write executor reached through the real orchestrators, end to end.

``tests/test_write_executor.py`` pins the scheduler and the safety set directly.
This file pins the *wiring*: that ``--exec-subagents`` and
``--dry-run-subagents`` on the actual ``implement`` / ``apply-*`` entry points
materialize write tasks, drive the scheduler, and return its exit code — and that
with both flags off every one of them is byte-for-byte the path that shipped before
the write executor.

Real subprocesses against ``tests/mocks/mock_llm.py`` only; never a live API.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import implement_orchestrator  # noqa: E402
from utils import prior_changes  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_safety  # noqa: E402
from utils.code_fix_batcher import batch_code_fixes  # noqa: E402
from utils.suggestion_batcher import group_suggestions_for_subagents  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    WRITE_TASK_FILENAME,
    materialize_write_tasks,
    write_task_file_for_phase,
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"

PLAN = """# Sample Implementation Plan

## Overview
A plan with three dependent tasks.

## Tasks

### T001: Create the module
Create the module file.
- Depends on: none
- Files to create: src/main.py

### T002: Implement the core
Implement the core functionality.
- Depends on: T001
- Files to modify: src/main.py

### T003: Add tests
Write unit tests.
- Depends on: T002
- Files to create: tests/test_main.py
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with the plan, a mock PATH and a driver config."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "plans").mkdir()
    plan = root / "plans" / "sample.md"
    plan.write_text(PLAN, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    # A real commit: `implement` records `head_before_implement`, and
    # `git rev-parse HEAD` in a repo with no commits is a hard error.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, env=env)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "codex"
    try:
        target.symlink_to(MOCK_LLM)
    except (OSError, NotImplementedError):  # pragma: no cover
        shutil.copyfile(MOCK_LLM, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", os.pathsep.join(
        [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
    ))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    config = tmp_path / "driver.yaml"
    config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    monkeypatch.chdir(root)

    class Repo:
        pass

    obj = Repo()
    obj.root = root
    obj.plan = plan
    obj.tmp_path = tmp_path
    obj.monkeypatch = monkeypatch
    yield obj
    registry._config = None
    registry._config_key = None


#: What a *compliant* runner attests to: an implement task's `expected_ids` is
#: `[task_id]`, and a batch that moves bytes while attesting to none of its
#: expected ids is a partial application, not a success (D3g/D3n). Extra ids are
#: ignored by the classifier, so one superset serves every batch of the run.
IMPLEMENT_WRITE_IDS = json.dumps(["T001", "T002", "T003"])


def _phase_dir(repo) -> Path:
    from utils.output_handler import get_phase_dir

    return Path(get_phase_dir(repo.plan, "implement"))


def _seed_generate_tasks(repo) -> None:
    """Mark the upstream phases done so `implement` reaches its own work."""
    from utils.state_manager import get_or_create_state

    state = get_or_create_state(repo.plan)
    state.mark_phase_completed("generate-tasks")
    state.mark_phase_completed("apply-suggestions")
    state.save()


class TestImplementOrchestratorWiring:

    def test_exec_subagents_schedules_the_batches_and_finalizes(self, repo):
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ])
        assert code == 0

        phase_dir = _phase_dir(repo)
        # 1. the materialized write task file exists and is v2-shaped
        document = json.loads((phase_dir / WRITE_TASK_FILENAME).read_text())
        assert document["tasks_schema_version"] == 2
        assert document["kind"] == "implement"
        assert [t["task_id"] for t in document["batches"]] == ["T001", "T002", "T003"]
        assert [t["batch_index"] for t in document["batches"]] == [0, 1, 2]
        assert document["batches"][1]["depends_on"] == ["T001"]

        # 2. one batch result per task, named by the TASK ID (a positional name
        #    is what let a resumed run adopt a predecessor's result), and they
        #    are schema-valid
        for task_id in ("T001", "T002", "T003"):
            payload = json.loads(
                (phase_dir / f"implement_batch_{task_id}.json").read_text()
            )
            assert payload["status"] in ("applied", "no-op")
            assert payload["summary"]

        # 3. the audit log alternates dispatched/completed — verification gated
        events = [r["event"] for r in
                  write_safety.AuditLog(str(phase_dir)).records()]
        assert events == ["dispatched", "completed"] * 3

        # 4. finalization ran: phase completed, summary written, plan reference
        from utils.state_manager import get_or_create_state
        state = get_or_create_state(repo.plan)
        assert state.is_phase_completed("implement")
        assert "## Implementation Summary" in repo.plan.read_text()
        report = json.loads((phase_dir / "write_batch_report.json").read_text())
        assert report["finalized"] is True
        assert report["exit"] == 0

    def test_a_stopped_batch_still_seeds_pending_for_every_undispatched_task(
        self, repo
    ):
        """F5: `check_workflow_prerequisites`'s denominator is `len(task_status)`
        (`state.implementation_task_statuses()`), and nothing used to seed it
        before a task's first finish/skip — a batch that stops the loop
        (partial application, D3g/D3n) leaves later batches entirely
        undispatched, and before the seed, entirely absent from `task_status`
        too. A workflow snapshot taken mid-run read `(1/2 tasks)`, then
        `(6/7 tasks)`, on the same 14-task plan this bug was measured against.
        """
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        # Every dispatch attests only to T001, so T002's batch (expected_ids
        # ["T002"]) edits bytes but reports an id it never asked for — a
        # partial application, which stops the loop at the batch boundary.
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ])
        assert code == 3  # the loop stopped at a batch boundary

        from utils.state_manager import get_or_create_state
        state = get_or_create_state(repo.plan)
        # T003's batch was never dispatched — its status is seeded, not
        # absent.
        assert state.get_task_status("T003") == "pending"
        statuses = state.implementation_task_statuses()
        assert set(statuses) == {"T001", "T002", "T003"}
        assert statuses["T001"] == "completed"
        assert statuses["T002"] == "failed"  # recorded by the scheduler itself

    def test_a_resume_redispatches_the_task_the_last_run_recorded_failed(
        self, repo
    ):
        """The orchestrator's own resume, end to end (F2, 2026-09-16).

        Run 1 stops at T002 (a partial application, recorded `failed`). The
        resumed run used to re-materialize only `pending` tasks, so T002 was
        never handed to the scheduler: T003 ran, the phase was stamped
        completed and T002 was absent from every artifact but `task_status`.
        """
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 3

        from utils.state_manager import get_or_create_state
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "failed", "T003": "pending",
        }

        # Run 2: a compliant runner, and the resume the exit-3 message names.
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        log = repo.tmp_path / "calls.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume",
        ]) == 0

        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "completed", "T003": "completed",
        }
        assert state.is_phase_completed("implement")
        dispatched = [
            json.loads(line)["prompt"] for line in log.read_text().splitlines()
            if json.loads(line).get("prompt")
        ]
        assert any("T002" in p for p in dispatched), "T002 was re-dispatched"
        assert not any("T001:" in p.split("\n", 1)[0] for p in dispatched)
        phase_dir = _phase_dir(repo)
        payload = json.loads((phase_dir / "implement_batch_T002.json").read_text())
        assert "T002" in payload["ids"]
        assert payload["verdict"]["ok"] is True

    def test_a_resume_after_a_failed_check_reruns_the_check_before_completing(
        self, repo
    ):
        """F3 end to end: every task applied, `driver.verify_command` failed,
        exit 3. The prescribed `--resume` materializes NO task, and used to
        complete the phase without running the check again."""
        _seed_generate_tasks(repo)
        verify = repo.tmp_path / "verify.sh"
        verify.write_text("#!/bin/sh\necho 'typecheck failed'\nexit 2\n",
                          encoding="utf-8")
        verify.chmod(verify.stat().st_mode | stat.S_IXUSR)
        config = repo.tmp_path / "driver.yaml"
        config.write_text(
            "driver:\n  subagent_runner: codex:gpt-5.5\n"
            f"  verify_command: {verify}\n", encoding="utf-8",
        )
        registry._config = None
        registry._config_key = None
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 3

        from utils.state_manager import get_or_create_state
        state = get_or_create_state(repo.plan)
        assert set(state.implementation_task_statuses().values()) == {"completed"}
        assert not state.is_phase_completed("implement")

        # The resume, with nothing fixed: the check must fail the phase again.
        log = repo.tmp_path / "calls.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume",
        ]) == 3
        assert not log.exists() or not log.read_text().strip(), "nothing re-dispatched"
        state = get_or_create_state(repo.plan)
        assert not state.is_phase_completed("implement")
        mirror = json.loads((_phase_dir(repo) / ".status.json").read_text())
        assert mirror["progress"]["write_subagents"]["verification"]["exit_code"] == 2

        # The repository fixed: the same resume completes the phase.
        verify.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume",
        ]) == 0
        state = get_or_create_state(repo.plan)
        assert state.is_phase_completed("implement")

    def test_dry_run_subagents_starts_no_child(self, repo):
        _seed_generate_tasks(repo)
        log = repo.tmp_path / "calls.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        before = (repo.root / "src" / "main.py").read_text()

        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--dry-run-subagents",
        ])
        assert code == 0

        phase_dir = _phase_dir(repo)
        report = json.loads((phase_dir / "subagent_dry_run.json").read_text())
        assert len(report["tasks"]) == 3
        assert all(entry["would_dispatch"] for entry in report["tasks"])
        assert all(entry["argv"] for entry in report["tasks"])
        # Zero dispatch and nothing changed.
        prompts = [
            r for r in (json.loads(line) for line in
                        (log.read_text().splitlines() if log.exists() else []))
            if r.get("prompt")
        ]
        assert prompts == []
        assert (repo.root / "src" / "main.py").read_text() == before
        assert not (phase_dir / "pre_batch").exists()

    def test_with_the_flag_off_the_orchestrator_emits_task_json_as_before(self, repo):
        """The additivity assertion: the native path is untouched."""
        _seed_generate_tasks(repo)
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes",
        ])
        assert code == 0
        phase_dir = _phase_dir(repo)
        for artifact in (WRITE_TASK_FILENAME, "subagent_audit.jsonl",
                         "write_batch_report.json", "subagent_dry_run.json"):
            assert not (phase_dir / artifact).exists(), artifact
        assert not (phase_dir / "pre_batch").exists()
        assert "## Implementation Summary" not in repo.plan.read_text()

    def test_a_seeded_human_task_resolves_to_skip_dependents_under_yes(self, repo):
        """The orchestrator's resolved strategy reaches the scheduler (D6 row 8).

        A human task is seeded on the decomposed graph rather than in the plan text:
        the markdown parser has no human marker — ``executor_role: human`` comes from
        task *generation* — so seeding it here is what a generated task list would
        have produced, without depending on a generator that lands in Phase 3.
        """
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)

        original = implement_orchestrator.TaskDecomposer

        class SeededDecomposer(original):
            def parse_plan(self, content):
                tasks = super().parse_plan(content)
                # T002 becomes the manual step; T003 depends on it transitively.
                self.tasks["T002"].executor_role = "human"
                self.tasks["T002"].subagent_type = "human"
                return tasks

        repo.monkeypatch.setattr(
            implement_orchestrator, "TaskDecomposer", SeededDecomposer
        )
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ])
        # `--yes` resolves the strategy to skip-dependents, so the run completes
        # with the human task AND its dependents skipped rather than blocking — the
        # point being that it resolves, never asks.
        assert code == 0
        report = json.loads(
            (_phase_dir(repo) / "write_batch_report.json").read_text()
        )
        assert report["human_task_strategy"] == "skip-dependents"
        assert set(report["skipped_tasks"]) == {"T002", "T003"}
        assert report["pending_human_tasks"] == []
        # Only the unblocked task was dispatched.
        assert [t["task_id"] for t in report["tasks"]] == ["T001"]

    def test_a_detached_run_hands_back_its_artifacts_and_resolution(
        self, repo, monkeypatch
    ):
        """F7/F8: implement never goes through ``ApplyOrchestratorBase``.

        A fully successful detached run used to report ``artifacts: []`` and no
        ``output_files``, and ``.status.json.resolutions`` stayed empty even
        though the log printed ``[AUTO_RESOLVED] human-task-strategy: …`` — the
        row-8 decision was recorded before this process owned the phase dir.

        Mirrors ``test_unattended.TestApplyPhaseRowsAreRecorded.
        test_a_detached_apply_run_lands_both_in_status_json``: a fake
        ``RunHandle`` is installed as ``detach._ACTIVE`` (claimed via
        ``status_file.claim_run``, never ``.start()``, so no lock/heartbeat/
        signal-handler machinery runs) and the orchestrator is driven in the
        foreground (no ``--detach``/``--internal-worker``) — ``detach.
        record_resolution``/``record_output_files`` only check whether
        ``_ACTIVE`` names this phase dir, not how it got there.
        """
        from utils import detach, status_file

        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)

        original = implement_orchestrator.TaskDecomposer

        class SeededDecomposer(original):
            def parse_plan(self, content):
                tasks = super().parse_plan(content)
                self.tasks["T002"].executor_role = "human"
                self.tasks["T002"].subagent_type = "human"
                return tasks

        repo.monkeypatch.setattr(
            implement_orchestrator, "TaskDecomposer", SeededDecomposer
        )

        phase_dir = _phase_dir(repo)
        handle = detach.RunHandle(
            str(phase_dir), run_id="run-1", mode="implement",
            plan_file=str(repo.plan), detached=True,
        )
        status_file.claim_run(str(phase_dir), {"run_id": "run-1", "state": "running"})
        monkeypatch.setattr(detach, "_ACTIVE", handle)

        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ])
        assert code == 0

        # F8: the row-8 auto-resolution buffered before the detach boundary now
        # reaches `run.resolutions`.
        run = status_file.read_run(str(phase_dir))
        assert [r["site"] for r in run["resolutions"]] == ["human-task-strategy"]
        assert "skip-dependents" in run["resolutions"][0]["decision"]

        # F7: the phase hands back its own artifacts.
        status = json.loads((phase_dir / ".status.json").read_text())
        output_files = status.get("output_files") or {}
        for key in (
            "write_tasks", "write_batch_report", "subagent_batch_report",
            "prior_changes", "summary",
        ):
            assert key in output_files, output_files
            assert Path(output_files[key]).is_file()
        touched = [
            path for key, path in output_files.items()
            if key.startswith("touched_file_")
        ]
        assert any(Path(path).name == "main.py" for path in touched), output_files

    def test_force_strategy_skip_continue_records_the_override(self, repo):
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        original = implement_orchestrator.TaskDecomposer

        class SeededDecomposer(original):
            def parse_plan(self, content):
                tasks = super().parse_plan(content)
                self.tasks["T002"].executor_role = "human"
                self.tasks["T002"].subagent_type = "human"
                return tasks

        repo.monkeypatch.setattr(
            implement_orchestrator, "TaskDecomposer", SeededDecomposer
        )
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
            "--force-strategy", "skip-continue",
        ])
        assert code == 0
        report = json.loads(
            (_phase_dir(repo) / "write_batch_report.json").read_text()
        )
        assert report["human_task_strategy"] == "skip-continue"
        assert set(report["skipped_tasks"]) == {"T002"}
        # T003's human dependency was treated as satisfied, and that is RECORDED.
        assert report["dependency_overrides"] == {"T003": ["T002"]}
        assert {t["task_id"] for t in report["tasks"]} == {"T001", "T003"}


class TestApplyOrchestratorWiring:
    """The three apply phases declare a write kind and reach the same scheduler."""

    @pytest.mark.parametrize("module_name,kind", [
        ("apply_suggestions_orchestrator", "apply_suggestions"),
        ("apply_task_suggestions_orchestrator", "apply_task_suggestions"),
        ("apply_code_fixes_orchestrator", "apply_code_fixes"),
    ])
    def test_each_apply_phase_declares_its_write_kind(self, module_name, kind):
        import importlib

        module = importlib.import_module(module_name)
        classes = [
            obj for obj in vars(module).values()
            if isinstance(obj, type) and getattr(obj, "write_kind", "") == kind
        ]
        assert classes, f"{module_name} declares no class with write_kind={kind!r}"

    def test_the_shared_artifact_phases_snapshot_the_file_they_edit(self, repo):
        """apply-suggestions snapshots the plan; apply-task-suggestions the tasks file.

        Getting this wrong is silent: snapshotting the plan for a phase that edits
        `tasks.md` would make every batch report "unchanged" and every outcome read
        as a clean failure.
        """
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        suggestions = ApplySuggestionsOrchestrator.__new__(
            ApplySuggestionsOrchestrator
        )
        suggestions.plan_path = str(repo.plan)
        assert suggestions.write_phase_artifact() == str(repo.plan)

        tasks = ApplyTaskSuggestionsOrchestrator.__new__(
            ApplyTaskSuggestionsOrchestrator
        )
        tasks.plan_path = str(repo.plan)
        tasks.tasks_file = str(repo.root / "plans" / "tasks.md")
        assert tasks.write_phase_artifact() == str(repo.root / "plans" / "tasks.md")


class TestBridge:
    """One conversion for all four phases, with stable prior-changes keys."""

    def test_apply_suggestions_declares_the_plan_and_a_stable_key(self, repo):
        tasks = materialize_write_tasks(
            kind="apply_suggestions",
            batches=[
                {"prompt": "p0 {prior_changes_context}", "section_key": "step_1",
                 "suggestions": [{"group_id": "abc"}]},
                {"prompt": "p1 {prior_changes_context}", "section_key": "step_2",
                 "suggestions": [{"group_id": "def"}]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        assert [t["output_name"] for t in tasks] == [
            "apply_batch_0.json", "apply_batch_1.json",
        ]
        # `{section_key}#{digest of the batch's member ids}` — the section key
        # alone is NOT a batch identity (one section spills into several batches),
        # and `prior_changes.append` is idempotent on `(id, phase)`.
        assert [t["prior_changes_id"] for t in tasks] == [
            "step_1#ba7816bf", "step_2#cb8379ac",
        ]
        assert all(t["declared_files"] == [str(repo.plan.resolve())] for t in tasks)
        assert [t["expected_ids"] for t in tasks] == [["abc"], ["def"]]
        assert all(t["can_parallelize"] is False for t in tasks)
        assert all(t["context_mode"] == "workspace-read" for t in tasks)

    def test_code_fix_batches_are_keyed_and_may_parallelize(self, repo):
        tasks = materialize_write_tasks(
            kind="apply_code_fixes",
            batches=[
                {"prompt": "p {prior_changes_context}", "file_key": "src/a.py",
                 "fixes": [{"id": "i1"}]},
                {"prompt": "p {prior_changes_context}", "file_key": "src/b.py",
                 "fixes": [{"id": "i2"}]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        # `{file_key}#{digest of the batch's member ids}`: the file key names the
        # partition, never the batch — see TestPriorChangesIdsAreUniquePerBatch.
        assert [t["prior_changes_id"] for t in tasks] == [
            "src/a.py#4cd9b767", "src/b.py#420fce31",
        ]
        assert [t["declared_files"] for t in tasks] == [["src/a.py"], ["src/b.py"]]
        assert all(t["can_parallelize"] is True for t in tasks)

    def test_implement_tasks_keep_their_edges_and_get_unique_result_names(self, repo):
        tasks = materialize_write_tasks(
            kind="implement",
            batches=[
                {"batch_index": 0, "can_parallelize": True, "tasks": [
                    {"id": "T001", "prompt": "a", "files_to_create": ["src/a.py"],
                     "depends_on": [], "title": "A"},
                    {"id": "T002", "prompt": "b", "files_to_create": ["src/b.py"],
                     "depends_on": [], "title": "B"},
                ]},
                {"batch_index": 1, "can_parallelize": False, "tasks": [
                    {"id": "T003", "prompt": "c", "files_to_modify": ["src/a.py"],
                     "depends_on": ["T001"], "title": "C"},
                ]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        assert [t["task_id"] for t in tasks] == ["T001", "T002", "T003"]
        assert [t["batch_index"] for t in tasks] == [0, 0, 1]
        # Two tasks in one batch cannot share a result filename.
        names = [t["output_name"] for t in tasks]
        assert len(set(names)) == 3, names
        assert tasks[2]["depends_on"] == ["T001"]
        assert tasks[0]["can_parallelize"] is True
        assert tasks[2]["can_parallelize"] is False

    def test_a_human_implement_task_carries_the_human_role(self, repo):
        tasks = materialize_write_tasks(
            kind="implement",
            batches=[{"batch_index": 0, "can_parallelize": False, "tasks": [
                {"id": "T001", "prompt": "manual", "executor_role": "human",
                 "depends_on": [], "title": "Rotate key",
                 "files_to_modify": ["src/a.py"]},
            ]}],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        assert tasks[0]["executor_role"] == "human"
        assert tasks[0]["is_human"] is True
        assert tasks[0]["subagent_type"] == "human"

    def test_a_tasks_file_stamps_tasks_sha256_on_every_task_and_the_document(
        self, repo
    ):
        """F8: apply-task-suggestions' tasks.md invalidates a stored task file too.

        ``build_inputs`` already supports ``tasks_file=`` (review-tasks' own
        validation tasks use it), but neither this bridge's per-task ``inputs``
        nor ``write_tasks.json``'s document-level ``inputs`` forwarded it, so a
        hand-edited tasks.md never invalidated a stored write-task file the way
        an edited plan already does.
        """
        tasks_file = repo.root / "plans" / "tasks.md"
        tasks_file.write_text("# Tasks\n\n### Task T001\n", encoding="utf-8")
        tasks = materialize_write_tasks(
            kind="apply_task_suggestions",
            batches=[
                {"prompt": "p {prior_changes_context}", "section_key": "step_1",
                 "suggestions": [{"group_id": "abc"}]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
            tasks_file=str(tasks_file),
        )
        assert "tasks_sha256" in tasks[0]["inputs"]

        task_file_path = write_task_file_for_phase(
            kind="apply_task_suggestions",
            tasks=tasks,
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
            tasks_file=str(tasks_file),
        )
        document = json.loads(Path(task_file_path).read_text(encoding="utf-8"))
        assert document["inputs"]["tasks_sha256"] == tasks[0]["inputs"]["tasks_sha256"]

        # Editing tasks.md changes the hash, exactly like an edited plan does.
        tasks_file.write_text("# Tasks\n\n### Task T001\n### Task T002\n", encoding="utf-8")
        retasked = materialize_write_tasks(
            kind="apply_task_suggestions",
            batches=[
                {"prompt": "p {prior_changes_context}", "section_key": "step_1",
                 "suggestions": [{"group_id": "abc"}]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
            tasks_file=str(tasks_file),
        )
        assert retasked[0]["inputs"]["tasks_sha256"] != tasks[0]["inputs"]["tasks_sha256"]

    def test_no_tasks_file_means_no_tasks_sha256(self, repo):
        """apply-suggestions/apply-code-fixes edit the plan, not a tasks file."""
        tasks = materialize_write_tasks(
            kind="apply_suggestions",
            batches=[
                {"prompt": "p {prior_changes_context}", "section_key": "step_1",
                 "suggestions": [{"group_id": "abc"}]},
            ],
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        assert "tasks_sha256" not in tasks[0]["inputs"]

        task_file_path = write_task_file_for_phase(
            kind="apply_suggestions",
            tasks=tasks,
            phase_dir=str(repo.root / "phase"),
            plan_file=str(repo.plan),
        )
        document = json.loads(Path(task_file_path).read_text(encoding="utf-8"))
        assert "tasks_sha256" not in document["inputs"]


class TestPriorChangesIdsAreUniquePerBatch:
    """The regression: many batches per key, one prior-changes entry each.

    ``prior_changes.append`` is idempotent on ``(id, phase)`` — deliberately, so a
    re-dispatched batch cannot double its summary. Keying that id on the bare
    ``file_key``/``section_key`` therefore *drops* every batch after the first on
    the same file or section, silently and at exit 0, leaving each later batch to
    read a prior-changes context missing its own predecessors. Both batchers
    routinely emit several batches per key (``MAX_FIXES_PER_BATCH`` is 3, the char
    caps close a batch sooner, and each security/HIGH fix is isolated), so this is
    the normal case, not an edge one.
    """

    @staticmethod
    def _fix_batches(count, file_key="packaging/opencode/src/index.mjs"):
        return [
            {"prompt": "p {prior_changes_context}", "file_key": file_key,
             "fixes": [{"id": f"fix-{n}-{k}"} for k in range(3)]}
            for n in range(count)
        ]

    @staticmethod
    def _suggestion_batches(count, section_key="step_3"):
        return [
            {"prompt": "p {prior_changes_context}", "section_key": section_key,
             "suggestions": [{"group_id": f"g-{n}-{k}"} for k in range(2)]}
            for n in range(count)
        ]

    def _materialize(self, repo, kind, batches, sub="phase"):
        return materialize_write_tasks(
            kind=kind, batches=batches,
            phase_dir=str(repo.root / sub), plan_file=str(repo.plan),
        )

    def test_many_code_fix_batches_on_one_file_get_distinct_ids(self, repo):
        tasks = self._materialize(repo, "apply_code_fixes", self._fix_batches(19))
        ids = [t["prior_changes_id"] for t in tasks]
        assert len(set(ids)) == 19, ids
        assert all(i.startswith("packaging/opencode/src/index.mjs#") for i in ids)

    @pytest.mark.parametrize("kind", ["apply_suggestions", "apply_task_suggestions"])
    def test_many_suggestion_batches_in_one_section_get_distinct_ids(self, repo, kind):
        tasks = self._materialize(repo, kind, self._suggestion_batches(7))
        ids = [t["prior_changes_id"] for t in tasks]
        assert len(set(ids)) == 7, ids
        assert all(i.startswith("step_3#") for i in ids)

    def test_the_ids_are_stable_across_a_re_materialization(self, repo):
        """Unique is only half of it: an unstable id DOUBLES an entry instead."""
        batches = self._fix_batches(5)
        first = self._materialize(repo, "apply_code_fixes", batches, sub="a")
        second = self._materialize(repo, "apply_code_fixes", list(batches), sub="b")
        assert [t["prior_changes_id"] for t in first] == [
            t["prior_changes_id"] for t in second
        ]

    def test_a_resumed_subset_keeps_each_batch_s_own_id(self, repo):
        """A re-dispatch of batches 3..5 alone must not renumber them.

        This is what a loop-index id gets wrong: the surviving batches would take
        the ids of the ones that already ran, and every summary would land on an
        `(id, phase)` pair already in the log — dropped, again silently.
        """
        batches = self._fix_batches(6)
        full = self._materialize(repo, "apply_code_fixes", batches, sub="full")
        tail = self._materialize(repo, "apply_code_fixes", batches[3:], sub="tail")
        assert [t["prior_changes_id"] for t in tail] == [
            t["prior_changes_id"] for t in full[3:]
        ]

    def test_every_batch_on_one_file_is_recorded_in_the_log(self, repo, tmp_path):
        """The user-visible bug: 19 batches, 19 entries — not 1.

        Asserts the entry *count* through the real `prior_changes.append`, because
        distinct ids are only the mechanism; the log is the artifact every later
        batch reads.
        """
        tasks = self._materialize(repo, "apply_code_fixes", self._fix_batches(19))
        out_dir = str(tmp_path / "out")
        written = [
            prior_changes.append(
                out_dir, task["prior_changes_id"], "apply-fixes",
                f"batch {n} applied",
            )
            for n, task in enumerate(tasks)
        ]
        assert all(written)
        assert len(prior_changes.entries(out_dir)) == 19

    def test_a_redispatched_batch_still_appends_only_once(self, repo, tmp_path):
        """The idempotency the ids must not break, asserted on the same log."""
        tasks = self._materialize(repo, "apply_code_fixes", self._fix_batches(4))
        out_dir = str(tmp_path / "out")
        for task in tasks:
            prior_changes.append(out_dir, task["prior_changes_id"], "apply-fixes", "s")
        again = prior_changes.append(
            out_dir, tasks[2]["prior_changes_id"], "apply-fixes", "s again"
        )
        assert again is False
        assert len(prior_changes.entries(out_dir)) == 4

    def test_the_native_batch_dict_carries_the_bridge_s_own_id(self, repo):
        """Both invokers, one id — computed once in Python, read off the batch.

        The native path (an agent following `instructions/apply-code-fixes.md`)
        passes `batch.prior_changes_id` to `util prior-changes append`; the Python
        path materializes a write task. They must be the same string or a run that
        switched invokers mid-phase would append the same batch twice.
        """
        fixes = [
            {"id": f"i{n}", "file": "src/api.ts", "description": "d" * 40,
             "type": "bug", "importance": "MEDIUM"}
            for n in range(7)
        ]
        batches = batch_code_fixes(fixes)
        assert len(batches) > 1, "fixture must span more than one batch per file"
        serialized = [
            {**b.to_dict(), "prompt": "p {prior_changes_context}"} for b in batches
        ]
        tasks = self._materialize(repo, "apply_code_fixes", serialized)
        assert [b["prior_changes_id"] for b in serialized] == [
            t["prior_changes_id"] for t in tasks
        ]
        assert len({t["prior_changes_id"] for t in tasks}) == len(tasks)

    def test_the_native_suggestion_batch_carries_the_bridge_s_own_id(self, repo):
        """The same equivalence for the two suggestion phases' shared batcher."""
        suggestions = [
            {"group_id": f"g{n}", "reference": "### Step 3: Do it",
             "description": "d" * 40, "type": "modification",
             "importance": "MEDIUM"}
            for n in range(9)
        ]
        batches = group_suggestions_for_subagents(suggestions)
        assert len(batches) > 1, "fixture must span more than one batch per section"
        serialized = [
            {**b.to_dict(), "prompt": "p {prior_changes_context}"} for b in batches
        ]
        for kind in ("apply_suggestions", "apply_task_suggestions"):
            tasks = self._materialize(repo, kind, serialized, sub=f"phase-{kind}")
            assert [b["prior_changes_id"] for b in serialized] == [
                t["prior_changes_id"] for t in tasks
            ]

    def test_implement_still_keys_on_the_task_id(self, repo):
        """Untouched: one materialized task per decomposed task, already unique."""
        tasks = materialize_write_tasks(
            kind="implement",
            batches=[{"batch_index": 0, "can_parallelize": False, "tasks": [
                {"id": "T001", "prompt": "a", "files_to_create": ["src/a.py"]},
                {"id": "T002", "prompt": "b", "files_to_create": ["src/b.py"]},
            ]}],
            phase_dir=str(repo.root / "phase"), plan_file=str(repo.plan),
        )
        assert [t["prior_changes_id"] for t in tasks] == ["T001", "T002"]


class TestUtilRouting:
    """`restore-batch` is reachable through the one entry point."""

    def test_restore_batch_is_a_util_command(self):
        import multi_llm

        assert "restore-batch" in multi_llm.UTIL_COMMANDS
        module, attr, summary = multi_llm.UTIL_COMMANDS["restore-batch"]
        assert (module, attr) == ("utils.restore_batch", "main")
        assert summary
        assert "restore-batch" in multi_llm.render_util_help()

    def test_it_is_not_a_user_facing_mode(self):
        import modes as mode_table

        assert mode_table.resolve("restore-batch") is None
        assert "restore-batch" not in mode_table.render_argument_hint()


# ===========================================================================
# D4 — every executor-backed mode is genuinely detachable
# ===========================================================================

UV_STUB = '''#!/usr/bin/env python3
"""A `uv` that asserts the argv shape D4e specifies, then execs it."""
import os
import sys

argv = sys.argv[1:]
assert argv[0] == "run", argv
# The uv flags sit between `run` and `--project`, and the worker must carry the
# same ones the launcher did. `--no-dev` is the load-bearing half: without it
# `uv run` syncs the dev group, so a careful launch would hand off to a child
# that pulled 17 packages -- or failed outright offline -- with the failure
# moving out of the launcher and into a log.
flags = []
index = 1
while index < len(argv) and argv[index] != "--project":
    flags.append(argv[index])
    index += 1
assert "--frozen" in flags, argv
assert "--no-dev" in flags, argv
assert argv[index] == "--project", argv
assert os.path.isdir(argv[index + 1]), argv
assert argv[index + 2] == "--", argv
assert argv[index + 3] == "python", argv
assert argv[index + 4].endswith("multi_llm.py"), argv
os.execv(sys.executable, [sys.executable, *argv[index + 4:]])
'''


class TestDetachBoundary:
    """`--detach` on a write mode reaches the SAME boundary the fan-out uses.

    Registering ``supports_detach`` without wiring the boundary would give a caller
    a run to poll that was never detached — the failure mode the mode table's
    exit-3 exists to prevent — so this asserts the call, not just the column.
    """

    def test_implement_crosses_the_boundary_after_preflight(self, repo,
                                                            monkeypatch):
        _seed_generate_tasks(repo)
        seen = {}

        def fake_maybe_detach(**kwargs):
            seen.update(kwargs)
            return 0  # pretend the invoker forked

        from utils import review_orchestrator_base

        monkeypatch.setattr(
            review_orchestrator_base, "maybe_detach", fake_maybe_detach
        )
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--detach",
        ])
        assert code == 0
        assert seen["mode"] == "implement"
        assert seen["phase"] == "implement"
        assert Path(seen["plan_file"]) == repo.plan.absolute()
        assert Path(seen["phase_dir"]).is_dir()
        # Nothing was dispatched: the boundary is BEFORE the first child, so a bad
        # flag or a missing runner fails on the caller's terminal.
        assert not (Path(seen["phase_dir"]) / "subagent_audit.jsonl").exists()

    @pytest.mark.parametrize("module_name,mode,phase", [
        ("apply_suggestions_orchestrator", "apply-suggestions", "apply-suggestions"),
        ("apply_task_suggestions_orchestrator", "apply-task-suggestions",
         "apply-task-suggestions"),
        ("apply_code_fixes_orchestrator", "apply-code-fixes", "apply-fixes"),
    ])
    def test_each_apply_phase_names_its_own_mode_word_at_the_boundary(
        self, repo, monkeypatch, module_name, mode, phase
    ):
        """apply-code-fixes' phase name and mode word differ; both must be right."""
        import importlib

        module = importlib.import_module(module_name)
        cls = next(
            obj for obj in vars(module).values()
            if isinstance(obj, type) and getattr(obj, "write_kind", "")
        )
        probe = cls.__new__(cls)
        probe.plan_path = str(repo.plan)
        probe.out_dir = str(repo.root / "out")
        probe.args = type("A", (), {"detach": True, "internal_worker": False})()

        seen = {}

        from utils import review_orchestrator_base

        monkeypatch.setattr(
            review_orchestrator_base, "maybe_detach",
            lambda **kwargs: seen.update(kwargs) or 0,
        )
        assert probe.maybe_detach_write_phase() == 0
        assert seen["mode"] == mode
        assert seen["phase"] == phase
        assert seen["phase_dir"].endswith(phase)

    def test_without_detach_the_boundary_is_not_crossed(self, repo, monkeypatch):
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from utils import review_orchestrator_base

        def explode(**kwargs):  # pragma: no cover - must never run
            raise AssertionError("maybe_detach must not be called without --detach")

        monkeypatch.setattr(review_orchestrator_base, "maybe_detach", explode)
        probe = ApplySuggestionsOrchestrator.__new__(ApplySuggestionsOrchestrator)
        probe.plan_path = str(repo.plan)
        probe.out_dir = str(repo.root / "out")
        probe.args = type("A", (), {"detach": False, "internal_worker": False})()
        assert probe.maybe_detach_write_phase() is None


@pytest.mark.timeout(300)
class TestFullStackDetachedImplement:

    def test_a_detached_implement_run_reaches_exactly_one_terminal_state(
        self, repo, tmp_path
    ):
        """The Done-when row, observed rather than inferred.

        A real double-spawned worker schedules three real mock dispatches, and
        ``status`` is polled until the run is terminal. ``run.state`` must be
        ``done`` with a recorded ``exit_code`` — never left ``running``.
        """
        if os.name == "nt":  # pragma: no cover - POSIX double-spawn path
            pytest.skip("the POSIX double-spawn path; Windows is covered in CI")
        import time

        from utils import detach, status_file

        _seed_generate_tasks(repo)
        stub = tmp_path / "uv-stub"
        stub.write_text(UV_STUB, encoding="utf-8")
        stub.chmod(0o755)

        skill_root = Path(__file__).parent.parent
        env = {
            **os.environ,
            detach.UV_ENV: str(stub),
            "MULTI_LLM_HEARTBEAT_INTERVAL_S": "1",
            "MOCK_LLM_EDITS": json.dumps([
                {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
            ]),
            "MOCK_LLM_WRITE_IDS": IMPLEMENT_WRITE_IDS,
        }
        launch = subprocess.run(
            [sys.executable, str(skill_root / "multi_llm.py"), "implement",
             str(repo.plan), "--detach", "--exec-subagents"],
            cwd=str(skill_root), env=env, capture_output=True, text=True,
            timeout=120,
        )
        assert launch.returncode == 0, launch.stdout + launch.stderr
        assert "=== DETACHED RUN STARTED ===" in launch.stdout

        phase_dir = _phase_dir(repo)
        deadline = time.time() + 120
        run = status_file.read_run(str(phase_dir)) or {}
        while run.get("state") in (None, "starting", "running") and \
                time.time() < deadline:
            time.sleep(0.5)
            run = status_file.read_run(str(phase_dir)) or {}

        assert run["state"] == "done", run
        assert run["exit_code"] == 0
        assert run["detached"] is True
        # Per-task write progress is observable with no chat context at all.
        status = json.loads((phase_dir / ".status.json").read_text())
        progress = status["progress"]["write_subagents"]
        assert progress["total"] == 3
        assert progress["finalized"] is True
        assert [t["task_id"] for t in progress["tasks"]] == ["T001", "T002", "T003"]

    def test_resume_after_a_mid_batch_kill_redispatches_only_the_incomplete(
        self, repo, tmp_path
    ):
        """A killed run leaves artifacts, and artifacts define what is done (D3i).

        The first run is killed after batch 0 by pointing the mock at a result that
        fails batch 1; the second run then re-dispatches ONLY batches 1 and 2, with
        batch 0's summary replayed from ``prior_changes.jsonl`` on disk rather than
        reconstructed from a memory the killed process took with it.
        """
        _seed_generate_tasks(repo)
        log = tmp_path / "calls.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))

        # Run 1: batch 0 applies, then a timeout kills the run mid-batch-1.
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# t1\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        phase_dir = _phase_dir(repo)
        from utils.write_batch_scheduler import WriteBatchScheduler
        from utils.write_phase_bridge import materialize_write_tasks

        tasks = materialize_write_tasks(
            kind="implement",
            batches=[
                {"batch_index": i, "can_parallelize": False, "tasks": [
                    # implement prompts carry the token too since F6 (2026-09-16).
                    {"id": f"T00{i+1}", "prompt": f"do {i} {{prior_changes_context}}",
                     "files_to_modify": ["src/main.py"], "depends_on": [],
                     "title": f"Task {i}"},
                ]}
                for i in range(3)
            ],
            phase_dir=str(phase_dir),
            plan_file=str(repo.plan),
        )
        first = WriteBatchScheduler(
            tasks=tasks[:1], phase_dir=str(phase_dir), kind="implement",
            output_dir=str(repo.root / ".multi-llm"), plan_file=str(repo.plan),
            anchor=str(repo.plan), repo_root=str(repo.root), finalize=False,
            log=lambda line: None,
        ).run()
        assert first.exit_code == 0, ([(o.status, o.write_outcome, o.reason) for o in first.outcomes], first.messages)

        # The "kill": the process is gone, batches 1 and 2 have no result file.
        assert (phase_dir / "implement_batch_T001.json").is_file()
        assert not (phase_dir / "implement_batch_T002.json").exists()
        log.unlink(missing_ok=True)

        # Run 2: resume.
        second = WriteBatchScheduler(
            tasks=tasks, phase_dir=str(phase_dir), kind="implement",
            output_dir=str(repo.root / ".multi-llm"), plan_file=str(repo.plan),
            anchor=str(repo.plan), repo_root=str(repo.root), resume=True,
            log=lambda line: None,
        ).run()
        assert second.exit_code == 0, second.messages
        skipped = [o.task_id for o in second.outcomes if o.skipped]
        assert skipped == ["T001"]
        dispatched = [
            json.loads(line)["prompt"]
            for line in (log.read_text().splitlines() if log.exists() else [])
            if json.loads(line).get("prompt")
        ]
        assert dispatched, "the incomplete batches must be re-dispatched"
        assert not any("do 0" in p for p in dispatched)
        assert any("do 1" in p for p in dispatched)
        assert any("do 2" in p for p in dispatched)
        # Exactly one terminal state, and it is the successful one.
        assert second.finalized is True
        assert second.stopped_at_batch is None

    def test_a_resume_over_a_shorter_task_list_still_runs_the_unrun_tasks(
        self, repo, tmp_path
    ):
        """The resume the implement orchestrator actually builds, end to end.

        `implement_orchestrator` re-materializes from
        `format_batches_for_output(skip_completed=--resume)`, so run 2's list holds
        only the tasks that never finished — it is SHORTER than run 1's, which is
        the case `test_resume_after_a_mid_batch_kill_redispatches_only_the_incomplete`
        never covered (it hands the scheduler the same list twice). Under the old
        positional result name every survivor inherited a predecessor's file and
        was skipped: measured 2026-09-16, T006-T010 were reported `applied` at
        exit 0 having never been dispatched.
        """
        from utils.write_batch_scheduler import WriteBatchScheduler
        from utils.write_phase_bridge import materialize_write_tasks

        _seed_generate_tasks(repo)
        log = tmp_path / "calls.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# t1\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        phase_dir = _phase_dir(repo)

        def materialize(task_ids):
            return materialize_write_tasks(
                kind="implement",
                batches=[
                    {"batch_index": index, "can_parallelize": False, "tasks": [
                        {"id": task_id,
                         "prompt": f"do {task_id}\n\n{{prior_changes_context}}",
                         "files_to_modify": ["src/main.py"], "depends_on": [],
                         "title": task_id},
                    ]}
                    for index, task_id in task_ids
                ],
                phase_dir=str(phase_dir), plan_file=str(repo.plan),
            )

        all_three = [(0, "T001"), (1, "T002"), (2, "T003")]
        first = WriteBatchScheduler(
            tasks=materialize(all_three)[:1], phase_dir=str(phase_dir),
            kind="implement", output_dir=str(repo.root / ".multi-llm"),
            plan_file=str(repo.plan), anchor=str(repo.plan),
            repo_root=str(repo.root), finalize=False, log=lambda line: None,
        ).run()
        assert first.exit_code == 0, ([(o.status, o.write_outcome, o.reason) for o in first.outcomes], first.messages)
        log.unlink(missing_ok=True)

        # The resume: T001 completed, so the orchestrator materializes T002/T003
        # ONLY — and their batch_index values keep their original 1 and 2.
        remaining = materialize([(1, "T002"), (2, "T003")])
        second = WriteBatchScheduler(
            tasks=remaining, phase_dir=str(phase_dir), kind="implement",
            output_dir=str(repo.root / ".multi-llm"), plan_file=str(repo.plan),
            anchor=str(repo.plan), repo_root=str(repo.root), resume=True,
            finalize=False, log=lambda line: None,
        ).run()

        assert second.exit_code == 0, second.messages
        assert [o.task_id for o in second.outcomes] == ["T002", "T003"]
        assert [o.skipped for o in second.outcomes] == [False, False]
        dispatched = [
            json.loads(line)["prompt"]
            for line in (log.read_text().splitlines() if log.exists() else [])
            if json.loads(line).get("prompt")
        ]
        assert any("do T002" in p for p in dispatched)
        assert any("do T003" in p for p in dispatched)
        assert (phase_dir / "implement_batch_T002.json").is_file()
        assert (phase_dir / "implement_batch_T003.json").is_file()


# ---------------------------------------------------------------------------
# #28, the apply half: an apply phase's summary.md and write_batch_report.json
# on `--resume` (three-host racing run, 2026-10-01)
# ---------------------------------------------------------------------------

APPLY_PLAN = """# Apply Plan

## Step 1: Alpha
Alpha text.

## Step 2: Beta
Beta text.

## Step 3: Gamma
Gamma text.
"""

APPLY_SECTIONS = ("Alpha", "Beta", "Gamma")


def _groups(prefix):
    """One HIGH suggestion per plan section, so each lands in a batch of its own."""
    return [
        {"theme": f"{prefix} theme {n}", "category": "c", "models": ["m"],
         "suggestions": [{
             "title": f"{prefix} suggestion {n}", "desc": f"Do {prefix} {n}.",
             "type": "addition", "reference": f"## Step {n + 1}: {name}",
             "importance": "HIGH", "source_model": "m",
         }]}
        for n, name in enumerate(APPLY_SECTIONS)
    ]


def _validation(statuses):
    return {
        "metadata": {"schema_version": "2.0", "validated_at": "2026-10-02T10:00:00",
                     "model": "m", "plan_hash": "x", "total_groups": len(statuses)},
        "groups": [{"group_index": i, "status": s, "reason": "r", "confidence": 0.9}
                   for i, s in enumerate(statuses)],
    }


class TestAnApplyResumeKeepsTheEarlierRunsRows:
    """#28: implement's `--resume` kept the first run's rows; the apply phases'
    `summary.md` did not. It was rendered from this run's outcomes, so a batch
    the stopped run had applied read as this run's (`Runner: executor`, 0.0s,
    no marker), and a batch whose positional `apply-batch-N` id a changed
    selection handed to other items vanished from both reports."""

    def _review(self, repo, prefix, statuses=("valid",) * 3):
        """A review-plan cycle on disk, completed now."""
        from utils.state_manager import get_or_create_state

        plan = repo.root / "plans" / "apply.md"
        if not plan.exists():
            plan.write_text(APPLY_PLAN, encoding="utf-8")
        review = repo.root / "plans" / "apply" / "review-plan"
        review.mkdir(parents=True, exist_ok=True)
        (review / "grouped.json").write_text(json.dumps(_groups(prefix)), encoding="utf-8")
        (review / "validation.json").write_text(
            json.dumps(_validation(list(statuses))), encoding="utf-8")
        state = get_or_create_state(plan)
        state.mark_phase_completed("review-plan")
        state.save()
        return plan

    def _ids(self, plan):
        """Each batch's member ids, as the phase materializes them."""
        import apply_suggestions_orchestrator as orchestrator

        assert orchestrator.main(
            ["--plan-file", str(plan), "--yes", "--dry-run-subagents"]) == 0
        tasks = json.loads((self._phase_dir(plan) / WRITE_TASK_FILENAME).read_text())
        return [task["expected_ids"] for task in tasks["batches"]]

    @staticmethod
    def _phase_dir(plan):
        return plan.parent / plan.stem / "apply-suggestions"

    def _run(self, repo, plan, attest, *flags):
        import apply_suggestions_orchestrator as orchestrator

        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "plans/apply.md", "mode": "patch", "content": "edit\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(attest))
        return orchestrator.main(
            ["--plan-file", str(plan), "--yes", "--exec-subagents", *flags])

    def _report(self, plan):
        return json.loads(
            (self._phase_dir(plan) / "write_batch_report.json").read_text())["tasks"]

    def _summary(self, plan):
        return (self._phase_dir(plan) / "summary.md").read_text(encoding="utf-8")

    @staticmethod
    def _section(summary, heading):
        """The lines under one `### …` heading, up to the next blank line."""
        lines = summary.splitlines()
        start = lines.index(heading)
        return lines[start:lines.index("", start)]

    def test_a_resume_keeps_the_stopped_runs_finished_batch_in_summary_md(self, repo):
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        # Run 1: batch 0 attested and applied, batch 1 a partial application.
        assert self._run(repo, plan, ids[0]) == 3
        assert self._run(repo, plan, sum(ids, []), "--resume") == 0

        rows = self._report(plan)
        assert [(r["task_id"], bool(r.get("from_earlier_run"))) for r in rows] == [
            ("apply-batch-0", True), ("apply-batch-1", False),
            ("apply-batch-2", False)]
        summary = self._summary(plan)
        earlier = self._section(summary, "### apply-batch-0 (earlier run) — ok (applied)")
        # The earlier run's own row, not this run's resume skip of it.
        assert "**Runner**: codex:gpt-5.5" in earlier[1]
        assert "executor" not in earlier[1]
        assert "1. **one suggestion 0** - Applied" in earlier
        assert "### apply-batch-1 — ok (applied)" in summary
        assert "### apply-batch-2 — ok (applied)" in summary
        # The count lines agree with the table under them.
        assert summary.count("\n### apply-") == len(rows) == 3
        assert ("- Batches processed: 3/3 (1 from an earlier run of this phase, "
                "not dispatched again)") in summary
        assert "- Suggestions applied: 3" in summary

    def test_a_batch_whose_id_a_changed_selection_reused_keeps_its_row(self, repo):
        """The stopped batch's item is skipped before the resume, so its
        `apply-batch-1` id goes to the next item. Its partial edits are still
        in the plan; the row saying so must survive."""
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, ids[0]) == 3
        stopped = next(r for r in self._report(plan) if r["task_id"] == "apply-batch-1")
        assert stopped["write_outcome"] == "partial-application"

        review = repo.root / "plans" / "apply" / "review-plan"
        (review / "validation.json").write_text(
            json.dumps(_validation(["valid", "invalid", "valid"])), encoding="utf-8")
        assert self._run(repo, plan, sum(ids, []), "--resume") == 0

        rows = self._report(plan)
        kept = [r for r in rows if r.get("expected_ids") == ids[1]]
        assert len(kept) == 1
        assert kept[0]["from_earlier_run"] is True
        assert kept[0]["write_outcome"] == "partial-application"
        # The item that now holds the id is this run's own row beside it.
        current = [r for r in rows if r["task_id"] == "apply-batch-1"
                   and not r.get("from_earlier_run")]
        assert [r["expected_ids"] for r in current] == [ids[2]]
        summary = self._summary(plan)
        assert ("### apply-batch-1 (earlier run) — partial_application "
                "(partial-application)") in summary
        assert "### apply-batch-1 — ok (applied)" in summary
        ok = sum(1 for r in rows if r["status"] == "ok")
        assert f"- Batches processed: {ok}/{len(rows)}" in summary
        assert summary.count("\n### apply-") == len(rows)

    def test_a_new_review_cycle_drops_the_old_rows_and_never_leaks_them(self, repo):
        """Cycle 1 stops part-applied; the review is re-run with other
        findings. Neither cycle 2's first run nor its resume may show a row of
        cycle 1, and the resume still keeps cycle 2's own earlier row."""
        plan = self._review(repo, "one")
        old_ids = self._ids(plan)
        assert self._run(repo, plan, old_ids[0]) == 3

        plan = self._review(repo, "two")
        new_ids = self._ids(plan)
        assert self._run(repo, plan, new_ids[0]) == 3
        rows = self._report(plan)
        assert not any(r.get("from_earlier_run") for r in rows)
        assert "(earlier run)" not in self._summary(plan)
        assert "- Batches processed: 1/2" in self._summary(plan)

        assert self._run(repo, plan, sum(new_ids, []), "--resume") == 0
        rows = self._report(plan)
        old = {i for batch in old_ids for i in batch}
        assert not any(set(r.get("expected_ids") or []) & old for r in rows)
        assert [r["expected_ids"] for r in rows if r.get("from_earlier_run")] == [
            new_ids[0]]
        summary = self._summary(plan)
        assert "one suggestion" not in summary
        assert "- Batches processed: 3/3 (1 from an earlier run" in summary

    def test_a_re_stamped_review_reuses_results_without_claiming_an_earlier_run(
        self, repo
    ):
        """Same findings, review re-stamped: the scheduler still skips every
        batch whose result and inputs match, but the old cycle's rows (its
        runner, its duration) are not carried; each row says this run reused
        an earlier result."""
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, [])) == 0
        plan = self._review(repo, "one")
        assert self._run(repo, plan, sum(ids, [])) == 0

        rows = self._report(plan)
        assert not any(r.get("from_earlier_run") for r in rows)
        assert all(r.get("skipped") for r in rows)
        summary = self._summary(plan)
        assert "(earlier run)" not in summary
        assert summary.count("(earlier result, not dispatched again) — ok") == 3
        assert ("- Batches processed: 3/3 (3 earlier result(s) reused, "
                "not dispatched again)") in summary
