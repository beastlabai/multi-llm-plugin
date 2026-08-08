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
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_safety  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    WRITE_TASK_FILENAME,
    materialize_write_tasks,
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

        # 2. one batch result per task, and they are schema-valid
        for index in range(3):
            payload = json.loads(
                (phase_dir / f"implement_batch_{index}.json").read_text()
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

    def test_force_strategy_skip_continue_records_the_override(self, repo):
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
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
        assert [t["prior_changes_id"] for t in tasks] == ["step_1", "step_2"]
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
        assert [t["prior_changes_id"] for t in tasks] == ["src/a.py", "src/b.py"]
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
assert argv[1] == "--project", argv
assert os.path.isdir(argv[2]), argv
assert argv[3] == "--", argv
assert argv[4] == "python", argv
assert argv[5].endswith("multi_llm.py"), argv
os.execv(sys.executable, [sys.executable, *argv[5:]])
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
        phase_dir = _phase_dir(repo)
        from utils.write_batch_scheduler import WriteBatchScheduler
        from utils.write_phase_bridge import materialize_write_tasks

        tasks = materialize_write_tasks(
            kind="implement",
            batches=[
                {"batch_index": i, "can_parallelize": False, "tasks": [
                    {"id": f"T00{i+1}", "prompt": f"do {i}",
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
        assert first.exit_code == 0

        # The "kill": the process is gone, batches 1 and 2 have no result file.
        assert (phase_dir / "implement_batch_0.json").is_file()
        assert not (phase_dir / "implement_batch_1.json").exists()
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
