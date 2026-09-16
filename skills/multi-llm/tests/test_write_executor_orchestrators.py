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
                        {"id": task_id, "prompt": f"do {task_id}",
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
        assert first.exit_code == 0
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
