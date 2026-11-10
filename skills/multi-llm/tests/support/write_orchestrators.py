"""The write orchestrators' world: a real git repository with a three-task plan, a mock runner and a driver config.

``orchestrator_repo`` provides the ``repo`` fixture (a committed repository,
``codex`` on ``PATH`` as the mock runner, ``MULTI_LLM_PROVIDERS_CONFIG`` set,
cwd at the root). It is not registered suite-wide, because
``tests.support.gate_fixture`` owns the suite's ``repo``; import
``orchestrator_repo`` to use it. ``_ApplyRig`` is a mixin that runs the real
apply-suggestions orchestrator over that repository; ``APPLY_PLAN`` and
``_validation`` are its plan and ``validation.json`` builder.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from utils import provider_registry as registry
from utils import subagent_executor as executor
from utils.write_phase_bridge import WRITE_TASK_FILENAME

from tests.mock_launchers import install_mock_llm, mock_path


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


@pytest.fixture(name="repo")
def orchestrator_repo(tmp_path, monkeypatch):
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
    install_mock_llm(bin_dir, "codex")
    monkeypatch.setenv("PATH", mock_path(bin_dir, ("codex",)))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    config = tmp_path / "driver.yaml"
    config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
    registry._config = None
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


class _ApplyRig:
    """The real apply-suggestions orchestrator over the mock runner."""

    def _review(self, repo, prefix, statuses=("valid",) * 3):
        """A review-plan cycle on disk, completed now."""
        from utils.state_manager import get_or_create_state

        plan = repo.root / "plans" / "apply.md"
        if not plan.exists():
            # Bytes: `backup.<sha8>.md` names the sha256 of APPLY_PLAN's LF
            # bytes, which a text-mode write turns into CRLF on Windows.
            plan.write_bytes(APPLY_PLAN.encode("utf-8"))
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

    def _run(self, repo, plan, attest, *flags, content="edit\n"):
        import apply_suggestions_orchestrator as orchestrator

        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "plans/apply.md", "mode": "patch", "content": content}
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

    def _deselect(self, repo, index):
        """The user skips one item between the runs (here: in validation)."""
        statuses = ["valid"] * 3
        statuses[index] = "invalid"
        review = repo.root / "plans" / "apply" / "review-plan"
        (review / "validation.json").write_text(
            json.dumps(_validation(statuses)), encoding="utf-8")
