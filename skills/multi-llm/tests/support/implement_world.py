"""The implement world: a git repository with a five-task plan, run against the mock runner.

``make_repo`` builds it (an edit, a read-only check and an independent edit, an
edit after the check, and a manual task), with the plan text a parameter for
files that need another graph. The helpers drive the real ``implement``
orchestrator in-process and read back what the mock runner was handed.

Fixtures: ``implement_repo`` provides ``repo`` (``make_repo`` with the default
plan). It is not registered suite-wide, because ``tests.support.gate_fixture``
owns the suite's ``repo``; import ``implement_repo`` to use it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import implement_orchestrator
from utils import provider_registry as registry
from utils import subagent_executor as executor
from utils.output_handler import get_phase_dir
from utils.state_manager import get_or_create_state

from tests.mock_launchers import install_mock_llm, mock_path


#: Five tasks: an edit, a read-only check and an independent edit in one
#: batch, an edit that depends on the check, and a manual task after that —
#: the racing plan's T008 / T009 / T012 / T013 / T010 shape.
PLAN = """# Checked plan

## Overview
A plan with a check in the middle.

## Tasks

### T001: Edit: create the module
Create the module file.
- Depends on: none
- Files to modify: src/main.py

### T002: Check the module
Run the validator on the module. Read-only.
- Depends on: T001
- Files to modify: none

### T003: Edit: add tests
Write unit tests.
- Depends on: T001
- Files to modify: src/main.py

### T004: Edit: wire the fallback
Add the fallback.
- Depends on: T002
- Files to modify: src/main.py

### T005: Play through by hand
**Type**: Manual (user action required)
Open the page and race.
- Depends on: T004
"""

EDIT_GATE = "Title: Edit:"
CHECK_GATE = "Title: Check the module"
CHECK_SUMMARY = "html-validate found 2 errors"
CHECK_NOTES = (
    "Ran `npx --yes html-validate index.html` (exit 1): index.html:1:1 "
    "doctype should be uppercase; index.html:128:30 aria-label is not "
    "recommended on this element."
)


def make_repo(tmp_path, monkeypatch, plan_text: str = PLAN):
    """The fixture's body, with the plan text a parameter: shared with
    ``test_human_task_answers.py`` (fix round 8), whose plans add a code task
    after the manual one or drop the manual one."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "plans").mkdir()
    plan = root / "plans" / "checked.md"
    plan.write_text(plan_text, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
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

    state = get_or_create_state(plan)
    state.mark_phase_completed("generate-tasks")
    state.mark_phase_completed("apply-suggestions")
    state.save()

    # Every edit task appends to src/main.py; the check edits nothing.
    monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", EDIT_GATE)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        ["T001", "T002", "T003", "T004"]
    ))
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))

    class Repo:
        pass

    obj = Repo()
    obj.root = root
    obj.plan = plan
    obj.tmp_path = tmp_path
    obj.monkeypatch = monkeypatch
    obj.log = log
    obj.phase_dir = Path(get_phase_dir(plan, "implement"))
    return obj


@pytest.fixture(name="repo")
def implement_repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None


def _check_fails(repo) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", CHECK_GATE)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_SUMMARY", CHECK_SUMMARY)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_NOTES", CHECK_NOTES)


def _check_passes(repo) -> None:
    for name in ("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "MOCK_LLM_WRITE_FAIL_SUMMARY",
                 "MOCK_LLM_WRITE_FAIL_NOTES"):
        repo.monkeypatch.delenv(name, raising=False)


def _implement(repo, *flags: str) -> int:
    return implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents", *flags,
    ])


def _dispatched_titles(repo) -> list:
    if not repo.log.exists():
        return []
    titles = []
    for line in repo.log.read_text().splitlines():
        prompt = json.loads(line).get("prompt") or ""
        for row in prompt.splitlines():
            if row.startswith("Title: "):
                titles.append(row[len("Title: "):])
                break
    return titles


def _statuses(repo) -> dict:
    return get_or_create_state(repo.plan).implementation_task_statuses()


def _run_one_that_stops(repo, capsys) -> str:
    """Run 1 of the story: T001 applies, the check fails, the rest never runs."""
    _check_fails(repo)
    assert _implement(repo) == 3
    captured = capsys.readouterr()
    return captured.out + captured.err


PLAN_WITH_CODE_DEPENDENT = PLAN + """
### T006: Edit: after the manual step
Needs the manual play-through first.
- Depends on: T005
- Files to modify: src/main.py
"""
