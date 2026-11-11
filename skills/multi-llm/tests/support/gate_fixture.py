"""The check-gate world: a seven-task graph where a policy stop sits beside a check with findings.

T001 → T002 (a read-only check that reports findings) → T007, and
T001 → T003 → T004 → T005 → T006, where T005 declares a file the ``DENY``
policy refuses, so ``implement`` stops on it before dispatching it. Built on
``tests.support.implement_world.make_repo``; ``_policy`` swaps the
``driver.write_paths`` policy and ``_check`` scripts the check's reply.

Fixtures (registered suite-wide by ``tests/conftest.py``): ``repo``, the world
with the ``DENY`` policy and every edit and check scripted. A test file that
defines its own ``repo`` overrides it. ``tests.support.shared_stop`` runs this
world to its first stop once per test module, for the tests that only read it.
"""

from __future__ import annotations

import json
from typing import Tuple

import pytest

from utils import provider_registry as registry

from tests.support.implement_world import _implement, make_repo


#: The fixture's graph. Every edit task patches src/main.py (the mock's one
#: edit); T005 also declares the file the policy denies.
PLAN = """# Check gate

## Overview
The check-gate fixture's shape.

## Tasks

### T001: Edit: create the configuration file
Create the configuration.
- Depends on: none
- Files to modify: src/main.py

### T002: Check the module
Run the release check. Read-only.
- Depends on: T001
- Files to modify: none

### T003: Edit: add the greeting module
Add the greeting.
- Depends on: T001
- Files to modify: src/main.py

### T004: Edit: add the command-line entry point
Add the entry point.
- Depends on: T003
- Files to modify: src/main.py

### T005: Edit: add the build stamp module
Add the stamp.
- Depends on: T004
- Files to modify: src/main.py, src/vendor/stamp.js

### T006: Edit: print the build stamp
Print it.
- Depends on: T005
- Files to modify: src/main.py

### T007: Edit: write the release note
Write the note.
- Depends on: T002
- Files to modify: src/main.py
"""

CHECK_GATE = "Title: Check the module"
FINDINGS = ["retries must be 3 (found 2)"]
DENY = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**', 'src/vendor/**']\n"
)
ALLOW = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**']\n"
)
#: The way past T005's refusal (fix round 21): a deny hit names the deny
#: rule to narrow, since the deny list wins over `allow` and "allow it in
#: driver.write_paths" resumed into the same refusal (live run 8's case).
DENY_CLAUSE = (
    "if you want src/vendor/stamp.js written, remove or narrow the deny rule "
    "`src/vendor/**` in driver.write_paths (.multi-llm/providers.yaml) so the "
    "rule no longer covers it"
)
STAMP = "Edit: add the build stamp module"
NOTE = "Edit: write the release note"
PRINT = "Edit: print the build stamp"


def _policy(repo, text: str) -> None:
    (repo.tmp_path / "driver.yaml").write_text(text, encoding="utf-8")
    registry._config = None


def _gate_world(tmp_path, monkeypatch, plan_text: str = PLAN, policy: str = DENY):
    """The ``repo`` fixture's body, with the plan and the policy parameters, so
    a shared stop (``tests.support.shared_stop``) builds exactly the world a
    fresh test gets."""
    obj = make_repo(tmp_path, monkeypatch, plan_text)
    _policy(obj, policy)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 8)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    return obj


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield _gate_world(tmp_path, monkeypatch)
    registry._config = None


def _check(repo, *, passed: bool, reply_status=None) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": passed, "findings": [] if passed else FINDINGS}))
    if reply_status:
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_STATUS", reply_status)
    else:
        repo.monkeypatch.delenv("MOCK_LLM_WRITE_CHECK_STATUS", raising=False)


def _first_run_parts(repo, capsys, reply_status=None) -> Tuple[str, str]:
    """:func:`_first_run`'s stdout and stderr, kept apart."""
    _check(repo, passed=False, reply_status=reply_status)
    assert _implement(repo) == 3
    captured = capsys.readouterr()
    repo.log.unlink(missing_ok=True)
    return captured.out, captured.err


def _first_run(repo, capsys, reply_status=None) -> str:
    out, err = _first_run_parts(repo, capsys, reply_status)
    return out + err


def _dispatched_prompt(repo, title: str) -> str:
    """The last prompt the mock runner was handed for the task titled *title*."""
    found = ""
    for line in repo.log.read_text().splitlines():
        prompt = json.loads(line).get("prompt") or ""
        if f"Title: {title}\n" in prompt:
            found = prompt
    assert found, f"no dispatch of {title!r} in the mock log"
    return found
