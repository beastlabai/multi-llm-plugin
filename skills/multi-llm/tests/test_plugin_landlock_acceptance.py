"""Acceptance for the plugin-landlock write rung, end to end (plan §8.3).

Plan ``codex-sandbox-unavailable-findings-and-fixes`` §8.3 items 3, 7, 8, 9,
10, 11, 13 and 14, and §8.2 "A write dispatch never falls through to the
uncontained rung" (2026-09-30). The executor's and the scheduler's own test
files (``test_plugin_landlock_executor.py``, ``test_plugin_landlock_write.py``)
pin each half against a stub of the other. This file pins the two together,
the way a user reaches them: ``multi_llm.main`` runs the real phase
orchestrator in-process (``apply-suggestions``, ``apply-task-suggestions``,
``apply-code-fixes``, ``implement``) with ``--yes --exec-subagents`` in a
scratch git repo under ``tmp_path``, so the real scheduler hands its
``JailRequest`` to the real executor, the real preflight walks codex's ladder,
and the real ``landlock_dispatch.wrap_argv`` builds the argv the child gets.
In-process, not a subprocess: the fake wrapper is installed by monkeypatching
``landlock_dispatch.WRAPPER_PATHS``, which a child interpreter would not see.

Tier 1 (§8.1), kernel-independent and zero spend:

* codex is the repo's mock launcher, and its own ``codex sandbox`` probe
  answers the bwrap failure line (``MOCK_LLM_SANDBOX_BROKEN=1``), so the
  standard ``workspace-write`` probe is ``unavailable`` exactly as on a stock
  Ubuntu 24.04 (item 7's contrast unsets it);
* the Landlock wrapper is ``test_plugin_landlock_executor``'s fake: it logs its
  argv and ``--allow`` entries, answers the rung's probe as scripted and
  ``exec``s the mock for a dispatch. It does not jail — what the jail would
  have refused is scripted into codex's stream instead;
* what the mock "codex" does on a write turn is scripted per batch through the
  mock's supported one-off seam (``MOCK_LLM_PLUGIN``, a module under
  ``tmp_path``): the in-place edit it makes, the codex events it adds to its
  stream (a failed ``file_change`` is what codex prints for a patch the jail
  refused) and the result it claims.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import codex as codex_module  # noqa: E402
from utils.providers.base import (  # noqa: E402
    HOST_SANDBOX_POSTURE_ENV,
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    SKIP_RUNG_ENV,
)
from utils.state_manager import StateManager  # noqa: E402
from utils.write_safety import AuditLog, safety_dir  # noqa: E402
from tests.mock_launchers import is_this_interpreter  # noqa: E402
from tests.test_plugin_landlock_executor import (  # noqa: E402
    BWRAP_LINE,
    F6_LEAD,
    FakeWrapper,
    _slot_dirs,
    use_own_temp_base,
)

# Windows runs no plugin jail (codex has its own sandbox there), and this suite
# could not stand one in even as a fake: the scratch repo's codex is a symlink to
# a shebang script on a replaced PATH, which Windows can neither find nor run.
# macOS does run it (tier 1 is kernel-independent; the real-jail legs below skip
# themselves where Landlock is missing).
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="symlinked shebang mock codex on a replaced PATH; Windows runs no plugin jail",
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
SCENARIO_MATCHER = Path(__file__).parent / "scenario_matcher.py"
RUNNER = "codex:gpt-5.5"
#: The codex host's credential-free floor (`hosts/capabilities.yaml`).
FLOOR_RUNNER = "codex:gpt-6-astra"
LEVER_ENV = "MULTI_LLM_CODEX_UNSANDBOXED_WRITES"
SESSION_ENV = registry.HOST_SESSION_MODEL_ENV
REFUSED_BEFORE_CHILD = "refused before any child started"
CONTRACT_MARK = "## Result (required"

#: The two kinds step 3 option (b) put on the rung, and their scopes.
STEP_3 = [("apply-suggestions", "plan-file"), ("apply-task-suggestions", "tasks")]
#: The two kinds step 3b (2026-10-02) added, under the repo-wide scope.
REPO_WIDE = ["apply-code-fixes", "implement"]
#: Every editing kind and its scope: all four run on the rung since step 3b.
ENABLED = STEP_3 + [(mode, "repo-wide") for mode in REPO_WIDE]
PHASE_DIRS = {
    "apply-suggestions": "apply-suggestions",
    "apply-task-suggestions": "apply-task-suggestions",
    "apply-code-fixes": "apply-fixes",
    "implement": "implement",
}
#: A phrase only batch 0's prompt carries, and one only batch 1's does, per
#: kind — what the mock's script keys a write turn by.
BATCH_TEXT = {
    "apply-suggestions": ("Name the upload size limit", "Say where the config file lives"),
    "apply-task-suggestions": ("Name the upload size limit", "Say where the config file lives"),
    "apply-code-fixes": ("Guard the division", "Check the input"),
    # implement's T002 prompt names T001 (its dependency) by title and
    # description, so the id line of the result contract is what tells the
    # two batches apart.
    "implement": ("this task's id, `T001`", "this task's id, `T002`"),
}

#: One sentence per refusal leg (plan §7 "Step 4"; the executor's own tests
#: pin the same sentences against a stand-in adapter).
LEG_SENTENCES = {
    "opt-in-off": "does not take this job: write jobs are not enabled on it",
    "absent": "cannot run here either: this kernel or container lacks Landlock "
              "(unavailable: Landlock ABI none (ENOSYS)",
    "abi2": "cannot run here either: this kernel or container lacks Landlock "
            "(unavailable: Landlock ABI 2 found",
    "contained": "does not take this job: providers.codex.sandbox_fallback=contained",
    "off": "does not take this job: providers.codex.sandbox_fallback=off",
    "required": "failed its self-test (landlock_exec: required-entry:",
    "not_denied": "failed its self-test (a write or truncation its canary made was not denied",
    "seed": "failed its self-test (a write or truncation its canary made was not denied",
    "positive": "failed its self-test (an allowed write failed",
    "startup": "Codex did not start under the allow-list: Error: codex could not start",
    "exit0": "failed its self-test (the canary never reported a denied write)",
    "hang": "failed its self-test (its probe gave no answer",
}

PLAN_BODY = """# Plan

<!-- TASKS_FILE: p/tasks/tasks.md -->

## Overview
A plan whose review found two things to change.

## Step 1
Upload the file.

## Step 2
Write the config.
"""

TASKS_BODY = """# Tasks

## Tasks

### T001: Create the module
Create the module file.
- Depends on: none
- Files to create: src/main.py

### T002: Implement the core
Implement the core functionality.
- Depends on: T001
- Files to modify: src/main.py
"""


def _suggestion_groups() -> List[Dict[str, Any]]:
    """Two valid groups on two sections, so the batcher makes two batches."""
    return [
        {
            "theme": "Name the upload size limit in Step 1",
            "category": "clarity",
            "models": ["model-a", "model-b"],
            "suggestions": [{
                "title": "Name the upload size limit",
                "desc": "State the maximum upload size the step accepts.",
                "type": "addition", "reference": "## Step 1",
                "importance": "HIGH", "source_model": "model-a",
            }],
        },
        {
            "theme": "Say where the config lives in Step 2",
            "category": "clarity",
            "models": ["model-b"],
            "suggestions": [{
                "title": "Say where the config file lives",
                "desc": "Name the config path the step writes.",
                "type": "addition", "reference": "## Step 2",
                "importance": "MEDIUM", "source_model": "model-b",
            }],
        },
    ]


#: A third group the validators could not decide: `--claude-decide` (implied
#: by `--yes` as `--agent-decide`) sends it to the §2b judge.
JUDGE_GROUP = {
    "theme": "Decide the retry policy for the upload",
    "category": "reliability",
    "models": ["model-a"],
    "suggestions": [{
        "title": "Decide the retry policy",
        "desc": "The plan does not say whether a failed upload is retried.",
        "type": "addition", "reference": "## Step 1",
        "importance": "MEDIUM", "source_model": "model-a",
    }],
}

FIX_GROUPS = [
    {
        "theme": "Guard the division in app.py",
        "category": "bug",
        "models": ["model-a"],
        "suggestions": [{
            "title": "Guard the division", "desc": "x may be 0 here.",
            "type": "bug", "importance": "HIGH", "file": "src/app.py",
            "line_range": "1-1", "source_model": "model-a",
        }],
    },
    {
        # A second, MEDIUM fix on the same file: the HIGH one is batched
        # alone, so apply-code-fixes has two batches like the other kinds.
        "theme": "Check the input in app.py",
        "category": "bug",
        "models": ["model-b"],
        "suggestions": [{
            "title": "Check the input", "desc": "x is never validated.",
            "type": "bug", "importance": "MEDIUM", "file": "src/app.py",
            "line_range": "1-1", "source_model": "model-b",
        }],
    },
]


# ---------------------------------------------------------------------------
# The scripted write turn: a one-off mock seam (MOCK_LLM_PLUGIN), under tmp_path
# ---------------------------------------------------------------------------

#: Replaces two of the mock's four supported seams (``get_response_content``
#: and ``format_output``). A write-batch prompt — the scheduler appends its
#: result contract to every one — is answered from the script: the first rule
#: whose ``match`` text the prompt carries picks its steps, consumed in order
#: per rule (the last one repeats); with no rule, ``default``. A step's
#: in-place appends land on disk, its events are spliced into codex's stream
#: before the agent message, and its claim is the agent message (``ids``
#: "all" = every id the prompt's contract lists). Every other prompt (a
#: judge's) gets the mock's own answer.
MOCK_PLUGIN = r'''
import json, os, re, sys

_MAIN = sys.modules["__main__"]
_ORIGINAL_CONTENT = _MAIN.get_response_content
_ORIGINAL_FORMAT = _MAIN.format_output
_EXTRA = []
_CONTRACT = "## Result (required"


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def get_response_content(prompt):
    if _CONTRACT not in prompt:
        return _ORIGINAL_CONTENT(prompt)
    ids = re.findall(r"(?:suggestion|fix) \d+ = `([^`]+)`", prompt)
    ids += re.findall(r"this task's id, `([^`]+)`", prompt)
    script = _load(os.environ["LANDLOCK_ACCEPT_SCRIPT"], {})
    rule = next((r for r in script.get("rules", []) if r["match"] in prompt), None)
    steps = rule["steps"] if rule else [script.get("default") or {}]
    counter_path = os.environ["LANDLOCK_ACCEPT_COUNTER"]
    counters = _load(counter_path, {})
    key = rule["match"] if rule else "default"
    seen = counters.get(key, 0)
    counters[key] = seen + 1
    with open(counter_path, "w", encoding="utf-8") as handle:
        json.dump(counters, handle)
    step = steps[min(seen, len(steps) - 1)]
    label = ids[0] if ids else "?"
    shell_log = os.environ.get("LANDLOCK_ACCEPT_SHELL_LOG")
    for name, command in step.get("shell", {}).items():
        # The denied-write cases, run by the "model" inside whatever jail this
        # process is in; the exit code is what the test reads back.
        import subprocess
        done = subprocess.run(command, shell=True, capture_output=True, text=True)
        if shell_log:
            with open(shell_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"name": name, "rc": done.returncode,
                                         "out": (done.stdout + done.stderr)[-400:]}) + "\n")
    for path, text in step.get("append", {}).items():
        # In place, the same inode — the way `apply_patch` writes a file.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text.replace("{id}", label))
    _EXTRA.extend(step.get("events", []))
    claim = dict(step.get("claim") or {"status": "applied"})
    if claim.get("ids", "all") == "all":
        claim["ids"] = ids
    claim.setdefault("files_touched", sorted(step.get("append", {})))
    claim.setdefault("summary", "scripted write turn")
    return json.dumps(claim)


def format_output(provider, content):
    rendered = _ORIGINAL_FORMAT(provider, content)
    if not _EXTRA:
        return rendered
    lines = rendered.splitlines()
    at = next((i for i, line in enumerate(lines) if '"agent_message"' in line),
              len(lines))
    extra = [json.dumps(event) for event in _EXTRA]
    return "\n".join(lines[:at] + extra + lines[at:])
'''


def file_change(status: str, *paths: str) -> Dict[str, Any]:
    """codex's ``file_change`` item. ``failed`` is what it prints for an
    ``apply_patch`` the jail refused (preflight's ``jail_writes`` shape: the
    whole patch's paths, not only the refused one)."""
    return {"type": "item.completed", "item": {
        "id": f"item_patch_{status}", "type": "file_change", "status": status,
        "changes": [{"path": p, "kind": "update"} for p in paths],
    }}


def marker(key: str) -> str:
    return f"\n<!-- applied {key} -->\n"


# ---------------------------------------------------------------------------
# The scratch repo
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env,
    ).stdout


class Scratch:
    """A committed git repo holding a reviewed plan, its tasks, a code review
    and one source file; mock codex on PATH; the fake wrapper installed.

    ``real_jail`` (step 3b's tier-2 leg) installs no fake wrapper, so the
    production ``landlock_exec.py`` jails the mock for real; the mock's own
    files (call log, the script counter, the shell log) then live under the
    throwaway ``$CODEX_HOME``, the one place outside the repository the jail
    grants, and the mock runs ``danger-full-access`` probes while its
    ``workspace-write`` probe still dies on the bwrap line.
    """

    def __init__(self, tmp_path: Path, monkeypatch, *, judge: bool = False,
                 real_jail: bool = False) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.real_jail = real_jail
        home = tmp_path / "home"
        self.codex_home = home / ".codex"
        self.codex_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CODEX_HOME", str(self.codex_home))
        mock_files = self.codex_home if real_jail else tmp_path

        self.root = (tmp_path / "repo").resolve()
        self.root.mkdir()
        _git(self.root, "init", "-q")
        self.plan = self.root / "plans" / "p.md"
        self.plan.parent.mkdir(parents=True)
        self.plan.write_text(PLAN_BODY, encoding="utf-8")
        self.out_dir = self.root / "plans" / "p"
        self.tasks = self.out_dir / "tasks" / "tasks.md"
        self.tasks.parent.mkdir(parents=True)
        self.tasks.write_text(TASKS_BODY, encoding="utf-8")
        self.readme = self.root / "README.md"
        self.readme.write_text("scratch\n", encoding="utf-8")
        self.app = self.root / "src" / "app.py"
        self.app.parent.mkdir()
        self.app.write_text("x = 1\n", encoding="utf-8")
        self.main_py = self.root / "src" / "main.py"
        # The protected set a repo-wide batch must leave alone (step 3b).
        (self.root / ".codex").mkdir()
        (self.root / ".codex" / "config.toml").write_text("# project codex config\n", encoding="utf-8")
        (self.root / ".multi-llm").mkdir()
        (self.root / ".multi-llm" / "providers.yaml").write_text("# project config\n", encoding="utf-8")
        self._review("review-plan", _suggestion_groups() + ([JUDGE_GROUP] if judge else []),
                     undecided=[2] if judge else [])
        self._review("review-tasks", _suggestion_groups())
        self._review("code-review", FIX_GROUPS)
        state = StateManager(Path(self.plan))
        # The phases under test are left open; only their upstream is done.
        for phase in ("review-plan", "generate-tasks", "review-tasks", "code-review"):
            state.mark_phase_completed(phase)
        state.save()
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-qm", "init")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        launcher = bin_dir / "codex"
        try:
            launcher.symlink_to(MOCK_LLM)
        except (OSError, NotImplementedError):  # pragma: no cover
            shutil.copyfile(MOCK_LLM, launcher)
            launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
            shutil.copyfile(SCENARIO_MATCHER, bin_dir / SCENARIO_MATCHER.name)
        self.codex = launcher
        # The dispatcher's uv backstop (D6c) reads PATH; keep the real uv
        # reachable by its env name without putting its directory — which may
        # hold a real provider CLI — on PATH.
        uv = shutil.which("uv")
        if uv:
            monkeypatch.setenv("MULTI_LLM_UV", uv)
        # REPLACED, not prepended: nothing here may reach a real codex.
        monkeypatch.setenv("PATH", os.pathsep.join(
            [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
        ))
        assert shutil.which("codex") == str(launcher)
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        self.calls = mock_files / "codex-calls.jsonl"
        monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(self.calls))
        # The standard probe: bwrap cannot start (the mock's own replay).
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        if real_jail:
            monkeypatch.setenv("MOCK_LLM_SANDBOX_RUNS_FULL_ACCESS", "1")
        else:
            monkeypatch.delenv("MOCK_LLM_SANDBOX_RUNS_FULL_ACCESS", raising=False)
        monkeypatch.setenv("MOCK_LLM_ECHO", "human_decision_judge")
        plugin = tmp_path / "mock_plugin.py"
        plugin.write_text(MOCK_PLUGIN, encoding="utf-8")
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        self.script_path = tmp_path / "write-script.json"
        self.counter_path = mock_files / "write-counter.json"
        self.shell_log = mock_files / "write-shell.jsonl"
        monkeypatch.setenv("LANDLOCK_ACCEPT_SCRIPT", str(self.script_path))
        monkeypatch.setenv("LANDLOCK_ACCEPT_COUNTER", str(self.counter_path))
        monkeypatch.setenv("LANDLOCK_ACCEPT_SHELL_LOG", str(self.shell_log))
        for name in (
            SKIP_RUNG_ENV, HOST_SANDBOX_POSTURE_ENV, LEVER_ENV, SESSION_ENV,
            "MULTI_LLM_HOST", "MOCK_LLM_EDITS", "MOCK_LLM_WRITE_RESULT",
            "MOCK_LLM_WRITE_IDS", "MOCK_LLM_PROSE",
        ):
            monkeypatch.delenv(name, raising=False)
        self.configure()
        executor.reset_version_cache()
        monkeypatch.chdir(self.root)
        self.wrapper = None if real_jail else FakeWrapper(tmp_path / "wrapper", monkeypatch)

    # -- inputs -----------------------------------------------------------

    def _review(self, subdir: str, groups: List[Dict[str, Any]], undecided=()) -> None:
        review = self.out_dir / subdir
        review.mkdir(parents=True)
        (review / "grouped.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
        verdicts = []
        for index in range(len(groups)):
            if index in undecided:
                verdicts.append({
                    "group_index": index, "status": "needs-human-decision",
                    "reason": "a business call", "confidence": 0.5,
                    "error_type": "real_ambiguity", "recoverable": False,
                })
            else:
                verdicts.append({"group_index": index, "status": "valid",
                                 "reason": "clear", "confidence": 0.9})
        validation = {
            "metadata": {"schema_version": "2.0", "model": "mock",
                         "validated_at": "2026-09-30T00:00:00",
                         "total_groups": len(groups)},
            "groups": verdicts,
        }
        (review / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    def configure(self, *, fallback: Optional[str] = None, driver: bool = True) -> None:
        """The explicit override layer: the runner (unless ``driver`` is off,
        which leaves the session and host rungs to pick it) and the policy."""
        lines = []
        if driver:
            lines += ["driver:", f"  subagent_runner: {RUNNER}"]
        if fallback is not None:
            lines += ["providers:", "  codex:", f"    sandbox_fallback: {fallback}"]
        if lines:
            path = self.tmp_path / "providers-override.yaml"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        else:
            self.monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
        registry._config = None
        registry._config_key = None

    def target(self, mode: str) -> Path:
        return {
            "apply-suggestions": self.plan,
            "apply-task-suggestions": self.tasks,
            "apply-code-fixes": self.app,
            "implement": self.main_py,
        }[mode]

    def script(self, mode: str, *, batch0: Optional[list] = None,
               batch1: Optional[list] = None) -> None:
        """What each write turn does. A batch with no steps of its own appends
        its marker to ``mode``'s target in place and claims every expected id."""
        rules = []
        first, second = BATCH_TEXT[mode]
        if batch0 is not None:
            rules.append({"match": first, "steps": batch0})
        if batch1 is not None:
            rules.append({"match": second, "steps": batch1})
        default = {"append": {str(self.target(mode)): marker("{id}")}}
        self.script_path.write_text(
            json.dumps({"rules": rules, "default": default}), encoding="utf-8",
        )

    def complete(self, *phases: str) -> None:
        state = StateManager(Path(self.plan))
        for phase in phases:
            state.mark_phase_completed(phase)
        state.save()

    # -- the run ----------------------------------------------------------

    def run(self, mode: str, *extra: str) -> int:
        if not self.script_path.exists():
            self.script(mode)
        if mode == "implement":
            self.complete("apply-suggestions")
        registry._config = None
        registry._config_key = None
        if hasattr(registry, "_RUN_ANCHOR"):
            registry._RUN_ANCHOR = None
        return multi_llm.main([mode, str(self.plan), "--yes", "--exec-subagents", *extra])

    # -- what it left -----------------------------------------------------

    def phase_dir(self, mode: str) -> Path:
        return self.out_dir / PHASE_DIRS[mode]

    def status(self, mode: str) -> Dict[str, Any]:
        return json.loads((self.phase_dir(mode) / ".status.json").read_text(encoding="utf-8"))

    def report(self, mode: str) -> Dict[str, Any]:
        return json.loads(
            (self.phase_dir(mode) / "write_batch_report.json").read_text(encoding="utf-8")
        )

    def write_tasks(self, mode: str) -> List[Dict[str, Any]]:
        document = json.loads(
            (self.phase_dir(mode) / "write_tasks.json").read_text(encoding="utf-8")
        )
        return document["batches"]

    def keys(self, mode: str) -> List[str]:
        """Each batch's first expected id: what its marker names."""
        return [task["expected_ids"][0] for task in self.write_tasks(mode)]

    def completed(self, mode: str) -> bool:
        state = StateManager(Path(self.plan))
        return any(state.is_phase_completed(name) for name in {mode, PHASE_DIRS[mode]})

    def exec_calls(self) -> List[Dict[str, Any]]:
        if not self.calls.exists():
            return []
        rows = [json.loads(line) for line in self.calls.read_text(encoding="utf-8").splitlines() if line]
        return [row for row in rows if row["args"][:1] == ["exec"]]

    def write_calls(self) -> List[Dict[str, Any]]:
        return [row for row in self.exec_calls() if CONTRACT_MARK in row["prompt"]]

    def changed_outside(self, mode: str) -> List[str]:
        """``git status`` paths that are neither the run's output folder nor
        ``mode``'s target."""
        target = str(self.target(mode).relative_to(self.root))
        # implement's finalization writes the summary back-reference into
        # the plan: the phase's own write, never the runner's.
        own = {target} | ({"plans/p.md"} if mode == "implement" else set())
        rows = _git(self.root, "status", "--porcelain", "--untracked-files=all").splitlines()
        return [
            row[3:] for row in rows
            if not row[3:].startswith("plans/p/") and row[3:] not in own
        ]

    def shell_results(self) -> Dict[str, int]:
        """``name -> exit code`` of every ``shell`` step the mock ran."""
        if not self.shell_log.exists():
            return {}
        rows = [json.loads(line) for line in self.shell_log.read_text(encoding="utf-8").splitlines() if line]
        return {row["name"]: row["rc"] for row in rows}


@pytest.fixture(autouse=True)
def own_temp_base(tmp_path, monkeypatch):
    """Every slot a test's run makes lands in a temp dir of its own, so
    ``_slot_dirs()`` never reads another process's live slots."""
    return use_own_temp_base(tmp_path, monkeypatch)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    yield Scratch(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


@pytest.fixture
def judged(tmp_path, monkeypatch):
    yield Scratch(tmp_path, monkeypatch, judge=True)
    registry._config = None
    registry._config_key = None


def _in_flight(status: Dict[str, Any]) -> list:
    return ((status.get("progress") or {}).get("write_subagents") or {}).get("in_flight") or []


def _refusal(err: str) -> str:
    """The batch-0 refusal's reason line, as the scheduler's exit-3 block
    prints it (``  - <reason>``)."""
    lines = [line.strip() for line in err.splitlines() if "Refusing to dispatch" in line]
    assert lines, err[-3000:]
    return lines[0][2:] if lines[0].startswith("- ") else lines[0]


def _allow(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The fake wrapper's recorded ``--allow``/``--allow-optional`` pairs, with
    the attempt's own ``TMPDIR`` (fresh per attempt) normalized."""
    tmpdir = row["env"]["TMPDIR"] or "\0"
    flags = row["allow"]
    return [
        (flags[i], flags[i + 1].replace(tmpdir, "<TMPDIR>"))
        for i in range(0, len(flags) - 1, 2)
    ]


def _developer_instructions(cmd: List[str]) -> List[str]:
    return [
        cmd[i + 1] for i, arg in enumerate(cmd[:-1])
        if arg == "-c" and cmd[i + 1].startswith("developer_instructions=")
    ]


def _no_uncontained_rung(scratch: Scratch, text: str) -> None:
    """§8.3 item 13's invariant: the recorded rung is never ``unsandboxed`` or
    ``landlock``, no ``[SANDBOX_BYPASSED]`` line, and every argv carrying
    ``danger-full-access`` that reached the mock went through the wrapper."""
    assert "[SANDBOX_BYPASSED]" not in text
    for scope in ("plan-file", "tasks", "repo-wide"):
        for runner in (RUNNER, FLOOR_RUNNER):
            assert executor.sandbox_rung(runner, PROFILE_AGENT_WRITE, write_scope=scope) in (
                None, "plugin-landlock",
            )
    assert "unsandboxed" not in executor._SANDBOX_DECISIONS.values()
    assert "landlock" not in executor._SANDBOX_DECISIONS.values()
    wrapped = scratch.wrapper.records(probe=False)
    full_access = [
        call for call in scratch.exec_calls() if "danger-full-access" in call["args"]
    ]
    # The wrapper logs itself before it execs, so every exec the mock logged
    # with full access has a wrapper row in front of it.
    assert len(full_access) <= len(wrapped)
    for row in wrapped:
        assert row["cmd"][0] == str(scratch.codex)


# ---------------------------------------------------------------------------
# §8.3 items 8 and 9 (and item 13's "Intended edits"): the enabled kinds take
# the rung and land their edits
# ---------------------------------------------------------------------------

class TestTheEnabledKindsTakeTheRung:
    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_each_batch_runs_on_the_rung_and_its_edit_lands(
        self, scratch, capsys, mode, scope, own_temp_base,
    ):
        target = scratch.target(mode)
        # implement's finalization writes the summary back-reference into
        # the plan: the phase's own write, so the plan is not "another" file.
        untouched = [scratch.tasks, scratch.app] + ([] if mode == "implement" else [scratch.plan])
        others = {p: p.read_bytes() for p in untouched if p != target}
        protected = {p: p.read_bytes() for p in (
            scratch.root / ".git" / "HEAD", scratch.root / ".git" / "config",
            scratch.root / ".codex" / "config.toml",
            scratch.root / ".multi-llm" / "providers.yaml",
        )}
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        slots_before = _slot_dirs()

        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        assert REFUSED_BEFORE_CHILD not in text
        _no_uncontained_rung(scratch, text)
        count = len(scratch.write_tasks(mode))
        assert count == 2

        # The stray-writes-only line, once before each batch, naming the scope.
        lines = cap.out.splitlines()
        notices = [i for i, line in enumerate(lines) if "[SANDBOX_INCIDENTAL]" in line]
        batches = [i for i, line in enumerate(lines) if line.startswith("[write-subagents] batch ")]
        assert len(notices) == len(batches) == count, text[-4000:]
        assert notices[0] < batches[0] < notices[1] < batches[1]
        for index in notices:
            line = lines[index]
            assert f"write phase, scope {scope}" in line
            assert "'plugin-landlock'" in line and BWRAP_LINE in line
            assert "network access is not confined" in line
        for index in batches:
            assert "applied" in lines[index] and "[on plugin-landlock]" in lines[index]
        assert executor.sandbox_rung(RUNNER, PROFILE_AGENT_WRITE, write_scope=scope) == "plugin-landlock"

        # Each batch's edit landed once, in the target, and nothing else moved
        # outside the run's output folder.
        assert target.read_text(encoding="utf-8") == before + "".join(
            marker(key) for key in scratch.keys(mode))
        assert scratch.changed_outside(mode) == []
        assert all(path.read_bytes() == data for path, data in others.items())
        assert {p: p.read_bytes() for p in protected} == protected

        # Completed, exit 0, nothing in flight, the rung on every batch.
        assert scratch.completed(mode)
        assert _in_flight(scratch.status(mode)) == []
        report = scratch.report(mode)
        assert [row.get("sandbox_rung") for row in report["tasks"]] == ["plugin-landlock"] * count
        assert [row["write_outcome"] for row in report["tasks"]] == ["applied"] * count

        # One rung probe for this scope's grant, then the wrapped dispatches.
        # A step-3 phase's closing next-step block asks the early warning,
        # which probes once more under the repo-wide scope key (the cache is
        # scope-keyed): that probe moves its canary workdir outside the
        # repository (root grant, 2026-10-05) and is not this phase's.
        probes = scratch.wrapper.records(probe=True)
        own = [r for r in probes if (r["env"]["MULTI_LLM_CANARY_WORKDIR"] is None) == (scope != "repo-wide")]
        [probe] = own
        assert len(probes) == (1 if scope == "repo-wide" else 2), probes
        rows = scratch.wrapper.records(probe=False)
        assert len(rows) == len(scratch.exec_calls()) == len(scratch.write_calls()) == count
        real_target = os.path.realpath(target)
        for row in rows:
            assert is_this_interpreter(row["orig_argv"][0])
            assert row["orig_argv"][1:4] == ["-I", "-S", str(scratch.wrapper.path)]
            cmd = row["cmd"]
            assert cmd[:2] == [str(scratch.codex), "exec"]
            assert cmd[cmd.index("--sandbox") + 1] == "danger-full-access"
            [developer] = _developer_instructions(cmd)
            in_repo = [entry for _flag, entry in _allow(row) if str(scratch.root) in entry]
            if scope == "plan-file":
                assert f"You may write only {real_target}." in developer
                assert in_repo == [f"file:{real_target}"]
            elif scope == "tasks":
                tasks_dir = os.path.realpath(scratch.tasks.parent)
                assert f"is the directory {tasks_dir}" in developer
                assert in_repo == [f"dir:{tasks_dir}"]
            else:
                # Step 3b, revised 2026-10-05: one required dir rule on the
                # repository root and nothing else in the tree.
                out_dir = os.path.realpath(scratch.out_dir)
                assert f"The whole repository {scratch.root} is writable" in developer
                assert f"this run's output folder {out_dir}" in developer
                assert "Do NOT modify .git/" in developer
                assert "/tmp and /dev/shm are denied" in developer
                assert in_repo == [f"dir:{scratch.root}"]
                assert ("--allow", f"dir:{scratch.root}") in _allow(row)
                assert row["env"]["UV_CACHE_DIR"] == os.path.join(row["env"]["TMPDIR"], "uv-cache")
                assert row["env"]["PYTEST_ADDOPTS"].startswith("-o cache_dir=")
                assert "write phase, scope repo-wide" in lines[notices[0]]
        for call in scratch.write_calls():
            assert call["args"][call["args"].index("--sandbox") + 1] == "danger-full-access"
        # Each attempt's TMPDIR existed for the child and is gone after it.
        assert all(row["tmpdir_exists"] for row in rows)
        assert not any(os.path.exists(row["env"]["TMPDIR"]) for row in rows)
        assert {Path(row["env"]["TMPDIR"]).parent for row in rows} == {own_temp_base}
        assert _slot_dirs() <= slots_before


# ---------------------------------------------------------------------------
# §8.3 items 10 and 13 ("Policy words and Landlock", "Intended edits"): the
# refusal legs hold
# ---------------------------------------------------------------------------

def _assert_refused_at_batch_0(scratch: Scratch, mode: str, code: int, text: str,
                               err: str, before: Optional[bytes]) -> str:
    """Exit 3 at batch 0, led by F6, I11's status, and nothing dispatched."""
    assert code == 3, text[-4000:]
    reason = _refusal(err)
    assert reason.startswith(F6_LEAD), reason
    assert reason.endswith("Nothing was dispatched."), reason
    assert f"batch 0 was {REFUSED_BEFORE_CHILD}" in err
    # I11: the refusal is named in `.status.json` and nothing is in flight.
    status = scratch.status(mode)
    assert _in_flight(status) == []
    assert status.get("state") == "runner_rejected"
    assert status.get("reason_code") == "sandbox_unavailable"
    assert status.get("refused_batch") == 0
    # No batch reached the mock, nothing moved, the phase is open.
    assert scratch.write_calls() == []
    assert scratch.wrapper.records(probe=False) == []
    target = scratch.target(mode)
    assert (target.read_bytes() if target.exists() else None) == before
    assert not scratch.completed(mode)
    assert "[SANDBOX_INCIDENTAL]" not in text
    return reason


def _assert_session_refused(out: str, reason: str) -> None:
    """The session rung's spec is what was refused — or, where the preflight
    lets the session rung fall through by name, the codex host's floor after
    it (``_preflight_batch_runners``). Measured 2026-09-30: the write
    scheduler's own preflight refuses the session spec itself at batch 0 and
    does not fall through; either way the run ends on a refusal."""
    fell = (
        f"session model {RUNNER} failed preflight (sandbox_unavailable); "
        f"falling back to {FLOOR_RUNNER}"
    ) in out
    assert f"'{FLOOR_RUNNER}'" in reason if fell else f"'{RUNNER}'" in reason, reason


def _bytes(path: Path) -> Optional[bytes]:
    return path.read_bytes() if path.exists() else None


class TestTheRefusalLegsStillHold:
    @pytest.mark.parametrize("lever", [False, True], ids=["plain", "I8-lever"])
    @pytest.mark.parametrize("leg", ["opt-in-off", "absent", "abi2", "contained", "off"])
    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_the_enabled_kinds_are_refused_at_batch_0(
        self, scratch, capsys, monkeypatch, mode, scope, leg, lever,
    ):
        """Item 10 (the step-3 opt-in off, Landlock absent) and item 13's
        policy words, each with and without the I8 lever, which changes none
        of it."""
        if leg == "opt-in-off":
            monkeypatch.setattr(codex_module, "WRITE_RUNG_OPT_IN", False)
        elif leg in ("contained", "off"):
            scratch.configure(fallback=leg)
        else:
            scratch.wrapper.set(probe=leg)
        if lever:
            monkeypatch.setenv(LEVER_ENV, "1")
        before = _bytes(scratch.target(mode))
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        reason = _assert_refused_at_batch_0(scratch, mode, code, text, cap.err, before)
        assert LEG_SENTENCES[leg] in reason, reason
        _no_uncontained_rung(scratch, text)
        assert executor.sandbox_rung(RUNNER, PROFILE_AGENT_WRITE, write_scope=scope) is None
        if leg in ("off", "opt-in-off"):
            # `off` and a kept-off opt-in spawn no rung probe at all.
            assert scratch.wrapper.records() == []
        else:
            # `contained` still measures the rung (one probe), and never takes
            # it; a failed probe is one probe and a refusal.
            assert len(scratch.wrapper.records(probe=True)) == 1

    @pytest.mark.parametrize("mode", REPO_WIDE)
    def test_repo_wide_never_takes_the_uncontained_rung_by_consent(
        self, scratch, capsys, monkeypatch, mode,
    ):
        """Step 3b put repo-wide edits on the jail and nowhere else: with the
        jail unusable, the chat's `danger-full-access` posture (F5) opens rung
        3 for the step-3 scopes only — a repo-wide batch is refused, naming
        the jail's own case, and no `[SANDBOX_BYPASSED]` line appears."""
        scratch.wrapper.set(probe="absent")
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        before = _bytes(scratch.target(mode))
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        reason = _assert_refused_at_batch_0(scratch, mode, code, text, cap.err, before)
        assert LEG_SENTENCES["absent"] in reason, reason
        assert "F6" not in reason or "driver.subagent_runner" in reason
        _no_uncontained_rung(scratch, text)
        assert executor.SANDBOX_DECISION_WRITE_BY_CONSENT not in executor._SANDBOX_DECISIONS.values()
        assert len(scratch.wrapper.records(probe=True)) == 1
        assert scratch.exec_calls() == []


# ---------------------------------------------------------------------------
# §8.2 "A write dispatch never falls through to the uncontained rung", and
# §8.3 item 13 "No route reaches `unsandboxed`", for the enabled kinds
# ---------------------------------------------------------------------------

class TestNoRouteReachesUnsandboxed:
    @pytest.mark.parametrize("policy", ["auto", "contained", "off"])
    @pytest.mark.parametrize(
        "probe", ["absent", "abi2", "required", "not_denied", "seed", "positive",
                  "startup", "exit0", "hang"],
    )
    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_a_failed_rung_probe_is_refused_never_walked_past(
        self, scratch, capsys, monkeypatch, mode, scope, probe, policy,
    ):
        """Every probe-time failure shape the rung can produce, under every
        policy word and with no posture: refused at batch 0 as
        `sandbox_unavailable` naming its case (under `auto` the probe's; under
        `contained`/`off` the policy's); the mock's log records no `exec`."""
        scratch.wrapper.set(probe=probe)
        if policy != "auto":
            scratch.configure(fallback=policy)
        if probe == "hang":
            # Room for the wrapper to start and record the probe before it hangs:
            # 0.8 s was not always enough on the macOS runner (2026-10-23).
            monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 2.5)
        before = _bytes(scratch.target(mode))
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        reason = _assert_refused_at_batch_0(scratch, mode, code, text, cap.err, before)
        assert LEG_SENTENCES[probe if policy == "auto" else policy] in reason, reason
        _no_uncontained_rung(scratch, text)
        assert scratch.exec_calls() == []
        if policy == "off":
            assert scratch.wrapper.records() == []
        else:
            # The case was measured (one probe) and no dispatch followed it.
            assert len(scratch.wrapper.records(probe=True)) == 1

    @pytest.mark.parametrize("attempt", ["lever", "posture", "session"])
    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_every_other_attempt_ends_on_the_rung(
        self, scratch, capsys, monkeypatch, mode, scope, attempt,
    ):
        """The I8 lever, the chat's `danger-full-access` posture (step 6: F5
        sits BELOW the jail) and a Codex session model all end on
        plugin-landlock while the rung is usable."""
        runner = RUNNER
        if attempt == "lever":
            monkeypatch.setenv(LEVER_ENV, "1")
        elif attempt == "posture":
            monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        else:
            scratch.configure(driver=False)
            monkeypatch.setenv("MULTI_LLM_HOST", "codex")
            monkeypatch.setenv(SESSION_ENV, "gpt-5.1-session")
            runner = "codex:gpt-5.1-session"
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        _no_uncontained_rung(scratch, text)
        assert executor.sandbox_rung(runner, PROFILE_AGENT_WRITE, write_scope=scope) == "plugin-landlock"
        rows = scratch.wrapper.records(probe=False)
        assert len(rows) == len(scratch.write_calls()) == 2
        # Under `--host codex` the preflight also runs the host's nested-exec
        # probe: one unwrapped read-only turn, never a write and never full
        # access.
        for call in scratch.exec_calls():
            if CONTRACT_MARK not in call["prompt"]:
                assert attempt == "session"
                assert call["args"][call["args"].index("--sandbox") + 1] == "read-only"
        for row in rows:
            assert row["cmd"][row["cmd"].index("--model") + 1] == runner.split(":", 1)[1]
            assert row["cmd"][row["cmd"].index("--sandbox") + 1] == "danger-full-access"
        assert [row.get("sandbox_rung") for row in scratch.report(mode)["tasks"]] == [
            "plugin-landlock"] * 2
        assert scratch.completed(mode)

    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_a_session_model_with_no_usable_jail_is_refused(
        self, scratch, capsys, monkeypatch, mode, scope,
    ):
        """No usable jail: the Codex session model's write is refused (or
        falls through by name to the codex host's floor, which is refused
        too) — never `unsandboxed`."""
        scratch.configure(driver=False)
        monkeypatch.setenv("MULTI_LLM_HOST", "codex")
        monkeypatch.setenv(SESSION_ENV, "gpt-5.5")
        scratch.wrapper.set(probe="absent")
        before = _bytes(scratch.target(mode))
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        reason = _assert_refused_at_batch_0(scratch, mode, code, text, cap.err, before)
        assert LEG_SENTENCES["absent"] in reason
        _assert_session_refused(cap.out, reason)
        _no_uncontained_rung(scratch, text)
        assert scratch.exec_calls() == []

    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_a_mid_turn_jail_denial_is_reported_and_never_retried(
        self, scratch, capsys, mode, scope,
    ):
        """§8.2's last shape: a codex-internal EACCES during the turn is a
        `[JAIL_DENIED]` failure, one attempt, never moved to another rung."""
        denied = Path(os.environ["HOME"]) / "elsewhere" / "state.db"
        denied.parent.mkdir(parents=True)
        scratch.wrapper.set(dispatch=f"eacces:{denied}")
        before = _bytes(scratch.target(mode))
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 3, text[-4000:]
        jail_lines = [line for line in text.splitlines() if "[JAIL_DENIED]" in line]
        assert jail_lines, text[-4000:]
        assert "on rung 'plugin-landlock'" in jail_lines[0]
        assert "denied ~/elsewhere/state.db" in jail_lines[0]
        # One attempt, on the rung; the fake wrapper failed it before any exec.
        assert len(scratch.wrapper.records(probe=False)) == 1
        assert scratch.exec_calls() == []
        assert _bytes(scratch.target(mode)) == before
        assert not scratch.completed(mode)
        assert _in_flight(scratch.status(mode)) == []
        _no_uncontained_rung(scratch, text)


# ---------------------------------------------------------------------------
# §7 step 6 (F5, adopted): the chat's posture, only below the jail
# ---------------------------------------------------------------------------

class TestF5BelowTheJail:
    @pytest.mark.parametrize("mode, scope", STEP_3)
    def test_the_posture_takes_the_uncontained_rung_only_when_the_jail_cannot(
        self, scratch, capsys, monkeypatch, mode, scope,
    ):
        scratch.wrapper.set(probe="absent")
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        before = scratch.target(mode).read_text(encoding="utf-8")
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        bypassed = [line for line in text.splitlines() if "[SANDBOX_BYPASSED]" in line]
        assert len(bypassed) == 1, bypassed
        assert "F5" in bypassed[0] and "danger-full-access sandbox posture" in bypassed[0]
        assert "write phase" in bypassed[0] and "read-only phase" not in bypassed[0]
        assert "fallback rung 'plugin-landlock' absent" in bypassed[0]
        assert "[SANDBOX_INCIDENTAL]" not in text
        assert executor.sandbox_rung(RUNNER, PROFILE_AGENT_WRITE, write_scope=scope) == "unsandboxed"
        # The argv reached the mock unwrapped, with codex's full access.
        assert scratch.wrapper.records(probe=False) == []
        calls = scratch.write_calls()
        assert len(calls) == 2
        for call in calls:
            assert call["args"][call["args"].index("--sandbox") + 1] == "danger-full-access"
            assert not _developer_instructions(call["args"])
        assert scratch.target(mode).read_text(encoding="utf-8") == before + "".join(
            marker(key) for key in scratch.keys(mode))
        assert scratch.completed(mode)


# ---------------------------------------------------------------------------
# §8.3 item 11, and item 13 "Out-of-scope edits are denied"
# ---------------------------------------------------------------------------

class TestADeniedEditFailsTheBatch:
    def test_a_denied_plan_edit_is_a_failed_batch_never_applied(self, scratch, capsys):
        """The jail refused the patch on the plan; codex ends the turn with
        exit 0 and the runner claims `applied` anyway."""
        mode = "apply-suggestions"
        scratch.script(mode, batch0=[{
            "events": [file_change("failed", str(scratch.plan))],
            "claim": {"status": "applied"},
        }])
        before = scratch.plan.read_bytes()
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 3, text[-4000:]
        assert scratch.plan.read_bytes() == before
        assert not scratch.completed(mode)
        row = scratch.report(mode)["tasks"][0]
        assert row["status"] == executor.STATUS_BATCH_FAILED
        assert row["write_outcome"] != "applied"
        assert row.get("sandbox_rung") == "plugin-landlock"
        # The loop stopped: batch 1 never reached the mock.
        assert len(scratch.write_calls()) == 1
        assert "FAILED (jail-denied edit)" in text
        assert "plugin-landlock jail refused the runner's write to plans/p.md" in text
        assert _in_flight(scratch.status(mode)) == []
        _no_uncontained_rung(scratch, text)

    @pytest.mark.parametrize("edits_target", [False, True], ids=["nothing-moved", "target-moved"])
    def test_task_suggestions_denied_on_the_plan_and_state(self, scratch, capsys, edits_target):
        """The same shape for the tasks scope, the plan file and state.json
        being the paths outside it: a failed batch when no bytes moved, a
        partial application when the tasks file changed — never applied."""
        mode = "apply-task-suggestions"
        step: Dict[str, Any] = {
            "events": [file_change(
                "failed", str(scratch.plan), str(scratch.out_dir / "state.json"),
            )],
            "claim": {"status": "applied"},
        }
        if edits_target:
            step["append"] = {str(scratch.tasks): marker("{id}")}
        scratch.script(mode, batch0=[step])
        plan_before = scratch.plan.read_bytes()
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 3, text[-4000:]
        assert scratch.plan.read_bytes() == plan_before
        assert not scratch.completed(mode)
        row = scratch.report(mode)["tasks"][0]
        assert row["write_outcome"] != "applied"
        assert row["status"] == (
            executor.STATUS_PARTIAL if edits_target else executor.STATUS_BATCH_FAILED
        )
        assert "plugin-landlock jail refused the runner's write to" in text
        assert "plans/p.md" in text and "plans/p/state.json" in text
        assert len(scratch.write_calls()) == 1
        assert _in_flight(scratch.status(mode)) == []


# ---------------------------------------------------------------------------
# §8.3 item 14: a partial edit stays recoverable, and a retry does not repeat
# work
# ---------------------------------------------------------------------------

BATCH1_EDIT = "\n<!-- batch-1 edit -->\n"


def _denied_outside(scratch: Scratch, mode: str) -> Path:
    """The path outside ``mode``'s grant that batch 1 tries: README.md for the
    step-3 scopes, `.git/config` for the repo-wide kinds (§8.3 "Step 3b is
    done when": item 14's partial application is an edit in a subdirectory
    beside a denied write to `.git/config`)."""
    return scratch.root / ".git" / "config" if mode in REPO_WIDE else scratch.readme


def _partial_batch_1(scratch: Scratch, mode: str, resumed: Dict[str, Any]) -> None:
    """Batch 0 applies cleanly. Batch 1's turn edits its target with one
    `apply_patch`, then tries a second on a path outside the scope, which the
    jail refused; ``resumed`` is what the re-dispatched turn does."""
    target = str(scratch.target(mode))
    scratch.script(mode, batch1=[
        {
            "append": {target: BATCH1_EDIT},
            "events": [file_change("completed", target),
                       file_change("failed", str(_denied_outside(scratch, mode)))],
            "claim": {"status": "applied"},
        },
        resumed,
    ])


def _first_run_stops_at_batch_1(scratch: Scratch, mode: str, capsys) -> Tuple[str, str]:
    """Run the phase; return (the target's bytes before batch 1, batch 1's id)."""
    target = scratch.target(mode)
    original = target.read_text(encoding="utf-8") if target.exists() else ""
    denied = _denied_outside(scratch, mode)
    denied_before = denied.read_bytes()
    code = scratch.run(mode)
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert code == 3, text[-4000:]
    keys = scratch.keys(mode)
    pre_batch_1 = original + marker(keys[0])
    assert scratch.target(mode).read_text(encoding="utf-8") == pre_batch_1 + BATCH1_EDIT
    rows = scratch.report(mode)["tasks"]
    assert [row["write_outcome"] for row in rows] == [
        executor.WRITE_APPLIED, executor.WRITE_PARTIAL]
    assert rows[1]["status"] == executor.STATUS_PARTIAL
    assert rows[1].get("sandbox_rung") == "plugin-landlock"
    assert "PARTIAL APPLICATION (jail-denied edit)" in text
    assert str(denied.relative_to(scratch.root)) in text
    assert denied.read_bytes() == denied_before
    assert not scratch.completed(mode)
    assert _in_flight(scratch.status(mode)) == []
    task_1 = scratch.write_tasks(mode)[1]["task_id"]
    # Batch 1's pre-batch snapshot holds the scheduler's own copy of the
    # bytes before batch 1 (what `util restore-batch` restores from) — in
    # the phase dir for the two narrow scopes, outside the grant for a
    # repo-wide batch on the rung (review fixes 2026-10-05).
    phase_dir = scratch.phase_dir(mode)
    snapshot_root = Path(safety_dir(str(phase_dir)))
    assert (snapshot_root / "pre_batch" / task_1).is_dir()
    if mode in REPO_WIDE:
        assert snapshot_root != phase_dir and not snapshot_root.is_relative_to(scratch.root)
    else:
        assert snapshot_root == phase_dir
    _no_uncontained_rung(scratch, text)
    return pre_batch_1, task_1


def _in_repo(row: Dict[str, Any], root: Path) -> List[str]:
    return [entry for _flag, entry in _allow(row) if str(root) in entry]


class TestAPartialEditStaysRecoverable:
    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_restore_then_resume_redispatches_batch_1_alone(self, scratch, capsys, mode, scope):
        _partial_batch_1(scratch, mode, {
            "append": {str(scratch.target(mode)): BATCH1_EDIT},
            "claim": {"status": "applied"},
        })
        pre_batch_1, task_1 = _first_run_stops_at_batch_1(scratch, mode, capsys)
        first_rows = scratch.wrapper.records(probe=False)
        assert len(first_rows) == 2

        # `util restore-batch --all-failed` returns the target to its bytes
        # before batch 1: batch 0's edit present, batch 1's absent.
        phase_dir = scratch.phase_dir(mode)
        code = multi_llm.main(["util", "restore-batch", "--phase-dir", str(phase_dir),
                               "--all-failed"])
        restored = capsys.readouterr()
        assert code == 0, restored.out + restored.err
        assert scratch.target(mode).read_text(encoding="utf-8") == pre_batch_1
        assert any(record["event"] == "restored" and record.get("task_id") == task_1
                   for record in AuditLog(str(phase_dir)).records())

        # `--resume` skips batch 0: the mock's log shows one exec, batch 1's,
        # on the rung, under the same allow-list as its first dispatch.
        execs_before = len(scratch.exec_calls())
        code = scratch.run(mode, "--resume")
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        resumed = scratch.exec_calls()[execs_before:]
        assert len(resumed) == 1
        assert BATCH_TEXT[mode][1] in resumed[0]["prompt"]
        assert BATCH_TEXT[mode][0] not in resumed[0]["prompt"]
        rows = scratch.wrapper.records(probe=False)
        assert len(rows) == 3
        assert _allow(rows[2]) == _allow(first_rows[1])
        real_target = os.path.realpath(scratch.target(mode))
        if scope == "plan-file":
            assert _in_repo(rows[2], scratch.root) == [f"file:{real_target}"]
        elif scope == "tasks":
            assert _in_repo(rows[2], scratch.root) == [f"dir:{os.path.realpath(scratch.tasks.parent)}"]
        else:
            assert _in_repo(rows[2], scratch.root) == [f"dir:{scratch.root}"]
        # Batch 1's edit appears once; the restore's new inode is not read as
        # a replaced target (the identity compared is this dispatch's own).
        assert scratch.target(mode).read_text(encoding="utf-8") == pre_batch_1 + BATCH1_EDIT
        assert "target replaced" not in text
        assert scratch.completed(mode)
        outcomes = {row["task_id"]: row for row in scratch.report(mode)["tasks"]}
        assert outcomes[task_1]["write_outcome"] == executor.WRITE_APPLIED
        assert outcomes[task_1].get("sandbox_rung") == "plugin-landlock"
        _no_uncontained_rung(scratch, text)

    @pytest.mark.parametrize("mode, scope", ENABLED)
    def test_resume_without_a_restore_reuses_the_first_snapshot(self, scratch, capsys, mode, scope):
        """The resumed turn finds its edit already present and reports it
        applied without writing again: recorded `applied`, the edit appears
        once, and `util restore-batch` afterwards still reaches the bytes
        before batch 1 (the first run's snapshot was reused, not retaken)."""
        _partial_batch_1(scratch, mode, {"claim": {"status": "applied"}})
        pre_batch_1, task_1 = _first_run_stops_at_batch_1(scratch, mode, capsys)
        batch_1_ids = scratch.write_tasks(mode)[1]["expected_ids"]

        execs_before = len(scratch.exec_calls())
        code = scratch.run(mode, "--resume")
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        resumed = scratch.exec_calls()[execs_before:]
        assert len(resumed) == 1
        prompt = resumed[0]["prompt"]
        assert all(f"`{key}`" in prompt for key in batch_1_ids)
        already = "already implemented" if mode == "implement" else "found already applied"
        assert already in prompt
        assert scratch.target(mode).read_text(encoding="utf-8") == pre_batch_1 + BATCH1_EDIT
        assert scratch.completed(mode)
        outcomes = {row["task_id"]: row for row in scratch.report(mode)["tasks"]}
        assert outcomes[task_1]["write_outcome"] == executor.WRITE_APPLIED

        code = multi_llm.main(["util", "restore-batch", "--phase-dir",
                               str(scratch.phase_dir(mode)), "--task-id", task_1])
        restored = capsys.readouterr()
        assert code == 0, restored.out + restored.err
        assert scratch.target(mode).read_text(encoding="utf-8") == pre_batch_1


# ---------------------------------------------------------------------------
# §8.3 item 7 and item 13's last leg: the standard sandbox still wins
# ---------------------------------------------------------------------------

class TestTheStandardSandboxStillWins:
    @pytest.mark.parametrize("mode", [mode for mode, _ in ENABLED])
    def test_a_passing_standard_probe_dispatches_today_s_argv(
        self, scratch, capsys, monkeypatch, mode,
    ):
        """With codex's own `workspace-write` probe `ok`, every kind — the
        repo-wide ones included, since their gate applies to the rung only —
        dispatches today's argv and lands its edit; no jail line is printed."""
        monkeypatch.delenv("MOCK_LLM_SANDBOX_BROKEN")
        target = scratch.target(mode)
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        assert "[SANDBOX_INCIDENTAL]" not in text and "[SANDBOX_BYPASSED]" not in text
        assert scratch.wrapper.records() == []
        calls = scratch.write_calls()
        assert len(calls) == len(scratch.write_tasks(mode)) >= 1
        for call in calls:
            assert call["args"][call["args"].index("--sandbox") + 1] == "workspace-write"
            assert not _developer_instructions(call["args"])
        assert target.read_text(encoding="utf-8") == before + "".join(
            marker(key) for key in scratch.keys(mode))
        assert scratch.completed(mode)
        assert executor.sandbox_rung(
            RUNNER, PROFILE_AGENT_WRITE,
            write_scope=executor.WRITE_SCOPE_BY_KIND[scratch.write_tasks(mode)[0]["kind"]],
        ) is None


# ---------------------------------------------------------------------------
# §8.3 item 3: the §2b judges run on the rung
# ---------------------------------------------------------------------------

class TestTheJudgesRunOnTheRung:
    def test_the_judge_dispatch_is_jailed_with_one_incidental_line(self, judged, capsys):
        code = judged.run("apply-suggestions", "--claude-decide")
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        assert "[SANDBOX_BYPASSED]" not in text
        read_lines = [line for line in text.splitlines()
                      if "[SANDBOX_INCIDENTAL]" in line and "read-only phase" in line]
        assert len(read_lines) == 1, read_lines
        line = read_lines[0]
        assert "'plugin-landlock'" in line and BWRAP_LINE in line
        # The executor lane runs no mode check, and says so.
        assert "a changed file mode is neither stopped nor checked" in line
        assert executor.sandbox_rung(RUNNER, PROFILE_AGENT_READONLY) == "plugin-landlock"

        judge_calls = [call for call in judged.exec_calls() if CONTRACT_MARK not in call["prompt"]]
        assert len(judge_calls) == 1
        assert judge_calls[0]["args"][judge_calls[0]["args"].index("--sandbox") + 1] == (
            "danger-full-access")
        rows = judged.wrapper.records(probe=False)
        assert len(rows) == len(judged.exec_calls())
        judge_row = rows[0]
        [developer] = _developer_instructions(judge_row["cmd"])
        assert "You may write only" not in developer and "is the directory" not in developer
        # A read-only grant: nothing in the repository is writable.
        assert _in_repo(judge_row, judged.root) == []
        # The judge's approval reached the write batches, which ran on the rung.
        assert judged.completed("apply-suggestions")
        expected = [key for task in judged.write_tasks("apply-suggestions")
                    for key in task["expected_ids"]]
        assert len(expected) == 3
        _no_uncontained_rung(judged, text)


# ---------------------------------------------------------------------------
# Step 3b (2026-10-02): the repo-wide kinds' own legs — item 13's out-of-scope
# targets, a declared new top-level entry refused before dispatch, and the
# denied-write cases through the real scheduler inside the real jail (tier 2)
# ---------------------------------------------------------------------------

class TestRepoWideDeniedEdits:
    @pytest.mark.parametrize("edits_target", [False, True], ids=["nothing-moved", "target-moved"])
    @pytest.mark.parametrize("mode", REPO_WIDE)
    def test_a_denied_write_to_git_config_and_state_never_reads_applied(
        self, scratch, capsys, mode, edits_target,
    ):
        """Item 13 for the repo-wide kinds, under the root grant (2026-10-05):
        `.git/config` and this run's `state.json` are inside the grant now, so
        a `file_change` codex reports as failed there is the jail's own
        failure — still never applied: a failed batch when nothing moved, a
        partial application when the declared file changed, `.git/config`
        intact."""
        config = scratch.root / ".git" / "config"
        step: Dict[str, Any] = {
            "events": [file_change("failed", str(config), str(scratch.out_dir / "state.json"))],
            "claim": {"status": "applied"},
        }
        if edits_target:
            step["append"] = {str(scratch.target(mode)): marker("{id}")}
        scratch.script(mode, batch0=[step])
        config_before = config.read_bytes()
        code = scratch.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 3, text[-4000:]
        assert config.read_bytes() == config_before
        assert not scratch.completed(mode)
        row = scratch.report(mode)["tasks"][0]
        assert row["write_outcome"] != "applied"
        assert row["status"] == (
            executor.STATUS_PARTIAL if edits_target else executor.STATUS_BATCH_FAILED
        )
        assert row.get("sandbox_rung") == "plugin-landlock"
        assert "plugin-landlock jail refused the runner's write to" in text
        assert ".git/config" in text and "plans/p/state.json" in text
        assert "which the jail granted whole" in text
        assert "outside this batch's grant" not in text
        assert len(scratch.write_calls()) == 1
        assert _in_flight(scratch.status(mode)) == []
        _no_uncontained_rung(scratch, text)


class TestADeclaredNewTopLevelEntryIsDispatched:
    def test_implement_declaring_a_new_root_file_runs_on_the_jail(
        self, scratch, capsys,
    ):
        """Root grant (2026-10-05): a declared file that is a new entry at
        the repository root is dispatched on the jail like any other and
        lands; until then it was refused before any child started, naming
        F6 (the racing re-run's `index.html`)."""
        scratch.tasks.write_text(
            TASKS_BODY.replace("Files to create: src/main.py", "Files to create: newtop.py"),
            encoding="utf-8",
        )
        scratch.script("implement", batch0=[
            {"append": {str(scratch.root / "newtop.py"): marker("{id}")}},
        ])
        code = scratch.run("implement")
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-4000:]
        assert "would be a new entry directly in" not in text
        assert "(F6)" not in text
        assert (scratch.root / "newtop.py").read_text(encoding="utf-8") == marker("T001")
        assert len(scratch.write_calls()) == 2
        rows = scratch.wrapper.records(probe=False)
        assert len(rows) == 2
        assert all(_in_repo(row, scratch.root) == [f"dir:{scratch.root}"] for row in rows)
        assert len(scratch.wrapper.records(probe=True)) == 1
        assert scratch.completed("implement")
        assert _in_flight(scratch.status("implement")) == []

    def test_the_same_task_runs_on_codex_s_own_sandbox(self, scratch, capsys, monkeypatch):
        """The refusal is the jail's: with codex's own `workspace-write`
        probe ok the new top-level file is codex's to allow, and it lands."""
        monkeypatch.delenv("MOCK_LLM_SANDBOX_BROKEN")
        scratch.tasks.write_text(
            TASKS_BODY.replace("Files to create: src/main.py", "Files to create: newtop.py"),
            encoding="utf-8",
        )
        scratch.script_path.write_text(json.dumps({"rules": [], "default": {
            "append": {str(scratch.root / "newtop.py"): marker("{id}")},
        }}), encoding="utf-8")
        code = scratch.run("implement")
        cap = capsys.readouterr()
        assert code == 0, (cap.out + cap.err)[-4000:]
        assert (scratch.root / "newtop.py").exists()
        assert scratch.wrapper.records() == []


#: Writes outside the repository, as the "model" runs them from inside its
#: batch, each tagged so the test can grep for a leak: still the jail's to
#: deny under the root grant.
DENIED_SHELL = {
    "tmp": "echo MARK3B > /tmp/mllm-3b-accept-$$",
    "shm": "echo MARK3B > /dev/shm/mllm-3b-accept-$$",
}
#: Writes the root grant admits (2026-10-05): the hands-off set is the
#: developer message's and the scheduler's business, not the jail's. `.git/`
#: is pruned from the sweep, so its writes surface only through the
#: `[GIT_CONFIG_CHANGED]` line; `newtop.txt` and `.codex/x` are undeclared
#: writes the sweep names and the policy admits.
POSITIVE_SHELL = {
    "tmpdir": 'echo MARK3B-ok > "$TMPDIR/scratch.txt"',
    "status": "git status --porcelain",
    "gitx": "echo MARK3B > .git/x",
    "config": "git config user.name MARK3B",
    "hook": "echo MARK3B > .git/hooks/pre-commit",
    "newtop": "echo MARK3B > newtop.txt",
    "codexx": "echo MARK3B > .codex/x",
}


@pytest.mark.landlock
@pytest.mark.sandbox_probe  # the suite's guard would answer the production wrapper 126
class TestTheRealJailThroughTheScheduler:
    """Tier 2: the real `landlock_exec.py`, the real scheduler, the mock codex
    jailed for real. Under the root grant (2026-10-05) the jail denies only
    writes outside the repository; what the model is told to leave alone is
    checked afterwards by the sweep, `driver.write_paths` and the warn-only
    `[GIT_CONFIG_CHANGED]` line — nothing is backed up or restored."""

    @pytest.fixture
    def jailed(self, tmp_path, monkeypatch):
        from tests.test_landlock_exec import require_landlock
        require_landlock()
        yield Scratch(tmp_path, monkeypatch, real_jail=True)
        registry._config = None
        registry._config_key = None

    @pytest.mark.parametrize("mode", REPO_WIDE)
    def test_outside_writes_are_denied_inside_ones_land_and_the_metadata_line_fires(
        self, jailed, capsys, mode,
    ):
        root = jailed.root
        head = _git(root, "rev-parse", "HEAD")
        commits = _git(root, "rev-list", "--count", "HEAD")
        keep = {
            p: p.read_bytes() for p in (
                root / ".git" / "HEAD", root / ".codex" / "config.toml",
                root / ".multi-llm" / "providers.yaml", root / "README.md",
            )
        }
        config_before = (root / ".git" / "config").read_bytes()
        target = jailed.target(mode)
        jailed.script(mode)
        script = json.loads(jailed.script_path.read_text(encoding="utf-8"))
        script["rules"] = [{
            "match": BATCH_TEXT[mode][0],
            "steps": [{"shell": {**DENIED_SHELL, **POSITIVE_SHELL},
                       "append": {str(target): marker("{id}")}}],
        }]
        jailed.script_path.write_text(json.dumps(script), encoding="utf-8")

        code = jailed.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-6000:]
        assert jailed.completed(mode)
        report = jailed.report(mode)
        assert [row.get("sandbox_rung") for row in report["tasks"]] == ["plugin-landlock"] * 2
        assert [row["write_outcome"] for row in report["tasks"]] == ["applied"] * 2
        assert "[SANDBOX_INCIDENTAL]" in text and "[SANDBOX_BYPASSED]" not in text

        results = jailed.shell_results()
        assert set(results) == set(DENIED_SHELL) | set(POSITIVE_SHELL), results
        leaked = {name for name in DENIED_SHELL if results[name] == 0}
        assert leaked == set(), leaked
        assert all(results[name] == 0 for name in POSITIVE_SHELL), results
        assert not list(Path("/tmp").glob("mllm-3b-accept-*"))
        assert not list(Path("/dev/shm").glob("mllm-3b-accept-*"))
        # Inside the repository the root grant admitted everything, the
        # hands-off set included — and nothing restored it.
        assert (root / ".git" / "x").read_bytes() == b"MARK3B\n"
        assert (root / ".git" / "hooks" / "pre-commit").read_bytes() == b"MARK3B\n"
        assert b"MARK3B" in (root / ".git" / "config").read_bytes()
        assert (root / "newtop.txt").read_bytes() == b"MARK3B\n"
        assert (root / ".codex" / "x").read_bytes() == b"MARK3B\n"
        assert {p: p.read_bytes() for p in keep} == keep
        assert _git(root, "rev-parse", "HEAD") == head
        assert _git(root, "rev-list", "--count", "HEAD") == commits
        # The warn-only line, once, for the batch that changed the metadata,
        # naming what moved; the second batch drew none.
        warned = [line for line in text.splitlines() if "[GIT_CONFIG_CHANGED]" in line]
        assert len(warned) == 1, warned
        assert ".git/config changed" in warned[0]
        assert ".git/hooks/pre-commit added" in warned[0]
        assert "warn-only" in warned[0]
        # The sweep named the undeclared in-tree writes; the policy admitted them.
        assert "changed outside this task's declared set" in text
        assert "newtop.txt" in text and ".codex/x" in text
        # The declared edit landed, once per batch.
        assert target.read_text(encoding="utf-8").count("<!-- applied ") == 2
        assert config_before != (root / ".git" / "config").read_bytes()

    @pytest.mark.parametrize("mode", REPO_WIDE)
    def test_a_denied_path_written_inside_the_jail_is_caught_by_the_policy(
        self, jailed, capsys, mode,
    ):
        """`driver.write_paths` stays the detective control it is for every
        other runner: the jail lets a write to `.multi-llm/providers.yaml`
        land (the root grant covers it), the post-run sweep sees it moved,
        the policy denies it and the batch fails without a restore (no
        rollback point is ever taken for a denied path)."""
        root = jailed.root
        target = jailed.target(mode)
        jailed.script(mode)
        script = json.loads(jailed.script_path.read_text(encoding="utf-8"))
        script["rules"] = [{
            "match": BATCH_TEXT[mode][0],
            "steps": [{"shell": {"providers": "echo MARK3B >> .multi-llm/providers.yaml"},
                       "append": {str(target): marker("{id}")}}],
        }]
        jailed.script_path.write_text(json.dumps(script), encoding="utf-8")

        code = jailed.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 3, text[-6000:]
        assert jailed.shell_results() == {"providers": 0}
        assert b"MARK3B" in (root / ".multi-llm" / "providers.yaml").read_bytes()
        row = jailed.report(mode)["tasks"][0]
        assert row["status"] == executor.STATUS_POLICY_VIOLATION
        assert row["write_outcome"] == executor.WRITE_POLICY_VIOLATION
        assert ".multi-llm/providers.yaml" in text and "driver.write_paths.deny" in text
        assert not jailed.completed(mode)
        assert len(jailed.write_calls()) == 1


    def test_a_forged_in_phase_rollback_point_is_inert_on_the_real_jail(self, jailed, capsys):
        """Review fixes 2026-10-05: under the root grant the batch CAN write
        the phase dir's `pre_batch/`, `rollback_store/` and audit log — and
        does, here — but the run's own evidence lives under the state home,
        which the jail denies, and `util restore-batch` reads only that."""
        from utils import write_safety
        from utils.write_safety import AuditLog, TaskSnapshot
        mode = "implement"
        root = jailed.root
        phase_dir = jailed.phase_dir(mode)
        external = Path(write_safety.external_safety_dir(str(phase_dir)))
        assert not external.is_relative_to(root)
        jailed.script(mode)
        script = json.loads(jailed.script_path.read_text(encoding="utf-8"))
        forged_manifest = json.dumps({
            "snapshot_version": 1, "task_id": "T001", "root": str(root),
            "files": [{"path": "src/main.py", "absolute": str(jailed.main_py), "existed": True,
                       "sha256_before": "0" * 64, "size_before": 8,
                       "payload": "000-src_main.py", "declared": True}],
            "expected_ids": ["T001"],
        })
        forged_audit = json.dumps({
            "event": "completed", "task_id": "T001", "kind": "implement", "status": "partial",
            "write_outcome": "partial-application", "files_touched": ["src/main.py"],
            "observed": {"src/main.py": {"sha256_after": "0" * 64}}, "expected_ids": ["T001"],
        })
        shell = {
            "forge_dir": "mkdir -p plans/p/implement/pre_batch/T001/files plans/p/implement/rollback_store/ab",
            "forge_payload": "echo planted > plans/p/implement/pre_batch/T001/files/000-src_main.py",
            "forge_manifest": f"printf %s {shlex.quote(forged_manifest)} > plans/p/implement/pre_batch/T001/manifest.json",
            "forge_audit": f"printf '%s\\n' {shlex.quote(forged_audit)} >> plans/p/implement/subagent_audit.jsonl",
            "forge_blob": "echo planted > plans/p/implement/rollback_store/ab/abcd",
            "outside": f"echo planted > {shlex.quote(str(external / 'subagent_audit.jsonl'))}",
        }
        script["rules"] = [{
            "match": BATCH_TEXT[mode][0],
            "steps": [{"shell": shell, "append": {str(jailed.main_py): marker("{id}")}}],
        }]
        jailed.script_path.write_text(json.dumps(script), encoding="utf-8")

        code = jailed.run(mode)
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-6000:]
        results = jailed.shell_results()
        assert {name for name in shell if results[name] == 0} == set(shell) - {"outside"}, results
        assert results["outside"] != 0
        # The in-phase forgeries landed (the root grant covers them) …
        assert (phase_dir / "pre_batch" / "T001" / "manifest.json").is_file()
        assert "planted" in (phase_dir / "pre_batch" / "T001" / "files" / "000-src_main.py").read_text()
        assert '"sha256_after": "000' in (phase_dir / "subagent_audit.jsonl").read_text(encoding="utf-8")
        # … and the run's own evidence is elsewhere, untouched by them.
        assert external.is_dir()
        assert (external / "pre_batch" / "T001" / "manifest.json").is_file()
        assert str(external) in text and "live outside it" in text
        real = TaskSnapshot.load(str(phase_dir), "T001", expected_root=str(root))
        assert real is not None and os.path.realpath(real.root) == str(root)
        assert [e.existed for e in real.entries] == [False], "main.py did not exist before T001"
        records = AuditLog(str(phase_dir)).records()
        assert all(r.get("observed", {}).get("src/main.py", {}).get("sha256_after") != "0" * 64
                   for r in records if r.get("event") == "completed")
        # The restore the stop message would name lands the REAL rollback:
        # the file is removed (absent before T001), never written "planted".
        code = multi_llm.main(["util", "restore-batch", "--phase-dir", str(phase_dir),
                               "--task-id", "T001", "--force"])
        restored = capsys.readouterr()
        assert code == 0, restored.out + restored.err
        assert not jailed.main_py.exists()


# ---------------------------------------------------------------------------
# The root grant (2026-10-05, a user decision): a repo-wide batch may create a
# file directly in the repository root — the racing re-run's `index.html`
# (plans/2026-10-04-racing-rerun/FINDINGS-codex.md, issues 2 and 3) — and the
# plugin's own git calls afterwards run nothing the batch could have planted.
# ---------------------------------------------------------------------------

ROOT_TASKS_BODY = """# Tasks

## Tasks

### T001: Create the page
Create the single-file game page at the repository root.
- Depends on: none
- Files to create: index.html

### T002: Tune the app
Tune the module.
- Depends on: T001
- Files to modify: src/app.py
"""


@pytest.mark.landlock
@pytest.mark.sandbox_probe
class TestARootLevelFileLandsOnTheRealJail:
    """Tier 2: the real wrapper, the real scheduler, the mock codex jailed for
    real. Before the root grant the first task was refused before any child
    started (``index.html would be a new entry directly in <root>``, F6)."""

    @pytest.fixture
    def jailed(self, tmp_path, monkeypatch):
        from tests.test_landlock_exec import require_landlock
        require_landlock()
        yield Scratch(tmp_path, monkeypatch, real_jail=True)
        registry._config = None
        registry._config_key = None

    def test_a_new_root_file_and_a_subdirectory_edit_both_land(self, jailed, capsys):
        root = jailed.root
        jailed.tasks.write_text(ROOT_TASKS_BODY, encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "root tasks")
        head = _git(root, "rev-parse", "HEAD")
        index_html = root / "index.html"
        assert not index_html.exists()
        jailed.script_path.write_text(json.dumps({"rules": [
            {"match": BATCH_TEXT["implement"][0],
             "steps": [{"append": {str(index_html): "<!doctype html>\n"}}]},
            {"match": BATCH_TEXT["implement"][1],
             "steps": [{"append": {str(jailed.app): marker("{id}")}}]},
        ], "default": {}}), encoding="utf-8")

        code = jailed.run("implement")
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert code == 0, text[-6000:]
        assert "would be a new entry directly in" not in text
        assert "(F6)" not in text
        assert jailed.completed("implement")
        report = jailed.report("implement")
        assert [row.get("sandbox_rung") for row in report["tasks"]] == ["plugin-landlock"] * 2
        assert [row["write_outcome"] for row in report["tasks"]] == ["applied"] * 2
        assert "[SANDBOX_INCIDENTAL]" in text and "[SANDBOX_BYPASSED]" not in text
        # Both edits landed: the new root-level file and the subdirectory edit.
        assert index_html.read_text(encoding="utf-8") == "<!doctype html>\n"
        assert jailed.app.read_text(encoding="utf-8") == "x = 1\n" + marker("T002")
        # The batch made no commit and planted nothing in the metadata.
        assert _git(root, "rev-parse", "HEAD") == head
        assert "[GIT_CONFIG_CHANGED]" not in text
        # Each wrapped dispatch carried one dir rule on the repository root.
        rows = [json.loads(line) for line in jailed.calls.read_text(encoding="utf-8").splitlines() if line]
        assert len(jailed.write_calls()) == 2
        assert len(rows) >= 2
