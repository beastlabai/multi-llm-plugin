#!/usr/bin/env python3
"""``improve-plan`` and ``improve-code`` end to end: the real dispatcher, real phases.

**The gap this closes.** ``tests/test_improve_orchestrator.py`` and
``tests/test_improve_steps.py`` stub ``FullWorkflow.run_mode``: they record the
argv each phase *would* receive and write the completion a real phase *would*
record, so no review ever fans out, no validation merges, no write batch is
classified and no ``apply-*`` phase ever finishes under a composite. Every claim
those files make about the pair — "one review, then the apply of its findings",
"the apply gets ``--resume`` only when ...", "``[ALREADY_COMPLETE]`` launches
nothing" — is therefore a claim about argv lists and fake state, and the
composite's contract with the *phases* is exactly what they cannot see. The
container lane (``tests/test_container_improve_modes.py``) covers the real CLIs;
this module covers the cheap, hermetic, every-PR half: a ``multi_llm.py
improve-plan|improve-code`` **subprocess** in a throwaway git repository, with the
suite's mock provider binaries on ``PATH`` and nothing stubbed inside the skill.

**How one process serves both halves.** A composite runs a read phase (a model
fan-out, then validation subagents) and a write phase (apply batches) under ONE
environment, and the mock's write path is environment-driven: ``MOCK_LLM_EDITS``
makes *every* invocation edit files and answer with a batch result, and the
``MOCK_LLM_EDITS_IF_PROMPT_HAS`` gate does not help, because a gated mock still
prints the write-batch document for every prompt (validation included, which is
then a rung-6 ``schema_invalid``). ``tests/test_container_full_chain.py`` solved
that by never running the two in one process; a composite cannot. So the write
side is a **mock plugin** (``MOCK_LLM_PLUGIN``, the supported one-off seam, see
``tests/test_mock_plugin_hook.py``): a module under ``tmp_path`` that wraps the
mock's own ``parse_provider_args`` and, only for a prompt carrying the write
batch's result contract, sets ``MOCK_LLM_EDITS`` (one marker line appended to the
file the batch names) and ``MOCK_LLM_WRITE_IDS`` (the batch's own ``expected_ids``,
read back out of the prompt, so no content hash is hard-coded). Every read prompt
keeps the mock's normal answer: the reviewer from a scenario file, the validation
subagents echoed valid (``MOCK_LLM_ECHO=validation``).

**What the assertions are anchored on**, in order of weight: bytes on disk (the
plan or source file moved by exactly one marker per batch, and ``backup.md`` holds
the plan from before), ``state.json`` (which phases are recorded), the mock call
log (how many model and subagent calls a re-run really made), the apply phase's
own ``write_batch_report.json`` verdicts, and the ``MULTI_LLM_RESULT:`` line. The
printed ``$ uv run ...`` step lines are used once, for the flags each real phase
was handed.

**What is covered**, each through the real dispatcher and the real phases:

* ``improve-plan`` in one invocation (review, validation, apply; no later phase;
  the bytes of the plan; ``backup.md``; the result line's ``user_next_step``), with
  ``--yes`` typed and implied;
* ``improve-code`` on an uncommitted change (scope, fix batches, the files edited);
* nothing to do: exit 2 and ``[ALREADY_COMPLETE]`` with no provider call, also
  under the ``--detach`` every tool route appends (no worker, no claimed run dir);
* ``--interactive``/``-i``: exit 3 before ``--yes`` is implied, nothing written;
* ``--force``: a new round (``--force --rerun-all`` on the review, no ``--resume``
  on the apply, ``backup.md`` re-stamped with the earlier one kept aside), and the
  round whose batches all come back as already landed: the apply is stamped after
  the new review, so the next re-run is ``[ALREADY_COMPLETE]``;
* a run stopped mid-apply, re-run with no flag: the review is not paid for again,
  ``--resume`` is inferred, only the remaining batches are dispatched, and the
  improve-code resume keeps the review's diff base;
* a review already on record is not run again;
* zero findings and every-finding-invalid: the apply is recorded completed, exit 0,
  the re-run exits 2;
* a ``needs-human-decision`` item is decided by the judge agent, nobody is asked
  (stdin is ``/dev/null``).

Usage:
    uv run --project skills/multi-llm -- pytest tests/test_e2e_improve_modes.py -v
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import PERF_SCALE, SkillRunner  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
MULTI_LLM = SKILL_DIR / "multi_llm.py"
MOCK_LLM = Path(__file__).resolve().parent / "mocks" / "mock_llm.py"
RESPONSES = Path(__file__).resolve().parent / "fixtures" / "e2e" / "responses"
PLAN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "e2e" / "plans" / "auth-feature.md"

PLAN_REL = "plans/auth-feature.md"
OUT_REL = "plans/auth-feature"
#: The one reviewer. ``cursor-agent:auto`` is in ``providers.yaml``'s catalog, so
#: the run prints no "unknown model" warning; the mock answers it from the scenario.
MODEL = "cursor-agent:auto"
#: The runner the executor dispatches every subagent to (validation and write).
RUNNER = "codex:gpt-5.5"

#: Every workflow phase a composite must NOT have reached. ``improve-plan`` stops
#: at apply-suggestions, ``improve-code`` at apply-code-fixes; the rest of this
#: list is what a leak into ``full`` would record.
LATER_THAN_PLAN = (
    "review-tasks", "apply-task-suggestions", "generate-tasks", "implement",
    "code-review", "apply-code-fixes",
)
LATER_THAN_CODE = ("generate-tasks", "implement")

# --------------------------------------------------------------------------- #
# The mock plugin: write batches edit a file, everything else is untouched.
# --------------------------------------------------------------------------- #

PLUGIN_SOURCE = '''\
"""Scratch mock plugin for tests/test_e2e_improve_modes.py (see its docstring)."""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("improve_e2e_orig_mock", {mock!r})
_orig = importlib.util.module_from_spec(_spec)
sys.modules["improve_e2e_orig_mock"] = _orig
_spec.loader.exec_module(_orig)

# `suggestion 1 = `<id>``: the result contract spells every expected id of the
# batch, in prompt order, and a compliant runner attests to them.
_IDS = re.compile(r"(?:suggestion|fix|task) \\d+ = `([^`]+)`")
_RESULT_CONTRACT = "## Result (required"


def _die(message):
    print("IMPROVE E2E PLUGIN ERROR: " + message, file=sys.stderr)
    sys.exit(9)


def parse_provider_args(provider):
    args, prompt = _orig.parse_provider_args(provider)
    if not prompt:
        prompt = _orig.read_stdin_prompt()
    if _RESULT_CONTRACT not in prompt:
        return args, prompt   # a read prompt: the mock answers it as it always does
    rules = json.loads(os.environ["IMPROVE_E2E_WRITES"])
    rule = next((r for r in rules if r["when"] in prompt), None)
    if rule is None:
        _die("a write batch matched none of the rules: " + prompt[:80])
    ids = _IDS.findall(prompt)
    if not ids:
        _die("the write batch spelled no expected ids")

    counter = Path(os.environ["IMPROVE_E2E_COUNTER"])
    seen = int(counter.read_text() or "0") + 1 if counter.exists() else 1
    counter.write_text(str(seen))
    if seen == int(os.environ.get("IMPROVE_E2E_FAIL_CALL") or 0):
        # The scripted stop: a runner that tried and could not. No edit, no ids.
        os.environ["MOCK_LLM_WRITE_RESULT"] = json.dumps({{
            "status": "failed", "ids": [], "files_touched": [],
            "summary": "scripted failure of write call %d" % seen,
        }})
        return args, prompt

    path = rule.get("path")
    if path is None:
        found = re.search(rule["path_regex"], prompt)
        if not found:
            _die("no file named in the write batch: " + rule["path_regex"])
        path = next(group for group in found.groups() if group)
    os.environ["MOCK_LLM_WRITE_IDS"] = json.dumps(ids)
    os.environ["MOCK_LLM_EDITS"] = json.dumps([{{
        "path": path, "mode": "patch",
        "content": rule["content"].replace("{{id}}", ids[0]),
    }}])
    return args, prompt


def get_response_content(prompt):
    """Shape the verdicts of the validation subagents; judge what they leave open.

    ``needs-human-decision`` marks the first group of EACH validation batch.
    """
    verdict = os.environ.get("IMPROVE_E2E_VERDICT")
    if verdict and "You are validating review suggestions" in prompt:
        echoed = _orig.echo_subagent_answer("validation", prompt)
        if echoed is not None:
            document = json.loads(echoed)
            for position, group in enumerate(document["groups"]):
                if verdict == "invalid":
                    group["status"] = "invalid"
                elif verdict == "needs-human-decision" and position == 0:
                    group["status"] = "needs-human-decision"
            return json.dumps(document)
    if os.environ.get("IMPROVE_E2E_JUDGE") and "You are validating" not in prompt:
        echoed = _orig.echo_subagent_answer("human_decision_judge", prompt)
        if echoed is not None:
            return echoed
    return _orig.get_response_content(prompt)
'''

#: What each apply phase's write batch appends to the file it names. One line per
#: batch, carrying the batch's first expected id: a batch that ran twice, or one
#: that never ran, is visible in the bytes.
PLAN_MARKER = "\n<!-- improve-modes-e2e applied {id} -->\n"
CODE_MARKER = "\n# improve-modes-e2e fixed {id}\n"
WRITE_RULES = [
    {"when": "Apply the following", "path": PLAN_REL, "content": PLAN_MARKER},
    {"when": "Fix the following",
     "path_regex": r"\*\*File\*\*: (\S+)|related issues in `([^`]+)`",
     "content": CODE_MARKER},
]

# --------------------------------------------------------------------------- #
# The throwaway repository the composites run in
# --------------------------------------------------------------------------- #

COMMITTED = {
    "src/__init__.py": '"""A small auth service."""\n',
    "src/auth.py": '''"""Authentication: registration, login, and token issue."""

import hashlib
import re

EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")


def validate_email(email: str) -> bool:
    """Whether ``email`` looks like an address."""
    return bool(EMAIL_RE.match(email or ""))


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether ``password`` matches ``stored_hash``."""
    return hash_password(password) == stored_hash
''',
    "src/security.py": '''"""Rate limiting and account lockout (stub)."""

MAX_ATTEMPTS = 5
''',
}

#: The uncommitted change ``improve-code`` reviews: it adds a login that records
#: attempts, and the rate limiter it records them in. The code-review fixture
#: (``code_review/container_issues.json``) cites exactly these two files.
WORKING_DIFF = {
    "src/security.py": '''"""Rate limiting and account lockout (work in progress)."""

import time

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300

_attempts = {}


def record_attempt(email, success):
    """Record a login attempt for ``email``."""
    bucket = _attempts.setdefault(email, [])
    bucket.append((time.time(), success))
    return len(bucket)


def is_locked_out(email):
    """Whether ``email`` has too many recent failures."""
    bucket = _attempts.get(email, [])
    recent = [t for t, ok in bucket if not ok and time.time() - t < WINDOW_SECONDS]
    return len(recent) > MAX_ATTEMPTS


def compare_tokens(a, b):
    """Compare two tokens."""
    return a == b
''',
    "src/auth.py": COMMITTED["src/auth.py"].replace(
        "import re\n", "import re\n\nfrom .security import is_locked_out, record_attempt\n", 1
    ) + '''

def login(email: str, password: str, stored_hash: str) -> bool:
    """Attempt a login, recording the attempt for rate limiting."""
    if is_locked_out(email):
        return False
    ok = verify_password(password, stored_hash)
    record_attempt(email, ok)
    return ok
''',
}

GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        env={**os.environ, **GIT_IDENTITY, "GIT_TERMINAL_PROMPT": "0"},
    )


# --------------------------------------------------------------------------- #
# The world: a repo, a mock PATH, a driver config and a plugin
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    """One ``multi_llm.py`` subprocess."""

    argv: List[str]
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    result: Optional[dict] = field(default=None)

    def step_flags(self) -> Dict[str, List[str]]:
        """``{mode word: flags after the plan}`` of each ``$ uv run ...`` step line.

        The composite prints the dispatcher command it hands each real phase; that
        is the one place the argv a phase actually got is observable without
        stubbing the phase.
        """
        steps: Dict[str, List[str]] = {}
        for line in self.stdout.splitlines():
            if not line.startswith("$ "):
                continue
            tokens = shlex.split(line[2:])
            at = next(i for i, token in enumerate(tokens) if token.endswith("multi_llm.py"))
            steps[tokens[at + 1]] = tokens[at + 3:]
        return steps

    def explain(self) -> str:
        return (
            f"argv: {' '.join(self.argv)}\nexit: {self.exit_code}\n"
            f"--- stdout (tail) ---\n{self.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{self.stderr[-3000:]}"
        )


class World:
    """A git repository holding the plan, run as cwd, with the mocks on ``PATH``."""

    def __init__(self, tmp_path: Path, *, working_diff: bool = False):
        self.tmp_path = tmp_path
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.counter = tmp_path / "write-calls.txt"

        for rel, body in COMMITTED.items():
            target = self.repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        (self.repo / "plans").mkdir()
        shutil.copyfile(PLAN_FIXTURE, self.repo / PLAN_REL)
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "init")
        self.original_plan = (self.repo / PLAN_REL).read_bytes()
        if working_diff:
            for rel, body in WORKING_DIFF.items():
                (self.repo / rel).write_text(body, encoding="utf-8")
        self.sources_before = {rel: (self.repo / rel).read_bytes() for rel in WORKING_DIFF}

        self.scenario = tmp_path / "scenario.yaml"
        self.set_review_answers()
        plugin = tmp_path / "improve_mock_plugin.py"
        plugin.write_text(PLUGIN_SOURCE.format(mock=str(MOCK_LLM)), encoding="utf-8")

        self.runner = SkillRunner(
            tmp_path / "harness",
            scenario_path=self.scenario,
            mock_plugin=plugin,
            extra_env={
                # Validation subagents answer every group in their prompt valid.
                "MOCK_LLM_ECHO": "validation",
                "IMPROVE_E2E_WRITES": json.dumps(WRITE_RULES),
                "IMPROVE_E2E_COUNTER": str(self.counter),
            },
        )
        base = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text(encoding="utf-8"))
        base.setdefault("driver", {})["subagent_runner"] = RUNNER
        config = tmp_path / "driver-providers.yaml"
        config.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        self.runner.extra_env["MULTI_LLM_PROVIDERS_CONFIG"] = str(config)

    def set_review_answers(
        self, plan_review: Optional[Path] = None, code_review: Optional[Path] = None
    ) -> None:
        """What the one mock reviewer answers, for the plan and for the code review.

        A second round that must find something DIFFERENT (so the apply phase has
        new batches to dispatch) swaps the plan-review fixture between runs, and a
        review that finds NOTHING points either at ``empty.json``; the scenario is
        read afresh by every mock invocation.
        """
        self.scenario.write_text(yaml.safe_dump({
            "name": "improve modes",
            # An unanswered read prompt fails loudly instead of getting a generic
            # list that would look like a review.
            "on_unmatched": "fail",
            "prompts": [
                {"pattern": "You are reviewing an implementation plan for completeness",
                 "fixture": str(plan_review or RESPONSES / "review_plan" / "raw_suggestions.json")},
                {"pattern": "You are reviewing code changes that implement a feature plan",
                 "fixture": str(code_review or RESPONSES / "code_review" / "container_issues.json")},
            ],
        }, sort_keys=False), encoding="utf-8")

    # -- running ----------------------------------------------------------- #

    def env(self, **extra: str) -> Dict[str, str]:
        env = self.runner._build_env()
        # The route and host pick only the spelling of a printed command; a
        # developer shell exporting either would move every `command` asserted here.
        for name in ("MULTI_LLM_ROUTE", "MULTI_LLM_HOST", "MULTI_LLM_HOST_SESSION_MODEL",
                     "MULTI_LLM_LEGACY_INSTRUCTIONS"):
            env.pop(name, None)
        env.update(extra)
        return env

    def run(self, mode: str, *flags: str, plan: str = PLAN_REL, env: Optional[dict] = None,
            timeout: int = 120) -> Run:
        argv = [sys.executable, str(MULTI_LLM), mode, plan, *flags]
        started = time.monotonic()
        done = subprocess.run(
            # stdin is /dev/null: an unattended run that asked anything would see EOF
            argv, cwd=self.repo, env=env or self.env(), stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout * PERF_SCALE,
        )
        return Run(
            argv=argv, exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr,
            seconds=time.monotonic() - started,
            result=parse_result_line(done.stdout),
        )

    def composite(self, mode: str, *flags: str, **kwargs) -> Run:
        """``<mode> <plan> --models <one reviewer> --yes --exec-subagents``."""
        return self.run(
            mode, "--models", MODEL, "--yes", "--exec-subagents", *flags, **kwargs
        )

    # -- reading what a run left ------------------------------------------- #

    @property
    def out(self) -> Path:
        return self.repo / OUT_REL

    @property
    def plan_bytes(self) -> bytes:
        return (self.repo / PLAN_REL).read_bytes()

    def state(self) -> dict:
        return json.loads((self.out / "state.json").read_text(encoding="utf-8"))

    def recorded_phases(self) -> set:
        state = self.state()
        return set(state.get("phases_completed") or {}) | set(state.get("phases_skipped") or {})

    def report(self, phase_dir: str) -> dict:
        return json.loads((self.out / phase_dir / "write_batch_report.json").read_text())

    def calls(self) -> List[dict]:
        log = self.runner.call_log_path
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    def calls_of(self, kind: str) -> List[dict]:
        return [call for call in self.calls() if classify(call) == kind]


def classify(call: dict) -> str:
    """What a mock invocation was, by what its prompt asked."""
    prompt = call["prompt"]
    if "## Result (required" in prompt:
        return "write"
    if prompt.startswith("You are validating review suggestions"):
        return "validate"
    if prompt.startswith("You are deciding what to do with"):
        return "judge"
    if prompt.startswith("You are reviewing"):
        return "review"
    return "other"


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


@pytest.fixture
def code_world(tmp_path) -> World:
    return World(tmp_path, working_diff=True)


def _applied_ids(report: dict) -> List[str]:
    """The first expected id of each batch, in dispatch order (the marker's id)."""
    return [task["expected_ids"][0] for task in report["tasks"]]


# --------------------------------------------------------------------------- #
# 1. improve-plan in one invocation
# --------------------------------------------------------------------------- #

class TestImprovePlanInOneInvocation:

    @pytest.mark.parametrize("typed_yes", [True, False], ids=["--yes typed", "--yes implied"])
    def test_review_then_apply_then_stop(self, world, typed_yes):
        flags = ["--models", MODEL, "--exec-subagents"] + (["--yes"] if typed_yes else [])
        run = world.run("improve-plan", *flags)
        assert run.exit_code == 0, run.explain()

        # -- the record: exactly the two phases, and nothing after them --------
        state = world.state()
        assert set(state["phases_completed"]) == {"review-plan", "apply-suggestions"}, state
        assert not (set(state.get("phases_skipped") or {}) & set(LATER_THAN_PLAN))
        # the composite's own record (display only; it also names the advisory
        # consolidation step, which had nothing to do on a four-group review)
        record = state["improve_plan"]
        assert record["current_phase"] == "complete" and record["failed_phase"] is None
        assert {k: record["steps"][k] for k in ("review-plan", "apply-suggestions")} == {
            "review-plan": "done", "apply-suggestions": "done"}, record
        leaked = [name for name in LATER_THAN_PLAN + ("tasks", "full")
                  if (world.out / name).exists()]
        assert not leaked, f"a composite reached past apply-suggestions: {leaked}"

        # -- the flags each REAL phase was handed -------------------------------
        steps = run.step_flags()
        assert list(steps) == ["review-plan", "apply-suggestions"], steps
        review, apply = steps["review-plan"], steps["apply-suggestions"]
        assert "--yes" in review and review[review.index("--models") + 1] == MODEL
        assert "--force" not in review and "--rerun-all" not in review
        assert apply[:3] == ["--yes", "--agent-decide", "--no-confirm"], apply
        for flag in ("--resume", "--force", "--fresh"):
            assert flag not in apply, (flag, apply)

        # -- the model and subagent traffic: one review, then writes -----------
        assert len(world.calls_of("review")) == 1, "one reviewer, reviewed once"
        assert world.calls_of("validate"), "the review's findings were validated"
        assert not world.calls_of("other"), [c["prompt"][:60] for c in world.calls_of("other")]

        # -- the apply batch really moved the plan's bytes ---------------------
        report = world.report("apply-suggestions")
        assert report["exit"] == 0 and report["finalized"] is True
        assert report["stop"] is None and report["tasks"]
        for task in report["tasks"]:
            assert task["status"] == "ok", task
            assert task["write_outcome"] == "applied", task
            assert task["files_touched"] == [PLAN_REL], task
            assert task["returned_ids"] == task["expected_ids"], task
        ids = _applied_ids(report)
        assert len(world.calls_of("write")) == len(ids)
        markers = [PLAN_MARKER.format(id=item) for item in ids]
        assert world.plan_bytes == world.original_plan + "".join(markers).encode("utf-8"), (
            "each batch appended its marker once, in dispatch order, to the original plan"
        )
        # the plan as it was before the apply is kept, byte for byte, on both legs
        assert (world.out / "apply-suggestions" / "backup.md").read_bytes() == world.original_plan

        # -- the result line ----------------------------------------------------
        result = run.result
        assert result is not None, run.explain()
        assert (result["mode"], result["phase"], result["state"], result["exit"]) == (
            "improve-plan", "improve-plan", "done", 0)
        assert result["next"] is None
        # the next step answers as the row's last phase: what follows apply-suggestions
        step = result["user_next_step"]
        assert step["phase"] == "generate-tasks" and step["complete"] is False, step
        assert step["command"].endswith(f"--generate-tasks {PLAN_REL}"), step


# --------------------------------------------------------------------------- #
# 2. Nothing to do: exit 2, before anything is spent
# --------------------------------------------------------------------------- #

class TestNothingToDo:

    def test_a_finished_improve_plan_is_already_complete(self, world):
        first = world.composite("improve-plan")
        assert first.exit_code == 0, first.explain()
        assert "[ALREADY_COMPLETE]" not in first.stdout
        calls_before = len(world.calls())
        plan_before = world.plan_bytes
        state_before = world.state()

        again = world.composite("improve-plan")

        assert again.exit_code == 2, again.explain()
        assert "[ALREADY_COMPLETE] improve-plan:" in again.stdout, again.explain()
        # no provider was called: not the reviewer, not a validator, not a writer
        assert calls_before >= 3, "the first run made the calls the re-run is compared with"
        assert len(world.calls()) == calls_before
        assert world.plan_bytes == plan_before
        assert world.state()["phases_completed"] == state_before["phases_completed"]
        assert "$ uv run" not in again.stdout, "a phase was started"

        result = again.result
        assert result is not None, again.explain()
        assert (result["mode"], result["exit"], result["state"]) == (
            "improve-plan", 2, "nothing-to-do")
        assert "[ALREADY_COMPLETE] improve-plan:" in result["notice"]
        assert result["next"] is None
        # exit 2 carries the next step too: a re-run is when the user needs it most
        step = result["user_next_step"]
        assert step["phase"] == "generate-tasks", step
        assert step["command"].endswith(f"--generate-tasks {PLAN_REL}"), step

    def test_the_detach_rerun_of_a_finished_composite_launches_no_worker(self, world):
        """The tool routes append ``--detach`` to every call, a re-run included."""
        assert world.composite("improve-plan").exit_code == 0
        run_dir = world.out / "improve-plan"
        listing_before = sorted(path.name for path in run_dir.iterdir())
        calls_before = len(world.calls())

        again = world.composite("improve-plan", "--detach")

        assert again.exit_code == 2, again.explain()
        assert "[ALREADY_COMPLETE] improve-plan:" in again.stdout
        assert "[DETACHED_RUN]" not in again.stdout
        assert not list(world.repo.rglob("orchestrator-run*.log")), "a worker's log exists"
        assert sorted(path.name for path in run_dir.iterdir()) == listing_before, (
            "the run dir was claimed by a launch that should never have happened")
        assert len(world.calls()) == calls_before
        assert again.result["user_next_step"]["phase"] == "generate-tasks"

    def test_a_finished_improve_code_is_already_complete(self, code_world):
        world = code_world
        assert world.composite("improve-code").exit_code == 0
        calls_before = len(world.calls())
        sources_before = {rel: (world.repo / rel).read_bytes() for rel in WORKING_DIFF}

        again = world.composite("improve-code")

        assert again.exit_code == 2, again.explain()
        assert "[ALREADY_COMPLETE] improve-code:" in again.stdout
        assert len(world.calls()) == calls_before
        assert {rel: (world.repo / rel).read_bytes() for rel in WORKING_DIFF} == sources_before
        result = again.result
        assert (result["mode"], result["exit"], result["state"]) == (
            "improve-code", 2, "nothing-to-do")
        # the last workflow phase: the answer on exit 2 is "complete", not absent
        assert result["user_next_step"]["complete"] is True, result["user_next_step"]


# --------------------------------------------------------------------------- #
# 3. improve-code on an uncommitted change
# --------------------------------------------------------------------------- #

class TestImproveCode:

    def test_review_code_then_apply_code_fixes(self, code_world):
        world = code_world
        run = world.composite("improve-code")
        assert run.exit_code == 0, run.explain()

        # -- the record ---------------------------------------------------------
        state = world.state()
        assert set(state["phases_completed"]) == {"code-review", "apply-code-fixes"}, state
        assert not (set(state.get("phases_skipped") or {}) & set(LATER_THAN_CODE))
        record = state["improve_code"]
        assert record["current_phase"] == "complete" and record["failed_phase"] is None
        assert record["steps"] == {"code-review": "done", "apply-code-fixes": "done"}, record
        leaked = [name for name in ("review-plan", "apply-suggestions", "generate-tasks",
                                    "implement", "tasks", "full")
                  if (world.out / name).exists()]
        assert not leaked, f"a composite reached outside its two phases: {leaked}"

        # -- the flags each real phase was handed -------------------------------
        steps = run.step_flags()
        assert list(steps) == ["review-code", "apply-code-fixes"], steps
        assert "--yes" in steps["review-code"] and "--force" not in steps["review-code"]
        assert steps["apply-code-fixes"][:3] == ["--yes", "--agent-decide", "--no-confirm"]
        for flag in ("--resume", "--force", "--fresh"):
            assert flag not in steps["apply-code-fixes"], steps

        # -- the review saw the uncommitted change ------------------------------
        reviews = world.calls_of("review")
        assert len(reviews) == 1
        for rel in WORKING_DIFF:
            assert rel in reviews[0]["prompt"], f"{rel} was not in the reviewer's scope"
        assert world.calls_of("validate"), "the issues were validated"
        assert not world.calls_of("other")

        # -- the fix batches edited the source files ---------------------------
        report = world.report("apply-fixes")
        assert report["exit"] == 0 and report["finalized"] is True and report["stop"] is None
        assert report["tasks"]
        expected_text = {rel: world.sources_before[rel].decode("utf-8") for rel in WORKING_DIFF}
        for task in report["tasks"]:
            assert task["status"] == "ok" and task["write_outcome"] == "applied", task
            assert task["returned_ids"] == task["expected_ids"], task
            assert len(task["files_touched"]) == 1, task
            expected_text[task["files_touched"][0]] += CODE_MARKER.format(id=task["expected_ids"][0])
        assert len(world.calls_of("write")) == len(report["tasks"])
        edited = {rel for task in report["tasks"] for rel in task["files_touched"]}
        assert edited == set(WORKING_DIFF), "the review's findings name both changed files"
        for rel, text in expected_text.items():
            # Decoded bytes, not read_text: that folds a Windows `\r\n` away.
            assert (world.repo / rel).read_bytes().decode("utf-8") == text, (
                f"{rel}: one marker per batch that touched it, appended to the reviewed bytes")

        # -- the result line ----------------------------------------------------
        result = run.result
        assert (result["mode"], result["phase"], result["state"], result["exit"]) == (
            "improve-code", "improve-code", "done", 0)
        assert result["next"] is None
        # apply-code-fixes is the workflow's last phase: nothing follows it
        assert result["user_next_step"]["complete"] is True, result["user_next_step"]
        assert result["user_next_step"]["command"] is None
        assert "[WORKFLOW_COMPLETE]" in run.stdout


# --------------------------------------------------------------------------- #
# 4. --interactive is refused before anything runs
# --------------------------------------------------------------------------- #

class TestInteractiveIsRefused:

    @pytest.mark.parametrize("flag", ["--interactive", "-i"])
    @pytest.mark.parametrize("mode, first, second", [
        ("improve-plan", "review-plan", "apply-suggestions"),
        ("improve-code", "review-code", "apply-code-fixes"),
    ])
    def test_exit_three_with_nothing_run(self, world, mode, first, second, flag):
        run = world.run(mode, "--models", MODEL, "--exec-subagents", flag)

        assert run.exit_code == 3, run.explain()
        assert run.stderr.strip() == (
            f"ERROR: {mode} runs unattended and does not take --interactive (-i). "
            f"Run --{first} --interactive, then --{second}."), run.stderr
        assert run.result is not None
        assert (run.result["mode"], run.result["exit"], run.result["state"]) == (
            mode, 3, "needs-human")
        assert not world.calls(), "a provider was called"
        assert not world.out.exists(), "the refusal wrote into the plan's output folder"
        assert world.plan_bytes == world.original_plan
        assert "$ uv run" not in run.stdout, "a phase was started"


# --------------------------------------------------------------------------- #
# 5. --force: a new round
# --------------------------------------------------------------------------- #

#: A second round that finds different things, so the apply has new batches.
ROUND_TWO_FINDINGS = [
    {"title": "Specify audit log retention",
     "desc": "The plan lists audit logging as a goal but never says how long entries "
             "are kept; state a retention period and where it is configured.",
     "type": "addition", "reference": "## Goals", "importance": "MEDIUM"},
    {"title": "Name the password hashing algorithm",
     "desc": "Say which algorithm and cost factor hash passwords, so the registration "
             "step and the login step agree.",
     "type": "clarification", "reference": "## Implementation Steps", "importance": "HIGH"},
]


class TestForceIsANewRound:

    def test_a_forced_improve_plan_reviews_again_and_applies_the_new_findings(self, world):
        first = world.composite("improve-plan")
        assert first.exit_code == 0, first.explain()
        round_one_ids = {i for task in world.report("apply-suggestions")["tasks"]
                         for i in task["expected_ids"]}
        after_round_one = world.plan_bytes
        assert len(world.calls_of("review")) == 1

        findings = world.tmp_path / "round-two-findings.json"
        findings.write_text(json.dumps(ROUND_TWO_FINDINGS), encoding="utf-8")
        world.set_review_answers(plan_review=findings)
        second = world.composite("improve-plan", "--force")
        assert second.exit_code == 0, second.explain()

        # the review ran again, for every model, and the apply started a new cycle
        assert len(world.calls_of("review")) == 2
        steps = second.step_flags()
        assert list(steps) == ["review-plan", "apply-suggestions"], steps
        review = steps["review-plan"]
        assert review.count("--force") == 1 and review.count("--rerun-all") == 1, review
        for flag in ("--resume", "--force", "--fresh"):
            assert flag not in steps["apply-suggestions"], steps["apply-suggestions"]

        # it applied THIS round's findings, to the plan as round one left it
        report = world.report("apply-suggestions")
        round_two_ids = [task["expected_ids"][0] for task in report["tasks"]]
        assert len(round_two_ids) == len(ROUND_TWO_FINDINGS)
        assert not set(round_two_ids) & round_one_ids
        assert all(task["write_outcome"] == "applied" for task in report["tasks"])
        assert world.plan_bytes == after_round_one + "".join(
            PLAN_MARKER.format(id=item) for item in round_two_ids).encode("utf-8")

        # backup.md is the plan as this round's apply found it; round one's is kept aside
        apply_dir = world.out / "apply-suggestions"
        assert (apply_dir / "backup.md").read_bytes() == after_round_one
        import hashlib
        aside = apply_dir / f"backup.{hashlib.sha256(world.original_plan).hexdigest()[:8]}.md"
        assert aside.read_bytes() == world.original_plan, sorted(p.name for p in apply_dir.iterdir())

        # and the new round is itself finished
        assert world.composite("improve-plan").exit_code == 2

    @staticmethod
    def _round_one_then_an_identical_forced_round(tmp_path, mode):
        """A finished composite, then ``--force`` over a review that finds the same things."""
        world = World(tmp_path, working_diff=(mode == "improve-code"))
        assert world.composite(mode).exit_code == 0
        touched = list(WORKING_DIFF) + [PLAN_REL]
        bytes_after_round_one = {rel: (world.repo / rel).read_bytes() for rel in touched}
        writes_before = len(world.calls_of("write"))
        assert len(world.calls_of("review")) == 1
        second = world.composite(mode, "--force")
        return world, second, bytes_after_round_one, writes_before

    @pytest.mark.parametrize("mode", ["improve-plan", "improve-code"])
    def test_a_forced_round_that_finds_the_same_things_applies_nothing_twice(
        self, tmp_path, mode
    ):
        """Plan §2.5, *A review that re-reports a batch byte for byte*: the review
        runs again, every batch it finds is one that already landed, and none is
        dispatched a second time."""
        world, second, bytes_after_round_one, writes_before = (
            self._round_one_then_an_identical_forced_round(tmp_path, mode))

        assert second.exit_code == 0, second.explain()
        assert len(world.calls_of("review")) == 2, "the review was not run again"
        steps = second.step_flags()
        review = steps["review-plan" if mode == "improve-plan" else "review-code"]
        assert review.count("--force") == 1 and review.count("--rerun-all") == 1, review
        assert len(world.calls_of("write")) == writes_before, "a landed batch was dispatched again"
        assert {rel: (world.repo / rel).read_bytes() for rel in bytes_after_round_one} == (
            bytes_after_round_one), "an edit was applied twice"
        assert second.result["state"] == "done"

    @pytest.mark.parametrize("mode", ["improve-plan", "improve-code"])
    def test_the_next_rerun_of_such_a_round_is_already_complete(self, tmp_path, mode):
        """Plan §2.5: the apply that kept every batch "records itself completed with
        a stamp newer than the review and exits 0, and the next re-run exits 2"."""
        world, second, _, _ = self._round_one_then_an_identical_forced_round(tmp_path, mode)
        assert second.exit_code == 0, second.explain()
        review_phase, apply_phase = (
            ("review-plan", "apply-suggestions") if mode == "improve-plan"
            else ("code-review", "apply-code-fixes"))
        stamps = world.state()["phases_completed"]
        assert stamps[apply_phase] > stamps[review_phase], (
            "the apply phase was recorded completed for the OLD review only", stamps)
        again = world.composite(mode)
        assert again.exit_code == 2, again.explain()
        assert f"[ALREADY_COMPLETE] {mode}:" in again.stdout


# --------------------------------------------------------------------------- #
# 6. Stopped mid-apply, re-run with no flag
# --------------------------------------------------------------------------- #

class TestAnInterruptedRunContinues:

    def test_a_bare_rerun_applies_only_the_remaining_batches(self, world):
        # The third write dispatch fails the way a runner that tried and could not does.
        stopped = world.composite("improve-plan", env=world.env(IMPROVE_E2E_FAIL_CALL="3"))
        assert stopped.exit_code == 3, stopped.explain()
        assert (stopped.result["state"], stopped.result["exit"]) == ("needs-human", 3)
        assert "--resume" in stopped.result["next"] and "improve-plan" in stopped.result["next"]
        assert "apply-batch-2" in stopped.result["error"], stopped.result["error"]
        assert "user_next_step" not in stopped.result, "a stop owes no next phase"

        # review recorded, apply not; two batches landed
        state = world.state()
        assert set(state["phases_completed"]) == {"review-plan"}, state["phases_completed"]
        assert state["improve_plan"]["failed_phase"] == "apply-suggestions"
        report = world.report("apply-suggestions")
        assert [task["status"] for task in report["tasks"]] == ["ok", "ok", "batch_failed"]
        assert report["stopped_at_batch"] == 2
        ids = [task["expected_ids"][0] for task in report["tasks"]]
        landed = world.plan_bytes
        assert landed == world.original_plan + "".join(
            PLAN_MARKER.format(id=item) for item in ids[:2]).encode("utf-8")
        reviews, validations = len(world.calls_of("review")), len(world.calls_of("validate"))
        assert (reviews, len(world.calls_of("write"))) == (1, 3)

        # the same command again, no --resume typed
        resumed = world.composite("improve-plan")
        assert resumed.exit_code == 0, resumed.explain()

        steps = resumed.step_flags()
        assert list(steps) == ["apply-suggestions"], "the review was run again"
        assert steps["apply-suggestions"].count("--resume") == 1, steps
        assert "--force" not in steps["apply-suggestions"]
        # the review and its validation were not paid for twice
        assert len(world.calls_of("review")) == reviews
        assert len(world.calls_of("validate")) == validations
        final = world.report("apply-suggestions")
        batches = len(final["tasks"])
        assert batches >= 4, "the fixture's four groups are four batches; the stop needs one after it"
        # exactly the failed batch and the ones never reached were dispatched
        assert len(world.calls_of("write")) == 3 + (batches - 2)
        assert [task["write_outcome"] for task in final["tasks"]] == ["applied"] * batches
        all_ids = [task["expected_ids"][0] for task in final["tasks"]]
        assert all_ids[:3] == ids
        assert world.plan_bytes == world.original_plan + "".join(
            PLAN_MARKER.format(id=item) for item in all_ids).encode("utf-8"), (
            "every batch landed exactly once")
        assert set(world.state()["phases_completed"]) == {"review-plan", "apply-suggestions"}
        assert resumed.result["user_next_step"]["phase"] == "generate-tasks"


# --------------------------------------------------------------------------- #
# 7. A review already on record is not run again
# --------------------------------------------------------------------------- #

class TestAReviewOnRecord:

    def test_improve_plan_after_a_standalone_review_runs_only_the_apply(self, world):
        review = world.run("review-plan", "--models", MODEL, "--yes", "--exec-subagents")
        assert review.exit_code == 0, review.explain()
        assert set(world.state()["phases_completed"]) == {"review-plan"}
        calls_before = len(world.calls())
        assert not world.calls_of("write")

        run = world.composite("improve-plan")

        assert run.exit_code == 0, run.explain()
        assert list(run.step_flags()) == ["apply-suggestions"], run.step_flags()
        assert "--resume" not in run.step_flags()["apply-suggestions"], (
            "nothing of the apply is in progress, so nothing is resumed")
        new_calls = world.calls()[calls_before:]
        assert new_calls and {classify(call) for call in new_calls} == {"write"}
        assert set(world.state()["phases_completed"]) == {"review-plan", "apply-suggestions"}
        report = world.report("apply-suggestions")
        assert world.plan_bytes == world.original_plan + "".join(
            PLAN_MARKER.format(id=task["expected_ids"][0]) for task in report["tasks"]
        ).encode("utf-8")


# --------------------------------------------------------------------------- #
# 8. An interrupted improve-code keeps the review's diff base
# --------------------------------------------------------------------------- #

class TestAnInterruptedImproveCode:

    def test_the_resumed_apply_diffs_against_the_base_the_review_used(self, code_world):
        world = code_world
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=world.repo, capture_output=True, text=True,
            check=True).stdout.strip()

        stopped = world.composite("improve-code", env=world.env(IMPROVE_E2E_FAIL_CALL="2"))
        assert stopped.exit_code == 3, stopped.explain()
        assert "--resume" in stopped.result["next"] and "improve-code" in stopped.result["next"]
        assert set(world.state()["phases_completed"]) == {"code-review"}
        report = world.report("apply-fixes")
        assert [task["status"] for task in report["tasks"]] == ["ok", "batch_failed"]
        reviews = len(world.calls_of("review"))

        # a new commit moves HEAD: "the base the review used" is no longer HEAD
        (world.repo / "NOTES.md").write_text("notes\n", encoding="utf-8")
        _git(world.repo, "add", "NOTES.md")
        _git(world.repo, "commit", "-q", "-m", "notes")

        resumed = world.composite("improve-code")
        assert resumed.exit_code == 0, resumed.explain()
        steps = resumed.step_flags()
        assert list(steps) == ["apply-code-fixes"], "the review ran again"
        apply = steps["apply-code-fixes"]
        assert apply.count("--resume") == 1
        assert apply[apply.index("--base-ref") + 1] == head, apply
        assert len(world.calls_of("review")) == reviews
        assert set(world.state()["phases_completed"]) == {"code-review", "apply-code-fixes"}
        final = world.report("apply-fixes")
        assert [task["write_outcome"] for task in final["tasks"]] == ["applied"] * 3
        expected = {rel: world.sources_before[rel].decode("utf-8") for rel in WORKING_DIFF}
        for task in final["tasks"]:
            expected[task["files_touched"][0]] += CODE_MARKER.format(id=task["expected_ids"][0])
        for rel, text in expected.items():
            assert (world.repo / rel).read_bytes().decode("utf-8") == text, rel


# --------------------------------------------------------------------------- #
# 8b. A review that fails stops the composite before its apply
# --------------------------------------------------------------------------- #

class TestAReviewThatFails:

    def test_no_apply_runs_and_the_same_command_again_goes_on_from_the_review(self, world):
        # every mock invocation fails; only the reviewer is ever invoked
        failed = world.composite("improve-plan", env=world.env(MOCK_LLM_FAIL="1"))

        assert failed.exit_code == 3, failed.explain()
        assert failed.result["state"] == "needs-human"
        assert "--resume" in failed.result["next"] and "improve-plan" in failed.result["next"]
        assert "review-plan did not complete" in failed.result["error"]
        assert list(failed.step_flags()) == ["review-plan"], "the apply must not start"
        assert not world.state().get("phases_completed")
        assert world.state()["improve_plan"]["failed_phase"] == "review-plan"
        assert not (world.out / "apply-suggestions").exists()
        assert world.plan_bytes == world.original_plan
        assert [classify(call) for call in world.calls()] == ["review"]

        again = world.composite("improve-plan")

        assert again.exit_code == 0, again.explain()
        assert list(again.step_flags()) == ["review-plan", "apply-suggestions"]
        assert set(world.state()["phases_completed"]) == {"review-plan", "apply-suggestions"}
        assert world.state()["improve_plan"]["failed_phase"] is None


# --------------------------------------------------------------------------- #
# 8c. A different --files scope is a fresh review
# --------------------------------------------------------------------------- #

class TestARescopedImproveCode:

    def test_a_typed_files_scope_that_differs_from_the_recorded_one_reviews_again(
        self, code_world
    ):
        world = code_world
        assert world.composite("improve-code").exit_code == 0
        first_review = world.calls_of("review")[0]["prompt"]
        assert "src/security.py" in first_review and "src/auth.py" in first_review

        run = world.composite("improve-code", "--files", "src/auth.py")

        assert run.exit_code == 0, run.explain()
        assert "[IMPROVE_SCOPE_CHANGED] improve-code:" in run.stdout
        review = run.step_flags()["review-code"]
        assert review.count("--force") == 1 and "--rerun-all" not in review, review
        assert review[review.index("--files") + 1] == "src/auth.py"
        reviews = world.calls_of("review")
        assert len(reviews) == 2, "the reviewer was asked about the new scope"
        assert "src/auth.py" in reviews[1]["prompt"]
        manifest = (world.out / "code-review" / "review-scope.md").read_text(encoding="utf-8")
        assert "`src/auth.py`" in manifest and "`src/security.py`" not in manifest, manifest
        assert world.state()["review_scope_files"] == ["src/auth.py"]


# --------------------------------------------------------------------------- #
# 8d. Detached, the way every tool route runs it
# --------------------------------------------------------------------------- #

def _stub_uv(tmp_path: Path) -> Path:
    """A ``uv`` that runs ``uv run <flags> --project <dir> -- python <args>`` with this
    interpreter, so the detached worker re-executes the dispatcher with no ``uv``
    resolution (and no network) in the way. Everything else about the worker is real:
    the double fork, the phase lock, the heartbeat, the log it owns."""
    path = tmp_path / "stub-uv"
    path.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done\n'
        "shift; shift\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


class TestADetachedImprovePlan:

    def test_the_worker_runs_both_phases_and_the_terminal_poll_names_the_next_step(
        self, world
    ):
        from utils import status_file

        env = world.env(MULTI_LLM_UV=str(_stub_uv(world.tmp_path)))
        launch = world.composite("improve-plan", "--detach", env=env)

        assert launch.exit_code == 0, launch.explain()
        assert "run_id:" in launch.stdout, launch.explain()
        # the launch owes the poll, not the next phase
        assert "status" in launch.result["next"] and "--phase improve-plan" in launch.result["next"]
        assert "user_next_step" not in launch.result

        run_dir = world.out / "improve-plan"
        deadline = time.monotonic() + 60 * PERF_SCALE
        run: dict = {}
        while time.monotonic() < deadline:
            run = status_file.read_run(str(run_dir))
            if run.get("state") in status_file.TERMINAL_RUN_STATES:
                break
            time.sleep(0.2)
        assert run.get("state") == "done", (run, (run_dir / "orchestrator-run.log").read_text()[-2000:])
        assert run.get("exit_code") == 0 and run.get("mode") == "improve-plan", run

        log = (run_dir / "orchestrator-run.log").read_text(encoding="utf-8")
        assert "=== Step 1: review-plan ===" in log and "=== Step 2: apply-suggestions ===" in log
        assert set(world.state()["phases_completed"]) == {"review-plan", "apply-suggestions"}
        report = world.report("apply-suggestions")
        assert world.plan_bytes == world.original_plan + "".join(
            PLAN_MARKER.format(id=task["expected_ids"][0]) for task in report["tasks"]
        ).encode("utf-8")

        # acceptance criterion 2: the next command arrives on the terminal poll
        poll = world.run("status", "--view", "run", "--phase", "improve-plan", env=env)
        assert poll.exit_code == 0, poll.explain()
        assert poll.result["mode"] == "improve-plan" and poll.result["state"] == "done"
        step = poll.result["user_next_step"]
        assert step["phase"] == "generate-tasks", step
        assert step["command"].endswith(f"--generate-tasks {PLAN_REL}"), step


# --------------------------------------------------------------------------- #
# 9. Nothing to apply is still a finished run
# --------------------------------------------------------------------------- #

EMPTY_PLAN_REVIEW = RESPONSES / "review_plan" / "empty.json"
EMPTY_CODE_REVIEW = RESPONSES / "code_review" / "empty_results.json"


class TestAReviewThatLeavesNothingToApply:
    """Acceptance criterion 4, on the executor leg and through the real phases:
    zero findings, or every finding invalid, records the apply completed, exits 0
    (never 3 on a missing ``validation.json``) and a re-run exits 2."""

    @pytest.mark.parametrize("cause", ["no findings", "every finding invalid"])
    @pytest.mark.parametrize("mode", ["improve-plan", "improve-code"])
    def test_the_apply_is_recorded_completed_and_a_rerun_is_already_complete(
        self, tmp_path, mode, cause
    ):
        world = World(tmp_path, working_diff=(mode == "improve-code"))
        env = None
        if cause == "no findings":
            world.set_review_answers(plan_review=EMPTY_PLAN_REVIEW, code_review=EMPTY_CODE_REVIEW)
        else:
            env = world.env(IMPROVE_E2E_VERDICT="invalid")
        review_phase, apply_phase = (
            ("review-plan", "apply-suggestions") if mode == "improve-plan"
            else ("code-review", "apply-code-fixes"))

        run = world.composite(mode, env=env)

        assert run.exit_code == 0, run.explain()
        assert set(world.state()["phases_completed"]) == {review_phase, apply_phase}
        stamps = world.state()["phases_completed"]
        assert stamps[apply_phase] > stamps[review_phase], stamps
        assert not world.calls_of("write"), "there was nothing to write"
        assert world.plan_bytes == world.original_plan
        assert {rel: (world.repo / rel).read_bytes() for rel in WORKING_DIFF} == world.sources_before
        assert (run.result["mode"], run.result["state"], run.result["exit"]) == (mode, "done", 0)
        if cause == "no findings":
            assert not world.calls_of("validate")

        calls = len(world.calls())
        again = world.composite(mode)
        assert again.exit_code == 2, again.explain()
        assert f"[ALREADY_COMPLETE] {mode}:" in again.stdout
        assert len(world.calls()) == calls


# --------------------------------------------------------------------------- #
# 10. Nobody is asked: the judge decides what validation could not settle
# --------------------------------------------------------------------------- #

class TestTheJudgeDecides:

    def test_needs_human_decision_items_are_judged_then_applied_unattended(self, world):
        run = world.composite(
            "improve-plan",
            env=world.env(IMPROVE_E2E_VERDICT="needs-human-decision", IMPROVE_E2E_JUDGE="approve"),
        )
        assert run.exit_code == 0, run.explain()

        undecided = run.result["validation_counts"]["needs_human_decision"]
        assert undecided >= 1, run.result["validation_counts"]
        judged = world.calls_of("judge")
        assert len(judged) == undecided, "each open item reached the judge agent"
        apply_dir = world.out / "apply-suggestions"
        assert (apply_dir / "subagent_batch_report_human_decision_judge.json").is_file()

        # the judge approved them, so every group was applied, none left open
        report = world.report("apply-suggestions")
        assert len(report["tasks"]) == undecided + run.result["validation_counts"]["valid"]
        assert all(task["write_outcome"] == "applied" for task in report["tasks"])
        assert world.plan_bytes == world.original_plan + "".join(
            PLAN_MARKER.format(id=task["expected_ids"][0]) for task in report["tasks"]
        ).encode("utf-8")
        # the run's summary says who decided
        assert "judge" in (apply_dir / "summary.md").read_text(encoding="utf-8").lower()
        assert "Decided by the judge agent" in run.stdout
