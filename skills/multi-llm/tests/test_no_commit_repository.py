#!/usr/bin/env python3
"""Every mode the ``multi_llm`` tool can reach works in a git repository with no commit.

**The failure this pins.** A first-time user's project is a plain ``git init``
with no commit (plans/opencode-plugin-publish.md, step 5), and the Codex network
rehearsal found that the shared ``status`` raised ``GitError: git rev-parse HEAD
failed`` there. It was not ``status``'s alone: a new ``state.json`` recorded
``head_at_start`` through ``git_utils.get_current_head``, so every mode that opens
the state — ``status``, every review, every composite — exited 1 with a
traceback before it did anything (measured on this tree before the fix:
``status``, ``review-plan``, ``improve-plan``, ``review-code``, ``improve-code``
and ``full`` exit 1; the other phases only got as far as their prerequisite
refusals, because the phase before them could not run). The decision
(plans/opencode-plugin-publish.md, "Decided: fix it in the shared runtime code")
is to fix the shared runtime, never to document an initial-commit prerequisite
and never to make a fixture pass by committing: **nothing in this module makes
a commit in a repository a mode runs in**, and every end-to-end world asserts
afterwards that HEAD is still unborn.

**What "against HEAD" means before the first commit.** The empty tree: nothing
is committed, so every file is new. ``git_utils.head_or_empty_tree`` is what the
state records as ``head_at_start`` and ``head_before_implement``;
``validate_git_ref`` falls back to it rather than to ``HEAD~1`` and then nothing;
``paths_in_commit`` reads a tree, so the empty one holds no path and an
implement-created file is ``created`` rather than unknown; the review-scope
manifest names the base as the empty tree. No mode needs history it cannot
have, so none ends at exit 3 for the lack of a commit.

**How the modes are driven.** Exactly as the OpenCode plugin's ``execute``
composes them (``packaging/opencode/src/index.mjs``, ``composeDispatcherArgv``):
the mode word, its positionals, the surface's ``baked_flags``, ``--detach`` for
every ``supports_detach`` row, then the surface's own poll argv (``status --view
run --wait --run-id``) until the run is terminal — all read from the generated
CLI surface (``packaging/common/surface.py``), so the argv here cannot drift from
the tool's. Model calls go to the suite's mock CLIs (``tests/mocks/mock_llm.py``)
on ``PATH``, through a throwaway mock plugin that answers each prompt kind and
makes each write batch's edit in the file the prompt names; nothing spends and
nothing leaves the machine.

**Parametrized over the tool's mode table.** ``TOOL_MODES`` is the surface's
exposed rows; a mode added there without a recipe in ``RECIPES`` fails
``test_every_mode_the_tool_exposes_has_a_recipe`` and its own parametrized row,
naming itself, rather than going untested.

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_no_commit_repository.py -v
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TESTS_DIR.parent
REPO_ROOT = SKILL_DIR.parent.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(SKILL_DIR))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

import finalize_tracking  # noqa: E402
import modes  # noqa: E402
from common import surface as cli_surface  # noqa: E402
from harness import PERF_SCALE, SkillRunner  # noqa: E402
from utils import detach, status_file  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.git_utils import (  # noqa: E402
    EMPTY_TREE_SHA1,
    GitError,
    empty_tree_id,
    get_current_head,
    head_is_unborn,
    head_or_empty_tree,
    is_empty_tree,
    load_review_scope,
    paths_in_commit,
    validate_git_ref,
    write_review_diff,
)
from utils.output_handler import PHASE_DIRECTORIES  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_e2e_improve_modes import (  # noqa: E402
    COMMITTED as SOURCES,
    MOCK_LLM,
    MODEL,
    PLAN_FIXTURE,
    PLAN_REL,
    RESPONSES,
    RUNNER,
    _stub_uv,
)

#: The host whose tool route this module drives. The fix is in shared code, so
#: one route is the whole of the coverage; OpenCode is the route the release
#: plan ships to first-time users.
TOOL_HOST = "opencode"
SURFACE = cli_surface.cli_surface(TOOL_HOST)
TOOL_ROWS: Dict[str, dict] = {row["name"]: row for row in SURFACE["modes"]}
TOOL_MODES: Tuple[str, ...] = tuple(TOOL_ROWS)

OUT_REL = PLAN_REL[: -len(".md")]
QUESTION = "Is the plan complete?"

GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, **GIT_IDENTITY, "GIT_TERMINAL_PROMPT": "0"},
    )


def _unborn_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    return root


def _has_commit(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", "HEAD", check=False).returncode == 0


# --------------------------------------------------------------------------- #
# 1. The unborn-HEAD helpers
# --------------------------------------------------------------------------- #

class TestUnbornHead:
    def test_a_fresh_init_is_unborn_and_its_base_is_the_empty_tree(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        cwd = str(repo)
        assert head_is_unborn(cwd)
        assert head_or_empty_tree(cwd) == EMPTY_TREE_SHA1
        assert empty_tree_id(cwd) == EMPTY_TREE_SHA1
        # git resolves the empty tree without having stored it.
        assert _git(repo, "rev-parse", "--verify", EMPTY_TREE_SHA1).returncode == 0

    def test_get_current_head_stays_strict(self, tmp_path, monkeypatch):
        repo = _unborn_repo(tmp_path / "repo")
        monkeypatch.chdir(repo)
        with pytest.raises(GitError):
            get_current_head()

    def test_a_repository_with_a_commit_is_not_unborn(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "one")
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert not head_is_unborn(str(repo))
        assert head_or_empty_tree(str(repo)) == sha

    def test_an_orphan_branch_is_unborn_although_the_repository_has_history(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "one")
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

        _git(repo, "checkout", "-q", "--orphan", "fresh")
        assert head_is_unborn(str(repo))
        assert head_or_empty_tree(str(repo)) == EMPTY_TREE_SHA1
        # The history is still there, it is just not HEAD's.
        assert _git(repo, "rev-parse", "--verify", "main").stdout.strip() == sha

        _git(repo, "checkout", "-q", "main")
        assert not head_is_unborn(str(repo))
        assert head_or_empty_tree(str(repo)) == sha

    def test_a_detached_head_is_not_unborn(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "one")
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "checkout", "-q", "--detach")
        assert not head_is_unborn(str(repo))
        assert head_or_empty_tree(str(repo)) == sha

    def test_outside_a_repository_nothing_is_unborn_and_the_base_still_raises(
        self, tmp_path, monkeypatch,
    ):
        outside = tmp_path / "plain"
        outside.mkdir()
        # Resolved: macOS's temp dir is behind a `/private` symlink.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", os.path.realpath(tmp_path))
        assert not head_is_unborn(str(outside))
        with pytest.raises(GitError):
            head_or_empty_tree(str(outside))

    def test_the_empty_tree_follows_the_object_format(self, tmp_path):
        repo = tmp_path / "sha256"
        repo.mkdir()
        made = _git(repo, "init", "-q", "--object-format=sha256", "-b", "main", check=False)
        if made.returncode != 0:
            pytest.skip(f"this git cannot make a SHA-256 repository: {made.stderr.strip()}")
        tree = empty_tree_id(str(repo))
        assert re.fullmatch(r"[0-9a-f]{64}", tree), tree
        assert head_or_empty_tree(str(repo)) == tree
        assert is_empty_tree(tree, str(repo))
        assert _git(repo, "rev-parse", "--verify", tree).returncode == 0

    def test_is_empty_tree(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        assert is_empty_tree(EMPTY_TREE_SHA1, str(repo))
        assert not is_empty_tree("HEAD", str(repo))
        assert not is_empty_tree("", str(repo))
        assert not is_empty_tree(None, str(repo))


# --------------------------------------------------------------------------- #
# 2. The callers: each diff base, baseline and file action
# --------------------------------------------------------------------------- #

class TestTheDiffBaseBeforeTheFirstCommit:
    @pytest.mark.parametrize("ref", ["HEAD", "HEAD~1", "no-such-ref"])
    def test_validate_git_ref_falls_back_to_the_empty_tree(self, tmp_path, monkeypatch, capsys, ref):
        repo = _unborn_repo(tmp_path / "repo")
        monkeypatch.chdir(repo)
        assert validate_git_ref(ref) == EMPTY_TREE_SHA1
        assert "HEAD has no commit yet" in capsys.readouterr().err

    def test_validate_git_ref_keeps_the_empty_tree_it_was_given(self, tmp_path, monkeypatch, capsys):
        repo = _unborn_repo(tmp_path / "repo")
        monkeypatch.chdir(repo)
        assert validate_git_ref(EMPTY_TREE_SHA1) == EMPTY_TREE_SHA1
        assert capsys.readouterr().err == ""

    def test_validate_git_ref_is_unchanged_once_there_is_history(self, tmp_path, monkeypatch, capsys):
        """One commit: ``HEAD~1`` does not resolve and the base is empty, as
        before. Two: the fallback is ``HEAD~1``. The empty tree is only ever
        the answer before the first commit."""
        repo = _unborn_repo(tmp_path / "repo")
        monkeypatch.chdir(repo)
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "one")
        assert validate_git_ref("no-such-ref") == ""
        (repo / "a.txt").write_text("b\n", encoding="utf-8")
        _git(repo, "commit", "-q", "-am", "two")
        assert validate_git_ref("no-such-ref") == "HEAD~1"
        assert "HEAD has no commit yet" not in capsys.readouterr().err

    def test_paths_in_commit_reads_the_empty_tree_as_holding_nothing(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
        assert paths_in_commit(EMPTY_TREE_SHA1, ["new.py"], cwd=str(repo)) == set()
        # An unborn HEAD is still no tree to ask; the caller picks the base.
        assert paths_in_commit("HEAD", ["new.py"], cwd=str(repo)) is None

    def test_paths_in_commit_is_unchanged_for_a_commit(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
        _git(repo, "add", "a.py")
        _git(repo, "commit", "-q", "-m", "one")
        assert paths_in_commit("HEAD", ["a.py", "b.py"], cwd=str(repo)) == {"a.py"}

    def test_a_new_state_records_the_empty_tree(self, tmp_path, monkeypatch):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "plans").mkdir()
        plan = repo / "plans" / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.chdir(repo)
        state = StateManager(plan)
        assert state.get("head_at_start") == EMPTY_TREE_SHA1
        assert state.get("branch_name") == "main"
        state.save()
        on_disk = json.loads((repo / "plans" / "p" / "state.json").read_text(encoding="utf-8"))
        assert on_disk["head_at_start"] == EMPTY_TREE_SHA1

    def test_a_file_made_before_the_first_commit_is_created(self, tmp_path, monkeypatch):
        """``tracked_files``' action, read against the empty tree: was
        ``modified`` for every path, because there was no tree to ask."""
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.chdir(repo)
        for base in (None, EMPTY_TREE_SHA1):
            assert finalize_tracking._path_states(["new.py"], base_ref=base) == {"new.py": "created"}
            assert finalize_tracking._file_actions(["new.py"], base_ref=base) == {"new.py": "created"}
        # A path the pre-run baseline names itself existed before the phase.
        assert finalize_tracking._path_states(
            ["new.py"], base_ref=EMPTY_TREE_SHA1, baseline=["new.py"],
        ) == {"new.py": "modified"}

    def test_the_review_diff_against_the_empty_tree_is_every_line_added(self, tmp_path):
        repo = _unborn_repo(tmp_path / "repo")
        (repo / "app.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        _git(repo, "add", "app.py")
        out = tmp_path / "code-review"
        result = write_review_diff(EMPTY_TREE_SHA1, ["app.py"], out, cwd=str(repo))
        assert result.coverage == "complete" and result.failure is None, result
        assert result.base_ref == EMPTY_TREE_SHA1 and result.base_commit is None
        text = (out / "review-scope.md").read_text(encoding="utf-8")
        assert f"- Base ref: `{EMPTY_TREE_SHA1}` (the empty tree: the repository had no commit" in text
        assert "`app.py` (+2 -0) — full: `diff/1-app.py.diff`" in text, text
        diff = (out / "diff" / "1-app.py.diff").read_text(encoding="utf-8")
        assert "new file mode" in diff and "+x = 1\n+y = 2\n" in diff
        header = load_review_scope(out)
        assert header is not None and header.base_ref == EMPTY_TREE_SHA1


# --------------------------------------------------------------------------- #
# 3. Every tool-reachable mode, through the tool's own argv
# --------------------------------------------------------------------------- #

#: The throwaway mock plugin (``MOCK_LLM_PLUGIN``, see tests/test_mock_plugin_hook.py).
#: One process serves every phase, so it answers by prompt kind: a validation or
#: task-generation subagent is echoed from the ids its prompt embeds, a judge or
#: consolidation prompt likewise, a write batch (the ``## Result (required``
#: contract) appends one marker line to the file its prompt names and attests to
#: the batch's own ids, and everything else falls through to the scenario file.
PLUGIN_SOURCE = r'''"""Scratch mock plugin for tests/test_no_commit_repository.py."""
import importlib.util
import json
import os
import re
import sys

_spec = importlib.util.spec_from_file_location("no_commit_orig_mock", os.environ["NO_COMMIT_MOCK"])
_orig = importlib.util.module_from_spec(_spec)
sys.modules["no_commit_orig_mock"] = _orig
_spec.loader.exec_module(_orig)

_RESULT_CONTRACT = "## Result (required"
_IDS = re.compile(r"(?:suggestion|fix|task) \d+ = `([^`]+)`|this task's id, `([^`]+)`")
#: Where each write prompt names the file it edits, most specific first:
#: apply-task-suggestions, apply-suggestions, apply-code-fixes, implement.
_TARGETS = (
    r"\*\*Target file \(to edit\)\*\*: (\S+)",
    r"\*\*Plan file\*\*: (\S+)",
    r"\*\*File\*\*: (\S+)",
    r"related issues in `([^`]+)`",
    r"## Output Files Expected\n- (\S+)",
    r"(?m)^- (src/\S+)$",
)


def _die(message):
    print("NO-COMMIT PLUGIN ERROR: " + message, file=sys.stderr)
    sys.exit(9)


def parse_provider_args(provider):
    args, prompt = _orig.parse_provider_args(provider)
    if not prompt:
        prompt = _orig.read_stdin_prompt()
    if _RESULT_CONTRACT not in prompt:
        return args, prompt
    ids = [a or b for a, b in _IDS.findall(prompt)]
    if not ids:
        _die("the write batch spelled no expected ids: " + prompt[:80])
    path = next(
        (m.group(1) for m in (re.search(rx, prompt) for rx in _TARGETS) if m), None
    )
    if path is None:
        _die("no file named in the write batch: " + prompt[:80])
    os.environ["MOCK_LLM_WRITE_IDS"] = json.dumps(ids)
    os.environ["MOCK_LLM_EDITS"] = json.dumps([{
        "path": path, "mode": "patch",
        "content": "\n" + os.environ["NO_COMMIT_MARKER"].replace("{id}", ids[0]) + "\n",
    }])
    return args, prompt


def get_response_content(prompt):
    for kind, marker in (("validation", "You are validating"),
                         ("task_generation", "You are decomposing")):
        if marker in prompt:
            echoed = _orig.echo_subagent_answer(kind, prompt)
            if echoed is not None:
                return echoed
    for kind in ("human_decision_judge", "consolidation"):
        echoed = _orig.echo_subagent_answer(kind, prompt)
        if echoed is not None:
            return echoed
    return _orig.get_response_content(prompt)
'''

#: The line each write batch appends; ``{id}`` is the batch's first expected id.
MARKER = "<!-- no-commit-e2e applied {id} -->"

#: What the one mock reviewer answers, per prompt opener. ``on_unmatched: fail``
#: so a prompt nobody planned for fails loudly instead of getting a generic list.
SCENARIO = {
    "name": "no-commit repository",
    "on_unmatched": "fail",
    "prompts": [
        {"pattern": "You are reviewing an implementation plan for completeness",
         "fixture": str(RESPONSES / "review_plan" / "raw_suggestions.json")},
        {"pattern": "You are reviewing implementation tasks that were generated from a feature plan",
         "fixture": str(RESPONSES / "review_tasks" / "coverage_suggestions.json")},
        {"pattern": "You are reviewing code changes that implement a feature plan",
         "fixture": str(RESPONSES / "code_review" / "container_issues.json")},
        {"pattern": "You are answering a question about an implementation plan",
         "fixture": str(RESPONSES / "ask" / "plan_question.md")},
    ],
}

#: The variables a developer shell may export that would move a host, a route
#: or the repository git acts on away from the one composed here.
_SCRUBBED_ENV = (
    "MULTI_LLM_HOST", "MULTI_LLM_HOST_SESSION_MODEL", "MULTI_LLM_LEGACY_INSTRUCTIONS",
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
    detach.PLAN_ARG_ENV, detach.WORKER_ENV,
)


@dataclass
class Call:
    """One dispatcher subprocess."""

    argv: List[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def result(self) -> Optional[dict]:
        return parse_result_line(self.stdout)

    @property
    def result_lines(self) -> List[str]:
        return [line for line in self.stdout.splitlines() if line.startswith("MULTI_LLM_RESULT:")]

    def banner(self, name: str) -> Optional[str]:
        """A field of the detached-run banner (``detach._print_banner``)."""
        found = re.search(rf"^\s*{name}:\s*(\S.*?)\s*$", self.stdout, re.MULTILINE)
        return found.group(1) if found else None

    @property
    def run_id(self) -> Optional[str]:
        return self.banner(SURFACE["dispatcher"]["banner"]["run_id_field"])

    def explain(self) -> str:
        return (
            f"argv: {' '.join(self.argv)}\nexit: {self.exit_code}\n"
            f"--- stdout (tail) ---\n{self.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{self.stderr[-3000:]}"
        )


@dataclass
class ToolCall:
    """One ``multi_llm`` tool call: the launch, and the polls of a detached run."""

    mode: str
    launch: Call
    polls: List[Call] = field(default_factory=list)

    @property
    def final(self) -> Call:
        """The call whose result is the tool's answer: the last poll, else the launch."""
        return self.polls[-1] if self.polls else self.launch

    def calls(self) -> List[Call]:
        return [self.launch, *self.polls]

    def explain(self) -> str:
        return "\n\n".join(call.explain() for call in self.calls())


class NoCommitWorld:
    """A ``git init`` with a plan and some source, never committed.

    ``staged`` adds the source to the index (still no commit): the first-time
    user who ran ``git add .``, whose code review has a diff to show before any
    implement phase recorded files.
    """

    def __init__(self, root: Path, *, staged: bool = False):
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir(parents=True)
        for rel, body in SOURCES.items():
            target = self.repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        (self.repo / "plans").mkdir()
        shutil.copyfile(PLAN_FIXTURE, self.repo / PLAN_REL)
        _git(self.repo, "init", "-q", "-b", "main")
        if staged:
            _git(self.repo, "add", "--", *SOURCES)
        assert not _has_commit(self.repo)
        self.original_plan = (self.repo / PLAN_REL).read_bytes()

        scenario = root / "scenario.yaml"
        scenario.write_text(yaml.safe_dump(SCENARIO, sort_keys=False), encoding="utf-8")
        plugin = root / "no_commit_mock_plugin.py"
        plugin.write_text(PLUGIN_SOURCE, encoding="utf-8")
        self.runner = SkillRunner(
            root / "harness", scenario_path=scenario, mock_plugin=plugin,
            extra_env={"NO_COMMIT_MOCK": str(MOCK_LLM), "NO_COMMIT_MARKER": MARKER},
        )
        # The tool passes no `--models`: the reviewers are the config's, so the
        # one mock reviewer is pinned there, with the executor's runner.
        base = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text(encoding="utf-8"))
        base.setdefault("defaults", {})["models"] = [MODEL]
        base.setdefault("driver", {})["subagent_runner"] = RUNNER
        config = root / "providers.yaml"
        config.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

        env = self.runner._build_env()
        for name in _SCRUBBED_ENV:
            env.pop(name, None)
        env.update({
            "MULTI_LLM_PROVIDERS_CONFIG": str(config),
            # Set here, not left to the autouse fixtures: a module-scoped world
            # is built before any of them runs.
            "XDG_STATE_HOME": str(root / "xdg-state"),
            "MULTI_LLM_UV": str(_stub_uv(root)),
            SURFACE["dispatcher"]["route_env"]["name"]: SURFACE["dispatcher"]["route_env"]["value"],
        })
        self.env = env

    # -- the tool's argv ----------------------------------------------------- #

    def _dispatch(self, argv: List[str], timeout: int = 180) -> Call:
        done = subprocess.run(
            [sys.executable, str(SKILL_DIR / SURFACE["dispatcher"]["script"]), *argv],
            cwd=self.repo, env=self.env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout * PERF_SCALE,
        )
        return Call(argv, done.returncode, done.stdout, done.stderr)

    @staticmethod
    def compose(mode: str, *, plan: str = PLAN_REL, question: str = QUESTION,
                extra: Tuple[str, ...] = ()) -> List[str]:
        """``composeDispatcherArgv``: word, positionals, baked flags, detach, extras."""
        row = TOOL_ROWS[modes.resolve(mode).name]
        argv = [mode]
        for kind in row["positionals"]:
            if kind == SURFACE["positional_kinds"]["plan"]:
                argv.append(plan)
            elif kind == SURFACE["positional_kinds"]["question"]:
                argv.append(question)
        argv += SURFACE["baked_flags"]
        if row["supports_detach"]:
            argv.append(SURFACE["detach_flag"])
        return argv + list(extra)

    def tool(self, mode: str) -> ToolCall:
        """One tool call: launch, then poll a detached run until it is terminal."""
        call = ToolCall(mode, self._dispatch(self.compose(mode)))
        # Only a detach row launches a run; a status view prints `run_id:` too.
        run_id = call.launch.run_id if TOOL_ROWS[modes.resolve(mode).name]["supports_detach"] else None
        if run_id is None:
            return call
        deadline = time.monotonic() + 240 * PERF_SCALE
        # `--wait` re-reads the run every `WAIT_POLL_INTERVAL_S` (5 s), and a
        # mock run is over long before its second read: waiting on the run's
        # own record first leaves the tool's poll below nothing to wait for,
        # and its answer is the terminal one the tool would relay either way.
        run_dir = call.launch.banner("phase")
        while run_dir and time.monotonic() < deadline:
            if status_file.read_run(run_dir).get("state") in status_file.TERMINAL_RUN_STATES:
                break
            time.sleep(0.1)
        poll = SURFACE["dispatcher"]["poll"]
        argv = self.compose(poll[0], extra=(*poll[1:], SURFACE["dispatcher"]["run_id_option"], run_id))
        while True:
            reply = self._dispatch(argv)
            call.polls.append(reply)
            if reply.exit_code != SURFACE["dispatcher"]["exit_codes"]["nothing_to_do"]:
                return call
            if time.monotonic() > deadline:
                return call

    # -- what a run left ----------------------------------------------------- #

    @property
    def out(self) -> Path:
        return self.repo / OUT_REL

    def state(self) -> dict:
        return json.loads((self.out / "state.json").read_text(encoding="utf-8"))

    def completed(self) -> set:
        return set(self.state().get("phases_completed") or {})

    def phase_dir(self, phase: str) -> Path:
        return self.out / PHASE_DIRECTORIES.get(phase, phase)

    def report_tasks(self, phase: str) -> List[dict]:
        report = json.loads((self.phase_dir(phase) / "write_batch_report.json").read_text())
        return report["tasks"]


# --------------------------------------------------------------------------- #
# The recipes: which world each mode runs in, and its documented result
# --------------------------------------------------------------------------- #

#: The workflow the chain world runs, in order: ``full``'s own sequence, so a
#: phase added to the workflow joins the chain without an edit here.
WORKFLOW = modes.resolve("full").sequence

#: The first-time user's session in one repository, as the release plan's
#: no-commit project runs it: ``init``, a ``status`` before anything ran, a
#: question, every workflow phase in order, then ``status`` again.
CHAIN = ("init", "status", "ask", *WORKFLOW, "status")

#: The worlds and the tool calls each makes, in order. A composite runs alone in
#: a fresh repository, as a user would start it.
WORLDS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "chain": (CHAIN, False),
    "improve-plan": (("improve-plan",), False),
    "improve-code": (("improve-code",), True),
    "full": (("full",), False),
}

EXIT_OK = SURFACE["dispatcher"]["exit_codes"]["ok"]
EXIT_NOTHING = SURFACE["dispatcher"]["exit_codes"]["nothing_to_do"]
STATE_DONE = SURFACE["dispatcher"]["states"]["ok"]
STATE_NOTHING = SURFACE["dispatcher"]["states"]["nothing_to_do"]


def _phases_of(mode: str) -> Tuple[str, ...]:
    row = modes.resolve(mode)
    return row.sequence_phases or ((row.phase,) if row.is_workflow_phase else ())


def _check_review_scope(world: NoCommitWorld, *, staged: bool) -> None:
    scope = (world.phase_dir("code-review") / "review-scope.md").read_text(encoding="utf-8")
    assert f"- Base ref: `{EMPTY_TREE_SHA1}` (the empty tree" in scope, scope
    assert "Diff coverage: complete" in scope, scope
    header = load_review_scope(world.phase_dir("code-review"))
    assert header is not None and header.diff_files, scope
    for rel in header.diff_files:
        diff = (world.phase_dir("code-review") / rel).read_text(encoding="utf-8")
        assert "new file mode" in diff, (rel, diff)
    if staged:
        # The staged source is the scope: git diff against the empty tree.
        for rel in ("src/auth.py", "src/security.py"):
            assert f"`{rel}`" in scope, scope


def _check_writes(world: NoCommitWorld, phase: str) -> None:
    tasks = world.report_tasks(phase)
    assert tasks and all(task.get("status") == "ok" for task in tasks), tasks


def _check(mode: str, world: NoCommitWorld, calls: List[ToolCall]) -> None:
    """Each mode's own artifacts, beyond the exit and the result line."""
    state = world.state() if (world.out / "state.json").exists() else {}
    if mode == "init":
        assert (world.repo / ".multi-llm" / "providers.yaml").is_file()
    elif mode == "ask":
        answers = list((world.out / "ask").glob("*/answers.md"))
        assert len(answers) == 1 and answers[0].read_text(encoding="utf-8").strip(), answers
    elif mode == "status":
        # The views read the state the earlier calls left; the shared GitError
        # was raised here, by the read-only state a status opens.
        assert state.get("head_at_start") == EMPTY_TREE_SHA1, state
    elif mode in ("review-plan", "review-tasks"):
        assert (world.phase_dir(mode) / "validation.json").is_file()
        assert (world.phase_dir(mode) / "report.md").is_file()
    elif mode == "apply-suggestions":
        _check_writes(world, mode)
        assert MARKER.split("{id}")[0] in (world.repo / PLAN_REL).read_text(encoding="utf-8")
    elif mode == "generate-tasks":
        assert (world.out / "tasks" / "tasks.md").is_file()
    elif mode == "apply-task-suggestions":
        _check_writes(world, mode)
        tasks_md = (world.out / "tasks" / "tasks.md").read_text(encoding="utf-8")
        assert MARKER.split("{id}")[0] in tasks_md
    elif mode == "implement":
        _check_writes(world, mode)
        assert state.get("head_before_implement") == EMPTY_TREE_SHA1, state
        tracked = state.get("tracked_files") or []
        assert tracked, state
        for entry in tracked:
            assert (world.repo / entry["path"]).is_file(), entry
            assert entry["task_id"] == "implementation", entry
    elif mode == "review-code":
        assert state.get("head_at_start") == EMPTY_TREE_SHA1, state
        _check_review_scope(world, staged=False)
    elif mode == "apply-code-fixes":
        _check_writes(world, "apply-code-fixes")
    elif mode == "improve-plan":
        _check_writes(world, "apply-suggestions")
        assert (world.repo / PLAN_REL).read_bytes() != world.original_plan
    elif mode == "improve-code":
        _check_review_scope(world, staged=True)
        _check_writes(world, "apply-code-fixes")
    elif mode == "full":
        _check_review_scope(world, staged=False)
        assert state.get("head_before_implement") == EMPTY_TREE_SHA1, state


@dataclass(frozen=True)
class Recipe:
    """Where a mode runs and what the tool must answer each time it does."""

    world: str
    #: ``(exit, state)`` per call of the mode in its world, in order.
    expect: Tuple[Tuple[int, str], ...]
    check: Callable[[str, NoCommitWorld, List[ToolCall]], None] = _check


DONE = (EXIT_OK, STATE_DONE)

RECIPES: Dict[str, Recipe] = {
    "init": Recipe("chain", (DONE,)),
    # Before any run: "no run recorded" is exit 2 by contract (`exit_code_for`).
    # After the chain: every detached run it polled is done.
    "status": Recipe("chain", ((EXIT_NOTHING, STATE_NOTHING), DONE)),
    "ask": Recipe("chain", (DONE,)),
    **{mode: Recipe("chain", (DONE,)) for mode in WORKFLOW},
    "improve-plan": Recipe("improve-plan", (DONE,)),
    "improve-code": Recipe("improve-code", (DONE,)),
    "full": Recipe("full", (DONE,)),
}


class _Worlds:
    """Each world built and run once per module, on first use."""

    def __init__(self, factory: pytest.TempPathFactory):
        self._factory = factory
        self._runs: Dict[str, Tuple[NoCommitWorld, List[ToolCall]]] = {}

    def run(self, name: str) -> Tuple[NoCommitWorld, List[ToolCall]]:
        if name not in self._runs:
            sequence, staged = WORLDS[name]
            world = NoCommitWorld(self._factory.mktemp(f"no-commit-{name}"), staged=staged)
            self._runs[name] = (world, [world.tool(mode) for mode in sequence])
        return self._runs[name]


@pytest.fixture(scope="module")
def worlds(tmp_path_factory) -> _Worlds:
    return _Worlds(tmp_path_factory)


def test_every_mode_the_tool_exposes_has_a_recipe():
    """A mode added to the tool's surface fails here, by name, until it has one."""
    assert set(TOOL_MODES) == {m.name for m in modes.user_facing_modes()}
    missing = sorted(set(TOOL_MODES) - set(RECIPES))
    assert not missing, (
        f"tool mode(s) {missing} have no recipe in RECIPES: say which world each "
        "runs in and its documented result in a repository with no commit"
    )
    assert not set(RECIPES) - set(TOOL_MODES), "a recipe names a mode the tool no longer exposes"
    for name, recipe in RECIPES.items():
        sequence, _staged = WORLDS[recipe.world]
        assert sequence.count(name) == len(recipe.expect), name


@pytest.mark.slow
@pytest.mark.parametrize("mode", TOOL_MODES)
def test_the_mode_works_in_a_repository_with_no_commit(worlds, mode):
    recipe = RECIPES.get(mode)
    if recipe is None:
        pytest.fail(f"tool mode {mode!r} has no recipe in RECIPES (see the module docstring)")
    world, calls = worlds.run(recipe.world)
    sequence, _staged = WORLDS[recipe.world]
    mine = [call for name, call in zip(sequence, calls) if name == mode]
    assert len(mine) == len(recipe.expect)

    for call, (exit_code, state) in zip(mine, recipe.expect):
        for one in call.calls():
            # One result line per invocation, and never the shared traceback.
            assert len(one.result_lines) == 1, one.explain()
            assert "GitError" not in one.stdout + one.stderr, one.explain()
            assert "Traceback" not in one.stdout + one.stderr, one.explain()
        if TOOL_ROWS[mode]["supports_detach"]:
            assert call.launch.exit_code == EXIT_OK and call.launch.run_id, call.explain()
            assert call.polls, call.explain()
        final = call.final
        assert final.exit_code == exit_code, call.explain()
        assert final.result["state"] == state and final.result["exit"] == exit_code, call.explain()
        assert final.result["mode"] == mode, call.explain()

    for phase in _phases_of(mode):
        assert phase in world.completed(), (phase, world.state())
    recipe.check(mode, world, mine)
    # The plugin never commits for the user, and no fixture did either.
    assert not _has_commit(world.repo)
    assert head_is_unborn(str(world.repo))


# --------------------------------------------------------------------------- #
# 4. The dispatcher commands the tool composes for itself
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestTheToolsOwnCommands:
    """``abort``, ``instructions`` and ``resume`` (the surface's
    ``dispatcher`` block) in a no-commit repository with a finished chain. The
    poll is every detached row above."""

    def test_abort_with_nothing_running_is_nothing_to_do(self, worlds):
        world, _calls = worlds.run("chain")
        abort = SURFACE["dispatcher"]["abort"]
        reply = world._dispatch(world.compose(abort[0], extra=tuple(abort[1:])))
        assert reply.exit_code == EXIT_NOTHING, reply.explain()
        assert len(reply.result_lines) == 1 and reply.result["state"] == STATE_NOTHING

    @pytest.mark.parametrize("mode", TOOL_MODES)
    def test_instructions(self, worlds, mode):
        world, _calls = worlds.run("chain")
        # `fetchInstructions`: the mode word, then the host option.
        reply = world._dispatch([
            *SURFACE["dispatcher"]["instructions"], mode,
            SURFACE["dispatcher"]["host_option"], TOOL_HOST,
        ])
        assert reply.exit_code == EXIT_OK, reply.explain()
        assert len(reply.result_lines) == 1

    def test_resume_reads_the_state_a_no_commit_run_wrote(self, worlds):
        world, _calls = worlds.run("chain")
        resume = SURFACE["dispatcher"]["resume"]
        reply = world._dispatch([
            *resume["argv"], PLAN_REL, resume["option"], "implement",
            SURFACE["dispatcher"]["host_option"], TOOL_HOST,
        ])
        assert reply.exit_code == EXIT_OK, reply.explain()
        assert len(reply.result_lines) == 1
        assert "GitError" not in reply.stdout + reply.stderr
