"""The commit front doors' core (plans/review-commits.md §2.1, §2.3, §2.4, T5).

``utils/commit_review.py``'s second part runs §2.1's seven steps for
``--review-commits``, ``--apply-commit-fixes``, ``--improve-commits`` and
``util commit-brief``: resolve, print the block (with the drift check),
decide the identity, take ownership, write or stage the brief, record
``commit_range``, then re-record the invocation and call the fronted ``main``.

Every fixture is a throwaway repository under ``tmp_path``. The fronted mains
(review-code's, apply-code-fixes', the composite's ``entry_point``) are
monkeypatched to record the argv they are handed, the invocation recorded at
that moment and what ``state.json`` already holds, so nothing runs a model.
Three seams belong to other tasks and are used when they are importable,
else replaced by the small stand-ins below: the brief writer
(``utils/commit_brief.py``), the shared change test
(``git_utils.paths_changed_since``) and, always stubbed, the apply lookups
(``utils/commit_apply.py``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import apply_code_fixes_orchestrator  # noqa: E402
import code_review_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
import utils  # noqa: E402
from utils import cli_common, commit_review, detach, git_utils, improve_workflow  # noqa: E402
from utils import process_control, status_file  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.commit_review import (  # noqa: E402
    BRIEF_END_MARKER,
    BRIEF_MARKER,
    CommitReviewError,
    output_dir_for_brief,
)
from utils.state_manager import StateManager, compute_plan_hash  # noqa: E402

HOST = ["--host", "claude-code"]
FOLDER = "plans/commit-reviews"
#: review-code's real ``main``, kept before the ``fronted`` fixture swaps it out.
REVIEW_CODE_MAIN = code_review_orchestrator.main


# ---------------------------------------------------------------------------
# Stand-ins for the seams other tasks own (used only when the real one is absent)
# ---------------------------------------------------------------------------


@dataclass
class _Counts:
    skips: int = 0
    overrides: int = 0
    edited: int = 0

    def any(self) -> bool:
        return bool(self.skips or self.overrides or self.edited)


def _stand_in_commit_brief() -> types.ModuleType:
    """The brief writer's contract (CONTRACTS.md), minimally."""
    module = types.ModuleType("utils.commit_brief")
    module.BRIEF_MAX_COMMITS = 100
    module.CONTEXT_PLAN_SNAPSHOT = "context-plan.md"
    module.STAGE_FILE_PATTERN = ".commit-stage-*"
    module.ARCHIVE_PREFIX = "restarted-"
    module.RESTART_KEEP = (".lock", ".status.lock", ".status.json", "orchestrator-run.log")
    module.DecisionCounts = _Counts

    def header_identity(*, base, head, focus, plan, plan_sha256):
        identity = {"base": base, "head": head, "focus": focus or None,
                    "plan": plan or None, "plan_sha256": plan_sha256 or None}
        text = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def render_brief_header(cr, *, cwd, focus, plan, plan_snapshot, plan_sha256):
        lines = [BRIEF_MARKER, f"# Commit brief {cr.display}", ""]
        lines += [f"- {c.sha[:12]} {c.date} {c.subject}" for c in cr.commits[:module.BRIEF_MAX_COMMITS]]
        if cr.phrase:
            lines.append(f"Phrase: {cr.phrase}")
        if focus:
            lines.append(f"Focus: {focus}")
        if plan:
            lines.append(f"Context plan: {plan} (snapshot {plan_snapshot}, {plan_sha256})")
        lines.append(BRIEF_END_MARKER)
        return "\n".join(lines) + "\n"

    def split_brief(text):
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != BRIEF_MARKER:
            return None, text
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == BRIEF_END_MARKER:
                return "".join(lines[:index + 1]), "".join(lines[index + 1:])
        return text, ""

    def write_brief(path, header, *, exclusive=False):
        path = Path(path)
        if exclusive:
            with open(path, "x", encoding="utf-8", newline="") as handle:
                handle.write(header)
            return
        tail = ""
        if path.exists():
            tail = split_brief(path.read_text(encoding="utf-8"))[1]
        path.write_text(header + tail, encoding="utf-8", newline="")

    def normalize_plan_path(typed, *, cwd, root):
        text = typed if os.path.isabs(typed) else os.path.join(cwd, typed)
        return git_utils.git_path(os.path.normpath(text), repo_root=root)

    def file_sha256(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return None

    def snapshot_context_plan(source, out_dir, *, data=None):
        data = Path(source).read_bytes() if data is None else data
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        target = Path(out_dir) / module.CONTEXT_PLAN_SNAPSHOT
        target.write_bytes(data)
        return target, hashlib.sha256(data).hexdigest()

    def count_decisions(phase_dir):
        return _Counts()

    def archive_round(phase_dir, *, keep_extra=()):
        phase = Path(phase_dir)
        keep = set(module.RESTART_KEEP) | {os.path.basename(os.fspath(k)) for k in keep_extra}
        names = [n for n in sorted(os.listdir(phase))
                 if n not in keep and not fnmatch.fnmatchcase(n, module.STAGE_FILE_PATTERN)
                 and not (n.startswith(module.ARCHIVE_PREFIX) and (phase / n).is_dir())]
        if not names:
            return None
        archive = phase / f"{module.ARCHIVE_PREFIX}{datetime.now():%Y%m%d-%H%M%S}"
        while archive.exists():
            archive = Path(f"{archive}.{datetime.now():%f}")
        archive.mkdir()
        for name in names:
            os.rename(phase / name, archive / name)
        return archive

    for name, value in list(locals().items()):
        if callable(value) and not name.startswith("_") and name != "module":
            setattr(module, name, value)
    return module


def _stand_in_paths_changed_since(head, paths, *, cwd=None):
    """The shared change test (§2.5 "Drift"), minimally: diff against head,
    plus a path absent at head that is on disk."""
    paths = list(paths)
    if not paths:
        return set()
    out = subprocess.run(["git", "diff", "--name-only", "--no-renames", "-z", head, "--", *paths],
                         cwd=cwd, capture_output=True, text=True, check=True).stdout
    changed = {p for p in out.split("\0") if p}
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", "-z", head, "--", *paths],
                            cwd=cwd, capture_output=True, text=True, check=True).stdout
    at_head = {p for p in listed.split("\0") if p}
    for path in paths:
        if path not in at_head and os.path.lexists(os.path.join(cwd or ".", path)):
            changed.add(path)
    return changed


Selection = namedtuple("Selection", "brief line payload")


@pytest.fixture(autouse=True)
def seams(monkeypatch):
    """The brief writer, the change test and the apply lookups, as above."""
    try:
        brief_api = importlib.import_module("utils.commit_brief")
    except ImportError:
        brief_api = _stand_in_commit_brief()
        monkeypatch.setitem(sys.modules, "utils.commit_brief", brief_api)
        monkeypatch.setattr(utils, "commit_brief", brief_api, raising=False)
    if not hasattr(git_utils, "paths_changed_since"):
        monkeypatch.setattr(git_utils, "paths_changed_since", _stand_in_paths_changed_since,
                            raising=False)
    try:
        apply_api = importlib.import_module("utils.commit_apply")
    except ImportError:
        apply_api = types.ModuleType("utils.commit_apply")
        monkeypatch.setitem(sys.modules, "utils.commit_apply", apply_api)
        monkeypatch.setattr(utils, "commit_apply", apply_api, raising=False)

    def require_completed_review(brief, *, pinned, cwd):
        state = output_dir_for_brief(brief) / "state.json"
        try:
            document = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = {}
        if not (document.get("phases_completed") or {}).get("code-review"):
            raise CommitReviewError(f"[COMMIT_REVIEW_NOT_FOUND] {pinned}")

    def select_review(*, cwd):
        raise CommitReviewError("[COMMIT_REVIEW_NOT_FOUND] no completed commit review (stand-in)")

    monkeypatch.setattr(apply_api, "require_completed_review", require_completed_review,
                        raising=False)
    monkeypatch.setattr(apply_api, "select_review", select_review, raising=False)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    cli_common.clear_invocation()
    yield types.SimpleNamespace(brief=brief_api, apply=apply_api)
    cli_common.clear_invocation()


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


def _git(root, *args) -> str:
    return subprocess.run(["git", *args], cwd=str(root), check=True,
                          capture_output=True, text=True).stdout


def _commit(root: Path, path: str, text: str, message: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _init(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    return Path(os.path.realpath(root))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Three commits (c1 a.py+b.py+docs, c2 b.py, c3 a.py); cwd is the root."""
    root = _init(tmp_path / "repo")
    (root / "docs").mkdir()
    (root / "docs" / "x.md").write_text("# Context plan\n\nKeep errors loud.\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "keep.txt").write_text("keep\n", encoding="utf-8")
    _commit(root, "a.py", "a = 1\n", "c1: start")
    _commit(root, "b.py", "b = 2\n", "c2: add b")
    _commit(root, "a.py", "a = 3\n", "c3: change a")
    monkeypatch.chdir(root)
    return root


def _sha(root: Path, rev: str) -> str:
    return _git(root, "rev-parse", rev).strip()


def _brief_of(root: Path, base: Optional[str], head: str) -> Path:
    return root / FOLDER / f"{'root' if base is None else base[:12]}-{head[:12]}.md"


def _state(brief: Path) -> dict:
    return json.loads((output_dir_for_brief(brief) / "state.json").read_text(encoding="utf-8"))


def _stage_files(root: Path):
    return sorted(str(p) for p in (root / FOLDER).rglob(".commit-stage-*"))


# ---------------------------------------------------------------------------
# The fronted mains
# ---------------------------------------------------------------------------


@pytest.fixture
def fronted(monkeypatch):
    """Record each fronted call: argv, the invocation at that moment, and the
    ``commit_range`` already in ``state.json`` (None when absent)."""
    calls = []

    def _snapshot(argv):
        brief = cli_common.flag_value(argv, "--plan-file")
        record = None
        if brief:
            state = output_dir_for_brief(Path(os.path.abspath(brief))) / "state.json"
            if state.is_file():
                record = json.loads(state.read_text(encoding="utf-8")).get("commit_range")
        invocation = cli_common.invocation()
        stage = cli_common.flag_value(argv, "--commit-stage")
        staged = (json.loads(Path(stage).read_text(encoding="utf-8"))
                  if stage and Path(stage).is_file() else None)
        return {"argv": list(argv), "brief": brief, "record": record, "staged": staged,
                "invocation": json.loads(json.dumps(invocation)) if invocation else None}

    def review(argv):
        calls.append({"target": "review-code", **_snapshot(argv)})
        return calls[-1].get("exit", 0)

    def apply(argv):
        calls.append({"target": "apply-code-fixes", **_snapshot(argv)})
        return 0

    def entry_point(mode, argv):
        calls.append({"target": f"entry_point:{mode}", **_snapshot(argv)})
        return 0

    monkeypatch.setattr(code_review_orchestrator, "main", review)
    monkeypatch.setattr(apply_code_fixes_orchestrator, "main", apply)
    monkeypatch.setattr(improve_workflow, "entry_point", entry_point)
    return calls


def _door(mode, *argv):
    return commit_review.front_door(mode, list(argv))


def _review(*argv):
    return _door("review-commits", *argv)


def _complete_review(brief: Path) -> None:
    """Mark the brief's code-review phase completed (what apply requires)."""
    manager = StateManager(brief)
    manager.mark_phase_completed("code-review")
    manager.save()


# ===========================================================================
# Step 7: what is re-recorded, and what the fronted main is handed
# ===========================================================================


class TestReRecord:
    def test_a_detached_review_rerecords_the_full_argv_as_review_code(self, repo, fronted, capsys):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        code = _review("--revs", "last 2 commits", "--models", "A,B", *HOST, "--quick",
                       "--files", "a.py", "--force", "--focus", "X", "--detach",
                       "--yes", "--exec-subagents")
        assert code == 0
        [call] = fronted
        argv = call["argv"]
        assert call["invocation"]["mode"] == "review-code"
        assert call["invocation"]["argv"] == argv
        for kept in (["--models", "A,B"], HOST, ["--quick"], ["--files", "a.py"], ["--force"],
                     ["--detach"]):
            position = argv.index(kept[0])
            assert argv[position:position + len(kept)] == kept
        brief = _brief_of(repo, base, head)
        assert cli_common.flag_value(argv, "--plan-file") == f"{FOLDER}/{brief.name}"
        assert cli_common.flag_value(argv, "--base-ref") == base
        assert cli_common.flag_value(argv, "--head-ref") == head
        for dropped in ("--revs", "--focus", "--context-plan"):
            assert dropped not in argv
        stage = cli_common.flag_value(argv, "--commit-stage")
        assert stage and Path(stage).name.startswith(".commit-stage-")
        # The launcher published nothing but the new brief's exclusive create.
        assert brief.is_file()
        assert call["record"] is None
        staged = call["staged"]
        assert staged["fields"]["focus"] == "X" and staged["fields"]["phrase"] == "last 2 commits"
        assert staged["typed_force"] is True
        # Never consumed (no worker here), so the launcher deleted it.
        assert not Path(stage).exists()
        assert "[COMMIT_BRIEF]" in capsys.readouterr().out

    def test_a_foreground_review_records_commit_range_before_the_main(self, repo, fronted):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        observed = {}

        def review(argv):
            observed["lock_free"] = process_control.lock_is_free(
                detach.lock_path(str(output_dir_for_brief(brief) / "code-review")))
            observed["record"] = _state(brief)["commit_range"]
            observed["argv"] = list(argv)
            observed["invocation"] = cli_common.invocation()
            return 0

        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        assert observed["lock_free"] is False          # held through the fronted main
        assert observed["record"]["focus"] == "X"
        assert observed["record"]["base"] == base and observed["record"]["head"] == head
        assert observed["invocation"]["mode"] == "review-code"
        assert "--force" not in observed["argv"] and "--rerun-all" not in observed["argv"]
        assert "--commit-stage" not in observed["argv"]
        # Released afterwards.
        assert process_control.lock_is_free(
            detach.lock_path(str(output_dir_for_brief(brief) / "code-review")))

    def test_apply_rerecords_as_apply_code_fixes_with_the_brief_only(self, repo, fronted):
        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        _complete_review(brief)
        fronted.clear()
        assert _door("apply-commit-fixes", "--revs", "last 2 commits", "--yes") == 0
        [call] = fronted
        assert call["target"] == "apply-code-fixes"
        assert call["invocation"]["mode"] == "apply-code-fixes"
        assert call["argv"] == ["--yes", "--plan-file", f"{FOLDER}/{brief.name}"]
        assert call["record"]["resolved_at"]

    def test_improve_commits_records_itself_with_the_pinned_range(self, repo, fronted):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        observed = {}

        review_dir = output_dir_for_brief(brief) / "code-review"

        def entry_point(mode, argv):
            observed["lock_free"] = process_control.lock_is_free(
                detach.lock_path(str(review_dir)))
            observed["handed_over"] = improve_workflow._held_lock(review_dir) is not None
            observed["owner"] = improve_workflow.live_owner(
                mode, Path(os.path.abspath(cli_common.flag_value(argv, "--plan-file"))),
                skip_own_run_dir=False)
            observed["argv"], observed["mode"] = list(argv), mode
            observed["invocation"] = cli_common.invocation()
            return 0

        improve_workflow.entry_point = entry_point
        assert _door("improve-commits", "--revs", "last 2 commits", "--focus", "X",
                     "--models", "A") == 0
        assert observed["mode"] == "improve-commits"
        # §2.1: held through the composite too, handed over so its preflight
        # finds no other owner (its review step runs under it); released after.
        assert observed["lock_free"] is False
        assert observed["handed_over"] is True and observed["owner"] is None
        assert process_control.lock_is_free(detach.lock_path(str(review_dir)))
        assert improve_workflow._HELD_LOCKS == {}
        argv = observed["argv"]
        assert argv[:2] == ["--models", "A"]
        assert cli_common.flag_value(argv, "--plan-file") == f"{FOLDER}/{brief.name}"
        assert cli_common.flag_value(argv, "--base-ref") == base
        assert cli_common.flag_value(argv, "--head-ref") == head
        recorded = observed["invocation"]
        assert recorded["mode"] == "improve-commits"
        assert recorded["argv"] == ["--models", "A", "--revs", f"{base[:12]}..{head[:12]}"]

    def test_the_improve_worker_takes_the_stage_and_writes_nothing(self, repo, fronted, capsys):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        stage_seen = {}

        def launcher(mode, argv):
            stage_seen["stage"] = cli_common.flag_value(argv, "--commit-stage")
            stage_seen["recorded"] = list(cli_common.invocation()["argv"])
            # Keep the stage, as a launched worker would see it.
            shutil.copy(stage_seen["stage"], stage_seen["stage"] + ".keep")
            return 0

        improve_workflow.entry_point = launcher
        assert _door("improve-commits", "--revs", "last 2 commits", "--models", "A",
                     "--detach", "--yes", "--exec-subagents") == 0
        stage = Path(stage_seen["stage"])
        os.replace(str(stage) + ".keep", stage)
        brief = _brief_of(repo, base, head)
        assert not (output_dir_for_brief(brief) / "state.json").exists()
        capsys.readouterr()

        worker_calls = []
        improve_workflow.entry_point = lambda mode, argv: worker_calls.append(list(argv)) or 0
        worker_argv = [t for t in stage_seen["recorded"] if t != "--detach"] + [
            "--internal-worker", "--internal-run-id", "r1"]
        assert _door("improve-commits", *worker_argv) == 0
        assert capsys.readouterr().out == ""
        [argv] = worker_calls
        assert cli_common.flag_value(argv, "--base-ref") == base
        assert cli_common.flag_value(argv, "--head-ref") == head
        assert cli_common.flag_value(argv, "--commit-stage") == str(stage)
        assert cli_common.flag_value(argv, "--plan-file") == str(brief)
        assert "--revs" not in argv and "--models" in argv
        # Nothing written by the child before the worker's lock.
        assert stage.exists()
        assert not (output_dir_for_brief(brief) / "state.json").exists()


# ===========================================================================
# §2.3: the output shape
# ===========================================================================


@pytest.fixture
def long_repo(tmp_path, monkeypatch):
    root = _init(tmp_path / "long")
    for index in range(26):
        _commit(root, f"f{index:02d}.py", f"x = {index}\n", f"commit {index:02d}")
    monkeypatch.chdir(root)
    return root


def _dispatch(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _lines(out, prefix):
    return [line for line in out.splitlines() if line.startswith(prefix)]


class TestOutputShape:
    def test_a_first_review_points_at_git_log_then_at_the_brief(self, long_repo, fronted, capsys):
        base, head = _sha(long_repo, "HEAD~25"), _sha(long_repo, "HEAD")
        code, out, err = _dispatch(["review-commits", "last", "25", "commits", *HOST], capsys)
        assert code == 0, err
        lines = out.splitlines()
        end = lines.index("[COMMIT_RANGE_END]")
        assert f"(full list: git log --first-parent {base[:12]}..{head[:12]})" in out
        assert lines[end + 1].startswith("[COMMIT_BRIEF] ")
        shown = lines[end + 1].split(" ", 1)[1]
        assert (long_repo / shown).is_file()
        result = parse_result_line(out)
        assert result["commit_range"]["brief"] == shown
        assert len(result["commit_range"]["commits"]) == 20
        assert result["commit_range"]["more"] == 5
        assert result["commit_range"]["full_list"] == shown

        code, out, err = _dispatch(["review-commits", f"{base[:12]}..{head[:12]}", *HOST], capsys)
        assert code == 0, err
        assert f"… and 5 more (full list: {shown})" in out

    def test_past_the_brief_cap_the_result_names_the_git_command(self, long_repo, fronted, seams,
                                                                 monkeypatch, capsys):
        monkeypatch.setattr(seams.brief, "BRIEF_MAX_COMMITS", 10)
        base, head = _sha(long_repo, "HEAD~25"), _sha(long_repo, "HEAD")
        code, out, _err = _dispatch(["review-commits", "-25", *HOST], capsys)
        assert code == 0
        result = parse_result_line(out)
        assert result["commit_range"]["full_list"] == (
            f"git log --first-parent {base[:12]}..{head[:12]}")

    def test_a_short_range_has_no_pointer(self, repo, fronted, capsys):
        code, out, _err = _dispatch(["review-commits", "last", "2", "commits", *HOST], capsys)
        assert code == 0
        result = parse_result_line(out)
        assert result["commit_range"]["more"] == 0
        assert result["commit_range"]["full_list"] is None
        assert "full list" not in out

    def test_a_launch_refused_at_step_4_prints_the_block_and_no_brief(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        capsys.readouterr()
        holder = process_control.PhaseLock(
            detach.lock_path(str(output_dir_for_brief(brief) / "code-review")))
        holder.acquire({"holder": "another shell", "pid": os.getpid()})
        try:
            fronted.clear()
            assert _review("--revs", "last 2 commits", "--focus", "Y") == 3
        finally:
            holder.release()
        out, err = capsys.readouterr()
        assert "[COMMIT_RANGE]" in out and "[COMMIT_RANGE_END]" in out
        assert "[COMMIT_BRIEF]" not in out
        assert "brief and state left unchanged" in err
        assert fronted == []


# ===========================================================================
# Step 4: ownership
# ===========================================================================


def _live_run(directory: Path, run_id: str, mode: str) -> process_control.PhaseLock:
    """A held ``.lock`` plus a live ``run`` record in *directory* (this process)."""
    directory.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(directory), {
        "run_id": run_id, "pid": os.getpid(),
        "process_start_time": process_control.process_start_time(os.getpid()),
        "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
        "state": "running", "detached": True, "mode": mode,
    })
    lock = process_control.PhaseLock(detach.lock_path(str(directory)))
    lock.acquire({"run_id": run_id, "pid": os.getpid(),
                  "process_start_time": process_control.process_start_time(os.getpid())})
    return lock


class TestOwnership:
    @pytest.fixture
    def reviewed(self, repo, fronted):
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        fronted.clear()
        cli_common.record_invocation("review-code", ["--sentinel"])
        return brief

    def _inputs(self, brief):
        state = output_dir_for_brief(brief) / "state.json"
        return brief.read_bytes(), state.read_bytes(), cli_common.invocation()

    def test_a_detached_live_run_is_reattached_and_left_alone(self, repo, reviewed, fronted, capsys):
        before = self._inputs(reviewed)
        lock = _live_run(output_dir_for_brief(reviewed) / "code-review", "run-1", "review-code")
        try:
            code = _review("--revs", "last 2 commits", "--focus", "other", "--detach",
                           "--if-not-running", "--yes", "--exec-subagents")
        finally:
            lock.release()
        assert code == 0
        [call] = fronted
        # launch re-attaches: no restart flags, no stage, the brief untouched.
        assert "--commit-stage" not in call["argv"]
        assert "--rerun-all" not in call["argv"]
        assert reviewed.read_bytes() == before[0]
        assert (output_dir_for_brief(reviewed) / "state.json").read_bytes() == before[1]
        assert _stage_files(repo) == []
        assert "[COMMIT_REVIEW_RESTARTED]" not in capsys.readouterr().out

    def test_without_if_not_running_a_live_run_refuses(self, repo, reviewed, fronted, capsys):
        before = self._inputs(reviewed)
        lock = _live_run(output_dir_for_brief(reviewed) / "code-review", "run-1", "review-code")
        try:
            code = _review("--revs", "last 2 commits", "--focus", "other", "--detach",
                           "--yes", "--exec-subagents")
        finally:
            lock.release()
        assert code == 3
        assert fronted == []
        assert self._inputs(reviewed) == before
        assert _stage_files(repo) == []
        out, err = capsys.readouterr()
        assert "[COMMIT_BRIEF]" not in out
        assert "run run-1" in err and "brief and state left unchanged" in err

    def test_a_foreground_holder_refuses_a_second_foreground_launch(self, repo, reviewed, fronted):
        held = commit_review.prepare("review-commits", ["--revs", "last 2 commits"])
        before = self._inputs(reviewed)
        try:
            assert held.lock is not None
            assert _review("--revs", "last 2 commits", "--focus", "Y") == 3
        finally:
            held.release()
        assert self._inputs(reviewed)[:2] == before[:2]
        assert fronted == []

    def test_concurrent_launches_leave_one_consistent_owner(self, repo, fronted, seams):
        first = commit_review.prepare("review-commits", ["--revs", "last 2 commits",
                                                         "--focus", "first"])
        try:
            assert _review("--revs", "last 2 commits", "--focus", "second") == 3
        finally:
            first.release()
        record = _state(first.brief)["commit_range"]
        assert record["focus"] == "first"
        header = seams.brief.split_brief(first.brief.read_text(encoding="utf-8"))[0]
        assert "first" in header and "second" not in header
        assert record["header_sha256"] == seams.brief.header_identity(
            base=record["base"], head=record["head"], focus="first", plan=None, plan_sha256=None)

    def test_a_restart_never_takes_a_live_run_over_without_a_typed_force(self, repo, reviewed,
                                                                         fronted):
        phase = output_dir_for_brief(reviewed) / "code-review"
        # With a typed --force, the launch may take over: the stage says so.
        lock = _live_run(phase, "run-1", "review-code")
        seen = {}
        try:
            def review(argv):
                stage = cli_common.flag_value(argv, "--commit-stage")
                seen["typed"] = commit_review.stage_takes_over(stage)
                return 0

            code_review_orchestrator.main = review
            assert _review("--revs", "last 2 commits", "--focus", "Y", "--force", "--detach",
                           "--yes", "--exec-subagents") == 0
        finally:
            lock.release()
        assert seen["typed"] is True

    def test_launch_ignores_a_restarts_synthesised_force(self, repo, reviewed, fronted, monkeypatch):
        """detach.launch's take-over decision reads the stage: a restart's own
        --force refuses a live run rather than breaking its lock."""
        phase = output_dir_for_brief(reviewed) / "code-review"
        stage = phase / ".commit-stage-test.json"
        stage.write_text(json.dumps({"version": 1, "typed_force": False}), encoding="utf-8")
        lock = _live_run(phase, "run-1", "review-code")
        monkeypatch.setattr(detach, "resolve_uv", lambda: "/usr/bin/uv")
        args = types.SimpleNamespace(force=True, if_not_running=False, host=None,
                                     commit_stage=str(stage), exec_subagents=False)
        try:
            with pytest.raises(detach.DetachError, match="already owns"):
                detach.launch(mode="review-code", argv=["--force"], phase_dir=str(phase),
                              plan_file=str(reviewed), phase="code-review", args=args)
        finally:
            lock.release()
        assert commit_review.stage_takes_over(None) is True


# ===========================================================================
# Step 6: the commit_range record
# ===========================================================================


class TestStateRecord:
    def test_a_new_brief_gets_an_initial_state_with_the_whole_record(self, repo, fronted):
        assert _review("--revs", "last 2 commits") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        state = _state(_brief_of(repo, base, head))
        for key in ("schema_version", "plan_path", "plan_hash", "head_at_start", "branch_name",
                    "phases_completed"):
            assert key in state
        record = state["commit_range"]
        for key in ("base", "head", "commits", "phrase", "window", "notes", "size", "brief",
                    "plan", "plan_snapshot", "plan_sha256", "focus", "header_sha256",
                    "drifted", "resolved_at"):
            assert key in record, key
        assert "reviewed_header_sha256" not in record and "guard" not in record
        assert [c["sha"] for c in record["commits"]] == [head, _sha(repo, "HEAD~1")]
        assert record["brief"] == f"{FOLDER}/{base[:12]}-{head[:12]}.md"
        assert fronted[0]["record"] == record

    def test_a_reuse_keeps_what_no_front_door_passes(self, repo, fronted, monkeypatch):
        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        manager = StateManager(brief)
        manager.state["commit_range"].update({"reviewed_header_sha256": "R", "guard": {"g": 1}})
        manager.save()
        monkeypatch.setattr(commit_review, "_now_iso", lambda: "2030-01-01T00:00:00Z")
        assert _review("--revs", "-2") == 0
        record = _state(brief)["commit_range"]
        assert record["reviewed_header_sha256"] == "R" and record["guard"] == {"g": 1}
        assert record["phrase"] == "last 2 commits"
        assert record["resolved_at"] == "2030-01-01T00:00:00Z"

    def test_the_phases_carry_the_record_unchanged(self, repo, fronted, monkeypatch, capsys):
        """review-code's own initialisation, its ``--force`` branch and a
        ``--force --rerun-all`` restart, run for real on the argv the front
        door hands it, keep the record (§2.1 "The phase carries the record and
        never rebuilds it"). Each run stops where the fan-out would start a
        model, after the phase has saved its round start."""
        assert _review("--revs", "last 2 commits") == 0
        handed = fronted[0]["argv"]
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        manager = StateManager(brief)
        manager.state["commit_range"].update({"header_sha256": "H", "drifted": ["d.py"],
                                              "resolved_at": "T", "guard": {"g": 1}})
        manager.save()
        expected = {key: value for key, value in _state(brief)["commit_range"].items()
                    if key != "guard"}

        class FanOut(Exception):
            """Where the fan-out would run a model."""

        async def no_fan_out(*args, **kwargs):
            raise FanOut

        monkeypatch.setattr(code_review_orchestrator, "run_all_reviews", no_fan_out)
        for restart in ([], ["--force"], ["--force", "--rerun-all"]):
            if restart:
                _complete_review(brief)
            with pytest.raises(FanOut):
                commit_review._call_main(REVIEW_CODE_MAIN,
                                         [*handed, "--models", "test:model", *restart])
            state = _state(brief)
            # The round start wrote only the review's own fields.
            assert state["commit_range"] == {**expected, "reviewed_header_sha256": "H"}, restart
            assert "code-review" not in state.get("phases_completed", {}), restart
            out = capsys.readouterr().out
            assert ("NOTE: Cleared previous phase completion (--force mode)" in out) is bool(
                restart), restart

    def test_a_racing_locked_writer_keeps_both_records(self, repo, fronted):
        from utils.metrics import state_file_lock

        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        state_path = output_dir_for_brief(brief) / "state.json"
        started = threading.Event()

        def metrics_writer():
            with state_file_lock(state_path):
                document = json.loads(state_path.read_text(encoding="utf-8"))
                started.set()
                time.sleep(0.3)
                document.setdefault("metrics", {})["review-code"] = [
                    {"label": "t1", "timestamp": "2026-10-26T10:00:00"}]
                state_path.write_text(json.dumps(document), encoding="utf-8")

        worker = threading.Thread(target=metrics_writer)
        worker.start()
        started.wait(5)
        StateManager(brief).record_commit_range({"resolved_at": "raced"})
        worker.join()
        document = json.loads(state_path.read_text(encoding="utf-8"))
        assert document["metrics"]["review-code"][0]["label"] == "t1"
        assert document["commit_range"]["resolved_at"] == "raced"
        assert document["commit_range"]["head"] == _sha(repo, "HEAD")

    @pytest.mark.parametrize("broken", ["{not json", "[]"], ids=["unparseable", "not an object"])
    def test_an_unparseable_state_is_replaced_by_one_carrying_the_record(
        self, repo, fronted, capsys, broken,
    ):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        assert _review("--revs", "last 2 commits") == 0
        notice = "[STATE_REPLACED]"
        assert notice not in capsys.readouterr().out
        state_path = output_dir_for_brief(brief) / "state.json"
        state_path.write_text(broken, encoding="utf-8")
        assert _review("--revs", "last 2 commits") == 0
        state = _state(brief)
        assert state["commit_range"]["head"] == head
        assert "schema_version" in state
        # Review R2, item 6: the replacement is said, in one line.
        shown = os.path.relpath(state_path).replace(os.sep, "/")
        lines = [line for line in capsys.readouterr().out.splitlines()
                 if line.startswith(notice)]
        assert lines == [f"{notice} {shown} did not parse; a fresh state was written and "
                         f"the earlier round's records are gone"]
        # A state that parses is recorded over without a word.
        assert _review("--revs", "last 2 commits") == 0
        assert notice not in capsys.readouterr().out

    def test_record_commit_range_replaces_only_the_given_fields(self, repo):
        brief = repo / FOLDER / "manual.md"
        brief.parent.mkdir(parents=True)
        brief.write_text("# brief\n", encoding="utf-8")
        StateManager(brief).record_commit_range({"head": "h", "focus": "x", "phrase": "p"})
        StateManager(brief).record_commit_range({"focus": None, "resolved_at": "t"},
                                                refresh_plan_hash=True)
        record = _state(brief)["commit_range"]
        assert record == {"head": "h", "focus": None, "phrase": "p", "resolved_at": "t"}


# ===========================================================================
# §2.4: identity
# ===========================================================================


class TestIdentity:
    def test_the_continuation_brief_path_and_dash_two_keep_the_identity(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        before_record = _state(brief)["commit_range"]
        before_bytes = brief.read_bytes()
        capsys.readouterr()
        for argv in (["--revs", f"{base[:12]}..{head[:12]}"],
                     ["--plan-file", f"{FOLDER}/{brief.name}"],
                     ["--revs", "-2"]):
            fronted.clear()
            assert _review(*argv) == 0
            record = _state(brief)["commit_range"]
            for key in ("header_sha256", "phrase", "window", "focus"):
                assert record[key] == before_record[key], (argv, key)
            assert brief.read_bytes() == before_bytes
            assert "--rerun-all" not in fronted[0]["argv"]
            assert "[COMMIT_REVIEW_RESTARTED]" not in capsys.readouterr().out

    @pytest.mark.parametrize("focus, recorded", [("Y", "Y"), ("", None)])
    def test_a_typed_focus_that_differs_restarts(self, repo, fronted, capsys, focus, recorded):
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        before = _state(brief)["commit_range"]["header_sha256"]
        capsys.readouterr()
        fronted.clear()
        assert _review("--revs", "last 2 commits", "--focus", focus) == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED] focus" in out
        argv = fronted[0]["argv"]
        assert "--force" in argv and "--rerun-all" in argv
        record = _state(brief)["commit_range"]
        assert record["focus"] == recorded
        assert record["header_sha256"] != before
        assert fronted[0]["record"]["focus"] == recorded

    def test_the_context_plan_retyped_in_another_spelling_keeps_the_identity(
            self, repo, fronted, capsys, monkeypatch):
        assert _review("--revs", "last 2 commits", "--context-plan", "docs/x.md") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        record = _state(brief)["commit_range"]
        snapshot = repo / record["plan_snapshot"]
        snapshot_bytes = snapshot.read_bytes()
        assert record["plan"] == "docs/x.md"
        capsys.readouterr()
        for spelling, where in (("./docs/x.md", repo), (str(repo / "docs" / "x.md"), repo),
                                ("../docs/x.md", repo / "src")):
            monkeypatch.chdir(where)
            assert _review("--revs", "last 2 commits", "--context-plan", spelling) == 0
            assert _state(brief)["commit_range"]["header_sha256"] == record["header_sha256"]
            assert snapshot.read_bytes() == snapshot_bytes
            assert "[COMMIT_REVIEW_RESTARTED]" not in capsys.readouterr().out


class TestThePrintedFollowUp:
    """PD-4 (2026-10-27): the follow-up review-code prints on a brief is the
    commit mode with the pinned range (`review-commits <base12>..<head12> …
    --reaggregate`, plans/review-commits.md §2.8), and `review-commits.md` §3
    now has the host run it as printed. That is safe only if the second
    front-door pass it makes lands on the same brief, the same two commits and
    the same round however far the branch has moved, which is what this pins,
    and only with the pinned range: the phrase typed again is resolved afresh."""

    def test_it_reaches_the_same_review_after_the_branch_moved(self, repo, fronted, capsys):
        import shlex

        from utils import commit_spelling

        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        before_record = _state(brief)["commit_range"]
        before_bytes = brief.read_bytes()
        printed = commit_spelling.spell_retry_command(
            "review-code", brief, "--reaggregate", flags=["--quick", "--yes", *HOST])
        argv = shlex.split(printed)
        rest = argv[argv.index("review-commits") + 1:]
        assert rest[0] == f"{base[:12]}..{head[:12]}" and rest[-1] == "--reaggregate", printed
        _commit(repo, "c.py", "c = 4\n", "c4: after the review")
        capsys.readouterr()
        fronted.clear()

        assert _review("--revs", *rest) == 0
        [call] = fronted
        out = capsys.readouterr().out
        assert call["target"] == "review-code"
        assert "--reaggregate" in call["argv"]
        assert cli_common.flag_value(call["argv"], "--plan-file") == f"{FOLDER}/{brief.name}"
        assert cli_common.flag_value(call["argv"], "--base-ref") == base
        assert cli_common.flag_value(call["argv"], "--head-ref") == head
        assert "--force" not in call["argv"] and "--rerun-all" not in call["argv"]
        record = _state(brief)["commit_range"]
        for key in ("base", "head", "phrase", "focus", "header_sha256"):
            assert record[key] == before_record[key], key
        assert brief.read_bytes() == before_bytes
        assert "[COMMIT_REVIEW_RESTARTED]" not in out
        # The pass prints the block again, which the core says not to relay.
        assert "[COMMIT_RANGE]" in out and f"[COMMIT_BRIEF] {FOLDER}/{brief.name}" in out

        # The words the user typed are not a substitute: resolved afresh, the
        # phrase now names the moved branch's last two commits, another brief.
        fronted.clear()
        assert _review("--revs", "last 2 commits", "--reaggregate") == 0
        assert cli_common.flag_value(fronted[0]["argv"], "--plan-file") != f"{FOLDER}/{brief.name}"


# ===========================================================================
# §2.4: the restart's archive
# ===========================================================================


class TestRestartArchive:
    @pytest.fixture
    def old_round(self, repo, fronted):
        assert _review("--revs", "last 2 commits", "--focus", "X", "--models", "A", "B") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        phase = output_dir_for_brief(brief) / "code-review"
        for name, text in (("A.json", "[]"), ("B.json", "[]"), ("report.md", "# report\n"),
                           ("user_selections.json", "{}"), (".status.json", "{}")):
            (phase / name).write_text(text, encoding="utf-8")
        (phase / ".lock").write_text("", encoding="utf-8")
        fronted.clear()
        return brief, phase

    def test_the_old_round_moves_out_before_the_main_is_called(self, old_round, fronted, seams,
                                                               monkeypatch, capsys):
        brief, phase = old_round
        monkeypatch.setattr(seams.brief, "count_decisions",
                            lambda phase_dir: _Counts(skips=2, overrides=1, edited=0))
        seen = {}

        def review(argv):
            seen["out"] = capsys.readouterr().out
            seen["present"] = sorted(p.name for p in phase.iterdir())
            return 0

        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--focus", "Y", "--models", "A") == 0
        out = seen["out"]
        assert "[COMMIT_REVIEW_ARCHIVED]" in out
        assert ("[COMMIT_DECISIONS_DROPPED] 2 skips, 1 valid/invalid overrides, 0 edited "
                "descriptions") in out
        assert out.index("[COMMIT_REVIEW_ARCHIVED]") < out.index("[COMMIT_REVIEW_RESTARTED]")
        assert out.index("[COMMIT_REVIEW_RESTARTED]") < out.index("[COMMIT_BRIEF]")
        [archive] = [p for p in phase.iterdir() if p.name.startswith("restarted-")]
        assert {"B.json", "report.md", "user_selections.json", "A.json"} <= {
            p.name for p in archive.iterdir()}
        assert "B.json" not in seen["present"] and "report.md" not in seen["present"]
        assert ".lock" in seen["present"] and ".status.json" in seen["present"]

    def test_with_no_decisions_only_the_archive_line_prints(self, old_round, fronted, seams,
                                                            monkeypatch, capsys):
        monkeypatch.setattr(seams.brief, "count_decisions", lambda phase_dir: _Counts())
        capsys.readouterr()
        assert _review("--revs", "last 2 commits", "--focus", "Y") == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_ARCHIVED]" in out
        assert "[COMMIT_DECISIONS_DROPPED]" not in out

    def test_a_crash_between_the_move_and_the_header_keeps_the_old_identity(
            self, old_round, fronted, seams, monkeypatch):
        brief, phase = old_round
        before_bytes = brief.read_bytes()
        before = _state(brief)["commit_range"]["header_sha256"]

        def crash(path, header, *, exclusive=False):
            raise OSError("injected crash")

        with monkeypatch.context() as patched:
            patched.setattr(seams.brief, "write_brief", crash)
            with pytest.raises(OSError, match="injected"):
                _review("--revs", "last 2 commits", "--focus", "Y")
        assert brief.read_bytes() == before_bytes
        assert _state(brief)["commit_range"]["header_sha256"] == before
        assert [p for p in phase.iterdir() if p.name.startswith("restarted-")]
        capsys_free = _review("--revs", "last 2 commits", "--focus", "Y")
        assert capsys_free == 0
        assert _state(brief)["commit_range"]["focus"] == "Y"
        assert "Y" in seams.brief.split_brief(brief.read_text(encoding="utf-8"))[0]

    def test_the_restarts_record_pops_the_completion_even_if_the_phase_never_runs(
            self, old_round, fronted):
        """Review fix 3, item 3: the archived round stops reading as a
        completed review in the front door's own step-6 write, so a phase
        that never runs (killed; here a stand-in that does nothing) cannot
        leave it to ``select_review``/``require_completed_review``."""
        brief, phase = old_round
        _complete_review(brief)
        assert _state(brief)["phases_completed"].get("code-review")
        seen = {}

        def review(argv):
            seen["phases"] = dict(_state(brief).get("phases_completed") or {})
            return 0

        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--focus", "Y") == 0
        assert "code-review" not in seen["phases"]
        assert "code-review" not in _state(brief)["phases_completed"]

    def test_a_crash_after_the_move_never_leaves_the_archived_round_completed(
            self, old_round, fronted, seams, monkeypatch):
        brief, phase = old_round
        _complete_review(brief)

        def crash(path, header, *, exclusive=False):
            raise OSError("injected crash")

        with monkeypatch.context() as patched:
            patched.setattr(seams.brief, "write_brief", crash)
            with pytest.raises(OSError, match="injected"):
                _review("--revs", "last 2 commits", "--focus", "Y")
        assert [p for p in phase.iterdir() if p.name.startswith("restarted-")]
        assert "code-review" not in _state(brief)["phases_completed"]
        # The record is untouched apart from the completion.
        assert _state(brief)["commit_range"]["focus"] == "X"

    def test_a_header_a_crash_left_ahead_of_the_record_is_rewritten_on_reuse(
            self, old_round, fronted, seams, monkeypatch, capsys):
        """Review fix 3, A3: a crash between step 5's header write and step
        6's record leaves a header (focus Y) the record (focus X) does not
        describe. The next entry that reuses the record rewrites the header
        to the recorded identity, with ``plan_hash`` re-stamped; a reuse
        whose header already matches leaves the brief's bytes alone."""
        brief, phase = old_round

        def crash(*args, **kwargs):
            raise OSError("injected crash after the header")

        with monkeypatch.context() as patched:
            patched.setattr(commit_review, "_record_state", crash)
            with pytest.raises(OSError, match="injected"):
                _review("--revs", "last 2 commits", "--focus", "Y")
        split = seams.brief.split_brief
        assert "Y" in split(brief.read_text(encoding="utf-8"))[0]
        assert _state(brief)["commit_range"]["focus"] == "X"
        capsys.readouterr()

        assert _review("--revs", "last 2 commits") == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED]" not in out
        header = split(brief.read_text(encoding="utf-8"))[0]
        assert "Y" not in header and "X" in header
        state = _state(brief)
        assert state["commit_range"]["focus"] == "X"
        assert state["plan_hash"] == compute_plan_hash(brief.resolve())

        settled = brief.read_bytes()
        assert _review("--revs", "last 2 commits") == 0
        assert brief.read_bytes() == settled

    @pytest.mark.parametrize("broken", ["{not json", "[]"], ids=["unparseable", "not an object"])
    def test_a_changed_focus_over_an_unreadable_state_restarts(self, old_round, fronted, seams,
                                                               capsys, broken):
        """With no readable record, nothing says which header the round in
        code-review/ answers; a header rewrite (focus Y over X) sets that round
        aside and restarts, as a recorded change does, rather than leave review-code
        to resume the old results under the new brief. [STATE_REPLACED] is
        still said. The same header again over a broken state restarts nothing."""
        brief, phase = old_round
        state_path = output_dir_for_brief(brief) / "state.json"
        state_path.write_text(broken, encoding="utf-8")
        capsys.readouterr()
        seen = {}

        def review(argv):
            seen["argv"] = list(argv)
            seen["present"] = sorted(p.name for p in phase.iterdir())
            return 0

        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--focus", "Y") == 0
        out = capsys.readouterr().out
        assert "--force" in seen["argv"] and "--rerun-all" in seen["argv"]
        for name in ("A.json", "B.json", "report.md", "user_selections.json"):
            assert name not in seen["present"], name
        [archive] = [p for p in phase.iterdir() if p.name.startswith("restarted-")]
        assert {"A.json", "B.json", "report.md", "user_selections.json"} <= {
            p.name for p in archive.iterdir()}
        shown = os.path.relpath(state_path).replace(os.sep, "/")
        assert f"[COMMIT_REVIEW_RESTARTED] {shown} did not parse" in out
        assert f"[STATE_REPLACED] {shown} did not parse" in out
        assert _state(brief)["commit_range"]["focus"] == "Y"
        assert "Y" in seams.brief.split_brief(brief.read_text(encoding="utf-8"))[0]

        state_path.write_text(broken, encoding="utf-8")
        fronted.clear()
        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--focus", "Y") == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED]" not in out and "[STATE_REPLACED]" in out
        assert "--rerun-all" not in seen["argv"]
        assert len([p for p in phase.iterdir() if p.name.startswith("restarted-")]) == 1

    def test_a_reuse_keeps_the_completion(self, old_round, fronted):
        brief, phase = old_round
        _complete_review(brief)
        assert _review("--revs", "last 2 commits") == 0
        assert _state(brief)["phases_completed"].get("code-review")

    def test_a_refused_detached_launch_moves_nothing(self, old_round, fronted):
        brief, phase = old_round
        lock = _live_run(phase, "run-9", "review-code")
        try:
            assert _review("--revs", "last 2 commits", "--focus", "Y", "--detach", "--yes",
                           "--exec-subagents") == 3
        finally:
            lock.release()
        assert (phase / "B.json").exists() and (phase / "report.md").exists()
        assert not [p for p in phase.iterdir() if p.name.startswith("restarted-")]


# ===========================================================================
# §2.4: the context-plan snapshot
# ===========================================================================


class TestContextPlanSnapshot:
    @pytest.mark.parametrize("damage", ["changed", "missing"])
    def test_resume_refuses_a_damaged_snapshot_and_explicit_plan_repairs_it(
        self, repo, fronted, capsys, damage,
    ):
        assert _review("--revs", "last 2 commits", "--context-plan", "docs/x.md") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        record = _state(brief)["commit_range"]
        snapshot = repo / record["plan_snapshot"]
        original = snapshot.read_bytes()
        if damage == "changed":
            snapshot.write_bytes(b"# corrupted snapshot\n")
        else:
            snapshot.unlink()
        called = len(fronted)
        assert _review("--revs", "last 2 commits") == 3
        assert len(fronted) == called
        assert "[CONTEXT_PLAN_SNAPSHOT_CHANGED]" in capsys.readouterr().out
        assert _review("--revs", "last 2 commits", "--context-plan", "docs/x.md") == 0
        assert snapshot.read_bytes() == original
        assert _state(brief)["commit_range"]["header_sha256"] == record["header_sha256"]

    def test_the_snapshot_freezes_the_plan_until_it_is_typed_again(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits", "--context-plan", "docs/x.md") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = _brief_of(repo, base, head)
        record = _state(brief)["commit_range"]
        snapshot = output_dir_for_brief(brief) / "context-plan.md"
        original = (repo / "docs" / "x.md").read_bytes()
        assert snapshot.read_bytes() == original
        assert record["plan_snapshot"] == f"{FOLDER}/{brief.stem}/context-plan.md"
        assert record["plan_sha256"] == hashlib.sha256(original).hexdigest()

        (repo / "docs" / "x.md").write_text("# Context plan\n\nNew text.\n", encoding="utf-8")
        capsys.readouterr()
        assert _review("--revs", f"{base[:12]}..{head[:12]}") == 0
        out = capsys.readouterr().out
        assert f"[CONTEXT_PLAN_CHANGED] docs/x.md differs from the snapshot this review uses " \
               f"({record['plan_sha256'][:12]})" in out
        assert "[COMMIT_REVIEW_RESTARTED]" not in out
        assert snapshot.read_bytes() == original

        assert _review("--revs", f"{base[:12]}..{head[:12]}", "--context-plan", "docs/x.md") == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED] context plan docs/x.md changed" in out
        updated = (repo / "docs" / "x.md").read_bytes()
        assert snapshot.read_bytes() == updated
        assert _state(brief)["commit_range"]["plan_sha256"] == hashlib.sha256(updated).hexdigest()

    def test_apply_keeps_the_recorded_review(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        _complete_review(brief)
        before_bytes = brief.read_bytes()
        before = _state(brief)["commit_range"]["header_sha256"]
        capsys.readouterr()
        assert _door("apply-commit-fixes", "--revs", "last 2 commits", "--focus", "Y") == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_KEPT] --focus Y is not what the review on record used (X)" in out
        assert brief.read_bytes() == before_bytes
        assert _state(brief)["commit_range"]["header_sha256"] == before


# ===========================================================================
# §2.2: [RANGE_AFTER_FLAGS]
# ===========================================================================


AFTER_FLAGS = [
    (["--focus", "x", "last", "2", "commits"], "last 2 commits"),
    (["--quick", "-2"], "-2"),
    (["--models", "A", "B..C"], "B..C"),
    (["--models", "A", "last", "2", "commits"], "last 2 commits"),
]


#: Flags apply-code-fixes' parser does not have: on apply-commit-fixes the
#: rows starting with them are an unknown flag, the fronted parser's error
#: (review fix 3, A2), never ``[RANGE_AFTER_FLAGS]``.
NOT_APPLY_FLAGS = ("--quick", "--models")


class TestRangeAfterFlags:
    @pytest.mark.parametrize("mode, tokens, words", [
        (mode, tokens, words)
        for mode in commit_review.COMMIT_MODES for tokens, words in AFTER_FLAGS
        if not (mode == "apply-commit-fixes" and tokens[0] in NOT_APPLY_FLAGS)
    ])
    def test_a_range_after_a_flag_is_refused_before_any_write(self, repo, fronted, capsys,
                                                              mode, tokens, words):
        assert _door(mode, *tokens) == 3
        out, err = capsys.readouterr()
        [marker] = _lines(out, "[RANGE_AFTER_FLAGS] ")
        assert words in marker
        assert f"--{mode} " in err and words in err.split(f"--{mode} ", 1)[1]
        assert not (repo / FOLDER).exists()
        assert fronted == []

    @pytest.mark.parametrize("tokens, words", AFTER_FLAGS)
    def test_util_commit_brief_refuses_it_too(self, repo, capsys, tokens, words):
        assert commit_review.commit_brief_main(tokens) == 3
        out, err = capsys.readouterr()
        assert f"[RANGE_AFTER_FLAGS] {words}" in out
        assert f"util commit-brief {words}" in err
        assert not (repo / FOLDER).exists()

    def test_two_models_still_forward_two_models(self, repo, fronted):
        assert _review("--revs", "last 2 commits", "--models", "A", "B") == 0
        argv = fronted[0]["argv"]
        position = argv.index("--models")
        assert argv[position:position + 3] == ["--models", "A", "B"]

    def test_an_unknown_flag_is_the_fronted_parsers_error_before_any_write(self, repo, fronted):
        with pytest.raises(SystemExit) as exc:
            _review("--revs", "last 2 commits", "--no-such-flag")
        assert exc.value.code == 2
        assert not (repo / FOLDER).exists()

    @pytest.mark.parametrize("mode, tokens", [
        ("review-commits", ["--revs", "HEAD", "--nosuch", "value"]),
        ("review-commits", ["--revs", "HEAD", "--nosuch", "a", "b"]),
        ("review-commits", ["--revs", "HEAD", "--nosuch", "--focus", "x", "last", "2"]),
        # apply-code-fixes' parser has no --models or --quick: unknown there.
        ("apply-commit-fixes", ["--revs", "HEAD", "--models", "cursor-agent:auto"]),
        ("apply-commit-fixes", ["--revs", "HEAD", "--models", "A", "B..C"]),
        ("apply-commit-fixes", ["--revs", "HEAD", "--quick", "-2"]),
    ])
    def test_an_unknown_flags_value_is_not_taken_for_a_range(self, repo, fronted, capsys,
                                                             mode, tokens):
        """Review fix 3, A2: the word after an unknown flag may be its value,
        so it keeps the fronted parser's unrecognized-arguments error, never
        a ``[RANGE_AFTER_FLAGS]`` naming it with a wrong ``Run:`` command."""
        with pytest.raises(SystemExit) as exc:
            _door(mode, *tokens)
        out, err = capsys.readouterr()
        assert exc.value.code == 2
        assert "[RANGE_AFTER_FLAGS]" not in out
        leftovers = [t for t in tokens[2:] if t not in ("--focus", "x")]
        assert f"unrecognized arguments: {' '.join(leftovers)}" in err
        assert not (repo / FOLDER).exists()
        assert fronted == []

    def test_a_range_word_after_a_known_flags_value_is_still_moved(self, repo, fronted,
                                                                    capsys):
        assert _review("--revs", "HEAD", "--timeout", "30", "last", "2") == 3
        assert "[RANGE_AFTER_FLAGS] last 2" in capsys.readouterr().out

    def test_through_the_dispatcher(self, repo, fronted, capsys):
        code, out, _err = _dispatch(["review-commits", "--quick", "-2", *HOST], capsys)
        assert code == 3
        assert "[RANGE_AFTER_FLAGS] -2" in out
        assert not (repo / FOLDER).exists()


# ===========================================================================
# Step 1's refusals
# ===========================================================================


class TestStepOne:
    def test_a_deleted_briefs_path_is_not_found(self, repo, fronted, capsys):
        missing = f"{FOLDER}/0123456789ab-fedcba987654.md"
        assert _review("--revs", missing) == 3
        out = capsys.readouterr().out
        assert out.startswith(f"[COMMIT_REVIEW_NOT_FOUND] {missing}: no commit brief at {missing}")
        assert not (repo / FOLDER / "0123456789ab-fedcba987654").exists()

    def test_a_brief_with_no_recorded_range_is_not_found(self, repo, fronted, capsys):
        brief = repo / FOLDER / "0123456789ab-fedcba987654.md"
        brief.parent.mkdir(parents=True)
        brief.write_text(f"{BRIEF_MARKER}\n{BRIEF_END_MARKER}\n", encoding="utf-8")
        assert _review("--plan-file", f"{FOLDER}/{brief.name}") == 3
        assert "no commit range recorded for this brief" in capsys.readouterr().out
        assert not output_dir_for_brief(brief).exists()

    @pytest.mark.parametrize("root_range", [True, False])
    def test_a_brief_left_without_a_record_is_re_entered_by_its_name(self, repo, fronted,
                                                                     capsys, root_range):
        """Review fix 3, A5: a detached first launch refused after the brief's
        exclusive create leaves a brief with no ``commit_range``; its printed
        path (a root range's only pinned spelling) resolves its slug strictly
        and takes the first resolution."""
        head = _sha(repo, "HEAD")
        base = None if root_range else _sha(repo, "HEAD~2")
        brief = _brief_of(repo, base, head)
        brief.parent.mkdir(parents=True)
        brief.write_text(f"{BRIEF_MARKER}\n{BRIEF_END_MARKER}\n", encoding="utf-8")
        assert _review("--revs", f"{FOLDER}/{brief.name}") == 0
        out = capsys.readouterr().out
        assert f"[COMMIT_BRIEF] {FOLDER}/{brief.name}" in out
        record = _state(brief)["commit_range"]
        assert (record["base"], record["head"]) == (base, head)
        assert len(record["commits"]) == (3 if root_range else 2)
        assert fronted[0]["argv"][fronted[0]["argv"].index("--head-ref") + 1] == head

    def test_an_a_dot_dot_b_whose_b_ends_in_md_is_a_range(self, repo, fronted):
        """Review fix 3, A4: a tag ``v1.md`` on the B side; the file-hint path
        tried the word as one revision only and refused it."""
        _git(repo, "tag", "v1.md", "HEAD~1")
        assert _review("--revs", "HEAD~2..v1.md") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD~1"))
        assert brief.is_file()
        assert _state(brief)["commit_range"]["head"] == _sha(repo, "HEAD~1")

    def test_a_dot_dot_md_word_that_names_nothing_keeps_the_file_message(self, repo, fronted,
                                                                         capsys):
        assert _review("--revs", "HEAD~2..nosuch.md") == 3
        out = capsys.readouterr().out
        assert out.startswith("[RANGE_UNRECOGNIZED] HEAD~2..nosuch.md: no such file or revision")
        assert not (repo / FOLDER).exists()

    def test_a_plan_is_never_a_brief(self, repo, fronted, capsys):
        assert _review("--revs", "docs/x.md") == 3
        out = capsys.readouterr().out
        assert "[RANGE_UNRECOGNIZED] docs/x.md: a file, but not a commit brief" in out
        assert "--context-plan docs/x.md" in out and "--review-code docs/x.md" in out
        assert not (repo / FOLDER).exists()

    def test_a_no_argument_apply_on_an_unborn_head_is_exit_two(self, tmp_path, monkeypatch,
                                                              fronted, capsys):
        root = _init(tmp_path / "empty")
        monkeypatch.chdir(root)
        assert _door("apply-commit-fixes") == 2
        assert "[NOTHING_TO_REVIEW] no commits yet on main" in capsys.readouterr().out

    def test_a_no_argument_apply_opens_the_selected_review(self, repo, fronted, seams,
                                                         monkeypatch, capsys):
        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        _complete_review(brief)
        line = "[COMMIT_REVIEW_SELECTED] x completed today, ends at HEAD"
        monkeypatch.setattr(seams.apply, "select_review",
                            lambda *, cwd: Selection(brief, line, {"completed_at": "t", "behind": 0}))
        fronted.clear()
        capsys.readouterr()
        assert _door("apply-commit-fixes") == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert lines[lines.index("[COMMIT_RANGE_END]") - 1] == "  " + line
        assert fronted[0]["argv"] == ["--plan-file", f"{FOLDER}/{brief.name}"]


# ===========================================================================
# §2.5: the drift check
# ===========================================================================


class TestDrift:
    def test_a_drifted_scope_file_is_in_the_block_and_the_record(self, repo, fronted, capsys):
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        head = _sha(repo, "HEAD")
        assert _review("--revs", "last 2 commits") == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        drift = f"  [SCOPE_DRIFT] 1 of 2 reviewed files differ on disk from {head[:7]}: a.py"
        assert drift in lines
        assert lines.index(drift) < lines.index("[COMMIT_RANGE_END]")
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), head)
        assert _state(brief)["commit_range"]["drifted"] == ["a.py"]

    def test_files_narrow_the_drift_scope(self, repo, fronted, capsys):
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        assert _review("--revs", "last 2 commits", "--files", "b.py") == 0
        assert "[SCOPE_DRIFT]" not in capsys.readouterr().out
        argv = fronted[0]["argv"]
        assert argv[argv.index("--files") + 1] == "b.py"

    def test_under_detach_the_drift_is_staged(self, repo, fronted, capsys):
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        staged = {}

        def review(argv):
            stage = cli_common.flag_value(argv, "--commit-stage")
            staged.update(json.loads(Path(stage).read_text(encoding="utf-8")))
            return 0

        code_review_orchestrator.main = review
        assert _review("--revs", "last 2 commits", "--detach", "--yes", "--exec-subagents") == 0
        assert "[SCOPE_DRIFT] 1 of 2" in capsys.readouterr().out
        assert staged["fields"]["drifted"] == ["a.py"]


# ===========================================================================
# The configured folder (§2.4, "Relative to the repository root")
# ===========================================================================


@pytest.mark.config_override
class TestConfiguredFolder:
    def test_a_project_value_from_a_subdirectory(self, repo, fronted, monkeypatch, capsys):
        (repo / ".multi-llm").mkdir()
        (repo / ".multi-llm" / "providers.yaml").write_text(
            "code_review:\n  commit_reviews_dir: reviews/commits\n", encoding="utf-8")
        monkeypatch.chdir(repo / "src")
        assert _review("--revs", "last 2 commits") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        brief = repo / "reviews" / "commits" / f"{base[:12]}-{head[:12]}.md"
        assert brief.is_file()
        assert not (repo / "src" / "reviews").exists()
        capsys.readouterr()
        monkeypatch.chdir(repo)
        assert _review("--revs", "last 2 commits") == 0
        out = capsys.readouterr().out
        assert f"[COMMIT_BRIEF] reviews/commits/{brief.name}" in out
        assert "[COMMIT_REVIEW_RESTARTED]" not in out

    def test_an_env_layer_value_is_relative_to_the_root(self, repo, fronted, tmp_path, monkeypatch):
        config = tmp_path / "elsewhere" / "providers.yaml"
        config.parent.mkdir()
        config.write_text("code_review:\n  commit_reviews_dir: env/briefs\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        monkeypatch.chdir(repo / "src")
        assert _review("--revs", "last 2 commits") == 0
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        assert (repo / "env" / "briefs" / f"{base[:12]}-{head[:12]}.md").is_file()
        assert not (config.parent / "env").exists()
        assert not (repo / "src" / "env").exists()


# ===========================================================================
# The folder ignores itself (§2.4)
# ===========================================================================


class TestGitIgnore:
    def test_a_new_folder_ignores_itself_and_git_status_is_unchanged(self, repo, fronted):
        (repo / "plans").mkdir()
        (repo / "plans" / "feature.md").write_text("# f\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "plans")
        before = _git(repo, "status", "--porcelain")
        assert _review("--revs", "last 2 commits") == 0
        assert _git(repo, "status", "--porcelain") == before
        assert (repo / FOLDER / ".gitignore").read_text(encoding="utf-8") == "*\n"
        assert not (repo / "plans" / ".gitignore").exists()

    def test_an_existing_unignored_folder_warns_and_is_left_alone(self, repo, fronted, capsys):
        (repo / FOLDER).mkdir(parents=True)
        assert _review("--revs", "last 2 commits") == 0
        out = capsys.readouterr().out
        assert out.count("[COMMIT_REVIEWS_NOT_IGNORED]") == 1
        assert not (repo / FOLDER / ".gitignore").exists()

    def test_a_deleted_gitignore_is_not_restored(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits") == 0
        (repo / FOLDER / ".gitignore").unlink()
        capsys.readouterr()
        assert _review("--revs", "last 2 commits") == 0
        assert "[COMMIT_REVIEWS_NOT_IGNORED]" in capsys.readouterr().out
        assert not (repo / FOLDER / ".gitignore").exists()


# ===========================================================================
# The stage and the detach hook
# ===========================================================================


class TestStage:
    def _staged(self, repo, *argv):
        kept = {}

        def review(argv_):
            stage = cli_common.flag_value(argv_, "--commit-stage")
            shutil.copy(stage, stage + ".keep")
            kept["stage"] = Path(stage)
            return 0

        code_review_orchestrator.main = review
        assert _review(*argv, "--detach", "--yes", "--exec-subagents") == 0
        os.replace(str(kept["stage"]) + ".keep", kept["stage"])
        return kept["stage"]

    def test_apply_stage_publishes_a_new_briefs_record(self, repo, fronted):
        stage = self._staged(repo, "--revs", "last 2 commits", "--focus", "X")
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        assert not (output_dir_for_brief(brief) / "state.json").exists()
        commit_review.apply_stage(stage)
        assert _state(brief)["commit_range"]["focus"] == "X"
        assert not stage.exists()
        commit_review.apply_stage(stage)   # gone: a no-op

    def test_apply_stage_restarts_with_the_archive_and_the_header(self, repo, fronted, seams,
                                                                  capsys):
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        phase = output_dir_for_brief(brief) / "code-review"
        (phase / "B.json").write_text("[]", encoding="utf-8")
        before_bytes = brief.read_bytes()
        stage = self._staged(repo, "--revs", "last 2 commits", "--focus", "Y")
        out = capsys.readouterr().out
        # No worker consumed it before the launcher settled: nothing restarted
        # yet, so nothing is announced.
        assert "[COMMIT_REVIEW_RESTARTED]" not in out
        # The launcher moved nothing and rewrote nothing.
        assert (phase / "B.json").exists() and brief.read_bytes() == before_bytes
        assert _state(brief)["commit_range"]["focus"] == "X"
        commit_review.apply_stage(stage)
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_ARCHIVED]" in out
        assert not (phase / "B.json").exists()
        assert _state(brief)["commit_range"]["focus"] == "Y"
        assert "Y" in seams.brief.split_brief(brief.read_text(encoding="utf-8"))[0]
        assert not stage.exists()

    def test_a_detached_restart_is_announced_only_once_a_worker_consumed_its_stage(
            self, repo, fronted, seams, monkeypatch, capsys):
        """A launch that detach.launch refuses after the stage is written
        restarted nothing, so it prints neither [COMMIT_REVIEW_RESTARTED] nor
        [COMMIT_DECISIONS_DROPPED]; one whose worker consumed the stage prints
        both after the launch, in the foreground's order."""
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        phase = output_dir_for_brief(_brief_of(repo, _sha(repo, "HEAD~2"),
                                               _sha(repo, "HEAD"))) / "code-review"
        (phase / "B.json").write_text("[]", encoding="utf-8")
        monkeypatch.setattr(seams.brief, "count_decisions", lambda phase_dir: _Counts(skips=1))
        detached = ("--revs", "last 2 commits", "--focus", "Y", "--detach", "--yes",
                    "--exec-subagents")
        capsys.readouterr()

        code_review_orchestrator.main = lambda argv: 3       # refused at the launch
        assert _review(*detached) == 3
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED]" not in out
        assert "[COMMIT_DECISIONS_DROPPED]" not in out
        assert (phase / "B.json").exists() and _stage_files(repo) == []

        def launched(argv):
            commit_review.apply_stage(cli_common.flag_value(argv, "--commit-stage"))
            print("[launched]")
            return 0

        code_review_orchestrator.main = launched
        assert _review(*detached) == 0
        before, _sep, after = capsys.readouterr().out.partition("[launched]")
        assert "[COMMIT_REVIEW_RESTARTED]" not in before
        assert after.index("[COMMIT_DECISIONS_DROPPED] 1 skips") < after.index(
            "[COMMIT_REVIEW_RESTARTED] focus")
        assert not (phase / "B.json").exists()

    def test_the_staged_restart_pops_the_completion_when_the_worker_applies_it(
            self, repo, fronted):
        """Review fix 3, item 3 under ``--detach``: the launcher writes no
        state (a refused launch keeps the completed review), and the worker's
        ``apply_stage`` pops the completion with the archive."""
        assert _review("--revs", "last 2 commits", "--focus", "X") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        _complete_review(brief)
        stage = self._staged(repo, "--revs", "last 2 commits", "--focus", "Y")
        assert _state(brief)["phases_completed"].get("code-review")
        commit_review.apply_stage(stage)
        assert "code-review" not in _state(brief)["phases_completed"]

    def test_the_worker_applies_the_stage_right_after_its_lock(self, repo, fronted):
        stage = self._staged(repo, "--revs", "last 2 commits", "--focus", "X")
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        handle = detach.RunHandle(str(output_dir_for_brief(brief) / "code-review"), run_id="r1",
                                  mode="review-code", plan_file=str(brief), detached=False,
                                  commit_stage=str(stage))
        handle._acquire_lock()
        try:
            handle._apply_commit_stage()
        finally:
            handle._lock.release()
        assert _state(brief)["commit_range"]["focus"] == "X"
        assert not stage.exists()

    def test_a_worker_that_loses_the_lock_applies_nothing(self, repo, fronted):
        stage = self._staged(repo, "--revs", "last 2 commits", "--focus", "X")
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        phase = output_dir_for_brief(brief) / "code-review"
        holder = process_control.PhaseLock(detach.lock_path(str(phase)))
        holder.acquire({"run_id": "other", "pid": os.getpid(),
                        "process_start_time": process_control.process_start_time(os.getpid())})
        try:
            handle = detach.RunHandle(str(phase), run_id="r1", mode="review-code",
                                      plan_file=str(brief), detached=False,
                                      commit_stage=str(stage))
            with pytest.raises(detach.DetachError):
                handle.start()
        finally:
            holder.release()
        assert stage.exists()
        assert not (output_dir_for_brief(brief) / "state.json").exists()

    def test_begin_worker_run_hands_the_stage_over(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(detach.RunHandle, "start",
                            lambda self: seen.setdefault("stage", self.commit_stage))
        args = types.SimpleNamespace(internal_run_id="r1", host=None,
                                     commit_stage=str(tmp_path / ".commit-stage-x.json"))
        detach.begin_worker_run(phase_dir=str(tmp_path), mode="review-code",
                                plan_file=str(tmp_path / "b.md"), args=args)
        assert seen["stage"] == str(tmp_path / ".commit-stage-x.json")

    def test_the_fronted_parsers_take_the_hidden_stage_flag(self, repo):
        for parser in (commit_review._fronted_parser("review-commits"),
                       commit_review._fronted_parser("apply-commit-fixes"),
                       commit_review._fronted_parser("improve-commits")):
            assert "--commit-stage" in parser._option_string_actions
            assert "--commit-stage" not in parser.format_help()
        assert "--commit-stage" not in improve_workflow.build_parser("improve-code")._option_string_actions


# ===========================================================================
# util commit-brief (§2.8)
# ===========================================================================


class TestUtilCommitBrief:
    def test_it_prepares_and_hands_back_the_remaining_flags(self, repo, capsys):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        assert commit_review.commit_brief_main(
            ["last", "2", "commits", "--focus", "X", "--quick", "--models", "A", "B"]) == 0
        out = capsys.readouterr().out
        brief = _brief_of(repo, base, head)
        assert "[COMMIT_RANGE]" in out
        assert f"[COMMIT_BRIEF] {FOLDER}/{brief.name}" in out
        assert f"[COMMIT_RANGE_PINNED] {base}..{head}" in out
        assert "[COMMIT_REMAINING_FLAGS] --quick --models A B" in out
        assert _state(brief)["commit_range"]["focus"] == "X"
        assert process_control.lock_is_free(
            detach.lock_path(str(output_dir_for_brief(brief) / "code-review")))

    def test_it_writes_the_same_brief_bytes_as_the_executor_front_door(self, repo, fronted, capsys):
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        native = brief.read_bytes()
        shutil.rmtree(repo / FOLDER)
        assert _door("improve-commits", "--revs", "last 2 commits", "--focus", "X") == 0
        assert brief.read_bytes() == native

    def test_a_live_review_refuses_it(self, repo, fronted, capsys):
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        before = brief.read_bytes(), (output_dir_for_brief(brief) / "state.json").read_bytes()
        lock = _live_run(output_dir_for_brief(brief) / "code-review", "run-1", "review-code")
        try:
            assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "Y"]) == 3
        finally:
            lock.release()
        assert (brief.read_bytes(),
                (output_dir_for_brief(brief) / "state.json").read_bytes()) == before

    def test_a_root_range_pins_as_root(self, tmp_path, monkeypatch, capsys):
        root = _init(tmp_path / "one")
        head = _commit(root, "a.py", "a = 1\n", "only")
        monkeypatch.chdir(root)
        assert commit_review.commit_brief_main([]) == 0
        out = capsys.readouterr().out
        assert f"[COMMIT_RANGE_PINNED] root..{head}" in out
        assert f"[COMMIT_BRIEF] {FOLDER}/root-{head[:12]}.md" in out


# ===========================================================================
# --help
# ===========================================================================


class TestHelp:
    @pytest.mark.parametrize("mode", commit_review.COMMIT_MODES)
    def test_each_door_has_a_useful_help(self, mode, capsys):
        with pytest.raises(SystemExit) as exc:
            commit_review.parse_args(mode, ["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--focus" in out and "--context-plan" in out and "Examples:" in out
        assert "--revs" not in out and "--commit-stage" not in out

    def test_front_door_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            commit_review.front_door("review-commits", ["--help"])
        assert exc.value.code == 0
        assert "range goes right after the mode word" in capsys.readouterr().out

    def test_the_util_command_has_a_module_main(self):
        assert commit_review.main is not None
        with pytest.raises(SystemExit):
            commit_review.main(["--help"])
