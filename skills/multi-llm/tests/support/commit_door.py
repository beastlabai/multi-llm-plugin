"""The commit modes' front door: a three-commit repository and the fronted mains, recorded.

``_door`` runs a commit mode's front door, ``_review`` / ``_complete_review``
lay out a review's record, ``_brief_of`` names a range's brief under ``FOLDER``
and ``_sha`` resolves a revision. The repository is built with
``tests.support.gitrepo``.

Fixtures, not registered suite-wide (``repo`` belongs to
``tests.support.gate_fixture`` there), so import the function name:
``door_seams`` provides the autouse ``seams`` (stand-ins for the brief and
diff seams), ``door_repo`` provides ``repo`` (c1 a.py+b.py+docs, c2 b.py, c3
a.py; cwd at the root) and ``door_fronted`` provides ``fronted`` (each fronted
call's argv and invocation).
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import pytest

import apply_code_fixes_orchestrator
import code_review_orchestrator
import utils
from utils import cli_common, commit_review, git_utils, improve_workflow
from utils.commit_review import (
    BRIEF_END_MARKER,
    BRIEF_MARKER,
    CommitReviewError,
    output_dir_for_brief,
)
from utils.state_manager import StateManager

from tests.support.gitrepo import commit_file as _commit, git as _git, init_repo

#: The front door's repositories: on ``main``, with an identity in their own
#: config, which the commit modes under test read too.
_init = partial(init_repo, branch="main", config={
    "user.email": "test@example.com", "user.name": "Test",
    "commit.gpgsign": "false"})


FOLDER = "plans/commit-reviews"


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
    module.RESTART_KEEP = (".lock", ".status.lock", ".status.json", "orchestrator-run.log",
                           ".views.lock")
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


@pytest.fixture(name="seams", autouse=True)
def door_seams(monkeypatch):
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


@pytest.fixture(name="repo")
def door_repo(tmp_path, monkeypatch):
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


@pytest.fixture(name="fronted")
def door_fronted(monkeypatch):
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
