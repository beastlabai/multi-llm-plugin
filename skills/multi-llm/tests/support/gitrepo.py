"""Hermetic git for tests: the developer's ``~/.gitconfig`` never reaches a test repository.

``git(cwd, *args)`` runs git with ``GIT_CONFIG_GLOBAL`` pointing at
``tests/support/gitconfig`` (identity ``t <t@example.invalid>``, no commit or
tag signing, ``init.defaultBranch=master``) and ``GIT_OPTIONAL_LOCKS=0``.
Those are defaults, not overrides: a repository's own ``git config`` and
``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*`` in the environment still win, as they
would over a user's global file. The system file is still read: Git for Windows
keeps ``core.autocrlf`` there, and the code under test runs with it too.

``init_repo`` makes a repository, ``commit_file`` writes, stages and commits one
file, and ``make_repo_fixture`` builds a ``repo`` fixture from them. Copies in
test files that differ on purpose stay where they are: the Codex jail rig's
own hooks path and identity, the package builds' environment identity, the
build-context check that runs git on the real repository.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pytest

#: The global configuration every call reads instead of the developer's.
GITCONFIG = Path(__file__).with_name("gitconfig")


def git_env(base: Optional[dict] = None) -> dict:
    """*base* (default ``os.environ``) with the hermetic git settings."""
    env = dict(os.environ if base is None else base)
    env["GIT_CONFIG_GLOBAL"] = str(GITCONFIG)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def git(cwd, *args: str, check: bool = True, text_out: bool = True):
    """``git *args`` in *cwd*: its stdout when *text_out*, else the CompletedProcess."""
    done = subprocess.run(["git", *args], cwd=str(cwd), env=git_env(), check=check,
                          capture_output=True, text=True)
    return done.stdout if text_out else done


def commit_file(root, rel: str, text: str, message: str) -> str:
    """Write *text* to *rel* under *root*, commit everything, return the new HEAD."""
    target = Path(root) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", message)
    return git(root, "rev-parse", "HEAD").strip()


def init_repo(path, *, branch: str = "master", config: Optional[Dict[str, str]] = None,
              commits: Iterable[Tuple[str, str, str]] = ()) -> Path:
    """A new repository at *path* (made with its parents) on *branch*, with
    *config* written to its own ``.git/config`` (which the code under test reads
    too, unlike the hermetic defaults) and one commit per ``(rel, text, message)``
    in *commits*; returns its real path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "--quiet", "-b", branch)
    root = Path(os.path.realpath(path))
    for key, value in (config or {}).items():
        git(root, "config", key, value)
    for rel, text, message in commits:
        commit_file(root, rel, text, message)
    return root


def make_repo_fixture(*, branch: str = "master", chdir: bool = True,
                      subdir: Optional[str] = "repo", name: str = "repo"):
    """A function-scoped fixture named *name*: ``init_repo(tmp_path / subdir)``,
    with the cwd moved there when *chdir*; its value is the repository's real path.

    Bind it to another name (``repo_fixture = make_repo_fixture()``), so a test's
    ``repo`` parameter does not shadow a module-level ``repo``."""

    def fixture(tmp_path, monkeypatch):
        root = init_repo(tmp_path / subdir if subdir else tmp_path, branch=branch)
        if chdir:
            monkeypatch.chdir(root)
        return root

    fixture.__name__ = fixture.__qualname__ = f"{name}_fixture"
    return pytest.fixture(name=name)(fixture)
