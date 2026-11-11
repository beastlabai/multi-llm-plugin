"""A stable git repository shape, built once per session and copied into each test.

On the Windows CI runner every git process costs tens of milliseconds or more,
so a fixture that runs ``git init``, a few ``git config`` calls and an
``add``/``commit``/``rev-parse`` round per commit (13 processes for a
three-commit repository) made the commit-mode files 12-25x slower there than
on Ubuntu. ``copy_of`` builds a shape once per pytest session, which under
xdist means once per worker, and gives each test a ``shutil.copytree`` copy at
the path its fixture always used. A copy costs at most one git process: the
index refresh below.

What keeps a copy the repository the fixture used to build:

* **Built the same way.** ``build`` is the fixture's own construction, run in
  the environment of the first test that asks for the shape. The template is
  keyed by the shape's name plus everything in the environment
  git reads while building one (every ``GIT_*`` variable, the home and config
  directories that locate the global ignore and attributes files, ``EMAIL``,
  and the git binary on ``PATH``). A test that sets an identity, a date or a
  config file before asking therefore gets a template built under that
  setting, never one built without it. System configuration (Git for
  Windows keeps ``core.autocrlf`` there) is read by the build exactly as by
  the old per-test construction.
* **Self-contained.** ``.git`` must be a plain directory with no
  ``alternates`` and no ``commondir`` (a linked worktree's), its own config
  must set no ``core.worktree``, hooks path, remote, include or submodule,
  no file outside the object store may name the template's own path, and
  every entry must be a regular file or a directory with a single link. A
  shape that breaks this is refused when it is built, because its copies
  would point back at the template or share files with it.
* **Fresh stat data.** A copied index still holds the template's inode and
  ctime, so every entry looks stat-dirty. Porcelain (``git status``,
  ``git diff``) re-reads the files and reports nothing, but plumbing that
  trusts the index (``git diff-files``, ``git diff-index HEAD``) reports every
  file modified, and the helpers set ``GIT_OPTIONAL_LOCKS=0`` so a status
  never writes the refreshed data back. One ``git update-index -q --refresh``
  per copy records the copy's own stat data, as the fixture's last
  ``git add`` did; ``-q`` keeps a deliberately dirty shape's edits as they are.
  Files are copied without their timestamps, so a copy's files are as new as
  a fresh build's. Git compares stat times in whole seconds and re-reads an
  entry no older than the index, which is how a fresh build sees a same-size
  edit made in the second it was built; a copy keeping a minute-old
  template's mtimes would miss such an edit whose mtime was put back.
* **Only a pristine seed.** The template holds ``.git`` and the worktree as
  the build left them. Workflow state, backups, configuration files and mock
  launchers carry absolute paths, so the caller creates them after the copy,
  at the destination.

Commits in one session share their SHAs across copies. A repository built
afresh already shared them with every test that built it in the same second,
so no test could rely on unique SHAs; anything keyed by SHA lives under the
test's own ``tmp_path`` or the per-test state and cache directories
``conftest.py`` sets. Commit dates are the template's build time, not the
test's: a shape whose tests resolve clock-relative revisions (``today``,
``last 2 hours``) against the real clock must pin its dates or stay built per
test. ``tests/test_git_template.py`` checks all of the above.
"""

from __future__ import annotations

import copy
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Callable, Dict, Hashable, Mapping, Tuple

from tests.support.gitrepo import git_env as hermetic_git_env

#: The non-``GIT_*`` variables git reads while building a shape: the home
#: directory (and Windows' spellings of it) and ``XDG_CONFIG_HOME`` locate the
#: default ``core.excludesFile`` and ``core.attributesFile`` that ``git add -A``
#: honours even with ``GIT_CONFIG_GLOBAL`` set; ``EMAIL`` is git's last
#: identity fallback.
_ENVIRONMENT_NAMES = ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "XDG_CONFIG_HOME",
                      "EMAIL")

#: Local config keys (lower-case, as ``git config --name-only`` prints them)
#: that would tie a copy to a path outside it or to another repository.
_REFUSED_KEYS = ("core.worktree", "core.hookspath", "extensions.worktreeconfig")
_REFUSED_PREFIXES = ("remote.", "include.", "includeif.", "submodule.")

#: Built templates, by session and shape. Module state rather than a session
#: fixture so the fixtures that copy a shape need no extra fixture imported
#: next to them; the session's base temporary directory in the key keeps two
#: sessions in one process (an in-process pytester run) apart.
_TEMPLATES: Dict[Tuple[Hashable, ...], Tuple[Path, object]] = {}


class TemplateRefused(AssertionError):
    """A shape that cannot be copied faithfully; build it per test instead."""


def _environment_key() -> Tuple[Hashable, ...]:
    relevant = sorted((name, value) for name, value in os.environ.items()
                      if name.upper().startswith("GIT_") or name.upper() in _ENVIRONMENT_NAMES)
    return (shutil.which("git"), tuple(relevant))


def _run_git(root: Path, env: Mapping[str, str], *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(root), env=dict(env), check=True,
                          capture_output=True, text=True).stdout


def _check_self_contained(root: Path, env: Mapping[str, str]) -> None:
    """Raise :class:`TemplateRefused` unless every copy of *root* stands alone."""
    dot_git = root / ".git"
    problems = []
    if dot_git.is_symlink() or not dot_git.is_dir():
        raise TemplateRefused(f"{dot_git} is not a directory (a linked worktree or a gitfile)")
    for rel in ("objects/info/alternates", "objects/info/http-alternates", "commondir"):
        if (dot_git / rel).exists():
            problems.append(f".git/{rel} exists")
    names = _run_git(root, env, "config", "--local", "--name-only", "--list").split()
    for name in names:
        lowered = name.lower()
        if lowered in _REFUSED_KEYS or lowered.startswith(_REFUSED_PREFIXES):
            problems.append(f"its config sets {name}")
    spellings = set()
    for spelling in (str(root), os.path.realpath(root)):
        spellings |= {os.fsencode(spelling), os.fsencode(spelling.replace(os.sep, "/"))}
    objects = dot_git / "objects"
    for directory, subdirs, files in os.walk(root):
        for name in subdirs + files:
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                problems.append(f"{path.relative_to(root)} is not a regular file")
            elif info.st_nlink != 1:
                problems.append(f"{path.relative_to(root)} has {info.st_nlink} links")
            elif dot_git in path.parents and objects not in path.parents:
                data = path.read_bytes()
                if any(spelling in data for spelling in spellings):
                    problems.append(f"{path.relative_to(root)} names the template's path")
    if problems:
        raise TemplateRefused(f"cannot template {root}: " + "; ".join(problems))


def copy_of(tmp_path_factory, name: str, build: Callable[[Path], object], dest, *,
            env: Callable[[], Mapping[str, str]] = hermetic_git_env) -> Tuple[Path, object]:
    """A copy of the shape *name* at *dest* (absent, or an empty directory such
    as the test's cwd, which Windows cannot remove): its real path, and a deep
    copy of what *build* returned when it built the template (a test may
    extend a list of SHAs without the next test seeing it).

    *build(path)* makes the shape at *path* (a fresh, not-yet-existing
    directory) the way the fixture always did, and may return values the
    fixture hands on, such as the SHAs it committed. It runs once per session
    per *name* and git environment, so each shape needs its own *name*, and a
    shape with inputs (plan text, say) needs them in it. *env* is the
    environment the shape's own git calls use, for the self-containment check
    and the copy's index refresh: the hermetic ``tests.support.gitrepo``
    environment by default, ``os.environ`` as the test left it
    (``lambda: os.environ``) for a shape built by plain
    ``subprocess.run(["git", ...])``.
    """
    key = (str(tmp_path_factory.getbasetemp()), name, _environment_key())
    entry = _TEMPLATES.get(key)
    if entry is None or not entry[0].is_dir():
        template = tmp_path_factory.mktemp(f"git-template-{name}") / "repo"
        built = build(template)
        _check_self_contained(template, env())
        entry = _TEMPLATES[key] = (template, built)
    template, built = entry
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty; a template is only copied into a new "
                              "or empty directory")
    shutil.copytree(template, dest, copy_function=shutil.copy, dirs_exist_ok=True)
    root = Path(os.path.realpath(dest))
    if (root / ".git" / "index").is_file():
        _run_git(root, env(), "update-index", "-q", "--refresh")
    return root, copy.deepcopy(built)
