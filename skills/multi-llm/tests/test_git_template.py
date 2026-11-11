"""``tests.support.git_template``: a copied template is the repository its fixture built.

The fixtures that copy a template instead of building per test (the commit
front door's three commits first) rely on four things checked here: a copy
equals a fresh build in branch, config, history, index, bytes and modes; the
status and diff reads the commit modes use see a copy's clean tree, unstaged,
staged and deleted files exactly as in a fresh build; copies share nothing
with each other or with the template; and a shape that could not stand alone
is refused when it is built.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from utils import git_utils

from tests.support import commit_door, git_template
from tests.support.git_template import TemplateRefused, copy_of
from tests.support.gitrepo import commit_file, git, init_repo


def _door_copy(tmp_path_factory, dest: Path) -> Path:
    """What ``commit_door.door_repo`` hands a test."""
    root, _ = copy_of(tmp_path_factory, "commit-door", commit_door._build_door_repo, dest)
    return root


def _tree_files(root: Path) -> Dict[str, Tuple[bytes, int]]:
    """Every worktree file outside ``.git``: its bytes and permission bits."""
    found = {}
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [name for name in subdirs if name != ".git"]
        for name in files:
            path = Path(directory) / name
            found[path.relative_to(root).as_posix()] = (
                path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
    return found


def _git_entries(root: Path) -> List[str]:
    """``.git``'s files outside the object store (whose commit objects differ
    whenever two builds fall in different seconds)."""
    dot_git = root / ".git"
    return sorted(path.relative_to(dot_git).as_posix() for path in dot_git.rglob("*")
                  if path.is_file() and "objects" not in path.relative_to(dot_git).parts)


class TestACopyIsTheFixturesRepository:
    def test_it_matches_a_fresh_build_in_everything_but_commit_times(self, tmp_path,
                                                                     tmp_path_factory):
        fresh = tmp_path / "fresh"
        commit_door._build_door_repo(fresh)
        copy = _door_copy(tmp_path_factory, tmp_path / "copy")
        assert copy == Path(os.path.realpath(tmp_path / "copy"))
        # Commit ids carry the commit times, so history is compared by its
        # trees, parent counts, identities and messages.
        for args in (["symbolic-ref", "HEAD"],
                     ["config", "--local", "--list"],
                     ["log", "--format=%T %p%n%an <%ae> %cn <%ce>%n%B"],
                     ["ls-files", "--stage", "--eol"],
                     ["status", "--porcelain=v1", "--untracked-files=all"]):
            fresh_out, copy_out = git(fresh, *args), git(copy, *args)
            if args[0] == "log":
                fresh_out, copy_out = _parent_counts(fresh_out), _parent_counts(copy_out)
            assert copy_out == fresh_out, args
        assert _tree_files(copy) == _tree_files(fresh)
        assert _git_entries(copy) == _git_entries(fresh)

    def test_an_identity_override_gets_a_template_built_under_it(self, tmp_path,
                                                                 tmp_path_factory, monkeypatch):
        plain = _door_copy(tmp_path_factory, tmp_path / "plain")
        monkeypatch.setenv("GIT_AUTHOR_NAME", "someone else")
        overridden = _door_copy(tmp_path_factory, tmp_path / "overridden")
        assert git(plain, "log", "-1", "--format=%an").strip() == "Test"
        assert git(overridden, "log", "-1", "--format=%an").strip() == "someone else"


def _parent_counts(log: str) -> List[str]:
    """``%T %p`` lines with the abbreviated parents replaced by their count."""
    lines = []
    for line in log.splitlines():
        tree, _, parents = line.partition(" ")
        if len(tree) >= 40 and all(c in "0123456789abcdef" for c in tree):
            line = f"{tree} parents={len(parents.split())}"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# The reads the commit modes make, on a copy
# ---------------------------------------------------------------------------


def _reads(root: Path, head: str) -> Dict[str, object]:
    """What each status or diff read reports for *root*.

    ``diff-files`` and ``diff-index`` trust the stat data in the index on
    disk, so they come first: ``git diff`` writes a refreshed index back
    whatever ``GIT_OPTIONAL_LOCKS`` says, and would hide a copy whose index
    still held the template's stat data. The porcelain reads refresh in
    memory, and ``paths_changed_since`` is the commit modes' own change test."""
    reads: Dict[str, object] = {
        "diff-files": git(root, "diff-files", "--name-only").split(),
        "diff-index HEAD": git(root, "diff-index", "--name-only", head).split(),
    }
    tracked = [path for path in git(root, "ls-files", "-z").split("\0") if path]
    reads.update({
        "status": sorted(git(root, "status", "--porcelain=v1").splitlines()),
        "diff": git(root, "diff", "--name-only").split(),
        "diff --cached": git(root, "diff", "--cached", "--name-only").split(),
        "diff HEAD": git(root, "diff", "--name-only", head).split(),
        "paths_changed_since": sorted(git_utils.paths_changed_since(head, tracked,
                                                                    cwd=str(root))),
    })
    return reads


def _expect(status: str, path: str, *, staged: bool) -> Dict[str, object]:
    unstaged = [] if staged else [path]
    return {
        "status": [f"{status}  {path}"] if staged else [f" {status} {path}"],
        "diff": unstaged,
        "diff --cached": [path] if staged else [],
        "diff HEAD": [path],
        "diff-files": unstaged,
        "diff-index HEAD": [path],
        "paths_changed_since": [path],
    }


def _build_aged_door_repo(path: Path) -> None:
    """The door's shape with every file a day old, as a template built long
    before the copy would be if copies kept the template's timestamps."""
    commit_door._build_door_repo(path)
    day_ago = time.time() - 86400
    for found in path.rglob("*"):
        os.utime(found, (day_ago, day_ago), follow_symlinks=False)


class TestTheReadsSeeACopyAsTheySeeAFreshBuild:
    @pytest.fixture
    def copy(self, tmp_path, tmp_path_factory, monkeypatch):
        root = _door_copy(tmp_path_factory, tmp_path / "repo")
        monkeypatch.chdir(root)
        return root, git(root, "rev-parse", "HEAD").strip()

    def test_a_clean_copy_is_clean_to_every_read(self, copy):
        root, head = copy
        assert _reads(root, head) == {name: [] for name in _expect("M", "a.py", staged=False)}

    def test_a_same_size_edit_is_unstaged_to_every_read(self, copy):
        root, head = copy
        (root / "a.py").write_text("a = 9\n", encoding="utf-8")
        assert _reads(root, head) == _expect("M", "a.py", staged=False)

    @pytest.mark.skipif(os.name == "nt", reason="Git for Windows records no inode and takes "
                        "ctime from the creation time, so a same-size write with its mtime "
                        "put back is invisible to git in any repository, copied or built")
    @pytest.mark.parametrize("name,build", [
        ("commit-door", commit_door._build_door_repo),
        ("commit-door-aged", _build_aged_door_repo),
    ], ids=["this-sessions-template", "a-day-old-template"])
    def test_a_same_size_edit_with_its_mtime_put_back_is_still_seen(
            self, tmp_path, tmp_path_factory, monkeypatch, name, build):
        root, _ = copy_of(tmp_path_factory, name, build, tmp_path / "repo")
        monkeypatch.chdir(root)
        head = git(root, "rev-parse", "HEAD").strip()
        target = root / "a.py"
        before = target.stat()
        target.write_text("a = 9\n", encoding="utf-8")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = target.stat()
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        assert _reads(root, head) == _expect("M", "a.py", staged=False)

    def test_a_staged_edit_is_staged_to_every_read(self, copy):
        root, head = copy
        (root / "b.py").write_text("b = 7\n", encoding="utf-8")
        git(root, "add", "b.py")
        assert _reads(root, head) == _expect("M", "b.py", staged=True)

    def test_a_deletion_is_seen_by_every_read(self, copy):
        root, head = copy
        (root / "src" / "keep.txt").unlink()
        assert _reads(root, head) == _expect("D", "src/keep.txt", staged=False)


# ---------------------------------------------------------------------------
# Copies share nothing
# ---------------------------------------------------------------------------


def _build_returning_its_path(path: Path) -> Path:
    init_repo(path, branch="main", commits=[("a.py", "a = 1\n", "one"),
                                            ("b.py", "b = 1\n", "two")])
    return path


def _build_returning_a_list(path: Path) -> List[str]:
    init_repo(path, commits=[("a.py", "a = 1\n", "one")])
    return ["one"]


class TestCopiesAreIndependent:
    def test_a_commit_and_an_edit_in_one_copy_reach_neither_the_other_nor_the_template(
            self, tmp_path, tmp_path_factory):
        first, template = copy_of(tmp_path_factory, "independence", _build_returning_its_path,
                                  tmp_path / "first")
        second, again = copy_of(tmp_path_factory, "independence", _build_returning_its_path,
                                tmp_path / "second")
        assert again == template
        base = git(template, "rev-parse", "HEAD").strip()
        assert git(first, "rev-parse", "HEAD").strip() == base
        assert git(second, "rev-parse", "HEAD").strip() == base

        moved = commit_file(first, "a.py", "a = 2\n", "three")
        (first / "b.py").write_text("b = 99\n", encoding="utf-8")

        assert moved != base
        for other in (second, template):
            assert git(other, "rev-parse", "HEAD").strip() == base
            assert (other / "a.py").read_text(encoding="utf-8") == "a = 1\n"
            assert (other / "b.py").read_text(encoding="utf-8") == "b = 1\n"
            assert git(other, "status", "--porcelain").strip() == ""
            assert git(other, "cat-file", "-e", moved, check=False, text_out=False).returncode
        assert git(first, "status", "--porcelain").splitlines() == [" M b.py"]

    def test_what_build_returned_is_each_copys_own(self, tmp_path, tmp_path_factory):
        _, first = copy_of(tmp_path_factory, "returned-list", _build_returning_a_list,
                           tmp_path / "first")
        first.append("added by one test")
        _, second = copy_of(tmp_path_factory, "returned-list", _build_returning_a_list,
                            tmp_path / "second")
        assert second == ["one"]

    def test_no_copied_file_is_a_link_to_the_template(self, tmp_path, tmp_path_factory):
        copy, template = copy_of(tmp_path_factory, "independence", _build_returning_its_path,
                                 tmp_path / "copy")
        for path in template.rglob("*"):
            if path.is_file():
                twin = copy / path.relative_to(template)
                assert not os.path.samefile(path, twin), path
                assert twin.stat().st_nlink == 1, twin


# ---------------------------------------------------------------------------
# Shapes that cannot stand alone
# ---------------------------------------------------------------------------


def _with_remote(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    git(root, "remote", "add", "origin", str(path.parent / "elsewhere"))


def _with_worktree_setting(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    git(root, "config", "core.worktree", str(root))


def _with_alternates(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    (root / ".git" / "objects" / "info" / "alternates").write_text("/elsewhere/objects\n")


def _with_hook_naming_its_path(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    (root / ".git" / "hooks" / "pre-commit").write_text(f"#!/bin/sh\ncd {root}\n")


def _as_a_gitfile(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")


def _with_a_hardlink(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    os.link(root / "a.py", root / "twin.py")


def _with_a_symlink(path: Path) -> None:
    root = init_repo(path, commits=[("a.py", "a\n", "one")])
    os.symlink("a.py", root / "link.py")


def _can_symlink(where: Path) -> bool:
    probe = where / "symlink-probe"
    try:
        os.symlink("target", probe)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


class TestAShapeThatCannotStandAloneIsRefused:
    @pytest.mark.parametrize("build,reason", [
        (_with_remote, "remote.origin"),
        (_with_worktree_setting, "core.worktree"),
        (_with_alternates, "alternates"),
        (_with_hook_naming_its_path, "names the template's path"),
        (_as_a_gitfile, "not a directory"),
        (_with_a_hardlink, "has 2 links"),
    ])
    def test_it_is_refused_when_built(self, tmp_path, tmp_path_factory, build, reason):
        with pytest.raises(TemplateRefused, match=reason):
            copy_of(tmp_path_factory, f"refused-{build.__name__}", build, tmp_path / "copy")
        assert not (tmp_path / "copy").exists()
        assert not any(key[1] == f"refused-{build.__name__}" for key in git_template._TEMPLATES)

    def test_a_symlink_is_refused_when_built(self, tmp_path, tmp_path_factory):
        if not _can_symlink(tmp_path):
            pytest.skip("this account cannot create symlinks here")
        with pytest.raises(TemplateRefused, match="not a regular file"):
            copy_of(tmp_path_factory, "refused-symlink", _with_a_symlink, tmp_path / "copy")
