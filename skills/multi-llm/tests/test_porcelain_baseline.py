"""Plan §4.1 (option A): the porcelain check compares per-path baselines.

Every case runs real ``git`` in a temp repository: take a snapshot, change the
tree the way the case names, take another, compare.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from utils import review_orchestrator_base as rob


def _git(root, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    subprocess.run(["git", *args], cwd=root, check=True, env=env,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("o1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _changed(root, mutate, *, ignore=()):
    before = rob.porcelain_snapshot(str(root), ignore_prefixes=ignore)
    mutate(root)
    after = rob.porcelain_snapshot(str(root), ignore_prefixes=ignore)
    return rob.unexpected_writes(before, after, ignore_prefixes=ignore)


def _subcommand(argv) -> str:
    """The git subcommand of ``argv`` as ``git_utils.git_argv`` renders it
    (the hardening ``-c key=value`` pairs come first)."""
    index = 1
    while index < len(argv) and str(argv[index]).startswith("-"):
        index += 2 if argv[index] in ("-c", "-C") else 1
    return argv[index] if argv[0] == "git" and index < len(argv) else ""


class TestOptionA:

    def test_a_write_to_a_pre_dirty_file_is_seen(self, repo):
        """The known gap: its porcelain line stays ` M`, its blob id moves."""
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        assert _changed(repo, lambda r: (r / "tracked.txt").write_text(
            "dirtier\n", encoding="utf-8")) == ["tracked.txt"]

    def test_an_unchanged_pre_dirty_file_is_clean(self, repo):
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        (repo / "untracked.md").write_text("u\n", encoding="utf-8")
        assert _changed(repo, lambda r: None) == []

    def test_a_file_changed_inside_a_pre_existing_untracked_directory(self, repo):
        (repo / "newdir").mkdir()
        (repo / "newdir" / "a.md").write_text("a\n", encoding="utf-8")
        (repo / "newdir" / "b.md").write_text("b\n", encoding="utf-8")
        assert _changed(repo, lambda r: (r / "newdir" / "a.md").write_text(
            "changed\n", encoding="utf-8")) == ["newdir/a.md"]

    def test_a_pre_existing_untracked_file_deleted(self, repo):
        (repo / "untracked.md").write_text("u\n", encoding="utf-8")
        assert _changed(repo, lambda r: (r / "untracked.md").unlink()) == ["untracked.md"]

    def test_a_dirty_file_restored_to_clean(self, repo):
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        assert _changed(repo, lambda r: (r / "tracked.txt").write_text(
            "v1\n", encoding="utf-8")) == ["tracked.txt"]

    def test_a_mode_only_change_to_a_pre_dirty_file(self, repo):
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        assert _changed(repo, lambda r: os.chmod(r / "tracked.txt", 0o755)) == [
            "tracked.txt"]

    def test_a_retargeted_symlink(self, repo):
        os.symlink("tracked.txt", repo / "link")

        def retarget(r):
            os.unlink(r / "link")
            os.symlink("other.txt", r / "link")

        assert _changed(repo, retarget) == ["link"]

    def test_a_symlink_is_never_followed(self, repo, monkeypatch):
        os.symlink("tracked.txt", repo / "link")
        snap = rob.porcelain_snapshot(str(repo))
        assert snap.entries["link"].kind == "symlink"
        assert snap.entries["link"].blob == "symlink:tracked.txt"

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user (root can read anything)",
    )
    def test_an_unreadable_path_always_counts_as_changed(self, repo):
        (repo / "secret.md").write_text("s\n", encoding="utf-8")
        os.chmod(repo / "secret.md", 0)
        try:
            snap = rob.porcelain_snapshot(str(repo))
            assert snap.entries["secret.md"].kind == "unreadable"
            assert _changed(repo, lambda r: None) == ["secret.md"]
        finally:
            os.chmod(repo / "secret.md", 0o644)

    def test_a_status_change_with_equal_content_is_clean(self, repo):
        (repo / "new.md").write_text("n\n", encoding="utf-8")
        assert _changed(repo, lambda r: _git(r, "add", "new.md")) == []

    def test_a_rename_reports_both_paths(self, repo):
        assert _changed(repo, lambda r: _git(r, "mv", "other.txt", "renamed.txt")) == [
            "other.txt", "renamed.txt"]

    def test_a_deletion_always_counts(self, repo):
        assert _changed(repo, lambda r: (r / "other.txt").unlink()) == ["other.txt"]

    def test_over_the_cap_the_comparison_is_path_and_status_only(
        self, repo, monkeypatch, capsys
    ):
        monkeypatch.setattr(rob, "PORCELAIN_HASH_CAP_BYTES", 4)
        (repo / "tracked.txt").write_text("dirty and bigger than four\n",
                                          encoding="utf-8")
        before = rob.porcelain_snapshot(str(repo))
        assert before.hashed is False
        assert "hashing cap" in capsys.readouterr().out
        (repo / "tracked.txt").write_text("rewritten, still dirty\n", encoding="utf-8")
        after = rob.porcelain_snapshot(str(repo))
        # The pre-dirty write is not seen (today's behaviour), and nothing raised.
        assert rob.unexpected_writes(before, after) == []
        (repo / "brand-new.md").write_text("x\n", encoding="utf-8")
        assert rob.unexpected_writes(before, rob.porcelain_snapshot(str(repo))) == [
            "brand-new.md"]

    def test_a_failing_hash_object_degrades_without_raising(self, repo, monkeypatch):
        real = subprocess.run

        def fake(argv, *a, **kw):
            if _subcommand(argv) == "hash-object":
                return subprocess.CompletedProcess(argv, 128, "", "fatal")
            return real(argv, *a, **kw)

        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        monkeypatch.setattr(rob.subprocess, "run", fake)
        snap = rob.porcelain_snapshot(str(repo))
        assert snap is not None and snap.hashed is False

    def test_ignored_prefixes_are_never_hashed(self, repo, monkeypatch):
        (repo / ".multi-llm").mkdir()
        (repo / ".multi-llm" / "snap.md").write_text("s\n", encoding="utf-8")
        (repo / "plans").mkdir()
        (repo / "plans" / "log.txt").write_text("l\n", encoding="utf-8")
        (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        real = subprocess.run
        hashed = []

        def spy(argv, *a, **kw):
            if _subcommand(argv) == "hash-object":
                hashed.append(kw.get("input", ""))
            return real(argv, *a, **kw)

        monkeypatch.setattr(rob.subprocess, "run", spy)
        snap = rob.porcelain_snapshot(str(repo), ignore_prefixes=[".multi-llm", "plans"])
        assert set(snap.entries) == {"tracked.txt"}
        assert hashed == ["tracked.txt\n"]

    def test_one_hash_object_call_per_snapshot(self, repo, monkeypatch):
        for n in range(5):
            (repo / f"u{n}.md").write_text(f"{n}\n", encoding="utf-8")
        real = subprocess.run
        calls = []

        def spy(argv, *a, **kw):
            calls.append(_subcommand(argv))
            return real(argv, *a, **kw)

        monkeypatch.setattr(rob.subprocess, "run", spy)
        rob.porcelain_snapshot(str(repo))
        assert calls.count("hash-object") == 1

    def test_no_repository_is_none(self, tmp_path):
        assert rob.porcelain_snapshot(None) is None
        assert rob.porcelain_snapshot(str(tmp_path)) is None
        assert rob.unexpected_writes(None, None) == []

    def test_output_that_is_not_porcelain_is_none(self, repo, monkeypatch):
        monkeypatch.setattr(rob.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, '{"type": "thread.started"}\n', ""))
        assert rob.porcelain_snapshot(str(repo)) is None
