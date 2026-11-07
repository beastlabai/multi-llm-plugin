"""The Claude release helpers: the pre-push guard, its hook file, and ``publish``.

Scope: ``packaging/claude/release.py``. Claude Code installs read ``master``, and
``master`` has no branch protection, so these are what stand between a push and
users. Each row builds a throwaway repository with this one's shape where the rules
look (the version owner, the generated tree) and a bare repository as its remote.
Nothing here touches the network or this repository's refs, except the one row that
rebuilds this checkout's own ``HEAD`` in a throwaway worktree.

The clean-rebuild check is stubbed everywhere but that row and its own unit rows:
the fixture repositories carry no builder, and what the rebuild decides is
``build.py --require-no-diff``'s, which ``test_claude_package.py`` covers.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
RELEASE = REPO_ROOT / "packaging" / "claude" / "release.py"

_spec = importlib.util.spec_from_file_location("claude_release_helpers", RELEASE)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

TREE = release.build.PLUGIN_RELATIVE
OWNER = release.build.VERSION_PATH
ZERO = "0" * 40

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="the hook is a POSIX sh script")


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory, monkeypatch):
    """No user or system git configuration, and a fixed identity: a signing or
    hooksPath setting on the developer's machine must not decide a row."""
    home = tmp_path_factory.mktemp("gitconfig")
    config = home / "config"
    config.write_text(
        "[init]\n\tdefaultBranch = master\n[commit]\n\tgpgsign = false\n"
        "[tag]\n\tgpgsign = false\n[advice]\n\tdetachedHead = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "t")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "t@example.invalid")
    for name in release.GIT_LOCAL_ENV:
        monkeypatch.delenv(name, raising=False)


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout.strip()


class Repo:
    """A clone with the plugin's shape and a bare ``origin``.

    :meth:`commit` writes the version owner and a shipped file, in the source and in
    the tree, so a default commit is a shipped-byte edit carrying a version.
    """

    def __init__(self, root: Path):
        self.root = root / "work"
        self.remote = root / "origin.git"
        _git(root, "init", "-q", "--bare", str(self.remote))
        _git(root, "init", "-q", str(self.root))
        _git(self.root, "remote", "add", "origin", str(self.remote))
        self.write(".gitattributes", "* text=auto eol=lf\n")

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def commit(self, version: str, body: str = "a", *, tree: bool = True,
               message: str = "") -> str:
        pyproject = f'[project]\nname = "multi-llm"\nversion = "{version}"\n'
        self.write(OWNER, pyproject)
        self.write("skills/multi-llm/multi_llm.py", f"BODY = {body!r}\n")
        if tree:
            self.write(f"{TREE}/skills/multi-llm/pyproject.toml", pyproject)
            self.write(f"{TREE}/skills/multi-llm/multi_llm.py", f"BODY = {body!r}\n")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "--allow-empty", "-m", message or f"{version} {body}")
        return self.head()

    def docs_commit(self, text: str) -> str:
        """A commit that ships nothing."""
        self.write("docs/notes.md", text)
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", f"docs {text}")
        return self.head()

    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD")

    def push_master(self) -> str:
        _git(self.root, "push", "-q", "origin", "HEAD:refs/heads/master")
        return self.head()

    def remote_master(self) -> Optional[str]:
        done = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "refs/heads/master"],
                              cwd=str(self.remote), capture_output=True, text=True)
        return done.stdout.strip() or None

    def remote_tags(self) -> List[str]:
        out = _git(self.remote, "tag", "--list")
        return [line for line in out.splitlines() if line]

    def tag(self, name: str, ref: str = "HEAD") -> None:
        _git(self.root, "tag", "-a", name, "-m", name, ref)


@pytest.fixture
def repo(tmp_path) -> Repo:
    return Repo(tmp_path)


class Stale:
    """The clean-rebuild check, stubbed: records what it was asked."""

    def __init__(self, problems: Optional[List[str]] = None):
        self.problems = problems or []
        self.calls: List[str] = []

    def __call__(self, checkout: Path, commit: str) -> List[str]:
        self.calls.append(commit)
        return list(self.problems)


def _line(local_ref: str, local: str, remote_ref: str, remote: str) -> tuple:
    return (local_ref, local, remote_ref, remote)


# ===========================================================================
# git's stdin
# ===========================================================================

class TestParseUpdates:
    def test_each_line_is_four_fields(self):
        text = f"refs/heads/master {'a' * 40} refs/heads/master {ZERO}\n\n"
        assert release.parse_updates(text) == [
            ("refs/heads/master", "a" * 40, "refs/heads/master", ZERO)]

    def test_a_malformed_line_is_an_error_not_a_pass(self):
        with pytest.raises(release.ReleaseError, match="unexpected pre-push line"):
            release.parse_updates("refs/heads/master abc\n")

    def test_zero_is_git_s_no_object_in_either_hash_width(self):
        assert release.is_zero(ZERO) and release.is_zero("0" * 64)
        assert not release.is_zero("") and not release.is_zero("0" * 39 + "1")


# ===========================================================================
# pre-push: master
# ===========================================================================

class TestPrePushMaster:
    def test_a_branch_other_than_master_is_not_checked(self, repo):
        sha = repo.commit("0.1.0")
        stale = Stale(["never"])
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/feature", sha, "refs/heads/feature", ZERO)], stale=stale) == 0
        assert stale.calls == []

    def test_deleting_master_is_not_this_check_s_business(self, repo):
        sha = repo.commit("0.1.0")
        stale = Stale(["never"])
        assert release.pre_push(repo.root, "origin", [
            _line("(delete)", ZERO, "refs/heads/master", sha)], stale=stale) == 0
        assert stale.calls == []

    def test_a_bumped_tree_passes_and_says_so(self, repo, capsys):
        base = repo.commit("0.1.0")
        sha = repo.commit("0.1.1", "b")
        stale = Stale()
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", base)], stale=stale) == 0
        assert stale.calls == [sha]
        assert f"pass for master {sha[:12]}" in capsys.readouterr().err

    def test_a_moved_tree_under_the_same_version_is_refused(self, repo, capsys):
        """The first release has no tag yet: the remote's master is what holds it."""
        base = repo.commit("0.1.0")
        sha = repo.commit("0.1.0", "b")
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", base)], stale=Stale()) == 1
        err = capsys.readouterr().err
        assert f"REFUSING: {TREE}/ moved since {base}" in err
        assert "--no-verify" in err and "nothing was pushed" in err

    def test_the_newest_tag_holds_a_push_to_a_new_remote(self, repo, capsys):
        repo.commit("0.1.0")
        repo.tag("claude-v0.1.0")
        sha = repo.commit("0.1.0", "b")
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", ZERO)], stale=Stale()) == 1
        assert "moved since claude-v0.1.0" in capsys.readouterr().err

    def test_the_pushed_commit_is_judged_not_head(self, repo):
        base = repo.commit("0.1.0")
        moved = repo.commit("0.1.0", "b")
        repo.commit("0.1.1", "c")
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/x", moved, "refs/heads/master", base)], stale=Stale()) == 1

    def test_a_remote_master_this_clone_lacks_is_refused_by_name(self, repo, capsys):
        sha = repo.commit("0.1.0")
        unknown = "1" * 40
        stale = Stale()
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", unknown)], stale=stale) == 1
        err = capsys.readouterr().err
        assert unknown[:12] in err and "git fetch origin" in err
        assert stale.calls == []

    def test_a_stale_generated_tree_is_refused(self, repo, capsys):
        base = repo.commit("0.1.0")
        sha = repo.commit("0.1.1", "b")
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", base)],
            stale=Stale(["stale: run `make claude-regen`"])) == 1
        assert "REFUSING: stale: run `make claude-regen`" in capsys.readouterr().err

    def test_a_commit_without_the_tree_is_refused(self, repo, capsys):
        base = repo.commit("0.1.0", tree=False)
        sha = repo.commit("0.1.1", "b", tree=False)
        assert release.pre_push(repo.root, "origin", [
            _line("refs/heads/master", sha, "refs/heads/master", base)], stale=Stale()) == 1
        assert f"carries no {TREE}/" in capsys.readouterr().err


# ===========================================================================
# pre-push: claude-v* tags
# ===========================================================================

class TestPrePushTag:
    def _push_tag(self, repo: Repo, tag: str, ref: str = "HEAD", extra=()) -> int:
        repo.tag(tag, ref)
        sha = _git(repo.root, "rev-parse", f"refs/tags/{tag}")
        return release.pre_push(repo.root, "origin", [
            *extra, _line(f"refs/tags/{tag}", sha, f"refs/tags/{tag}", ZERO)], stale=Stale())

    def test_a_tag_on_the_pushed_master_passes(self, repo):
        repo.commit("0.1.0")
        repo.push_master()
        _git(repo.root, "fetch", "-q", "origin")
        assert self._push_tag(repo, "claude-v0.1.0") == 0

    def test_a_tag_pushed_with_its_master_passes(self, repo):
        sha = repo.commit("0.1.0")
        master = _line("refs/heads/master", sha, "refs/heads/master", ZERO)
        assert self._push_tag(repo, "claude-v0.1.0", extra=[master]) == 0

    def test_a_tag_on_a_commit_the_remote_lacks_is_refused(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._push_tag(repo, "claude-v0.1.0") == 1
        assert "which master on origin does not contain" in capsys.readouterr().err

    def test_the_destination_is_asked_not_a_stale_tracking_ref(self, repo, capsys):
        """Review finding: the remote's master rolled back, origin/master did not."""
        first = repo.commit("0.1.0")
        repo.commit("0.1.1", "b")
        repo.push_master()
        _git(repo.root, "fetch", "-q", "origin")
        _git(repo.remote, "update-ref", "refs/heads/master", first)
        assert self._push_tag(repo, "claude-v0.1.1") == 1
        assert "which master on origin does not contain" in capsys.readouterr().err

    def test_a_push_to_a_bare_url_reads_that_url(self, repo):
        """Review finding: no remote name, so no tracking ref to read."""
        sha = repo.commit("0.1.0")
        repo.push_master()
        repo.tag("claude-v0.1.0")
        url = str(repo.remote)
        assert release.pre_push(repo.root, url, [
            _line("refs/tags/claude-v0.1.0", sha, "refs/tags/claude-v0.1.0", ZERO)],
            url=url, stale=Stale()) == 0

    def test_a_destination_master_this_clone_lacks_is_named(self, repo, tmp_path, capsys):
        repo.commit("0.1.0")
        repo.push_master()
        other = tmp_path / "other"
        _git(tmp_path, "clone", "-q", str(repo.remote), str(other))
        (other / "x").write_text("x", encoding="utf-8")
        _git(other, "add", "x")
        _git(other, "commit", "-q", "-m", "elsewhere")
        _git(other, "push", "-q", "origin", "HEAD:master")
        assert self._push_tag(repo, "claude-v0.1.0") == 1
        assert "which this clone does not have" in capsys.readouterr().err

    def test_a_tag_naming_another_version_is_refused(self, repo, capsys):
        repo.commit("0.1.1")
        repo.push_master()
        _git(repo.root, "fetch", "-q", "origin")
        assert self._push_tag(repo, "claude-v0.1.0") == 1
        assert "whose version is 0.1.1" in capsys.readouterr().err

    def test_a_tag_on_a_commit_without_the_tree_is_refused(self, repo, capsys):
        repo.commit("0.1.0", tree=False)
        repo.push_master()
        _git(repo.root, "fetch", "-q", "origin")
        assert self._push_tag(repo, "claude-v0.1.0") == 1
        assert f"carries no {TREE}/" in capsys.readouterr().err

    def test_a_tag_that_is_no_version_is_refused(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._push_tag(repo, "claude-vnext") == 1
        assert "does not name a strict semver version" in capsys.readouterr().err

    def test_other_families_tags_pass_untouched(self, repo):
        repo.commit("0.1.0")
        assert self._push_tag(repo, "codex-v9.9.9") == 0


# ===========================================================================
# The clean rebuild
# ===========================================================================

class TestStaleTreeCheck:
    def _record(self, calls, code=0, out=""):
        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, code, stdout="", stderr=out)
        return runner

    def _with_builders(self, repo: Repo) -> str:
        for builder in ("packaging/claude/build.py", "packaging/codex/build.py"):
            repo.write(builder, "# builder\n")
        repo.write("packaging/codex/multi-llm/README.md", "x\n")
        return repo.commit("0.1.0")

    def test_each_tree_is_rebuilt_by_its_own_commit_s_builder(self, repo):
        sha = self._with_builders(repo)
        calls: list = []
        assert release.stale_tree_problems(repo.root, sha, runner=self._record(calls)) == []
        assert [Path(argv[-2]).as_posix().split("/")[-3:] for argv, _ in calls] == [
            ["packaging", "claude", "build.py"], ["packaging", "codex", "build.py"]]
        for argv, kwargs in calls:
            assert argv[-1] == "--require-no-diff" and "--frozen" in argv
            worktree = Path(kwargs["cwd"])
            assert Path(argv[-2]).is_relative_to(worktree) and worktree != repo.root
            env = kwargs["env"]
            for name in ("MULTI_LLM_ALLOW_DIRTY", "UV_PROJECT_ENVIRONMENT", "GIT_DIR",
                         "GIT_INDEX_FILE"):
                assert name not in env

    def test_the_throwaway_worktree_is_gone_afterwards(self, repo):
        sha = self._with_builders(repo)
        before = _git(repo.root, "worktree", "list", "--porcelain")
        release.stale_tree_problems(repo.root, sha, runner=self._record([], code=1))
        assert _git(repo.root, "worktree", "list", "--porcelain") == before

    def test_a_failed_rebuild_names_the_tree_and_its_regen(self, repo):
        sha = self._with_builders(repo)
        problems = release.stale_tree_problems(
            repo.root, sha, runner=self._record([], code=1, out="ERROR: differs\n"))
        assert len(problems) == 2
        assert f"{TREE}/ at {sha[:12]}" in problems[0] and "`make claude-regen`" in problems[0]
        assert "ERROR: differs" in problems[0]
        assert "`make codex-regen`" in problems[1]

    def test_a_missing_builder_or_tree_is_refused_not_skipped(self, repo):
        """Review finding: a commit that deleted its builder must not pass unrebuilt."""
        sha = repo.commit("0.1.0")
        calls: list = []
        problems = release.stale_tree_problems(repo.root, sha, runner=self._record(calls))
        assert calls == []
        assert len(problems) == 2
        assert "no packaging/claude/build.py" in problems[0]
        assert "no packaging/codex/multi-llm/ and no packaging/codex/build.py" in problems[1]

    @pytest.mark.skipif(sys.platform != "linux" or shutil.which("uv") is None,
                        reason="rebuilds this checkout with the real builders (Linux, uv)")
    def test_this_checkout_s_head_rebuilds_to_its_own_trees(self):
        """The real builders, the real argv: HEAD's committed trees are current, as
        CI's `package` job holds every commit to."""
        assert release.stale_tree_problems(REPO_ROOT, "HEAD") == []


# ===========================================================================
# The hook file
# ===========================================================================

class TestHookFile:
    def test_install_writes_an_executable_marked_hook(self, repo):
        status, message = release.install_hook(repo.root)
        path = repo.root / ".git" / "hooks" / "pre-push"
        assert status == 0 and str(path) in message
        assert release.HOOK_MARK in path.read_text(encoding="utf-8")
        if sys.platform != "win32":
            assert path.stat().st_mode & stat.S_IXUSR

    def test_install_follows_core_hooks_path(self, repo):
        _git(repo.root, "config", "core.hooksPath", "my-hooks")
        release.install_hook(repo.root)
        assert (repo.root / "my-hooks" / "pre-push").is_file()

    def test_a_worktree_installs_into_the_shared_hooks(self, repo, tmp_path):
        repo.commit("0.1.0")
        linked = tmp_path / "linked"
        _git(repo.root, "worktree", "add", "-q", "--detach", str(linked))
        release.install_hook(linked)
        assert (repo.root / ".git" / "hooks" / "pre-push").is_file()

    def test_someone_else_s_hook_is_left_alone(self, repo):
        path = repo.root / ".git" / "hooks" / "pre-push"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        status, message = release.install_hook(repo.root)
        assert status == 1 and "left alone" in message
        assert path.read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"
        status, message = release.uninstall_hook(repo.root)
        assert status == 1 and path.is_file()

    def test_reinstall_and_uninstall_touch_only_this_hook(self, repo):
        assert release.install_hook(repo.root)[0] == 0
        assert release.install_hook(repo.root)[0] == 0
        assert release.uninstall_hook(repo.root)[0] == 0
        assert not (repo.root / ".git" / "hooks" / "pre-push").exists()
        assert release.uninstall_hook(repo.root) == (
            0, f"no pre-push hook at {repo.root / '.git' / 'hooks' / 'pre-push'}")


@POSIX_ONLY
class TestTheInstalledHook:
    """The shell file git runs, driven by a real ``git push`` with ``uv`` faked."""

    @pytest.fixture
    def fake_uv(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "uv"
        script = bin_dir / "uv"
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > '{log}.args'\n"
            f"cat > '{log}.stdin'\n"
            "pwd > '" f"{log}.cwd'\n"
            f"env > '{log}.env'\n"
            "exit ${FAKE_UV_EXIT:-0}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        return log

    @pytest.fixture
    def hooked(self, repo):
        repo.write("packaging/claude/release.py", "# stand-in\n")
        repo.commit("0.1.0")
        assert release.install_hook(repo.root)[0] == 0
        return repo

    def _push(self, repo: Repo, *refspec: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "push", "origin", *refspec], cwd=str(repo.root),
                              capture_output=True, text=True)

    def test_a_branch_push_never_starts_uv(self, hooked, fake_uv):
        assert self._push(hooked, "HEAD:refs/heads/feature").returncode == 0
        assert not Path(f"{fake_uv}.args").exists()

    def test_a_master_push_runs_the_check_with_git_s_lines(self, hooked, fake_uv):
        assert self._push(hooked, "HEAD:refs/heads/master").returncode == 0
        args = Path(f"{fake_uv}.args").read_text(encoding="utf-8").splitlines()
        assert args[-4:] == ["packaging/claude/release.py", "pre-push", "origin",
                             str(hooked.remote)]
        assert "--frozen" in args
        stdin = Path(f"{fake_uv}.stdin").read_text(encoding="utf-8")
        assert f" {hooked.head()} refs/heads/master {ZERO}" in stdin
        assert Path(f"{fake_uv}.cwd").read_text(encoding="utf-8").strip() == str(
            hooked.root.resolve())

    def test_a_refusal_stops_the_push(self, hooked, fake_uv, monkeypatch):
        monkeypatch.setenv("FAKE_UV_EXIT", "1")
        assert self._push(hooked, "HEAD:refs/heads/master").returncode != 0
        assert hooked.remote_master() is None

    def test_a_claude_tag_push_runs_the_check(self, hooked, fake_uv):
        hooked.tag("claude-v0.1.0")
        assert self._push(hooked, "refs/tags/claude-v0.1.0").returncode == 0
        assert "refs/tags/claude-v0.1.0" in Path(f"{fake_uv}.stdin").read_text(encoding="utf-8")

    def test_a_checkout_without_the_script_refuses_a_master_push(self, repo, fake_uv):
        """Review finding: an older checkout must not wave a bad master through."""
        repo.commit("0.1.0")
        release.install_hook(repo.root)
        done = self._push(repo, "HEAD:refs/heads/master")
        assert done.returncode != 0 and "has no packaging/claude/release.py" in done.stderr
        assert repo.remote_master() is None
        assert not Path(f"{fake_uv}.args").exists()
        assert self._push(repo, "HEAD:refs/heads/feature").returncode == 0

    def test_the_hook_s_uv_never_sees_an_outside_environment(self, hooked, fake_uv,
                                                              monkeypatch, tmp_path):
        """Review finding: uv would sync into UV_PROJECT_ENVIRONMENT."""
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "elsewhere"))
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
        assert self._push(hooked, "HEAD:refs/heads/master").returncode == 0
        env = Path(f"{fake_uv}.env").read_text(encoding="utf-8")
        assert "UV_PROJECT_ENVIRONMENT=" not in env and "VIRTUAL_ENV=" not in env

    def test_without_uv_a_master_push_is_refused(self, hooked, tmp_path, monkeypatch):
        bin_dir = tmp_path / "nouv"
        bin_dir.mkdir()
        (bin_dir / "git").symlink_to(shutil.which("git"))
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
        if shutil.which("uv"):
            pytest.skip("uv is installed system-wide here")
        done = self._push(hooked, "HEAD:refs/heads/master")
        assert done.returncode != 0 and "uv is not on PATH" in done.stderr


# ===========================================================================
# publish
# ===========================================================================

def _history_ok(checkout, commit):
    return ["release history: read"], []


class TestPublish:
    def _publish(self, repo: Repo, *, push: bool, stale=None, history=_history_ok,
                 fetch: bool = True) -> int:
        return release.publish(repo.root, push=push, stale=stale or Stale(), history=history,
                               fetch=fetch)

    def test_it_runs_from_master_only(self, repo, capsys):
        repo.commit("0.1.0")
        _git(repo.root, "checkout", "-q", "-b", "feature")
        assert self._publish(repo, push=True) == 1
        assert "this checkout is on feature" in capsys.readouterr().err
        assert repo.remote_master() is None

    def test_a_dry_run_prints_the_plan_and_pushes_nothing(self, repo, capsys):
        sha = repo.commit("0.1.0")
        assert self._publish(repo, push=False) == 0
        out = capsys.readouterr().out
        assert f"tag:    create claude-v0.1.0 on {sha[:12]} and push it" in out
        assert "dry run, nothing pushed" in out and "release history: read" in out
        assert repo.remote_master() is None and repo.remote_tags() == []
        assert _git(repo.root, "tag", "--list") == ""

    def test_the_first_release_pushes_master_and_its_tag(self, repo, capsys):
        sha = repo.commit("0.1.0")
        assert self._publish(repo, push=True) == 0
        assert repo.remote_master() == sha
        assert repo.remote_tags() == ["claude-v0.1.0"]
        assert _git(repo.remote, "rev-parse", "claude-v0.1.0^{commit}") == sha
        assert _git(repo.remote, "cat-file", "-t", "claude-v0.1.0") == "tag"
        assert "released Claude Code plugin 0.1.0" in capsys.readouterr().out

    def test_a_bump_is_a_fast_forward_and_a_new_tag(self, repo):
        repo.commit("0.1.0")
        assert self._publish(repo, push=True) == 0
        sha = repo.commit("0.1.1", "b")
        assert self._publish(repo, push=True) == 0
        assert repo.remote_master() == sha
        assert repo.remote_tags() == ["claude-v0.1.0", "claude-v0.1.1"]

    def test_a_push_that_ships_nothing_needs_no_tag(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._publish(repo, push=True) == 0
        capsys.readouterr()
        sha = repo.docs_commit("x")
        assert self._publish(repo, push=True) == 0
        assert repo.remote_master() == sha and repo.remote_tags() == ["claude-v0.1.0"]
        assert "none needed" in capsys.readouterr().out

    def test_an_untagged_release_already_pushed_is_only_tagged(self, repo, capsys):
        sha = repo.commit("0.1.0")
        repo.push_master()
        assert self._publish(repo, push=True) == 0
        assert repo.remote_tags() == ["claude-v0.1.0"]
        assert "push:   nothing" in capsys.readouterr().out
        assert repo.remote_master() == sha

    def test_an_unbumped_shipped_change_is_refused_and_nothing_moves(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._publish(repo, push=True) == 0
        before = repo.remote_master()
        repo.commit("0.1.0", "b")
        assert self._publish(repo, push=True) == 1
        err = capsys.readouterr().err
        assert "moved since" in err and "nothing was pushed" in err
        assert repo.remote_master() == before and repo.remote_tags() == ["claude-v0.1.0"]

    def test_a_stale_tree_is_refused(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._publish(repo, push=True, stale=Stale(["stale tree"])) == 1
        assert "REFUSING: stale tree" in capsys.readouterr().err
        assert repo.remote_master() is None

    def test_a_shared_history_refusal_is_refused(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._publish(repo, push=True,
                             history=lambda c, s: ([], ["0.1.0 is on npm already"])) == 1
        assert "0.1.0 is on npm already" in capsys.readouterr().err
        assert repo.remote_master() is None

    def test_the_push_url_is_the_destination_not_the_fetch_url(self, repo, tmp_path, capsys):
        """Review finding: origin fetches from one server and pushes to another, which
        already serves 0.1.1. An unbumped 0.1.1 change must be refused against it."""
        repo.commit("0.1.0")
        repo.push_master()
        served = tmp_path / "served.git"
        _git(tmp_path, "clone", "-q", "--bare", str(repo.remote), str(served))
        released = repo.commit("0.1.1", "b")
        _git(repo.root, "push", "-q", str(served), "HEAD:refs/heads/master")
        repo.commit("0.1.1", "c")
        _git(repo.root, "remote", "set-url", "--push", "origin", str(served))
        assert self._publish(repo, push=True) == 1
        assert f"moved since {released}" in capsys.readouterr().err
        assert _git(served, "rev-parse", "refs/heads/master") == released

    def test_several_push_urls_are_refused(self, repo, tmp_path, capsys):
        repo.commit("0.1.0")
        _git(repo.root, "remote", "set-url", "--add", "--push", "origin", str(repo.remote))
        _git(repo.root, "remote", "set-url", "--add", "--push", "origin", str(tmp_path / "b.git"))
        assert self._publish(repo, push=True) == 1
        assert "pushes to 2 URLs" in capsys.readouterr().err
        assert repo.remote_master() is None

    def test_a_local_tag_the_destination_lacks_is_pushed(self, repo, capsys):
        """Review finding: a tag made by hand, or left by a failed push, still ships."""
        sha = repo.commit("0.1.0")
        repo.tag("claude-v0.1.0")
        assert self._publish(repo, push=True) == 0
        assert repo.remote_tags() == ["claude-v0.1.0"]
        assert _git(repo.remote, "rev-parse", "claude-v0.1.0^{commit}") == sha
        assert "push claude-v0.1.0" in capsys.readouterr().out

    def test_a_destination_tag_elsewhere_is_refused(self, repo, capsys):
        first = repo.commit("0.1.0")
        repo.push_master()
        _git(repo.root, "push", "-q", "origin", f"{first}:refs/tags/claude-v0.1.0")
        repo.commit("0.1.0", "b")
        _git(repo.root, "tag", "-f", "-a", "claude-v0.1.0", "-m", "x")
        moved = repo.head()
        # git's own fetch refuses to clobber the disagreeing tag, and says which.
        assert self._publish(repo, push=True) == 1
        err = capsys.readouterr().err
        assert "claude-v0.1.0" in err and "disagrees" in err
        # Without the fetch, the destination's advertised tags are compared directly.
        assert self._publish(repo, push=True, fetch=False) == 1
        err = capsys.readouterr().err
        assert f"claude-v0.1.0 on {repo.remote} marks {first[:12]}" in err
        assert f"has it at {moved[:12]}" in err
        assert repo.remote_master() == first

    def test_a_local_tag_off_master_is_refused(self, repo, capsys):
        repo.commit("0.1.0")
        _git(repo.root, "checkout", "-q", "-b", "side")
        repo.commit("0.1.0", "side")
        repo.tag("claude-v0.1.0")
        _git(repo.root, "checkout", "-q", "master")
        assert self._publish(repo, push=True) == 1
        assert "never published" in capsys.readouterr().err
        assert repo.remote_master() is None

    def test_a_remote_master_ahead_of_local_is_never_overwritten(self, repo, tmp_path, capsys):
        repo.commit("0.1.0")
        repo.push_master()
        other = tmp_path / "other"
        _git(tmp_path, "clone", "-q", str(repo.remote), str(other))
        (other / "x").write_text("x", encoding="utf-8")
        _git(other, "add", "x")
        _git(other, "commit", "-q", "-m", "elsewhere")
        _git(other, "push", "-q", "origin", "HEAD:master")
        theirs = _git(other, "rev-parse", "HEAD")
        repo.commit("0.1.1", "b")
        assert self._publish(repo, push=True) == 1
        assert "never force-pushes" in capsys.readouterr().err
        assert repo.remote_master() == theirs


# ===========================================================================
# The CLI and the make targets
# ===========================================================================

class TestCli:
    def test_install_and_uninstall_through_main(self, repo, capsys):
        assert release.main(["--repo", str(repo.root), "install-hook"]) == 0
        assert (repo.root / ".git" / "hooks" / "pre-push").is_file()
        assert release.main(["--repo", str(repo.root), "uninstall-hook"]) == 0
        assert not (repo.root / ".git" / "hooks" / "pre-push").exists()

    def test_pre_push_reads_stdin(self, repo, monkeypatch, capsys):
        sha = repo.commit("0.1.0")
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(
            f"refs/heads/feature {sha} refs/heads/feature {ZERO}\n"))
        assert release.main(["--repo", str(repo.root), "pre-push", "origin", "url"]) == 0

    def test_a_malformed_stdin_refuses(self, repo, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("garbage\n"))
        assert release.main(["--repo", str(repo.root), "pre-push", "origin"]) == 1
        assert "unexpected pre-push line" in capsys.readouterr().err

    def test_the_make_targets_reach_the_helpers(self):
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        for target, needle in (
            ("claude-hook-install", "install-hook"),
            ("claude-hook-uninstall", "uninstall-hook"),
            ("claude-publish", "publish $(if $(PUSH),--push)"),
            ("bump-version", "bump_version.py"),
        ):
            recipe = makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
            assert needle in recipe, target
            assert target in makefile.split(".PHONY:", 1)[1].split("\n\n", 1)[0], target
