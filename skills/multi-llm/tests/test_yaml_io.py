"""``utils/yaml_io``: the safe loader, its stat-keyed file cache, and the config
caches built on it (``provider_registry.load_config``, ``host_capabilities.load``).

The contract is in ``utils/yaml_io.py``'s module docstring and
``docs/development/providers.md#configuration-and-runner-resolution``.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import utils.provider_registry as registry
from utils import host_capabilities, yaml_io
from tests.source_cache import python_files, source_tree

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parent.parent

#: Directories that hold no YAML of ours (environments, caches, test scratch).
_PRUNE = {".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "plans", ".git"}


def _repository_yaml_files():
    """Every YAML file the skill or its tests parse, relative to the repo root."""
    found = []
    for root, dirs, files in os.walk(SKILL_DIR):
        dirs[:] = sorted(d for d in dirs if d not in _PRUNE)
        for name in sorted(files):
            if name.endswith((".yaml", ".yml")) or name == "cli-versions.lock":
                found.append(Path(root) / name)
    workflows = REPO_ROOT / ".github"
    if workflows.is_dir():
        found.extend(sorted(workflows.rglob("*.yml")))
    return found


YAML_FILES = _repository_yaml_files()


def _ids(paths):
    return [p.relative_to(REPO_ROOT).as_posix() for p in paths]


def _load_all(text, loader):
    try:
        return "ok", list(yaml.load_all(text, Loader=loader))
    except yaml.YAMLError as exc:
        return "error", type(exc).__name__


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

class TestTheLoader:
    def test_the_sweep_finds_the_files_the_skill_reads(self):
        names = set(_ids(YAML_FILES))
        assert "skills/multi-llm/providers.yaml" in names
        assert "skills/multi-llm/hosts/capabilities.yaml" in names
        assert "skills/multi-llm/tests/container/cli-versions.lock" in names
        assert len(names) >= 40

    @pytest.mark.skipif(not yaml.__with_libyaml__, reason="PyYAML built without libyaml")
    @pytest.mark.parametrize("path", YAML_FILES, ids=_ids(YAML_FILES))
    def test_both_loaders_build_the_same_documents(self, path):
        """CSafeLoader is a drop-in for SafeLoader on every file in the repo.

        Compared by ``repr`` as well as ``==`` so key order and scalar types
        (``1`` against ``"1"``, ``True`` against ``"yes"``) count too.
        """
        text = path.read_text(encoding="utf-8")
        pure = _load_all(text, yaml.SafeLoader)
        fast = _load_all(text, yaml.CSafeLoader)
        assert fast == pure
        assert repr(fast) == repr(pure)

    def test_the_loader_is_libyaml_when_present_and_always_safe(self):
        expected = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader
        assert yaml_io.SAFE_LOADER is expected
        with pytest.raises(yaml.constructor.ConstructorError):
            yaml_io.safe_load("!!python/object/apply:os.getcwd []\n")

    def test_without_libyaml_it_falls_back_to_the_pure_loader(self):
        """A wheel without libyaml (possible on some platforms) still loads."""
        code = (
            "import sys\n"
            "sys.modules['yaml._yaml'] = None\n"
            "import yaml\n"
            "assert not yaml.__with_libyaml__ and not hasattr(yaml, 'CSafeLoader')\n"
            f"sys.path.insert(0, {str(SKILL_DIR)!r})\n"
            "from utils import yaml_io\n"
            "assert yaml_io.SAFE_LOADER is yaml.SafeLoader, yaml_io.SAFE_LOADER\n"
            "print(yaml_io.load_file(sys.argv[1])['default_provider'])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(SKILL_DIR / "providers.yaml")],
            capture_output=True, text=True, cwd=str(SKILL_DIR), timeout=120,
        )
        assert result.returncode == 0, result.stderr
        expected = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text("utf-8"))
        assert result.stdout.strip() == expected["default_provider"]

    def test_no_product_module_parses_yaml_itself(self):
        """Every runtime parse goes through yaml_io (one loader, one cache)."""
        offenders = []
        for path in python_files(SKILL_DIR):
            rel = path.relative_to(SKILL_DIR)
            if rel.parts[0] in _PRUNE | {"tests"} or rel == Path("utils/yaml_io.py"):
                continue
            tree = source_tree(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(a.name == "yaml" for a in node.names):
                    offenders.append(f"{rel}:{node.lineno} import yaml")
                elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
                    offenders.append(f"{rel}:{node.lineno} from yaml import")
        assert offenders == []


# ---------------------------------------------------------------------------
# The file cache
# ---------------------------------------------------------------------------

@pytest.fixture
def parses(monkeypatch):
    """Count real parses; every load_file miss calls yaml.load once."""
    yaml_io.clear_cache()
    calls = []
    real = yaml.load

    def counting(stream, Loader):
        calls.append(getattr(stream, "name", stream))
        return real(stream, Loader=Loader)

    monkeypatch.setattr(yaml_io.yaml, "load", counting)
    yield calls
    yaml_io.clear_cache()


@pytest.fixture
def settled(monkeypatch):
    """Treat every file as settled, so a just-written fixture is cacheable."""
    monkeypatch.setattr(yaml_io, "SETTLE_S", 0.0)


class TestTheFileCache:
    def test_an_unchanged_file_is_parsed_once(self, tmp_path, parses, settled):
        path = tmp_path / "a.yaml"
        path.write_text("a: [1, 2]\n", encoding="utf-8")
        assert yaml_io.load_file(path) == {"a": [1, 2]}
        assert yaml_io.load_file(str(path)) == {"a": [1, 2]}
        assert len(parses) == 1

    def test_every_caller_gets_its_own_copy(self, tmp_path, parses, settled):
        path = tmp_path / "a.yaml"
        path.write_text("a: {b: [1]}\n", encoding="utf-8")
        first = yaml_io.load_file(path)
        first["a"]["b"].append("planted")
        first["c"] = 1
        assert yaml_io.load_file(path) == {"a": {"b": [1]}}
        assert len(parses) == 1

    def test_a_changed_file_is_parsed_again(self, tmp_path, parses, settled):
        path = tmp_path / "a.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        assert yaml_io.load_file(path) == {"a": 1}
        path.write_text("a: 22\n", encoding="utf-8")
        assert yaml_io.load_file(path) == {"a": 22}
        assert len(parses) == 2

    def test_files_never_share_an_entry(self, tmp_path, parses, settled):
        one, two = tmp_path / "one" / "p.yaml", tmp_path / "two" / "p.yaml"
        for path, value in ((one, "one"), (two, "two")):
            path.parent.mkdir()
            path.write_text(f"v: {value}\n", encoding="utf-8")
        for _ in range(2):
            assert yaml_io.load_file(one) == {"v": "one"}
            assert yaml_io.load_file(two) == {"v": "two"}
        assert len(parses) == 2

    def test_a_just_written_file_is_never_cached(self, tmp_path, parses):
        """Inside SETTLE_S a same-size rewrite can keep the same signature."""
        path = tmp_path / "a.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        before = os.stat(path)
        assert yaml_io.load_file(path) == {"a": 1}
        path.write_text("a: 2\n", encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert yaml_io.load_file(path) == {"a": 2}
        assert len(parses) == 2

    def test_settledness_is_decided_before_the_read(self, tmp_path, monkeypatch):
        """A read that starts inside a timestamp tick and ends after SETTLE_S
        may have seen bytes a same-size rewrite in that tick replaced, so the
        clock is read when the signature is taken, not after the parse."""
        tick = 1_000 * 10**9
        clock = [tick + 5 * 10**8]  # half a second into the tick
        monkeypatch.setattr(yaml_io, "time", SimpleNamespace(time_ns=lambda: clock[0]))
        # A filesystem whose tick swallows both writes: same timestamps for each.
        monkeypatch.setattr(
            yaml_io, "_signature",
            lambda resolved, st: (resolved, st.st_dev, st.st_ino, st.st_size, tick, tick),
        )
        real = yaml.load

        def slow(stream, Loader):
            document = real(stream, Loader=Loader)
            clock[0] = tick + 10 * 10**9  # the read outlasted SETTLE_S
            return document

        monkeypatch.setattr(yaml_io.yaml, "load", slow)
        yaml_io.clear_cache()
        path = tmp_path / "a.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        assert yaml_io.load_file(path) == {"a": 1}
        path.write_text("a: 2\n", encoding="utf-8")  # same inode, same size
        assert yaml_io.load_file(path) == {"a": 2}
        yaml_io.clear_cache()

    def test_a_file_that_changes_during_the_read_is_not_cached(
        self, tmp_path, monkeypatch, settled
    ):
        yaml_io.clear_cache()
        path = tmp_path / "a.yaml"
        path.write_text("a: 1\n", encoding="utf-8")
        real = yaml.load

        def racing(stream, Loader):
            document = real(stream, Loader=Loader)
            if document == {"a": 1}:
                path.write_text("a: 22\n", encoding="utf-8")
            return document

        monkeypatch.setattr(yaml_io.yaml, "load", racing)
        assert yaml_io.load_file(path) == {"a": 1}
        assert os.path.realpath(path) not in yaml_io._CACHE
        assert yaml_io.load_file(path) == {"a": 22}
        assert os.path.realpath(path) in yaml_io._CACHE
        yaml_io.clear_cache()

    def test_errors_are_not_cached(self, tmp_path, parses, settled):
        path = tmp_path / "a.yaml"
        path.write_text("a: [1\n", encoding="utf-8")
        for _ in range(2):
            with pytest.raises(yaml_io.YAMLError):
                yaml_io.load_file(path)
        assert len(parses) == 2
        with pytest.raises(OSError):
            yaml_io.load_file(tmp_path / "missing.yaml")

    def test_settled_means_older_than_the_slowest_timestamp_tick(self):
        now = time.time_ns()
        old = now - int((yaml_io.SETTLE_S + 1) * 1e9)
        assert yaml_io.is_settled(None)
        assert yaml_io.is_settled(("/p", 1, 2, 3, old, old))
        assert not yaml_io.is_settled(("/p", 1, 2, 3, old, now))
        assert not yaml_io.is_settled(("/p", 1, 2, 3, now, old))

    def test_file_signature_tells_missing_from_a_broken_link(self, tmp_path):
        path = tmp_path / "a.yaml"
        assert yaml_io.file_signature(path) is None
        path.write_bytes(b"a: 1\n")  # bytes: a text write is "\r\n" (6 bytes) on Windows
        signature = yaml_io.file_signature(path)
        st = os.stat(path)
        assert signature == (st.st_dev, st.st_ino, len("a: 1\n"), st.st_mtime_ns, st.st_ctime_ns)
        link = tmp_path / "link.yaml"
        try:
            os.symlink(tmp_path / "gone.yaml", link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        assert yaml_io.file_signature(link)[0] == "link"


# ---------------------------------------------------------------------------
# provider_registry.load_config: one entry per set of unchanged layer files
# ---------------------------------------------------------------------------

def _repo(path: Path, project_layer: str = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "plans").mkdir(exist_ok=True)
    (path / "plans" / "p.md").write_text("# plan\n", encoding="utf-8")
    if project_layer is not None:
        _layer(path, project_layer)
    return path


def _layer(repo: Path, text: str) -> Path:
    layer = repo / ".multi-llm" / "providers.yaml"
    layer.parent.mkdir(exist_ok=True)
    layer.write_text(text, encoding="utf-8")
    return layer


def _symlink(target: Path, link: Path) -> None:
    if os.path.lexists(link):
        os.unlink(link)
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")


@pytest.fixture
def merges(monkeypatch):
    """Count uncached config merges."""
    calls = []
    real = registry._merge_layers

    def counting(layers, permissive):
        calls.append(tuple(str(path) for _, path in layers))
        return real(layers, permissive)

    monkeypatch.setattr(registry, "_merge_layers", counting)
    registry._config = None
    yield calls
    registry._config = None


@pytest.fixture
def git_lookups(monkeypatch):
    """Count the git root lookups discovery really runs."""
    calls = []
    real = registry.get_project_root_from_dir

    def counting(directory=None):
        calls.append(directory)
        return real(directory)

    monkeypatch.setattr(registry, "get_project_root_from_dir", counting)
    registry._config = None
    yield calls
    registry._config = None


BASE_PROVIDER = registry.load_base_config()["default_provider"]


def test_a_file_that_is_not_utf8_is_a_parse_error(tmp_path):
    """One exception type for "does not parse", so the registry's override
    loader turns it into ``ConfigError`` like any other malformed layer."""
    path = tmp_path / "latin1.yaml"
    path.write_bytes("key: caf\xe9\n".encode("latin-1"))
    with pytest.raises(yaml_io.YAMLError, match="not UTF-8"):
        yaml_io.load_file(path)


@pytest.mark.config_override
class TestTheConfigCache:
    def test_anchors_in_one_repository_share_one_entry(self, tmp_path, merges, settled):
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        plan = repo / "plans" / "p.md"
        first = registry.load_config(anchor=str(plan))
        assert first["default_provider"] == "gemini"
        for anchor in (str(repo), str(repo / "plans"), str(plan)):
            assert registry.load_config(anchor=anchor) is first
        assert len(merges) == 1

    def test_each_layer_set_gets_its_own_entry(self, tmp_path, monkeypatch, merges, settled):
        """A repository's layer is never served for another repository, nor
        for a directory outside any repository (the user's own layers only)."""
        one = _repo(tmp_path / "one", "default_provider: gemini\n")
        two = _repo(tmp_path / "two", "default_provider: codex\n")
        plain = tmp_path / "plain"
        plain.mkdir()
        assert BASE_PROVIDER not in ("gemini", "codex")
        for _ in range(2):
            assert registry.load_config(anchor=str(one / "plans" / "p.md"))["default_provider"] == "gemini"
            assert registry.load_config(anchor=str(two / "plans" / "p.md"))["default_provider"] == "codex"
            assert registry.load_config(anchor=str(plain))["default_provider"] == BASE_PROVIDER
        assert len(merges) == 3

        env = tmp_path / "env.yaml"
        env.write_text("default_provider: opencode\n", encoding="utf-8")
        monkeypatch.setenv(registry.ENV_CONFIG_VAR, str(env))
        assert registry.load_config(anchor=str(one))["default_provider"] == "opencode"
        assert registry.load_config(anchor=str(plain))["default_provider"] == "opencode"
        layer_one = Path(os.path.realpath(one)) / ".multi-llm" / "providers.yaml"
        assert merges[-2:] == [(str(layer_one), str(env.resolve())), (str(env.resolve()),)]

    def test_a_lowered_cap_reaches_the_ordinary_accessor(self, tmp_path, merges, settled):
        """The write scheduler (U22) reads its caps through the ordinary
        accessors; a lowered cap must reach them on the next access, from the
        anchor that loaded the old value as much as from any other."""
        repo = _repo(tmp_path / "repo", "driver:\n  max_concurrent_write_subagents: 4\n")
        plan = str(repo / "plans" / "p.md")
        assert registry.get_max_concurrent_write_subagents(anchor=plan) == 4
        assert registry.get_max_concurrent_write_subagents(anchor=plan) == 4
        assert len(merges) == 1
        _layer(repo, "driver:\n  max_concurrent_write_subagents: 1  # lowered\n")
        assert registry.get_max_concurrent_write_subagents(anchor=plan) == 1
        assert registry.get_max_concurrent_write_subagents(anchor=str(repo)) == 1
        _layer(repo, "driver:\n  max_concurrent_write_subagents: 3\n")
        assert registry.get_max_concurrent_write_subagents(anchor=str(repo)) == 3
        assert registry.get_max_concurrent_write_subagents(anchor=plan) == 3
        assert len(merges) == 3

    def test_a_just_written_layer_is_never_pinned(self, tmp_path, merges):
        """Inside SETTLE_S a same-size rewrite can keep the file's signature,
        so a load of a just-written layer is not cached at all."""
        repo = _repo(tmp_path / "repo", "driver:\n  max_concurrent_write_subagents: 4\n")
        layer = repo / ".multi-llm" / "providers.yaml"
        before = os.stat(layer)
        assert registry.get_max_concurrent_write_subagents(anchor=str(repo)) == 4
        layer.write_text("driver:\n  max_concurrent_write_subagents: 1\n", encoding="utf-8")
        os.utime(layer, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert registry.get_max_concurrent_write_subagents(anchor=str(repo)) == 1
        assert not registry._config.merged

    def test_a_layer_created_or_removed_later_shows_the_same_from_every_anchor(
        self, tmp_path, merges, settled
    ):
        repo = _repo(tmp_path / "repo")
        plan = str(repo / "plans" / "p.md")
        assert registry.load_config(anchor=plan)["default_provider"] == BASE_PROVIDER
        layer = _layer(repo, "default_provider: gemini\n")
        (repo / "unseen").mkdir()
        for anchor in (plan, str(repo), str(repo / "unseen")):
            assert registry.load_config(anchor=anchor)["default_provider"] == "gemini"
        layer.unlink()
        for anchor in (plan, str(repo), str(repo / "unseen")):
            assert registry.load_config(anchor=anchor)["default_provider"] == BASE_PROVIDER

    def test_an_edit_during_the_merge_is_not_pinned(self, tmp_path, monkeypatch, settled):
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        real = registry._merge_layers
        edited = []

        def racing(layers, permissive):
            snapshot = real(layers, permissive)
            if not edited:
                edited.append(_layer(repo, "default_provider: codex  # edited\n"))
            return snapshot

        monkeypatch.setattr(registry, "_merge_layers", racing)
        registry._config = None
        assert registry.load_config(anchor=str(repo))["default_provider"] == "gemini"
        assert not registry._config.merged
        assert registry.load_config(anchor=str(repo))["default_provider"] == "codex"
        assert len(registry._config.merged) == 1

    def test_concurrent_first_loads_merge_once(self, tmp_path, merges, settled):
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(registry.load_config(anchor=str(repo)))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(merges) == 1
        assert all(result is results[0] for result in results)

    def test_the_base_layer_is_parsed_once_per_process(self, tmp_path, parses):
        registry._config = None
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        registry.load_config(anchor=str(repo))
        registry.load_config(anchor=str(tmp_path))
        registry.get_driver_provenance("subagent_runner", anchor=str(repo))
        base = os.path.realpath(SKILL_DIR / "providers.yaml")
        assert [name for name in parses if os.path.realpath(name) == base] == [base]


_SCRATCH_GLOBAL_GIT_CONFIG = "[user]\n\tname = Test\n\temail = test@example.invalid\n"


@pytest.fixture
def scratch_git(tmp_path, monkeypatch):
    """A private HOME and global git config (returned), so a test can edit git's
    global settings and the user's own never reach the repositories it builds."""
    home = tmp_path / "home"
    home.mkdir()
    config = home / ".gitconfig"
    config.write_text(_SCRATCH_GLOBAL_GIT_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES",
                 "GIT_TEST_ASSUME_DIFFERENT_OWNER"):
        monkeypatch.delenv(name, raising=False)
    return config


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _git_root(directory):
    """git's own answer for *directory*, asked directly: what fresh discovery sees."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=directory, capture_output=True, text=True,
    )
    return os.path.realpath(proc.stdout.strip()) if proc.returncode == 0 else None


def _verify(anchor):
    """``(verify_command, layer)`` as the write scheduler reads them."""
    return registry.get_verify_settings(anchor=anchor)[::2]


@pytest.mark.config_override
class TestFreshDiscovery:
    """Which layer files a load reads comes from git on every access, never from
    a remembered answer, so the executed ``driver.verify_command`` always belongs
    to the repository git selects now
    (docs/development/providers.md#configuration-and-runner-resolution)."""

    def test_every_access_asks_git_and_reuses_the_merge(
        self, tmp_path, git_lookups, merges, settled
    ):
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        plan = str(repo / "plans" / "p.md")
        for _ in range(5):
            assert registry.load_config(anchor=plan)["default_provider"] == "gemini"
        assert len(git_lookups) == 5
        assert len(merges) == 1

    def test_core_worktree_in_a_gitfile_target_is_followed_both_ways(
        self, tmp_path, scratch_git, settled
    ):
        """``core.worktree`` in the git directory a gitfile names moves the root,
        though nothing under the checkout changes."""
        checkout = tmp_path / "checkout"
        git_dir = tmp_path / "checkout.git"
        _git("init", "-q", f"--separate-git-dir={git_dir}", str(checkout))
        _layer(checkout, 'driver:\n  verify_command: "echo checkout"\n')
        other = tmp_path / "other"
        other.mkdir()
        _layer(other, 'driver:\n  verify_command: "echo other"\n')
        anchor = str(checkout)
        assert _verify(anchor) == ("echo checkout", "project")
        _git(f"--git-dir={git_dir}", "config", "core.worktree", str(other))
        assert _git_root(checkout) == os.path.realpath(other)
        assert _verify(anchor) == ("echo other", "project")
        _git(f"--git-dir={git_dir}", "config", "--unset", "core.worktree")
        assert _git_root(checkout) == os.path.realpath(checkout)
        assert _verify(anchor) == ("echo checkout", "project")

    def test_a_worktree_config_redirect_is_followed_both_ways(
        self, tmp_path, scratch_git, settled
    ):
        """With ``extensions.worktreeConfig``, a linked worktree's own
        ``config.worktree`` can move its root, and moving it back is seen too."""
        main = _repo(tmp_path / "main")
        _git("commit", "-q", "--allow-empty", "-m", "init", cwd=main)
        linked = tmp_path / "linked"
        _git("worktree", "add", "-q", str(linked), cwd=main)
        _git("config", "extensions.worktreeConfig", "true", cwd=main)
        _layer(linked, 'driver:\n  verify_command: "echo linked"\n')
        other = tmp_path / "other"
        other.mkdir()
        _layer(other, 'driver:\n  verify_command: "echo other"\n')
        anchor = str(linked)
        assert _verify(anchor) == ("echo linked", "project")
        _git("config", "--worktree", "core.worktree", str(other), cwd=linked)
        assert _git_root(linked) == os.path.realpath(other)
        assert _verify(anchor) == ("echo other", "project")
        _git("config", "--worktree", "--unset", "core.worktree", cwd=linked)
        assert _git_root(linked) == os.path.realpath(linked)
        assert _verify(anchor) == ("echo linked", "project")

    def test_a_revoked_safe_directory_drops_the_project_layer(
        self, tmp_path, scratch_git, monkeypatch, settled
    ):
        """git refuses a repository another user owns unless the global config
        lists it in ``safe.directory``; revoking that drops the project's command."""
        repo = _repo(tmp_path / "repo", 'driver:\n  verify_command: "echo repo"\n')
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
        if _git_root(repo) is not None:
            pytest.skip("this git cannot be told the repository has another owner")
        untrusted = scratch_git.read_text(encoding="utf-8")
        trusted = untrusted + f"[safe]\n\tdirectory = {Path(os.path.realpath(repo)).as_posix()}\n"
        scratch_git.write_text(trusted, encoding="utf-8")
        anchor = str(repo / "plans" / "p.md")
        assert _git_root(repo) == os.path.realpath(repo)
        assert _verify(anchor) == ("echo repo", "project")
        scratch_git.write_text(untrusted, encoding="utf-8")
        assert _git_root(repo) is None
        assert _verify(anchor) == (None, "base")
        scratch_git.write_text(trusted, encoding="utf-8")
        assert _verify(anchor) == ("echo repo", "project")

    def test_a_malformed_global_config_drops_the_project_layer(
        self, tmp_path, scratch_git, settled
    ):
        """git refuses to run at all over a broken global config."""
        repo = _repo(tmp_path / "repo", 'driver:\n  verify_command: "echo repo"\n')
        anchor = str(repo / "plans" / "p.md")
        assert _verify(anchor) == ("echo repo", "project")
        good = scratch_git.read_text(encoding="utf-8")
        scratch_git.write_text(good + "[[[ not git config\n", encoding="utf-8")
        assert _git_root(repo) is None
        assert _verify(anchor) == (None, "base")
        scratch_git.write_text(good, encoding="utf-8")
        assert _verify(anchor) == ("echo repo", "project")

    def test_a_retargeted_checkout_symlink_reads_the_new_repository(
        self, tmp_path, settled
    ):
        """The verify_command a load returns, and the layer named with it,
        follow the checkout symlink to whatever it points at now."""
        one = _repo(tmp_path / "one", 'driver:\n  verify_command: "echo A"\n')
        two = _repo(tmp_path / "two", 'driver:\n  verify_command: "echo B"\n')
        plain = tmp_path / "plain"
        (plain / "plans").mkdir(parents=True)
        checkout = tmp_path / "checkout"
        _symlink(one, checkout)
        plan = str(checkout / "plans" / "p.md")
        assert registry.get_verify_settings(anchor=plan)[::2] == ("echo A", "project")
        registry.load_config(anchor=str(tmp_path))  # an unrelated anchor between
        _symlink(two, checkout)
        assert registry.get_verify_settings(anchor=plan)[::2] == ("echo B", "project")
        _symlink(plain, checkout)
        assert registry.get_verify_settings(anchor=plan)[::2] == (None, "base")
        _symlink(one, checkout)
        assert registry.get_verify_settings(anchor=plan)[::2] == ("echo A", "project")

    def test_a_git_root_created_or_removed_on_the_path_reruns_git(self, tmp_path, settled):
        repo = _repo(tmp_path / "repo", "default_provider: gemini\n")
        plan = str(repo / "plans" / "p.md")
        assert registry.load_config(anchor=plan)["default_provider"] == "gemini"
        subprocess.run(["git", "init", "-q"], cwd=repo / "plans", check=True)
        assert registry.load_config(anchor=plan)["default_provider"] == BASE_PROVIDER
        shutil.rmtree(repo / "plans" / ".git")
        assert registry.load_config(anchor=plan)["default_provider"] == "gemini"
        shutil.rmtree(repo / ".git")
        assert registry.load_config(anchor=plan)["default_provider"] == BASE_PROVIDER

    def test_a_directory_that_joins_a_repository_later_is_rediscovered(
        self, tmp_path, settled
    ):
        later = tmp_path / "later"
        (later / "plans").mkdir(parents=True)
        _layer(later, "default_provider: gemini\n")
        plan = str(later / "plans" / "p.md")
        assert registry.load_config(anchor=plan)["default_provider"] == BASE_PROVIDER
        subprocess.run(["git", "init", "-q"], cwd=later, check=True)
        assert registry.load_config(anchor=plan)["default_provider"] == "gemini"

    def test_a_git_redirect_variable_reruns_git(self, tmp_path, monkeypatch, settled):
        one = _repo(tmp_path / "one", "default_provider: gemini\n")
        two = _repo(tmp_path / "two", "default_provider: codex\n")
        plan = str(one / "plans" / "p.md")
        assert registry.load_config(anchor=plan)["default_provider"] == "gemini"
        monkeypatch.setenv("GIT_DIR", str(two / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(two))
        assert registry.load_config(anchor=plan)["default_provider"] == "codex"


@pytest.mark.config_override
class TestOneSnapshotPerLoad:
    """A value and the layer named with it come from the same load."""

    _WITH = 'driver:\n  verify_command: "echo project"\n  subagent_runner: "codex:gpt-x"\n'
    _WITHOUT = "driver:\n  subagent_retries: 2  # the project's values are gone\n"

    def _flipping(self, monkeypatch, repo):
        """Rewrite the project layer after every merge, so no two loads agree."""
        real = registry._merge_layers
        state = {"with": True}

        def flip(layers, permissive):
            snapshot = real(layers, permissive)
            state["with"] = not state["with"]
            _layer(repo, self._WITH if state["with"] else self._WITHOUT)
            return snapshot

        monkeypatch.setattr(registry, "_merge_layers", flip)
        registry._config = None

    def test_the_layers_are_recorded_with_the_values(self, tmp_path, monkeypatch, settled):
        repo = _repo(tmp_path / "repo", self._WITH)
        env = tmp_path / "env.yaml"
        env.write_text('driver:\n  subagent_runner: "opencode:x"\n', encoding="utf-8")
        monkeypatch.setenv(registry.ENV_CONFIG_VAR, str(env))
        driver, layers = registry._driver_view(anchor=str(repo))
        assert (driver["verify_command"], layers["verify_command"]) == ("echo project", "project")
        assert (driver["subagent_runner"], layers["subagent_runner"]) == ("opencode:x", "env")
        assert layers["subagent_retries"] == "base"
        assert registry.get_driver_provenance("no_such_key", anchor=str(repo)) == "default"

    def test_the_verify_command_and_its_layer_come_from_one_load(
        self, tmp_path, monkeypatch, settled
    ):
        repo = _repo(tmp_path / "repo", self._WITH)
        self._flipping(monkeypatch, repo)
        # Two loads disagree here: the hazard the paired read exists for.
        assert registry.get_verify_command(anchor=str(repo)) == "echo project"
        assert registry.get_driver_provenance("verify_command", anchor=str(repo)) == "base"
        seen = set()
        for _ in range(4):
            command, _timeout, layer = registry.get_verify_settings(anchor=str(repo))
            seen.add((command, layer))
        assert seen == {("echo project", "project"), (None, "base")}

    def test_the_runner_and_its_layer_come_from_one_load(
        self, tmp_path, monkeypatch, settled
    ):
        repo = _repo(tmp_path / "repo", self._WITH)
        self._flipping(monkeypatch, repo)
        results = [
            registry.resolve_subagent_runner("full", anchor=str(repo), host_id="claude-code")
            for _ in range(4)
        ]
        assert ("codex:gpt-x", "project") in results
        for spec, layer in results:
            assert (spec == "codex:gpt-x") == (layer == "project"), results


def test_concurrent_capability_loads_parse_once(monkeypatch):
    """Concurrent first readers (doctor's probe threads) share one parse."""
    host_capabilities.clear_cache()
    calls = []
    real = host_capabilities._load_uncached

    def slow(root):
        calls.append(root)
        time.sleep(0.05)
        return real(root)

    monkeypatch.setattr(host_capabilities, "_load_uncached", slow)
    barrier = threading.Barrier(6)
    results = []

    def worker():
        barrier.wait()
        results.append(host_capabilities.load())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    host_capabilities.clear_cache()
    assert len(calls) == 1
    assert all(result is results[0] for result in results)
