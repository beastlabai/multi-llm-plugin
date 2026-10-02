"""The repo-wide grant (plan step 3b, 2026-10-02): the dispatch half.

Plan ``codex-sandbox-unavailable-findings-and-fixes`` §6 F1 "Protected paths
for repo-wide write profiles", "Editing turns that run tests and builds"
(step 1b's decision), "Allow-list entries at dispatch" (the ``repo-wide``
row) and §8.3 "Step 3b is done when". What this file pins is
``landlock_dispatch``'s share — the enumeration, the required/optional split,
the pre-dispatch refusals, step 1b's redirects, the canary's extra targets and
the stream reading — and, at tier 2, the production wrapper denying every
case in F1's denied-write list while the positive cases land.

Tier 1 needs no kernel; tier 2 (``landlock`` marker) needs Landlock ABI 3 and
skips without it (fails under ``MULTI_LLM_REQUIRE_LANDLOCK=1``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import landlock_dispatch as ld  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils.provider_registry import DEFAULT_WRITE_PATHS  # noqa: E402
from utils.providers import codex as codex_mod  # noqa: E402
from utils.providers.base import (  # noqa: E402
    SANDBOX_CANARY_DENIED,
    WRITE_SCOPE_PLAN_FILE,
    WRITE_SCOPE_REPO_WIDE,
    JailContext,
    JailRequest,
)
from tests.test_landlock_dispatch import _run  # noqa: E402
from tests.test_landlock_exec import require_landlock  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="the plugin-landlock rung is Linux-only; its dispatch half never runs on Windows",
)

RUNG = codex_mod.PLUGIN_LANDLOCK_RUNG
DENY = tuple(DEFAULT_WRITE_PATHS["deny"])


@pytest.fixture(autouse=True)
def temp_base(tmp_path, monkeypatch):
    base = Path(os.path.realpath(tmp_path)) / "tmpbase"
    base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    monkeypatch.setattr(ld, "_SWEPT_BASES", set())
    return base


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
        cwd=tree, check=True, capture_output=True, text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).stdout


@pytest.fixture
def wide(tmp_path):
    """A committed scratch repo with everything F1's protected set names, a
    denied entry of each kind, two symlinks, an ancestor (``plans/``) above
    the output folder, and ordinary files and directories."""
    root = Path(os.path.realpath(tmp_path))
    repo = root / "repo"
    out = repo / "plans" / "plan"
    phase = out / "implement"
    for d in ("src/pkg", "tests", ".codex", ".multi-llm/runs", ".multi-llm/tmp",
              "node_modules/dep", ".ssh", "plans/plan/tasks"):
        (repo / d).mkdir(parents=True)
    (phase / "pre_batch" / "t1").mkdir(parents=True)
    (phase / "rollback_store").mkdir()
    files = {
        "README.md": "readme\n", "pyproject.toml": "[project]\nname='x'\n",
        ".env": "SECRET=1\n", ".npmrc": "registry=x\n", "key.pem": "pem\n",
        "src/app.py": "x = 1\n", "src/pkg/__init__.py": "", "tests/test_app.py": "def test(): pass\n",
        "node_modules/dep/index.js": "", ".ssh/id": "",
        "plans/plan.md": "# plan\n", "plans/other.md": "# other\n",
        "plans/plan/state.json": "{}\n", "plans/plan/tasks/tasks.md": "# tasks\n",
        "plans/plan/implement/pre_batch/t1/snapshot.json": "{}\n",
        "plans/plan/implement/rollback_store/blob": "bytes\n",
        "plans/plan/implement/subagent_audit.jsonl": "{}\n",
        ".codex/config.toml": "# project\n", ".multi-llm/providers.yaml": "# cfg\n",
    }
    for rel, text in files.items():
        (repo / rel).write_text(text)
    (repo / ".gitignore").write_text("node_modules/\n.env\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "gitlink").symlink_to(".git/config")
    (repo / "outlink").symlink_to("/etc")
    (repo / "srclink").symlink_to("src")
    home = root / "codex-home"
    home.mkdir()
    slot = root / "slot"
    slot.mkdir()
    return SimpleNamespace(root=root, repo=repo, out=out, phase=phase, home=home, slot=slot)


def _request(wide, *, declared=(), output_dir=True, deny=DENY, kind="implement"):
    return JailRequest(
        repo_root=str(wide.repo), kind=kind, write_scope=WRITE_SCOPE_REPO_WIDE,
        output_dir=str(wide.out) if output_dir else None, deny_patterns=tuple(deny),
        declared=tuple(str(p) for p in declared),
    )


def _ctx(wide, **kw):
    return JailContext(request=_request(wide, **kw), tmpdir=str(wide.slot),
                       codex_home=str(wide.home))


def _rel(wide, path):
    return os.path.relpath(path, wide.repo)


def _tree_hashes(top: Path):
    out = {}
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                out[str(path)] = ("link", os.readlink(path))
            else:
                out[str(path)] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return out


# ===========================================================================
# Tier 1: the enumeration (F1 "Protected paths": prevention by enumeration)
# ===========================================================================


class TestTheEnumeration:
    def test_the_protected_set_and_the_ancestor_get_no_rule(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        assert plan.root == str(wide.repo)
        assert [_rel(wide, p) for p in plan.protected] == [
            ".git", ".hg", ".svn", ".codex", ".multi-llm", "plans/plan",
        ]
        assert [_rel(wide, a) for a in plan.ancestors] == ["plans"]
        ruled = {e.path for e in plan.entries}
        for name in (".", ".git", ".codex", ".multi-llm", "plans", "plans/plan"):
            assert os.path.realpath(wide.repo / name) not in ruled, name

    def test_one_rule_per_entry_outside_the_protected_set(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        assert [(_rel(wide, e.path), e.kind, e.required) for e in plan.entries] == [
            (".gitignore", "file", False),
            ("README.md", "file", False),
            ("plans/other.md", "file", False),
            ("plans/plan.md", "file", False),
            ("pyproject.toml", "file", False),
            ("src", "dir", False),
            ("tests", "dir", False),
        ]
        assert all(e.why.startswith("repo-wide: ") for e in plan.entries)

    def test_the_denied_entries_get_no_rule_and_are_named(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        assert sorted((_rel(wide, p), pat) for p, pat in plan.denied) == [
            (".env", "**/.env"), (".npmrc", "**/.npmrc"), (".ssh", "**/.ssh/**"),
            ("key.pem", "**/*.pem"), ("node_modules", "**/node_modules/**"),
        ]

    def test_symlinks_get_no_rule_of_their_own(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        names = {_rel(wide, e.path) for e in plan.entries}
        assert not names & {"gitlink", "outlink", "srclink"}

    def test_with_no_deny_list_every_unprotected_entry_is_ruled(self, wide):
        plan = ld.repo_wide_plan(_request(wide, deny=()))
        names = {_rel(wide, e.path) for e in plan.entries}
        assert {".env", ".npmrc", ".ssh", "key.pem", "node_modules"} <= names
        assert plan.denied == ()

    def test_an_output_dir_outside_the_repo_is_not_protected(self, wide, tmp_path):
        request = JailRequest(
            repo_root=str(wide.repo), kind="implement", write_scope=WRITE_SCOPE_REPO_WIDE,
            output_dir=str(tmp_path / "elsewhere"),
        )
        plan = ld.repo_wide_plan(request)
        assert [_rel(wide, p) for p in plan.protected] == [".git", ".hg", ".svn", ".codex", ".multi-llm"]
        assert plan.ancestors == ()
        # `plans/` is then an ordinary directory entry.
        assert ("plans", "dir") in {(_rel(wide, e.path), e.kind) for e in plan.entries}

    def test_a_missing_repo_root_is_refused(self, wide):
        request = JailRequest(repo_root=str(wide.repo / "gone"), kind="implement",
                              write_scope=WRITE_SCOPE_REPO_WIDE)
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.repo_wide_plan(request)
        assert excinfo.value.reason == "repo-root-missing"

    def test_the_vcs_names_are_the_sweeps(self):
        assert ld.VCS_DIR_NAMES == wbs.SWEEP_PRUNED_DIR_NAMES
        assert ld.PROTECTED_ROOT_NAMES == ld.VCS_DIR_NAMES + (".codex", ".multi-llm")

    def test_the_probe_controls_come_from_the_enumeration(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        assert not hasattr(plan, "write_dir")  # no positive target in the tree (2026-10-02)
        assert [_rel(wide, p) for p in plan.negative_dirs()] == [
            ".git", ".codex", ".multi-llm", "plans/plan",
        ]


class TestTheHooksPathAndTheProvidersConfigAreProtected:
    """Fix round 4 (2026-10-02). ``core.hooksPath`` names a directory git
    runs unconfined on the user's next commit; outside ``.git/`` it was an
    ordinary entry with a dir rule, so a jailed batch could plant a hook
    there. ``MULTI_LLM_PROVIDERS_CONFIG`` is the config whose
    ``verify_command`` the next run executes unjailed; pointed inside the
    repository it was a ruled file. Both join the protected set when they
    resolve inside the repository, as the output folder does."""

    def test_a_relative_hooks_path_is_protected_and_never_ruled(self, wide):
        (wide.repo / ".husky").mkdir()
        (wide.repo / ".husky" / "pre-commit").write_text("#!/bin/sh\n")
        _git(wide.repo, "config", "core.hooksPath", ".husky")
        plan = ld.repo_wide_plan(_request(wide))
        assert str(wide.repo / ".husky") in plan.protected
        assert ".husky" not in {_rel(wide, e.path) for e in plan.entries}
        assert str(wide.repo / ".husky") in plan.negative_dirs()
        assert plan.write_dir() == str(wide.repo / "src")
        refusal = ld.repo_wide_declared_refusal(plan, [str(wide.repo / ".husky" / "pre-commit")])
        assert refusal and "protected path" in refusal and "(F6)" in refusal

    def test_an_absolute_hooks_path_inside_the_repo_makes_its_parent_an_ancestor(self, wide):
        (wide.repo / "tools" / "hooks").mkdir(parents=True)
        (wide.repo / "tools" / "lint.sh").write_text("")
        _git(wide.repo, "config", "core.hooksPath", str(wide.repo / "tools" / "hooks"))
        plan = ld.repo_wide_plan(_request(wide))
        assert str(wide.repo / "tools" / "hooks") in plan.protected
        assert [_rel(wide, a) for a in plan.ancestors] == ["plans", "tools"]
        names = {_rel(wide, e.path) for e in plan.entries}
        assert "tools/lint.sh" in names and "tools" not in names and "tools/hooks" not in names

    def test_a_hooks_path_outside_the_repo_changes_nothing(self, wide, tmp_path):
        outside = tmp_path / "githooks"
        outside.mkdir()
        _git(wide.repo, "config", "core.hooksPath", str(outside))
        plan = ld.repo_wide_plan(_request(wide))
        assert [_rel(wide, p) for p in plan.protected] == [
            ".git", ".hg", ".svn", ".codex", ".multi-llm", "plans/plan",
        ]

    def test_a_hooks_path_under_git_is_already_covered(self, wide):
        _git(wide.repo, "config", "core.hooksPath", ".git/hooks")
        plan = ld.repo_wide_plan(_request(wide))
        assert [_rel(wide, p) for p in plan.protected] == [
            ".git", ".hg", ".svn", ".codex", ".multi-llm", "plans/plan",
        ]

    def test_a_providers_config_inside_the_repo_is_protected(self, wide, monkeypatch):
        (wide.repo / "config").mkdir()
        config = wide.repo / "config" / "providers.yaml"
        config.write_text("driver:\n  verify_command: true\n")
        (wide.repo / "config" / "other.yaml").write_text("")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        plan = ld.repo_wide_plan(_request(wide))
        assert str(config) in plan.protected
        assert [_rel(wide, a) for a in plan.ancestors] == ["config", "plans"]
        names = {_rel(wide, e.path) for e in plan.entries}
        assert "config/other.yaml" in names and "config/providers.yaml" not in names
        assert str(config) not in plan.negative_dirs()
        refusal = ld.repo_wide_declared_refusal(plan, [str(config)])
        assert refusal and "protected path" in refusal

    def test_a_relative_providers_config_resolves_against_the_cwd(self, wide, monkeypatch):
        config = wide.repo / "providers.yaml"
        config.write_text("")
        monkeypatch.chdir(wide.repo)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", "providers.yaml")
        plan = ld.repo_wide_plan(_request(wide))
        assert str(config) in plan.protected
        assert "providers.yaml" not in {_rel(wide, e.path) for e in plan.entries}

    @pytest.mark.parametrize("where", ["outside", "unset", "under-.multi-llm"])
    def test_a_providers_config_elsewhere_changes_nothing(self, wide, monkeypatch, tmp_path, where):
        if where == "outside":
            monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(tmp_path / "providers.yaml"))
        elif where == "under-.multi-llm":
            monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG",
                               str(wide.repo / ".multi-llm" / "providers.yaml"))
        else:
            monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
        plan = ld.repo_wide_plan(_request(wide))
        assert [_rel(wide, p) for p in plan.protected] == [
            ".git", ".hg", ".svn", ".codex", ".multi-llm", "plans/plan",
        ]


# ===========================================================================
# Tier 1: the declared set — required entries and the pre-dispatch refusals
# ===========================================================================


class TestTheDeclaredSet:
    def test_the_entries_covering_a_declared_file_are_required(self, wide):
        declared = (wide.repo / "src/app.py", wide.repo / "src/pkg/new.py",
                    wide.repo / "README.md", wide.repo / "tests/sub/deep/new.py")
        plan = ld.repo_wide_plan(_request(wide, declared=declared))
        required = {_rel(wide, e.path) for e in plan.entries_for([str(p) for p in declared]) if e.required}
        assert required == {"src", "README.md", "tests"}
        assert ld.repo_wide_declared_refusal(plan, [str(p) for p in declared]) is None

    @pytest.mark.parametrize("rel,case", [
        ("newtop.txt", "new entry directly in"),
        ("newdir/x.py", "covered by no enumerated entry"),
        ("plans/new.md", "new entry directly in"),
        (".git/hooks/pre-commit", "protected path"),
        (".git/x", "protected path"),
        (".codex/x", "protected path"),
        (".multi-llm/providers.yaml", "protected path"),
        (".multi-llm/runs/x", "protected path"),
        ("plans/plan/state.json", "protected path"),
        ("plans/plan/implement/pre_batch/t1/snapshot.json", "protected path"),
        ("plans/plan/implement/rollback_store/blob", "protected path"),
        ("plans/plan/implement/subagent_audit.jsonl", "protected path"),
        ("node_modules/dep/index.js", "driver.write_paths denies"),
        (".env", "driver.write_paths denies"),
        ("gitlink", "protected path"),
        ("outlink/passwd", "outside the repository"),
        ("../elsewhere.txt", "outside the repository"),
    ])
    def test_a_declared_path_the_grant_cannot_cover_is_refused_naming_f6(self, wide, rel, case):
        plan = ld.repo_wide_plan(_request(wide))
        reason = ld.repo_wide_declared_refusal(plan, [str(wide.repo / rel)])
        assert reason is not None and case in reason, reason
        assert reason.endswith(ld.REPO_WIDE_F6)
        assert "driver.subagent_runner" in reason
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.allow_entries(_ctx(wide, declared=(wide.repo / rel,)))
        assert excinfo.value.reason == "declared-uncovered"

    def test_a_symlink_into_the_tree_resolves_to_its_target(self, wide):
        plan = ld.repo_wide_plan(_request(wide))
        assert ld.repo_wide_declared_refusal(plan, [str(wide.repo / "srclink" / "z.py")]) is None

    def test_the_allow_list_is_base_plus_the_enumeration_and_never_the_root(self, wide):
        ctx = _ctx(wide, declared=(wide.repo / "src/app.py",))
        entries = ld.allow_entries(ctx)
        assert [(e.path, e.kind, e.required) for e in entries[:5]] == [
            (str(wide.home), "dir", True), (str(wide.slot), "dir", True),
            ("/dev/null", "file", True), ("/dev/ptmx", "file", False), ("/dev/pts", "dir", False),
        ]
        tail = [(_rel(wide, e.path), e.kind, e.required) for e in entries[5:]]
        assert tail == [
            (".gitignore", "file", False), ("README.md", "file", False),
            ("plans/other.md", "file", False), ("plans/plan.md", "file", False),
            ("pyproject.toml", "file", False), ("src", "dir", True), ("tests", "dir", False),
        ]
        assert str(wide.repo) not in {e.path for e in entries}
        argv, _env = ld.wrap_argv(["codex", "exec"], RUNG, ctx, {})
        assert "--allow" in argv and f"dir:{wide.repo / 'src'}" in argv
        assert f"dir:{wide.repo / 'tests'}" in argv
        assert argv[argv.index(f"dir:{wide.repo / 'tests'}") - 1] == "--allow-optional"

    def test_an_unknown_scope_is_still_refused(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="implement",
                              write_scope="everything", target=str(wide.repo / "README.md"))
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.allow_entries(JailContext(request, str(wide.slot), str(wide.home)))
        assert excinfo.value.reason == "scope-not-served"


# ===========================================================================
# Tier 1: step 1b's redirects, the preview, the developer sentences, the canary
# ===========================================================================


class TestStepOneBRedirects:
    def test_repo_wide_adds_uv_and_pytest_cache_redirects_under_tmpdir(self, wide):
        ctx = _ctx(wide)
        redirects = ld.child_env_redirects(ctx)
        assert redirects["UV_CACHE_DIR"] == str(wide.slot / "uv-cache")
        assert redirects["PYTEST_ADDOPTS"] == f"-o cache_dir={wide.slot / 'pytest-cache'}"
        # Step 1c's three stay.
        assert redirects["GIT_OPTIONAL_LOCKS"] == "0"
        assert redirects["PYTHONPYCACHEPREFIX"] == str(wide.slot / "pycache")
        assert redirects["npm_config_cache"] == str(wide.slot / "npm-cache")
        _argv, env = ld.wrap_argv(["codex", "exec"], RUNG, ctx, {"PATH": "/bin"})
        assert env["UV_CACHE_DIR"] == str(wide.slot / "uv-cache") and env["TMPDIR"] == str(wide.slot)

    def test_a_plan_file_grant_gets_neither(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="apply_suggestions",
                              write_scope=WRITE_SCOPE_PLAN_FILE,
                              target=str(wide.repo / "plans" / "plan.md"))
        redirects = ld.child_env_redirects(JailContext(request, str(wide.slot), str(wide.home)))
        assert "UV_CACHE_DIR" not in redirects and "PYTEST_ADDOPTS" not in redirects

    def test_a_tmpdir_with_a_space_is_quoted_for_pytest(self, wide):
        slot = wide.root / "slot with space"
        slot.mkdir()
        ctx = JailContext(_request(wide), str(slot), str(wide.home))
        value = ld.child_env_redirects(ctx)["PYTEST_ADDOPTS"]
        import shlex
        assert shlex.split(value) == ["-o", f"cache_dir={slot / 'pytest-cache'}"]


class TestThePreviewAndTheInstructions:
    def test_the_preview_shows_a_placeholder_and_enumerates_nothing(self, wide):
        argv = ld.preview_wrap_argv(["codex", "exec"], RUNG, _request(wide))
        assert "--allow-optional" in argv
        assert f"dir:{ld.PLACEHOLDER_REPO_ENTRIES}" in argv
        assert not any(str(wide.repo) in a for a in argv)

    def test_the_developer_message_names_the_grant_its_limits_and_the_toolchain(self, wide):
        text = codex_mod.jail_developer_instructions(JailContext(_request(wide), str(wide.slot)))
        assert f"anywhere in the repository {wide.repo}" in text
        assert f"this run's output folder {wide.out}" in text
        assert ".git/, .codex/, .multi-llm/" in text
        assert "The repository root itself (and plans/) is not writable" in text
        assert "never replaced (sed -i, a temp-file save, mv)" in text
        assert "/tmp and /dev/shm are denied" in text
        assert "uv, pytest, npm and Python bytecode already use caches under $TMPDIR" in text
        assert "never try sudo or chown" in text
        assert "You may write only" not in text and "is the directory" not in text

    def test_a_plan_file_message_carries_no_repo_wide_sentence(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="apply_suggestions",
                              write_scope=WRITE_SCOPE_PLAN_FILE,
                              target=str(wide.repo / "plans" / "plan.md"))
        text = codex_mod.jail_developer_instructions(JailContext(request, str(wide.slot)))
        assert "You may edit existing files anywhere" not in text and "/dev/shm" not in text


class TestTheCanary:
    def test_canary_env_names_every_protected_dir_and_no_target_in_the_tree(self, wide):
        env = ld.canary_env(_ctx(wide))
        assert env[codex_mod.CANARY_ENV_DENY_DIRS].split("\n") == [
            str(wide.repo / ".git"), str(wide.repo / ".codex"),
            str(wide.repo / ".multi-llm"), str(wide.out),
        ]
        # The repo-wide grant's positive control is the canary's own $TMPDIR
        # write; nothing in the source tree is ever a target (2026-10-02).
        assert codex_mod.CANARY_ENV_WRITE_DIR not in env
        assert codex_mod.CANARY_ENV_WRITE_FILE not in env

    def test_the_rung_probe_hands_the_controls_to_the_canary(self, wide):
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(wide.root),
               "CODEX_HOME": str(wide.home)}
        resolved = ["/opt/codex/bin/codex", *codex_mod.PLUGIN_LANDLOCK_PROBE_ARGV[1:]]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            assert codex_mod.CANARY_ENV_WRITE_DIR not in probe.env
            assert str(wide.repo / ".git") in probe.env[codex_mod.CANARY_ENV_DENY_DIRS]
            assert str(wide.out) in probe.env[codex_mod.CANARY_ENV_DENY_DIRS]
            allowed = [a for a in probe.argv if a.startswith(("dir:", "file:"))]
            assert f"dir:{wide.repo / 'src'}" in allowed
            assert f"dir:{wide.repo}" not in allowed
        # The canary script reads the extra targets.
        assert codex_mod.CANARY_ENV_DENY_DIRS in codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT


def _plain_repo(root: Path, *dirs: str) -> SimpleNamespace:
    """A repo for the permissions-jailed canary: a `.git` directory (made, not
    initialised — the canary stats it, nothing reads it), the probe workdir
    the walk would have prepared, the directories named, and a codex home
    beside it. `root` may carry a space."""
    repo = root / "repo"
    for d in (".git", ".multi-llm/tmp", *dirs):
        (repo / d).mkdir(parents=True, exist_ok=True)
    home = root / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(root=root, repo=repo, home=home)


@contextlib.contextmanager
def _perms_jail(probe, *read_only: Path):
    """Tier 1's stand-in for the kernel: inside an entered `rung_probe`, the
    seeded file goes 0444 and the probe workdir, its `.multi-llm/` parent and
    every `read_only` directory 0555, so a write there fails on plain permissions exactly as the
    jail would deny it, while `$TMPDIR` stays writable. Everything is restored
    before the context (and the probe's own clean-up) runs on."""
    seeded = Path(probe.seeded)
    workdir = seeded.parent
    # `.multi-llm/` (the workdir's parent) is itself a protected root and so
    # always one of the negative targets.
    dirs = [workdir, workdir.parent, *read_only]
    try:
        seeded.chmod(0o444)
        for d in dirs:
            d.chmod(0o555)
        yield
    finally:
        for d in reversed(dirs):
            d.chmod(0o755)
        seeded.chmod(0o644)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="plain permissions deny nothing to root")
class TestTheCanaryUnderPlainPermissions:
    """The canary script's reading of its targets, with the kernel stood in
    for by file permissions (adversarial review, 2026-10-02: the extra
    negative targets were handed to the script `os.pathsep`-joined and
    word-split by the shell, so a repository path holding a space or a colon
    produced nonexistent targets that read as denied)."""

    @pytest.fixture
    def sh(self):
        path = shutil.which("sh")
        if not path:  # pragma: no cover
            pytest.skip("no sh on PATH")
        return path

    def _probe(self, plain, sh, **request):
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(plain.root),
               "CODEX_HOME": str(plain.home)}
        req = JailRequest(repo_root=str(plain.repo), kind="implement",
                          write_scope=WRITE_SCOPE_REPO_WIDE, **request)
        return ld.rung_probe(RUNG, req, [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT],
                             cwd=str(plain.repo), env=env)

    def test_the_controls_hold_when_every_target_is_denied(self, tmp_path, sh):
        plain = _plain_repo(tmp_path, "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe, plain.repo / ".git"):
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout

    def test_a_repository_path_with_a_space_keeps_its_negative_targets(self, tmp_path, sh):
        """`.git/` left writable under a path with a space must fail the
        canary: the target is the one directory, not two fragments of it."""
        plain = _plain_repo(tmp_path / "with space", "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe):
            assert str(plain.repo / ".git") in probe.env[codex_mod.CANARY_ENV_DENY_DIRS]
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED not in proc.stdout
        assert "not denied" in proc.stderr

    def test_a_repository_path_with_a_colon_keeps_its_negative_targets(self, tmp_path, sh):
        plain = _plain_repo(tmp_path / "a:b", "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe):
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED not in proc.stdout

    def test_a_negative_target_that_is_not_a_directory_is_a_setup_failure(self, tmp_path, sh):
        """A target the script cannot stat is a mis-handed target, never a
        denied write: the control that would pass vacuously stops the probe."""
        plain = _plain_repo(tmp_path, "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe, plain.repo / ".git"):
            env = dict(probe.env)
            env[codex_mod.CANARY_ENV_DENY_DIRS] += "\n" + str(plain.repo / "gone")
            proc = _run(ld.unwrap_argv(probe.argv), env, plain.repo)
        assert proc.returncode == codex_mod.LANDLOCK_CANARY_SETUP_EXIT, (proc.stdout, proc.stderr)
        assert codex_mod.LANDLOCK_CANARY_TARGET_FAILED in proc.stderr
        assert SANDBOX_CANARY_DENIED not in proc.stdout

    def test_an_unwritable_first_directory_does_not_fail_the_probe(self, tmp_path, sh):
        """The probe's positive control never writes into the source tree:
        the first enumerated directory (sorted order puts `.github`, `.husky`
        and a root-owned volume dir first) is nobody's proven-writable
        target, and a control there failed on plain permissions where the
        jail was fine (adversarial review, 2026-10-02, repro case B)."""
        plain = _plain_repo(tmp_path, ".github", "src")
        with self._probe(plain, sh) as probe, _perms_jail(
            probe, plain.repo / ".git", plain.repo / ".github",
        ):
            assert codex_mod.CANARY_ENV_WRITE_DIR not in probe.env
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout
        assert os.listdir(plain.repo / ".github") == []

    def test_a_target_holding_a_newline_cannot_be_handed_over(self, tmp_path, sh):
        """The one byte the hand-over cannot carry: the probe is a setup
        failure (the walk's `unknown`), never a pass on a split target."""
        plain = _plain_repo(tmp_path, "src", "plans/a\nb")
        with pytest.raises(ld.ProbeSetupFailed, match="newline"):
            with self._probe(plain, sh, output_dir=str(plain.repo / "plans" / "a\nb")):
                pass


# ===========================================================================
# Tier 1: the stream under a repo-wide grant
# ===========================================================================


def _cmd(output: str) -> str:
    return json.dumps({"type": "item.completed", "item": {
        "type": "command_execution", "command": "sh", "aggregated_output": output,
        "exit_code": 1, "status": "failed"}})


class TestJailDeniedEditsRepoWide:
    def test_declared_paths_are_in_scope_and_the_jails_denials_are_not(self, wide):
        ctx = _ctx(wide, declared=(wide.repo / "src/app.py", wide.repo / "README.md"))
        stream = "\n".join([
            _cmd("sh: 1: cannot create src/app.py: Permission denied\n"),
            _cmd("sh: 1: cannot create README.md.tmp: Permission denied\n"),
            _cmd("fatal: Unable to create '.git/index.lock': Permission denied\n"),
            _cmd("PermissionError: [Errno 13] Permission denied: '/tmp/x'\n"),
            _cmd("sh: 1: cannot create newtop.txt: Permission denied\n"),
            _cmd("sh: 1: cannot create plans/plan/state.json: Permission denied\n"),
        ])
        denials = ld.jail_denied_edits(stream, ctx, cwd=str(wide.repo))
        assert denials.in_scope_paths == [str(wide.repo / "src/app.py")]
        # A new root name beside a declared root file could be sed -i's
        # random temp file: counted as a replacement, never as clean.
        assert denials.replacement_paths == [
            str(wide.repo / "README.md.tmp"), str(wide.repo / "newtop.txt"),
        ]
        assert denials.out_of_scope_count == 3
        assert denials.failed_file_changes == []

    def test_a_root_sibling_is_a_replacement_only_beside_a_declared_root_file(self, wide):
        ctx = _ctx(wide, declared=(wide.repo / "src/app.py",))
        stream = _cmd("sh: 1: cannot create README.md.tmp: Permission denied\n")
        denials = ld.jail_denied_edits(stream, ctx, cwd=str(wide.repo))
        assert denials.replacement_paths == [] and denials.out_of_scope_count == 1

    def test_a_failed_file_change_is_still_reported(self, wide):
        event = json.dumps({"type": "item.completed", "item": {
            "type": "file_change", "status": "failed",
            "changes": [{"path": ".git/config", "kind": "update"}]}})
        denials = ld.jail_denied_edits(event, _ctx(wide), cwd=str(wide.repo))
        assert denials.failed_file_changes == [[str(wide.repo / ".git/config")]]
        assert denials.any_in_scope

    #: A test suite's own output: a failing test's report (pytest indents the
    #: exception with ``E   ``), the short summary, a test id, and a print of
    #: the exception's text. None of them is a tool reporting a denied write.
    ORDINARY_OUTPUT = (
        "tests/test_perm.py::test_locked FAILED\n"
        "tests/test_perm.py::test_raises[Permission denied] PASSED\n"
        "    with open('src/app.py', 'w') as h:\n"
        "E   PermissionError: [Errno 13] Permission denied: 'src/app.py'\n"
        "E       PermissionError: [Errno 13] Permission denied: 'README.md.tmp'\n"
        "[Errno 13] Permission denied: 'src/app.py'\n"
        "FAILED tests/test_perm.py::test_locked - PermissionError: [Errno 13] "
        "Permission denied: 'src/app.py'\n"
        "expected EACCES on src/app.py, got it: permission denied\n"
    )

    def test_a_test_suites_own_permission_denied_text_names_no_path(self, wide):
        """Fix round 4 (2026-10-02): a ``Permission denied`` line in ordinary
        test output used to resolve to a declared path (in scope, so a
        nothing-moved batch read FAILED) or to a new root name beside a
        declared root file (a ``replacement`` denial, same verdict). Only the
        shapes the measured denials take — a tool naming itself at column 0
        — count."""
        ctx = _ctx(wide, declared=(wide.repo / "src/app.py", wide.repo / "README.md"))
        denials = ld.jail_denied_edits(_cmd(self.ORDINARY_OUTPUT), ctx, cwd=str(wide.repo))
        assert denials.total == 0, denials.to_dict()
        assert ld._denied_paths_in(self.ORDINARY_OUTPUT) == []

    @pytest.mark.parametrize("line, expected", [
        # The fixture's shapes (codex 0.158.0, jail_writes.out.jsonl).
        ("/bin/bash: line 1: src/app.py: Permission denied", ["src/app.py"]),
        ("PermissionError: [Errno 13] Permission denied: 'src/app.py'", ["src/app.py"]),
        # dash, coreutils, git, sed, tee and node, as measured on the rig and
        # in Appendix A of the plan.
        ("sh: 1: cannot create src/app.py: Permission denied", ["src/app.py"]),
        ("touch: cannot touch 'src/app.py': Permission denied", ["src/app.py"]),
        ("rm: cannot remove 'README.md': Permission denied", ["README.md"]),
        ("mv: cannot move 'README.md' to 'README2.md': Permission denied",
         ["README.md", "README2.md"]),
        ("mkdir: cannot create directory \u2018newtop\u2019: Permission denied", ["newtop"]),
        ("cp: cannot create regular file 'src/app.py': Permission denied", ["src/app.py"]),
        ("fatal: Unable to create '.git/index.lock': Permission denied", [".git/index.lock"]),
        ("sed: couldn't open temporary file ./sedAbC12: Permission denied", ["./sedAbC12"]),
        ("tee: src/app.py: Permission denied", ["src/app.py"]),
        ("Error: EACCES: permission denied, open '/x/src/app.py'", ["/x/src/app.py"]),
        ("[Error: EACCES: permission denied, open '/x/src/app.py'] {", ["/x/src/app.py"]),
        ("Can't do inplace edit on README.md: Cannot make temp name: Permission denied.",
         ["README.md"]),
    ])
    def test_every_measured_denial_shape_still_names_its_path(self, line, expected):
        assert ld._denied_paths_in(line + "\n") == expected


# ===========================================================================
# Tier 2: the production wrapper under a repo-wide grant (F1 "Denied-write
# acceptance cases, which must pass before step 3b enables either mode")
# ===========================================================================

DENIED_CASES = [
    ("commit", "git -c user.email=a@b.invalid -c user.name=a commit -q --allow-empty -m x"),
    ("reset", "git reset -q --hard"),
    ("config", "git config user.name xx"),
    ("hook", 'echo hook > .git/hooks/pre-commit'),
    ("gitx", 'echo x > .git/x'),
    ("codex", 'echo x > .codex/x'),
    ("codexcfg", 'echo x >> .codex/config.toml'),
    ("providers", 'echo x >> .multi-llm/providers.yaml'),
    ("runs", 'echo x > .multi-llm/runs/x'),
    ("mtmp", 'echo x > .multi-llm/tmp/x'),
    ("state", 'echo x >> plans/plan/state.json'),
    ("prebatch", 'echo x > plans/plan/implement/pre_batch/t1/x'),
    ("blob", 'echo x >> plans/plan/implement/rollback_store/blob'),
    ("audit", 'echo x >> plans/plan/implement/subagent_audit.jsonl'),
    ("newtop", 'echo x > newtop.txt'),
    ("newtopdir", 'mkdir newtop'),
    ("rm", 'rm README.md'),
    ("mv", 'mv README.md README2.md'),
    ("gitlink", 'echo x >> gitlink'),
    ("sedi", "sed -i s/readme/x/ README.md"),
    ("plansnew", 'echo x > plans/new.md'),
    ("env", 'echo x >> .env'),
    ("nodemod", 'echo x > node_modules/dep/new.js'),
    ("tmp", 'echo x > /tmp/mllm-3b-$$'),
    ("shm", 'echo x > /dev/shm/mllm-3b-$$'),
]
POSITIVE_CASES = [
    ("edit", 'echo edited > src/app.py'),
    ("new", 'echo new > src/new.py'),
    ("subdir", 'mkdir src/sub && echo s > src/sub/s.py'),
    ("topfile", 'echo top >> README.md'),
    ("tests", 'echo t >> tests/test_app.py'),
    ("tmpdir", 'echo t > "$TMPDIR/scratch.txt"'),
]


def _script(cases):
    return "; ".join(f"({cmd}) >/dev/null 2>&1; echo {name}=$?" for name, cmd in cases)


@pytest.mark.landlock
class TestTheRealJailRepoWide:
    @pytest.fixture(autouse=True)
    def _needs_landlock(self):
        require_landlock()

    @pytest.fixture
    def sh(self):
        path = shutil.which("sh")
        if not path:  # pragma: no cover
            pytest.skip("no sh on PATH")
        return path

    def _run_jailed(self, wide, sh, script, declared=()):
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        with ld.slot_tmpdir(str(wide.repo)) as tmpdir:
            ctx = ld.complete_context(_request(wide, declared=declared), tmpdir, env)
            argv, child_env = ld.wrap_argv([sh, "-c", script], RUNG, ctx, env)
            assert os.path.samefile(argv[3], SKILL_ROOT / "utils" / "landlock_exec.py")
            proc = _run(argv, child_env, wide.repo)
            scratch = sorted(os.listdir(tmpdir))
        return proc, scratch

    def test_every_denied_write_case_holds_and_the_positives_land(self, wide, sh):
        protected = [wide.repo / ".git", wide.repo / ".codex", wide.repo / ".multi-llm", wide.out]
        before = {str(p): _tree_hashes(p) for p in protected}
        readme_ino = os.stat(wide.repo / "README.md").st_ino
        head = _git(wide.repo, "rev-parse", "HEAD")
        top_before = sorted(os.listdir(wide.repo))
        proc, scratch = self._run_jailed(
            wide, sh, _script(DENIED_CASES + POSITIVE_CASES),
            declared=(wide.repo / "src/app.py",),
        )
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        results = dict(line.split("=", 1) for line in proc.stdout.split())
        denied = {name for name, _ in DENIED_CASES}
        failed = {name for name, _ in DENIED_CASES if results[name] != "0"}
        assert failed == denied, {n: results[n] for n in denied - failed}
        for name, _ in POSITIVE_CASES:
            assert results[name] == "0", (name, results[name], proc.stderr)
        # Byte-identical afterwards: every protected tree, the top level, HEAD.
        assert {str(p): _tree_hashes(p) for p in protected} == before
        assert _git(wide.repo, "rev-parse", "HEAD") == head
        assert sorted(os.listdir(wide.repo)) == top_before
        assert (wide.repo / "README.md").read_text() == "readme\ntop\n"
        assert os.stat(wide.repo / "README.md").st_ino == readme_ino
        assert (wide.repo / ".env").read_text() == "SECRET=1\n"
        assert not (wide.repo / "node_modules/dep/new.js").exists()
        assert not (wide.repo / "plans/new.md").exists()
        assert (wide.repo / "plans/other.md").read_text() == "# other\n"
        # The positives.
        assert (wide.repo / "src/app.py").read_text() == "edited\n"
        assert (wide.repo / "src/new.py").read_text() == "new\n"
        assert (wide.repo / "src/sub/s.py").read_text() == "s\n"
        assert (wide.repo / "tests/test_app.py").read_text().endswith("t\n")
        assert "scratch.txt" in scratch
        assert not list(Path("/tmp").glob("mllm-3b-*"))

    def test_the_hooks_path_and_the_providers_config_are_denied(self, wide, sh, monkeypatch):
        """Fix round 4: a ``core.hooksPath`` outside ``.git/`` and a
        ``MULTI_LLM_PROVIDERS_CONFIG`` inside the repository are denied and
        byte-identical afterwards, while the declared edit beside them lands."""
        (wide.repo / ".husky").mkdir()
        (wide.repo / ".husky" / "pre-commit").write_text("#!/bin/sh\n")
        _git(wide.repo, "config", "core.hooksPath", ".husky")
        (wide.repo / "config").mkdir()
        config = wide.repo / "config" / "providers.yaml"
        config.write_text("driver:\n  verify_command: true\n")
        (wide.repo / "config" / "other.yaml").write_text("x\n")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        cases = [
            ("hook", "echo x >> .husky/pre-commit"),
            ("newhook", "echo x > .husky/post-commit"),
            ("cfg", "echo x >> config/providers.yaml"),
            ("cfgnew", "echo x > config/new.yaml"),
            ("cfgmv", "mv config/other.yaml config/providers.yaml"),
        ]
        positives = [("sibling", "echo y >> config/other.yaml"), ("edit", "echo edited > src/app.py")]
        before = {str(p): _tree_hashes(p) for p in (wide.repo / ".husky", wide.repo / ".git")}
        config_before = config.read_text()
        proc, _scratch = self._run_jailed(
            wide, sh, _script(cases + positives), declared=(wide.repo / "src/app.py",),
        )
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        results = dict(line.split("=", 1) for line in proc.stdout.split())
        assert all(results[name] != "0" for name, _ in cases), results
        assert all(results[name] == "0" for name, _ in positives), results
        assert {str(p): _tree_hashes(p) for p in (wide.repo / ".husky", wide.repo / ".git")} == before
        assert config.read_text() == config_before
        assert sorted(os.listdir(wide.repo / "config")) == ["other.yaml", "providers.yaml"]
        assert (wide.repo / "config" / "other.yaml").read_text() == "x\ny\n"
        assert (wide.repo / "src" / "app.py").read_text() == "edited\n"

    def test_the_same_script_unjailed_names_what_the_jail_stops(self, wide, sh):
        """The control: outside the jail the denied cases land, so the pass
        above is the wrapper's doing."""
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        with ld.slot_tmpdir(str(wide.repo)) as tmpdir:
            ctx = ld.complete_context(_request(wide), tmpdir, env)
            argv, child_env = ld.wrap_argv([sh, "-c", _script(DENIED_CASES[4:15])], RUNG, ctx, env)
            proc = _run(ld.unwrap_argv(argv), child_env, wide.repo)
        results = dict(line.split("=", 1) for line in proc.stdout.split())
        assert all(code == "0" for code in results.values()), results
        assert (wide.repo / "newtop.txt").exists() and (wide.repo / ".git" / "x").exists()

    def test_the_real_canary_passes_for_a_repo_wide_grant(self, wide, sh):
        git_before = _tree_hashes(wide.repo / ".git")
        out_before = _tree_hashes(wide.out)
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        resolved = [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            assert probe.env[codex_mod.CANARY_ENV_DENY_DIRS].split("\n")[0] == str(wide.repo / ".git")
            proc = _run(probe.argv, probe.env, wide.repo)
            assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
            assert SANDBOX_CANARY_DENIED in proc.stdout
            assert probe.seed_intact()
            unjailed = _run(ld.unwrap_argv(probe.argv), probe.env, wide.repo)
            assert unjailed.returncode == 3, (unjailed.stdout, unjailed.stderr)
        assert _tree_hashes(wide.repo / ".git") == git_before
        assert _tree_hashes(wide.out) == out_before
        assert sorted(os.listdir(wide.repo / "src")) == ["app.py", "pkg"]

    def test_the_real_canary_passes_with_an_unwritable_first_directory(self, wide, sh):
        """Repro case B (2026-10-02): the first sorted directory 0555 is a
        plain-permissions fact the jail has no part in, so the probe passes
        and nothing was ever created outside `.multi-llm/` and `$TMPDIR`."""
        first = wide.repo / ".husky"
        first.mkdir()
        (first / "pre-commit").write_text("#!/bin/sh\n")
        first.chmod(0o555)
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        resolved = [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT]
        try:
            with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
                assert codex_mod.CANARY_ENV_WRITE_DIR not in probe.env
                proc = _run(probe.argv, probe.env, wide.repo)
        finally:
            first.chmod(0o755)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout
        assert sorted(os.listdir(first)) == ["pre-commit"]

    def test_a_write_into_git_alone_fails_the_canary(self, wide, sh):
        """A jail that left `.git/` writable (no deny target) must not pass:
        the extra negative control is what catches it. Simulated by handing
        the canary a deny dir that IS writable (the slot's own TMPDIR)."""
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        resolved = [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            leaky = dict(probe.env)
            leaky[codex_mod.CANARY_ENV_DENY_DIRS] = probe.env["TMPDIR"]
            proc = _run(probe.argv, leaky, wide.repo)
            assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
            assert SANDBOX_CANARY_DENIED not in proc.stdout


class TestAnInheritedPytestAddopts:
    def test_the_redirect_is_prepended_to_the_inherited_value(self, wide):
        """The parent's ``PYTEST_ADDOPTS`` (``-p no:cacheprovider
        --import-mode=importlib`` in a CI job, say) used to be clobbered by
        the redirect, so the child's pytest ran without the options the
        project relies on. The redirect is prepended; the inherited value
        keeps the last word, as argv does."""
        inherited = "-p no:cacheprovider --import-mode=importlib"
        _argv, env = ld.wrap_argv(["codex", "exec"], RUNG, _ctx(wide),
                                  {"PATH": "/bin", "PYTEST_ADDOPTS": inherited})
        assert env["PYTEST_ADDOPTS"] == (
            f"-o cache_dir={wide.slot / 'pytest-cache'} {inherited}"
        )

    def test_an_empty_inherited_value_is_just_the_redirect(self, wide):
        _argv, env = ld.wrap_argv(["codex", "exec"], RUNG, _ctx(wide),
                                  {"PATH": "/bin", "PYTEST_ADDOPTS": ""})
        assert env["PYTEST_ADDOPTS"] == f"-o cache_dir={wide.slot / 'pytest-cache'}"
