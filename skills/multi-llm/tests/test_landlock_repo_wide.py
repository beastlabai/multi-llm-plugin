"""The repo-wide grant (plan step 3b, 2026-10-02; revised 2026-10-05): the
dispatch half.

Plan ``codex-sandbox-unavailable-findings-and-fixes`` §6 F1 "Editing turns
that run tests and builds" (step 1b's decision), "Allow-list entries at
dispatch" (the ``repo-wide`` row) and §3's 2026-10-05 decision: the grant is
**one dir rule on the repository root**. Until then it was one rule per
enumerated root entry outside a protected set, which kept the root unruled
and so refused a declared new top-level file before dispatch — the racing
re-run's whole plan was one root-level ``index.html``. Landlock is allow-only
and a rule inherits down the tree, so the root rule necessarily covers
``.git``, ``.codex``, ``.multi-llm`` and the run's output folder; the user
chose that over a backup-and-restore of the protected set around every
batch. What this file pins is ``landlock_dispatch``'s share — the root entry,
step 1b's redirects, the probe's out-of-repository negative controls, the
developer message and the stream reading — and, at tier 2, the production
wrapper admitting every write inside the repository while denying every
write outside it.

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
    """A committed scratch repo with everything the hands-off set names, a
    denied entry of each kind, two symlinks, the output folder under
    ``plans/``, and ordinary files and directories."""
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


def _request(wide, *, output_dir=True, kind="implement"):
    return JailRequest(
        repo_root=str(wide.repo), kind=kind, write_scope=WRITE_SCOPE_REPO_WIDE,
        output_dir=str(wide.out) if output_dir else None,
    )


def _ctx(wide, **kw):
    return JailContext(request=_request(wide, **kw), tmpdir=str(wide.slot),
                       codex_home=str(wide.home))


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
# Tier 1: the root grant (2026-10-05)
# ===========================================================================

class TestTheRootGrant:
    def test_the_allow_list_is_base_plus_one_required_rule_on_the_root(self, wide):
        entries = ld.allow_entries(_ctx(wide))
        assert [(e.path, e.kind, e.required) for e in entries[:5]] == [
            (str(wide.home), "dir", True), (str(wide.slot), "dir", True),
            ("/dev/null", "file", True), ("/dev/ptmx", "file", False), ("/dev/pts", "dir", False),
        ]
        assert [(e.path, e.kind, e.required) for e in entries[5:]] == [
            (str(wide.repo), "dir", True),
        ]
        assert entries[5].why.startswith("repo-wide: ")
        argv, _env = ld.wrap_argv(["codex", "exec"], RUNG, _ctx(wide), {})
        assert argv[argv.index(f"dir:{wide.repo}") - 1] == "--allow"
        # Nothing below the root gets a rule of its own any more.
        assert not any(str(wide.repo / "src") in a for a in argv)

    def test_the_root_is_realpath_resolved(self, wide):
        dotted = wide.repo / "plans" / ".." / "src" / ".."
        request = JailRequest(repo_root=str(dotted), kind="implement",
                              write_scope=WRITE_SCOPE_REPO_WIDE, output_dir=str(wide.out))
        assert ld.repo_wide_entry(request).path == str(wide.repo)

    def test_a_missing_repo_root_is_refused(self, wide):
        request = JailRequest(repo_root=str(wide.repo / "gone"), kind="implement",
                              write_scope=WRITE_SCOPE_REPO_WIDE)
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.allow_entries(JailContext(request, str(wide.slot), str(wide.home)))
        assert excinfo.value.reason == "repo-root-missing"

    def test_the_hands_off_names_are_instruction_level_only(self, wide):
        """The old protected set is now the developer message's: no entry
        for it, no refusal of a declared path under it."""
        assert ld.REPO_WIDE_HANDS_OFF_NAMES == (".git", ".codex", ".multi-llm")
        assert ld.VCS_DIR_NAMES == wbs.SWEEP_PRUNED_DIR_NAMES
        assert not hasattr(ld, "PROTECTED_ROOT_NAMES")
        assert not hasattr(ld, "repo_wide_declared_refusal")
        assert not hasattr(ld, "repo_wide_plan")
        assert not hasattr(ld, "extra_protected_paths")
        assert not hasattr(ld, "REPO_WIDE_F6")

    def test_the_two_narrow_scopes_still_never_rule_the_root(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="apply_suggestions",
                              write_scope=WRITE_SCOPE_PLAN_FILE, target=str(wide.repo))
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.allow_entries(JailContext(request, str(wide.slot), str(wide.home)))
        assert excinfo.value.reason in ("repo-root", "target-missing")

    def test_an_unknown_scope_is_still_refused(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="implement",
                              write_scope="everything", target=str(wide.repo / "README.md"))
        with pytest.raises(ld.JailRefused) as excinfo:
            ld.allow_entries(JailContext(request, str(wide.slot), str(wide.home)))
        assert excinfo.value.reason == "scope-not-served"

    def test_the_request_carries_no_per_task_path(self):
        fields = set(JailRequest.__dataclass_fields__)
        assert "declared" not in fields and "deny_patterns" not in fields


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
    def test_the_preview_names_the_root_rule_and_computes_nothing(self, wide):
        argv = ld.preview_wrap_argv(["codex", "exec"], RUNG, _request(wide))
        assert argv[argv.index(f"dir:{wide.repo}") - 1] == "--allow"
        assert [a for a in argv if str(wide.repo) in a] == [f"dir:{wide.repo}"]
        assert not hasattr(ld, "PLACEHOLDER_REPO_ENTRIES")

    def test_the_developer_message_says_the_repository_is_writable_and_what_to_leave_alone(self, wide):
        text = codex_mod.jail_developer_instructions(JailContext(_request(wide), str(wide.slot)))
        assert f"The whole repository {wide.repo} is writable" in text
        assert "including directly in its root" in text
        assert "every write outside it is denied" in text
        assert f"this run's output folder {wide.out}" in text
        assert "Do NOT modify .git/" in text and ".codex/, .multi-llm/" in text
        assert "no edits to .git/config or .git/hooks/" in text
        assert "the jail does not enforce this" in text
        assert "/tmp and /dev/shm are denied" in text
        assert "uv, pytest, npm and Python bytecode already use caches under $TMPDIR" in text
        assert "never try sudo or chown" in text
        # The old enumeration's wording and the narrow scopes' sentences are gone.
        for absent in ("You may write only", "is the directory", "is not writable",
                       "never replaced", "including anywhere in the repository, is denied",
                       "are denied."):
            assert absent not in text, absent

    def test_a_plan_file_message_is_byte_identical_to_before(self, wide):
        request = JailRequest(repo_root=str(wide.repo), kind="apply_suggestions",
                              write_scope=WRITE_SCOPE_PLAN_FILE,
                              target=str(wide.repo / "plans" / "plan.md"))
        text = codex_mod.jail_developer_instructions(JailContext(request, str(wide.slot)))
        assert text == (
            "Filesystem writes are restricted, whatever your environment context says "
            f"about sandboxing. Write scratch files only under {wide.slot} (also $TMPDIR). "
            "Every other write, including anywhere in the repository, is denied."
            f" You may write only {wide.repo / 'plans' / 'plan.md'}. Edit it in place with "
            "apply_patch, or by rewriting it through a shell redirect. Commands that "
            "replace the file are denied and leave it unchanged: sed -i, perl -i, editors "
            "or scripts that save through a temporary file and a rename, mv over it, or "
            "deleting and re-creating it. Git commands that write .git/ (add, commit, "
            "stash, checkout, restore, fetch) are denied. npm and npx use a per-dispatch "
            "cache under $TMPDIR. A Permission denied error here comes from this "
            "restriction, not from file ownership: never try sudo or chown to work "
            "around it."
        )
        assert "whole repository" not in text and "/dev/shm" not in text


class TestTheCanary:
    def test_canary_env_hands_over_no_target_in_the_tree(self, wide):
        env = ld.canary_env(_ctx(wide))
        assert env == {}
        assert not hasattr(codex_mod, "CANARY_ENV_DENY_DIRS")

    def test_the_rung_probe_moves_its_negative_controls_outside_the_repository(self, wide, temp_base):
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(wide.root),
               "CODEX_HOME": str(wide.home)}
        resolved = ["/opt/codex/bin/codex", *codex_mod.PLUGIN_LANDLOCK_PROBE_ARGV[1:]]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            workdir = probe.env[codex_mod.CANARY_ENV_WORKDIR]
            assert Path(workdir).is_dir()
            assert Path(workdir).parent == temp_base
            assert workdir != probe.env["TMPDIR"]
            assert Path(probe.seeded).parent == Path(workdir)
            assert probe.seed_intact()
            assert codex_mod.CANARY_ENV_WRITE_DIR not in probe.env
            assert codex_mod.CANARY_ENV_WRITE_FILE not in probe.env
            allowed = [a for a in probe.argv if a.startswith(("dir:", "file:"))]
            assert f"dir:{wide.repo}" in allowed
            assert not any(workdir in a for a in allowed)
        assert not Path(workdir).exists()
        # The canary script reads the override ahead of its default workdir.
        assert codex_mod.CANARY_ENV_WORKDIR in codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT
        assert codex_mod.LANDLOCK_CANARY_WORKDIR in codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT

    def test_a_plan_file_probe_keeps_the_in_repo_workdir(self, wide):
        (wide.repo / ".multi-llm" / "tmp").mkdir(exist_ok=True)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(wide.root),
               "CODEX_HOME": str(wide.home)}
        request = JailRequest(repo_root=str(wide.repo), kind="apply_suggestions",
                              write_scope=WRITE_SCOPE_PLAN_FILE,
                              target=str(wide.repo / "plans" / "plan.md"))
        resolved = ["/opt/codex/bin/codex", *codex_mod.PLUGIN_LANDLOCK_PROBE_ARGV[1:]]
        with ld.rung_probe(RUNG, request, resolved, cwd=str(wide.repo), env=env) as probe:
            assert codex_mod.CANARY_ENV_WORKDIR not in probe.env
            assert Path(probe.seeded).parent == wide.repo / ".multi-llm" / "tmp"


def _plain_repo(root: Path, *dirs: str) -> SimpleNamespace:
    """A repo for the permissions-jailed canary: a `.git` directory (made, not
    initialised), the in-repo workdir the walk would have prepared, the
    directories named, and a codex home beside it."""
    repo = root / "repo"
    for d in (".git", ".multi-llm/tmp", *dirs):
        (repo / d).mkdir(parents=True, exist_ok=True)
    home = root / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(root=root, repo=repo, home=home)


@contextlib.contextmanager
def _perms_jail(probe):
    """Tier 1's stand-in for the kernel: inside an entered `rung_probe`, the
    seeded file goes 0444 and its directory — the probe's workdir — 0555, so
    a write there fails on plain permissions exactly as the jail would deny
    it, while `$TMPDIR` stays writable. Restored before the context (and the
    probe's own clean-up) runs on."""
    seeded = Path(probe.seeded)
    workdir = seeded.parent
    try:
        seeded.chmod(0o444)
        workdir.chmod(0o555)
        yield
    finally:
        workdir.chmod(0o755)
        seeded.chmod(0o644)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="plain permissions deny nothing to root")
class TestTheCanaryUnderPlainPermissions:
    """The canary script's reading of its workdir, with the kernel stood in
    for by file permissions: a repo-wide probe's negative controls live in
    the out-of-repository directory `$MULTI_LLM_CANARY_WORKDIR` names."""

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

    def test_the_controls_hold_when_the_outside_workdir_is_denied(self, tmp_path, sh):
        plain = _plain_repo(tmp_path, "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe):
            assert Path(probe.env[codex_mod.CANARY_ENV_WORKDIR]) == Path(probe.seeded).parent
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout
        # Nothing was created in the repository's own `.multi-llm/tmp`.
        assert os.listdir(plain.repo / ".multi-llm" / "tmp") == []

    def test_a_writable_in_repo_workdir_does_not_fail_a_repo_wide_probe(self, tmp_path, sh):
        """The root grant makes `.multi-llm/tmp` writable: the canary's
        negative control must not look there, or a working jail reads as
        "not denied"."""
        plain = _plain_repo(tmp_path, "src")
        with self._probe(plain, sh) as probe, _perms_jail(probe):
            assert os.access(plain.repo / ".multi-llm" / "tmp", os.W_OK)
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout

    def test_a_writable_outside_workdir_fails_the_probe(self, tmp_path, sh):
        """A jail that left the out-of-repository directory writable (no
        boundary at all) must not pass."""
        plain = _plain_repo(tmp_path, "src")
        with self._probe(plain, sh) as probe:
            proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED not in proc.stdout
        assert "not denied" in proc.stderr

    def test_a_repository_path_with_a_space_or_a_colon_is_fine(self, tmp_path, sh):
        for name in ("with space", "a:b"):
            plain = _plain_repo(tmp_path / name, "src")
            with self._probe(plain, sh) as probe, _perms_jail(probe):
                proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
            assert proc.returncode == 0, (name, proc.returncode, proc.stdout, proc.stderr)
            assert SANDBOX_CANARY_DENIED in proc.stdout

    def test_an_unwritable_first_directory_does_not_fail_the_probe(self, tmp_path, sh):
        """The probe's positive control never writes into the source tree
        (adversarial review, 2026-10-02, repro case B)."""
        plain = _plain_repo(tmp_path, ".github", "src")
        (plain.repo / ".github").chmod(0o555)
        try:
            with self._probe(plain, sh) as probe, _perms_jail(probe):
                assert codex_mod.CANARY_ENV_WRITE_DIR not in probe.env
                proc = _run(ld.unwrap_argv(probe.argv), probe.env, plain.repo)
        finally:
            (plain.repo / ".github").chmod(0o755)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert SANDBOX_CANARY_DENIED in proc.stdout
        assert os.listdir(plain.repo / ".github") == []


# ===========================================================================
# Tier 1: the stream under a repo-wide grant
# ===========================================================================


def _cmd(output: str) -> str:
    return json.dumps({"type": "item.completed", "item": {
        "type": "command_execution", "command": "sh", "aggregated_output": output,
        "exit_code": 1, "status": "failed"}})


class TestJailDeniedEditsRepoWide:
    def test_a_denial_inside_the_repository_is_in_scope_and_one_outside_is_not(self, wide):
        """The root rule covers everything under the repository, so a denial
        there is the jail's own failure (in scope); `/tmp` and the like are
        the jail working (out of scope). No `replacement` reading survives:
        a dir rule admits every write shape."""
        stream = "\n".join([
            _cmd("sh: 1: cannot create src/app.py: Permission denied\n"),
            _cmd("sh: 1: cannot create README.md.tmp: Permission denied\n"),
            _cmd("fatal: Unable to create '.git/index.lock': Permission denied\n"),
            _cmd("sh: 1: cannot create newtop.txt: Permission denied\n"),
            _cmd("sh: 1: cannot create plans/plan/state.json: Permission denied\n"),
            _cmd("PermissionError: [Errno 13] Permission denied: '/tmp/x'\n"),
            _cmd("sh: 1: cannot create /dev/shm/x: Permission denied\n"),
        ])
        denials = ld.jail_denied_edits(stream, _ctx(wide), cwd=str(wide.repo))
        assert denials.in_scope_paths == [
            str(wide.repo / "src/app.py"), str(wide.repo / "README.md.tmp"),
            str(wide.repo / ".git/index.lock"), str(wide.repo / "newtop.txt"),
            str(wide.repo / "plans/plan/state.json"),
        ]
        assert denials.replacement_paths == []
        assert denials.out_of_scope_count == 2
        assert denials.out_of_scope_paths == ["/tmp/x", "/dev/shm/x"]
        assert denials.failed_file_changes == []

    def test_a_symlink_out_of_the_repository_is_out_of_scope(self, wide):
        stream = _cmd("sh: 1: cannot create outlink/passwd: Permission denied\n")
        denials = ld.jail_denied_edits(stream, _ctx(wide), cwd=str(wide.repo))
        assert denials.in_scope_paths == [] and denials.out_of_scope_count == 1

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
        test output used to resolve to a path inside the repository (in
        scope, so a nothing-moved batch read FAILED). Only the shapes the
        measured denials take — a tool naming itself at column 0 — count."""
        denials = ld.jail_denied_edits(_cmd(self.ORDINARY_OUTPUT), _ctx(wide), cwd=str(wide.repo))
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
# Tier 2: the production wrapper under a repo-wide grant (the root rule)
# ===========================================================================

#: Writes outside the repository: still denied by the jail.
DENIED_CASES = [
    ("tmp", 'echo x > /tmp/mllm-3b-$$'),
    ("shm", 'echo x > /dev/shm/mllm-3b-$$'),
    ("home", 'echo x > "$HOME/mllm-3b-$$"'),
    ("outlink", 'echo x > outlink/mllm-3b-$$'),
    ("tmpbase", 'echo x > "$(dirname "$TMPDIR")/mllm-3b-$$"'),
]
#: Writes inside the repository: every shape lands under the root rule,
#: the old protected set and the two top-level shapes included. What the
#: model is told not to touch is the developer message's business and the
#: scheduler's checks', not the jail's.
POSITIVE_CASES = [
    ("edit", 'echo edited > src/app.py'),
    ("new", 'echo new > src/new.py'),
    ("subdir", 'mkdir src/sub && echo s > src/sub/s.py'),
    ("topfile", 'echo top >> README.md'),
    ("newtop", 'echo x > newtop.txt'),
    ("newtopdir", 'mkdir newtop'),
    ("sedi", "sed -i s/readme/x/ README.md"),
    ("mv", 'mv pyproject.toml pyproject.moved'),
    ("rm", 'rm key.pem'),
    ("plansnew", 'echo x > plans/new.md'),
    ("gitx", 'echo x > .git/x'),
    ("codex", 'echo x > .codex/x'),
    ("mtmp", 'echo x > .multi-llm/tmp/x'),
    ("state", 'echo x >> plans/plan/state.json'),
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

    def _run_jailed(self, wide, sh, script):
        env = dict(os.environ, CODEX_HOME=str(wide.home), HOME=str(wide.root))
        with ld.slot_tmpdir(str(wide.repo)) as tmpdir:
            ctx = ld.complete_context(_request(wide), tmpdir, env)
            argv, child_env = ld.wrap_argv([sh, "-c", script], RUNG, ctx, env)
            assert os.path.samefile(argv[3], SKILL_ROOT / "utils" / "landlock_exec.py")
            proc = _run(argv, child_env, wide.repo)
            scratch = sorted(os.listdir(tmpdir))
        return proc, scratch

    def test_every_write_outside_the_repository_is_denied_and_every_one_inside_lands(self, wide, sh):
        head = _git(wide.repo, "rev-parse", "HEAD")
        proc, scratch = self._run_jailed(wide, sh, _script(DENIED_CASES + POSITIVE_CASES))
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        results = dict(line.split("=", 1) for line in proc.stdout.split())
        denied = {name for name, _ in DENIED_CASES}
        failed = {name for name, _ in DENIED_CASES if results[name] != "0"}
        assert failed == denied, {n: results[n] for n in denied - failed}
        for name, _ in POSITIVE_CASES:
            assert results[name] == "0", (name, results[name], proc.stderr)
        # Outside: nothing landed.
        assert not list(Path("/tmp").glob("mllm-3b-*"))
        assert not list(wide.root.glob("mllm-3b-*"))
        assert not list(Path("/etc").glob("mllm-3b-*"))
        # Inside: the root grant admitted every shape, the hands-off set included.
        assert (wide.repo / "src/app.py").read_text() == "edited\n"
        assert (wide.repo / "src/new.py").read_text() == "new\n"
        assert (wide.repo / "src/sub/s.py").read_text() == "s\n"
        assert (wide.repo / "README.md").read_text() == "x\ntop\n"
        assert (wide.repo / "newtop.txt").read_text() == "x\n"
        assert (wide.repo / "newtop").is_dir()
        assert (wide.repo / "pyproject.moved").exists() and not (wide.repo / "pyproject.toml").exists()
        assert not (wide.repo / "key.pem").exists()
        assert (wide.repo / "plans/new.md").read_text() == "x\n"
        assert (wide.repo / ".git/x").read_text() == "x\n"
        assert (wide.repo / ".codex/x").read_text() == "x\n"
        assert (wide.repo / ".multi-llm/tmp/x").read_text() == "x\n"
        assert (wide.repo / "plans/plan/state.json").read_text() == "{}\nx\n"
        assert "scratch.txt" in scratch
        # A shell script wrote files, not commits: HEAD did not move.
        assert _git(wide.repo, "rev-parse", "HEAD") == head

    def test_the_same_script_unjailed_names_what_the_jail_stops(self, wide, sh):
        """The control: outside the jail the denied cases land too, so the
        denials above are the wrapper's doing."""
        env = dict(os.environ, CODEX_HOME=str(wide.home), HOME=str(wide.root))
        with ld.slot_tmpdir(str(wide.repo)) as tmpdir:
            ctx = ld.complete_context(_request(wide), tmpdir, env)
            argv, child_env = ld.wrap_argv([sh, "-c", _script(DENIED_CASES[2:3])], RUNG, ctx, env)
            proc = _run(ld.unwrap_argv(argv), child_env, wide.repo)
        results = dict(line.split("=", 1) for line in proc.stdout.split())
        assert results == {"home": "0"}, results
        assert list(wide.root.glob("mllm-3b-*"))

    def test_the_real_canary_passes_for_a_repo_wide_grant(self, wide, sh, temp_base):
        git_before = _tree_hashes(wide.repo / ".git")
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        resolved = [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            workdir = probe.env[codex_mod.CANARY_ENV_WORKDIR]
            assert Path(workdir).parent == temp_base
            proc = _run(probe.argv, probe.env, wide.repo)
            assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
            assert SANDBOX_CANARY_DENIED in proc.stdout
            assert probe.seed_intact()
            unjailed = _run(ld.unwrap_argv(probe.argv), probe.env, wide.repo)
            assert unjailed.returncode == 3, (unjailed.stdout, unjailed.stderr)
        assert not Path(workdir).exists()
        assert _tree_hashes(wide.repo / ".git") == git_before
        assert sorted(os.listdir(wide.repo / "src")) == ["app.py", "pkg"]
        assert os.listdir(wide.repo / ".multi-llm" / "tmp") == []

    def test_the_real_canary_passes_with_an_unwritable_first_directory(self, wide, sh):
        """Repro case B (2026-10-02): the first sorted directory 0555 is a
        plain-permissions fact the jail has no part in."""
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

    def test_a_canary_pointed_at_the_in_repo_workdir_would_fail(self, wide, sh):
        """Why the workdir moved: under the root rule the repository's
        `.multi-llm/tmp` is writable, so a canary there reads "not denied".
        Simulated by handing the jailed canary the in-repo directory."""
        env = dict(os.environ, CODEX_HOME=str(wide.home))
        resolved = [sh, "-c", codex_mod.PLUGIN_LANDLOCK_CANARY_SCRIPT]
        with ld.rung_probe(RUNG, _request(wide), resolved, cwd=str(wide.repo), env=env) as probe:
            leaky = dict(probe.env)
            leaky[codex_mod.CANARY_ENV_WORKDIR] = str(wide.repo / ".multi-llm" / "tmp")
            proc = _run(probe.argv, leaky, wide.repo)
            assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
            assert SANDBOX_CANARY_DENIED not in proc.stdout
        assert os.listdir(wide.repo / ".multi-llm" / "tmp") == []


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
