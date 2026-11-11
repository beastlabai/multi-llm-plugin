"""The suite's probe guard (``tests/conftest.py``) never runs the production
Landlock wrapper, wherever the checkout lives and however a probe launches it.

The guard recognises the production wrapper by identity, fixed when the
conftest is imported: the canonical path, inode and bytes of every copy the
checkout holds (the source tree's and each generated package's). Not by
"outside the temp dir": a checkout that itself sits under the temp dir (an
agent worktree, a scratch copy) would otherwise run the real wrapper, take the
plugin-landlock rung on a Landlock-capable kernel and fail every test that
expects the rung `absent`. Relative words resolve against the probe's own
``cwd``. A probe whose program is shell text is refused unless the test opts in
with ``@pytest.mark.shell_probe``. A test's own fake under ``tmp_path`` still
runs for real, even while ``WRAPPER_PATHS`` points at it.

The installed-guard tests swap the spawn for a recorder, so a guard that lets
a probe through records it instead of running the real wrapper.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import conftest as suite_conftest
from utils import landlock_dispatch
import utils.subagent_executor as executor
import utils.sandbox_walk as sandbox_walk

SKILL_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_WRAPPER = SKILL_ROOT / "utils" / "landlock_exec.py"
GUARD_NOTE = "the default suite never runs the production Landlock wrapper"
SHELL_NOTE = "a shell-form probe runs only in a test marked @pytest.mark.shell_probe"


def _wrapped(wrapper, *flags) -> list:
    return [sys.executable, "-I", "-S", *flags, str(wrapper), "--allow", "dir:/x",
            "--", "codex", "sandbox", "linux", "true"]


def _canonical_dir(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def _fake_wrapper(tmp_path: Path, name: str = "fake_landlock_exec.py") -> Path:
    fake = tmp_path / name
    fake.write_text("import sys\nprint('fake ran')\nsys.exit(0)\n", encoding="utf-8")
    return fake


@pytest.fixture
def spawned(monkeypatch):
    """Every probe the guard lets through to the real spawn, recorded instead
    of run (``subprocess.run`` on POSIX, the job-object runner on Windows)."""
    calls = []

    def record(argv, **kwargs):
        calls.append([str(a) for a in argv])
        return subprocess.CompletedProcess(list(argv), 0, "recorded", "")

    monkeypatch.setattr(executor.subprocess, "run", record)
    monkeypatch.setattr(sandbox_walk, "_run_sandbox_probe_tree", record)
    return calls


def _probe(argv, cwd, env=None):
    return executor._run_sandbox_probe(argv, cwd=str(cwd), timeout=10, env=env)


def _assert_blocked(result, spawned):
    assert result.returncode == 126
    assert GUARD_NOTE in result.stderr
    assert spawned == []


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"no symlinks here: {exc}")


# --- the snapshot -------------------------------------------------------------


def test_the_snapshot_is_the_real_wrapper_in_this_checkout():
    snapshot = suite_conftest._PRODUCTION_WRAPPERS
    assert any(os.path.samefile(path, PRODUCTION_WRAPPER) for path in snapshot.paths)
    assert os.path.basename(PRODUCTION_WRAPPER) in snapshot.names
    assert snapshot.digests and snapshot.inodes and snapshot.sizes


@pytest.mark.parametrize("run_from", ["source", "codex-package", "opencode-package"])
def test_the_snapshot_holds_every_copy_in_the_checkout_stale_or_not(
    tmp_path, monkeypatch, run_from
):
    # A checkout laid out like this repository: the source payload and two
    # generated packages, one of them stale (other bytes).
    payload = Path("skills", "multi-llm", "utils")
    copies = {
        "source": tmp_path / payload,
        "codex-package": tmp_path / "packaging" / "codex" / "multi-llm" / payload,
        "opencode-package": tmp_path / "packaging" / "opencode" / payload,
    }
    for name, directory in copies.items():
        directory.mkdir(parents=True)
        (directory / "landlock_exec.py").write_text(f"# {name}\n", encoding="utf-8")
        (directory / "landlock_dispatch.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        landlock_dispatch, "__file__", str(copies[run_from] / "landlock_dispatch.py")
    )

    found = {os.path.realpath(p) for p in suite_conftest._production_wrapper_files()}

    assert found == {
        os.path.realpath(d / "landlock_exec.py") for d in copies.values()
    }


# --- where the checkout lives -------------------------------------------------


def test_a_temp_dir_that_contains_the_checkout_still_blocks_the_production_wrapper():
    # The checkout's own parent stands in for a temp dir that contains it.
    temp_root = _canonical_dir(SKILL_ROOT.parent)
    assert suite_conftest._under(str(PRODUCTION_WRAPPER), temp_root)
    assert suite_conftest._probe_guard_verdict(
        _wrapped(PRODUCTION_WRAPPER), cwd=None, temp_root=temp_root
    ) == "wrapper"


def test_every_packaged_copy_is_blocked_from_a_checkout_under_the_temp_dir():
    here = os.path.realpath(PRODUCTION_WRAPPER)
    others = [
        p for p in suite_conftest._production_wrapper_files()
        if os.path.realpath(p) != here
    ]
    if not others:
        pytest.skip("this checkout holds no generated package")
    temp_root = _canonical_dir(Path(os.path.commonpath([here, *others])).parent)
    for copy in others:
        # Found at import by path, so a stale copy (other bytes) counts too.
        assert suite_conftest._canonical_file(copy) in (
            suite_conftest._PRODUCTION_WRAPPERS.paths
        ), copy
        assert suite_conftest._probe_guard_verdict(
            _wrapped(copy), cwd=None, temp_root=temp_root
        ) == "wrapper", copy


def test_a_fake_under_the_temp_dir_runs_and_one_outside_it_is_blocked(
    tmp_path, monkeypatch
):
    fake = _fake_wrapper(tmp_path)
    monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(fake))
    assert suite_conftest._probe_guard_verdict(
        _wrapped(fake), cwd=None, temp_root=_canonical_dir(tmp_path.parent)
    ) is None
    # Outside the temp dir: blocked, exactly as before the identity check.
    assert suite_conftest._probe_guard_verdict(
        _wrapped(fake), cwd=None, temp_root=_canonical_dir(SKILL_ROOT)
    ) == "wrapper"


def test_the_installed_guard_answers_126_while_the_table_points_at_a_fake(
    tmp_path, monkeypatch
):
    fake = _fake_wrapper(tmp_path)
    monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(fake))

    blocked = executor._run_sandbox_probe(
        _wrapped(PRODUCTION_WRAPPER), cwd=str(tmp_path), timeout=10
    )
    assert blocked.returncode == 126
    assert GUARD_NOTE in blocked.stderr

    ran = executor._run_sandbox_probe(_wrapped(fake), cwd=str(tmp_path), timeout=10)
    assert ran.returncode == 0
    assert "fake ran" in ran.stdout


# --- relative words resolve against the child's cwd ---------------------------


@pytest.mark.parametrize("table", ["production", "fake-with-another-name"])
def test_a_relative_wrapper_resolves_against_the_child_cwd(
    tmp_path, monkeypatch, spawned, table
):
    # pytest sits in the temp dir; the child starts in the skill root, where
    # `utils/landlock_exec.py` is the production wrapper.
    monkeypatch.chdir(tmp_path)
    if table != "production":
        fake = _fake_wrapper(tmp_path)
        monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(fake))
    argv = [sys.executable, "-I", "-S", os.path.join("utils", "landlock_exec.py"),
            "--", "true"]

    _assert_blocked(_probe(argv, cwd=SKILL_ROOT), spawned)


def test_a_relative_fake_in_the_child_cwd_runs_beside_a_real_one_in_pytest_s_cwd(
    tmp_path, monkeypatch, spawned
):
    # The other way round: pytest sits beside the production wrapper, the
    # child starts where the same relative word is a fake.
    child = tmp_path / "child"
    child.mkdir()
    fake = _fake_wrapper(child, name="landlock_exec.py")
    monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(fake))
    monkeypatch.chdir(PRODUCTION_WRAPPER.parent)
    argv = [sys.executable, "-I", "-S", "landlock_exec.py", "--", "true"]

    result = _probe(argv, cwd=child)

    assert result.returncode == 0
    assert [call[1:] for call in spawned] == [argv[1:]]


# --- copies and links are the wrapper -----------------------------------------


@pytest.mark.parametrize("name", ["landlock_exec.py", "anything.py"])
def test_a_byte_for_byte_copy_under_the_temp_dir_is_the_wrapper(
    tmp_path, spawned, name
):
    copy = tmp_path / name
    shutil.copyfile(PRODUCTION_WRAPPER, copy)

    _assert_blocked(_probe(_wrapped(copy), cwd=tmp_path), spawned)


def test_a_hard_link_is_the_wrapper_by_inode_whatever_its_bytes_become(
    tmp_path, monkeypatch, spawned
):
    # A stand-in production file whose snapshot records its inode alone (no
    # bytes), so only the inode rule can recognise the link: a hard link to
    # the real wrapper cannot be made across the checkout's filesystem and the
    # temp dir's.
    real = tmp_path / "checkout" / "real_wrapper.py"
    real.parent.mkdir()
    real.write_text("# real\n", encoding="utf-8")
    info = os.stat(real)
    if not info.st_ino:
        pytest.skip("this filesystem reports no inode numbers")
    monkeypatch.setattr(suite_conftest, "_PRODUCTION_WRAPPERS", suite_conftest._WrapperIdentity(
        paths=frozenset({suite_conftest._canonical_file(str(real))}),
        inodes=frozenset({(info.st_dev, info.st_ino)}),
        digests=frozenset(), sizes=frozenset(), names=frozenset({"real_wrapper.py"}),
    ))
    link = tmp_path / "linked.py"
    try:
        os.link(real, link)
    except OSError as exc:
        pytest.skip(f"no hard links here: {exc}")
    link.write_text("# edited after the snapshot\n", encoding="utf-8")

    _assert_blocked(_probe(_wrapped(link), cwd=tmp_path), spawned)


# --- launch forms -------------------------------------------------------------


def test_a_suffixless_symlink_to_the_wrapper_is_the_wrapper(tmp_path, spawned):
    link = tmp_path / "wrapper"
    _symlink_or_skip(link, PRODUCTION_WRAPPER)

    _assert_blocked(_probe(_wrapped(link), cwd=tmp_path), spawned)


def test_the_wrapper_past_the_first_six_words_is_the_wrapper(tmp_path, spawned):
    argv = _wrapped(PRODUCTION_WRAPPER, "-B", "-E", "-s", "-u")
    assert argv.index(str(PRODUCTION_WRAPPER)) > 5

    _assert_blocked(_probe(argv, cwd=tmp_path), spawned)


def _indirect_env() -> dict:
    return {**os.environ, "PY": sys.executable, "WRAPPER": str(PRODUCTION_WRAPPER)}


INDIRECT_SH = ["sh", "-c", 'exec "$PY" -I -S "$WRAPPER" -- true']


@pytest.mark.parametrize("argv", [
    INDIRECT_SH,
    ["sh", "-c", "exit 0"],
    [sys.executable, "-c", "print('text')"],
    [sys.executable, "-I", "-m", "json.tool"],
    ["env", "-S", "sh -c true"],
    ["env", "A=1", sys.executable, "-I", "-S", "probe.py"],
    ["timeout", "5", "sh", "-c", "true"],
], ids=["sh-env-indirection", "sh-plain", "python-c", "python-m", "env-S", "env-prefix",
        "timeout-prefix"])
def test_a_shell_form_is_refused_unless_the_test_opts_in(
    tmp_path, request, spawned, argv
):
    result = _probe(argv, cwd=tmp_path, env=_indirect_env())

    assert result.returncode == 126
    assert spawned == []
    refused = request.node.stash[suite_conftest.PROBE_GUARD_REFUSALS]
    if SHELL_NOTE in result.stderr:
        assert refused == [[str(a) for a in argv]]
        refused.clear()  # the refusal is this test's subject, not a failure
    else:
        # The indirection resolves to the wrapper itself: blocked as one.
        assert GUARD_NOTE in result.stderr and argv is INDIRECT_SH


@pytest.mark.shell_probe
@pytest.mark.parametrize("script", [
    'exec "$PY" -I -S "$WRAPPER" -- true',
    'exec "$PY" -I -S ${WRAPPER} -- true',
    'CMD="$PY -I -S $WRAPPER --"; eval "$CMD true"',
    'exec "$PY" -I -S "$(dirname "$0")/landlock_exec.py" -- true',
    'exec "$PY" -I -m utils.landlock_exec -- true',
], ids=["quoted", "braced", "eval", "built-path", "module"])
def test_an_opted_in_shell_probe_that_reaches_the_wrapper_is_still_blocked(
    tmp_path, spawned, script
):
    _assert_blocked(
        _probe(["sh", "-c", script], cwd=SKILL_ROOT, env=_indirect_env()),
        spawned,
    )


@pytest.mark.shell_probe
def test_an_opted_in_prefix_launcher_still_resolves_the_wrapper_in_the_child_cwd(
    tmp_path, monkeypatch, spawned
):
    monkeypatch.chdir(tmp_path)
    argv = ["env", "A=1", sys.executable, "-I", "-S",
            os.path.join("utils", "landlock_exec.py"), "--", "true"]

    _assert_blocked(_probe(argv, cwd=SKILL_ROOT), spawned)


@pytest.mark.shell_probe
def test_an_opted_in_plain_shell_probe_runs(tmp_path, spawned):
    argv = ["sh", "-c", "echo 'multi-llm sandbox canary: the write was denied'; exit 0"]

    result = _probe(argv, cwd=tmp_path)

    assert result.returncode == 0
    assert [call[1:] for call in spawned] == [argv[1:]]
