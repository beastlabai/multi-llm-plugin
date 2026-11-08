"""The plugin-landlock rung under test: a scripted fake ``landlock_exec.py`` and a temp dir of the test's own.

``FakeWrapper`` (or ``install_fake_wrapper``) puts a fake wrapper in place of
the real one, records each call's argv and env as JSONL, and does what the test
scripted for a probe and for a dispatch apart. ``use_own_temp_base`` points
``gettempdir()`` at the test's own directory, and ``_slot_dirs`` lists the slot
directories made there. ``_plan_request`` builds a ``JailRequest``.

Fixtures, not registered suite-wide (``repo`` belongs to
``tests.support.gate_fixture`` there), so import the function name:
``landlock_jail_home`` provides ``jail_home`` (a temp ``$HOME`` with
``$CODEX_HOME`` inside), ``landlock_repo`` provides ``repo`` (a plan in a
folder at the cwd, the rung variables cleared) and ``landlock_wrapper``
provides ``wrapper`` (a ``FakeWrapper``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from utils import landlock_dispatch
from utils.providers.base import (
    HOST_SANDBOX_POSTURE_ENV,
    SANDBOX_CANARY_DENIED,
    SANDBOX_CANARY_NOT_DENIED,
    SKIP_RUNG_ENV,
    JailRequest,
)


BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
F6_LEAD = (
    "To run this now, set `driver.subagent_runner` to a runner that is not "
    "Codex (for example `claude-code:<model>`) in `.multi-llm/providers.yaml`."
)
ABSENT_NOTE = "landlock_exec: unavailable: Landlock ABI none (ENOSYS); the floor is 3"
ABI2_NOTE = "landlock_exec: unavailable: Landlock ABI 2 found; the floor is 3"
REQUIRED_NOTE = "landlock_exec: required-entry: dir:/nowhere/codex-home: missing"


FAKE_WRAPPER = r'''
import json, os, subprocess, sys, time
LOG = {log!r}
MODES = {modes!r}
args = sys.argv[1:]
sep = args.index("--")
allow, cmd = args[:sep], args[sep + 1:]
# Where codex is an npm-style shim (a Windows mock) the launcher is `node
# <cli.js>`, so the subcommand is the second or third word.
probe = "sandbox" in cmd[1:3]
record = {{
    "argv": sys.argv, "orig_argv": getattr(sys, "orig_argv", None),
    "executable": sys.executable, "allow": allow, "cmd": cmd, "probe": probe,
    "pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None,
    "env": {{k: os.environ.get(k) for k in (
        "TMPDIR", "MULTI_LLM_CANARY_SEEDED", "MULTI_LLM_CANARY_WRITE_FILE",
        "MULTI_LLM_CANARY_WRITE_DIR", "MULTI_LLM_CANARY_WORKDIR", "CODEX_HOME",
        "GIT_OPTIONAL_LOCKS", "UV_CACHE_DIR", "PYTEST_ADDOPTS")}},
    "tmpdir_exists": os.path.isdir(os.environ.get("TMPDIR") or "/nonexistent"),
}}
with open(LOG, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
with open(MODES, encoding="utf-8") as handle:
    modes = json.load(handle)
mode = modes["probe" if probe else "dispatch"]
kind, _, arg = mode.partition(":")

def say(text, code, err=True):
    print(text, file=sys.stderr if err else sys.stdout, flush=True)
    sys.exit(code)

def run(argv):
    # exec(2) where there is one. Windows has none (os.execvp starts the child
    # and ends this process with status 0), so the child runs under this
    # process and its status is relayed.
    if os.name == "nt":
        sys.exit(subprocess.run(argv).returncode)
    os.execvp(argv[0], argv)

if kind == "exec":
    run(cmd)
if kind == "deny":
    say({denied!r}, 0, err=False)
if kind == "absent":
    say({absent!r}, 126)
if kind == "abi2":
    say({abi2!r}, 126)
if kind == "required":
    say({required!r}, 125)
if kind == "not_denied":
    say({not_denied!r}, 3)
if kind == "seed":
    os.truncate(os.environ["MULTI_LLM_CANARY_SEEDED"], 0)
    say({denied!r}, 0, err=False)
if kind == "positive":
    say("multi-llm sandbox canary: an allowed write failed", 5)
if kind == "exit0":
    sys.exit(0)
if kind == "startup":
    say("Error: codex could not start: Permission denied (os error 13)", 1)
if kind == "hang":
    time.sleep(30)
    sys.exit(0)
if kind == "fail":
    sys.exit(1)
if kind == "eacces":
    say("Error: failed to write " + arg.replace("$TMPDIR", os.environ.get("TMPDIR", "")) +
        ": Permission denied (os error 13)", 1)
if kind == "eacces_nopath":
    say("Error: Permission denied (os error 13)", 1)
if kind == "eacces_secret":
    say("Error: failed to write " + arg + ": Permission denied (os error 13) "
        "(auth: Authorization: Bearer " + "sk-proj-" + "A1b2C3d4" * 6 + ")", 1)
if kind == "tree":
    os.execvp("sh", ["sh", "-c", "sleep 300 & echo $! > " + arg + "; wait"])
if kind == "home":
    home = os.environ["CODEX_HOME"]
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write(arg.replace("\\n", "\n"))
    run(cmd)
if kind == "link":
    # uv traffic on a shared inode: a name made and removed (ctime only).
    time.sleep(0.02)
    os.link(arg, arg + ".spare")
    os.unlink(arg + ".spare")
    run(cmd)
if kind == "linkwrite":
    # The same inode rewritten at the same size, mtime put back.
    info = os.stat(arg)
    with open(arg, "r+b") as handle:
        data = handle.read()
        handle.seek(0)
        handle.write(data.replace(b"6.0.3", b"6.6.6"))
    os.utime(arg, ns=(info.st_atime_ns, info.st_mtime_ns))
    run(cmd)
if kind == "cmdeny":
    done = subprocess.run(cmd, capture_output=True, text=True)
    item = {{"type": "item.completed", "item": {{
        "type": "command_execution", "command": "sh -c 'echo x > /etc/stray'",
        "aggregated_output": "sh: 1: cannot create " + arg + ": Permission denied\n",
        "exit_code": 2, "status": "failed"}}}}
    print(json.dumps(item), flush=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.exit(done.returncode)
say("fake wrapper: unknown mode " + mode, 99)
'''


class FakeWrapper:
    """A fake ``landlock_exec.py`` under ``tmp_path`` and its JSONL log."""

    def __init__(self, root: Path, monkeypatch, probe: str = "deny", dispatch: str = "exec"):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "landlock_exec.py"
        self.log = root / "wrapper.jsonl"
        self.modes = root / "modes.json"
        self.set(probe=probe, dispatch=dispatch)
        self.path.write_text(FAKE_WRAPPER.format(
            log=str(self.log), modes=str(self.modes),
            denied=f"{SANDBOX_CANARY_DENIED}", absent=ABSENT_NOTE, abi2=ABI2_NOTE,
            required=REQUIRED_NOTE, not_denied=SANDBOX_CANARY_NOT_DENIED,
        ), encoding="utf-8")
        monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(self.path))

    def set(self, *, probe: str = None, dispatch: str = None) -> None:
        current = {"probe": "deny", "dispatch": "exec"}
        if self.modes.exists():
            current = json.loads(self.modes.read_text(encoding="utf-8"))
        if probe is not None:
            current["probe"] = probe
        if dispatch is not None:
            current["dispatch"] = dispatch
        self.modes.write_text(json.dumps(current), encoding="utf-8")

    def records(self, probe=None):
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if probe is None or r["probe"] is probe]


def install_fake_wrapper(root: Path, monkeypatch, **modes) -> FakeWrapper:
    """For other test files: a scripted fake wrapper in place of the real one."""
    return FakeWrapper(root, monkeypatch, **modes)


@pytest.fixture(name="jail_home")
def landlock_jail_home(tmp_path, monkeypatch):
    """A temp ``$HOME`` with an existing ``$CODEX_HOME`` inside it (the rung
    refuses a missing home and never creates one)."""
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    return codex_home


@pytest.fixture(name="repo")
def landlock_repo(tmp_path, monkeypatch, jail_home):
    root = tmp_path / "repo"
    (root / "plans").mkdir(parents=True)
    (root / "plans" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv(SKIP_RUNG_ENV, raising=False)
    monkeypatch.delenv(HOST_SANDBOX_POSTURE_ENV, raising=False)
    return root


@pytest.fixture(name="wrapper")
def landlock_wrapper(tmp_path, monkeypatch):
    return FakeWrapper(tmp_path / "wrapper", monkeypatch)


#: The temp dir every process on the machine shares. :func:`_slot_dirs` never
#: scans it: another pytest worker's, or another run's, live slots are there,
#: and the cleanup assertions read them as this test's leftovers (the user's
#: commit review of 5025463..fb0634f, 2026-10-08).
_SHARED_TEMP = os.path.realpath(tempfile.gettempdir())


def use_own_temp_base(tmp_path: Path, monkeypatch) -> Path:
    """Point ``gettempdir()`` at a directory of this test's own, where every
    slot ``TMPDIR``, probe directory and seed the code under test makes lands,
    with a fresh first-call sweep memo. For other test files too."""
    base = Path(os.path.realpath(tmp_path)) / "own-temp"
    base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    monkeypatch.setattr(landlock_dispatch, "_SWEPT_BASES", set())
    return base


def _slot_dirs():
    """The slot directories in this test's own temp dir (:func:`use_own_temp_base`).

    A test that saw a slot also asserts it was made in that directory, so an
    empty answer here is never a scan of the wrong place."""
    base = os.path.realpath(tempfile.gettempdir())
    assert base != _SHARED_TEMP, (
        "_slot_dirs() would scan the shared temp dir, which holds other "
        "processes' slots; give the test its own with use_own_temp_base()"
    )
    return {name for name in os.listdir(base) if name.startswith(landlock_dispatch.SLOT_PREFIX)}


def _plan_request(repo_root: Path, *, scope="plan-file", kind="apply_suggestions"):
    plan = repo_root / "plans" / "plan.md"
    return JailRequest(
        repo_root=str(repo_root), kind=kind, write_scope=scope,
        target=str(plan) if scope != "repo-wide" else None,
        output_dir=str(repo_root / "plans" / "plan") if scope == "repo-wide" else None,
    )
