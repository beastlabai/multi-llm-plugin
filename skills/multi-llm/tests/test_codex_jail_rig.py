#!/usr/bin/env python3
"""Tier 3 of the ``plugin-landlock`` rung: the installed ``codex`` inside the
real jail, against a localhost stub (plan ``codex-sandbox-unavailable-findings-and-fixes``
§8.1 "Tier 3 isolation", §8.2 "The allow-list is re-measured on each Codex
upgrade (the preflight rig, promoted)", and the tier-3 legs of the step-2/3
bullets and of §9 item 12).

Why this file exists (2026-09-30). The rung's allow-list (``$CODEX_HOME``, the
attempt's ``TMPDIR``, ``/dev/null``, ``/dev/ptmx``, ``/dev/pts``, plus a write
scope's target) and step 1c's env redirects were measured by one-off rigs — the
preflight's ``run_step1.py`` under the gitignored ``plans/`` output folder, and
step 1c's copy under ``scratch/`` — so the measurement could not be repeated
from the repo and would go stale with the next Codex release. This promotes the
parts the rung depends on into the suite, run through the PRODUCTION code:

* the argv is rendered by ``CodexProvider.build_command_for_transport`` (the
  fan-out's ``generate`` leg) or ``build_agent_command_for_transport`` (the
  executor's ``agent-write`` leg) with ``sandbox_fallback="plugin-landlock"``
  and a :class:`JailContext`, so the ``-c developer_instructions=…`` splice is
  the shipped one;
* the scratch dir is ``landlock_dispatch.slot_tmpdir``, the context
  ``complete_context`` (which reads ``$CODEX_HOME`` from the composed env), the
  wrap ``wrap_argv`` (the wrapper prefix, ``TMPDIR`` and step 1c's
  ``GIT_OPTIONAL_LOCKS``/``PYTHONPYCACHEPREFIX``/``npm_config_cache``
  redirects), and the wrapper ``utils/landlock_exec.py`` itself;
* the stream is read back with ``jail_denied_edits`` / ``is_jail_failure``,
  and the home with ``home_snapshot`` / ``compare_home``.

The only test-side additions are the ``-c model_provider=stub …`` overrides
(inserted before the prompt, so the model is a localhost stub and nothing is
spent) and a throwaway ``CODEX_HOME`` under ``tmp_path`` for EVERY codex
process — without it codex appends trust tables to the real
``~/.codex/config.toml`` (memory ``codex-offline-capture-via-stub``; I12).

Tests:

1. **strace, outside the jail** — every write-type syscall the Codex process
   tree makes during the ``read_only`` turn lands under
   ``allow_entries(context)``; for ``plan_edit`` (its scripted edit only) those
   plus the plan file. A write anywhere else fails naming the path and the
   installed Codex version. A control row runs the ``writes`` script the same
   way and proves the trace names every repo write it lands.
2. **inside the production wrapper** — ``read_only`` completes with the tree
   untouched; ``writes``' repo writes are all denied and the turn completes;
   ``plan_edit`` under a ``plan-file`` grant lands the ``apply_patch`` edit in
   place (same inode) and nothing else; ``sed -i`` on the plan fails and leaves
   it byte-identical.
3. **the stub's recorded request** carries the rendered developer instructions
   as a developer message, and no request advertises a ``node_repl`` tool.
4. **step 1c's tool turn** (``git status``, ``git diff``, an npm command, a
   Python import) exits 0 command by command under the redirects, and the only
   EACCES in the trace or the outputs is ``/dev/tty``.
5. **the home check** — a clean ``read_only`` turn leaves the
   ``home_snapshot`` fingerprint of a seeded throwaway home unchanged, on its first
   turn and on a second one.

Every test records the installed Codex version next to the rung's
``measured_through`` (printed, attached as a junit property, and in every
assertion message): a pass on a newer build is what licenses raising that
version (§8.2 "When").

Opt-in: ``-m codex_jail`` (deselected by default via ``addopts``); every test
also carries ``landlock`` and skips (or, with ``MULTI_LLM_REQUIRE_LANDLOCK=1``,
fails) when the kernel's Landlock ABI is below 3. Network: none of the scripted
commands needs it; a fresh ``CODEX_HOME`` still makes codex clone its plugin
catalog into ``$CODEX_HOME/.tmp`` in the background (no spend).
"""

from __future__ import annotations

import codecs
import functools
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import landlock_dispatch as ld  # noqa: E402
from utils import process_control  # noqa: E402
from utils.llm_client import _resolve_executable  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    SANDBOX_RUNG_PLUGIN_LANDLOCK,
    WRITE_SCOPE_PLAN_FILE,
    JailContext,
    JailRequest,
)
from utils.providers.codex import (  # noqa: E402
    PLUGIN_LANDLOCK_RUNG,
    CodexProvider,
    jail_developer_instructions,
)
from tests.test_landlock_exec import require_landlock  # noqa: E402

pytestmark = [pytest.mark.codex_jail, pytest.mark.landlock]

RIG = Path(__file__).resolve().parent / "fixtures" / "codex" / "landlock" / "rig"
STUB = RIG / "stub_record.py"

#: The model codex is told it talks to. `gpt-5.5` makes codex advertise its
#: `apply_patch` custom tool (fixtures/codex/0.156.1 capture notes), which the
#: plan-edit turns call.
MODEL = "gpt-5.5"

#: Wall-clock cap per turn. The rig's turns took 1-5 s; strace roughly doubles it.
TURN_TIMEOUT_S = 180

REAL_CODEX_HOME = os.path.realpath(os.path.expanduser("~/.codex"))

PLAN_REL = "plans/p.md"
PLAN_BEFORE = "# Plan\n\nstep one\nstep two\n"
#: What `plan_edit.json`'s apply_patch makes of it.
PLAN_AFTER = "# Plan\n\nstep one\nstep 2\n"
AGENTS_TEXT = "line one\nline two\nline three\n"

#: The syscalls traced. `%file` covers every path-taking call (open*, creat,
#: truncate, mkdir*, mknod*, unlink*, rmdir, rename*, link*, symlink*, chdir,
#: execve…); `bind` for pathname Unix sockets (Landlock's MAKE_SOCK);
#: `fchdir` and the clone family to follow each pid's cwd and program.
STRACE_TRACE = "%file,bind,fchdir,clone,clone3,fork,vfork"


# ===========================================================================
# Gates and the version record
# ===========================================================================

def _require_codex() -> str:
    path = shutil.which("codex")
    if not path:
        pytest.skip("codex is not on PATH: tier 3 drives the installed codex binary")
    return path


@pytest.fixture(scope="module")
def codex_version(tmp_path_factory) -> str:
    """``codex --version``, read with a throwaway ``CODEX_HOME`` like every
    other codex process this module starts."""
    _require_codex()
    home = tmp_path_factory.mktemp("codex-version-home")
    proc = subprocess.run(
        ["codex", "--version"], capture_output=True, text=True, timeout=60,
        stdin=subprocess.DEVNULL, env=dict(os.environ, CODEX_HOME=str(home)),
    )
    match = re.search(r"(\d+\.\d+\.\d+\S*)", proc.stdout)
    return match.group(1) if match else (proc.stdout.strip() or "unknown")


@pytest.fixture
def version_note(codex_version, record_property) -> str:
    """The installed codex beside the rung's measured-through version, recorded
    for the run (§8.1 "The test records the codex version it ran")."""
    measured = PLUGIN_LANDLOCK_RUNG.measured_through[0]
    note = f"installed codex {codex_version}; plugin-landlock measured through {measured}"
    record_property("codex_version", codex_version)
    record_property("plugin_landlock_measured_through", measured)
    print(note)
    return note


@pytest.fixture
def jail_ready(version_note) -> str:
    """Every row: codex on PATH and a Landlock ABI >= 3 (skip, or fail under
    ``MULTI_LLM_REQUIRE_LANDLOCK=1``)."""
    _require_codex()
    require_landlock()
    return version_note


# ===========================================================================
# The scratch repo, the stub and one turn
# ===========================================================================

def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=tier3@example.invalid", "-c", "user.name=tier3",
         "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main",
         "-c", "core.hooksPath=/dev/null", *args],
        cwd=tree, check=True, capture_output=True, text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).stdout


def make_tree(tmp_path: Path, *, with_helper: bool = False) -> Path:
    """The rig's scratch repo: ``AGENTS.md``, ``plans/p.md`` (and ``helper.py``
    for the import step), committed."""
    tree = tmp_path / "tree"
    (tree / "plans").mkdir(parents=True)
    (tree / "AGENTS.md").write_text(AGENTS_TEXT)
    (tree / PLAN_REL).write_text(PLAN_BEFORE)
    if with_helper:
        (tree / "helper.py").write_text("VALUE = 42\n")
    _git(tree, "init", "-q")
    _git(tree, "add", ".")
    _git(tree, "commit", "-qm", "init")
    return Path(os.path.realpath(tree))


def make_home(tmp_path: Path, name: str = "codex-home") -> Path:
    """A throwaway ``CODEX_HOME`` (it must exist: ``complete_context`` refuses a
    missing home and the plugin never creates one)."""
    home = tmp_path / name
    home.mkdir()
    return Path(os.path.realpath(home))


def seed_home(home: Path) -> List[str]:
    """Give a throwaway home what a used one carries in ``home_snapshot``'s
    set: a user ``config.toml`` (a comment, so no behaviour changes), a global
    ``AGENTS.md``, a rules file, a user skill and a custom prompt. Returns the
    tree entries (home-relative) the snapshot must list."""
    (home / "config.toml").write_text("# seeded by tests/test_codex_jail_rig.py\n")
    (home / "AGENTS.md").write_text("Seeded global instructions (tier 3).\n")
    trees = {
        "rules/default.rules": "# seeded by tier 3; no rules\n",
        "skills/tier3-seed/SKILL.md": (
            "---\nname: tier3-seed\ndescription: A seeded skill tier 3 checks codex "
            "leaves untouched.\n---\nNothing to do.\n"
        ),
        "prompts/tier3-seed.md": "A seeded custom prompt (tier 3).\n",
    }
    for rel, text in trees.items():
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(trees)


def git_status(tree: Path) -> str:
    return _git(tree, "status", "--porcelain", "--untracked-files=all")


def tree_state(tree: Path) -> Dict[str, Tuple]:
    """Every entry under ``tree``, ``.git/`` included: dirs by inode, files by
    content hash and inode, symlinks by target. Two equal states mean the turn
    wrote nothing in the repository."""
    state: Dict[str, Tuple] = {}
    for root, dirs, files in os.walk(tree):
        dirs.sort()
        for name in dirs:
            path = os.path.join(root, name)
            state[os.path.relpath(path, tree) + "/"] = ("dir", os.lstat(path).st_ino)
        for name in sorted(files):
            path = os.path.join(root, name)
            info = os.lstat(path)
            if os.path.islink(path):
                state[os.path.relpath(path, tree)] = ("link", os.readlink(path))
                continue
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            state[os.path.relpath(path, tree)] = ("file", digest, info.st_ino)
    return state


def state_diff(before: Dict[str, Tuple], after: Dict[str, Tuple]) -> Dict[str, List[str]]:
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
    }


class Stub:
    """The recording Responses stub on a free localhost port."""

    def __init__(self, script: Path, log: Path) -> None:
        self.log = log
        self.proc = subprocess.Popen(
            [sys.executable, str(STUB), "0", str(script), str(log)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            stdin=subprocess.DEVNULL,
        )
        line = self.proc.stdout.readline().strip()
        if not line.startswith("port="):
            self.close()
            raise RuntimeError(f"the stub did not start: {line!r} {self.proc.stderr.read()!r}")
        self.port = int(line.split("=", 1)[1])

    def overrides(self) -> List[str]:
        """The ``-c`` pairs that point codex at the stub (memory
        ``codex-offline-capture-via-stub``)."""
        return [
            "-c", "model_provider=stub",
            "-c", "model_providers.stub.name=stub",
            "-c", f"model_providers.stub.base_url=http://127.0.0.1:{self.port}/v1",
            "-c", "model_providers.stub.wire_api=responses",
            "-c", "model_providers.stub.env_key=STUB_KEY",
        ]

    def bodies(self) -> List[dict]:
        """Every request body, in the order codex sent them."""
        out: List[dict] = []
        n = 1
        while True:
            path = Path(f"{self.log}.body{n}.json")
            if not path.exists():
                return out
            try:
                out.append(json.loads(path.read_bytes()))
            except ValueError:
                out.append({})
            n += 1

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()


@dataclass
class Turn:
    """One codex turn and what the parent read back while its context was alive."""

    rc: object
    stdout: str
    stderr: str
    context: JailContext
    entries: List[ld.AllowEntry]
    argv: List[str]
    env_redirects: Dict[str, str]
    bodies: List[dict]
    denials: ld.JailDenials
    evidence: Optional[ld.JailEvidence]
    trace: Optional["Trace"] = None
    tmp_files: List[str] = field(default_factory=list)

    @property
    def events(self) -> List[dict]:
        out = []
        for raw in self.stdout.splitlines():
            raw = raw.strip()
            if raw.startswith("{"):
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
        return out

    @property
    def items(self) -> List[dict]:
        return [e.get("item") or {} for e in self.events if e.get("type") == "item.completed"]

    @property
    def commands(self) -> List[dict]:
        return [i for i in self.items if i.get("type") == "command_execution"]

    @property
    def file_changes(self) -> List[dict]:
        return [i for i in self.items if i.get("type") == "file_change"]

    @property
    def completed(self) -> bool:
        return any(e.get("type") == "turn.completed" for e in self.events)

    @property
    def failed(self) -> bool:
        return any(e.get("type") in ("turn.failed", "error") for e in self.events)

    def describe(self) -> str:
        cmds = [
            (c.get("command"), c.get("exit_code"), (c.get("aggregated_output") or "")[-300:])
            for c in self.commands
        ]
        changes = [(c.get("status"), c.get("changes")) for c in self.file_changes]
        return (
            f"rc={self.rc} completed={self.completed} failed={self.failed}\n"
            f"commands={json.dumps(cmds, indent=1)}\nfile_changes={changes}\n"
            f"stderr tail={self.stderr[-1500:]!r}"
        )


def run_turn(
    tmp_path: Path,
    tree: Path,
    script: Path,
    *,
    plan_file: bool = False,
    jailed: bool = True,
    strace: bool = False,
    home: Optional[Path] = None,
    label: str = "turn",
) -> Turn:
    """One codex turn, rendered, wrapped and spawned the way the lanes do it
    (landlock_dispatch's module docstring, steps 1-9).

    ``plan_file`` gives the request the ``plan-file`` scope on ``plans/p.md``
    and renders the executor's ``agent-write`` leg; otherwise the fan-out's
    read-only ``generate`` leg. ``jailed=False`` runs the same codex argv and
    the same child env (``TMPDIR`` + redirects) without the wrapper prefix — the
    strace leg's "outside the jail". ``strace`` wraps whatever is spawned.
    """
    work = tmp_path / label
    work.mkdir()
    home = home or make_home(work)
    assert os.path.realpath(home) != REAL_CODEX_HOME
    if plan_file:
        request = JailRequest(
            repo_root=str(tree), kind="apply_suggestions",
            write_scope=WRITE_SCOPE_PLAN_FILE, target=str(tree / PLAN_REL),
        )
        prompt = "Apply the accepted suggestions to plans/p.md."
    else:
        request = JailRequest(repo_root=str(tree), kind=PROFILE_GENERATE)
        prompt = "Review the plan."
    provider = CodexProvider()
    stub = Stub(script, work / "stub.log")
    try:
        with ld.slot_tmpdir(str(tree)) as tmpdir:
            # Step 2: render with the attempt's context (no home yet, as the
            # fan-out does: `JailContext(jail.request, jail.tmpdir)`).
            render_context = JailContext(request, tmpdir)
            if plan_file:
                cmd, stdin_text = provider.build_agent_command_for_transport(
                    prompt, MODEL, PROFILE_AGENT_WRITE, "argv",
                    sandbox_fallback=SANDBOX_RUNG_PLUGIN_LANDLOCK,
                    jail_context=render_context,
                )
            else:
                cmd, stdin_text = provider.build_command_for_transport(
                    prompt, MODEL, "argv",
                    sandbox_fallback=SANDBOX_RUNG_PLUGIN_LANDLOCK,
                    cwd=str(tree), jail_context=render_context,
                )
            assert stdin_text is None and cmd[-1] == prompt, cmd
            cmd = [*cmd[:-1], *stub.overrides(), cmd[-1]]
            # Step 3: executable resolution.
            launcher, _ = _resolve_executable(cmd[0])
            resolved = [*launcher, *cmd[1:]]
            # Step 4: the child env — the throwaway home, the stub's key.
            env = dict(os.environ, CODEX_HOME=str(home), STUB_KEY="dummy")
            # Steps 5-6: the checked context and the one wrap.
            context = ld.complete_context(request, tmpdir, env)
            entries = ld.allow_entries(context)
            wrapped, child_env = ld.wrap_argv(resolved, PLUGIN_LANDLOCK_RUNG, context, env)
            assert os.path.realpath(child_env["CODEX_HOME"]) == os.path.realpath(home)
            assert os.path.realpath(child_env["CODEX_HOME"]).startswith(
                os.path.realpath(tmp_path)
            ), "every codex process runs on a throwaway CODEX_HOME under tmp_path"
            argv = wrapped if jailed else ld.unwrap_argv(wrapped)
            assert argv, wrapped
            strace_log = work / "strace.log"
            if strace:
                argv = [
                    "strace", "-f", "-qq", "-yy", "-s", "400",
                    "-o", str(strace_log), "-e", f"trace={STRACE_TRACE}", *argv,
                ]
            proc = subprocess.Popen(
                argv, cwd=str(tree), env=child_env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                **process_control.new_process_group_kwargs(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=TURN_TIMEOUT_S)
                rc: object = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                stdout, stderr = proc.communicate()
                rc = "timeout"
            # Step 8: read back while the context is alive.
            denials = ld.jail_denied_edits(stdout, context, cwd=str(tree))
            evidence = ld.is_jail_failure(stderr, stdout, context)
            tmp_files = sorted(
                os.path.relpath(os.path.join(root, name), tmpdir)
                for root, _dirs, names in os.walk(tmpdir) for name in names
            )
            trace = parse_strace(strace_log, str(tree)) if strace else None
            redirects = {k: child_env[k] for k in ld.child_env_redirects(context)}
    finally:
        stub.close()
    return Turn(
        rc=rc, stdout=stdout, stderr=stderr, context=context, entries=entries,
        argv=argv, env_redirects=redirects, bodies=stub.bodies(), denials=denials,
        evidence=evidence, trace=trace, tmp_files=tmp_files,
    )


def scripted(tmp_path: Path, name: str, steps: Optional[int] = None) -> Path:
    """A rig turn script, or its first ``steps`` steps written under tmp_path."""
    source = RIG / f"{name}.json"
    if steps is None:
        return source
    out = tmp_path / f"{name}.first{steps}.json"
    out.write_text(json.dumps(json.loads(source.read_text())[:steps]))
    return out


# ===========================================================================
# strace, parsed
# ===========================================================================

_LINE = re.compile(r"^(\d+)\s+(\w+)\((.*)\)\s+=\s+(.*)$")
_UNFINISHED = re.compile(r"^(\d+)\s+(.*?)\s*<unfinished \.\.\.>$")
_RESUMED = re.compile(r"^(\d+)\s+<\.\.\. \w+ resumed>(.*)$")
_DECORATED = re.compile(r"^(-?\d+|AT_FDCWD)<(.*)>$")
_DEVICE_SUFFIX = re.compile(r"<(?:char|block) [^>]*>$")
_SUN_PATH = re.compile(r'sun_path=(@?)"((?:[^"\\]|\\.)*)"')
_WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC")

#: Landlock-mediated path writes: syscall -> (dirfd arg index or None, path arg
#: index), one tuple per path the call writes. chmod/chown/utimes/xattr are not
#: here on purpose: the jail does not mediate them (F1 Limits), so they are no
#: measure of the allow-list.
_PATH_WRITES = {
    "truncate": [(None, 0)],
    "mkdir": [(None, 0)], "mkdirat": [(0, 1)],
    "mknod": [(None, 0)], "mknodat": [(0, 1)],
    "unlink": [(None, 0)], "unlinkat": [(0, 1)],
    "rmdir": [(None, 0)],
    "rename": [(None, 0), (None, 1)],
    "renameat": [(0, 1), (2, 3)], "renameat2": [(0, 1), (2, 3)],
    "link": [(None, 1)], "linkat": [(2, 3)],
    "symlink": [(None, 1)], "symlinkat": [(1, 2)],
}


def _split_args(args: str) -> List[str]:
    """strace's argument list split on top-level commas (not inside a quoted
    string, a ``<…>`` decoration, ``{…}`` or ``[…]``)."""
    parts, depth, cur, in_str, esc = [], 0, [], False, False
    for ch in args:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "<{[":
            depth += 1
        elif ch in ">}]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _unquote(token: str) -> Optional[str]:
    token = token.strip()
    if token.endswith("..."):
        token = token[:-3]
    if len(token) < 2 or not (token.startswith('"') and token.endswith('"')):
        return None
    raw = token[1:-1]
    try:
        return codecs.escape_decode(raw.encode("latin-1", "backslashreplace"))[0].decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - keep the raw text rather than lose the path
        return raw


def _decoration(token: str) -> Optional[str]:
    match = _DECORATED.match(token.strip())
    if not match:
        return None
    return _DEVICE_SUFFIX.sub("", match.group(2))


@dataclass
class TraceEvent:
    pid: str
    prog: str
    call: str
    path: str
    ok: bool
    err: Optional[str]
    write: bool


@dataclass
class Trace:
    events: List[TraceEvent]

    @property
    def writes(self) -> List[TraceEvent]:
        """Write-type calls that succeeded (a write that landed)."""
        return [e for e in self.events if e.write and e.ok]

    @property
    def eacces(self) -> List[TraceEvent]:
        return [e for e in self.events if e.err == "EACCES"]


def _resolve_path(
    args: List[str],
    pid: str,
    cwd: Dict[str, str],
    root_cwd: str,
    dirfd_index: Optional[int],
    path_index: int,
) -> Optional[str]:
    """The absolute path ``args[path_index]`` names: as written when absolute,
    else against the ``-yy`` decoration of ``args[dirfd_index]`` (a dirfd or
    ``AT_FDCWD</cwd>``, which also refreshes the pid's cwd), else the pid's
    tracked cwd."""
    if path_index >= len(args):
        return None
    path = _unquote(args[path_index])
    if path is None:
        return None
    if path.startswith("/"):
        return os.path.normpath(path)
    base = None
    if dirfd_index is not None and dirfd_index < len(args):
        base = _decoration(args[dirfd_index])
        if base is not None and args[dirfd_index].startswith("AT_FDCWD"):
            cwd[pid] = base
    if base is None:
        base = cwd.get(pid, root_cwd)
    return os.path.normpath(os.path.join(base, path))


def parse_strace(log: Path, root_cwd: str) -> Trace:
    """Every write-type path syscall (plus every EACCES) in an ``strace -f -yy``
    log, resolved to absolute paths and attributed to the program each pid (or
    the ancestor it forked from) exec'd. Ported from step 1c's rig parser."""
    cwd: Dict[str, str] = {}
    prog: Dict[str, str] = {}
    pending: Dict[str, str] = {}
    events: List[TraceEvent] = []
    root_pid: Optional[str] = None
    if not log.exists():
        return Trace(events)
    for raw in log.read_text(errors="replace").splitlines():
        raw = raw.strip()
        unfinished = _UNFINISHED.match(raw)
        if unfinished:
            pending[unfinished.group(1)] = unfinished.group(2)
            continue
        resumed = _RESUMED.match(raw)
        if resumed:
            if resumed.group(1) not in pending:
                continue
            raw = f"{resumed.group(1)} {pending.pop(resumed.group(1))}{resumed.group(2)}"
        match = _LINE.match(raw)
        if not match:
            continue
        pid, call, args_text, result = match.groups()
        if root_pid is None:
            root_pid = pid
            cwd.setdefault(pid, root_cwd)
        ok = not result.startswith("-1") and not result.startswith("?")
        err = None
        if result.startswith("-1"):
            bits = result.split()
            err = bits[1] if len(bits) > 1 else "?"
        args = _split_args(args_text)
        label = os.path.basename(prog.get(pid, "?"))
        resolve = functools.partial(_resolve_path, args, pid, cwd, root_cwd)

        if call in ("clone", "clone3", "fork", "vfork"):
            if ok and result.split("<")[0].strip().lstrip("-").isdigit():
                child = result.split("<")[0].strip()
                prog[child] = prog.get(pid, "?")
                cwd.setdefault(child, cwd.get(pid, root_cwd))
            continue
        if call in ("execve", "execveat"):
            if ok:
                target = resolve(None if call == "execve" else 0, 0 if call == "execve" else 1)
                if target:
                    prog[pid] = target
            elif err == "EACCES":
                target = resolve(None, 0) or "?"
                events.append(TraceEvent(pid, label, call, target, ok, err, False))
            continue
        if call == "chdir":
            target = resolve(None, 0)
            if ok and target:
                cwd[pid] = target
            continue
        if call == "fchdir":
            target = _decoration(args[0]) if args else None
            if ok and target:
                cwd[pid] = target
            continue
        if call in ("open", "openat", "openat2", "creat"):
            dirfd, index, flags_index = {
                "open": (None, 0, 1), "creat": (None, 0, None),
                "openat": (0, 1, 2), "openat2": (0, 1, 2),
            }[call]
            flags = args[flags_index] if flags_index is not None and flags_index < len(args) else ""
            write = call == "creat" or any(f in flags for f in _WRITE_FLAGS)
            if not write and err != "EACCES":
                continue
            path = _decoration(result.split(" ", 1)[0]) if ok else None
            path = path or resolve(dirfd, index)
            if path:
                events.append(TraceEvent(pid, label, call, path, ok, err, write))
            continue
        if call == "bind":
            sun = _SUN_PATH.search(args_text)
            if sun and not sun.group(1) and "AF_UNIX" in args_text:
                path = sun.group(2)
                if not path.startswith("/"):
                    path = os.path.normpath(os.path.join(cwd.get(pid, root_cwd), path))
                events.append(TraceEvent(pid, label, "bind", path, ok, err, True))
            continue
        if call in _PATH_WRITES:
            for dirfd, index in _PATH_WRITES[call]:
                path = resolve(dirfd, index)
                if path:
                    events.append(TraceEvent(pid, label, call, path, ok, err, True))
            continue
        if err == "EACCES":
            # Any other path call refused with EACCES (stat, access, readlink…):
            # kept for the tool turn's "only /dev/tty" check.
            path = resolve(0 if call.endswith("at") else None, 1 if call.endswith("at") else 0)
            events.append(TraceEvent(pid, label, call, path or "?", ok, err, False))
    return Trace(events)


def _under(path: str, entries: Sequence[ld.AllowEntry]) -> bool:
    real = os.path.realpath(path)
    for entry in entries:
        root = entry.path.rstrip(os.sep) or os.sep
        if entry.kind == ld.KIND_DIR and (real == root or real.startswith(root + os.sep)):
            return True
        if entry.kind == ld.KIND_FILE and real == entry.path:
            return True
    return False


def writes_outside(result: Turn) -> List[Tuple[str, str, str]]:
    """``(path, syscall, program)`` for every traced write that landed outside
    the turn's allow-list — each one an EACCES inside the jail."""
    assert result.trace is not None
    return sorted({
        (e.path, e.call, e.prog) for e in result.trace.writes
        if not _under(e.path, result.entries)
    })


def render_outside(outside: Sequence[Tuple[str, str, str]]) -> str:
    return "\n".join(f"  {path}  [{call} by {prog}]" for path, call, prog in outside)


# ===========================================================================
# 1. strace outside the jail: the allow-list still covers codex
# ===========================================================================

@pytest.mark.parametrize("turn", ["read_only", "plan_edit"])
def test_every_codex_write_lands_under_the_allow_list(tmp_path, jail_ready, turn):
    """§8.2 "Outside the jail, under strace -f": the allow-list is re-measured
    against the installed codex. ``plan_edit`` runs its scripted edit only (the
    apply_patch and the ``cat``); the rig script's two stray writes are the
    jail's legs (test 2) and would land for real outside it."""
    if not shutil.which("strace"):
        pytest.skip("strace is not installed: the allow-list re-measurement traces codex's syscalls")
    tree = make_tree(tmp_path)
    plan = turn == "plan_edit"
    script = scripted(tmp_path, "plan_edit", steps=2) if plan else scripted(tmp_path, "read_only")
    result = run_turn(tmp_path, tree, script, plan_file=plan, jailed=False, strace=True, label=turn)
    assert result.rc == 0 and result.completed, f"{jail_ready}\n{result.describe()}"
    assert result.trace is not None and result.trace.events, (
        f"{jail_ready}: strace recorded nothing; the parser or the trace set is broken"
    )
    outside = writes_outside(result)
    assert not outside, (
        f"{jail_ready}: codex's {turn} turn wrote outside the plugin-landlock allow-list "
        f"({ld.render_allow_list(result.entries)}) — every path below would be an EACCES "
        f"inside the jail; re-measure before raising measured_through:\n"
        + render_outside(outside)
    )
    # The measurement is not vacuous: codex wrote its own state under the home
    # and, for the plan-file turn, the plan itself.
    homes = [e for e in result.trace.writes if _under(e.path, result.entries[:1])]
    assert homes, f"{jail_ready}: no write under $CODEX_HOME was traced"
    if plan:
        assert (tree / PLAN_REL).read_text() == PLAN_AFTER, result.describe()
        plan_writes = [e for e in result.trace.writes if e.path == str(tree / PLAN_REL)]
        assert plan_writes, f"{jail_ready}: the apply_patch write of the plan was not traced"


def test_the_trace_catches_a_write_outside_the_allow_list(tmp_path, jail_ready):
    """Control for the row above: the ``writes`` script's first seven steps,
    outside the jail under strace, land a shell redirect, a heredoc, a Python
    ``open`` and codex's own in-process ``apply_patch`` (add + update) in the
    scratch repo, and :func:`writes_outside` names every one of them — so an
    empty result above is a measurement, not a parser that sees nothing. The
    script's last steps (a tty ``bash`` session, which writes the real
    ``~/.bash_history`` on exit outside the jail) are left out."""
    if not shutil.which("strace"):
        pytest.skip("strace is not installed: the allow-list re-measurement traces codex's syscalls")
    tree = make_tree(tmp_path)
    result = run_turn(tmp_path, tree, scripted(tmp_path, "writes", steps=7), jailed=False, strace=True)
    assert result.rc == 0 and result.completed, f"{jail_ready}\n{result.describe()}"
    flagged = {path for path, _call, _prog in writes_outside(result)}
    expected = {str(tree / name) for name in ("f.txt", "h.txt", "p.txt", "a.txt", "AGENTS.md")}
    assert expected <= flagged, (
        f"{jail_ready}: the trace missed a repo write; flagged:\n"
        + render_outside(writes_outside(result))
    )
    progs = {prog for path, _call, prog in writes_outside(result) if path == str(tree / "a.txt")}
    assert "codex" in progs, progs


# ===========================================================================
# 2. inside the production wrapper
# ===========================================================================

def test_read_only_turn_completes_in_the_jail_and_leaves_the_tree_untouched(tmp_path, jail_ready):
    """§8.2 "Inside the production wrapper: the same turns complete"."""
    tree = make_tree(tmp_path)
    before = tree_state(tree)
    result = run_turn(tmp_path, tree, scripted(tmp_path, "read_only"))
    after = tree_state(tree)
    assert result.rc == 0 and result.completed and not result.failed, f"{jail_ready}\n{result.describe()}"
    assert len(result.commands) == 6, result.describe()
    assert result.evidence is None, f"{jail_ready}: codex-internal EACCES {result.evidence}"
    assert result.denials.total == 0, (jail_ready, result.denials.to_dict())
    assert before == after, (jail_ready, state_diff(before, after))
    assert git_status(tree) == ""


#: The `writes` script's repo writes (shell redirect, heredoc, python open,
#: apply_patch add + update, the shell `apply_patch` helper, a background
#: `sleep; echo >` and a tty session). None may land.
WRITES_TARGETS = ("f.txt", "h.txt", "p.txt", "a.txt", "b.txt", "s.txt", "t.txt")


def test_writes_turn_is_denied_in_the_repo_and_still_completes(tmp_path, jail_ready):
    """§8.2 step 2 "Write attempts are denied", tier 3: real codex makes the
    attempts through stub-scripted shell and apply_patch calls; every one is
    denied, the tree is untouched, and the turn still completes (the slot's
    result is kept: no codex-internal EACCES)."""
    tree = make_tree(tmp_path)
    before = tree_state(tree)
    result = run_turn(tmp_path, tree, scripted(tmp_path, "writes"))
    after = tree_state(tree)
    assert result.rc == 0 and result.completed and not result.failed, f"{jail_ready}\n{result.describe()}"
    assert before == after, (jail_ready, state_diff(before, after))
    assert git_status(tree) == ""
    for name in WRITES_TARGETS:
        assert not (tree / name).exists(), f"{jail_ready}: {name} landed in the repo"
    assert (tree / "AGENTS.md").read_text() == AGENTS_TEXT
    assert result.evidence is None, f"{jail_ready}: codex-internal EACCES {result.evidence}"
    # The stream says so too: the apply_patch item failed, and the shell
    # writes were denied (read back by the production reader).
    failed_changes = [c for c in result.file_changes if c.get("status") == "failed"]
    assert failed_changes, result.describe()
    assert result.denials.failed_file_changes, result.denials.to_dict()
    denied = [
        c for c in result.commands
        if "permission denied" in (c.get("aggregated_output") or "").lower()
    ]
    assert len(denied) >= 3, f"{jail_ready}\n{result.describe()}"
    assert result.denials.out_of_scope_count >= 3, result.denials.to_dict()


def test_plan_edit_lands_in_place_and_nothing_else_changes(tmp_path, jail_ready):
    """§8.2 step 3 "Plan file: supported edits", tier 3: under a ``plan-file``
    grant the apply_patch edit lands on the same inode, the script's stray
    writes (``plans/other.md``, an append to ``AGENTS.md``) are denied, and
    ``git status`` shows only the plan. If a Codex update switched apply_patch
    to temp-and-rename, this fails here rather than in the field."""
    tree = make_tree(tmp_path)
    plan = tree / PLAN_REL
    inode = (os.stat(plan).st_dev, os.stat(plan).st_ino)
    before = tree_state(tree)
    result = run_turn(tmp_path, tree, scripted(tmp_path, "plan_edit"), plan_file=True)
    after = tree_state(tree)
    assert result.rc == 0 and result.completed and not result.failed, f"{jail_ready}\n{result.describe()}"
    assert plan.read_text() == PLAN_AFTER, f"{jail_ready}\n{result.describe()}"
    assert (os.stat(plan).st_dev, os.stat(plan).st_ino) == inode, (
        f"{jail_ready}: apply_patch replaced the plan instead of rewriting it in place"
    )
    diff = state_diff(before, after)
    assert diff == {"added": [], "removed": [], "changed": [PLAN_REL]}, (jail_ready, diff)
    assert git_status(tree) == f" M {PLAN_REL}\n"
    assert [c.get("status") for c in result.file_changes] == ["completed"], result.describe()
    assert result.evidence is None, f"{jail_ready}: codex-internal EACCES {result.evidence}"
    assert not result.denials.failed_file_changes, result.denials.to_dict()
    denied = set(result.denials.denied_paths()) | set(result.denials.out_of_scope_paths)
    assert str(tree / "plans" / "other.md") in denied, result.denials.to_dict()
    assert str(tree / "AGENTS.md") in denied, result.denials.to_dict()


def test_sed_i_on_the_plan_fails_and_leaves_it_byte_identical(tmp_path, jail_ready):
    """§8.2 step 3, tier 3's second turn: ``sed -i`` replaces the file through a
    temp file beside it, which a file-only grant does not cover. The command
    fails, the plan keeps its bytes and inode, ``plans/`` gains nothing, and
    the production reader counts it as a replacement denial."""
    tree = make_tree(tmp_path)
    plan = tree / PLAN_REL
    inode = (os.stat(plan).st_dev, os.stat(plan).st_ino)
    before = tree_state(tree)
    result = run_turn(tmp_path, tree, scripted(tmp_path, "plan_sed_i"), plan_file=True)
    after = tree_state(tree)
    assert result.rc == 0 and result.completed, f"{jail_ready}\n{result.describe()}"
    sed = [c for c in result.commands if "sed -i" in (c.get("command") or "")]
    assert len(sed) == 1 and sed[0].get("exit_code") not in (0, None), f"{jail_ready}\n{result.describe()}"
    assert "permission denied" in (sed[0].get("aggregated_output") or "").lower(), result.describe()
    assert plan.read_text() == PLAN_BEFORE
    assert (os.stat(plan).st_dev, os.stat(plan).st_ino) == inode
    assert before == after, (jail_ready, state_diff(before, after))
    assert git_status(tree) == ""
    assert result.denials.replacement_paths, result.denials.to_dict()
    assert all(
        os.path.dirname(p) == str(tree / "plans") for p in result.denials.replacement_paths
    ), result.denials.to_dict()


# ===========================================================================
# 3. the developer message and the tool list, as the stub recorded them
# ===========================================================================

def _developer_texts(body: dict) -> List[str]:
    texts: List[str] = []
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("role") != "developer":
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def _tool_names(tools) -> Iterator[str]:
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        for key in ("name", "type"):
            if isinstance(tool.get(key), str):
                yield tool[key]
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            yield function["name"]
        if isinstance(tool.get("tools"), list):
            yield from _tool_names(tool["tools"])


@pytest.mark.parametrize("scope", ["read-only", "plan-file"])
def test_developer_instructions_reach_the_model_and_no_node_repl_tool(tmp_path, jail_ready, scope):
    """§8.2 "The developer_instructions override", tier 3 (and step 3's
    supported-edits sentences for ``plan-file``): the stub's first recorded
    request carries ``jail_developer_instructions(context)`` verbatim as a
    developer message. And, step 1c: no request advertises ``node_repl`` for
    the rung's model, so the jail never has to answer for it."""
    tree = make_tree(tmp_path)
    result = run_turn(
        tmp_path, tree, scripted(tmp_path, "noop"), plan_file=scope == "plan-file",
    )
    assert result.rc == 0 and result.completed, f"{jail_ready}\n{result.describe()}"
    assert result.bodies, f"{jail_ready}: the stub recorded no request"
    expected = jail_developer_instructions(result.context)
    assert result.context.tmpdir in expected
    if scope == "plan-file":
        assert str(tree / PLAN_REL) in expected and "sed -i" in expected
    first = _developer_texts(result.bodies[0])
    assert expected in first, (
        f"{jail_ready}: the rendered developer_instructions are not a developer "
        f"message of the first request; developer texts were:\n"
        + "\n---\n".join(t[:400] for t in first)
    )
    names = [name for body in result.bodies for name in _tool_names(body.get("tools"))]
    assert "exec_command" in names, (jail_ready, sorted(set(names)))
    assert not [n for n in names if "node_repl" in n], (
        f"{jail_ready}: codex advertised node_repl on the rung: {sorted(set(names))}"
    )


# ===========================================================================
# 4. step 1c's tool turn under the redirects
# ===========================================================================

def test_step_1c_tool_turn_runs_clean_under_the_redirects(tmp_path, jail_ready):
    """§9 item 12, tier 3: step 1c's tool turn — apply_patch on the plan, ``git
    status``, ``git diff``, an npm command (``npm ls``; the rig's ``npx --yes
    cowsay`` needed the registry, and ``npm ls --no-update-notifier`` writes the
    same kind of file, its debug log, into the cache dir with no network), a
    Python import from the repo, and a listing — completes in the jail under
    the ``plan-file`` grant with every command exiting 0. The redirects
    (``GIT_OPTIONAL_LOCKS``, ``PYTHONPYCACHEPREFIX``, ``npm_config_cache``)
    keep every write in the scratch dir, so the only EACCES left in the trace
    and the outputs is ``/dev/tty`` (bash probing its controlling terminal,
    denied silently)."""
    tree = make_tree(tmp_path, with_helper=True)
    trace = bool(shutil.which("strace"))
    result = run_turn(
        tmp_path, tree, scripted(tmp_path, "tools_plan_edit"), plan_file=True, strace=trace,
    )
    assert result.rc == 0 and result.completed and not result.failed, f"{jail_ready}\n{result.describe()}"
    assert set(result.env_redirects) == {"GIT_OPTIONAL_LOCKS", "PYTHONPYCACHEPREFIX", "npm_config_cache"}
    assert [c.get("status") for c in result.file_changes] == ["completed"], result.describe()
    assert len(result.commands) == 5, result.describe()
    bad = [(c.get("command"), c.get("exit_code")) for c in result.commands if c.get("exit_code") != 0]
    assert not bad, f"{jail_ready}: commands failed under the redirects: {bad}\n{result.describe()}"
    for command in result.commands:
        output = (command.get("aggregated_output") or "").lower()
        assert "permission denied" not in output and "eacces" not in output, (
            f"{jail_ready}: {command.get('command')} hit a denial:\n{output[-800:]}"
        )
    python = [c for c in result.commands if "import helper" in (c.get("command") or "")]
    assert python and python[0].get("aggregated_output", "").strip().endswith("42")
    assert result.denials.total == 0, result.denials.to_dict()
    assert result.evidence is None, f"{jail_ready}: codex-internal EACCES {result.evidence}"
    # The redirects were used: the bytecode and npm's log landed in the
    # attempt's scratch dir, and nothing new appeared in the repo.
    assert any(p.startswith("pycache/") for p in result.tmp_files), result.tmp_files[:40]
    assert any(p.startswith("npm-cache/") for p in result.tmp_files), result.tmp_files[:40]
    assert git_status(tree) == f" M {PLAN_REL}\n"
    assert not (tree / "__pycache__").exists()
    if result.trace is not None:
        eacces = sorted({(e.path, e.call, e.prog) for e in result.trace.eacces})
        assert {path for path, _call, _prog in eacces} <= {"/dev/tty"}, (
            f"{jail_ready}: EACCES beyond /dev/tty in the jailed tool turn:\n"
            + "\n".join(f"  {p}  [{c} by {g}]" for p, c, g in eacces)
        )


# ===========================================================================
# 5. the home check
# ===========================================================================

def test_clean_read_only_turn_leaves_the_home_fingerprint_unchanged(tmp_path, jail_ready):
    """§8.2 "The home check", tier 3: a clean ``read_only`` turn in the jail
    leaves ``home_snapshot``'s security-relevant set unchanged, apart from the
    run's own trust table (excluded because the repo is passed as a run cwd),
    so a Codex upgrade that starts rewriting one of those files is caught here
    before ``[CODEX_HOME_CHECK]`` fires on every slot.

    The home is seeded with what a used home carries in that set
    (:func:`seed_home`), so a rewrite of an existing file is caught as well as
    a new one. Two turns are measured: the first on this home (codex's
    first-use writes — the trust table, ``skills/.system/``, the sqlite
    state — must stay outside the set) and a second, steady-state one."""
    tree = make_tree(tmp_path)
    home = make_home(tmp_path)
    seeded = seed_home(home)
    snapshots = [ld.home_snapshot(str(home), [str(tree)])]
    first = snapshots[0]
    assert first is not None
    # The seeded set is really in the fingerprint.
    assert first.files["AGENTS.md"] not in ("absent", "unreadable")
    assert first.config_sha not in ("absent", "unreadable")
    assert set(first.trees) >= set(seeded), (sorted(first.trees), seeded)
    for label in ("first-turn", "second-turn"):
        result = run_turn(tmp_path, tree, scripted(tmp_path, "read_only"), home=home, label=label)
        assert result.rc == 0 and result.completed, f"{jail_ready}\n{result.describe()}"
        snapshots.append(ld.home_snapshot(str(home), [str(tree)]))
        change = ld.compare_home(snapshots[-2], snapshots[-1])
        assert change is not None
        assert not change.any, (
            f"{jail_ready}: the {label} clean read_only turn changed the home's "
            f"security-relevant set: {change.describe()} ({change.to_dict()})"
        )
        assert not change.foreign_trust, (label, change.to_dict())
