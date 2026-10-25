#!/usr/bin/env python3
"""Row 7 of the opencode release gate: the gated tarball, loaded by the pinned OpenCode.

Run it as::

    uv run --project skills/multi-llm -- python packaging/opencode/probe.py row7 \\
        --tarball packaging/opencode/opencode-multi-llm-<version>.tgz \\
        --opencode "$(uv run --project skills/multi-llm -- python \\
                      packaging/opencode/bump.py install --prefix <dir> | tail -1)" \\
        --registry "$(make -s opencode-print-registry)" \\
        --out <evidence.json> [--keep]

**What it proves**, and only by observation: the tarball the gate packed,
extracted into a temporary directory and named in an isolated OpenCode's
``plugin[]`` as that **directory**, (1) registers ``command["multi-llm"]`` with
the payload's own command body, resolved against the payload and carrying no
``{SKILL_DIR}`` token; (2) registers the ``multi_llm`` tool, which OpenCode
advertises in the ``tools[]`` of its model request; and (3) runs ``init`` and
then ``status`` through that tool, each handing its documented result back to
the model. In a project with an initial commit and in one with none.

**What it does not take as proof.** A config entry (``plugin[]`` lists a path
whether or not anything loaded from it); an exit code of zero (a plugin whose
tool hook registers nothing exits 0 too, the silent death the plugin source is
written against); importing the factory in Node (that is not OpenCode loading
it). So the tool is proved from the model's side of the wire: the localhost stub
model (``tests/container/stub_server.py``, its chat-completions
``scripted_tool_call`` leg) logs the ``tools[]`` OpenCode advertised, answers
with one ``multi_llm`` call, and logs the ``role: "tool"`` message that carries
the result back. That text must hash to what OpenCode's own ``tool_use`` event
says the tool returned, and must carry the dispatcher's ``MULTI_LLM_RESULT``.

**Never a ``.tgz`` in ``plugin[]``** (:func:`opencode_config` refuses one). The
loader treats a tarball path as its containing directory and never opens the
archive, so a tarball lying beside a staged build would load the build and pass.

**Isolation** (:func:`isolated_env`). Nothing in this process is reassigned: the
children get an environment built from nothing and rooted at a fresh
``probe_root`` (``HOME``, every ``XDG_*``, ``TMPDIR``, npm's cache and its user
and global config files, both empty, and uv's cache), with every inherited
``OPENCODE_*``, ``XDG_*``, ``npm_config_*``, ``UV_*``, ``MULTI_LLM_*`` and the
other families in :data:`CLEARED_PREFIXES` left behind and named in the
evidence. ``PATH`` is kept, minus the virtualenv this script runs in, with the
pinned binary's directory first.

**The registry** (:func:`discover_registry_settings`). OpenCode installs
``@opencode-ai/plugin`` into every config directory it loads, in the
background, through the npm config loader and Arborist bundled in its binary, so
which registry it resolves from is a setting this gate has to own rather than
inherit. Which setting the binary honours is measured, not assumed: one launch
per candidate (an ``npm_config_registry`` variable, a ``.npmrc`` in the isolated
home, a ``bunfig.toml``), each naming a localhost recording registry that only
that candidate points at. ``npm_config_registry``, the setting the isolated
environment carries, must be honoured; the others are recorded. After the runs,
the config directory's own ``package-lock.json`` names where every package it
installed came from (:func:`installed_from`), and each must be the single
registry.

**Cold first use.** uv's cache is empty and private to ``probe_root``, so the
first tool call's ``uv run --frozen --no-dev`` installs the payload's runtime
dependencies from the network, as a user's first call does. The offline,
pre-warmed path is row 8's evidence, not this row's.

**Every subprocess is bounded** (:func:`run_bounded`, a poll loop that kills the
child's whole session at its budget), and never by a ``timeout`` wrapper:
``timeout`` moves ``opencode run`` into a background process group of the
terminal, where it stops silently before it creates a session (measured, and
the reason this repository's captures never use one). Each child gets a session
of its own (``setsid``): no controlling terminal to be stopped by, and one kill
reaches the ``uv`` and Python processes the plugin starts in process groups of
their own.

**The no-commit project is graded on the committed project's terms.** Until the
shared runtime handles a repository without a ``HEAD``, its calls fail and the
probe fails with them. The plan's decision is to fix the runtime; adding a
commit to the fixture is the one thing never done here.

**Evidence.** ``--out`` always gets the row-7 JSON
(``plans/opencode-plugin-publish.impl-contract.md``), with ``"pass": false`` and
``failures`` on a refusal. Exit 0 only when every assertion passed. Diagnostics
(each child's stdout and stderr, the stub's scenarios and request logs, the
``debug config`` output) stay under ``probe_root/diagnostics``, whose path is
printed on failure; ``probe_root`` is removed only after a pass without
``--keep``.

**The building blocks are a library**, importable without side effects, for the
registry-install probe to reuse: :func:`isolated_env` and :func:`materialize`,
:func:`run_bounded`, :func:`read_debug_config` and :func:`check_command`,
:class:`StubModel`, :func:`scripted_call_scenario`, :func:`opencode_config`,
:func:`drive_scripted_call` (with :func:`assess_call`),
:func:`discover_registry_settings` (with :class:`RecordingRegistry`) and
:func:`installed_from`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socketserver
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_SOURCE = REPO_ROOT / "skills" / "multi-llm"
#: The scripted stub model. Read from the repository, not the payload: tests do
#: not ship, and this is the same server the container lane runs.
STUB_SERVER = SKILL_SOURCE / "tests" / "container" / "stub_server.py"

ROW = "7"
#: The slash command the plugin's ``config`` hook injects (``COMMAND_NAME`` in
#: ``src/index.mjs``). The tool's name is not spelled here: it is the payload's
#: own ``dist/cli-surface.json`` ``tool_name``.
COMMAND_NAME = "multi-llm"
#: Where the payload keeps the skill, relative to the package directory.
PAYLOAD_SKILL = ("skills", "multi-llm")
#: npm's prefix for every member of a packed tarball.
TARBALL_PREFIX = "package/"

PROVIDER = "stub"
MODEL = "m"
PROJECT_KINDS = ("committed", "no-commit")
#: The plan ``status`` reads. It exists in both projects; nothing ever reviews it.
PLAN = "plans/probe.md"
#: What ``init`` scaffolds, relative to the git root.
PROVIDERS_FILE = ".multi-llm/providers.yaml"
#: The two calls, spelled as the tool's own argument descriptions require: all
#: four properties, ``plan`` empty for a mode without a plan positional,
#: ``question`` empty for every mode but ``ask``, ``extra_flags`` ``[]``. The
#: plugin's ``validateArgs`` refuses a missing one before anything runs.
INIT_ARGS = {"mode": "init", "plan": "", "question": "", "extra_flags": []}
STATUS_ARGS = {"mode": "status", "plan": PLAN, "question": "", "extra_flags": []}
#: The run view's line for a plan with no run, which is what ``status``'s exit 2
#: means on a fresh plan ("2 = non-terminal (still running, or no run
#: recorded)", ``utils/run_status.py``'s exit-code epilog).
NO_RUN_LINE = "no run recorded"

#: Budgets in seconds, each a ceiling on one child. ``opencode run``'s is the
#: cold one: its first tool call syncs the payload's runtime from the network.
GIT_BUDGET_S = 60
DEBUG_CONFIG_BUDGET_S = 120
RUN_BUDGET_S = 600
DISCOVERY_BUDGET_S = 90
STUB_START_BUDGET_S = 20
KILL_GRACE_S = 5

#: Inherited variables that can select configuration, data or caches for one of
#: the children (OpenCode, npm, Bun, uv, git, Python, Node, this skill). None
#: reaches a child; the names found are the evidence's ``env_cleared``. Compared
#: upper-cased, so npm's lower-case ``npm_config_*`` spelling is covered.
CLEARED_PREFIXES = (
    "OPENCODE", "XDG_", "NPM_CONFIG_", "BUN_", "UV_", "VIRTUAL_ENV", "CONDA_",
    "PIP_", "PYTHON", "NODE_", "GIT_", "MULTI_LLM_", "MLLM_",
)
#: Inherited variables passed through: locale, CA bundles and proxies, which say
#: how to reach the network rather than what to load from it, and what Windows
#: needs to start a process at all. Nothing else is inherited but ``PATH``.
PASSTHROUGH = (
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR",
)
_PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

#: The candidate registry settings :func:`discover_registry_settings` measures,
#: in the order it measures them.
REGISTRY_CANDIDATES = ("npm_config_registry", "home_npmrc", "bunfig")
#: The one the isolated environment carries, and so the one that must be honoured.
REGISTRY_SETTING = "npm_config_registry"

#: OpenCode's session-title prompts (``scripted_opencode_read_deny.yaml``'s
#: rule, verbatim). Answered as text; the scripted call never touches them.
TITLE_PATTERN = (
    "Generate a concise, sentence-case title|You are a title generator\\. You "
    "output ONLY a thread title|Generate a short title \\(four words or less\\)|"
    "You are tasked with generating the session title"
)


class ProbeError(RuntimeError):
    """A refusal: the probe cannot go on, and the evidence says why."""


# ---------------------------------------------------------------------------
# Sibling modules, loaded by path when first needed
# ---------------------------------------------------------------------------

_SIBLINGS: Dict[str, Any] = {}


def _sibling(name: str) -> Any:
    """``packaging/opencode/<name>.py``, imported once, by path.

    By path because this script is also imported by the suite under its own
    module name, with no ``packaging/opencode`` on ``sys.path``; on first use
    because importing ``build`` imports the skill tree, which a caller that only
    wants :func:`isolated_env` should not pay for.
    """
    if name not in _SIBLINGS:
        module_name = f"opencode_{name}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, PACKAGE_DIR / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            # Registered before it runs, as an import would: a dataclass under
            # `from __future__ import annotations` looks its module up by name.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        _SIBLINGS[name] = module
    return _SIBLINGS[name]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(text: str) -> str:
    """The stub's ``_sha``: sha256 of the UTF-8 text, hex."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

#: Every directory and file an isolated environment owns, relative to its root.
LAYOUT = {
    "home": "home", "config": "config", "data": "data", "cache": "cache",
    "state": "state", "runtime": "runtime", "tmp": "tmp",
    "npm_cache": "npm-cache", "npm_userconfig": "npmrc-user",
    "npm_globalconfig": "npmrc-global", "uv_cache": "uv-cache",
}


@dataclass(frozen=True)
class IsolatedEnv:
    """An environment for OpenCode and everything it starts, owned by ``root``.

    ``env`` is the complete environment, never an overlay on this process's.
    ``cleared`` names the inherited variables left behind because they select
    configuration (see :data:`CLEARED_PREFIXES`); values are never recorded.
    ``registry`` is what ``npm_config_registry`` carries, or ``None`` for an
    environment that names no registry (registry discovery's baseline).
    """

    root: Path
    env: Dict[str, str]
    cleared: Tuple[str, ...]
    registry: Optional[str]

    def path(self, name: str) -> Path:
        return self.root / LAYOUT[name]

    @property
    def home(self) -> Path:
        return self.path("home")

    @property
    def config_home(self) -> Path:
        return self.path("config")

    @property
    def opencode_config_dir(self) -> Path:
        """OpenCode's global config directory under this home (``$XDG_CONFIG_HOME/opencode``)."""
        return self.path("config") / "opencode"

    @property
    def opencode_cache_dir(self) -> Path:
        return self.path("cache") / "opencode"

    @property
    def uv_cache(self) -> Path:
        return self.path("uv_cache")


def _selects_config(name: str) -> bool:
    upper = name.upper()
    return upper == "HOME" or any(upper.startswith(prefix) for prefix in CLEARED_PREFIXES)


def _venv_bin(inherited: Mapping[str, str]) -> Optional[str]:
    venv = inherited.get("VIRTUAL_ENV")
    if not venv:
        return None
    return os.path.normcase(os.path.abspath(os.path.join(venv, "Scripts" if os.name == "nt" else "bin")))


def isolated_env(root: Path, *, binary: str, registry: Optional[str],
                 inherited: Optional[Mapping[str, str]] = None) -> IsolatedEnv:
    """The environment for ``binary`` rooted at ``root``. Pure: creates nothing.

    Built from nothing rather than filtered from ``inherited`` (this process's
    environment by default): a variable nobody thought to strip cannot select
    the maintainer's installation if no variable is copied unless named.
    ``PATH`` is the exception, because ``uv``, ``git`` and ``node`` live on it:
    kept in order, minus the virtualenv this script runs in (``uv run`` puts it
    first, and it is the builder's environment, not a user's), with ``binary``'s
    directory first so the pinned ``opencode`` is the one any child finds.
    """
    root = Path(root).absolute()
    inherited = dict(os.environ if inherited is None else inherited)
    paths = {name: str(root / rel) for name, rel in LAYOUT.items()}

    bin_dir = os.path.abspath(os.path.dirname(os.path.abspath(binary)))
    venv_bin = _venv_bin(inherited)
    kept = [entry for entry in inherited.get("PATH", "").split(os.pathsep)
            if entry and os.path.normcase(os.path.abspath(entry)) != venv_bin
            and os.path.abspath(entry) != bin_dir]

    env = {name: value for name, value in inherited.items() if name in PASSTHROUGH}
    env.update({
        "PATH": os.pathsep.join([bin_dir] + kept),
        "HOME": paths["home"],
        "XDG_CONFIG_HOME": paths["config"],
        "XDG_DATA_HOME": paths["data"],
        "XDG_CACHE_HOME": paths["cache"],
        "XDG_STATE_HOME": paths["state"],
        "XDG_RUNTIME_DIR": paths["runtime"],
        "TMPDIR": paths["tmp"],
        # OpenCode replacing itself mid-gate, or fetching a model catalogue the
        # stub provider does not need, would make the run depend on more than
        # the binary under test.
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        # npm's own files, empty: the user file is what a maintainer's `.npmrc`
        # would otherwise be read as, the global one what a system prefix holds.
        "npm_config_cache": paths["npm_cache"],
        "npm_config_userconfig": paths["npm_userconfig"],
        "npm_config_globalconfig": paths["npm_globalconfig"],
        "npm_config_update_notifier": "false",
        # Empty: the first tool call's `uv run` is the cold first use.
        "UV_CACHE_DIR": paths["uv_cache"],
        "GIT_CONFIG_NOSYSTEM": "1",
    })
    if os.name == "nt":
        env.update(USERPROFILE=paths["home"], TMP=paths["tmp"], TEMP=paths["tmp"],
                   APPDATA=paths["config"], LOCALAPPDATA=paths["data"])
    if registry:
        env["npm_config_registry"] = registry
    if any(env.get(name) for name in _PROXIES):
        # The stub model is on loopback; a proxy must never be asked for it.
        for name in ("NO_PROXY", "no_proxy"):
            present = [part for part in env.get(name, "").split(",") if part]
            env[name] = ",".join(present + [h for h in ("127.0.0.1", "localhost") if h not in present])
    cleared = tuple(sorted(name for name in inherited
                           if name not in PASSTHROUGH and name != "PATH" and _selects_config(name)))
    return IsolatedEnv(root=root, env=env, cleared=cleared, registry=registry)


def materialize(iso: IsolatedEnv) -> None:
    """Create ``iso``'s directories and its two empty npm config files."""
    for name in LAYOUT:
        if name in ("npm_userconfig", "npm_globalconfig"):
            continue
        iso.path(name).mkdir(parents=True, exist_ok=True)
    iso.path("runtime").chmod(0o700)
    for name in ("npm_userconfig", "npm_globalconfig"):
        iso.path(name).write_text("", encoding="utf-8")


def real_home() -> str:
    """The invoking user's home, by every route a child could resolve it."""
    return os.path.abspath(os.path.expanduser("~"))


# ---------------------------------------------------------------------------
# Bounded children
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bounded:
    """One bounded child: how it ended and where its streams went."""

    argv: Tuple[str, ...]
    returncode: Optional[int]
    killed: bool
    reason: Optional[str]
    elapsed_s: float
    stdout: Path
    stderr: Path

    def describe(self) -> str:
        if self.killed:
            return f"killed ({self.reason}) after {self.elapsed_s:.0f}s"
        return f"exit {self.returncode} after {self.elapsed_s:.1f}s"


def _session_members(sid: int) -> List[int]:
    """Every live pid in session ``sid`` (Linux ``/proc``; empty elsewhere)."""
    members = []
    proc = Path("/proc")
    if not proc.is_dir():
        return members
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            fields = text[text.rfind(")") + 2:].split()
            if int(fields[3]) == sid:
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _signal_session(sid: int, sig: int) -> None:
    """``sig`` to the session leader's group and to every other member.

    A process group is not enough: the plugin starts ``uv`` in a group of its
    own so that its own kill can reach the Python under it, and that group is
    still inside this session.
    """
    targets = set(_session_members(sid))
    try:
        os.killpg(sid, sig)
    except OSError:
        pass
    for pid in targets:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _stop(proc: "subprocess.Popen[bytes]") -> None:
    if os.name == "nt":
        # The gate runs on Linux; on Windows this only has to end the child the
        # suite's own cases start, which spawn nothing.
        proc.kill()
        proc.wait()
        return
    _signal_session(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    _signal_session(proc.pid, signal.SIGKILL)
    proc.wait()


def run_bounded(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], budget_s: float,
                stdout: Path, stderr: Path, stop: Optional[Callable[[], bool]] = None,
                poll_s: float = 0.2) -> Bounded:
    """Run ``argv`` to its end, its budget, or ``stop()`` returning true.

    stdin is ``/dev/null`` and both streams go to files, so nothing can block on
    a pipe nobody drains. The child leads a new session; at the budget (or the
    stop, or this process being interrupted) the whole session is sent SIGTERM,
    then SIGKILL after :data:`KILL_GRACE_S`. A child that exited on its own
    still has its session swept, so a straggler it left cannot outlive the probe.
    """
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
        else {"start_new_session": True})
    start = time.monotonic()
    reason: Optional[str] = None
    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        proc = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kwargs)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > budget_s:
                    reason = "budget"
                    break
                if stop is not None and stop():
                    reason = "stopped"
                    break
                time.sleep(poll_s)
        except BaseException:
            reason = "interrupted"
            raise
        finally:
            if proc.poll() is None:
                _stop(proc)
            elif os.name != "nt":
                _signal_session(proc.pid, signal.SIGKILL)
    killed = reason is not None
    return Bounded(argv=tuple(argv), returncode=None if killed else proc.returncode,
                   killed=killed, reason=reason, elapsed_s=round(time.monotonic() - start, 2),
                   stdout=stdout, stderr=stderr)


def _git(args: Sequence[str], cwd: Path, env: Mapping[str, str]) -> None:
    subprocess.run(["git", "-c", "init.defaultBranch=main", *args], cwd=str(cwd), env=dict(env),
                   check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=GIT_BUDGET_S)


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------

def extract_package(tarball: Path, dest: Path) -> Path:
    """Extract ``tarball`` under ``dest``; return the package directory.

    The archive is held to the rules :func:`build.tarball_facts` holds it to
    (every member under ``package/``, no absolute or ``..`` path, no link or
    device), and file modes are kept as npm would keep them.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            parts = name.split("/")
            if name.startswith("/") or ".." in parts or not name.startswith(TARBALL_PREFIX):
                raise ProbeError(f"{tarball}: member {name!r} is not under {TARBALL_PREFIX!r}")
            target = dest / Path(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ProbeError(f"{tarball}: member {name!r} is a link or device")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ProbeError(f"{tarball}: member {name!r} has no content")
            with source:
                target.write_bytes(source.read())
            target.chmod((member.mode & 0o755) | stat.S_IRUSR | stat.S_IWUSR)
    package = dest / TARBALL_PREFIX.rstrip("/")
    if not (package / "package.json").is_file():
        raise ProbeError(f"{tarball}: no {TARBALL_PREFIX}package.json")
    return package


def payload_surface(package_dir: Path) -> Dict[str, Any]:
    """The payload's own generated CLI surface (``dist/cli-surface.json``)."""
    try:
        return json.loads((Path(package_dir) / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeError(f"{package_dir}: cannot read dist/cli-surface.json ({exc})") from exc


def skill_dir_of(package_dir: Path) -> str:
    """The skill directory the plugin resolves ``{SKILL_DIR}`` to.

    Real path, as ``import.meta.dirname`` reports it.
    """
    return os.path.realpath(os.path.join(str(package_dir), *PAYLOAD_SKILL))


def is_inside(path: str, parent: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(str(parent)))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# OpenCode configuration
# ---------------------------------------------------------------------------

_ARCHIVE_SUFFIXES = (".tgz", ".tar.gz", ".tar", ".zip")


def opencode_config(plugin_dir: Path, base_url: str) -> Dict[str, Any]:
    """An ``opencode.json`` loading ``plugin_dir`` and serving the stub model.

    Refuses anything but an existing directory for ``plugin[]``: a ``.tgz``
    entry loads whatever directory the archive sits in and never the archive,
    so it can pass beside a staged build without proving anything about the
    tarball.
    """
    spec = str(plugin_dir)
    if spec.lower().endswith(_ARCHIVE_SUFFIXES):
        raise ProbeError(f"refusing an archive in plugin[]: {spec} (extract it and name the directory)")
    if not Path(spec).is_dir():
        raise ProbeError(f"plugin[] must name an extracted package directory: {spec}")
    return {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [spec],
        "provider": {PROVIDER: {
            "npm": "@ai-sdk/openai-compatible", "name": PROVIDER,
            "options": {"baseURL": base_url, "apiKey": "row7-probe-not-a-key"},
            "models": {MODEL: {"name": MODEL, "tool_call": True}},
        }},
        "model": f"{PROVIDER}/{MODEL}",
        "small_model": f"{PROVIDER}/{MODEL}",
    }


def write_project_config(project: Path, plugin_dir: Path, base_url: str) -> Path:
    """Write ``project/opencode.json`` (:func:`opencode_config`); return its path."""
    path = Path(project) / "opencode.json"
    path.write_text(json.dumps(opencode_config(plugin_dir, base_url), indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class DebugConfig:
    bounded: Optional[Bounded]
    config: Optional[Dict[str, Any]]
    error: Optional[str]


def read_debug_config(binary: str, env: Mapping[str, str], cwd: Path, diag: Path,
                      label: str = "debug-config",
                      budget_s: float = DEBUG_CONFIG_BUDGET_S) -> DebugConfig:
    """``opencode debug config`` in ``cwd``, parsed. Never ``--pure``: that drops plugins."""
    bounded = run_bounded([binary, "debug", "config"], cwd=cwd, env=env, budget_s=budget_s,
                          stdout=diag / f"{label}.json", stderr=diag / f"{label}.stderr.txt")
    if bounded.killed or bounded.returncode != 0:
        return DebugConfig(bounded, None, f"opencode debug config: {bounded.describe()}")
    try:
        config = json.loads(bounded.stdout.read_text(encoding="utf-8"))
    except ValueError as exc:
        return DebugConfig(bounded, None, f"opencode debug config printed no JSON ({exc})")
    if not isinstance(config, dict):
        return DebugConfig(bounded, None, "opencode debug config printed JSON that is not an object")
    return DebugConfig(bounded, config, None)


_FRONTMATTER = re.compile(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$")
_SKILL_PATH = re.compile(r"(?:[A-Za-z]:)?[/\\][^\s\"'`]*?[/\\]skills[/\\]multi-llm(?=[/\\\s\"'`]|$)")


def expected_template(package_dir: Path, token: str) -> str:
    """The template the payload's ``config`` hook must register.

    ``dist/command.md`` with its frontmatter split off and every ``token``
    resolved to the payload's skill directory: ``injectCommand`` in
    ``src/index.mjs``, done again from the payload's own bytes.
    """
    raw = (Path(package_dir) / "dist" / "command.md").read_text(encoding="utf-8")
    body = skill_dir_of(package_dir).join(raw.split(token))
    match = _FRONTMATTER.match(body)
    return match.group(2) if match else body


def check_command(config: Optional[Mapping[str, Any]], package_dir: Path,
                  token: Optional[str] = None) -> Dict[str, Any]:
    """What ``debug config`` says about the command and the plugin that made it.

    ``registered``: ``command["multi-llm"]`` carries a non-empty template.
    ``matches_payload``: that template is the payload's own body resolved
    against the payload (:func:`expected_template`). ``token_left``: a
    ``{SKILL_DIR}`` survived. ``skill_paths``: every absolute
    ``…/skills/multi-llm`` the template names (the current body routes to the
    tool and names none), each of which must be inside the payload.
    ``plugin_specs``: the ``plugin[]`` entries OpenCode resolved, which must be
    the payload directory alone.
    """
    token = token or _sibling("build").SKILL_DIR_TOKEN
    command = ((config or {}).get("command") or {}).get(COMMAND_NAME)
    template = command.get("template") if isinstance(command, dict) else None
    plugins = [str(p) for p in ((config or {}).get("plugin") or [])]
    origins = [o.get("spec") for o in ((config or {}).get("plugin_origins") or []) if isinstance(o, dict)]
    package_real = os.path.realpath(str(package_dir))
    specs_ok = bool(plugins) and all(
        _spec_path(spec) == package_real for spec in plugins) and not any(
        spec.lower().endswith(_ARCHIVE_SUFFIXES) for spec in plugins)
    registered = isinstance(template, str) and bool(template.strip())
    paths = sorted(set(_SKILL_PATH.findall(template))) if registered else []
    return {
        "registered": registered,
        "matches_payload": registered and template == expected_template(package_dir, token),
        "token_left": bool(registered and token in template),
        "skill_paths": paths,
        "skill_paths_inside": all(is_inside(p, package_dir) for p in paths),
        "plugin_specs": plugins,
        "plugin_origins": origins,
        "plugin_is_payload": specs_ok,
    }


def _spec_path(spec: str) -> str:
    if spec.startswith("file://"):
        spec = spec[len("file://"):]
    return os.path.realpath(spec)


# ---------------------------------------------------------------------------
# The stub model
# ---------------------------------------------------------------------------

def scripted_call_scenario(tool_name: str, arguments: Mapping[str, Any], call_id: str) -> Dict[str, Any]:
    """A stub scenario: one ``tool_name`` call per request that advertises it.

    The request carrying our call's ``role: "tool"`` result gets a fixed
    terminal message, so the turn ends there. A session-title prompt gets a
    title, and anything else a plain ``OK``: a main request that does NOT
    advertise the tool is then answered as text, the run ends at exit 0 with no
    call, and the probe reports the tool unregistered rather than timing out.
    """
    return {
        "name": f"opencode_row7_{call_id}",
        "description": "row-7 probe: one scripted multi_llm call through opencode run",
        "scripted_tool_call": {
            "surfaces": ["chat"],
            "name": tool_name,
            "call_id": call_id,
            "arguments": dict(arguments),
            "terminal": f"row-7 probe: the {tool_name} call returned",
        },
        "prompts": [
            {"pattern": TITLE_PATTERN, "content": "Row seven probe"},
            {"pattern": "^", "content": "OK"},
        ],
    }


class StubModel:
    """``tests/container/stub_server.py`` on 127.0.0.1, armed with one scenario.

    A context manager: the server binds port 0 and publishes the port it got
    (no pick-then-bind race), its stdout and stderr and its request log stay in
    ``diag``, and it is stopped on exit. Each entry of :meth:`entries` is one
    request, written before its response was sent.
    """

    def __init__(self, scenario: Mapping[str, Any], diag: Path, label: str,
                 server: Path = STUB_SERVER, python: Optional[str] = None):
        self.scenario = dict(scenario)
        self.diag = Path(diag)
        self.label = label
        self.server = Path(server)
        self.python = python or sys.executable
        self.log = self.diag / f"{label}.stub-requests.jsonl"
        self.port: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._streams: List[Any] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "StubModel":
        self.diag.mkdir(parents=True, exist_ok=True)
        scenario_path = self.diag / f"{self.label}.scenario.json"
        scenario_path.write_text(json.dumps(self.scenario, indent=2) + "\n", encoding="utf-8")
        port_file = self.diag / f"{self.label}.stub-port"
        port_file.unlink(missing_ok=True)
        env = dict(os.environ)
        # The server imports `tests.scenario_matcher` from the skill tree.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SKILL_SOURCE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        out = open(self.diag / f"{self.label}.stub-stdout.txt", "wb")
        err = open(self.diag / f"{self.label}.stub-stderr.txt", "wb")
        self._streams = [out, err]
        self._proc = subprocess.Popen(
            [self.python, str(self.server), "--port", "0", "--port-file", str(port_file),
             "--log", str(self.log), "--scenario", str(scenario_path)],
            env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        deadline = time.monotonic() + STUB_START_BUDGET_S
        while True:
            if port_file.exists():
                text = port_file.read_text(encoding="utf-8").strip()
                if text:
                    self.port = int(text)
                    return self
            if self._proc.poll() is not None:
                self.__exit__(None, None, None)
                raise ProbeError(f"the stub model exited {self._proc.returncode} before binding; "
                                 f"see {self.diag / (self.label + '.stub-stderr.txt')}")
            if time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise ProbeError(f"the stub model did not bind within {STUB_START_BUDGET_S}s")
            time.sleep(0.05)

    def __exit__(self, *_exc: Any) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for stream in self._streams:
            stream.close()
        self._streams = []

    def entries(self) -> List[Dict[str, Any]]:
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


# ---------------------------------------------------------------------------
# One scripted call through `opencode run`
# ---------------------------------------------------------------------------

def parse_events(text: str) -> List[Dict[str, Any]]:
    """``opencode run --format json``'s NDJSON, skipping anything that is not an object."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def assess_call(events: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]], *,
                tool_name: str, call_id: str) -> Dict[str, Any]:
    """What one run proves about the tool, from both sides of the wire.

    * ``advertised``: a model request's ``tools[]`` named ``tool_name`` (the
      stub logs every request's tool names), so OpenCode registered the tool
      and exposed it to the model;
    * ``served_call``: the stub answered such a request with our one call;
    * ``tool_status`` / ``output``: OpenCode's own ``tool_use`` event for that
      call id, and the string the tool's ``execute`` returned;
    * ``returned``: the ``role: "tool"`` text the next model request carried,
      and ``returned_identical`` when it hashes to ``output``;
    * ``outcome``: that text parsed — the plugin's JSON result, whose
      ``result`` is the dispatcher's ``MULTI_LLM_RESULT``.

    ``problems`` names every part that did not hold.
    """
    problems: List[str] = []
    advertised = any(tool_name in (entry.get("tool_names") or []) for entry in entries)
    scripted = [entry.get("scripted") or {} for entry in entries]
    served = any(s.get("phase") == "call" and s.get("declaration") for s in scripted)
    terminal = next((s for s in scripted if s.get("phase") == "terminal"), None)
    unmatched = sum(1 for entry in entries if entry.get("unmatched") or entry.get("unhandled"))

    part = None
    errors = []
    for event in events:
        if event.get("type") == "error":
            errors.append(json.dumps(event.get("error"))[:500])
        candidate = event.get("part") or {}
        if (event.get("type") == "tool_use" and candidate.get("tool") == tool_name
                and candidate.get("callID") in (call_id, None)):
            part = candidate
    state = (part or {}).get("state") or {}
    output = state.get("output") if isinstance(state.get("output"), str) else None

    if not advertised:
        problems.append(f"no model request advertised {tool_name} in tools[]: the tool was not registered")
    elif not served:
        problems.append(f"the stub never served the scripted {tool_name} call")
    if part is None:
        problems.append(f"opencode emitted no tool_use event for {tool_name}")
    elif state.get("status") != "completed":
        problems.append(f"the {tool_name} call ended {state.get('status')!r}: {str(state.get('error'))[:300]}")

    returned = None
    identical = False
    if terminal is None:
        if advertised and served:
            problems.append("the tool's result never reached the model (no request carried it back)")
    else:
        identical = output is not None and terminal.get("output_sha") == _sha256(output)
        if identical:
            returned = output
        elif not terminal.get("output_truncated"):
            returned = terminal.get("output")
            problems.append("the text the model received differs from what the tool returned")
        else:
            problems.append("the model received a result that does not hash to the tool's output")

    outcome = None
    if returned is not None:
        try:
            outcome = json.loads(returned)
        except ValueError:
            problems.append("the returned result is not the plugin's JSON")
        if outcome is not None and not isinstance(outcome, dict):
            outcome = None
            problems.append("the returned result is not a JSON object")
    return {
        "advertised": advertised, "served_call": served, "tool_status": state.get("status"),
        "tool_input": state.get("input"), "output": output, "returned": returned,
        "returned_identical": identical, "outcome": outcome, "error_events": errors,
        "unmatched_requests": unmatched, "problems": problems,
    }


def drive_scripted_call(binary: str, env: Mapping[str, str], project: Path, stub: StubModel, *,
                        tool_name: str, call_id: str, mode: str, diag: Path, label: str,
                        budget_s: float = RUN_BUDGET_S) -> Dict[str, Any]:
    """One ``opencode run`` in ``project`` whose model is ``stub``; :func:`assess_call` of it.

    The stub must already be armed (:func:`scripted_call_scenario`) and the
    project's config must name it. Adds ``bounded`` (how the run ended) and the
    run's ``opencode_exit``.
    """
    prompt = f"Run the {tool_name} tool with mode {mode}."
    bounded = run_bounded(
        [binary, "run", "--format", "json", "--print-logs", "--log-level", "INFO",
         "--model", f"{PROVIDER}/{MODEL}", prompt],
        cwd=project, env=env, budget_s=budget_s,
        stdout=diag / f"{label}.stdout.jsonl", stderr=diag / f"{label}.stderr.txt")
    events = parse_events(bounded.stdout.read_text(encoding="utf-8", errors="replace"))
    verdict = assess_call(events, stub.entries(), tool_name=tool_name, call_id=call_id)
    verdict["bounded"] = bounded
    verdict["opencode_exit"] = bounded.returncode
    if bounded.killed:
        verdict["problems"].append(f"opencode run was {bounded.describe()}")
    elif bounded.returncode != 0:
        verdict["problems"].append(f"opencode run exited {bounded.returncode}")
    return verdict


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class _Loopback(ThreadingHTTPServer):
    """No reverse lookup of 127.0.0.1 at bind (it hung the macOS runner; see
    ``stub_server.LoopbackHTTPServer``)."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Quiet: a client killed mid-request is how every discovery run ends."""


class RecordingRegistry:
    """A localhost "registry" that records every request and answers 404.

    Enough to learn whether a client asked it at all, which is the only
    question registry discovery puts to it. A context manager; :attr:`url` has
    no trailing slash.

    It doubles as the discovery runs' model, and a model request (a POST) is
    held open until the context exits or :data:`DISCOVERY_BUDGET_S` passes:
    answered at once, ``opencode run`` ends within a second and takes its
    background install with it (measured), so a candidate that is not honoured
    could never be told from one that was never tried.
    """

    def __init__(self) -> None:
        self.requests: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._server: Optional[_Loopback] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "RecordingRegistry":
        registry = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                with registry._lock:
                    registry.requests.append((self.command, self.path))
                if self.command == "POST":
                    registry._closing.wait(DISCOVERY_BUDGET_S)
                body = b'{"error":"Not found"}'
                try:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the run was stopped while its request was held

            do_GET = do_POST = _answer

            def log_message(self, *_args: Any) -> None:
                pass

        self._server = _Loopback(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._closing.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def package_requests(self, since: int = 0) -> List[str]:
        """Paths asked after the first ``since`` requests, model calls excluded."""
        with self._lock:
            seen = self.requests[since:]
        return [path for method, path in seen if method == "GET" and not path.startswith("/v1/")]

    def count(self) -> int:
        with self._lock:
            return len(self.requests)


def _apply_registry_candidate(iso: IsolatedEnv, setting: str, url: str) -> Dict[str, str]:
    """``iso``'s environment with only ``setting`` naming ``url``."""
    env = dict(iso.env)
    env.pop("npm_config_registry", None)
    if setting == "npm_config_registry":
        env["npm_config_registry"] = url + "/"
    elif setting == "home_npmrc":
        # npm reads the user file from `npm_config_userconfig` when set, so the
        # home `.npmrc` is only reached with that variable gone.
        env.pop("npm_config_userconfig", None)
        (iso.home / ".npmrc").write_text(f"registry={url}/\n", encoding="utf-8")
    elif setting == "bunfig":
        for where in (iso.home, iso.config_home):
            (where / ".bunfig.toml").write_text(f'[install]\nregistry = "{url}/"\n', encoding="utf-8")
    else:
        raise ValueError(f"unknown registry setting {setting!r}")
    return env


def discover_registry_settings(binary: str, root: Path, *,
                               inherited: Optional[Mapping[str, str]] = None,
                               candidates: Sequence[str] = REGISTRY_CANDIDATES,
                               budget_s: float = DISCOVERY_BUDGET_S) -> List[Dict[str, Any]]:
    """Which registry settings ``binary``'s package installer honours, measured.

    One ``opencode run`` per candidate, each in its own isolated home whose
    environment names no registry except through that one candidate, which
    points at a :class:`RecordingRegistry`. OpenCode starts installing
    ``@opencode-ai/plugin`` into its config directory as soon as the config
    loads, so the run is stopped the moment the recorder is asked (honoured),
    or once that directory's ``package-lock.json`` shows the install completed
    somewhere else (not honoured; ``fell_back_to`` names where). Neither within
    the budget is ``honoured: None``, inconclusive. The model is the recorder
    too, so no model call can succeed.
    """
    results: List[Dict[str, Any]] = []
    with RecordingRegistry() as recorder:
        for setting in candidates:
            iso = isolated_env(Path(root) / setting, binary=binary, registry=None, inherited=inherited)
            materialize(iso)
            env = _apply_registry_candidate(iso, setting, recorder.url)
            iso.opencode_config_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "$schema": "https://opencode.ai/config.json",
                "provider": {PROVIDER: {"npm": "@ai-sdk/openai-compatible", "name": PROVIDER,
                                        "options": {"baseURL": recorder.url + "/v1",
                                                    "apiKey": "row7-probe-not-a-key"},
                                        "models": {MODEL: {"name": MODEL}}}},
                "model": f"{PROVIDER}/{MODEL}", "small_model": f"{PROVIDER}/{MODEL}",
            }
            (iso.opencode_config_dir / "opencode.json").write_text(json.dumps(config), encoding="utf-8")
            work = iso.root / "work"
            work.mkdir(exist_ok=True)
            _git(["init", "-q"], work, iso.env)
            mark = recorder.count()
            lock = iso.opencode_config_dir / "package-lock.json"
            bounded = run_bounded(
                [binary, "run", "--format", "json", "--print-logs", "--log-level", "INFO",
                 "--model", f"{PROVIDER}/{MODEL}", "registry discovery"],
                cwd=work, env=env, budget_s=budget_s,
                stdout=iso.root / "stdout.jsonl", stderr=iso.root / "stderr.txt",
                stop=lambda: bool(recorder.package_requests(mark)) or lock.exists())
            asked = recorder.package_requests(mark)
            honoured: Optional[bool]
            fell_back: List[str] = []
            if asked:
                honoured = True
            elif lock.exists():
                honoured = False
                fell_back = installed_from([iso.opencode_config_dir], None)["origins"]
            else:
                honoured = None
            results.append({"setting": setting, "honoured": honoured, "requests": len(asked),
                            "fell_back_to": fell_back, "elapsed_s": bounded.elapsed_s})
    return results


def _origin(url: str) -> str:
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*://[^/]+)", url)
    return match.group(1) if match else url


def installed_from(config_dirs: Iterable[Path], cache_dir: Optional[Path],
                   registry: Optional[str] = None) -> Dict[str, Any]:
    """Where every package OpenCode installed came from, per its own lockfiles.

    Reads each config directory's ``package-lock.json`` (what OpenCode's
    dependency install writes) and, when ``cache_dir`` is given, every
    ``package-lock.json`` under its ``packages/`` tree (the npm-spec plugin
    cache; its layout is discovered by walking, not assumed). ``foreign`` lists
    each ``resolved`` URL that is not under ``registry``.
    """
    lockfiles = [Path(d) / "package-lock.json" for d in config_dirs]
    if cache_dir is not None and (Path(cache_dir) / "packages").is_dir():
        lockfiles += sorted(p for p in (Path(cache_dir) / "packages").rglob("package-lock.json")
                            if "node_modules" not in p.relative_to(cache_dir).parts)
    read: List[str] = []
    resolved: List[str] = []
    for lockfile in lockfiles:
        if not lockfile.is_file():
            continue
        try:
            packages = json.loads(lockfile.read_text(encoding="utf-8")).get("packages") or {}
        except (OSError, ValueError):
            continue
        read.append(str(lockfile))
        resolved += [str(entry["resolved"]) for key, entry in packages.items()
                     if key and isinstance(entry, dict) and entry.get("resolved")]
    prefix = (registry.rstrip("/") + "/") if registry else None
    return {
        "lockfiles": read,
        "packages": len(resolved),
        "origins": sorted({_origin(url) for url in resolved}),
        "foreign": sorted(url for url in resolved if prefix and not url.startswith(prefix)),
    }


# ---------------------------------------------------------------------------
# Row 7
# ---------------------------------------------------------------------------

def _files(project: Path) -> set:
    found = set()
    for path in Path(project).rglob("*"):
        rel = path.relative_to(project)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if path.is_file():
            found.add(rel.as_posix())
    return found


def make_project(project: Path, kind: str, env: Mapping[str, str]) -> None:
    """A disposable git project: with one commit, or (``no-commit``) none at all.

    The no-commit project holds the same files, uncommitted, exactly as a
    repository is right after ``git init``. It is never given a commit.
    """
    if kind not in PROJECT_KINDS:
        raise ValueError(f"unknown project kind {kind!r}")
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Row-7 probe project\n", encoding="utf-8")
    plan = project / PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text("# Probe plan\n\nNothing reviews this plan; `status` reads it.\n", encoding="utf-8")
    _git(["init", "-q"], project, env)
    if kind == "committed":
        _git(["add", "-A"], project, env)
        _git(["-c", "user.name=row7-probe", "-c", "user.email=row7-probe@example.invalid",
              "-c", "commit.gpgsign=false", "commit", "-q", "-m", "row-7 probe fixture"],
             project, env)


def _project_dir(outcome: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The ``--project`` of the uv argv the tool ran: the skill dir it resolved."""
    argv = (outcome or {}).get("argv")
    if isinstance(argv, list) and "--project" in argv:
        index = argv.index("--project")
        if index + 1 < len(argv):
            return str(argv[index + 1])
    return None


def _error_text(outcome: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not outcome:
        return None
    # `output` too: an exit-1 dispatcher prints its error object on stdout (the
    # no-HEAD `GitError` arrives there, not on stderr).
    text = "\n".join(str(outcome.get(key) or "")
                     for key in ("message", "error", "stderr", "output")).strip()
    return text[-1500:] or None


def grade_call(call: Mapping[str, Any], *, mode: str, want_exit: int, want_state: str,
               surface: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """``call`` (:func:`drive_scripted_call`) against ``mode``'s documented result."""
    problems = list(call["problems"])
    outcome = call.get("outcome") or {}
    result = outcome.get("result") if isinstance(outcome.get("result"), dict) else None
    if call.get("outcome") is not None:
        if outcome.get("ok") is not True:
            problems.append(f"the tool reported ok={outcome.get('ok')!r}, state {outcome.get('state')!r}, "
                            f"exit {outcome.get('exit')!r}")
        if outcome.get("exit") != want_exit:
            problems.append(f"the dispatcher exited {outcome.get('exit')!r}, not {want_exit}")
        if result is None:
            problems.append(f"the result carries no {surface['dispatcher']['result_marker']} payload")
        else:
            if result.get("mode") != mode:
                problems.append(f"the result is mode {result.get('mode')!r}, not {mode}")
            if result.get("state") != want_state or result.get("exit") != want_exit:
                problems.append(f"the result is state {result.get('state')!r} exit "
                                f"{result.get('exit')!r}, not {want_state} exit {want_exit}")
    evidence = {
        "ok": not problems,
        "exit": outcome.get("exit"),
        "state": outcome.get("state"),
        "result": result,
        "opencode_exit": call.get("opencode_exit"),
        "tool_advertised": call["advertised"],
        "returned_to_model": call.get("returned") is not None,
        "returned_identical": call["returned_identical"],
        "skill_dir": _project_dir(call.get("outcome")),
    }
    if problems:
        evidence["problems"] = problems
        evidence["error"] = _error_text(call.get("outcome"))
    return evidence, problems


def run_project(kind: str, root: Path, iso: IsolatedEnv, package_dir: Path, binary: str,
                surface: Mapping[str, Any], diag_root: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Row 7 in one fresh project: debug config, then ``init``, then ``status``."""
    failures: List[str] = []
    diag = diag_root / kind
    diag.mkdir(parents=True, exist_ok=True)
    project = Path(root) / f"project-{kind}"
    make_project(project, kind, iso.env)
    tool = surface["tool_name"]
    codes = surface["dispatcher"]["exit_codes"]
    states = surface["dispatcher"]["states"]
    skill_dir = skill_dir_of(package_dir)
    entry: Dict[str, Any] = {"kind": kind, "project": str(project)}

    call_id = f"call_mllm_row7_init_{kind.replace('-', '_')}"
    with StubModel(scripted_call_scenario(tool, INIT_ARGS, call_id), diag, "init") as stub:
        write_project_config(project, package_dir, stub.base_url)
        debug = read_debug_config(binary, iso.env, project, diag)
        command = check_command(debug.config, package_dir)
        before = _files(project)
        init_call = drive_scripted_call(binary, iso.env, project, stub, tool_name=tool,
                                        call_id=call_id, mode="init", diag=diag, label="init")
        created = sorted(_files(project) - before)
    init, problems = grade_call(init_call, mode="init", want_exit=codes["ok"], want_state=states["ok"],
                                surface=surface)
    init["files_created"] = created
    if PROVIDERS_FILE not in created:
        problems.append(f"init created no {PROVIDERS_FILE} in the project")
        init["ok"] = False
    failures += [f"{kind}: init: {p}" for p in problems]

    call_id = f"call_mllm_row7_status_{kind.replace('-', '_')}"
    with StubModel(scripted_call_scenario(tool, STATUS_ARGS, call_id), diag, "status") as stub:
        write_project_config(project, package_dir, stub.base_url)
        status_call = drive_scripted_call(binary, iso.env, project, stub, tool_name=tool,
                                          call_id=call_id, mode="status", diag=diag, label="status")
    status, problems = grade_call(status_call, mode="status", want_exit=codes["nothing_to_do"],
                                  want_state=states["nothing_to_do"], surface=surface)
    if status_call.get("outcome") and NO_RUN_LINE not in str(status_call["outcome"].get("output") or ""):
        problems.append(f"status did not report {NO_RUN_LINE!r} for a plan with no run")
        status["ok"] = False
    failures += [f"{kind}: status: {p}" for p in problems]

    if debug.error:
        failures.append(f"{kind}: {debug.error}")
    if not command["registered"]:
        failures.append(f"{kind}: debug config has no command[{COMMAND_NAME!r}] template")
    elif not command["matches_payload"]:
        failures.append(f"{kind}: command[{COMMAND_NAME!r}] is not the payload's own body "
                        "resolved against the payload")
    if command["token_left"]:
        failures.append(f"{kind}: command[{COMMAND_NAME!r}] still carries an unresolved skill-dir token")
    if not command["plugin_is_payload"]:
        failures.append(f"{kind}: plugin[] is {command['plugin_specs']}, not the extracted payload alone")
    # `command_skill_dir` is observed, not computed. The command body this
    # build ships routes to the tool and names no path, so the skill directory
    # the plugin resolved (`SKILL_DIR` in `src/index.mjs`, the same constant
    # its config hook resolves `{SKILL_DIR}` with) is read off the tool's own
    # uv argv (`--project <dir>`), and any path the template does name must be
    # inside the payload as well.
    observed = init["skill_dir"] or status["skill_dir"]
    inside = bool(observed) and os.path.realpath(observed) == skill_dir and command["skill_paths_inside"]
    if not inside:
        failures.append(f"{kind}: the tool ran skill dir {observed!r}, not the payload's {skill_dir}")
    tool_registered = bool(init_call["advertised"] and status_call["advertised"])

    entry.update({
        "command_registered": command["registered"] and command["matches_payload"]
        and not command["token_left"],
        "command_skill_dir": observed,
        "command_inside_payload": inside and command["plugin_is_payload"],
        "tool_registered": tool_registered,
        "plugin_specs": command["plugin_specs"],
        "init": init,
        "status": status,
    })
    entry["ok"] = not [f for f in failures if f.startswith(f"{kind}:")]
    return entry, failures


@dataclass
class Steps:
    """The row's moving parts, swappable so the suite can grade the assembly offline."""

    read_version: Callable[[str], str] = None
    pinned_version: Callable[[], str] = None
    tarball_facts: Callable[[Path], Dict[str, Any]] = None
    discover: Callable[..., List[Dict[str, Any]]] = discover_registry_settings
    project: Callable[..., Tuple[Dict[str, Any], List[str]]] = run_project
    installed_from: Callable[..., Dict[str, Any]] = installed_from

    def __post_init__(self) -> None:
        if self.read_version is None:
            self.read_version = lambda binary: _sibling("bump")._read_version(binary)
        if self.pinned_version is None:
            self.pinned_version = lambda: _sibling("bump").pinned_version()
        if self.tarball_facts is None:
            self.tarball_facts = lambda path: _sibling("build").tarball_facts(path)


def new_evidence(tarball: Path, opencode: str, registry: str) -> Dict[str, Any]:
    """The row-7 evidence before anything ran: failing, with every required key."""
    return {
        "row": ROW, "pass": False, "opencode_version": None, "registry": registry,
        "tarball": {"path": str(Path(tarball).absolute()), "integrity": None, "digest": None},
        "probe_root": None,
        "isolation": {"env_cleared": [], "home": None},
        "projects": [], "failures": [],
        "opencode_binary": str(Path(opencode).absolute()),
        "started_at": _utc_now(), "finished_at": None,
    }


def _rmtree(path: Path) -> None:
    """``shutil.rmtree`` that also clears the read-only files uv's cache holds."""
    def retry(func: Callable, target: str, _exc: Any) -> None:
        try:
            os.chmod(target, stat.S_IRWXU)
            func(target)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:  # pragma: no cover - the builder's interpreter is newer
        shutil.rmtree(path, onerror=retry)


def row7(tarball: Path, opencode: str, registry: str, out: Path, keep: bool = False,
         steps: Optional[Steps] = None, tmp_parent: Optional[Path] = None) -> int:
    """The whole row; writes ``out`` in every case; 0 only on a pass."""
    steps = steps or Steps()
    evidence = new_evidence(tarball, opencode, registry)
    failures: List[str] = evidence["failures"]
    probe_root: Optional[Path] = None
    try:
        reg = _sibling("registry")
        try:
            evidence["registry"] = reg.require_single(registry)
        except reg.RegistryError as exc:
            raise ProbeError(str(exc)) from exc
        try:
            pin = steps.pinned_version()
        except Exception as exc:  # BumpError, a pin file gone
            raise ProbeError(f"no one opencode release to probe: {exc}") from exc
        evidence["pinned_version"] = pin
        binary = str(Path(opencode).absolute())
        if not os.path.isfile(binary):
            raise ProbeError(f"--opencode {binary} is not a file")
        try:
            answered = steps.read_version(binary)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProbeError(f"{binary} --version could not be read: {exc}") from exc
        evidence["opencode_version"] = answered
        if answered != pin:
            raise ProbeError(f"{binary} answers --version {answered!r}, not the pin {pin}; "
                             "refusing before the probe runs")
        tarball = Path(tarball).absolute()
        if not tarball.is_file():
            raise ProbeError(f"--tarball {tarball} is not a file")
        try:
            facts = steps.tarball_facts(tarball)
        except Exception as exc:  # build.PayloadError: an archive it cannot account for
            raise ProbeError(f"--tarball {tarball}: {exc}") from exc
        evidence["tarball"] = {"path": str(tarball), "integrity": facts["integrity"],
                               "digest": facts["digest"], "sha256": facts.get("sha256")}

        probe_root = Path(tempfile.mkdtemp(prefix="mllm-row7-", dir=tmp_parent)).resolve()
        evidence["probe_root"] = str(probe_root)
        diag = probe_root / "diagnostics"
        package_dir = extract_package(tarball, probe_root / "payload")
        surface = payload_surface(package_dir)
        iso = isolated_env(probe_root / "env", binary=binary, registry=evidence["registry"])
        if os.path.realpath(str(iso.home)) == os.path.realpath(real_home()):
            raise ProbeError("the isolated HOME resolved to the invoking user's home")
        materialize(iso)
        evidence["isolation"] = {
            "env_cleared": list(iso.cleared),
            "home": str(iso.home),
            "xdg": {name: iso.env[name] for name in sorted(iso.env) if name.startswith("XDG_")},
            "uv_cache": str(iso.uv_cache),
            "registry_setting": REGISTRY_SETTING,
            "registry_discovery": [],
        }

        discovery = steps.discover(binary, probe_root / "discovery")
        evidence["isolation"]["registry_discovery"] = discovery
        verdict = next((d for d in discovery if d.get("setting") == REGISTRY_SETTING), None)
        if not verdict or verdict.get("honoured") is not True:
            failures.append(f"registry: this opencode's package installer did not honour "
                            f"{REGISTRY_SETTING} ({verdict}), so the single registry cannot be "
                            "enforced the way the isolated environment sets it")

        for kind in PROJECT_KINDS:
            try:
                entry, problems = steps.project(kind, probe_root, iso, package_dir, binary,
                                                surface, diag)
            except (ProbeError, OSError, subprocess.SubprocessError) as exc:
                entry = {"kind": kind, "ok": False, "error": str(exc)}
                problems = [f"{kind}: {exc}"]
            evidence["projects"].append(entry)
            failures.extend(problems)
            if kind == PROJECT_KINDS[0]:
                evidence["isolation"]["uv_cache_entries_after_first_project"] = (
                    sum(1 for _ in iso.uv_cache.rglob("*")) if iso.uv_cache.is_dir() else 0)

        installed = steps.installed_from([iso.opencode_config_dir], iso.opencode_cache_dir,
                                         evidence["registry"])
        evidence["installed_from"] = installed
        if installed["foreign"]:
            failures.append("registry: OpenCode installed packages from outside the single "
                            f"registry: {installed['foreign'][:5]}")
        if not installed["lockfiles"]:
            # Recorded, not failed: the payload under test is loaded from a
            # directory, so nothing of ours comes from a registry in this row,
            # and OpenCode's own background install may simply not have
            # finished before the last run ended. A package that DID install
            # from elsewhere is the failure above.
            installed["note"] = ("no package-lock.json yet in " + str(iso.opencode_config_dir)
                                 + ": OpenCode's background install did not finish in time")
    except ProbeError as exc:
        failures.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - the evidence must still be written
        failures.append(f"the probe itself failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        evidence["pass"] = not failures and len(evidence["projects"]) == len(PROJECT_KINDS) and all(
            p.get("ok") for p in evidence["projects"])
        if not evidence["pass"] and not failures:
            failures.append("not every project ran")
        evidence["finished_at"] = _utc_now()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if not evidence["pass"]:
            print(f"row 7: FAIL ({len(failures)} failure(s)); evidence: {out}", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            if probe_root is not None:
                print(f"  diagnostics kept under {probe_root / 'diagnostics'}", file=sys.stderr)
        elif probe_root is not None and not keep:
            _rmtree(probe_root)
    if not evidence["pass"]:
        return 1
    print(f"row 7: PASS; evidence: {out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r7 = sub.add_parser("row7", help="the clean-install probe (gate row 7)")
    r7.add_argument("--tarball", required=True, type=Path, help="the gated .tgz")
    r7.add_argument("--opencode", required=True, help="the pinned binary (bump.py install)")
    r7.add_argument("--registry", required=True, help="Makefile OPENCODE_REGISTRY")
    r7.add_argument("--out", required=True, type=Path, help="where the row-7 evidence JSON goes")
    r7.add_argument("--keep", action="store_true", help="keep probe_root after a pass")
    args = parser.parse_args(argv)
    try:
        return row7(args.tarball, args.opencode, args.registry, args.out, keep=args.keep)
    except Exception:  # noqa: BLE001 - the evidence is written; this is the diagnostic
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
