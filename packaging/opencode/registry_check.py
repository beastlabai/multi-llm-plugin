#!/usr/bin/env python3
"""The registry-install probe: the bytes npm serves, installed by OpenCode as a user would.

Run it through ``make opencode-registry-check`` or as::

    uv run --frozen --no-dev --project skills/multi-llm -- \\
        python packaging/opencode/registry_check.py \\
        --spec opencode-multi-llm@<version> --registry "$(make -s opencode-print-registry)" \\
        --tag opencode-v<version> --environment ci-runner|container|vm --out <record.json> \\
        [--rehearsal-registry http://127.0.0.1:<port>] [--workflow-run-id N] \\
        [--opencode <binary>] [--previous-pin <name>@<version>] [--repo <path>] [--keep]

**One probe, two records** (publishing plan, step 0's bootstrap job and step 5).
Step 5 runs it with ``container`` (or ``vm``) and writes the canonical
``<UTC>-opencode-release-network-check.json``. ``--environment ci-runner``
writes ``<UTC>-opencode-bootstrap-check.json``, the record of the laptop path's
first-release tag push; the workflow's bootstrap path it was written for was
retired after ``opencode-v0.1.1``, and that record covers a first release
only, so every later tag refuses it. Only the environment, and which criteria can be
attempted there, differ. Both records are ``registry-record.schema.json``'s
shape and are validated by ``registry_record.py`` before this script reports
success.

**What it proves, in order, each step's output preserved:**

1. ``npm_spec_resolves`` — a bounded ``npm view <spec> version dist --json``
   poll against the registry (registry and CDN lag the upload; an expired poll
   is "not yet", reported as such), then ``npm pack <spec>`` downloads the
   registry's tarball and its bytes must hash to the ``dist.integrity`` npm
   served, under a tarball URL on that same registry.
2. ``installed_tree_matches_candidate`` — the candidate rebuilt from the
   **tag's commit** with ``--frozen --no-dev`` (a temporary ``git worktree``
   of that commit, never the working tree), compared with the extracted
   registry files **and** with the payload OpenCode installed, **before the
   first tool call** creates a venv or any runtime state. Exact publishable
   inventory both ways (no file missing, none extra) plus the tree digest; the
   only other files tolerated are the installer metadata named in
   :data:`INSTALLER_METADATA`, which lie outside the package directory. A
   mismatch stops the probe there: no tool call runs on bytes that are not
   the candidate.
3. ``manifest_version_matches`` — the installed ``package.json`` (and the
   installer's lockfile entry) name the spec's name and version.
4. ``command_resolves_installed_skill`` — ``opencode debug config`` registers
   ``command["multi-llm"]`` whose template is the installed payload's own
   body (an old file-route shim cannot pass), with no ``{SKILL_DIR}`` left,
   and ``plugin[]`` is exactly the spec.
5. ``tool_init_runs_installed_payload`` — ``init`` through ``multi_llm``
   against the localhost stub model: its documented result, the files it
   wrote, and the skill directory uv was handed (``--project``) must be the
   installed payload's.
6. ``cold_dependencies_install`` — uv's cache was empty and the payload had no
   venv before that first call; afterwards the venv exists and every runtime
   dependency the payload's ``pyproject.toml`` declares is installed in it.
7. ``same_pin_restart_matches`` — a new OpenCode process with the same pin
   resolves the same version at the same location with the same digest, and
   ``status`` behaves as it did before the restart.
8. ``warm_offline_operation`` (``container``/``vm`` only) — see below.
9. ``upgrade_runs_new_payload`` (a later release only, ``previous_pin``) — an
   installation of the previous pin, in its own home and project, holds
   runtime state; the pin changes; after the restart the new payload is the
   one executing.

**Where the payload is, is measured.** ``opencode debug paths`` names the
binary's own directories, and the probe walks them for the one package whose
manifest is the spec. Measured on 1.19.4: ``<cache>/packages/<spec>/
node_modules/<name>``, byte-identical to the tarball, with Arborist's
metadata (``package.json``, ``package-lock.json``,
``node_modules/.package-lock.json``) in the install root beside it; the
lockfile's ``resolved`` is which registry the package actually came from.
Nothing asserts that path: a binary that moves its cache moves the probe's
answer with it, and the record names what was found. On every environment the
location must lie under the probe's root, the isolated home it built.

**The registry** is the single one (``registry.require_single``), or, for a
rehearsal before anything is published, a localhost stub
(``registry_stub.py``) named by ``--rehearsal-registry``, admitted only as
``http://127.0.0.1:<port>``. A rehearsal's record says ``rehearsal: true`` and
is validated structurally only (``registry_record.py validate
--rehearsal-structure-only``); it is never release evidence, and it is never
written under ``tests/container/records/``, where the network-check exception
in ``.gitignore`` would make it committable.

**Isolation** is row 7's (``probe.isolated_env``): a ``probe_root`` holding
``HOME``, every ``XDG_*``, npm's cache and config files and uv's cache, all
empty at the start (recorded), every inherited ``OPENCODE_*`` and friends
left behind. ``npm_config_registry`` carries the registry, the setting
OpenCode 1.19.4's bundled installer honours (measured by row 7). The probe's
own npm reads use caches of their own, so they never warm OpenCode's. The
project is a plain ``git init`` with no commit. The root is made inside
``--tmp-parent`` whenever it is given (the retired bootstrap job handed it a
fresh, checked-empty directory, and its guard required the payload under that
directory), else in ``$RUNNER_TEMP`` on a runner, else the system temp; on a
runner it lies outside the checkout, which only rebuilds the candidate.

**The container environment** is this script driving Docker from the host,
because the warm-offline criterion needs a network cut that a process cannot
make for itself (this machine restricts unprivileged user namespaces, so no
``unshare -n``). In order:

* a STOCK container (``cli-versions.lock``'s base image, by digest; never the
  pre-warmed harness image, and the source tree is never mounted) with its
  network; ``apt-get`` adds git, CA certificates and Python (3.11, Debian's,
  plus ``python3-yaml`` for the stub model's scenario loader), uv is the
  pinned release fetched by URL and checked against its sha256 before it
  runs, and OpenCode is ``npm ci --ignore-scripts`` of the pinned install's
  lockfile subset (``bump.install_subset``), checked by
  ``bump.verify_installed`` as each phase starts (every installed package
  from the single registry with the committed integrity, nothing else in
  ``node_modules``), which also chooses the binary the phase runs: a verified
  platform package's own ``bin/opencode``, in the order ``opencode-ai``'s
  install script would have tried them, the script itself never run; then
  ``--version``;
* ``docker cp`` of the probe KIT only (:data:`KIT_FILES`: this script, the
  row-7 probe library, the pinned install's check, the stub model and its
  scenario matcher, the payload digest module) and a facts file, never the
  candidate's sources: the candidate is rebuilt on the host and only its
  inventory and digest go in;
* the install phase, inside, as an unprivileged user (criteria 1-7, and 9);
* ``docker stop``, ``docker commit``, then the committed image run with
  ``--network none``: the warm-offline ``status`` through the installed
  payload, with ``OPENCODE_DISABLE_MODELS_FETCH=1`` and nothing else added
  (``UV_OFFLINE`` and ``OPENCODE_DISABLE_AUTOUPDATE`` stay unset, every
  variable is recorded). Around it the proof: ``NetworkMode`` from ``docker
  inspect`` before and after, the interfaces that are up (loopback only), an
  ``npm view`` against the single registry failing with a NETWORK code (not
  ``E404``, not an auth error), and the payload's location and digest
  unchanged. Then the containers and the committed image are deleted; the
  image is never pushed.

For a rehearsal the stub runs on the host's loopback, and how the networked
container reaches it is chosen by what Docker this is and recorded: Docker
Desktop's ``--network host`` is its VM's network, where the host's loopback
answers ``ECONNREFUSED`` (measured 2026-10-25), so there the container gets
``--add-host host.docker.internal:host-gateway`` and a loopback relay
(``registry_stub.py relay``) keeps the in-container URL
``http://127.0.0.1:<port>``, the URL the stub's packument names; Docker Engine
gets ``--network host``. The offline phase has neither.

**Interfaces that are up.** The proof lists the interfaces whose ``IFF_UP``
flag is set, and the criterion's evidence keeps the full listing with flags
and the local addresses. Measured on Docker Desktop's linuxkit kernel: every
network namespace, ``--network none`` included, holds the kernel's fallback
tunnel devices (``tunl0``, ``gre0``, ``sit0``, ``ip6tnl0`` …), all down and
without an address; only ``lo`` is up, and the only routes are
``127.0.0.0/8``.

**The VM environment** runs the phases in this process; for the offline
criterion it waits, bounded, for the VM's network to be detached (only
loopback up), runs the same operation and proof, and records
``network_mode: none`` only if the cut was observed before and after.

**Evidence and exit.** ``--out`` gets the record whenever the criteria ran
(failing criteria and all; a refusal before that writes nothing). The
supporting output (every child's streams, the stub logs, the container's raw
phase results, the candidate's build log) goes to ``<out stem>.support/``
beside it, on a refusal too: a ``CheckError`` is when it is needed most. Exit
0 only when every criterion passed and the validator accepted the record.
``probe_root`` is removed after a pass unless ``--keep``.

Every subprocess is bounded, and never by a ``timeout`` wrapper around
``opencode run``: every long one (the candidate's build, the pinned install,
OpenCode, Docker) through ``probe.run_bounded``, whose session kill reaches the
grandchildren a direct-child ``timeout`` would leave behind; only short ``git``
reads keep ``subprocess.run(timeout=)``. ``main`` installs
``probe.install_exit_handlers``, so a SIGTERM unwinds through every sweep
rather than ending the interpreter with OpenCode still running. Importable
without side effects; the suite drives the assembly with fakes (:class:`Steps`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
PAYLOAD_MODULE = REPO_ROOT / "packaging" / "common" / "payload.py"
RECORDS_DIR = REPO_ROOT / "skills" / "multi-llm" / "tests" / "container" / "records"
CLI_VERSIONS_LOCK = REPO_ROOT / "skills" / "multi-llm" / "tests" / "container" / "cli-versions.lock"

KIND = "opencode-registry-install-check"
SCHEMA_VERSION = 1
ENVIRONMENTS = ("ci-runner", "container", "vm")

#: The criteria, by name (``registry_record``'s tuples are the source of truth
#: for which sets a profile requires; these are the names this probe emits).
NPM = "npm_spec_resolves"
MANIFEST = "manifest_version_matches"
TREE = "installed_tree_matches_candidate"
COMMAND = "command_resolves_installed_skill"
INIT = "tool_init_runs_installed_payload"
COLD = "cold_dependencies_install"
OFFLINE = "warm_offline_operation"
RESTART = "same_pin_restart_matches"
UPGRADE = "upgrade_runs_new_payload"

#: The registry stub, as a rehearsal names it: loopback, a port, nothing else.
_REHEARSAL = re.compile(r"^http://127\.0\.0\.1:([0-9]{1,5})/?$")
#: ``<name>@<exact version>``; a scoped name keeps its leading ``@``.
_SPEC = re.compile(r"^((?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*)@"
                   r"([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)$")
_VERSION_TEXT = re.compile(r"([0-9]+\.[0-9]+\.[0-9]+)")

#: The files OpenCode's installer writes beside the package rather than in it,
#: relative to the install root (the directory holding ``node_modules``).
#: Measured on 1.19.4 (npm's Arborist, bundled): nothing inside the package
#: directory, these three outside it. Any other file there is a mismatch.
INSTALLER_METADATA = ("package.json", "package-lock.json", "node_modules/.package-lock.json")
#: The build's own digest file: shipped, but outside the digest it records.
DIGEST_FILE = "dist/tree.sha256"
#: The placeholder the plugin resolves at startup (``SKILL_DIR_TOKEN`` in
#: ``src/index.mjs`` and ``common/surface.py``; the suite pins the two equal).
#: Spelled here because the container kit carries no skill tree to import it from.
SKILL_DIR_TOKEN = "{SKILL_DIR}"

#: Bounded npm reads, as row 9 and the laptop path's post-publish read bound theirs.
NPM_FETCH_FLAGS = (
    "--fetch-retries=2", "--fetch-retry-mintimeout=2000",
    "--fetch-retry-maxtimeout=10000", "--fetch-timeout=30000",
)
#: The offline proof wants an answer, not npm's retries.
NPM_PROOF_FLAGS = ("--fetch-retries=0", "--fetch-timeout=10000")
#: How long ``npm view`` is polled for the version to become visible (step 4's lag).
NPM_POLL_S = 300.0
NPM_POLL_INTERVAL_S = 10.0

#: Budgets in seconds, each a ceiling on one child.
NPM_BUDGET_S = 150
GIT_BUDGET_S = 120
BUILD_BUDGET_S = 900
VERSION_BUDGET_S = 60
HOST_INSTALL_BUDGET_S = 900
DOCKER_SHORT_S = 120
DOCKER_SETUP_S = 1200
DOCKER_PHASE_S = 2400
DOCKER_OFFLINE_S = 900
VM_CUT_WAIT_S = 900

#: The placeholder model URL for ``debug config`` runs, which never call a model
#: (port 9, discard: refused if anything ever did).
NO_MODEL_URL = "http://127.0.0.1:9/v1"
#: ``IFF_UP`` in ``/sys/class/net/<if>/flags``.
IFF_UP = 0x1
_LOOPBACK = re.compile(r"^lo[0-9]*$")

#: The probe KIT the container driver copies in, relative to the repository.
#: Only test equipment: the candidate is rebuilt on the host, and nothing of
#: the plugin's own sources enters the container except as npm serves it.
#: ``registry_record`` comes along because it is where the criteria sets are
#: defined; its PyYAML import (the tag read) never runs inside. ``bump`` is the
#: pinned install's own check (:func:`bump.verify_installed`), run inside on
#: what ``npm ci`` put down there, and the one that chooses the binary.
KIT_FILES = (
    "packaging/opencode/registry_check.py",
    "packaging/opencode/probe.py",
    "packaging/opencode/bump.py",
    "packaging/opencode/registry.py",
    "packaging/opencode/registry_record.py",
    "packaging/opencode/registry-record.schema.json",
    "packaging/opencode/registry_stub.py",
    "packaging/common/payload.py",
    "skills/multi-llm/tests/__init__.py",
    "skills/multi-llm/tests/scenario_matcher.py",
    "skills/multi-llm/tests/container/stub_server.py",
)
#: Where things live inside the container.
KIT_ROOT = "/opt/mllm-registry-check"
PIN_PREFIX = "/opt/opencode-pin"
WORK = "/work"
CONTAINER_UID = 10001
CONTAINER_USER = "mllm"
IMAGE_LABEL = "mllm.registry-check"


class CheckError(RuntimeError):
    """A refusal: the probe cannot go on, and says why."""


# ---------------------------------------------------------------------------
# Sibling modules, loaded by path when first needed
# ---------------------------------------------------------------------------

_MODULES: Dict[str, Any] = {}


def _load(module_name: str, path: Path) -> Any:
    """``path`` imported once under ``module_name``, as ``probe._sibling`` does.

    By path, and on first use: inside the container only the kit exists, so
    ``build`` (which imports the skill tree) and ``registry_record`` (which
    needs PyYAML for the tag) must not be imported by a phase that never
    calls them.
    """
    if module_name not in _MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        _MODULES[module_name] = module
    return _MODULES[module_name]


def _probe() -> Any:
    return _load("opencode_probe", PACKAGE_DIR / "probe.py")


def _payload() -> Any:
    return _load("opencode_common_payload", PAYLOAD_MODULE)


def _registry() -> Any:
    return _load("opencode_registry", PACKAGE_DIR / "registry.py")


def _record_module() -> Any:
    return _load("opencode_registry_record", PACKAGE_DIR / "registry_record.py")


def _bump() -> Any:
    return _load("opencode_bump", PACKAGE_DIR / "bump.py")


def _build() -> Any:
    return _load("opencode_build", PACKAGE_DIR / "build.py")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _read(path: Path, limit: int = 4000) -> str:
    """The last ``limit`` characters of a text file, or ``""``."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _bounded_view(bounded: Any) -> Dict[str, Any]:
    """A :class:`probe.Bounded`, as evidence."""
    return {"argv": list(bounded.argv), "exit": bounded.returncode, "killed": bounded.killed,
            "reason": bounded.reason, "elapsed_s": bounded.elapsed_s,
            "stdout": bounded.stdout.name, "stderr": bounded.stderr.name}


def _rmtree(path: Path) -> None:
    if Path(path).exists():
        _probe()._rmtree(Path(path))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def parse_spec(spec: str) -> Tuple[str, str]:
    """``(name, version)`` of an exact ``<name>@<version>`` pin, or :class:`CheckError`."""
    match = _SPEC.match(spec or "")
    if not match:
        raise CheckError(f"--spec {spec!r} is not an exact <name>@<x.y.z> pin")
    return match.group(1), match.group(2)


def resolve_registries(registry: str, rehearsal_registry: Optional[str] = None,
                       makefile: Optional[Path] = None) -> Tuple[str, str, bool]:
    """``(single, read_registry, rehearsal)``.

    ``registry`` must be the single registry, whatever else is given: the
    record names it, and the offline proof always asks it. A rehearsal
    registry is admitted only as loopback, and makes every npm read and
    OpenCode's installer use it instead, and the record a rehearsal.
    """
    reg = _registry()
    try:
        single = reg.require_single(registry, makefile)
    except reg.RegistryError as exc:
        raise CheckError(str(exc)) from exc
    if rehearsal_registry is None:
        return single, single, False
    match = _REHEARSAL.match(rehearsal_registry)
    if not match or not 0 < int(match.group(1)) < 65536:
        raise CheckError(
            f"--rehearsal-registry {rehearsal_registry!r} is not a loopback stub "
            "(http://127.0.0.1:<port>): a rehearsal never reads another registry")
    return single, reg.normalize(rehearsal_registry), True


def release_kind(public_version: Optional[str]) -> str:
    """From the tag's commit, never from the version number (step 5's exception)."""
    return "first" if public_version is None else "later"


def criteria_for(environment: str, kind: str) -> Tuple[str, ...]:
    """Exactly the criteria a record of this environment and release kind carries."""
    rr = _record_module()
    if environment == "ci-runner":
        return rr.BOOTSTRAP_CRITERIA
    return rr.LATER_RELEASE_CRITERIA if kind == "later" else rr.FIRST_RELEASE_CRITERIA


def profile_for(environment: str) -> str:
    return "bootstrap" if environment == "ci-runner" else "release"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.realpath(str(path))).relative_to(os.path.realpath(str(parent)))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# The candidate, rebuilt from the tag's commit
# ---------------------------------------------------------------------------

def default_build_argv(worktree: Path, destination: Path) -> List[str]:
    """The tag's own build, under the tag's own lock and no dev group."""
    return ["uv", "run", "--frozen", "--no-dev", "--project", str(worktree / "skills" / "multi-llm"),
            "--", "python", str(worktree / "packaging" / "opencode" / "build.py"),
            "--destination", str(destination)]


def _plain_run(argv: Sequence[str], cwd: Path, env: Mapping[str, str], budget_s: float,
               log: Path) -> Dict[str, Any]:
    """A bounded child whose streams are appended to ``log``; ``exit`` is
    ``None`` when its budget (or an exit signal) ended it.

    ``probe.run_bounded``, not ``subprocess.run(timeout=)``: the build is
    ``uv run`` starting Python, and a timeout that kills only the direct child
    leaves the build running under the next step.
    """
    argv = [str(a) for a in argv]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        out.write(("$ " + " ".join(argv) + "\n").encode("utf-8"))
    try:
        bounded = _probe().run_bounded(argv, cwd=Path(cwd), env=env, budget_s=budget_s,
                                       stdout=log, stderr=log, append=True)
    except OSError as exc:
        with open(log, "ab") as out:
            out.write(f"cannot run: {exc}\n".encode("utf-8"))
        return {"argv": argv, "exit": 127}
    return {"argv": argv, "exit": bounded.returncode}


def _git_text(repo: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=GIT_BUDGET_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def file_inventory(directory: Path) -> Tuple[List[str], List[str]]:
    """``(regular files, anything else)``, as sorted ``/``-separated relative paths.

    A symlink, device or socket is never part of a publishable inventory, so
    it is reported rather than followed.
    """
    directory = Path(directory)
    files: List[str] = []
    odd: List[str] = []
    for root, dirs, names in os.walk(directory, followlinks=False):
        here = Path(root)
        for name in list(dirs):
            if (here / name).is_symlink():
                odd.append((here / name).relative_to(directory).as_posix())
                dirs.remove(name)
        for name in names:
            path = here / name
            rel = path.relative_to(directory).as_posix()
            if path.is_symlink() or not path.is_file():
                odd.append(rel)
            else:
                files.append(rel)
    return sorted(files), sorted(odd)


def inventory_digest(directory: Path, digest_files: Iterable[str]) -> str:
    """``payload.tree_digest`` over exactly ``digest_files`` under ``directory``.

    The build's digest rule, restricted to the candidate's own inventory, so a
    venv or a ``__pycache__`` the first tool call left inside the installed
    payload is runtime state rather than a change to the payload.
    """
    wanted = frozenset(digest_files)
    return _payload().tree_digest(Path(directory), exclude=lambda rel: rel not in wanted)


def rebuild_candidate(repo: Path, sha: str, scratch: Path, *,
                      build_argv: Callable[[Path, Path], List[str]] = default_build_argv,
                      budget_s: float = BUILD_BUDGET_S) -> Dict[str, Any]:
    """The candidate the tag's commit builds: its inventory and digest.

    A temporary ``git worktree`` of ``sha`` (detached), so what is built is
    that commit's files and nothing of the working tree: an uncommitted edit
    or an untracked file in ``repo`` cannot reach it. The build writes into a
    destination outside the worktree, with uv's project environment outside
    it too, so the worktree stays exactly the commit; it is removed afterwards
    whatever happened. Raises :class:`CheckError` when the build fails or its
    output cannot be accounted for.
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    worktree = scratch / "source"
    destination = scratch / "candidate"
    log = scratch / "build.log"
    env = {k: v for k, v in os.environ.items()
           if k not in ("MULTI_LLM_ALLOW_DIRTY", "VIRTUAL_ENV")}
    env["UV_PROJECT_ENVIRONMENT"] = str(scratch / "build-venv")
    added = _plain_run(["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), sha],
                       repo, env, GIT_BUDGET_S, log)
    if added["exit"] != 0:
        raise CheckError(f"cannot check out {sha} as a temporary worktree of {repo}; see {log}")
    try:
        head = _git_text(worktree, "rev-parse", "HEAD")
        if head != sha:
            raise CheckError(f"the temporary worktree is at {head}, not the tag's commit {sha}")
        built = _plain_run(build_argv(worktree, destination), worktree, env, budget_s, log)
        if built["exit"] != 0:
            raise CheckError(f"the tag's build exited {built['exit']}; see {log}")
        status = _git_text(worktree, "status", "--porcelain", "--untracked-files=all")
    finally:
        _plain_run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                   repo, env, GIT_BUDGET_S, log)
        _plain_run(["git", "-C", str(repo), "worktree", "prune"], repo, env, GIT_BUDGET_S, log)
    files, odd = file_inventory(destination)
    if odd:
        raise CheckError(f"the tag's build wrote links or devices: {odd[:5]}")
    recorded_path = destination / DIGEST_FILE
    if not recorded_path.is_file():
        raise CheckError(f"the tag's build wrote no {DIGEST_FILE}")
    recorded = recorded_path.read_text(encoding="utf-8").strip()
    digest_files = [f for f in files if f != DIGEST_FILE]
    recomputed = inventory_digest(destination, digest_files)
    if recomputed != recorded:
        raise CheckError(
            f"the tag's build records digest {recorded} but its inventory hashes to "
            f"{recomputed}: this probe cannot account for what the candidate digest covers")
    return {
        "sha": sha, "digest": recorded, "inventory": files, "digest_files": digest_files,
        "files": len(files), "built_in": "a temporary git worktree of the tag's commit",
        "build": {"argv": build_argv(Path("<worktree>"), Path("<destination>")),
                  "exit": built["exit"],
                  "worktree_status_after": status if status is not None else "<unreadable>"},
        "destination": str(destination),
    }


# ---------------------------------------------------------------------------
# Trees: the registry's files and OpenCode's install against the candidate
# ---------------------------------------------------------------------------

def compare_tree(directory: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """``directory`` against the candidate: exact inventory and the tree digest."""
    files, odd = file_inventory(directory)
    expected = set(candidate["inventory"])
    present = set(files)
    missing = sorted(expected - present)
    unexpected = sorted((present - expected) | set(odd))
    digest = inventory_digest(directory, candidate["digest_files"])
    recorded_path = Path(directory) / DIGEST_FILE
    recorded = (recorded_path.read_text(encoding="utf-8", errors="replace").strip()
                if recorded_path.is_file() else None)
    return {
        "location": str(directory), "files": len(files),
        "missing": missing[:50], "missing_count": len(missing),
        "unexpected": unexpected[:50], "unexpected_count": len(unexpected),
        "digest": digest, "recorded_digest": recorded,
        "matches": not missing and not unexpected and digest == candidate["digest"]
        and recorded == candidate["digest"],
    }


def payload_state(location: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Where the payload is and what its candidate inventory hashes to, now."""
    files, _odd = file_inventory(location) if Path(location).is_dir() else ([], [])
    missing = sorted(set(candidate["inventory"]) - set(files))
    return {"location": str(location),
            "digest": inventory_digest(location, candidate["digest_files"]) if not missing else
            "<missing files>", "missing": missing[:20]}


def find_installed(roots: Iterable[Path], name: str, version: str) -> List[Path]:
    """Every installed package directory under ``roots`` that is ``name@version``.

    A walk, not a path: a ``package.json`` naming ``name`` and ``version`` in a
    directory whose parent is a ``node_modules`` (or a scope directory inside
    one). Runtime state the first tool call leaves (``.venv``,
    ``__pycache__``) is not descended into.
    """
    found: List[Path] = []
    seen = set()
    leaf = name.split("/")[-1]
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            here = Path(current)
            dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__", ".git")]
            if "package.json" not in files or here.name != leaf:
                continue
            parent = here.parent if not name.startswith("@") else here.parent.parent
            if parent.name != "node_modules":
                continue
            try:
                manifest = json.loads((here / "package.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.get("name") == name and manifest.get("version") == version:
                real = os.path.realpath(str(here))
                if real not in seen:
                    seen.add(real)
                    found.append(here)
                dirs[:] = []
    return found


def install_root_of(package_dir: Path) -> Path:
    """The directory holding the ``node_modules`` the package was installed into."""
    current = Path(package_dir)
    while current.name != "node_modules":
        if current.parent == current:
            raise CheckError(f"{package_dir} is not inside a node_modules directory")
        current = current.parent
    return current.parent


def installer_metadata(install_root: Path, package_dir: Path) -> Dict[str, Any]:
    """The installer's files beside the package: identified, or unexpected."""
    files, odd = file_inventory(install_root)
    inside = Path(package_dir).relative_to(install_root).as_posix() + "/"
    outside = [f for f in files if not f.startswith(inside)]
    identified = [f for f in outside if f in INSTALLER_METADATA]
    unexpected = sorted([f for f in outside if f not in INSTALLER_METADATA] + odd)
    return {"install_root": str(install_root), "identified": identified,
            "unexpected": unexpected[:50], "unexpected_count": len(unexpected)}


def installed_origin(install_root: Path, name: str) -> Dict[str, Any]:
    """What the installer's lockfile says about ``name``: where it came from."""
    lockfile = Path(install_root) / "package-lock.json"
    try:
        packages = json.loads(lockfile.read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, ValueError) as exc:
        return {"lockfile": str(lockfile), "error": f"unreadable ({exc})"}
    entry = packages.get(f"node_modules/{name}") or {}
    resolved = entry.get("resolved")
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*://[^/]+)", resolved or "")
    return {"lockfile": str(lockfile), "version": entry.get("version"), "resolved": resolved,
            "integrity": entry.get("integrity"), "origin": match.group(1) if match else None}


# ---------------------------------------------------------------------------
# npm
# ---------------------------------------------------------------------------

def npm_error_code(stdout: str, stderr: str) -> Optional[str]:
    """npm's error code: ``--json``'s ``.error.code`` on stdout, else its stderr line."""
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            code = parsed["error"].get("code")
            if isinstance(code, str):
                return code
    except ValueError:
        pass
    match = re.search(r"^npm (?:ERR!|error) code (\S+)", stderr or "", re.M)
    return match.group(1) if match else None


def _npm_argv(*words: str, registry: str, cache: Path, userconfig: Path, prefix: Path,
              flags: Sequence[str] = NPM_FETCH_FLAGS) -> List[str]:
    """npm with only the probe's own settings: the registry and a fresh cache on
    the command line (they outrank every other layer), an empty user file (the
    isolated environment names an empty global one), and ``--prefix`` naming
    the probe's own npm scratch directory, which holds no ``package.json``,
    ``node_modules`` or ``.npmrc``. That flag is what pins npm's project root:
    without it npm walks up from its cwd to the first ``package.json`` or
    ``node_modules`` and reads the ``.npmrc`` there, above the empty files
    (measured on npm 10.9.3 and 11.5.1; ``registry_absent.mjs`` says how)."""
    return ["npm", *words, f"--registry={registry}", f"--cache={cache}",
            f"--userconfig={userconfig}", f"--prefix={prefix}", "--no-update-notifier",
            *flags]


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------

def parse_debug_paths(text: str) -> Dict[str, str]:
    """``opencode debug paths``: one ``<name> <path>`` per line."""
    paths = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and os.path.isabs(parts[1].strip()):
            paths[parts[0]] = parts[1].strip()
    return paths


def project_config(spec: str, base_url: str) -> Dict[str, Any]:
    """The project's ``opencode.json``: the exact npm pin and the stub model.

    The provider block is row 7's (``probe.opencode_config``), taken whole so
    the two probes drive OpenCode against the same stub the same way; only
    ``plugin[]`` differs: the spec, never a directory.
    """
    config = _probe().opencode_config(Path(__file__).resolve().parent, base_url)
    config["plugin"] = [spec]
    return config


def write_project_config(project: Path, spec: str, base_url: str) -> None:
    (Path(project) / "opencode.json").write_text(
        json.dumps(project_config(spec, base_url), indent=2) + "\n", encoding="utf-8")


def _default_debug_paths(binary: str, env: Mapping[str, str], cwd: Path, diag: Path,
                         label: str) -> Dict[str, Any]:
    bounded = _probe().run_bounded([binary, "debug", "paths"], cwd=cwd, env=env,
                                   budget_s=VERSION_BUDGET_S, stdout=diag / f"{label}.txt",
                                   stderr=diag / f"{label}.stderr.txt")
    paths = parse_debug_paths(_read(bounded.stdout, 20000)) if bounded.returncode == 0 else {}
    return {"bounded": _bounded_view(bounded), "paths": paths}


def _default_debug_config(binary: str, env: Mapping[str, str], cwd: Path, diag: Path,
                          label: str) -> Any:
    return _probe().read_debug_config(binary, env, cwd, diag, label=label)


def _default_tool_call(binary: str, env: Mapping[str, str], project: Path, spec: str, tool: str,
                       arguments: Mapping[str, Any], mode: str, call_id: str, diag: Path,
                       label: str) -> Dict[str, Any]:
    """One scripted ``multi_llm`` call through ``opencode run`` (row 7's driver)."""
    probe = _probe()
    with probe.StubModel(probe.scripted_call_scenario(tool, arguments, call_id), diag, label) as stub:
        write_project_config(project, spec, stub.base_url)
        call = probe.drive_scripted_call(binary, env, project, stub, tool_name=tool,
                                         call_id=call_id, mode=mode, diag=diag, label=label)
    write_project_config(project, spec, NO_MODEL_URL)
    return call


def _default_run(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], budget_s: float,
                 stdout: Path, stderr: Path) -> Any:
    return _probe().run_bounded(argv, cwd=cwd, env=env, budget_s=budget_s, stdout=stdout,
                                stderr=stderr)


@dataclass
class Steps:
    """The probe's moving parts, swappable so the suite can grade the assembly offline."""

    run: Callable[..., Any] = _default_run
    debug_paths: Callable[..., Dict[str, Any]] = _default_debug_paths
    debug_config: Callable[..., Any] = _default_debug_config
    tool_call: Callable[..., Dict[str, Any]] = _default_tool_call
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    sys_class_net: Path = Path("/sys/class/net")
    proc_net: Path = Path("/proc/net")


@dataclass
class Context:
    """One environment's run: what is probed, against what, where."""

    spec: str
    name: str
    version: str
    read_registry: str
    single_registry: str
    rehearsal: bool
    environment: str
    candidate: Mapping[str, Any]
    binary: str
    opencode_pin: str
    probe_root: Path
    previous_pin: Optional[str] = None
    poll_s: float = NPM_POLL_S
    interval_s: float = NPM_POLL_INTERVAL_S


def _version_of(steps: Steps, argv: Sequence[str], env: Mapping[str, str], cwd: Path,
                diag: Path, label: str) -> Optional[str]:
    bounded = steps.run(list(argv), cwd=cwd, env=env, budget_s=VERSION_BUDGET_S,
                        stdout=diag / f"{label}.txt", stderr=diag / f"{label}.stderr.txt")
    if bounded.returncode != 0:
        return None
    match = _VERSION_TEXT.search(_read(bounded.stdout) + " " + _read(bounded.stderr))
    return match.group(1) if match else None


def _entries(path: Path) -> int:
    """How many entries ``path`` holds, recursively; 0 for one that does not exist."""
    path = Path(path)
    return sum(1 for _ in path.rglob("*")) if path.is_dir() else 0


def runtime_requirements(skill_dir: Path) -> List[str]:
    """The payload's ``[project].dependencies``, as normalized distribution names."""
    text = (Path(skill_dir) / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib

        deps = (tomllib.loads(text).get("project") or {}).get("dependencies") or []
    except ImportError:  # pragma: no cover - Python 3.10
        block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
        deps = re.findall(r"[\"']([^\"']+)[\"']", block.group(1)) if block else []
    return sorted({_dist_name(d) for d in deps})


def _dist_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-") \
        .replace(".", "-")


_LIST_DISTS = ("import importlib.metadata as m, json, sys; "
               "print(json.dumps({'python': sys.version.split()[0], 'dists': sorted("
               "(d.metadata['Name'] or '').lower().replace('_', '-').replace('.', '-') "
               "for d in m.distributions())}))")


# ---------------------------------------------------------------------------
# The network, as the offline proof sees it
# ---------------------------------------------------------------------------

def interfaces_snapshot(sys_class_net: Path = Path("/sys/class/net")) -> Dict[str, Any]:
    """The interfaces that are up, and every interface with its flags.

    ``up`` is what the proof records: an interface whose ``IFF_UP`` is clear
    carries no packets. Every entry is kept in ``all``, so the reading can be
    checked (``bonding_masters``, a control file rather than an interface, is
    skipped because it has no ``flags``).
    """
    detail = []
    root = Path(sys_class_net)
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        flags_file = entry / "flags"
        if not flags_file.is_file():
            continue
        try:
            flags = int(flags_file.read_text(encoding="utf-8").strip(), 16)
        except (OSError, ValueError):
            continue
        operstate = _read(entry / "operstate", 64).strip() or None
        detail.append({"name": entry.name, "flags": hex(flags), "up": bool(flags & IFF_UP),
                       "operstate": operstate})
    return {"up": [d["name"] for d in detail if d["up"]], "all": detail}


def local_addresses(proc_net: Path = Path("/proc/net")) -> List[str]:
    """Every address this network namespace owns (IPv4 ``LOCAL`` routes, IPv6)."""
    addresses: List[str] = []
    lines = _read(Path(proc_net) / "fib_trie", 200000).splitlines()
    for index, line in enumerate(lines[:-1]):
        match = re.search(r"\|--\s+([0-9.]+)\s*$", line)
        if match and "host LOCAL" in lines[index + 1]:
            addresses.append(match.group(1))
    for line in _read(Path(proc_net) / "if_inet6", 200000).splitlines():
        parts = line.split()
        if parts and re.fullmatch(r"[0-9a-f]{32}", parts[0]):
            raw = parts[0]
            addresses.append(":".join(raw[i:i + 4] for i in range(0, 32, 4)))
    return sorted(set(addresses))


def network_snapshot(ctx: Context, env: Mapping[str, str], steps: Steps, diag: Path,
                     label: str, scratch: Path) -> Dict[str, Any]:
    """One side of the offline proof: interfaces, addresses, and an ``npm view``
    of the spec against the SINGLE registry (a rehearsal too: the cut is about
    the network, not about the stub)."""
    cache = scratch / f"npm-cache-{label}"
    cache.mkdir(parents=True, exist_ok=True)
    userconfig = scratch / "npmrc-empty"
    userconfig.write_text("", encoding="utf-8")
    argv = _npm_argv("view", ctx.spec, "version", "--json", registry=ctx.single_registry,
                     cache=cache, userconfig=userconfig, prefix=scratch, flags=NPM_PROOF_FLAGS)
    bounded = steps.run(argv, cwd=scratch, env=env, budget_s=NPM_BUDGET_S,
                        stdout=diag / f"{label}.npm-view.json",
                        stderr=diag / f"{label}.npm-view.stderr.txt")
    stdout, stderr = _read(bounded.stdout), _read(bounded.stderr)
    interfaces = interfaces_snapshot(steps.sys_class_net)
    return {
        "taken_at": _utc_now(),
        "interfaces": interfaces["up"],
        "npm_view": {"registry": ctx.single_registry, "exit": bounded.returncode,
                     "code": npm_error_code(stdout, stderr) if bounded.returncode != 0 else None,
                     "argv": [a if not a.startswith("--cache=") else "--cache=<fresh>" for a in argv]},
        "_detail": {"interfaces": interfaces["all"],
                    "addresses": local_addresses(steps.proc_net),
                    "npm_view_output": (stdout or stderr)[-800:]},
    }


def assess_network_proof(proof: Mapping[str, Any], single: str) -> List[str]:
    """Why ``proof`` does not show the offline run had no network, or ``[]``.

    The validator's own check (``registry_record._network_proof_errors``):
    ``NetworkMode`` ``none``, loopback-only interfaces and a NETWORK error code
    from ``npm view`` before and after (``E404`` and auth codes are answers
    from a registry, so they refute the cut), the payload unchanged. Run here
    too, so the probe fails the criterion itself rather than writing a record
    only the validator would refuse.
    """
    return list(_record_module()._network_proof_errors(dict(proof), single))


def _proof_side(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """A snapshot in the schema's closed ``offline_probe`` shape."""
    return {key: snapshot[key] for key in ("taken_at", "interfaces", "npm_view")}


# ---------------------------------------------------------------------------
# One environment's criteria
# ---------------------------------------------------------------------------

class _Run:
    """Criteria as they are decided, and the reasons the rest were not attempted."""

    def __init__(self) -> None:
        self.criteria: Dict[str, Dict[str, Any]] = {}
        self.failures: List[str] = []

    def decide(self, name: str, problems: List[str], evidence: Dict[str, Any]) -> bool:
        ok = not problems
        if problems:
            evidence = dict(evidence, problems=problems)
            self.failures += [f"{name}: {p}" for p in problems]
        self.criteria[name] = {"pass": ok, "evidence": evidence}
        return ok

    def skip(self, names: Iterable[str], reason: str) -> None:
        for name in names:
            if name not in self.criteria:
                self.criteria[name] = {"pass": False, "evidence": {"not_attempted": reason}}
                self.failures.append(f"{name}: not attempted: {reason}")


#: What a failed network attempt looks like in OpenCode's own log (``--print-logs``
#: puts it on the run's stderr): a WARN or ERROR line naming a fetch, an install
#: or a resolver/connection error. Measured on 1.19.4 offline: the config
#: directory's ``@opencode-ai/plugin`` install holds the session's start for
#: npm's retries (~70 s), then fails at WARN and the run goes on without it.
_NETWORK_ATTEMPT = re.compile(
    r"level=(?:WARN|ERROR)\b.*(?:fetch|install|upgrade|update|github|registry|ECONN|EAI_|ENOTFOUND|"
    r"ETIMEDOUT|ENETUNREACH)", re.I)


def network_attempts(text: str, limit: int = 20) -> List[str]:
    """The failed network attempts OpenCode logged, each line cut to 400 characters."""
    return [line[:400] for line in text.splitlines() if _NETWORK_ATTEMPT.search(line)][:limit]


def _grade(call: Mapping[str, Any], mode: str, surface: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """``call`` against ``mode``'s documented result (row 7's grading, and its
    no-run line for ``status``)."""
    probe = _probe()
    codes = surface["dispatcher"]["exit_codes"]
    states = surface["dispatcher"]["states"]
    if mode == "init":
        graded, problems = probe.grade_call(call, mode="init", want_exit=codes["ok"],
                                            want_state=states["ok"], surface=surface)
    else:
        graded, problems = probe.grade_call(call, mode="status", want_exit=codes["nothing_to_do"],
                                            want_state=states["nothing_to_do"], surface=surface)
        outcome = call.get("outcome") or {}
        if call.get("outcome") is not None and probe.NO_RUN_LINE not in str(outcome.get("output") or ""):
            problems.append(f"status did not report {probe.NO_RUN_LINE!r} for a plan with no run")
            graded["ok"] = False
    argv = (call.get("outcome") or {}).get("argv")
    graded["uv"] = argv[0] if isinstance(argv, list) and argv else None
    bounded = call.get("bounded")
    if bounded is not None and hasattr(bounded, "argv"):
        graded["run"] = _bounded_view(bounded)
        graded["network_attempts"] = network_attempts(_read(bounded.stderr, 400000))
    return graded, problems


def _locate(ctx: Context, steps: Steps, iso: Any, cwd: Path, diag: Path, label: str,
            version: Optional[str] = None) -> Dict[str, Any]:
    """Where the binary says its directories are, and the one installed package there."""
    paths = steps.debug_paths(ctx.binary, iso.env, cwd, diag, f"{label}.debug-paths")
    roots = [Path(p) for key, p in sorted(paths["paths"].items())
             if key in ("cache", "data", "config", "state")]
    found = find_installed(roots, ctx.name, version or ctx.version)
    return {"debug_paths": paths["paths"], "searched": [str(r) for r in roots],
            "found": [str(f) for f in found]}


def run_criteria(ctx: Context, steps: Optional[Steps] = None) -> Dict[str, Any]:
    """Every criterion this environment can attempt but the offline one, in order."""
    steps = steps or Steps()
    probe = _probe()
    run = _Run()
    root = Path(ctx.probe_root)
    diag = root / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    scratch = root / "probe-npm"
    scratch.mkdir(parents=True, exist_ok=True)
    userconfig = scratch / "npmrc-empty"
    userconfig.write_text("", encoding="utf-8")
    kind = "later" if ctx.previous_pin else "first"
    names = list(criteria_for(ctx.environment, kind))
    names = [n for n in names if n != OFFLINE]
    state: Dict[str, Any] = {}
    details: Dict[str, Any] = {"probe_root": str(root), "diagnostics": str(diag)}
    result = {"criteria": run.criteria, "failures": run.failures, "state": state,
              "details": details, "measured": {}, "variables_set": {}}

    # -- the isolated environment ------------------------------------------
    iso = probe.isolated_env(root / "env", binary=ctx.binary, registry=ctx.read_registry + "/")
    if os.path.realpath(str(iso.home)) == os.path.realpath(probe.real_home()):
        raise CheckError("the isolated HOME resolved to the invoking user's home")
    probe.materialize(iso)
    empty = {name: _entries(iso.path(name)) for name in
             ("home", "config", "data", "cache", "state", "npm_cache", "uv_cache")}
    details["empty_at_start"] = empty
    details["env_cleared"] = list(iso.cleared)
    result["variables_set"] = {k: v for k, v in sorted(iso.env.items())
                               if k not in probe.PASSTHROUGH}
    if any(empty.values()):
        raise CheckError(f"the probe's home and caches are not empty at the start: {empty}")
    project = root / "project"
    probe.make_project(project, "no-commit", iso.env)
    write_project_config(project, ctx.spec, NO_MODEL_URL)
    state.update(iso_root=str(iso.root), project=str(project), env=dict(iso.env))

    answered = _version_of(steps, [ctx.binary, "--version"], iso.env, project, diag,
                           "opencode-version")
    result["measured"]["opencode"] = answered or "<unread>"
    for tool in ("node", "npm"):
        result["measured"][tool] = _version_of(steps, [tool, "--version"], iso.env, project,
                                               diag, f"{tool}-version") or "<unread>"
    if answered != ctx.opencode_pin:
        raise CheckError(f"{ctx.binary} answers --version {answered!r}, not the pin "
                         f"{ctx.opencode_pin}; refusing before the probe runs")

    # -- 1. npm_spec_resolves ----------------------------------------------
    attempts = []
    view: Optional[Dict[str, Any]] = None
    deadline = steps.clock() + ctx.poll_s
    while True:
        index = len(attempts) + 1
        cache = scratch / f"view-cache-{index}"
        argv = _npm_argv("view", ctx.spec, "version", "dist", "--json", "--prefer-online",
                         registry=ctx.read_registry, cache=cache, userconfig=userconfig,
                         prefix=scratch)
        bounded = steps.run(argv, cwd=scratch, env=iso.env, budget_s=NPM_BUDGET_S,
                            stdout=diag / f"npm-view-{index}.json",
                            stderr=diag / f"npm-view-{index}.stderr.txt")
        stdout, stderr = _read(bounded.stdout, 20000), _read(bounded.stderr)
        attempt = {"at": _utc_now(), "exit": bounded.returncode, "killed": bounded.killed,
                   "code": npm_error_code(stdout, stderr) if bounded.returncode != 0 else None}
        attempts.append(attempt)
        if bounded.returncode == 0:
            try:
                view = json.loads(stdout)
            except ValueError:
                view = None
            if isinstance(view, dict) and view.get("version") == ctx.version:
                break
        if steps.clock() >= deadline:
            break
        steps.sleep(ctx.interval_s)
    npm_evidence: Dict[str, Any] = {
        "registry": ctx.read_registry,
        "argv": [a if not a.startswith("--cache=") else "--cache=<fresh per attempt>" for a in argv],
        "attempts": attempts, "poll_s": ctx.poll_s, "served": view}
    problems = []
    dist = (view or {}).get("dist") if isinstance(view, dict) else None
    if not isinstance(view, dict) or view.get("version") != ctx.version or not isinstance(dist, dict):
        problems.append(f"npm view never showed {ctx.spec} within {ctx.poll_s:.0f} s "
                        f"(last: exit {attempts[-1]['exit']}, code {attempts[-1]['code']}); "
                        "if this is registry lag, re-run once a clean-cache read serves it")
        run.decide(NPM, problems, npm_evidence)
        run.skip(names, f"{ctx.spec} never resolved from {ctx.read_registry}")
        return result
    served_integrity = dist.get("integrity")
    tarball_url = dist.get("tarball")
    if not re.fullmatch(r"sha512-[A-Za-z0-9+/]{86}==", str(served_integrity or "")):
        problems.append(f"dist.integrity {served_integrity!r} is not a sha512 integrity")
    if not str(tarball_url or "").startswith(ctx.read_registry + "/"):
        problems.append(f"dist.tarball {tarball_url!r} is not on {ctx.read_registry}")
    # The download: npm's own client, with its own fresh cache.
    download_dir = root / "registry-artifact"
    download_dir.mkdir(parents=True, exist_ok=True)
    argv = _npm_argv("pack", ctx.spec, "--json", f"--pack-destination={download_dir}",
                     registry=ctx.read_registry, cache=scratch / "pack-cache", userconfig=userconfig,
                     prefix=scratch)
    bounded = steps.run(argv, cwd=scratch, env=iso.env, budget_s=NPM_BUDGET_S,
                        stdout=diag / "npm-pack.json", stderr=diag / "npm-pack.stderr.txt")
    tarballs = sorted(download_dir.glob("*.tgz"))
    download = {"argv": [a if not a.startswith("--cache=") else "--cache=<fresh>" for a in argv],
                "exit": bounded.returncode,
                "code": npm_error_code(_read(bounded.stdout), _read(bounded.stderr))
                if bounded.returncode != 0 else None,
                "files": [t.name for t in tarballs]}
    npm_evidence.update(tarball_url=tarball_url, integrity_served=served_integrity, download=download)
    extracted: Optional[Path] = None
    if bounded.returncode != 0 or len(tarballs) != 1:
        problems.append(f"npm pack {ctx.spec} did not download one tarball "
                        f"(exit {bounded.returncode}, code {download['code']}, {len(tarballs)} files)")
    else:
        data = tarballs[0].read_bytes()
        stub = _load("opencode_registry_stub", PACKAGE_DIR / "registry_stub.py")
        downloaded = stub.integrity_of(data)
        download.update(bytes=len(data), integrity=downloaded)
        if downloaded != served_integrity:
            problems.append(f"the downloaded tarball hashes to {downloaded}, not the served "
                            f"{served_integrity}")
        try:
            extracted = probe.extract_package(tarballs[0], root / "registry-files")
        except probe.ProbeError as exc:
            problems.append(str(exc))
    state["integrity"] = served_integrity
    if not run.decide(NPM, problems, npm_evidence) or extracted is None:
        run.skip(names, "the registry's tarball could not be downloaded and verified")
        return result

    # -- OpenCode's first start: the install, before any tool call -----------
    # Walked before OpenCode has run at all: nothing of the spec may be there yet.
    preinstalled = [str(p) for p in find_installed(
        [iso.path(n) for n in ("home", "config", "data", "cache", "state")], ctx.name, ctx.version)]
    debug = steps.debug_config(ctx.binary, iso.env, project, diag, "debug-config-first")
    located_after = _locate(ctx, steps, iso, project, diag, "after-first-start")
    found = [Path(p) for p in located_after["found"]]
    location = found[0] if len(found) == 1 else None
    details["opencode_paths"] = located_after["debug_paths"]
    details["installed_before_first_start"] = preinstalled

    # -- 2. installed_tree_matches_candidate (and the location) --------------
    problems = []
    registry_files = compare_tree(extracted, ctx.candidate)
    tree_evidence: Dict[str, Any] = {
        "candidate": {"sha": ctx.candidate.get("sha"), "digest": ctx.candidate["digest"],
                      "files": len(ctx.candidate["inventory"])},
        "registry_files": registry_files, "located": located_after,
        "compared": "before the first tool call"}
    if preinstalled:
        problems.append(f"{ctx.spec} was installed before OpenCode's first start: {preinstalled}")
    if not registry_files["matches"]:
        problems.append("the registry's files are not the candidate "
                        f"(missing {registry_files['missing_count']}, unexpected "
                        f"{registry_files['unexpected_count']}, digest {registry_files['digest']})")
    if debug.error:
        problems.append(debug.error)
    if location is None:
        problems.append(f"OpenCode's first start left {len(found)} installed {ctx.spec} "
                        f"under {located_after['searched']}, not one")
    else:
        installed = compare_tree(location, ctx.candidate)
        metadata = installer_metadata(install_root_of(location), location)
        origin = installed_origin(install_root_of(location), ctx.name)
        origins = probe.installed_from([iso.opencode_config_dir],
                                       Path(located_after["debug_paths"].get("cache", iso.opencode_cache_dir)),
                                       ctx.read_registry)
        tree_evidence.update(installed=installed, installer_metadata=metadata, origin=origin,
                             installed_from=origins)
        if not _is_under(location, root):
            problems.append(f"the payload resolved to {location}, outside the probe's root {root}")
        if not installed["matches"]:
            problems.append("OpenCode's installed payload is not the candidate "
                            f"(missing {installed['missing_count']}, unexpected "
                            f"{installed['unexpected_count']}, digest {installed['digest']})")
        if metadata["unexpected"]:
            problems.append(f"the install root holds files that are neither the package nor "
                            f"identified installer metadata: {metadata['unexpected'][:5]}")
        if not str(origin.get("resolved") or "").startswith(ctx.read_registry + "/"):
            problems.append(f"the installed package came from {origin.get('resolved')!r}, "
                            f"not {ctx.read_registry}")
        if origin.get("integrity") != served_integrity:
            problems.append(f"the installer recorded integrity {origin.get('integrity')!r}, not the "
                            f"served {served_integrity}")
        if origins["foreign"]:
            problems.append(f"OpenCode installed packages from outside {ctx.read_registry}: "
                            f"{origins['foreign'][:5]}")
        state.update(location=str(location), skill_dir=probe.skill_dir_of(location),
                     installed_registry=origin.get("origin"))
        details["payload_location"] = str(location)
        details["installed_from"] = origins
    tree_ok = run.decide(TREE, problems, tree_evidence)

    # -- 3. manifest_version_matches ----------------------------------------
    if location is not None:
        problems = []
        manifest = json.loads((location / "package.json").read_text(encoding="utf-8"))
        origin = tree_evidence.get("origin") or {}
        if (manifest.get("name"), manifest.get("version")) != (ctx.name, ctx.version):
            problems.append(f"the installed manifest is {manifest.get('name')}@"
                            f"{manifest.get('version')}, not {ctx.spec}")
        if origin.get("version") != ctx.version:
            problems.append(f"the installer's lockfile records {origin.get('version')!r}")
        run.decide(MANIFEST, problems, {"manifest": {"name": manifest.get("name"),
                                                      "version": manifest.get("version")},
                                         "lockfile_version": origin.get("version"),
                                         "location": str(location)})

    # -- 4. command_resolves_installed_skill --------------------------------
    surface: Optional[Mapping[str, Any]] = None
    if location is not None:
        surface = probe.payload_surface(location)
        command = read_command(debug.config, location)
        problems = list(_command_problems(command, ctx.spec))
        run.decide(COMMAND, problems, {"command": command, "location": str(location),
                                        "skill_dir": probe.skill_dir_of(location),
                                        "debug_config": _bounded_view(debug.bounded)
                                        if debug.bounded is not None else None})
    if location is None or not tree_ok:
        run.skip(names, "the installed payload is not the candidate; no tool call runs on it")
        return result

    # -- 5. tool_init_runs_installed_payload, 6. cold_dependencies_install ---
    skill_dir = Path(probe.skill_dir_of(location))
    tool = surface["tool_name"]
    cold_before = {"uv_cache_entries": _entries(iso.uv_cache),
                   "venv_present": (skill_dir / ".venv").exists()}
    before = probe._files(project)
    call = steps.tool_call(ctx.binary, iso.env, project, ctx.spec, tool, probe.INIT_ARGS,
                           "init", "call_mllm_registry_init", diag, "init")
    created = sorted(probe._files(project) - before)
    graded, problems = _grade(call, "init", surface)
    graded["files_created"] = created
    if probe.PROVIDERS_FILE not in created:
        problems.append(f"init created no {probe.PROVIDERS_FILE} in the project")
    problems += _ran_from(graded, skill_dir)
    run.decide(INIT, problems, {"init": graded, "installed_skill_dir": str(skill_dir)})
    if graded.get("uv"):
        result["measured"]["uv"] = _version_of(steps, [graded["uv"], "--version"], iso.env,
                                               project, diag, "uv-version") or "<unread>"

    venv_python = skill_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    cold_after: Dict[str, Any] = {"uv_cache_entries": _entries(iso.uv_cache),
                                  "venv_present": (skill_dir / ".venv").exists()}
    problems = []
    required = runtime_requirements(skill_dir)
    if venv_python.exists():
        bounded = steps.run([str(venv_python), "-I", "-c", _LIST_DISTS], cwd=project, env=iso.env,
                            budget_s=VERSION_BUDGET_S, stdout=diag / "venv-dists.json",
                            stderr=diag / "venv-dists.stderr.txt")
        try:
            listed = json.loads(_read(bounded.stdout, 200000))
        except ValueError:
            listed = {}
        cold_after.update(python=listed.get("python"), dists=listed.get("dists"))
        result["measured"]["python"] = listed.get("python") or "<unread>"
        missing = [r for r in required if r not in (listed.get("dists") or [])]
        if missing:
            problems.append(f"the venv lacks the runtime dependencies {missing}")
    else:
        problems.append(f"the first call left no venv at {skill_dir / '.venv'}")
    if cold_before["uv_cache_entries"] or cold_before["venv_present"]:
        problems.append(f"the dependency install was not cold: {cold_before}")
    if not cold_after["uv_cache_entries"]:
        problems.append("uv's cache is still empty after the first call: nothing was installed")
    run.decide(COLD, problems, {"before_first_call": cold_before, "after_first_call": cold_after,
                                "runtime_requirements": required, "uv_cache": str(iso.uv_cache)})

    # -- 7. same_pin_restart_matches ----------------------------------------
    problems = []
    baseline_call = steps.tool_call(ctx.binary, iso.env, project, ctx.spec, tool, probe.STATUS_ARGS,
                                    "status", "call_mllm_registry_status_before", diag,
                                    "status-before-restart")
    baseline, base_problems = _grade(baseline_call, "status", surface)
    before_restart = payload_state(location, ctx.candidate)
    restart_debug = steps.debug_config(ctx.binary, iso.env, project, diag, "debug-config-restart")
    relocated = _locate(ctx, steps, iso, project, diag, "after-restart")
    restart_command = read_command(restart_debug.config, location)
    after_restart = payload_state(Path(relocated["found"][0]), ctx.candidate) \
        if len(relocated["found"]) == 1 else None
    restart_call = steps.tool_call(ctx.binary, iso.env, project, ctx.spec, tool, probe.STATUS_ARGS,
                                   "status", "call_mllm_registry_status_after", diag,
                                   "status-after-restart")
    restarted, restart_problems = _grade(restart_call, "status", surface)
    manifest_after = None
    if after_restart is not None:
        manifest_after = json.loads((Path(after_restart["location"]) / "package.json")
                                    .read_text(encoding="utf-8")).get("version")
    problems += [f"before the restart: status: {p}" for p in base_problems]
    problems += [f"before the restart: {p}" for p in _ran_from(baseline, skill_dir)]
    problems += [f"after the restart: status: {p}" for p in restart_problems]
    problems += [f"after the restart: {p}" for p in _command_problems(restart_command, ctx.spec)]
    problems += [f"after the restart: {p}" for p in _ran_from(restarted, skill_dir)]
    if restart_debug.error:
        problems.append(restart_debug.error)
    if after_restart is None:
        problems.append(f"after the restart {len(relocated['found'])} installed {ctx.spec}, not one")
    else:
        if after_restart["location"] != before_restart["location"]:
            problems.append(f"the payload moved: {before_restart['location']} -> "
                            f"{after_restart['location']}")
        if not (after_restart["digest"] == before_restart["digest"] == ctx.candidate["digest"]):
            problems.append(f"the payload's digest changed across the restart "
                            f"({before_restart['digest']} -> {after_restart['digest']}; "
                            f"candidate {ctx.candidate['digest']})")
        if manifest_after != ctx.version:
            problems.append(f"after the restart the installed version is {manifest_after!r}")
    behaviour = lambda g: ((g.get("result") or {}).get("mode"), g.get("state"), g.get("exit"),  # noqa: E731
                           g.get("skill_dir"))
    if behaviour(baseline) != behaviour(restarted):
        problems.append(f"status behaved differently after the restart: {behaviour(baseline)} -> "
                        f"{behaviour(restarted)}")
    # Read again now: OpenCode's config-directory install runs in the background,
    # so its lockfile may only exist by the end of the run.
    origins = probe.installed_from([iso.opencode_config_dir],
                                   Path(relocated["debug_paths"].get("cache", iso.opencode_cache_dir)),
                                   ctx.read_registry)
    details["installed_from"] = origins
    if origins["foreign"]:
        problems.append(f"by the end of the run OpenCode had installed packages from outside "
                        f"{ctx.read_registry}: {origins['foreign'][:5]}")
    run.decide(RESTART, problems, {"before": {"payload": before_restart, "status": baseline},
                                   "after": {"payload": after_restart, "version": manifest_after,
                                             "command": restart_command, "status": restarted},
                                   "installed_from_at_end": origins})
    state["payload"] = after_restart or before_restart

    # -- 9. upgrade_runs_new_payload (a later release) ----------------------
    if ctx.previous_pin:
        run.criteria[UPGRADE] = run_upgrade(ctx, steps, run)
    return result


def read_command(config: Optional[Mapping[str, Any]], location: Path) -> Dict[str, Any]:
    """``probe.check_command`` against the installed payload, minus its
    directory-route verdict (``plugin_is_payload``), which an npm pin can never
    satisfy and :func:`_command_problems` replaces with the exact-spec check."""
    command = _probe().check_command(config, location, token=SKILL_DIR_TOKEN)
    command.pop("plugin_is_payload", None)
    return command


def _command_problems(command: Mapping[str, Any], spec: str) -> List[str]:
    """Row 7's command reading, for an npm pin: ``plugin[]`` is the spec, not a path."""
    problems = []
    if not command["registered"]:
        problems.append("debug config has no command['multi-llm'] template")
    elif not command["matches_payload"]:
        problems.append("command['multi-llm'] is not the installed payload's own body")
    if command["token_left"]:
        problems.append("command['multi-llm'] still carries an unresolved {SKILL_DIR}")
    if not command["skill_paths_inside"]:
        problems.append(f"the template names skill paths outside the payload: {command['skill_paths']}")
    if command["plugin_specs"] != [spec]:
        problems.append(f"plugin[] is {command['plugin_specs']}, not exactly [{spec!r}]")
    return problems


def _ran_from(graded: Mapping[str, Any], skill_dir: Path) -> List[str]:
    """The call ran the installed payload: uv's ``--project`` and the script."""
    return _probe().invocation_problems(graded, skill_dir)


def run_upgrade(ctx: Context, steps: Steps, run: _Run) -> Dict[str, Any]:
    """Step 5's upgrade: the previous pin installed and used, then the new pin.

    Its own isolated home and project, in the same fresh environment. The new
    payload is compared with the candidate before its first tool call, like the
    first install; "the new payload is executing" is read off the call's own
    uv argv.
    """
    probe = _probe()
    root = Path(ctx.probe_root) / "upgrade"
    diag = Path(ctx.probe_root) / "diagnostics" / "upgrade"
    diag.mkdir(parents=True, exist_ok=True)
    previous_name, previous_version = parse_spec(ctx.previous_pin)
    iso = probe.isolated_env(root / "env", binary=ctx.binary, registry=ctx.read_registry + "/")
    probe.materialize(iso)
    project = root / "project"
    probe.make_project(project, "no-commit", iso.env)
    problems: List[str] = []
    evidence: Dict[str, Any] = {"previous_pin": ctx.previous_pin, "new_pin": ctx.spec}

    write_project_config(project, ctx.previous_pin, NO_MODEL_URL)
    first = steps.debug_config(ctx.binary, iso.env, project, diag, "debug-config-previous")
    prev_ctx = replace(ctx, name=previous_name, version=previous_version)
    located = _locate(prev_ctx, steps, iso, project, diag, "previous")
    if first.error or len(located["found"]) != 1:
        problems.append(f"the previous pin did not install once ({first.error or located['found']})")
        return _upgrade_verdict(run, problems, evidence)
    previous = Path(located["found"][0])
    evidence["before"] = {"version": json.loads((previous / "package.json").read_text(
        encoding="utf-8")).get("version"), "location": str(previous)}
    surface = probe.payload_surface(previous)
    call = steps.tool_call(ctx.binary, iso.env, project, ctx.previous_pin, surface["tool_name"],
                           probe.STATUS_ARGS, "status", "call_mllm_registry_upgrade_previous",
                           diag, "status-previous")
    graded, call_problems = _grade(call, "status", surface)
    evidence["before"]["status"] = graded
    problems += [f"previous pin: status: {p}" for p in call_problems]
    problems += [f"previous pin: {p}" for p in _ran_from(graded, Path(probe.skill_dir_of(previous)))]

    write_project_config(project, ctx.spec, NO_MODEL_URL)
    second = steps.debug_config(ctx.binary, iso.env, project, diag, "debug-config-upgraded")
    located = _locate(ctx, steps, iso, project, diag, "upgraded")
    if second.error or len(located["found"]) != 1:
        problems.append(f"the new pin did not install once after the change "
                        f"({second.error or located['found']})")
        return _upgrade_verdict(run, problems, evidence)
    new = Path(located["found"][0])
    tree = compare_tree(new, ctx.candidate)
    evidence["after"] = {"version": json.loads((new / "package.json").read_text(
        encoding="utf-8")).get("version"), "location": str(new), "digest": tree["digest"],
        "tree": tree}
    if not tree["matches"]:
        problems.append("the upgraded payload is not the candidate; no tool call runs on it")
        return _upgrade_verdict(run, problems, evidence)
    if evidence["after"]["version"] != ctx.version:
        problems.append(f"after the change the installed version is {evidence['after']['version']!r}")
    surface = probe.payload_surface(new)
    call = steps.tool_call(ctx.binary, iso.env, project, ctx.spec, surface["tool_name"],
                           probe.STATUS_ARGS, "status", "call_mllm_registry_upgrade_new",
                           diag, "status-upgraded")
    graded, call_problems = _grade(call, "status", surface)
    evidence["after"]["status"] = graded
    evidence["ran_from"] = graded.get("skill_dir")
    problems += [f"new pin: status: {p}" for p in call_problems]
    problems += [f"new pin: {p}" for p in _ran_from(graded, Path(probe.skill_dir_of(new)))]
    return _upgrade_verdict(run, problems, evidence)


def _upgrade_verdict(run: _Run, problems: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    run.decide(UPGRADE, problems, evidence)
    return run.criteria[UPGRADE]


# ---------------------------------------------------------------------------
# The warm offline operation
# ---------------------------------------------------------------------------

def offline_env(env: Mapping[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """The installed environment as the offline run gets it, and what it leaves unset.

    Step 5's rule: ``OPENCODE_DISABLE_MODELS_FETCH=1`` (an unrelated startup
    fetch) and nothing else that could make the run pass. ``UV_OFFLINE`` is
    never set: once warm, the payload's own ``uv run --frozen --no-dev`` must
    work offline. ``OPENCODE_DISABLE_AUTOUPDATE`` is unset here, unlike in the
    networked phases (where a self-update would change the binary under
    test): it was measured on 1.18.18 not to stop the update check anyway.
    """
    unset = ["OPENCODE_DISABLE_AUTOUPDATE", "UV_OFFLINE"]
    out = {k: v for k, v in env.items() if k not in unset}
    out["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
    return out, unset


def offline_operation(ctx: Context, state: Mapping[str, Any], steps: Optional[Steps] = None) -> Dict[str, Any]:
    """``status`` through the installed payload, with the proof taken around it.

    Run where the network is already cut (the committed image with
    ``--network none``, or a VM with its interface detached). Returns the
    proof's two sides, the payload's state before and after, the graded call
    and every variable the run had; ``network_mode`` is the caller's to add.
    """
    steps = steps or Steps()
    probe = _probe()
    diag = Path(ctx.probe_root) / "diagnostics" / "offline"
    diag.mkdir(parents=True, exist_ok=True)
    scratch = Path(ctx.probe_root) / "probe-npm" / "offline"
    scratch.mkdir(parents=True, exist_ok=True)
    env, unset = offline_env(state["env"])
    project = Path(state["project"])
    location = Path(state["location"])
    skill_dir = Path(probe.skill_dir_of(location))

    before = network_snapshot(ctx, env, steps, diag, "before", scratch)
    payload_before = payload_state(location, ctx.candidate)
    surface = probe.payload_surface(location)
    call = steps.tool_call(ctx.binary, env, project, ctx.spec, surface["tool_name"],
                           probe.STATUS_ARGS, "status", "call_mllm_registry_offline", diag,
                           "status-offline")
    graded, problems = _grade(call, "status", surface)
    problems += _ran_from(graded, skill_dir)
    payload_after = payload_state(location, ctx.candidate)
    after = network_snapshot(ctx, env, steps, diag, "after", scratch)
    # Recorded, never passed over: a fetch the run attempted and did without
    # (the config directory's dependency install, an update check) is part of
    # what "offline" meant for this binary. A fetch the run NEEDED shows up as
    # the status call failing instead.
    return {"before": before, "after": after, "payload": {"before": payload_before,
                                                          "after": payload_after},
            "status": graded, "problems": problems,
            "variables_set": {k: v for k, v in sorted(env.items()) if k not in probe.PASSTHROUGH},
            "left_unset": unset, "network_attempts": graded.get("network_attempts") or []}


def offline_criterion(offline: Mapping[str, Any], mode_before: Optional[str],
                      mode_after: Optional[str], single: str, how: str) -> Dict[str, Any]:
    """The ``warm_offline_operation`` entry: the operation and its network proof."""
    proof = {
        "network_mode": mode_before if mode_before == mode_after else
        f"{mode_before} -> {mode_after}",
        "before": _proof_side(offline["before"]),
        "after": _proof_side(offline["after"]),
        "payload": {side: {"location": offline["payload"][side]["location"],
                           "digest": offline["payload"][side]["digest"]}
                    for side in ("before", "after")},
    }
    problems = [f"status: {p}" for p in offline["problems"]]
    if mode_before != mode_after:
        problems.append(f"NetworkMode changed during the run: {mode_before} -> {mode_after}")
    problems += assess_network_proof(proof, single)
    for side in ("before", "after"):
        addresses = offline[side]["_detail"]["addresses"]
        foreign = [a for a in addresses if not (a.startswith("127.") or
                                                a == "0000:0000:0000:0000:0000:0000:0000:0001")]
        if foreign:
            problems.append(f"{side}: the namespace owns non-loopback addresses {foreign}")
    evidence = {
        "how": how, "status": offline["status"],
        "variables_set": offline["variables_set"], "left_unset": offline["left_unset"],
        "interfaces": {side: offline[side]["_detail"]["interfaces"] for side in ("before", "after")},
        "addresses": {side: offline[side]["_detail"]["addresses"] for side in ("before", "after")},
        "npm_view_output": {side: offline[side]["_detail"]["npm_view_output"]
                            for side in ("before", "after")},
        "network_attempts_without_effect": offline["network_attempts"],
    }
    if problems:
        evidence["problems"] = problems
    return {"pass": not problems, "evidence": evidence, "network_proof": proof}


def wait_for_cut(steps: Steps, budget_s: float = VM_CUT_WAIT_S) -> Dict[str, Any]:
    """A VM: wait, bounded, until only loopback interfaces are up."""
    deadline = steps.clock() + budget_s
    print(f"registry check (vm): detach this VM's network interface now; waiting up to "
          f"{budget_s:.0f} s for only loopback to be up", file=sys.stderr, flush=True)
    while True:
        up = interfaces_snapshot(steps.sys_class_net)["up"]
        if up and all(_LOOPBACK.match(name) for name in up):
            return {"cut": True, "up": up}
        if steps.clock() >= deadline:
            return {"cut": False, "up": up}
        steps.sleep(5.0)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def build_record(*, environment: str, rehearsal: bool, workflow_run_id: Optional[int],
                 spec: str, name: str, version: str, read_registry: str, installed_registry: Optional[str],
                 integrity: Optional[str], tag: str, sha: str, candidate_digest: str, kind: str,
                 previous_pin: Optional[str], measured: Mapping[str, str],
                 variables_set: Mapping[str, str], criteria: Mapping[str, Mapping[str, Any]],
                 payload_location: Optional[str], environment_details: Mapping[str, Any],
                 notes: str = "") -> Dict[str, Any]:
    """The record, with exactly the criteria this environment and kind carry, in order.

    A criterion the run never reached is present, failing, with the reason as
    its evidence: a record is never shorter than its profile, so a partial run
    cannot read as a different, smaller profile.
    """
    entries = []
    for criterion in criteria_for(environment, kind):
        decided = criteria.get(criterion) or {"pass": False,
                                              "evidence": {"not_attempted": "the run ended before it"}}
        entry = {"name": criterion, "pass": bool(decided.get("pass") is True),
                 "evidence": decided.get("evidence") or {"not_attempted": "no evidence"}}
        if criterion == OFFLINE and decided.get("network_proof") is not None:
            entry["network_proof"] = decided["network_proof"]
        entries.append(entry)
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "recorded_at": _utc_now(),
        "environment": environment, "workflow_run_id": workflow_run_id,
        "rehearsal": bool(rehearsal), "install_source": "npm",
        "spec": spec, "name": name, "version": version,
        "registry": read_registry, "installed_registry": installed_registry or "<not installed>",
        "integrity": integrity or "<not served>", "tag": tag, "sha": sha,
        "candidate_digest": candidate_digest, "release_kind": kind, "previous_pin": previous_pin,
        "measured": {key: str(value) for key, value in measured.items()},
        "variables_set": {key: str(value) for key, value in variables_set.items()},
        "environment_details": dict(environment_details),
        "criteria": entries,
    }
    if payload_location:
        record["payload_location"] = payload_location
    for key in ("opencode", "python", "uv"):
        record["measured"].setdefault(key, "<unread>")
    if notes:
        record["notes"] = notes
    return record


def validate_record(record: Mapping[str, Any], *, environment: str, tag: str, repo: Path,
                    digest: Optional[str]) -> List[str]:
    """``registry_record``'s verdict: the profile, or for a rehearsal its structure only."""
    rr = _record_module()
    return rr.validate_record(dict(record), profile=profile_for(environment), tag=tag,
                              repo=Path(repo), digest=digest,
                              rehearsal_structure_only=bool(record.get("rehearsal")))


# ---------------------------------------------------------------------------
# The container driver (host side)
# ---------------------------------------------------------------------------

def container_base(lock: Path = CLI_VERSIONS_LOCK) -> Dict[str, str]:
    """The stock base image and the pinned uv, read from ``cli-versions.lock``."""
    import yaml

    data = yaml.safe_load(Path(lock).read_text(encoding="utf-8")) or {}
    base = data.get("base_image") or {}
    uv = (data.get("artifacts") or {}).get("uv") or {}
    if not base.get("ref") or not base.get("digest") or not uv.get("url") or not uv.get("sha256"):
        raise CheckError(f"{lock} names no base image digest or uv artifact")
    return {"image": f"{base['ref']}@{base['digest']}", "platform": str(data.get("platform") or
                                                                       "linux/amd64"),
            "uv_version": str(uv.get("version")), "uv_url": str(uv["url"]),
            "uv_sha256": str(uv["sha256"])}


def setup_script(base: Mapping[str, str]) -> str:
    """Root's setup in the stock container: what a user's machine has, pinned where we can."""
    return "\n".join([
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -qq",
        "apt-get install -y -qq --no-install-recommends ca-certificates curl git python3 "
        "python3-yaml >/dev/null",
        "curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "
        f"-o /tmp/uv.tar.gz '{base['uv_url']}'",
        f"echo '{base['uv_sha256']}  /tmp/uv.tar.gz' | sha256sum --check --strict -",
        "tar -xzf /tmp/uv.tar.gz -C /tmp",
        "install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv",
        "install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx",
        "rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu",
        f"groupadd --gid {CONTAINER_UID} {CONTAINER_USER}",
        f"useradd --uid {CONTAINER_UID} --gid {CONTAINER_UID} --create-home "
        f"--home-dir /home/{CONTAINER_USER} --shell /bin/bash {CONTAINER_USER}",
        f"mkdir -p {WORK}/out {PIN_PREFIX} {KIT_ROOT}",
        f"chown -R {CONTAINER_UID}:{CONTAINER_UID} {WORK}",
        "uv --version; python3 --version; git --version; node --version; npm --version",
    ])


def pin_install_script(single: str) -> str:
    """``npm ci --ignore-scripts`` of the lockfile subset, from the single
    registry, with empty npm config and the prefix as npm's project root
    (``bump._npm_ci``'s flags, for the same reasons: no install script runs a
    binary before ``verify_installed`` chose it)."""
    return "\n".join([
        "set -euo pipefail",
        f"cd {PIN_PREFIX}",
        ": > .npmrc-empty",
        f"npm ci --ignore-scripts --no-audit --no-fund --registry={single} "
        f"--userconfig={PIN_PREFIX}/.npmrc-empty --prefix={PIN_PREFIX} "
        f"--cache={PIN_PREFIX}/.npm-cache",
        f"rm -rf {PIN_PREFIX}/.npm-cache",
        f"chmod -R a+rX {PIN_PREFIX}",
    ])


def stage_kit(destination: Path) -> Dict[str, str]:
    """Copy :data:`KIT_FILES` under ``destination``; their sha256, for the record."""
    import hashlib

    manifest = {}
    for rel in KIT_FILES:
        source = REPO_ROOT / rel
        target = Path(destination) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    return manifest


def docker_transport(operating_system: str) -> Dict[str, Any]:
    """How a networked container reaches a stub on the host's loopback.

    Docker Desktop runs containers in a VM: its ``--network host`` is the VM's
    network, where the host's ``127.0.0.1`` refuses (measured), while
    ``host.docker.internal`` via ``host-gateway`` is forwarded to the host's
    loopback, so a relay inside the container keeps the URL loopback. Docker
    Engine shares the host's network namespace with ``--network host``.
    """
    if "docker desktop" in (operating_system or "").lower():
        return {"kind": "host-gateway", "run_args": ["--add-host", "host.docker.internal:host-gateway"],
                "relay": True,
                "why": "Docker Desktop: --network host is the VM's network and the host's loopback "
                       "refuses there; host.docker.internal is forwarded to it"}
    return {"kind": "host", "run_args": ["--network", "host"], "relay": False,
            "why": "Docker Engine: --network host shares the host's loopback"}


class Docker:
    """The docker CLI, every call bounded, every call's streams kept."""

    def __init__(self, diag: Path, binary: str = "docker"):
        self.diag = Path(diag)
        self.binary = binary
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, args: Sequence[str], *, label: str, budget_s: float = DOCKER_SHORT_S,
                 check: bool = True) -> Tuple[Any, str]:
        index = len(self.calls) + 1
        bounded = _probe().run_bounded(
            [self.binary, *map(str, args)], cwd=self.diag, env=dict(os.environ), budget_s=budget_s,
            stdout=self.diag / f"{index:02d}-{label}.out.txt",
            stderr=self.diag / f"{index:02d}-{label}.err.txt")
        self.calls.append({"label": label, **_bounded_view(bounded)})
        stdout = _read(bounded.stdout, 200000)
        if check and bounded.returncode != 0:
            raise CheckError(f"docker {label} {bounded.describe()}: "
                             f"{_read(bounded.stderr, 600).strip() or stdout[-600:].strip()}")
        return bounded, stdout


def drive_container(*, ctx: Context, base: Mapping[str, str], pin_files: Tuple[Dict, Dict],
                    support: Path, docker: Optional[Docker] = None) -> Dict[str, Any]:
    """Step 5 in a stock container, then its warm-offline run with ``--network none``.

    Returns ``{"phase": <install-phase result>, "offline": <criterion or None>,
    "details": {...}}``. The containers and the committed image are removed in
    every case; ``details["image_deleted"]`` says whether that was confirmed.
    """
    support = Path(support)
    docker = docker or Docker(support / "docker")
    docker.diag.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    networked = f"mllm-regcheck-{run_id}"
    offline_name = f"mllm-regcheck-{run_id}-offline"
    image_tag = f"mllm-registry-check-offline:{run_id}"
    details: Dict[str, Any] = {"base_image": base["image"], "platform": base["platform"],
                               "uv_artifact": {"version": base["uv_version"], "url": base["uv_url"],
                                               "sha256": base["uv_sha256"]},
                               "run_id": run_id, "source_tree_mounted": False}
    _, info = docker(["info", "--format", "{{json .}}"], label="info")
    try:
        info_data = json.loads(info)
    except ValueError:
        info_data = {}
    details["docker"] = {"server": info_data.get("ServerVersion"),
                         "operating_system": info_data.get("OperatingSystem"),
                         "kernel": info_data.get("KernelVersion")}
    transport = docker_transport(str(info_data.get("OperatingSystem") or "")) if ctx.rehearsal \
        else {"kind": "default-bridge", "run_args": [], "relay": False}
    details["networked_phase"] = {"transport": transport["kind"],
                                  "why": transport.get("why", "the networked phase needs only the "
                                                       "internet, which the default bridge has")}
    phase: Optional[Dict[str, Any]] = None
    offline: Optional[Dict[str, Any]] = None
    image_id: Optional[str] = None
    staging = support / "container-in"
    try:
        # -- the stock container, networked ---------------------------------
        docker(["run", "-d", "--name", networked, "--platform", base["platform"],
                "--label", f"{IMAGE_LABEL}={run_id}", *transport["run_args"],
                base["image"], "sleep", "infinity"], label="run-networked", budget_s=900)
        _, base_id = docker(["image", "inspect", "--format", "{{.Id}}", base["image"]],
                            label="base-id")
        details["base_image_id"] = base_id.strip()
        _, mode = docker(["inspect", "--format", "{{.HostConfig.NetworkMode}}", networked],
                         label="networked-mode")
        details["networked_phase"]["network_mode"] = mode.strip()
        _, versions = docker(["exec", "-u", "0", networked, "bash", "-c", setup_script(base)],
                             label="setup", budget_s=DOCKER_SETUP_S)
        details["container_tools"] = versions.strip().splitlines()[-5:]
        manifest, lock = pin_files
        pin_dir = staging / "opencode-pin"
        _write_json(pin_dir / "package.json", manifest)
        _write_json(pin_dir / "package-lock.json", lock)
        docker(["cp", f"{pin_dir}/.", f"{networked}:{PIN_PREFIX}"], label="cp-pin")
        docker(["exec", "-u", "0", networked, "bash", "-c", pin_install_script(ctx.single_registry)],
               label="pin-install", budget_s=DOCKER_SETUP_S)
        kit = staging / "kit"
        details["kit"] = {"root": KIT_ROOT, "sha256": stage_kit(kit)}
        docker(["cp", f"{kit}/.", f"{networked}:{KIT_ROOT}"], label="cp-kit")
        facts = _facts(ctx)
        _write_json(staging / "facts.json", facts)
        docker(["cp", str(staging / "facts.json"), f"{networked}:{WORK}/facts.json"], label="cp-facts")
        if transport["relay"]:
            port = int(ctx.read_registry.rsplit(":", 1)[1])
            docker(["exec", "-d", "-u", str(CONTAINER_UID), networked, "python3",
                    f"{KIT_ROOT}/packaging/opencode/registry_stub.py", "relay",
                    "--listen-port", str(port), "--target", f"host.docker.internal:{port}",
                    "--port-file", f"{WORK}/relay.port"], label="relay")
            _wait_in(docker, networked, f"{WORK}/relay.port", "relay-ready")
            details["networked_phase"]["relay"] = (f"127.0.0.1:{port} -> host.docker.internal:{port}"
                                                   " (inside the networked container)")
        # -- the install phase ------------------------------------------------
        bounded, _ = docker(["exec", "-u", f"{CONTAINER_UID}:{CONTAINER_UID}", "-w", WORK,
                             networked, "python3", f"{KIT_ROOT}/packaging/opencode/registry_check.py",
                             "container-phase", "--phase", "install", "--facts", f"{WORK}/facts.json",
                             "--out", f"{WORK}/out/install.json"],
                            label="phase-install", budget_s=DOCKER_PHASE_S, check=False)
        docker(["cp", f"{networked}:{WORK}/out/.", str(support / "container-out")],
               label="cp-out-install", check=False)
        docker(["cp", f"{networked}:{ctx.probe_root}/diagnostics/.", str(support / "diagnostics")],
               label="cp-diagnostics-install", check=False)
        phase_path = support / "container-out" / "install.json"
        if not phase_path.is_file():
            raise CheckError(f"the install phase wrote no result ({bounded.describe()}); see "
                             f"{docker.diag}")
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        if phase.get("refusal"):
            raise CheckError(f"the install phase refused: {phase['refusal']}")
        if not phase.get("state", {}).get("payload"):
            return {"phase": phase, "offline": None, "details": details}

        # -- commit, then the committed image with no network -----------------
        docker(["stop", "-t", "3", networked], label="stop", budget_s=120)
        _, committed = docker(["commit", "--change", f"LABEL {IMAGE_LABEL}={run_id}", networked,
                               image_tag], label="commit", budget_s=900)
        image_id = committed.strip()
        details["offline_phase"] = {"image": image_tag, "image_id": image_id}
        docker(["run", "-d", "--name", offline_name, "--network", "none", "--label",
                f"{IMAGE_LABEL}={run_id}", image_tag, "sleep", "infinity"],
               label="run-offline", budget_s=300)
        _, mode_before = docker(["inspect", "--format", "{{.HostConfig.NetworkMode}}", offline_name],
                                label="offline-mode-before")
        bounded, _ = docker(["exec", "-u", f"{CONTAINER_UID}:{CONTAINER_UID}", "-w", WORK,
                             offline_name, "python3",
                             f"{KIT_ROOT}/packaging/opencode/registry_check.py", "container-phase",
                             "--phase", "offline", "--facts", f"{WORK}/facts.json",
                             "--state", f"{WORK}/out/install-state.json",
                             "--out", f"{WORK}/out/offline.json"],
                            label="phase-offline", budget_s=DOCKER_OFFLINE_S, check=False)
        _, mode_after = docker(["inspect", "--format", "{{.HostConfig.NetworkMode}}", offline_name],
                               label="offline-mode-after")
        docker(["cp", f"{offline_name}:{WORK}/out/.", str(support / "container-out")],
               label="cp-out-offline", check=False)
        docker(["cp", f"{offline_name}:{ctx.probe_root}/diagnostics/offline/.",
                str(support / "diagnostics" / "offline")], label="cp-diagnostics-offline", check=False)
        offline_path = support / "container-out" / "offline.json"
        if offline_path.is_file():
            raw = json.loads(offline_path.read_text(encoding="utf-8"))
            if raw.get("refusal"):
                offline = {"pass": False, "evidence": {"problems": [raw["refusal"]]}}
            else:
                offline = offline_criterion(
                    raw, mode_before.strip(), mode_after.strip(), ctx.single_registry,
                    f"docker commit of the warmed stock container, run with --network none "
                    f"({image_tag})")
        else:
            offline = {"pass": False, "evidence": {"problems": [
                f"the offline phase wrote no result ({bounded.describe()})"]}}
        return {"phase": phase, "offline": offline, "details": details}
    finally:
        for name in (offline_name, networked):
            docker(["rm", "-f", name], label=f"rm-{name.rsplit('-', 1)[-1]}", check=False)
        if image_id is not None:
            docker(["rmi", "-f", image_tag], label="rmi", check=False)
            gone, _ = docker(["image", "inspect", image_id], label="image-gone", check=False)
            details["image_deleted"] = gone.returncode != 0
        details["docker_calls"] = len(docker.calls)


def _wait_in(docker: Docker, container: str, path: str, label: str, budget_s: float = 20.0) -> None:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        bounded, _ = docker(["exec", container, "test", "-s", path], label=label, check=False)
        if bounded.returncode == 0:
            return
        time.sleep(0.5)
    raise CheckError(f"{path} never appeared in {container}")


def _facts(ctx: Context) -> Dict[str, Any]:
    """Everything a phase inside the container needs, and nothing it could load the plugin from."""
    return {"spec": ctx.spec, "name": ctx.name, "version": ctx.version,
            "read_registry": ctx.read_registry, "single_registry": ctx.single_registry,
            "rehearsal": ctx.rehearsal, "environment": ctx.environment,
            "candidate": {key: ctx.candidate[key] for key in
                          ("sha", "digest", "inventory", "digest_files")},
            "binary": ctx.binary, "opencode_pin": ctx.opencode_pin,
            "probe_root": str(ctx.probe_root), "previous_pin": ctx.previous_pin,
            "poll_s": ctx.poll_s, "interval_s": ctx.interval_s, "pin_prefix": PIN_PREFIX}


def _context_from_facts(facts: Mapping[str, Any]) -> Context:
    return Context(spec=facts["spec"], name=facts["name"], version=facts["version"],
                   read_registry=facts["read_registry"], single_registry=facts["single_registry"],
                   rehearsal=bool(facts["rehearsal"]), environment=facts["environment"],
                   candidate=facts["candidate"], binary=facts["binary"],
                   opencode_pin=facts["opencode_pin"], probe_root=Path(facts["probe_root"]),
                   previous_pin=facts.get("previous_pin"), poll_s=float(facts["poll_s"]),
                   interval_s=float(facts["interval_s"]))


def verify_pin(prefix: Path, ctx: Context) -> Dict[str, Any]:
    """``bump.verify_installed`` on the container's ``npm ci`` of the pin, before
    anything of it runs: refuses (``CheckError``) a package from elsewhere, one
    outside the subset, or no verified platform binary; else returns the one to
    run (``binary``), chosen in the container because the choice depends on its
    CPU and libc. The subset lockfile is the host's (``npm ci`` never rewrites
    it); the registry is the single one the host already checked."""
    bump = _bump()
    try:
        lock = json.loads((prefix / "package-lock.json").read_text(encoding="utf-8"))
        return bump.verify_installed(prefix, lock, ctx.single_registry)
    except (OSError, ValueError, bump.BumpError) as exc:
        raise CheckError(f"the pinned OpenCode under {prefix}: {exc}") from exc


def container_phase(argv: Sequence[str], steps: Optional[Steps] = None) -> int:
    """``container-phase --phase install|offline``: run INSIDE the container only."""
    parser = argparse.ArgumentParser(prog="registry_check.py container-phase")
    parser.add_argument("--phase", choices=("install", "offline"), required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv))
    try:
        if not (Path("/.dockerenv").exists() or os.environ.get("MLLM_REGISTRY_CHECK_PHASE_TEST")):
            raise CheckError("container-phase runs inside the probe's container only")
        facts = json.loads(args.facts.read_text(encoding="utf-8"))
        ctx = _context_from_facts(facts)
        # Both phases: the offline one runs from a commit of this container,
        # so the same check makes the same choice there.
        pin = verify_pin(Path(facts["pin_prefix"]), ctx) if facts.get("pin_prefix") else None
        if pin is not None:
            ctx = replace(ctx, binary=pin["binary"])
        if args.phase == "install":
            result = run_criteria(ctx, steps)
            if pin is not None:
                result["details"]["pinned_install"] = pin
            state = dict(result["state"])
            _write_json(args.out.with_name("install-state.json"), state)
            result["state"] = {k: v for k, v in state.items() if k != "env"}
            _write_json(args.out, result)
        else:
            state = json.loads(Path(args.state).read_text(encoding="utf-8"))
            _write_json(args.out, offline_operation(ctx, state, steps))
    except CheckError as exc:
        _write_json(args.out, {"refusal": str(exc)})
        print(f"registry check ({args.phase}): REFUSING: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - the host reads the result file, not this stream
        _write_json(args.out, {"refusal": f"the {args.phase} phase itself failed: "
                                          f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    return 0


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def _install_host_binary(single: str, root: Path) -> str:
    """The pinned OpenCode for an in-process run, with an npm cache of its own.

    ``bump.py install`` (the gate's own installer) as a child whose ``HOME``
    is fresh: npm's default cache is ``$HOME/.npm``, so nothing it downloads
    comes from, or goes into, the invoking user's cache. Bounded by
    ``probe.run_bounded``, so a hung registry connection ends with the budget.
    The path it prints last is a verified platform package's own binary.
    """
    prefix = root / "opencode-host"
    home = root / "opencode-host-home"
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home))
    diag = root / "diagnostics"
    log = diag / "opencode-host-install.log"
    bounded = _probe().run_bounded(
        [sys.executable, str(PACKAGE_DIR / "bump.py"), "install", "--prefix", str(prefix),
         "--registry", single], cwd=root, env=env, budget_s=HOST_INSTALL_BUDGET_S,
        stdout=diag / "opencode-host-install.stdout.txt", stderr=log)
    lines = [line for line in _read(bounded.stdout).splitlines() if line.strip()]
    if bounded.returncode != 0 or not lines:
        raise CheckError(f"bump.py install {bounded.describe()}; see {log}")
    return lines[-1].strip()


def _support_dir(out: Path) -> Path:
    return out.with_name(out.stem + ".support")


def probe_parent(tmp_parent: Optional[Path], environment: str,
                 environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """Where the probe's root is made: ``--tmp-parent`` whenever it is given.

    Then ``$RUNNER_TEMP`` on a runner, then the system temp (``None``). The
    flag must win: ``$RUNNER_TEMP`` is set on every runner, and the retired
    bootstrap job passed ``--tmp-parent`` a fresh, checked-empty directory
    precisely so its guard could require the payload under it. Preferring the
    variable rooted the probe in a sibling of that directory, and the guard
    refused every run.
    """
    if tmp_parent is not None:
        return str(tmp_parent)
    if environment == "ci-runner" and environ.get("RUNNER_TEMP"):
        return environ["RUNNER_TEMP"]
    return None


def collect_support(support: Path, scratch: Optional[Path], local_root: Optional[Path]) -> None:
    """What the probe left on this host, copied into ``support``: on a pass and
    on a refusal alike, so a ``CheckError`` keeps the diagnostics that explain
    it. ``local_root`` is the probe root when it lives on this host (never the
    container's, whose files the driver copies out itself)."""
    if local_root is not None:
        _copy_tree(Path(local_root) / "diagnostics", support / "diagnostics")
    if scratch is not None:
        _copy_tree(Path(scratch) / "diagnostics", support / "host-diagnostics")
        build_log = Path(scratch) / "candidate-build" / "build.log"
        if build_log.is_file():
            shutil.copyfile(build_log, support / "candidate-build.log")


def check(args: argparse.Namespace, steps: Optional[Steps] = None,
          docker: Optional[Docker] = None) -> int:
    """The whole probe; 0 only when every criterion passed and the record validated."""
    out = Path(args.out).absolute()
    scratch: Optional[Path] = None
    support: Optional[Path] = None
    local_root: Optional[Path] = None
    record: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    passed = False
    try:
        if args.environment not in ENVIRONMENTS:
            raise CheckError(f"--environment {args.environment!r} is not one of {ENVIRONMENTS}")
        single, read_registry, rehearsal = resolve_registries(args.registry, args.rehearsal_registry)
        name, version = parse_spec(args.spec)
        if rehearsal and _is_under(out, RECORDS_DIR):
            raise CheckError(f"refusing to write a rehearsal record under {RECORDS_DIR}: its "
                             ".gitignore exception would make it committable as release evidence")
        run_id = args.workflow_run_id
        if run_id is None and args.environment == "ci-runner" and \
                os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_RUN_ID"):
            run_id = int(os.environ["GITHUB_RUN_ID"])
        if args.environment == "ci-runner":
            if run_id is None and not rehearsal:
                raise CheckError("a bootstrap record names its workflow run: pass --workflow-run-id "
                                 "(or run it inside GitHub Actions)")
        elif run_id is not None:
            raise CheckError(f"--workflow-run-id belongs to the ci-runner (bootstrap) record, not "
                             f"a {args.environment} one")
        if run_id is not None and run_id < 1:
            raise CheckError(f"--workflow-run-id {run_id} is not a run")

        repo = Path(args.repo).absolute()
        rr = _record_module()
        try:
            tag = rr.tag_facts(repo, args.tag)
        except rr.RecordError as exc:
            raise CheckError(str(exc)) from exc
        if (tag["name"], tag["version"]) != (name, version):
            raise CheckError(f"--spec {args.spec} is not {args.tag}'s package.json "
                             f"({tag['name']}@{tag['version']})")
        if args.tag != f"{rr.TAG_PREFIX}{version}":
            raise CheckError(f"--tag {args.tag} does not name version {version}")
        kind = release_kind(tag["public_version"])
        previous = args.previous_pin
        if kind == "later":
            expected = f"{name}@{tag['public_version']}"
            if args.environment == "ci-runner":
                raise CheckError(f"{args.tag}'s capabilities name public_version "
                                 f"{tag['public_version']}: a bootstrap record covers the first "
                                 "release only")
            if previous is None:
                previous = expected
            elif previous != expected:
                raise CheckError(f"--previous-pin {previous} is not {expected}, the tag's "
                                 "public_version")
        elif previous is not None:
            raise CheckError(f"{args.tag}'s capabilities have no public_version: a first release "
                             f"has no previous pin (got {previous})")

        pin = _bump().pinned_version()
        parent = probe_parent(args.tmp_parent, args.environment)
        scratch = Path(tempfile.mkdtemp(prefix="mllm-regcheck-", dir=parent)).resolve()
        if _is_under(scratch, repo):
            raise CheckError(f"the probe's root {scratch} lies inside the checkout {repo}")
        # A rerun replaces both the record and its diagnostics. If it refuses
        # before building a new record, an earlier green one must not survive
        # beside this attempt's failed supporting output.
        try:
            out.unlink(missing_ok=True)
        except OSError as exc:
            raise CheckError(f"could not invalidate the previous record {out}: {exc}") from exc
        support = _support_dir(out)
        _rmtree(support)
        support.mkdir(parents=True, exist_ok=True)
        print(f"registry check: {args.spec} at {args.tag} ({tag['sha'][:12]}), {kind} release, "
              f"environment {args.environment}{', REHEARSAL against ' + read_registry if rehearsal else ''}; "
              f"root {scratch}", file=sys.stderr, flush=True)
        candidate = rebuild_candidate(repo, tag["sha"], scratch / "candidate-build")
        print(f"registry check: candidate rebuilt from {tag['sha'][:12]}: {candidate['files']} files, "
              f"digest {candidate['digest']}", file=sys.stderr, flush=True)

        details: Dict[str, Any] = {"single_registry": single, "support": support.name,
                                   "candidate": {k: candidate[k] for k in
                                                 ("sha", "digest", "files", "built_in", "build")}}
        if rehearsal:
            details["rehearsal_registry"] = read_registry
        offline: Optional[Dict[str, Any]] = None
        if args.environment == "container":
            if args.opencode:
                raise CheckError("--opencode names a host binary; the container installs the "
                                 "pinned OpenCode itself, from the lockfile")
            ctx = Context(spec=args.spec, name=name, version=version, read_registry=read_registry,
                          single_registry=single, rehearsal=rehearsal, environment=args.environment,
                          # Chosen inside, by `verify_pin`, from what npm put there.
                          candidate=candidate, binary=PIN_PREFIX,
                          opencode_pin=pin, probe_root=Path(WORK) / "probe-root",
                          previous_pin=previous, poll_s=args.poll_seconds)
            lock = json.loads((REPO_ROOT / _bump().LOCKFILE).read_text(encoding="utf-8"))
            driven = drive_container(ctx=ctx, base=container_base(), support=support, docker=docker,
                                     pin_files=_bump().install_subset(lock, pin, registry=single))
            phase, offline = driven["phase"], driven["offline"]
            details["container"] = driven["details"]
        else:
            local_root = scratch / "probe-root"
            binary = args.opencode or _install_host_binary(single, scratch)
            ctx = Context(spec=args.spec, name=name, version=version, read_registry=read_registry,
                          single_registry=single, rehearsal=rehearsal, environment=args.environment,
                          candidate=candidate, binary=str(Path(binary).absolute()), opencode_pin=pin,
                          probe_root=local_root, previous_pin=previous,
                          poll_s=args.poll_seconds)
            phase = run_criteria(ctx, steps)
            if args.environment == "vm" and phase["state"].get("payload"):
                cut = wait_for_cut(steps or Steps())
                if cut["cut"]:
                    raw = offline_operation(ctx, phase["state"], steps)
                    mode = "none" if all(_LOOPBACK.match(i) for i in
                                         raw["before"]["interfaces"] + raw["after"]["interfaces"]) \
                        else "attached"
                    offline = offline_criterion(raw, mode, mode, single,
                                                "this VM with its network interface detached")
                else:
                    offline = {"pass": False, "evidence": {"problems": [
                        f"the VM's network was never cut (up: {cut['up']})"]}}
        criteria = dict(phase["criteria"])
        if args.environment != "ci-runner":
            criteria[OFFLINE] = offline or {"pass": False, "evidence": {
                "not_attempted": "the networked phase left no installed payload to run offline"}}
        details.update(phase.get("details") or {})
        details["phase_failures"] = phase.get("failures") or []
        notes = ("REHEARSAL against a localhost registry stub serving the candidate tarball under "
                 "its own name and version; never release evidence." if rehearsal else "")
        record = build_record(
            environment=args.environment, rehearsal=rehearsal, workflow_run_id=run_id,
            spec=args.spec, name=name, version=version, read_registry=read_registry,
            installed_registry=(phase.get("state") or {}).get("installed_registry"),
            integrity=(phase.get("state") or {}).get("integrity"), tag=args.tag, sha=tag["sha"],
            candidate_digest=candidate["digest"], kind=kind, previous_pin=previous,
            measured=phase.get("measured") or {}, variables_set=phase.get("variables_set") or {},
            criteria=criteria, payload_location=(phase.get("state") or {}).get("location"),
            environment_details=details, notes=notes)
        _write_json(out, record)
        errors = validate_record(record, environment=args.environment, tag=args.tag, repo=repo,
                                 digest=candidate["digest"])
        passed = all(c["pass"] for c in record["criteria"]) and not errors
    except CheckError as exc:
        print(f"registry check: REFUSING: {exc}", file=sys.stderr)
        if scratch is not None:
            print(f"  kept {scratch}", file=sys.stderr)
        if support is not None:
            print(f"  supporting output: {support}", file=sys.stderr)
        return 1
    finally:
        if support is not None:
            try:
                collect_support(support, scratch, local_root)
            except OSError as exc:  # a copy that fails must not hide the verdict
                print(f"registry check: could not copy the supporting output: {exc}",
                      file=sys.stderr)
    if record is not None:
        for entry in record["criteria"]:
            verdict = "pass" if entry["pass"] else "FAIL"
            print(f"  {verdict:4}  {entry['name']}", file=sys.stderr)
            for problem in (entry["evidence"].get("problems") or [])[:6]:
                print(f"          - {problem}", file=sys.stderr)
            if entry["evidence"].get("not_attempted"):
                print(f"          - not attempted: {entry['evidence']['not_attempted']}", file=sys.stderr)
        for error in errors:
            print(f"  validator: {error}", file=sys.stderr)
    if passed:
        what = ("structurally valid REHEARSAL record (a rehearsal is never release evidence)"
                if record.get("rehearsal") else f"valid {profile_for(args.environment)} record")
        print(f"registry check: PASS; {what}: {out}")
        if scratch is not None and not args.keep:
            _rmtree(scratch)
        return 0
    print(f"registry check: FAIL; record: {out}" + (f"; kept {scratch}" if scratch else ""),
          file=sys.stderr)
    return 1


def _copy_tree(source: Path, destination: Path) -> None:
    if Path(source).is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="registry_check.py",
        description="The registry-install probe: what npm serves, installed by OpenCode.")
    parser.add_argument("--spec", required=True, help="the exact npm pin, <name>@<version>")
    parser.add_argument("--registry", required=True, help="Makefile OPENCODE_REGISTRY (the single registry)")
    parser.add_argument("--tag", required=True, help="the release tag, opencode-v<version>")
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument("--out", required=True, type=Path, help="where the record goes")
    parser.add_argument("--rehearsal-registry", default=None,
                        help="REHEARSAL ONLY: a localhost registry stub, http://127.0.0.1:<port>")
    parser.add_argument("--workflow-run-id", type=int, default=None,
                        help="ci-runner only (default: $GITHUB_RUN_ID inside GitHub Actions)")
    parser.add_argument("--opencode", default=None,
                        help="the pinned binary (ci-runner/vm; default: bump.py install under the root)")
    parser.add_argument("--previous-pin", default=None,
                        help="a later release's previous pin (default: <name>@<tag's public_version>)")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT,
                        help="the clone whose tag decides the commit (default: this one)")
    parser.add_argument("--poll-seconds", type=float, default=NPM_POLL_S,
                        help=f"how long npm view is polled for the version (default {NPM_POLL_S:.0f})")
    parser.add_argument("--tmp-parent", type=Path, default=None,
                        help="where the probe's root is made; wins whenever given (default: "
                             "$RUNNER_TEMP on a runner, else the system temp)")
    parser.add_argument("--keep", action="store_true", help="keep the probe's root after a pass")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # SIGTERM/SIGINT unwind through every sweep (`probe.install_exit_handlers`):
    # OpenCode and its uv child lead sessions of their own, out of reach of a
    # signal to this process's group.
    _probe().install_exit_handlers()
    if argv and argv[0] == "container-phase":
        return container_phase(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return check(args)
    except Exception:  # noqa: BLE001 - a traceback is the diagnostic; the exit is still 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
