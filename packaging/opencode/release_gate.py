#!/usr/bin/env python3
"""The opencode package's release gate: the rows it requires, and whether they
ran in **this** gate session (plans/opencode-plugin-publish.md, step 0's
gate-rows table).

``make opencode-package-check`` used to end in an unconditional row-7 refusal,
and before that its ``--write-stamp`` could write a full-tier stamp without
reading any evidence at all: recipe ordering was the only thing standing
between a stamp and rows nobody ran, and ordering is not evidence. This module
is the one executable contract that replaced both. It owns five things:

* **The rows.** :data:`ROWS` declares exactly the plan's row ids — ``1``, ``2``,
  ``3``, ``4``, ``5``, ``5b``, ``6``, ``7``, ``8``, ``9`` — each with what it
  proves, its grader and its evidence. ``8b`` (the serial ordinary suite) is
  not a stamp row: CI's three ``test`` check runs carry it. The static rows
  name the pytest node ids that prove them, and :func:`grade_ci` reads them out
  of junit XML one by one. pytest's exit code is not that: a module skipped for
  want of npm, a deselected row and an empty selection all exit 0 or 5, and
  each of those is a row that did not run.
* **The session.** :func:`begin_session` starts a gate: earlier evidence is
  deleted and a fresh id written beside the candidate commit, version and
  dirtiness. Every piece of evidence carries the id it was written under, and
  the stamp accepts only the running one — so a row that went green last week
  cannot be stamped beside today's. ``ALLOW_DIRTY=1`` (or a dirty tree at
  ``begin``) makes the session a **rehearsal**: every row may run, and the
  stamp and the release-gate record both refuse it by name.
* **The image binding (row 8).** :func:`bind_image` resolves the image the gate
  just built to its immutable id, reads the payload digest **out of that id**
  (never the checkout, never the mutable tag), refuses an image built from
  other bytes and an inherited ``MLLM_E2E_IMAGE`` that names another image, and
  :func:`check_row8` refuses a scenario record from another image, another run
  or another session, or one that no grader graded.
* **The stamp's input.** :func:`evidence_problems` is what ``build.py
  --write-stamp`` runs before writing anything, and :func:`stamp_summary` is the
  checked summary it embeds.
* **The release-gate record.** :func:`export_record` writes
  ``tests/container/records/<UTC>-opencode-release-gate.json`` from the session
  that authorized the stamp — the one way row 8's local evidence reaches CI,
  whose runners have no Docker daemon — and :func:`validate_records` is the one
  validator the publishing workflow's ``verify`` job and the suite both run.

**Stdlib only**, and that is a requirement rather than a preference: the
``verify`` job runs ``validate-record`` under ``uv run --frozen --no-dev``, which
installs the skill's runtime dependencies alone (``pyyaml``), so ``jsonschema``
— a dev dependency — is not there. :func:`schema_errors` therefore implements
the subset of JSON Schema the record's schema uses, and the suite holds the
schema to that subset. The one non-stdlib reach, the container harness's
PROVISIONAL/UNGRADED markers, happens inside :func:`check_row8` only, which
runs in the local gate and never in CI.

Loaded by path (``build.py`` does it, and so does the suite) under the name
``opencode_release_gate``: the codex route has a ``release_gate`` module of its
own, and one pytest session imports both.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "multi-llm"
CONTAINER_DIR = SKILL_DIR / "tests" / "container"
RECORDS_DIR = CONTAINER_DIR / "records"
VERIFY_RECORD = CONTAINER_DIR / "verify_record.py"
POSTURE = CONTAINER_DIR / "posture.yaml"
MANIFEST = PACKAGE_DIR / "package.json"
CANDIDATE_DIGEST = PACKAGE_DIR / "dist" / "tree.sha256"
RECORD_SCHEMA = PACKAGE_DIR / "release-gate-record.schema.json"
STAMP_NAME = ".package-check.stamp"

#: Where the session's evidence lands. Gitignored beside the stamp: it attests
#: to the machine that ran it, exactly like the stamp it feeds. Other work
#: packages never write here; their CLIs write to the ``--out`` path the
#: Makefile hands them, and :func:`record_row` is what brings that file in.
EVIDENCE_DIR = PACKAGE_DIR / ".gate"
#: Where ``export-record`` writes by default: inside the gitignored evidence
#: directory, NOT ``RECORDS_DIR``. ``records/`` lies under the payload prefix
#: ``skills/multi-llm/``, so an untracked record there makes the tree dirty and
#: ``build.py --check-stamp`` — and with it ``opencode-publish-first`` — refuse
#: the very candidate the gate just stamped. The record is committed to the
#: default branch only after the fast-forward (plan step 2), in a commit of its
#: own; the export prints the copy-and-commit lines for that step.
EXPORT_SUBDIR = "export"
EXPORT_DIR = EVIDENCE_DIR / EXPORT_SUBDIR
SESSION_NAME = "session.json"
IMAGE_NAME = "image.json"
#: Sub-directories :func:`begin_session` empties too: raw CLI output, step
#: logs, junit files and the scratch host install all belong to one session.
SESSION_SUBDIRS = ("raw", "logs", "junit", "host")

TAG_PREFIX = "opencode-v"
RECORD_SUFFIX = "-opencode-release-gate.json"

#: Inside the image (tests/container/Dockerfile §10b): the installed package and
#: the two files the gate reads out of the image id.
INSTALLED_ROOT = "/opt/mllm/opencode-plugin"
IMAGE_PAYLOAD_DIGEST = f"{INSTALLED_ROOT}/dist/tree.sha256"
IMAGE_INSTALLED_REPORT = "/opt/mllm/lock/cli-versions.installed.json"

_STAMP = re.compile(r"\d{8}T\d{6}Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class GateError(Exception):
    """A refusal, worded for the maintainer reading the make output."""


# ---------------------------------------------------------------------------
# The rows — plans/opencode-plugin-publish.md, step 0's table, and nothing else
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateRow:
    id: str
    proves: str
    grader: str
    evidence: str
    #: The pytest node ids that prove a static row, relative to the skill
    #: directory (pytest's rootdir). Empty for rows 7-9, which are not tests.
    nodes: Tuple[str, ...] = ()


_PKG = "tests/test_opencode_package.py"

ROWS: Tuple[GateRow, ...] = (
    GateRow(
        "1", "tarball inventory: the runtime skill files, instructions, lockfile, "
        "command and entry point present; tests, local state, credentials and "
        "virtualenvs absent",
        "grade-ci", "every node id below graded `passed` in this session's junit",
        (f"{_PKG}::TestPack::test_package_files_manifest",
         f"{_PKG}::TestPack::test_package_ships_license_and_readme",
         f"{_PKG}::TestPack::test_the_tarball_sweep_runs_over_what_ships",
         f"{_PKG}::TestStaging::test_the_required_inventory_is_complete",
         f"{_PKG}::TestStaging::test_every_mode_table_target_module_is_staged",
         f"{_PKG}::TestStaging::test_the_staged_set_equals_the_allowlist_expansion",
         f"{_PKG}::TestStaging::test_the_staging_sweep_refuses_a_planted_secret"),
    ),
    GateRow(
        "2", "no `dependencies` or `peerDependencies` in package.json",
        "grade-ci", "the node id below graded `passed`",
        (f"{_PKG}::TestManifest::test_package_declares_no_dependencies",),
    ),
    GateRow(
        "3", "no bare-specifier import in dist/index.mjs (a verbatim copy of "
        "src/index.mjs, which the second node licenses)",
        "grade-ci", "both node ids graded `passed`",
        (f"{_PKG}::TestJavaScriptSource::test_no_bare_specifiers_in_js",
         f"{_PKG}::TestBuildArtifacts::test_dist_mirrors_src"),
    ),
    GateRow(
        "4", "dist/command.md equals a fresh render of the command template",
        "grade-ci", "the node id below graded `passed`",
        (f"{_PKG}::TestBuildArtifacts::test_command_md_matches_render",),
    ),
    GateRow(
        "5", "the packaged pyproject.toml/uv.lock are the pair the image bakes",
        "grade-ci", "the node id below graded `passed`",
        (f"{_PKG}::TestStaging::test_lockfile_parity_with_image",),
    ),
    GateRow(
        "5b", "the default `uv run` argv carries `--no-dev`, and a "
        "`uv sync --frozen --no-dev --dry-run` of the staged payload installs "
        "exactly the one runtime package",
        "grade-ci", "both node ids graded `passed`",
        (f"{_PKG}::TestBuildArtifacts::test_default_uv_run_argv_carries_no_dev",
         f"{_PKG}::TestBuildArtifacts::test_the_no_dev_sync_installs_one_package"),
    ),
    GateRow(
        "6", "the engines.opencode floor equals the container's OpenCode pin, "
        "ceiling `<2.0.0`",
        "grade-ci", "both node ids graded `passed`",
        (f"{_PKG}::TestManifest::test_engines_floor_matches_the_container_pin",
         f"{_PKG}::TestManifest::test_engines_floor_is_the_container_pin"),
    ),
    GateRow(
        "7", "the gated tarball, extracted and loaded as a directory by the pinned "
        "OpenCode in an isolated home, registers command[\"multi-llm\"] resolving "
        "inside the payload and the multi_llm tool, and runs init and status "
        "through the tool, in a committed and a no-commit project",
        "probe.py row7, then `record 7`",
        "the exact `opencode --version`, the tarball's integrity, the resolved "
        "command path, both projects' tool results and the registry URL",
    ),
    GateRow(
        "8", "the installed package (route: package, at /opt/mllm/opencode-plugin) "
        "runs the five scenarios through a real `opencode run`, with stub models "
        "and no credentials, on an image whose payload digest is the candidate's",
        "the row-8 container module's per-scenario graders, then `check-row8`",
        "one run record per scenario naming the gate's image id, each carrying "
        "gate-grader.json and passing `make e2e-verify-record`",
    ),
    GateRow(
        "9", "opencode-multi-llm@<version> is absent from the single registry",
        "registry_absent.mjs, then `record 9`",
        "the registry URL and npm's error code; only E404 passes",
    ),
)
ROW_IDS: Tuple[str, ...] = tuple(row.id for row in ROWS)
CI_ROWS: Tuple[str, ...] = tuple(row.id for row in ROWS if row.nodes)
_BY_ID = {row.id: row for row in ROWS}

#: Row 8's five scenarios, by the row ids the container module writes records
#: under (impl contract). An init/status pair alone is not row 8.
ROW8_SCENARIOS: Tuple[str, ...] = (
    "opencode-package-gate-init-status",
    "opencode-package-gate-detach",
    "opencode-package-gate-supervision",
    "opencode-package-gate-completion",
    "opencode-package-gate-cancellation",
)
#: The one scenario whose grader must say how the abort was sent.
ROW8_ABORT_SCENARIO = "opencode-package-gate-cancellation"


# ---------------------------------------------------------------------------
# Small facts, each read from its one owner
# ---------------------------------------------------------------------------

def _sibling(stem: str, alias: str):
    """``packaging/opencode/<stem>.py``, imported once, by path.

    By path because the bare names collide: the codex route ships a ``build``
    and a ``release_gate`` of its own, and whichever a pytest session imported
    first would answer ``import build`` here. An already-loaded copy of *this*
    file is reused under either name, so a test that monkeypatches the module
    it imported patches the one this gate calls.
    """
    path = (PACKAGE_DIR / f"{stem}.py").resolve()
    for name in (stem, alias, "__main__"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None) \
                and Path(module.__file__).resolve() == path:
            return module
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _builder():
    return _sibling("build", "opencode_build")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(moment: datetime.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact(moment: datetime.datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or MANIFEST).read_text(encoding="utf-8"))


def package_version() -> str:
    return str(manifest()["version"])


def release_tag(version: Optional[str] = None) -> str:
    return f"{TAG_PREFIX}{version or package_version()}"


def gated_tarball() -> Path:
    """What ``npm pack`` writes for this version: ``<name>-<version>.tgz``, in
    the package directory. The full gate's ``make opencode-package`` packs and
    scans it, row 7 loads it, and the stamp binds it."""
    data = manifest()
    return PACKAGE_DIR / f"{data['name']}-{data['version']}.tgz"


def candidate_digest() -> str:
    """The clean build's ``dist/tree.sha256``, as ``make opencode-package`` left it."""
    try:
        return CANDIDATE_DIGEST.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GateError(f"{CANDIDATE_DIGEST} is unreadable ({exc}); the candidate "
                        "was never built — run `make opencode-package`") from exc


def single_registry() -> str:
    return _sibling("registry", "opencode_registry").single_registry()


def host_pin() -> str:
    """The one OpenCode release every pin names, or :class:`GateError`.

    ``bump.read_pins()`` is the reader ``test_every_pin_agrees`` uses; a
    disagreement among them is a half-landed bump, and no gate should pick
    one side of it.
    """
    pins = _sibling("bump", "opencode_bump").read_pins()
    values = sorted(set(map(str, pins.values())))
    if len(values) != 1:
        raise GateError("the opencode pins disagree ("
                        + ", ".join(f"{name}={value}" for name, value in sorted(pins.items()))
                        + "); `make opencode-bump` moves them together")
    return values[0]


def head_sha(root: Path = REPO_ROOT) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"could not run `git rev-parse HEAD`: {exc}") from exc
    sha = done.stdout.strip()
    if done.returncode != 0 or not _SHA.fullmatch(sha):
        raise GateError(f"`git rev-parse HEAD` failed in {root}: "
                        f"{(done.stderr or done.stdout).strip()[:300]} — a release "
                        "candidate is a commit")
    return sha


def _own_export(path: str) -> bool:
    return path.startswith("skills/multi-llm/tests/container/records/") \
        and path.endswith(RECORD_SUFFIX)


def dirty_paths(root: Path = REPO_ROOT) -> List[str]:
    """Every tracked change and untracked, unignored path, as ``git status``
    shows them — minus the release-gate records this gate exports, which are
    its own output and land in a later commit by design (plan step 0).

    The whole tree, not the payload prefixes ``build.py`` guards: an
    uncommitted edit to this module, the Makefile or a row's test changes what
    the gate *checks* without changing a shipped byte.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z",
             "--untracked-files=normal"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"could not run `git status`: {exc}") from exc
    if done.returncode != 0:
        raise GateError(f"`git status` failed in {root}: {done.stderr.strip()[:300]}")
    paths: List[str] = []
    tokens = done.stdout.split("\0")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 4:
            continue
        status, path = token[:2], token[3:]
        if status[0] in "RC":
            index += 1  # a rename's or copy's source path follows
        if not _own_export(path):
            paths.append(f"{status} {path}")
    return paths


# ---------------------------------------------------------------------------
# The session: one gate invocation's evidence, never a mix of several
# ---------------------------------------------------------------------------

def begin_session(directory: Path = EVIDENCE_DIR, *, allow_dirty: bool = False,
                  root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Start a gate session: earlier evidence deleted, a fresh id written.

    ``make opencode-package-check`` begins one before any row runs, and the
    stamp accepts only evidence carrying the current id. The session also
    fixes the candidate: the commit, the version, and whether the tree was
    dirty — a dirty tree, or ``ALLOW_DIRTY=1`` (``allow_dirty``), makes this a
    rehearsal that may run every row and authorizes nothing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for name in SESSION_SUBDIRS:
        shutil.rmtree(directory / name, ignore_errors=True)
    for name in ("raw", "logs", "junit"):
        (directory / name).mkdir()
    dirty = dirty_paths(root)
    moment = _now()
    session = {
        "id": uuid.uuid4().hex,
        "started_at": _iso(moment),
        "started_stamp": _compact(moment),
        "candidate_sha": head_sha(root),
        "version": package_version(),
        "dirty": bool(dirty),
        "dirty_paths": dirty[:40],
        "allow_dirty": bool(allow_dirty),
    }
    (directory / SESSION_NAME).write_text(json.dumps(session, indent=2) + "\n",
                                          encoding="utf-8")
    return session


def current_session(directory: Path = EVIDENCE_DIR) -> Optional[Dict[str, Any]]:
    session = _read_json(Path(directory) / SESSION_NAME)
    return session if isinstance(session, dict) and session.get("id") else None


def is_rehearsal(session: Mapping[str, Any]) -> bool:
    return bool(session.get("dirty")) or bool(session.get("allow_dirty"))


def _require_session(directory: Path) -> Dict[str, Any]:
    session = current_session(directory)
    if session is None:
        raise GateError(f"no gate session under {directory}: run `make "
                        "opencode-package-check`, which begins one before any row")
    return session


def _row_path(directory: Path, row_id: str) -> Path:
    return Path(directory) / f"row-{row_id}.json"


def write_row(row_id: str, evidence: Mapping[str, Any],
              directory: Path = EVIDENCE_DIR, *, root: Path = REPO_ROOT) -> Path:
    """Write one row's evidence under the running session.

    The session id and the commit at write time are attached here, never
    taken from the caller, so a row's evidence cannot claim a session it was
    not written in.
    """
    if row_id not in _BY_ID:
        raise GateError(f"unknown gate row {row_id!r}; the rows are {', '.join(ROW_IDS)}")
    session = _require_session(directory)
    stamped = {**evidence, "row": row_id, "session": session["id"],
               "candidate_sha": head_sha(root), "recorded_at": _iso(_now())}
    path = _row_path(directory, row_id)
    path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_evidence(directory: Path = EVIDENCE_DIR) -> Dict[str, Any]:
    directory = Path(directory)
    return {
        "session": current_session(directory),
        "image": _read_json(directory / IMAGE_NAME),
        "rows": {row_id: _read_json(_row_path(directory, row_id)) for row_id in ROW_IDS},
    }


# ---------------------------------------------------------------------------
# Rows 1-6 and 5b: graded node by node out of junit XML
# ---------------------------------------------------------------------------

def _junit_key(node: str) -> Tuple[str, str]:
    """``tests/x.py::Class::test`` → (``tests.x.Class``, ``test``): the two
    attributes pytest's junit writer derives from the same node id."""
    path, *parts = node.split("::")
    module = (path[:-3] if path.endswith(".py") else path).replace("/", ".")
    return ".".join([module, *parts[:-1]]), parts[-1]


def junit_cases(paths: Sequence[Path]) -> List[Tuple[str, str, str]]:
    """``(classname, name, outcome)`` for every ``<testcase>`` in ``paths``.

    ``outcome`` is ``failed`` for a ``<failure>`` or ``<error>`` child (a
    teardown error included), ``skipped`` for ``<skipped>`` (an xfail too), and
    ``passed`` otherwise. A missing or unparseable file is an error naming it:
    a junit file pytest never finished writing is not an empty pass.
    """
    cases: List[Tuple[str, str, str]] = []
    for path in paths:
        try:
            tree = ElementTree.parse(str(path))
        except (OSError, ElementTree.ParseError) as exc:
            raise GateError(f"{path}: not a readable junit file ({exc}) — did pytest "
                            "run at all?") from exc
        for case in tree.getroot().iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                outcome = "failed"
            elif case.find("skipped") is not None:
                outcome = "skipped"
            else:
                outcome = "passed"
            cases.append((case.get("classname") or "", case.get("name") or "", outcome))
    return cases


def _node_outcome(node: str, cases: Sequence[Tuple[str, str, str]]) -> str:
    want_class, want_name = _junit_key(node)
    found = [outcome for classname, name, outcome in cases
             if (classname == want_class or classname.endswith("." + want_class))
             and (name == want_name or name.startswith(want_name + "["))]
    if not found:
        return "absent"
    if all(outcome == "passed" for outcome in found):
        return "passed"
    return "failed" if "failed" in found else "skipped"


def grade_ci(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Each static row's verdict: ``{row: {"pass", "nodes", "junit", ...}}``.

    A row passes only when **every** node it declares is present and passed.
    ``absent`` is the deselection case — a marker expression that drops the
    row, a rename, a module that failed to collect — and it fails the row
    exactly as a skip does.
    """
    cases = junit_cases(paths)
    verdicts: Dict[str, Dict[str, Any]] = {}
    for row_id in CI_ROWS:
        nodes = {node: _node_outcome(node, cases) for node in _BY_ID[row_id].nodes}
        verdicts[row_id] = {
            "grader": "grade-ci",
            "pass": bool(nodes) and all(v == "passed" for v in nodes.values()),
            "nodes": nodes,
            "junit": [Path(path).name for path in paths],
            "cases_read": len(cases),
        }
    return verdicts


def _ci_row_problems(row_id: str, verdict: Mapping[str, Any]) -> List[str]:
    if not verdict.get("cases_read"):
        return [f"row {row_id}: the junit files hold no test case at all — an empty "
                "selection proves nothing"]
    nodes = verdict.get("nodes") or {}
    problems = []
    for node in _BY_ID[row_id].nodes:
        outcome = nodes.get(node, "absent")
        if outcome != "passed":
            problems.append(f"row {row_id}: {node} {outcome}"
                            + (" (deselected, renamed or never collected)"
                               if outcome == "absent" else ""))
    return problems


def session_for_grading(directory: Path = EVIDENCE_DIR,
                        root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """The session grade-ci writes into: one that exists **for HEAD**.

    The PR package job runs ``opencode-package-check-ci`` in a fresh clone with
    no session, and only grades. A stale session from an earlier commit is not
    this gate's either.
    """
    session = current_session(directory)
    if session is None:
        return None
    try:
        return session if session.get("candidate_sha") == head_sha(root) else None
    except GateError:
        return None


# ---------------------------------------------------------------------------
# Rows 7 and 9: another work package's CLI writes the JSON, this module checks it
# ---------------------------------------------------------------------------

def _is_true(value: Any) -> bool:
    return value is True


#: Each row-7 call's documented results, as ``(exit, MULTI_LLM_RESULT state)``.
#: Not "exit 0": the uniform exit codes make 2 "nothing to do", and that is
#: exactly what ``status`` answers for a project whose only run so far was
#: ``init`` ("no run recorded"). A status that found a finished run says
#: ``(0, "done")``. Anything else — 1, 3, an ``error`` state — is a failure.
ROW7_DOCUMENTED_RESULTS: Dict[str, frozenset] = {
    "init": frozenset({(0, "done")}),
    "status": frozenset({(0, "done"), (2, "nothing-to-do")}),
}


def _is_under(path: str, root: str) -> bool:
    """Is ``path`` ``root`` itself or inside it, compared resolved?"""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def row7_problems(raw: Any, *, host_version: str, registry: str,
                  tarball: Optional[Mapping[str, Any]], tarball_path: Path) -> List[str]:
    """Why ``probe.py row7 --out``'s file cannot be row 7's evidence.

    Shape and bindings at once: the probe ran the pinned host against the
    single registry on **the gated tarball** (same file, same integrity, same
    tree digest), and every assertion it reports passed. A probe can exit 0 and
    still describe a run that proves nothing; this reads what it says it did.
    """
    if not isinstance(raw, Mapping):
        return ["row 7: the probe output is not a JSON object"]
    problems: List[str] = []
    if raw.get("row") != "7":
        problems.append(f"row 7: the file says row {raw.get('row')!r}")
    if not _is_true(raw.get("pass")):
        failures = raw.get("failures") or []
        problems.append(f"row 7: the probe did not pass (pass: {raw.get('pass')!r}); "
                        f"failures: {json.dumps(failures)[:600]}")
    elif raw.get("failures"):
        problems.append(f"row 7: pass is true beside failures {json.dumps(raw['failures'])[:400]}")
    if raw.get("opencode_version") != host_version:
        problems.append(f"row 7: ran opencode {raw.get('opencode_version')!r}, the pinned "
                        f"host is {host_version} — the probe never tests whatever happened "
                        "to install")
    if raw.get("registry") != registry:
        problems.append(f"row 7: registry {raw.get('registry')!r}, the single registry is "
                        f"{registry}")
    for key in ("probe_root",):
        if not isinstance(raw.get(key), str) or not raw.get(key):
            problems.append(f"row 7: no {key}")
    isolation = raw.get("isolation")
    if not isinstance(isolation, Mapping) or not isinstance(isolation.get("env_cleared"), list) \
            or not isinstance(isolation.get("home"), str):
        problems.append("row 7: no isolation block (env_cleared list and home)")
    loaded = raw.get("tarball") if isinstance(raw.get("tarball"), Mapping) else {}
    if tarball is None:
        problems.append(f"row 7: the gated tarball {tarball_path} is missing; "
                        "`make opencode-package` packs it")
    else:
        if not loaded.get("path") or Path(str(loaded["path"])).resolve() != Path(tarball_path).resolve():
            problems.append(f"row 7: the probe loaded {loaded.get('path')!r}, the gated "
                            f"tarball is {tarball_path}")
        if loaded.get("integrity") != tarball.get("integrity"):
            problems.append(f"row 7: the probe's tarball integrity {loaded.get('integrity')!r} "
                            f"is not the gated tarball's {tarball.get('integrity')}")
        if loaded.get("digest") != tarball.get("digest"):
            problems.append(f"row 7: the probe's tarball digest {loaded.get('digest')!r} is not "
                            f"the gated tarball's {tarball.get('digest')}")
    projects = raw.get("projects")
    kinds = [p.get("kind") for p in projects] if isinstance(projects, list) \
        and all(isinstance(p, Mapping) for p in projects) else None
    if kinds is None or sorted(kinds) != ["committed", "no-commit"]:
        problems.append(f"row 7: projects {kinds!r}, the probe runs exactly a committed "
                        "and a no-commit project")
    for project in projects if kinds is not None else []:
        kind = project.get("kind")
        for flag in ("command_registered", "command_inside_payload", "tool_registered"):
            if not _is_true(project.get(flag)):
                problems.append(f"row 7: {kind} project: {flag} is {project.get(flag)!r}")
        skill_dir = project.get("command_skill_dir")
        if not isinstance(skill_dir, str) or not skill_dir:
            problems.append(f"row 7: {kind} project: no command_skill_dir")
        elif isinstance(raw.get("probe_root"), str) and raw.get("probe_root") and not \
                _is_under(skill_dir, raw["probe_root"]):
            problems.append(f"row 7: {kind} project: the tool ran the skill at {skill_dir}, "
                            f"outside the probe's own root {raw['probe_root']} — not the "
                            "extracted gated tarball")
        for call in ("init", "status"):
            result = project.get(call)
            documented = ROW7_DOCUMENTED_RESULTS[call]
            outcome = ((result.get("exit"), (result.get("result") or {}).get("state"))
                       if isinstance(result, Mapping)
                       and isinstance(result.get("result"), Mapping) else None)
            if not isinstance(result, Mapping) or not _is_true(result.get("ok")) \
                    or outcome not in documented \
                    or (result.get("result") or {}).get("mode") != call:
                problems.append(f"row 7: {kind} project: {call} did not return its "
                                f"documented result (one of {sorted(documented)}): "
                                f"{json.dumps(result)[:300]}")
        created = (project.get("init") or {}).get("files_created") \
            if isinstance(project.get("init"), Mapping) else None
        if not isinstance(created, list) or not created:
            problems.append(f"row 7: {kind} project: init created no project-local file")
    return problems


def row9_problems(raw: Any, *, registry: str, name: str, version: str) -> List[str]:
    """Why ``registry_absent.mjs --out``'s file cannot be row 9's evidence.

    Re-read here rather than trusted from the script's exit: ``pass`` must be
    exactly ``outcome == "absent"``, and absence is exactly npm's ``E404``.
    """
    if not isinstance(raw, Mapping):
        return ["row 9: the probe output is not a JSON object"]
    problems: List[str] = []
    if raw.get("row") != "9":
        problems.append(f"row 9: the file says row {raw.get('row')!r}")
    if raw.get("registry") != registry:
        problems.append(f"row 9: asked registry {raw.get('registry')!r}, the single registry "
                        f"is {registry} — another server's E404 is not this one's")
    if raw.get("name") != name or raw.get("version") != version:
        problems.append(f"row 9: asked about {raw.get('name')}@{raw.get('version')}, the "
                        f"candidate is {name}@{version}")
    outcome = raw.get("outcome")
    if outcome not in ("absent", "published", "unknown"):
        problems.append(f"row 9: outcome {outcome!r} is not absent, published or unknown")
    if raw.get("pass") is not (outcome == "absent"):
        problems.append(f"row 9: pass {raw.get('pass')!r} disagrees with outcome {outcome!r}")
    if outcome == "published":
        problems.append(f"row 9: {name}@{version} is already on {registry}; npm forbids "
                        "republishing a version — bump skills/multi-llm/pyproject.toml")
    elif outcome != "absent" or raw.get("code") != "E404":
        problems.append(f"row 9: npm answered code {raw.get('code')!r} (exit "
                        f"{raw.get('npm_exit')!r}): only E404 is absence — anything else "
                        "is an UNKNOWN, not a green")
    if not isinstance(raw.get("npm_exit"), int) or isinstance(raw.get("npm_exit"), bool):
        problems.append("row 9: npm_exit is not an integer")
    return problems


def record_row(row_id: str, source: Path, directory: Path = EVIDENCE_DIR, *,
               tarball: Optional[Path] = None) -> Path:
    """``record 7|9 --from <file>``: check the CLI's file, then keep it as the
    session's row evidence. Nothing is written for a refused file.

    ``tarball`` is the gated tarball row 7 must have loaded (default:
    :func:`gated_tarball`)."""
    raw = _read_json(Path(source))
    if raw is None:
        raise GateError(f"row {row_id}: {source} is missing or not JSON")
    _require_session(directory)
    data = manifest()
    if row_id == "7":
        path = Path(tarball) if tarball else gated_tarball()
        facts = _builder().tarball_facts(path) if path.is_file() else None
        problems = row7_problems(raw, host_version=host_pin(), registry=single_registry(),
                                 tarball=facts, tarball_path=path)
        evidence = {"grader": "probe.py row7", "pass": True,
                    "opencode_version": raw.get("opencode_version"),
                    "registry": raw.get("registry"),
                    "tarball": {k: (facts or {}).get(k)
                                for k in ("path", "integrity", "sha256", "digest")},
                    "projects": raw.get("projects"), "probe": raw}
    elif row_id == "9":
        problems = row9_problems(raw, registry=single_registry(), name=str(data["name"]),
                                 version=str(data["version"]))
        evidence = {"grader": "registry_absent.mjs", "pass": True,
                    "registry": raw.get("registry"), "name": raw.get("name"),
                    "version": raw.get("version"), "code": raw.get("code"),
                    "npm_exit": raw.get("npm_exit"), "outcome": raw.get("outcome")}
    else:
        raise GateError(f"`record` takes row 7 or 9; row {row_id} has its own grader")
    if problems:
        raise GateError("\n".join(problems))
    return write_row(row_id, evidence, directory)


# ---------------------------------------------------------------------------
# Row 8's image binding: the payload digest, read out of the immutable image id
# ---------------------------------------------------------------------------

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _docker(argv: Sequence[str], runner: Runner, what: str) -> str:
    try:
        done = runner(["docker", *argv], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"{what}: `docker {' '.join(argv)}` could not run: {exc}") from exc
    if done.returncode != 0:
        raise GateError(f"{what}: `docker {' '.join(argv)}` exited {done.returncode}: "
                        f"{(done.stderr or done.stdout or '').strip()[:400]}")
    return done.stdout or ""


def resolve_image_id(reference: str, runner: Runner = subprocess.run) -> str:
    out = _docker(["image", "inspect", "--format", "{{.Id}}", reference], runner,
                  f"resolving image {reference}").strip()
    if not _IMAGE_ID.fullmatch(out):
        raise GateError(f"image {reference} resolved to {out!r}, not a sha256 image id")
    return out


def read_from_image(image_id: str, paths: Sequence[str],
                    runner: Runner = subprocess.run) -> Dict[str, bytes]:
    """Each path's bytes, copied out of a container **created** (never started)
    from the image id: nothing in the image executes, so neither its
    entrypoint nor its non-root user has a say in what is read."""
    container = _docker(["create", image_id], runner, f"reading {image_id}").strip()
    if not container:
        raise GateError(f"`docker create {image_id}` named no container")
    found: Dict[str, bytes] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="mllm-gate-image-") as scratch:
            for index, path in enumerate(paths):
                target = Path(scratch) / f"{index}"
                _docker(["cp", f"{container}:{path}", str(target)], runner,
                        f"reading {path} from {image_id}")
                try:
                    found[path] = target.read_bytes()
                except OSError as exc:
                    raise GateError(f"{path} could not be read out of {image_id}: "
                                    f"{exc}") from exc
    finally:
        try:
            runner(["docker", "rm", "-f", container], capture_output=True, text=True,
                   timeout=120)
        except (OSError, subprocess.SubprocessError):
            pass
    return found


def bind_image(tag: str, *, runner: Runner = subprocess.run,
               environ: Optional[Mapping[str, str]] = None,
               directory: Path = EVIDENCE_DIR, digest: Optional[str] = None,
               host_version: Optional[str] = None) -> Dict[str, Any]:
    """``image --image <tag>``, run right after the gate's own ``docker build``.

    Refuses, **before row 8 runs**:

    * an inherited ``MLLM_E2E_IMAGE`` that resolves to another image id (or to
      none) — the gate sets the variable itself, to the id it built, and an
      override from the maintainer's shell would otherwise decide which bytes
      row 8 grades;
    * an image whose ``dist/tree.sha256`` is not the candidate's. A clean
      ``git status`` does not see ignored files, so this comparison — not the
      tree check — is what binds the image's payload;
    * an image whose installed OpenCode is not the pinned host.
    """
    environ = os.environ if environ is None else environ
    session = _require_session(directory)
    image_id = resolve_image_id(tag, runner)
    inherited = (environ.get("MLLM_E2E_IMAGE") or "").strip()
    if inherited:
        try:
            inherited_id = resolve_image_id(inherited, runner)
        except GateError as exc:
            raise GateError(f"MLLM_E2E_IMAGE={inherited} is set in your environment and "
                            f"resolves to no image ({exc}); the gate runs row 8 on the image "
                            f"it built, {tag} = {image_id}. Unset it.") from exc
        if inherited_id != image_id:
            raise GateError(f"MLLM_E2E_IMAGE={inherited} is set in your environment and "
                            f"resolves to {inherited_id}, but the gate built {tag} = "
                            f"{image_id}. Row 8 runs on the gate's image only; unset it.")
    files = read_from_image(image_id, [IMAGE_PAYLOAD_DIGEST, IMAGE_INSTALLED_REPORT], runner)
    payload_digest = files[IMAGE_PAYLOAD_DIGEST].decode("utf-8", "replace").strip()
    expected = digest or candidate_digest()
    if payload_digest != expected:
        raise GateError(f"image {tag} ({image_id}) carries payload digest {payload_digest!r} "
                        f"at {IMAGE_PAYLOAD_DIGEST}, the candidate's is {expected}: the "
                        "image was built from other bytes, so row 8 would grade a "
                        "different package. Rebuild it from this clean checkout.")
    try:
        report = json.loads(files[IMAGE_INSTALLED_REPORT].decode("utf-8"))
    except ValueError as exc:
        raise GateError(f"{IMAGE_INSTALLED_REPORT} in {image_id} is not JSON ({exc})") from exc
    host = host_version or host_pin()
    opencode = next((cli for cli in (report.get("clis") or [])
                     if isinstance(cli, Mapping) and cli.get("id") == "opencode"), {})
    if opencode.get("installed") != host or opencode.get("status") != "ok":
        raise GateError(f"image {image_id} installed opencode {opencode.get('installed')!r} "
                        f"(status {opencode.get('status')!r}); the pinned host is {host}")
    evidence = {
        "session": session["id"], "tag": tag, "image_id": image_id,
        "payload_digest": payload_digest, "candidate_digest": expected,
        "candidate_sha": head_sha(), "opencode_installed": opencode.get("installed"),
        "installed_versions": report, "inspected_at": _iso(_now()),
    }
    path = Path(directory) / IMAGE_NAME
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


# ---------------------------------------------------------------------------
# Row 8: the five scenario records, graded against the session's image
# ---------------------------------------------------------------------------

def _records_for(row_id: str, since: str, records_dir: Path) -> List[Path]:
    """Every record of ``row_id`` started at or after ``since``, oldest first."""
    found = []
    for path in sorted(Path(records_dir).glob(f"*-{row_id}.json")):
        stamp = path.name[: -len(f"-{row_id}.json")]
        if _STAMP.fullmatch(stamp) and stamp >= since:
            found.append(path)
    return found


def _harness_markers() -> Tuple[str, ...]:
    if str(SKILL_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_DIR))
    from tests.container import harness  # noqa: WPS433 — the record vocabulary

    return (harness.PROVISIONAL_REASON, harness.UNGRADED_REASON)


def _under(path: Any, root: str) -> bool:
    if not isinstance(path, str) or not path.startswith("/"):
        return False
    candidate, base = PurePosixPath(path), PurePosixPath(root)
    return ".." not in candidate.parts and candidate != base \
        and base in candidate.parents


def verify_bundle(record: Path) -> Dict[str, Any]:
    """``make e2e-verify-record REC=<record>``'s script, run on one record.

    MATCH is the only green here: ``verify_record.py`` exits 0 for a MISSING
    bundle too (a pruned bundle is not tampering), but a scenario whose
    bundle is gone has no ``gate-grader.json`` to bind to it.
    """
    try:
        done = subprocess.run([sys.executable, str(VERIFY_RECORD), str(record)],
                              capture_output=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"exit": None, "verdict": "error", "output": str(exc)[:400]}
    output = (done.stdout + done.stderr).strip()
    verdict = next((word for word in ("MATCH", "MISSING", "MISMATCH")
                    if f"verify-record: {word}" in output), "error")
    return {"exit": done.returncode, "verdict": verdict,
            "output": output.splitlines()[0][:300] if output else ""}


def _grader_problems(row_id: str, grader: Any) -> List[str]:
    if not isinstance(grader, Mapping):
        return [f"{row_id}: ungraded — its bundle carries no readable gate-grader.json"]
    problems = []
    if grader.get("row") != row_id:
        problems.append(f"{row_id}: gate-grader.json names row {grader.get('row')!r}")
    for key in ("scenario", "grader"):
        if not isinstance(grader.get(key), str) or not grader.get(key):
            problems.append(f"{row_id}: gate-grader.json has no {key}")
    if not _is_true(grader.get("pass")):
        problems.append(f"{row_id}: its grader did not pass (pass: {grader.get('pass')!r})")
    if grader.get("route") != "package":
        problems.append(f"{row_id}: graded route {grader.get('route')!r}, row 8 is the "
                        "package route")
    if grader.get("installed_root") != INSTALLED_ROOT:
        problems.append(f"{row_id}: installed_root {grader.get('installed_root')!r}, the "
                        f"package is installed at {INSTALLED_ROOT}")
    if not _under(grader.get("dispatcher_skill_dir"), INSTALLED_ROOT):
        problems.append(f"{row_id}: the dispatcher ran from "
                        f"{grader.get('dispatcher_skill_dir')!r}, not from inside "
                        f"{INSTALLED_ROOT} — a file-route shim or a checkout path")
    checks = grader.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        problems.append(f"{row_id}: gate-grader.json records no checks")
    else:
        failed = sorted(name for name, value in checks.items() if not _is_true(value))
        if failed:
            problems.append(f"{row_id}: checks not passed: {failed}")
    if row_id == ROW8_ABORT_SCENARIO and (
            not isinstance(grader.get("abort_method"), str) or not grader.get("abort_method")):
        problems.append(f"{row_id}: the grader does not say how the abort was sent")
    return problems


def _posture_of(bundle: Path) -> Dict[str, Any]:
    """The two halves of the container posture assertions, as the row left them.

    ``posture-inside.json`` exists only when the entrypoint's
    ``assert_posture`` passed (it exits before writing otherwise);
    ``posture-host.json`` is ``harness.host_posture_mismatches`` over
    ``docker inspect`` — ``captured: false`` is a lost inspect race the harness
    records rather than fails.
    """
    host = _read_json(bundle / "posture-host.json")
    host = host if isinstance(host, Mapping) else {}
    return {"inside": (bundle / "posture-inside.json").is_file(),
            "host_captured": bool(host.get("captured")),
            "mismatches": list(host.get("mismatches") or [])}


def _posture_matches_head(root: Path = REPO_ROOT) -> Tuple[str, bool]:
    relative = POSTURE.relative_to(REPO_ROOT).as_posix()
    digest = hashlib.sha256(POSTURE.read_bytes()).hexdigest()
    try:
        done = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{relative}"],
                              capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return digest, False
    return digest, done.returncode == 0 and hashlib.sha256(done.stdout).hexdigest() == digest


def check_row8(since: str, *, directory: Path = EVIDENCE_DIR,
               records_dir: Path = RECORDS_DIR, container_dir: Path = CONTAINER_DIR,
               verifier: Callable[[Path], Dict[str, Any]] = verify_bundle,
               markers: Optional[Sequence[str]] = None,
               posture_head: Optional[Callable[[], Tuple[str, bool]]] = None,
               ) -> Dict[str, Any]:
    """Row 8's evidence from this session's run, or :class:`GateError` naming
    every scenario that is missing, skipped, ungraded, failing, or from
    another image or run."""
    if not _STAMP.fullmatch(since or ""):
        raise GateError(f"--since {since!r} is not a YYYYmmddTHHMMSSZ stamp; it is "
                        "compared lexically with the record names")
    session = _require_session(directory)
    if since < str(session.get("started_stamp") or ""):
        raise GateError(f"--since {since} reaches back before this gate session began "
                        f"({session.get('started_stamp')}); row 8 counts this session's "
                        "run only")
    image = _read_json(Path(directory) / IMAGE_NAME)
    if not isinstance(image, Mapping) or image.get("session") != session["id"]:
        raise GateError("no image-build entry in this gate session: row 8 cannot be bound "
                        "to an image. The gate builds its own (`image --image`) before "
                        "row 8 runs.")
    image_id = image.get("image_id")
    reasons = tuple(markers) if markers is not None else _harness_markers()
    problems: List[str] = []
    scenarios: Dict[str, Any] = {}
    posture: Dict[str, Any] = {}
    runs: set = set()
    for row_id in ROW8_SCENARIOS:
        found = _records_for(row_id, since, records_dir)
        if not found:
            problems.append(f"{row_id}: no run record at or after {since} — the scenario "
                            "never ran in this gate (missing, deselected or skipped)")
            continue
        path = found[-1]
        stamp = path.name[: -len(f"-{row_id}.json")]
        runs.add(stamp)
        record = _read_json(path) or {}
        entries = record.get("rows") or []
        entry = entries[-1] if entries and isinstance(entries[-1], Mapping) else {}
        if record.get("outcome") != "pass":
            word = "skipped (not run)" if record.get("outcome") == "not-run" else "failed"
            problems.append(f"{row_id}: {word} — the record's outcome is "
                            f"{record.get('outcome')!r}")
        if entry.get("row_id") != row_id:
            problems.append(f"{row_id}: the record's last row entry is "
                            f"{entry.get('row_id')!r}")
        elif entry.get("outcome") != "pass":
            problems.append(f"{row_id}: its row entry says {entry.get('outcome')!r}")
        reason = str(entry.get("reason") or "")
        if any(marker in reason for marker in reasons):
            problems.append(f"{row_id}: ungraded — {reason[:200]}")
        if entry.get("route") != "package":
            problems.append(f"{row_id}: ran route {entry.get('route', 'file')!r}, row 8 is "
                            "the package route")
        if "cli_version" in entry and entry["cli_version"] != image.get("opencode_installed"):
            problems.append(f"{row_id}: ran opencode {entry['cli_version']!r}, the image "
                            f"installed {image.get('opencode_installed')!r}")
        recorded_image = (record.get("image") or {}).get("digest")
        if recorded_image != image_id:
            problems.append(f"{row_id}: ran on image {recorded_image!r}, the gate built "
                            f"{image_id}")
        root = (record.get("bundle") or {}).get("root")
        if not root:
            problems.append(f"{row_id}: the record carries no bundle block, so no grader "
                            "file can be bound to it")
            continue
        bundle = Path(container_dir) / str(root)
        grader = _read_json(bundle / "gate-grader.json")
        problems += _grader_problems(row_id, grader)
        verified = verifier(path)
        if verified.get("exit") != 0 or verified.get("verdict") != "MATCH":
            problems.append(f"{row_id}: `make e2e-verify-record` says "
                            f"{verified.get('verdict')} (exit {verified.get('exit')}): "
                            f"{verified.get('output')}")
        posture[row_id] = _posture_of(bundle)
        if not posture[row_id]["inside"]:
            problems.append(f"{row_id}: no posture-inside.json — the in-container posture "
                            "assertions did not pass")
        if posture[row_id]["mismatches"]:
            problems.append(f"{row_id}: docker's record of the run disagrees with "
                            f"posture.yaml: {posture[row_id]['mismatches']}")
        grader = grader if isinstance(grader, Mapping) else {}
        scenarios[row_id] = {
            "scenario": grader.get("scenario"), "grader": grader.get("grader"),
            "pass": _is_true(grader.get("pass")), "run_id": stamp, "record": path.name,
            "outcome": record.get("outcome"), "graded": True,
            "image_id": recorded_image, "route": grader.get("route"),
            "dispatcher_skill_dir": grader.get("dispatcher_skill_dir"),
            "checks": dict(grader.get("checks") or {}),
            "verify_record": verified,
            **({"abort_method": grader.get("abort_method")}
               if row_id == ROW8_ABORT_SCENARIO else {}),
        }
    if len(runs) > 1:
        # One pytest module run stamps every scenario it drives with the same
        # start time, so records from two stamps are two runs mixed — possibly
        # another lane's, in this shared records directory.
        problems.append(f"row 8: the newest records come from {len(runs)} different runs "
                        f"({', '.join(sorted(runs))}); re-run the row")
    if posture and not any(item["host_captured"] for item in posture.values()):
        problems.append("row 8: `docker inspect` caught none of the scenario containers, so "
                        "the host half of the posture assertions never ran")
    posture_sha, posture_clean = (posture_head or _posture_matches_head)()
    if not posture_clean:
        problems.append("row 8: tests/container/posture.yaml differs from HEAD's; the "
                        "posture assertions run with it unchanged")
    if problems:
        raise GateError("row 8 is not green:\n" + "\n".join(f"    {line}" for line in problems))
    return {
        "grader": "check-row8", "pass": True, "image_id": image_id,
        "image_tag": image.get("tag"), "since": since, "run": next(iter(runs)),
        "scenarios": scenarios,
        "posture": {"pass": True, "posture_yaml_sha256": posture_sha,
                    "posture_yaml_matches_head": True, "scenarios": posture},
    }


# ---------------------------------------------------------------------------
# What the stamp requires, and what it keeps
# ---------------------------------------------------------------------------

def session_problems(session: Optional[Mapping[str, Any]]) -> List[str]:
    """The two refusals that need no build to state, so they lead."""
    if not session:
        return ["no gate session: run `make opencode-package-check`, which begins one "
                "before any row and writes every row's evidence into it"]
    if is_rehearsal(session):
        why = ("ALLOW_DIRTY=1" if session.get("allow_dirty") else
               "a dirty tree at `begin` (" + "; ".join(session.get("dirty_paths") or [])[:400] + ")")
        return [f"gate session {session.get('id')} is a development REHEARSAL ({why}): it "
                "may run every row, but it authorizes no stamp and no release-gate record. "
                "Re-run `make opencode-package-check` from a clean checkout of the release "
                "commit, without ALLOW_DIRTY."]
    return []


def evidence_problems(evidence: Mapping[str, Any], *, digest: str, version: str,
                      head: str, host_version: str, registry: str,
                      tarball: Optional[Mapping[str, Any]],
                      dirty: Sequence[str] = ()) -> List[str]:
    """Why ``evidence`` cannot back a stamp over these bytes, as readable lines.

    ``digest`` is the fresh clean build's, ``tarball`` the gated tarball's
    :func:`build.tarball_facts` recomputed now (``None``: it is missing),
    ``head`` the commit checked out now, ``dirty`` what ``git status`` shows
    now. Every row of :data:`ROW_IDS` must have passed in the running session
    over exactly these.
    """
    session = evidence.get("session")
    problems = session_problems(session)
    if not session:
        return problems
    sid = session["id"]
    if session.get("candidate_sha") != head:
        problems.append(f"the gate session began at commit {session.get('candidate_sha')}, "
                        f"HEAD is now {head}: a commit landed mid-gate; re-run it")
    if session.get("version") != version:
        problems.append(f"the gate session ran version {session.get('version')!r}, "
                        f"package.json says {version}")
    if dirty:
        problems.append("the tree has uncommitted changes now, so no commit holds what the "
                        "gate checked: " + "; ".join(dirty[:10]))
    rows = evidence.get("rows") or {}
    for row_id in ROW_IDS:
        item = rows.get(row_id)
        if not isinstance(item, Mapping):
            problems.append(f"row {row_id}: no evidence in this session "
                            f"({_BY_ID[row_id].grader})")
            continue
        if item.get("session") != sid:
            problems.append(f"row {row_id}: its evidence belongs to another gate session "
                            f"({item.get('session')!r}); every row must come from this run")
            continue
        if item.get("candidate_sha") != session.get("candidate_sha"):
            problems.append(f"row {row_id}: graded at commit {item.get('candidate_sha')}, the "
                            f"session's candidate is {session.get('candidate_sha')}")
        if not _is_true(item.get("pass")):
            problems.append(f"row {row_id}: did not pass")
        if row_id in CI_ROWS:
            problems += _ci_row_problems(row_id, item)
    row7 = rows.get("7") if isinstance(rows.get("7"), Mapping) else {}
    if row7:
        if row7.get("opencode_version") != host_version:
            problems.append(f"row 7: ran opencode {row7.get('opencode_version')!r}, the "
                            f"pinned host is {host_version}")
        if row7.get("registry") != registry:
            problems.append(f"row 7: registry {row7.get('registry')!r}, the single registry "
                            f"is {registry}")
    if tarball is None:
        problems.append("the gated tarball is missing: the stamp binds the file "
                        "`make opencode-package` packed, scanned and row 7 loaded")
    else:
        if tarball.get("digest") != digest or tarball.get("recorded_digest") != digest:
            problems.append(f"the gated tarball's tree digest is {tarball.get('digest')!r} "
                            f"(it records {tarball.get('recorded_digest')!r}); the candidate "
                            f"is {digest}")
        loaded = row7.get("tarball") or {}
        if row7 and loaded.get("integrity") != tarball.get("integrity"):
            problems.append(f"the gated tarball's integrity is now {tarball.get('integrity')}, "
                            f"row 7 loaded {loaded.get('integrity')!r}: it was repacked")
    image = evidence.get("image")
    if not isinstance(image, Mapping):
        problems.append("no image-build entry in this session: row 8 is bound to the "
                        "image the gate built, and none was recorded")
    else:
        if image.get("session") != sid:
            problems.append("the image entry belongs to another gate session")
        if not _IMAGE_ID.fullmatch(str(image.get("image_id") or "")):
            problems.append(f"the image entry names no image id ({image.get('image_id')!r})")
        if image.get("payload_digest") != digest:
            problems.append(f"the image's payload digest {image.get('payload_digest')!r} is "
                            f"not the candidate's {digest}")
        if image.get("opencode_installed") != host_version:
            problems.append(f"the image installed opencode {image.get('opencode_installed')!r}, "
                            f"the pinned host is {host_version}")
    row8 = rows.get("8") if isinstance(rows.get("8"), Mapping) else {}
    if row8:
        if isinstance(image, Mapping) and row8.get("image_id") != image.get("image_id"):
            problems.append(f"row 8: its records name image {row8.get('image_id')!r}, the "
                            f"gate built {image.get('image_id')!r}")
        scenarios = row8.get("scenarios") or {}
        for scenario in ROW8_SCENARIOS:
            item = scenarios.get(scenario)
            if not isinstance(item, Mapping) or not _is_true(item.get("pass")) \
                    or not _is_true(item.get("graded")):
                problems.append(f"row 8: scenario {scenario} has no passing, graded record")
            elif item.get("image_id") != row8.get("image_id"):
                problems.append(f"row 8: scenario {scenario} ran on image {item.get('image_id')!r}")
        if len({(scenarios.get(s) or {}).get("run_id") for s in ROW8_SCENARIOS}) > 1:
            problems.append("row 8: its scenarios come from different runs")
        if not _is_true((row8.get("posture") or {}).get("pass")):
            problems.append("row 8: the posture assertions did not pass")
    row9 = rows.get("9") if isinstance(rows.get("9"), Mapping) else {}
    if row9:
        if row9.get("registry") != registry:
            problems.append(f"row 9: asked {row9.get('registry')!r}, the single registry is "
                            f"{registry}")
        if row9.get("version") != version or row9.get("code") != "E404":
            problems.append(f"row 9: {row9.get('name')}@{row9.get('version')} answered "
                            f"{row9.get('code')!r}; the stamp needs E404 for {version}")
    return problems


def stamp_summary(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """What the stamp embeds of a session that passed :func:`evidence_problems`:
    enough to say which run authorized it, on which image and host, without
    the stamp becoming a second copy of the evidence."""
    session = evidence["session"]
    rows = evidence["rows"]
    image = evidence["image"]
    return {
        "session": session["id"], "started_at": session.get("started_at"),
        "candidate_sha": session.get("candidate_sha"),
        "rows": {row_id: "pass" for row_id in ROW_IDS},
        "ci_nodes": {row_id: sorted((rows[row_id].get("nodes") or {})) for row_id in CI_ROWS},
        "row7": {"opencode_version": rows["7"].get("opencode_version"),
                 "registry": rows["7"].get("registry"),
                 "projects": sorted(p.get("kind") for p in rows["7"].get("projects") or [])},
        "image": {"id": image.get("image_id"), "tag": image.get("tag"),
                  "payload_digest": image.get("payload_digest")},
        "row8": {"run": rows["8"].get("run"), "ran": "local",
                 "scenarios": {s: (rows["8"].get("scenarios") or {}).get(s, {}).get("grader")
                               for s in ROW8_SCENARIOS}},
        "row9": {"registry": rows["9"].get("registry"), "code": rows["9"].get("code")},
    }


def describe(evidence: Mapping[str, Any]) -> List[str]:
    """One line per row, in words that never call an absence a pass."""
    session = evidence.get("session") or {}
    lines = [f"gate session {session.get('id', '<none>')} at "
             f"{session.get('candidate_sha', '?')}"
             + (" — REHEARSAL, authorizes nothing" if session and is_rehearsal(session) else "")]
    rows = evidence.get("rows") or {}
    for row in ROWS:
        item = rows.get(row.id)
        if not isinstance(item, Mapping):
            lines.append(f"row {row.id}: no evidence")
            continue
        state = "pass" if _is_true(item.get("pass")) else "FAIL"
        other = "" if item.get("session") == session.get("id") else " (another session)"
        lines.append(f"row {row.id}: {state}{other} — {row.proves[:90]}")
    image = evidence.get("image") or {}
    lines.append(f"image: {image.get('image_id', 'no image-build entry')}")
    return lines


# ---------------------------------------------------------------------------
# The release-gate record: row 8's local evidence, carried to CI
# ---------------------------------------------------------------------------

def _tarball_facts_now(path: Path) -> Optional[Dict[str, Any]]:
    return _builder().tarball_facts(path) if Path(path).is_file() else None


def build_record(evidence: Mapping[str, Any], stamp: Mapping[str, Any],
                 exported: datetime.datetime) -> Dict[str, Any]:
    session = evidence["session"]
    rows = evidence["rows"]
    image = evidence["image"]
    row8 = rows["8"]
    tarball = stamp["tarball"]
    return {
        "schema": 1,
        "kind": "opencode-release-gate",
        "exported_at": _compact(exported),
        "tag": release_tag(stamp["version"]),
        "sha": session["candidate_sha"],
        "version": stamp["version"],
        "candidate_digest": stamp["digest"],
        "tarball": {"name": Path(str(tarball["path"])).name,
                    "integrity": tarball["integrity"], "sha256": tarball["sha256"],
                    "file_count": len(tarball.get("files") or [])},
        "registry": stamp["registry"],
        "host_version": stamp["host_version"],
        "image": {"id": image["image_id"], "tag": image.get("tag"),
                  "payload_digest": image["payload_digest"],
                  "opencode_installed": image.get("opencode_installed"),
                  "installed_versions": image.get("installed_versions") or {}},
        "posture": row8["posture"],
        "session": session["id"],
        "rehearsal": False,
        "rows": {row_id: "pass" for row_id in ROW_IDS},
        "row8": {"ran": "local", "run_id": row8.get("run"), "since": row8.get("since")},
        "scenarios": [
            {"row": scenario, "session": session["id"],
             **{key: (row8["scenarios"][scenario]).get(key)
                for key in ("scenario", "grader", "pass", "run_id", "record", "outcome",
                            "graded", "image_id", "verify_record")}}
            for scenario in ROW8_SCENARIOS
        ],
    }


def export_record(*, directory: Path = EVIDENCE_DIR, records_dir: Optional[Path] = None,
                  stamp_path: Optional[Path] = None, tarball: Optional[Path] = None,
                  now: Optional[datetime.datetime] = None) -> Path:
    """Write ``<UTC>-opencode-release-gate.json`` from the session that
    authorized the stamp, then validate it with the workflow's own validator.

    Refuses a rehearsal by name, a session the stamp was not written from, and
    anything :func:`evidence_problems` still finds — the record is never more
    than the stamp vouched for, and it is never written by hand.
    """
    evidence = load_evidence(directory)
    early = session_problems(evidence.get("session"))
    if early:
        raise GateError("\n".join(early))
    session = evidence["session"]
    stamp_file = Path(stamp_path) if stamp_path else PACKAGE_DIR / STAMP_NAME
    stamp = _read_json(stamp_file)
    if not isinstance(stamp, Mapping) or stamp.get("schema") != 2:
        raise GateError(f"no schema-2 stamp at {stamp_file}: the record is exported from "
                        "the session that authorized a stamp, after `--write-stamp`")
    if stamp.get("session") != session["id"]:
        raise GateError(f"the stamp was written by gate session {stamp.get('session')!r}, the "
                        f"evidence is session {session['id']}; re-run the full gate")
    gated = Path(tarball) if tarball else Path(str(stamp["tarball"]["path"]))
    host = host_pin()
    registry = single_registry()
    problems = evidence_problems(
        evidence, digest=str(stamp.get("digest")), version=package_version(),
        head=head_sha(), host_version=host, registry=registry,
        tarball=_tarball_facts_now(gated), dirty=dirty_paths())
    for key, now_value in (("version", package_version()), ("host_version", host),
                           ("registry", registry), ("candidate_sha", session["candidate_sha"])):
        if stamp.get(key) != now_value:
            problems.append(f"the stamp's {key} {stamp.get(key)!r} is not {now_value!r}")
    if problems:
        raise GateError("the session does not support a release-gate record:\n"
                        + "\n".join(f"    {line}" for line in problems))
    exported = now or _now()
    record = build_record(evidence, stamp, exported)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    for private in {str(REPO_ROOT), str(Path.home())}:
        if private and private != "/" and private in text:
            raise GateError(f"the record would carry the local path {private}; it is "
                            "committed, so it must name none")
    records = Path(records_dir) if records_dir else Path(directory) / EXPORT_SUBDIR
    records.mkdir(parents=True, exist_ok=True)
    path = records / f"{record['exported_at']}{RECORD_SUFFIX}"
    if path.exists():
        raise GateError(f"{path} already exists; wait a second and export again")
    path.write_text(text, encoding="utf-8")
    try:
        validate_records(tag=record["tag"], sha=record["sha"], version=record["version"],
                         digest=record["candidate_digest"], host_version=host,
                         registry=registry, records_dir=records)
    except GateError:
        path.unlink()
        raise
    return path


# -- The validator. Stdlib only: the `verify` job runs it under --no-dev. -----

#: The JSON Schema keywords :func:`schema_errors` implements. The suite holds
#: the record schema to this set, so a schema edit cannot quietly lean on a
#: keyword the CI validator would ignore.
SCHEMA_KEYWORDS = frozenset({
    "$schema", "$id", "title", "description", "type", "const", "enum", "pattern",
    "minLength", "minimum", "required", "properties", "additionalProperties",
    "items", "minItems",
})

_TYPES: Dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_errors(node: Any, schema: Mapping[str, Any], path: str = "") -> List[str]:
    """Every way ``node`` disagrees with ``schema``, each naming its field."""
    where = path or "<root>"
    bad: List[str] = []
    expected = schema.get("type")
    if expected is not None:
        kinds = expected if isinstance(expected, list) else [expected]
        if not any(_TYPES[kind](node) for kind in kinds):
            return [f"{where}: {json.dumps(node)[:80]} is not of type {'/'.join(kinds)}"]
    if "const" in schema and (node != schema["const"]
                              or type(node) is not type(schema["const"])):
        bad.append(f"{where}: {json.dumps(node)[:80]} is not {json.dumps(schema['const'])}")
    if "enum" in schema and node not in schema["enum"]:
        bad.append(f"{where}: {json.dumps(node)[:80]} is not one of {schema['enum']}")
    if isinstance(node, str):
        if "pattern" in schema and not re.search(schema["pattern"], node):
            bad.append(f"{where}: {node[:80]!r} does not match {schema['pattern']}")
        if len(node) < int(schema.get("minLength", 0)):
            bad.append(f"{where}: shorter than {schema['minLength']}")
    if _TYPES["number"](node) and "minimum" in schema and node < schema["minimum"]:
        bad.append(f"{where}: {node} is below {schema['minimum']}")
    if isinstance(node, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in node:
                bad.append(f"{path + '/' if path else ''}{key}: missing")
        extra = schema.get("additionalProperties", True)
        for key, value in node.items():
            child = f"{path}/{key}" if path else str(key)
            if key in properties:
                bad += schema_errors(value, properties[key], child)
            elif extra is False:
                bad.append(f"{child}: not a field this schema allows")
            elif isinstance(extra, Mapping):
                bad += schema_errors(value, extra, child)
    if isinstance(node, list):
        if len(node) < int(schema.get("minItems", 0)):
            bad.append(f"{where}: fewer than {schema['minItems']} items")
        if isinstance(schema.get("items"), Mapping):
            for index, value in enumerate(node):
                bad += schema_errors(value, schema["items"], f"{path}[{index}]")
    return bad


def load_record_schema() -> Dict[str, Any]:
    return json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))


def _record_problems(record: Mapping[str, Any], name: str, *, tag: str, sha: str,
                     version: str, digest: str, host_version: str,
                     registry: str) -> List[str]:
    if record.get("rehearsal") is not False:
        return [f"rehearsal: {name} says rehearsal {record.get('rehearsal')!r}; a "
                "rehearsal record is never release evidence"]
    problems = schema_errors(record, load_record_schema())
    if problems:
        return [f"schema: {line}" for line in problems]
    stamp = name[: -len(RECORD_SUFFIX)]
    if record["exported_at"] != stamp:
        problems.append(f"exported_at: {record['exported_at']} is not the file name's {stamp}")
    if record["tag"] != release_tag(record["version"]):
        problems.append(f"tag: {record['tag']} does not name version {record['version']}")
    for field, expected, against in (
            ("sha", sha, "the tag's commit"), ("version", version, "its package.json"),
            ("candidate_digest", digest, "the rebuilt candidate"),
            ("host_version", host_version, "the pinned host"),
            ("registry", registry, "the single registry")):
        if record[field] != expected:
            problems.append(f"{field}: the record says {record[field]!r}, {against} is "
                            f"{expected!r}")
    image = record["image"]
    if not _IMAGE_ID.fullmatch(str(image.get("id") or "")):
        problems.append(f"image.id: missing or not an image id ({image.get('id')!r})")
    if image.get("payload_digest") != digest:
        problems.append(f"image.payload_digest: {image.get('payload_digest')!r} is not the "
                        f"candidate digest {digest}")
    if image.get("opencode_installed") != host_version:
        problems.append(f"image.opencode_installed: {image.get('opencode_installed')!r}, the "
                        f"pinned host is {host_version}")
    if not _is_true((record.get("posture") or {}).get("pass")):
        problems.append("posture.pass: the posture assertions did not pass")
    seen: Dict[str, int] = {}
    runs = set()
    for index, scenario in enumerate(record["scenarios"]):
        row = scenario.get("row")
        field = f"scenarios[{index}] ({row})"
        seen[row] = seen.get(row, 0) + 1
        if row not in ROW8_SCENARIOS:
            problems.append(f"{field}: not a row-8 scenario")
            continue
        if scenario.get("session") != record["session"]:
            problems.append(f"{field}.session: {scenario.get('session')!r}, the record's "
                            f"session is {record['session']} — scenarios from different "
                            "sessions")
        if scenario.get("image_id") != image.get("id"):
            problems.append(f"{field}.image_id: {scenario.get('image_id')!r}, the record's "
                            f"image is {image.get('id')!r} — scenarios from different images")
        runs.add(scenario.get("run_id"))
        if scenario.get("outcome") != "pass":
            word = "skipped" if scenario.get("outcome") in ("not-run", "skip") else "failed"
            problems.append(f"{field}.outcome: {scenario.get('outcome')!r} — {word}")
        if scenario.get("graded") is not True or not scenario.get("grader"):
            problems.append(f"{field}.graded: ungraded (graded {scenario.get('graded')!r}, "
                            f"grader {scenario.get('grader')!r})")
        if scenario.get("pass") is not True:
            problems.append(f"{field}.pass: {scenario.get('pass')!r} is not boolean true")
        verified = scenario.get("verify_record") or {}
        if verified.get("exit") != 0 or verified.get("verdict") != "MATCH":
            problems.append(f"{field}.verify_record: {verified.get('verdict')!r} (exit "
                            f"{verified.get('exit')!r}); its archived bundle must MATCH")
    for row in ROW8_SCENARIOS:
        if row not in seen:
            problems.append(f"scenarios: {row} is missing")
        elif seen[row] > 1:
            problems.append(f"scenarios: {row} appears {seen[row]} times")
    if len(runs) > 1:
        problems.append(f"scenarios.run_id: the scenarios come from {len(runs)} runs")
    return problems


def validate_records(*, tag: str, sha: str, version: str, digest: str,
                     host_version: str, registry: str,
                     records_dir: Path = RECORDS_DIR) -> Tuple[Path, Dict[str, Any]]:
    """The newest committed record naming ``tag``, validated, or
    :class:`GateError` naming every offending field.

    **Only the newest counts.** An older record naming the same tag is never a
    fallback: if the newest is wrong, the release is refused, because a newer
    record is what the maintainer last exported and an older valid one beside
    it would otherwise publish on evidence they superseded. A record left from
    an earlier candidate fails the SHA and digest checks; that is how a stale
    record is caught. To recover, re-run step 3 at the tag's commit and commit
    a fresh record — never edit one to match.
    """
    problems: List[str] = []
    registry = str(registry).strip().rstrip("/")
    try:
        expected_registry = single_registry()
        if registry != expected_registry:
            problems.append(f"registry: --registry {registry} is not the single registry "
                            f"{expected_registry}")
    except Exception as exc:  # noqa: BLE001 — any failure to read it refuses
        problems.append(f"registry: the single registry could not be read ({exc})")
    records = Path(records_dir)
    named: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in sorted(records.glob(f"*{RECORD_SUFFIX}")):
        stamp = path.name[: -len(RECORD_SUFFIX)]
        if not _STAMP.fullmatch(stamp):
            problems.append(f"{path.name}: not a <YYYYmmddTHHMMSSZ>{RECORD_SUFFIX} name, so "
                            "it cannot be ordered against the others")
            continue
        record = _read_json(path)
        if not isinstance(record, dict):
            problems.append(f"{path.name}: unreadable, so nothing can say whether it names "
                            f"{tag} — and it might be the newest that does")
            continue
        if record.get("tag") == tag:
            named.append((stamp, path, record))
    if not named:
        problems.append(f"tag: no committed release-gate record names {tag} under "
                        f"{records}. Run step 3 (`make opencode-package-check`) at the tag's "
                        "commit, commit the exported record to the default branch, and "
                        "dispatch again.")
        raise GateError("\n".join(problems))
    # Names are `<stamp><suffix>` in one directory, so no two share a stamp.
    named.sort(key=lambda item: item[0])
    stamp, path, record = named[-1]
    problems += _record_problems(record, path.name, tag=tag, sha=sha, version=version,
                                 digest=digest, host_version=host_version, registry=registry)
    if problems:
        older = len(named) - 1
        raise GateError(f"{path.name} (the newest of {len(named)} record(s) naming {tag}; "
                        f"{older} older one(s) never count as a fallback) is refused:\n"
                        + "\n".join(f"    {line}" for line in problems))
    return path, record


# ---------------------------------------------------------------------------
# One bounded gate step, with its log kept
# ---------------------------------------------------------------------------

def _kill(child: "subprocess.Popen[bytes]", grace: float) -> None:
    """The child's whole process group, TERM then KILL. It leads its own
    session (``start_new_session``), so every descendant that did not leave it
    goes too."""
    def signal_group(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(child.pid, sig)
            elif sig == signal.SIGTERM:
                child.terminate()
            else:
                child.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    signal_group(signal.SIGTERM)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def run_step(name: str, argv: Sequence[str], *, timeout: float,
             capture: Optional[Path] = None, directory: Path = EVIDENCE_DIR,
             grace: float = 10.0) -> int:
    """Run one gate step under a wall-clock bound, its output tee'd to
    ``.gate/logs/<name>.log``.

    **Its own session, with no terminal.** The child's stdin is ``/dev/null``
    and it leads a new session, so a timeout kills every descendant in one
    ``killpg``. That is also what makes the bound safe for ``opencode run``,
    which a plain ``timeout N`` wrapper wedges before it creates a session
    (measured: ``timeout`` moves the child into a background process group of
    the same terminal; with no controlling terminal there is nothing to be
    stopped by). ``capture`` keeps the last non-empty stdout line — the
    host-install path, the image id — for the recipe's next line, since each
    recipe line is its own shell.
    """
    logs = Path(directory) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    started = time.monotonic()
    last_stdout: List[str] = []
    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(argv)}\n")
        try:
            child = subprocess.Popen(
                list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=(os.name == "posix"),
                close_fds=False)
        except OSError as exc:
            print(f"release gate: step {name} could not start {argv[0]!r}: {exc}",
                  file=sys.stderr)
            return 127

        def pump(stream, sink, is_stdout: bool) -> None:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace")
                with lock:
                    log.write(text)
                    log.flush()
                    sink.write(text)
                    sink.flush()
                    if is_stdout and text.strip():
                        last_stdout[:] = [text.strip()]

        threads = [threading.Thread(target=pump, args=(child.stdout, sys.stdout, True),
                                    daemon=True),
                   threading.Thread(target=pump, args=(child.stderr, sys.stderr, False),
                                    daemon=True)]
        for thread in threads:
            thread.start()
        code: Optional[int]
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(child, grace)
            code = None
        except KeyboardInterrupt:
            _kill(child, grace)
            raise
        for thread in threads:
            thread.join(timeout=grace)
    elapsed = time.monotonic() - started
    if code is None:
        print(f"release gate: step {name} exceeded its {timeout:g} s bound and was killed "
              f"with its process group; its output is in {log_path}", file=sys.stderr)
        return 124
    if code != 0:
        print(f"release gate: step {name} failed (exit {code}) after {elapsed:.0f} s; its "
              f"output is in {log_path}", file=sys.stderr)
        return code
    if capture is not None:
        if not last_stdout:
            print(f"release gate: step {name} printed nothing on stdout to capture into "
                  f"{capture}; its output is in {log_path}", file=sys.stderr)
            return 1
        Path(capture).parent.mkdir(parents=True, exist_ok=True)
        Path(capture).write_text(last_stdout[0] + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stamp_check(directory: Path = EVIDENCE_DIR) -> List[str]:
    """``stamp-check``: :func:`evidence_problems` over the bytes on disk now —
    the check ``build.py --write-stamp`` runs, without its rebuild."""
    evidence = load_evidence(directory)
    early = session_problems(evidence.get("session"))
    if early:
        return early
    return evidence_problems(
        evidence, digest=candidate_digest(), version=package_version(), head=head_sha(),
        host_version=host_pin(), registry=single_registry(),
        tarball=_tarball_facts_now(gated_tarball()), dirty=dirty_paths())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="release_gate.py",
                                     description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evidence-dir", type=Path, default=EVIDENCE_DIR,
        help="the session's evidence directory (default: packaging/opencode/.gate). "
             "`build.py --write-stamp` reads the default only.")
    sub = parser.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("begin", help="start a gate session: delete earlier evidence")
    begin.add_argument("--allow-dirty", action="store_true",
                       help="a development rehearsal: rows run, nothing is authorized")
    sub.add_parser("rows", help="print the declared rows")
    grade = sub.add_parser("grade-ci", help="grade rows 1-6 and 5b from pytest junit XML")
    grade.add_argument("--junit", action="append", required=True, type=Path)
    record = sub.add_parser("record", help="check a row-7 or row-9 CLI's file; keep it")
    record.add_argument("row", choices=("7", "9"))
    record.add_argument("--from", dest="source", required=True, type=Path)
    image = sub.add_parser("image", help="bind the gate's image: id, payload digest")
    image.add_argument("--image", required=True)
    row8 = sub.add_parser("check-row8", help="grade row 8's scenario records")
    row8.add_argument("--since", required=True, help="YYYYmmddTHHMMSSZ, before the run")
    sub.add_parser("stamp-check", help="what --write-stamp would refuse, now")
    sub.add_parser("summary", help="describe the evidence on disk")
    export = sub.add_parser("export-record", help="write the release-gate record from this session")
    export.add_argument("--into", type=Path, default=None,
                        help=f"where to write it (default: {EXPORT_DIR}, gitignored)")
    validate = sub.add_parser("validate-record", help="the workflow's record validator")
    for flag in ("--tag", "--sha", "--version", "--digest", "--host-version", "--registry"):
        validate.add_argument(flag, required=True)
    validate.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    step = sub.add_parser("step", help="run one bounded gate step, keeping its log",
                          usage="release_gate.py step NAME --timeout S [--capture F] -- CMD...")
    step.add_argument("name")
    step.add_argument("--timeout", type=float, required=True)
    step.add_argument("--capture", type=Path, default=None)
    words = list(argv) if argv is not None else sys.argv[1:]
    # The step's command is everything after the first `--`, split off before
    # argparse sees it: a REMAINDER positional would swallow `--timeout` too.
    command: List[str] = []
    if "step" in words and "--" in words[words.index("step"):]:
        cut = words.index("--", words.index("step"))
        words, command = words[:cut], words[cut + 1:]
    args = parser.parse_args(words)
    directory = args.evidence_dir
    try:
        if args.command == "begin":
            session = begin_session(directory, allow_dirty=args.allow_dirty)
            print(f"gate session {session['id']} at {session['candidate_sha']} "
                  "(earlier evidence deleted)")
            for line in session_problems(session):
                print(f"NOTE: {line}", file=sys.stderr)
        elif args.command == "rows":
            for row in ROWS:
                print(f"row {row.id}: {row.proves}\n  grader: {row.grader}\n  evidence: "
                      f"{row.evidence}" + "".join(f"\n    {node}" for node in row.nodes))
        elif args.command == "grade-ci":
            verdicts = grade_ci(args.junit)
            session = session_for_grading(directory)
            failing = []
            for row_id, verdict in verdicts.items():
                problems = _ci_row_problems(row_id, verdict)
                print(f"row {row_id}: {'pass' if not problems else 'FAIL'} "
                      f"({len(verdict['nodes'])} node(s))")
                for line in problems:
                    print(f"    {line}", file=sys.stderr)
                if problems:
                    failing.append(row_id)
                if session is not None:
                    write_row(row_id, verdict, directory)
            if session is None:
                print("grade-ci: no gate session for HEAD; graded only, no evidence written")
            if failing:
                print(f"REFUSING: row(s) {', '.join(failing)} did not pass; a skip, a "
                      "deselection or an empty selection never satisfies a row",
                      file=sys.stderr)
                return 1
        elif args.command == "record":
            path = record_row(args.row, args.source, directory)
            print(f"row {args.row}: evidence {path}")
        elif args.command == "image":
            evidence = bind_image(args.image, directory=directory)
            print(f"image {args.image} = {evidence['image_id']}: payload digest "
                  f"{evidence['payload_digest']} is the candidate's", file=sys.stderr)
            print(evidence["image_id"])
        elif args.command == "check-row8":
            evidence = check_row8(args.since, directory=directory)
            path = write_row("8", evidence, directory)
            print(f"row 8: {len(evidence['scenarios'])} scenarios pass on "
                  f"{evidence['image_id']} (run {evidence['run']}); evidence {path}")
        elif args.command == "stamp-check":
            problems = _stamp_check(directory)
            if problems:
                for line in problems:
                    print(f"REFUSING: {line}", file=sys.stderr)
                return 1
            print("the session's evidence authorizes a stamp over the bytes on disk")
        elif args.command == "summary":
            for line in describe(load_evidence(directory)):
                print(line)
        elif args.command == "export-record":
            path = export_record(directory=directory,
                                 records_dir=args.into)
            target = RECORDS_DIR.relative_to(REPO_ROOT).as_posix()
            print(f"wrote {path} (validated). After the fast-forward (plan step 2), from a "
                  f"checkout of the default branch at the candidate:\n"
                  f"    cp {path} {target}/\n"
                  f"    git add {target}/{path.name}\n"
                  f"    git commit -m \"Record the opencode release gate for "
                  f"<tag>\" && git push origin HEAD\n"
                  "Then check the tag's commit out again before `make opencode-publish-first`.")
        elif args.command == "validate-record":
            path, record = validate_records(
                tag=args.tag, sha=args.sha, version=args.version, digest=args.digest,
                host_version=args.host_version, registry=args.registry,
                records_dir=args.records_dir)
            print(f"release-gate record {path.name} validates: {record['tag']} at "
                  f"{record['sha']}, {len(record['scenarios'])} row-8 scenarios on "
                  f"{record['image']['id']} (session {record['session']}, ran locally)")
        elif args.command == "step":
            if not command:
                parser.error("step needs a command after --")
            return run_step(args.name, command, timeout=args.timeout, capture=args.capture,
                            directory=directory)
    except GateError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
