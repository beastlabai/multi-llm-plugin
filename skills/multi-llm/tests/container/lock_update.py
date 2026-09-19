#!/usr/bin/env python3
"""Move the pins a `@latest` build PROVED have moved -- and only those (C9).

``make e2e-lock-update`` is the writer behind the drift probe. Its input is not a
changelog and not a human's reading of a report: it is the evidence a
``MLLM_PIN_MODE=latest`` image carries inside itself, extracted with ``docker cp``
and consumed here.

**THE SCOPE IS DECIDED BY WHAT THE IMAGE CAN PROVE, NOT BY WHAT WOULD BE
CONVENIENT.** Every row below is either rewritten with its proof named, or
refused with the command that would produce the proof it lacks. A pin this script
cannot substantiate stays exactly where it is:

======================  =========================================================
row                     what the @latest image proves, and what it does not
======================  =========================================================
``install: npm``        REWRITTEN. ``/opt/mllm/npm/package-lock.json`` is the
                        lockfile ``npm install <pkg>@latest`` actually resolved:
                        exact versions plus an ``integrity`` hash for the whole
                        transitive tree. Copying it out *is* the pin move; the
                        version in ``cli-versions.lock`` and ``package.json``
                        follows it, and the census's own ``installed`` value has
                        to agree with it before anything is written.
``install: download``   REWRITTEN. The Dockerfile installs these with
                        ``install -m 0755 /tmp/<x> /opt/mllm/real/<x>``, so the
                        file in the image *is* the downloaded artifact, byte for
                        byte -- confirmed against the pinned image, whose
                        ``/opt/mllm/real/grok`` hashes to exactly the ``sha256``
                        this lock names. So the new checksum is recoverable by
                        hashing the file, and the URL by substituting the version
                        inside the one the lock already carries.
``install: archive``    REFUSED. ``goose`` is unpacked from a ``.tar.bz2`` that
                        the image does not keep: ``/opt/mllm/real/goose`` is the
                        extracted binary and hashes to something that is not the
                        artifact's checksum (measured: ``19f40a9c…`` against the
                        recorded ``416de595…``). A pin is the bytes we fetch, and
                        those bytes are gone. Named, with the command.
``install: pypi``       REFUSED. ``aider``'s pin is a whole ``--generate-hashes``
                        requirements file. The image has the installed versions
                        but not the hashes, which come from a resolver run
                        against the index. Named, with the command.
base image, uv,         REFUSED, and not because of missing evidence: the
platform                ``latest`` build holds them pinned on purpose, so it can
                        never be the thing that moved them. They change by hand,
                        in a reviewable commit.
``cli-surface.yaml``    NOT TOUCHED. It is a contract, not a pin, and its own
                        probe reports a changed help digest as *"re-seed once
                        reviewed"* -- an act of review, not of bookkeeping.
                        Re-seeding it here would silently discard the one
                        upstream-facing signal in the design. The re-seed command
                        is printed against the rebuilt pinned image instead.
======================  =========================================================

**Nothing is written until every row is decided.** A half-moved npm layer is a
``package-lock.json`` that disagrees with ``package.json`` -- ``npm ci`` refuses
it and the next pinned build fails for a reason that has nothing to do with the
CLI that moved. So the whole npm layer moves or none of it does.

**The rewrite is a text edit, and then it is checked as a document.**
``cli-versions.lock`` is 270 lines of which ~150 are the reasoning behind the
pins; round-tripping it through PyYAML would delete all of it. So the edit is
line-scoped -- and afterwards the result is re-parsed and compared against the
document we *intended*, key by key, so "the line editor hit the wrong row" is a
failure here rather than a surprise three builds later.

**And the record is verified, not merely written.** ``apply`` emits a record of
every field it moved, with the before/after digest of every file it touched;
``verify`` re-reads that record against the census of the *rebuilt pinned image*
and fails unless every new pin is what that image really installed. The Makefile
runs both halves, which is what makes this target a proof rather than an edit.

Exit codes: ``0`` rewritten (or verified), ``2`` nothing to move, ``1`` refused
or failed -- nothing written in either failing case.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

#: The evidence layout `make e2e-lock-update` writes with `docker cp`. Named in
#: one place so the makefile recipe and this script cannot drift.
CENSUS_NAME = "cli-versions.installed.json"
NPM_SUBDIR = "npm"
ARTIFACT_SUBDIR = "artifacts"

#: Files this script is allowed to rewrite. The list is closed on purpose: it is
#: also the revert instruction printed when the verification step fails.
WRITABLE = ("cli-versions.lock", "package.json", "package-lock.json", "Dockerfile")


class Refusal(Exception):
    """A pin that cannot be substantiated. Nothing is written."""


class LayerRefusal(Refusal):
    """One *layer* cannot move, while the rest of the run still can.

    The npm layer is all-or-nothing by the format's own construction -- a
    lockfile describes one resolved tree, so there is no such thing as taking
    ``codex`` out of the ``@latest`` tree while leaving ``pi`` at the version that
    tree does not contain. But the artifact rows are independent files, and
    stopping them because npm could not move would report less than the evidence
    supports. So an npm-layer problem is recorded as a refusal with its reason and
    the run continues.
    """

    def __init__(self, message: str, command: str = ""):
        super().__init__(message)
        self.command = command


# --------------------------------------------------------------------------- #
# Digests and small readers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Change:
    """One field moved, and the evidence that moved it."""

    file: str
    target: str
    old: str
    new: str
    evidence: str


@dataclass
class Refused:
    """One pin left alone, with the reason and the command that would move it."""

    row: str
    install: str
    pinned: str
    installed: str
    reason: str
    command: str


@dataclass
class Plan:
    changes: list[Change] = field(default_factory=list)
    refusals: list[Refused] = field(default_factory=list)
    #: row id -> the version the rewritten pins now assert. A layer that moves as
    #: one tree contributes every one of its rows, including those whose own
    #: version did not change: that is precisely what `verify` has to re-check
    #: against the rebuilt image.
    new_versions: dict[str, str] = field(default_factory=dict)
    #: artifact name -> {version,url,sha256,manifest_url} deltas.
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    npm_lock: dict | None = None
    npm_manifest: dict | None = None


# --------------------------------------------------------------------------- #
# Line-scoped editing of a commented YAML file
# --------------------------------------------------------------------------- #
def _block_span(lines: list[str], start: int) -> tuple[int, int]:
    """The half-open span of the block whose first line is ``start``.

    A block ends at the next line no more indented than its own opener, which is
    the only structural fact needed here and the only one that survives comments,
    blank lines and folded scalars alike.
    """
    opener = lines[start]
    indent = len(opener) - len(opener.lstrip(" "))
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip(" ")) <= indent:
            return start, index
    return start, len(lines)


def find_cli_row(lines: list[str], row_id: str) -> tuple[int, int]:
    for index, line in enumerate(lines):
        if re.match(rf"^\s*-\s+id:\s+{re.escape(row_id)}\s*$", line):
            return _block_span(lines, index)
    raise Refusal(f"cli-versions.lock has no `- id: {row_id}` row to rewrite")


def find_artifact(lines: list[str], name: str) -> tuple[int, int]:
    inside = False
    for index, line in enumerate(lines):
        if re.match(r"^artifacts:\s*$", line):
            inside = True
            continue
        if inside and re.match(r"^\S", line):
            break
        if inside and re.match(rf"^\s\s{re.escape(name)}:\s*$", line):
            return _block_span(lines, index)
    raise Refusal(f"cli-versions.lock has no `artifacts.{name}` block to rewrite")


def set_scalar(lines: list[str], span: tuple[int, int], key: str, value: str) -> str:
    """Replace ``key:``'s value inside ``span``, in place. Returns the old value.

    Exactly one occurrence is required: a key that appears twice inside one block
    means the span is wrong, and guessing which one to edit is how a line editor
    corrupts a file it was only supposed to touch in one place.
    """
    start, end = span
    hits = [index for index in range(start, end)
            if re.match(rf"^(\s*){re.escape(key)}:\s", lines[index])]
    if len(hits) != 1:
        raise Refusal(
            f"expected exactly one `{key}:` in the block at line {start + 1}, "
            f"found {len(hits)} -- refusing to guess which one is the pin"
        )
    index = hits[0]
    prefix = re.match(rf"^(\s*){re.escape(key)}:\s", lines[index]).group(1)
    old = lines[index].split(":", 1)[1].strip().strip('"').strip("'")
    lines[index] = f'{prefix}{key}: "{value}"'
    return old


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def load_census(path: Path, expect_mode: str) -> dict:
    if not path.is_file():
        raise Refusal(
            f"no version census at {path}\n"
            f"  It is written by the build's own last layer; extract it with\n"
            f"  `docker cp <container>:/opt/mllm/lock/{CENSUS_NAME} {path}`."
        )
    census = json.loads(path.read_text(encoding="utf-8"))
    if census.get("mode") != expect_mode:
        raise Refusal(
            f"{path.name} records mode {census.get('mode')!r}, not {expect_mode!r}. "
            f"A pinned census is a restatement of the pins we already hold and "
            f"proves nothing about what moved; the evidence has to come from the "
            f"@latest image (`make e2e-image-latest`)."
        )
    return census


def census_rows(census: dict) -> dict[str, dict]:
    return {row["id"]: row for row in census.get("clis", [])}


def assert_image_is_sound(census: dict, lock: dict) -> None:
    """Refuse evidence from an image that is not the image we think it is.

    Three cheap facts, each of which invalidates every version below it: the
    platform the pins are written for, a toolchain that is still what the lock
    names (``latest`` inverts the CLI rows and nothing else), and C2's PATH shape.
    A drifted toolchain means the CLI versions were resolved by a different
    runtime than the one the pinned build will use.
    """
    if census.get("platform") != lock.get("platform"):
        raise Refusal(
            f"the census was produced on platform {census.get('platform')!r} but "
            f"cli-versions.lock pins {lock.get('platform')!r} -- an arm64 census "
            f"cannot move an amd64 pin"
        )
    drifted = [row["id"] for row in census.get("toolchain", [])
               if row.get("enforced") and row.get("status") != "ok"]
    if drifted:
        raise Refusal(
            f"the @latest image's own toolchain does not match the lock: "
            f"{', '.join(drifted)}. `latest` inverts the CLI rows and nothing "
            f"else, so this is an image-level change that has to move by hand "
            f"first, in a reviewable commit."
        )
    broken = [entry["binary"] for entry in census.get("path_shape", [])
              if entry.get("status") != "ok"]
    if broken:
        raise Refusal(
            f"the @latest image's wrapper tree is not intact ({', '.join(broken)}); "
            f"its version census is not evidence about anything"
        )


def retarget_url(url: str, old_version: str, new_version: str, label: str) -> str:
    """Move a version inside a URL the lock already carries -- or refuse.

    Composing a URL from a template kept in this file would make it a second
    source of truth for the download location. Substituting inside the recorded
    one keeps the single source, and the exactly-once requirement is what stops a
    coincidental match from rewriting the wrong part of the path. The result is
    still only a *proposal*: it is the pinned rebuild that fetches it and checks
    the checksum, which is where the proof actually lands.
    """
    if old_version == new_version:
        return url
    occurrences = url.count(old_version)
    if occurrences != 1:
        raise Refusal(
            f"{label}: the recorded URL names version {old_version!r} "
            f"{occurrences} times, so substituting {new_version!r} into it is a "
            f"guess. Move this row by hand."
        )
    return url.replace(old_version, new_version)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_npm(lock: dict, census: dict[str, dict], evidence: Path, plan: Plan,
             container_dir: Path = HERE) -> None:
    """The whole npm layer, or none of it.

    The lockfile is one file describing one resolved tree; rewriting the version
    of one row while another keeps its old pin produces a ``package.json`` that
    ``npm ci`` rejects outright. So a row that cannot be substantiated fails the
    layer rather than being skipped out of it.
    """
    rows = [row for row in lock["clis"] if row.get("install") == "npm"]
    if not rows:
        return
    lock_path = evidence / NPM_SUBDIR / "package-lock.json"
    if not lock_path.is_file():
        raise Refusal(
            f"no npm lockfile at {lock_path} -- extract it with "
            f"`docker cp <container>:/opt/mllm/npm/package-lock.json {lock_path}`"
        )
    resolved_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if int(resolved_lock.get("lockfileVersion", 0)) < 2:
        raise LayerRefusal(
            "the extracted package-lock.json is lockfileVersion "
            f"{resolved_lock.get('lockfileVersion')}, which carries no integrity "
            "hash for transitive dependencies -- it is not a pin"
        )
    packages = resolved_lock.get("packages") or {}

    resolved: dict[str, str] = {}
    for row in rows:
        row_id, package = row["id"], row["package"]
        seen = census.get(row_id)
        if seen is None:
            raise LayerRefusal(f"the census has no row for {row_id}")
        if seen.get("status") not in ("ok", "mismatch"):
            raise LayerRefusal(
                f"{row_id}: the census records status {seen.get('status')!r} "
                f"({seen.get('reason', '')}) -- an unreadable version is not "
                f"evidence of a new pin"
            )
        entry = packages.get(f"node_modules/{package}")
        if not entry or not entry.get("version"):
            raise LayerRefusal(
                f"{row_id}: the extracted lockfile has no resolved entry for "
                f"node_modules/{package}"
            )
        if entry["version"] != seen.get("installed"):
            raise LayerRefusal(
                f"{row_id}: the lockfile resolved {package}@{entry['version']} but "
                f"`{row['binary']} --version` in the same image answered "
                f"{seen.get('installed')!r}. Two disagreeing witnesses are not "
                f"evidence; move this row by hand."
            )
        resolved[package] = entry["version"]

    # A tarball that upstream ships without an integrity hash is installed by URL
    # alone. The committed lock has such entries, asserted exactly by
    # tests/test_dockerfile_pins.py; a new one is a real widening of the trust
    # boundary and is a human's decision, not a bookkeeping side effect.
    #
    # THE ALLOWLIST IS SUBTRACTED, and until 2026-09-19 it was not. This refusal
    # tells the reader to "widen UNHASHED_BY_UPSTREAM in
    # tests/test_dockerfile_pins.py" -- and then refused again just the same,
    # because the comparison was against the committed LOCKFILE alone and nothing
    # here ever read that allowlist. A remedy a tool names and then ignores is
    # worse than no remedy: it reads as the tool being broken, and the way past it
    # is to hand-write the very lockfile this evidence path exists to derive. So
    # the decision, once taken and committed, is honoured here.
    committed = json.loads(
        (container_dir / "package-lock.json").read_text(encoding="utf-8"))
    grew = (_unhashed(packages)
            - _unhashed(committed.get("packages") or {})
            - _adjudicated_unhashed(container_dir))
    if grew:
        raise LayerRefusal(
            "the @latest tree adds package(s) that upstream ships with a resolved "
            "URL and NO integrity hash, so a re-published tarball behind an "
            "unchanged version would install unnoticed:\n  "
            + "\n  ".join(sorted(grew)),
            "review the new entries, then widen UNHASHED_BY_UPSTREAM in "
            "tests/test_dockerfile_pins.py in its own commit -- that allowlist is "
            "a trust boundary and moving it is a decision, not bookkeeping",
        )

    moved = [row for row in rows if resolved[row["package"]] != str(row["version"])]
    if not moved:
        return

    for row in rows:
        new = resolved[row["package"]]
        old = str(row["version"])
        plan.new_versions[row["id"]] = new
        if new != old:
            plan.changes.append(Change(
                "cli-versions.lock", f"clis[{row['id']}].version", old, new,
                f"package-lock.json resolved {row['package']}@{new}; "
                f"`{row['binary']} --version` in the same image agreed"))
            plan.changes.append(Change(
                "package.json", f"dependencies[{row['package']}]", old, new,
                "kept exact and in step with the lock's npm rows "
                "(tests/test_dockerfile_pins.py::test_npm_rows_match_package_json)"))

    plan.npm_lock = resolved_lock
    plan.npm_manifest = {package: resolved[package] for package in resolved}


def _unhashed(packages: dict) -> set[str]:
    return {name for name, entry in packages.items()
            if name and entry.get("resolved") and not entry.get("integrity")}


def _adjudicated_unhashed(container_dir: Path) -> set[str]:
    """`UNHASHED_BY_UPSTREAM`, read out of the test module that owns it.

    Parsed with `ast` rather than imported: this tool runs on the host against a
    checkout, and importing a pytest module to read one constant would drag in
    pytest and every fixture beside it for a set of strings.

    A missing file or a renamed constant yields the empty set, which restores the
    strict pre-2026-09-19 behaviour -- the safe direction, since the only thing
    this set can do is let a refusal through.
    """
    source = container_dir.parent / "test_dockerfile_pins.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - a broken checkout
        return set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "UNHASHED_BY_UPSTREAM"
                   for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:  # pragma: no cover - not a literal any more
            return set()
        return {str(item) for item in value}
    return set()


def plan_artifacts(lock: dict, census: dict[str, dict], evidence: Path,
                   plan: Plan) -> None:
    """The three CLIs that ship as a file, split by what the image kept.

    ``install: download`` puts the fetched file on disk unchanged, so its
    checksum is recoverable. ``install: archive`` unpacks it and throws the
    artifact away, so it is not -- and that distinction is already in the lock,
    which is why it is the discriminator here rather than a hand-kept list.
    """
    for row in lock["clis"]:
        install = row.get("install")
        if install not in ("download", "archive"):
            continue
        row_id = row["id"]
        seen = census.get(row_id, {})
        pinned, installed = str(row["version"]), seen.get("installed")
        artifact = row.get("artifact") or row_id

        if install == "archive":
            if installed and installed != pinned:
                try:
                    url = retarget_url(lock["artifacts"][artifact]["url"],
                                       pinned, installed, row_id)
                except Refusal:
                    # This row is refused either way; a URL that cannot be
                    # composed only costs the hint its exact spelling.
                    url = f"<the {artifact} {installed} release artifact>"
                plan.refusals.append(Refused(
                    row_id, install, pinned, str(installed),
                    "the image unpacks this artifact and keeps only the extracted "
                    "binary, so the tarball's sha256 is not recoverable from it",
                    f"curl -sSL '{url}' | sha256sum   "
                    f"# then move artifacts.{artifact} + the ARG lines by hand"))
            continue

        if not installed or installed == pinned:
            continue
        if seen.get("status") not in ("ok", "mismatch"):
            plan.refusals.append(Refused(
                row_id, install, pinned, str(installed),
                f"the census records status {seen.get('status')!r}",
                "re-run `make e2e-image-latest` and read the build log"))
            continue

        binary = row.get("binary") or row_id
        path = evidence / ARTIFACT_SUBDIR / binary
        if not path.is_file():
            raise Refusal(
                f"{row_id}: no artifact at {path} -- extract it with "
                f"`docker cp <container>:/opt/mllm/real/{binary} {path}`"
            )
        entry = lock["artifacts"][artifact]
        delta = {
            "version": installed,
            "url": retarget_url(entry["url"], pinned, installed, row_id),
            "sha256": sha256_file(path),
        }
        if entry.get("manifest_url"):
            delta["manifest_url"] = retarget_url(
                entry["manifest_url"], pinned, installed, f"{row_id} manifest")
        plan.artifacts[artifact] = delta
        plan.new_versions[row_id] = installed

        evidence_note = (
            f"/opt/mllm/real/{binary} in the @latest image hashes to "
            f"{delta['sha256'][:16]}…; the Dockerfile installs this row's "
            f"download unchanged, so that file IS the artifact")
        plan.changes.append(Change(
            "cli-versions.lock", f"clis[{row_id}].version", pinned, installed,
            f"`{binary} --version` in the @latest image"))
        for key in ("version", "url", "sha256", "manifest_url"):
            if key in delta and delta[key] != entry.get(key):
                plan.changes.append(Change(
                    "cli-versions.lock", f"artifacts.{artifact}.{key}",
                    str(entry.get(key)), delta[key], evidence_note))
                plan.changes.append(Change(
                    "Dockerfile", f"ARG {artifact.upper()}_{key.upper()}",
                    str(entry.get(key)), delta[key],
                    "the Dockerfile ARG and the lock are two copies of one "
                    "checksum and are asserted to agree"))


def plan_untouchable(lock: dict, census: dict[str, dict], plan: Plan) -> None:
    """The rows this script will never move, stated as data rather than silence."""
    for row in lock["clis"]:
        if row.get("install") != "pypi":
            continue
        row_id = row["id"]
        seen = census.get(row_id, {})
        installed = seen.get("installed")
        if not installed or installed == str(row["version"]):
            continue
        plan.refusals.append(Refused(
            row_id, "pypi", str(row["version"]), str(installed),
            "the pin is a --generate-hashes requirements file; the image has the "
            "installed version but not the hashes, which only a resolver run "
            "against the index can produce",
            f"uv pip compile --generate-hashes --python-version 3.11 "
            f"-o {row['requirements']} - <<< '{row['package']}=={installed}'"))


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def rewrite_lock(text: str, plan: Plan) -> str:
    """Line-scoped edits, then a document-level check that they were the right ones."""
    lines = text.splitlines()
    before = yaml.safe_load(text)

    for row_id, version in plan.new_versions.items():
        set_scalar(lines, find_cli_row(lines, row_id), "version", version)
    for artifact, delta in plan.artifacts.items():
        span = find_artifact(lines, artifact)
        for key, value in delta.items():
            set_scalar(lines, span, key, value)

    rewritten = "\n".join(lines) + "\n"
    after = yaml.safe_load(rewritten)

    expected = json.loads(json.dumps(before))
    for row in expected["clis"]:
        if row["id"] in plan.new_versions:
            row["version"] = plan.new_versions[row["id"]]
    for artifact, delta in plan.artifacts.items():
        expected["artifacts"][artifact].update(delta)
    if after != expected:
        raise Refusal(
            "the line-scoped rewrite of cli-versions.lock did not produce the "
            "document it was supposed to -- nothing has been written. This is a "
            "bug in lock_update.py, not in the pins."
        )
    return rewritten


def rewrite_package_json(text: str, versions: dict[str, str]) -> str:
    """Exact versions, never the `^` range `npm install` leaves behind.

    ``npm install <pkg>@latest`` writes a caret range into the manifest it
    resolved; committing that would make ``package.json`` a range and the
    lockfile the only pin. The repo's rule is that both are exact and equal, so
    the manifest is rewritten from the resolved versions rather than copied.
    """
    manifest = json.loads(text)
    declared = manifest["dependencies"]
    if set(declared) != set(versions):
        raise Refusal(
            "the committed package.json and the resolved tree name different "
            f"packages: {sorted(set(declared) ^ set(versions))}"
        )
    for package in declared:
        declared[package] = versions[package]
    return json.dumps(manifest, indent=2) + "\n"


def rewrite_package_lock(resolved: dict, versions: dict[str, str]) -> str:
    """The image's lockfile, with the root requirement made exact.

    Only the root ``packages[""].dependencies`` map is touched, and only to the
    exact version already resolved beneath it: ``npm ci`` refuses to run when the
    manifest and the lockfile's root requirement differ, and the manifest above
    is exact.
    """
    root = (resolved.get("packages") or {}).get("")
    if root is None or "dependencies" not in root:
        raise Refusal("the extracted package-lock.json has no root package entry")
    if set(root["dependencies"]) != set(versions):
        raise Refusal(
            "the extracted lockfile's root dependencies do not match the resolved "
            f"set: {sorted(set(root['dependencies']) ^ set(versions))}"
        )
    for package, version in versions.items():
        root["dependencies"][package] = version
    return json.dumps(resolved, indent=2) + "\n"


def rewrite_dockerfile(text: str, plan: Plan) -> str:
    """The ARG copies of the artifact pins, which a test asserts must agree."""
    lines = text.splitlines()
    for artifact, delta in plan.artifacts.items():
        for key, value in delta.items():
            if key == "manifest_url":
                continue  # the build never fetches the manifest
            name = f"{artifact.upper()}_{key.upper()}"
            hits = [index for index, line in enumerate(lines)
                    if line.startswith(f"ARG {name}=")]
            if len(hits) != 1:
                raise Refusal(
                    f"expected exactly one `ARG {name}=` in the Dockerfile, "
                    f"found {len(hits)}"
                )
            lines[hits[0]] = f"ARG {name}={value}"
    return "\n".join(lines) + "\n"


def _git_dirty(container_dir: Path, names: tuple[str, ...]) -> list[str] | None:
    """Which of ``names`` already carry uncommitted edits (None = no git here).

    The revert instruction this target prints is ``git checkout --``, which is
    only safe when the sole thing it would discard is this script's own work.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *names],
            cwd=container_dir, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return [line[3:].strip() for line in completed.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_apply(args) -> int:
    container_dir = Path(args.container_dir).resolve()
    evidence = Path(args.evidence).resolve()
    lock_path = container_dir / "cli-versions.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    lock = yaml.safe_load(lock_text)

    census = load_census(evidence / CENSUS_NAME, "latest")
    assert_image_is_sound(census, lock)
    rows = census_rows(census)

    plan = Plan()
    try:
        plan_npm(lock, rows, evidence, plan, container_dir)
    except LayerRefusal as refusal:
        npm_rows = [row["id"] for row in lock["clis"] if row.get("install") == "npm"]
        moved = [f"{row_id} {rows[row_id]['expected']} -> {rows[row_id]['installed']}"
                 for row_id in npm_rows
                 if rows.get(row_id, {}).get("status") == "mismatch"]
        # The layer is one resolved tree: a partially-planned npm move is exactly
        # the inconsistent state this whole function refuses to produce.
        plan.changes = [change for change in plan.changes
                        if change.file not in ("package.json", "package-lock.json")]
        plan.new_versions = {row_id: version
                             for row_id, version in plan.new_versions.items()
                             if row_id not in npm_rows}
        plan.npm_lock = plan.npm_manifest = None
        plan.refusals.append(Refused(
            "npm layer", "npm", f"{len(npm_rows)} rows",
            f"{len(moved)} moved upstream",
            f"{refusal}\nrows that would have moved: "
            + (", ".join(moved) or "none"),
            refusal.command))
    plan_artifacts(lock, rows, evidence, plan)
    plan_untouchable(lock, rows, plan)

    for refusal in plan.refusals:
        print(f"NOT MOVED  {refusal.row:<12} {refusal.pinned} -> "
              f"{refusal.installed}  ({refusal.install})")
        for line in refusal.reason.splitlines():
            print(f"           {line}")
        if refusal.command:
            for line in refusal.command.splitlines():
                print(f"           {line}")
    if plan.refusals:
        print()

    if not plan.changes:
        print("lock-update: every pin this target can substantiate is already "
              "current; nothing written.")
        if plan.refusals:
            print("             The rows above moved upstream but their pins are "
                  "not recoverable from the image; they are named with the "
                  "command that would move them.")
        # The record is written on this path too: "goose moved and we cannot pin
        # it" is a finding, and a finding that exists only in a terminal scrollback
        # is one nobody can act on later.
        _write_record(args, census, plan, {})
        return 2

    for change in plan.changes:
        print(f"MOVE       {change.file:<20} {change.target:<34} "
              f"{change.old} -> {change.new}")
    print()

    # --dry-run is exempt: it writes nothing, so there is nothing for the revert
    # to be ambiguous about, and previewing the plan is exactly what you want to
    # do before deciding whether to commit the edits already in the tree.
    dirty = _git_dirty(container_dir, WRITABLE) if not args.dry_run else []
    if dirty and not args.allow_dirty:
        print("lock-update: refusing to rewrite files that already carry "
              "uncommitted edits:", file=sys.stderr)
        for name in dirty:
            print(f"  {name}", file=sys.stderr)
        print("  The verification step's revert is `git checkout --`, which would "
              "discard those edits too. Commit or stash them first, or pass "
              "--allow-dirty and own the revert yourself.", file=sys.stderr)
        return 1

    written: dict[str, dict[str, str]] = {}
    payloads: dict[str, str] = {}
    payloads["cli-versions.lock"] = rewrite_lock(lock_text, plan)
    if plan.npm_manifest is not None:
        payloads["package.json"] = rewrite_package_json(
            (container_dir / "package.json").read_text(encoding="utf-8"),
            plan.npm_manifest)
        payloads["package-lock.json"] = rewrite_package_lock(
            plan.npm_lock, plan.npm_manifest)
    if plan.artifacts:
        payloads["Dockerfile"] = rewrite_dockerfile(
            (container_dir / "Dockerfile").read_text(encoding="utf-8"), plan)

    for name, payload in payloads.items():
        path = container_dir / name
        written[name] = {"before": sha256_text(path.read_text(encoding="utf-8")),
                         "after": sha256_text(payload)}
        if not args.dry_run:
            path.write_text(payload, encoding="utf-8")

    _write_record(args, census, plan, written)
    if args.dry_run:
        print("--dry-run: nothing was written.")
    else:
        print("revert with: " + revert_command(container_dir, written))
    return 0


def _write_record(args, census: dict, plan: Plan, written: dict) -> None:
    """What moved, what did not, and the digest of every file that was touched.

    ``verify`` reads this back against the census of the image rebuilt from it,
    which is the step that turns the rewrite from an edit into a proof.
    """
    if not args.record:
        return
    record = {
        "kind": "lock-update",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": args.image,
        "dry_run": bool(args.dry_run),
        "census": {"mode": census.get("mode"), "platform": census.get("platform"),
                   "base_image": census.get("base_image")},
        "changed": [vars(change) for change in plan.changes],
        "not_moved": [vars(refusal) for refusal in plan.refusals],
        "new_versions": plan.new_versions,
        "files": written,
    }
    record_path = Path(args.record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"record: {record_path}")


def revert_command(container_dir: Path, written: dict) -> str:
    """The exact undo for this rewrite, naming only the files it touched.

    Not `git checkout -- <the container dir>`: that would also discard edits to
    the twenty other files beside these four, which is a worse outcome than the
    one it is offered to fix.
    """
    paths = []
    for name in written:
        absolute = container_dir / name
        relative = os.path.relpath(absolute, Path.cwd())
        # `make` runs this from the checkout root, where the relative form is the
        # readable one; anywhere else it would be a ladder of `../` and the
        # absolute path is what a reader can act on.
        paths.append(relative if not relative.startswith("..") else str(absolute))
    return "git checkout -- " + " ".join(sorted(paths))


def cmd_evidence(args) -> int:
    """Print `<path in the image>\\t<path under --evidence>`, one per line.

    The makefile recipe loops over this rather than carrying its own list of
    what to ``docker cp``: which rows have a recoverable artifact is decided by
    ``install:`` in ``cli-versions.lock``, and a hand-kept copy of that answer in
    a shell recipe is the second source of truth this file exists to avoid.
    """
    container_dir = Path(args.container_dir).resolve()
    lock = yaml.safe_load(
        (container_dir / "cli-versions.lock").read_text(encoding="utf-8"))
    rows = [
        (f"/opt/mllm/lock/{CENSUS_NAME}", CENSUS_NAME),
        ("/opt/mllm/npm/package.json", f"{NPM_SUBDIR}/package.json"),
        ("/opt/mllm/npm/package-lock.json", f"{NPM_SUBDIR}/package-lock.json"),
    ]
    for row in lock["clis"]:
        if row.get("install") != "download":
            continue
        binary = row.get("binary") or row["id"]
        rows.append((f"/opt/mllm/real/{binary}", f"{ARTIFACT_SUBDIR}/{binary}"))
    for source, destination in rows:
        print(f"{source}\t{destination}")
    return 0


def cmd_verify(args) -> int:
    """Grade the record against the census of the image built FROM it."""
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    container_dir = Path(args.container_dir).resolve()
    census = load_census(Path(args.installed), "pinned")
    rows = census_rows(census)

    failures: list[str] = []
    for name, digests in (record.get("files") or {}).items():
        actual = sha256_text((container_dir / name).read_text(encoding="utf-8"))
        if actual != digests["after"]:
            failures.append(
                f"{name}: changed since the rewrite ({digests['after'][:12]}… -> "
                f"{actual[:12]}…) -- something edited it between apply and verify, "
                f"so this census is not evidence about the record")

    for row_id, version in (record.get("new_versions") or {}).items():
        seen = rows.get(row_id)
        if seen is None:
            failures.append(f"{row_id}: the pinned image's census has no such row")
            continue
        if seen.get("status") != "ok" or seen.get("installed") != version:
            failures.append(
                f"{row_id}: the record moved the pin to {version!r} but the "
                f"rebuilt pinned image installed {seen.get('installed')!r} "
                f"(status {seen.get('status')!r})")
        else:
            print(f"VERIFIED   {row_id:<12} {version}")

    unmoved = [row["id"] for row in census.get("clis", [])
               if row.get("status") not in ("ok", "not-verifiable")]
    for row_id in unmoved:
        if row_id not in (record.get("new_versions") or {}):
            failures.append(
                f"{row_id}: the rebuilt pinned image does not match the lock "
                f"either, and this record never touched it")

    if failures:
        print("\nLOCK-UPDATE VERIFICATION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("\nRevert the rewrite with:\n  "
              + revert_command(container_dir, record.get("files") or {}),
              file=sys.stderr)
        return 1
    print(f"\nlock-update: verified {len(record.get('new_versions') or {})} pin(s) "
          f"against the rebuilt pinned image.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--container-dir", default=str(HERE),
                        help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply_parser = subparsers.add_parser(
        "apply", help="rewrite the pins the @latest image proves have moved")
    apply_parser.add_argument("--evidence", required=True,
                              help="directory holding the extracted evidence")
    apply_parser.add_argument("--image", default="mllm-e2e:latest-probe",
                              help="the image tag the evidence came from")
    apply_parser.add_argument("--record", default=None)
    apply_parser.add_argument("--dry-run", action="store_true")
    apply_parser.add_argument("--allow-dirty", action="store_true")
    apply_parser.add_argument("--container-dir", default=str(HERE),
                              help=argparse.SUPPRESS)

    verify_parser = subparsers.add_parser(
        "verify", help="grade a record against the rebuilt pinned image's census")
    verify_parser.add_argument("--record", required=True)
    verify_parser.add_argument("--installed", required=True,
                               help=f"the pinned image's {CENSUS_NAME}")
    verify_parser.add_argument("--container-dir", default=str(HERE),
                               help=argparse.SUPPRESS)

    evidence_parser = subparsers.add_parser(
        "evidence", help="print the image paths `apply` needs, one per line")
    evidence_parser.add_argument("--container-dir", default=str(HERE),
                                 help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    commands = {"apply": cmd_apply, "verify": cmd_verify, "evidence": cmd_evidence}
    try:
        return commands[args.command](args)
    except Refusal as refusal:
        print(f"lock-update: {refusal}", file=sys.stderr)
        print("nothing was written.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
