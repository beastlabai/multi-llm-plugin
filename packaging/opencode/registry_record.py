#!/usr/bin/env python3
"""The registry-install record's one validator, with two profiles.

Run it as::

    uv run --frozen --no-dev --project skills/multi-llm -- \\
        python packaging/opencode/registry_record.py validate \\
        --profile release|bootstrap <record.json> [--tag T] [--repo PATH]

**What the record is.** What a registry-install probe measured about the bytes
npm *serves* for one ``opencode-multi-llm`` version, installed by OpenCode from
the single registry into a fresh environment (publishing plan, step 5). It is
not row 7, which loads the gated tarball before publication; this one checks
what publication actually produced. Its shape is
``registry-record.schema.json``, beside this file.

**Two profiles, one file.**

* ``release`` — the canonical step-5 record,
  ``<UTC>-opencode-release-network-check.json``, committed sanitized under
  ``skills/multi-llm/tests/container/records/``. ``environment`` is
  ``container`` or ``vm``, the only places the network can be cut, and the
  criteria are exactly the release kind's: the eight below for a first
  release, those eight plus ``upgrade_runs_new_payload`` for a later one.
* ``bootstrap`` — the retired check the laptop path's ``opencode-v<first>``
  tag push was meant to run, ``<UTC>-opencode-bootstrap-check.json``, a
  workflow artifact when it existed.
  ``environment: "ci-runner"``, a workflow run ID, the first-release kind, and
  exactly SEVEN criteria: the eight without ``warm_offline_operation``, which a
  runner cannot attempt (no Docker daemon, no defined way to cut one process
  off the network). A record carrying that criterion is refused rather than
  trusted, so a bootstrap record never passes the release profile and a green
  bootstrap never stands in for step 5. The workflow's bootstrap path was
  retired after ``opencode-v0.1.1``, whose evidence is its step-5 record; no
  job writes this profile's record now, and since it admits a first release
  only, no later tag can.

**The first-release exception is decided from the tag's commit,** never from the
version number: ``git show <tag>:skills/multi-llm/hosts/capabilities.yaml``,
and whether it has ``hosts.opencode.package.public_version``. While it has none,
no version has been verified, so the record sets ``previous_pin: null`` and
carries no upgrade criterion (there is nothing to upgrade from); that also
covers a Recovery replacement for a first release whose own check failed.
Otherwise the release is a later one: ``previous_pin`` must be
``<name>@<public_version>``, older than the new version, and the upgrade
criterion must pass. Each side's record is refused on the other's shape.

**What it refuses.** A record whose tag, SHA, version or integrity is not the
release's (the tag resolved in ``--repo``; its ``package.json`` read at the
tag); a registry other than the single one; missing, duplicate or unexpected
criteria; a ``pass`` that is not the boolean ``true``; rehearsal evidence
(``rehearsal`` anything but ``false``, or a ``-rehearsal.json`` file); a record
without ``same_pin_restart_matches``; and a passing ``warm_offline_operation``
without its network proof: ``NetworkMode`` ``none``, a loopback-only interface
list and an ``npm view`` against the single registry failing with a NETWORK
error code (not ``E404``, not an auth error, not npm's own ``ENOTCACHED``)
before and after, with the installed payload's location and digest unchanged.

**Rehearsal records, structure only.** ``--rehearsal-structure-only`` checks a
record the registry-install probe wrote against a localhost registry stub
(``rehearsal: true``) for everything above EXCEPT what makes it a rehearsal:
``rehearsal`` must then be ``true``, ``registry`` and ``installed_registry`` may
be the loopback stub (``http://127.0.0.1:<port>``), a bootstrap rehearsal run
outside a workflow may have a ``null`` run ID, and the file must not carry a
release record's name. It exists so a rehearsal can prove the probe writes the
right shape; it never makes a rehearsal release evidence, and says so every
time it runs.

Stdlib plus PyYAML, the skill's one runtime dependency, so the workflow can run
it under ``--no-dev``. ``jsonschema`` is a dev extra: the schema is applied here
by a small interpreter of exactly the keywords the schema uses, and the suite
checks that it agrees with ``jsonschema`` on the same records.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SCHEMA_PATH = PACKAGE_DIR / "registry-record.schema.json"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import registry  # noqa: E402

#: Step 5's eight criteria, in the plan's order. Every first-release record.
FIRST_RELEASE_CRITERIA: Tuple[str, ...] = (
    "npm_spec_resolves",
    "manifest_version_matches",
    "installed_tree_matches_candidate",
    "command_resolves_installed_skill",
    "tool_init_runs_installed_payload",
    "cold_dependencies_install",
    "warm_offline_operation",
    "same_pin_restart_matches",
)
OFFLINE = "warm_offline_operation"
UPGRADE = "upgrade_runs_new_payload"
RESTART = "same_pin_restart_matches"
#: A later release also upgrades an existing installation of the previous pin.
LATER_RELEASE_CRITERIA: Tuple[str, ...] = FIRST_RELEASE_CRITERIA + (UPGRADE,)
#: What the bootstrap job can attempt on a runner: the eight without the offline one.
BOOTSTRAP_CRITERIA: Tuple[str, ...] = tuple(c for c in FIRST_RELEASE_CRITERIA if c != OFFLINE)

PROFILES = ("release", "bootstrap")
ENVIRONMENTS = {"release": ("container", "vm"), "bootstrap": ("ci-runner",)}
SUFFIXES = {
    "release": "-opencode-release-network-check.json",
    "bootstrap": "-opencode-bootstrap-check.json",
}
_FILE_NAME = re.compile(r"^\d{8}T\d{6}Z-")

#: Where the tag's commit says whether any version has been verified yet.
CAPABILITIES = "skills/multi-llm/hosts/capabilities.yaml"
TAG_MANIFEST = "packaging/opencode/package.json"
TAG_PREFIX = "opencode-v"

#: The codes npm reports when it cannot reach a registry at all. `E404` (an
#: answer), auth codes (an answer) and `ENOTCACHED` (npm's own offline mode, a
#: configuration rather than a cut network) are deliberately absent.
NETWORK_ERROR_CODES = frozenset({
    "ENOTFOUND", "EAI_AGAIN", "EAI_FAIL", "ECONNREFUSED", "ECONNRESET",
    "ENETUNREACH", "ENETDOWN", "EHOSTUNREACH", "ETIMEDOUT", "ESOCKETTIMEDOUT",
})
_LOOPBACK = re.compile(r"^lo\d*$")
#: The localhost registry stub a rehearsal reads from; nothing else is admitted.
_REHEARSAL_REGISTRY = re.compile(r"^http://127\.0\.0\.1:[0-9]{1,5}/?$")
#: Printed by every structure-only check, passing or not.
REHEARSAL_NOTICE = "a rehearsal is never release evidence"
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


class RecordError(RuntimeError):
    """The record could not be read at all."""


# ---------------------------------------------------------------------------
# The schema, without jsonschema
# ---------------------------------------------------------------------------

def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_errors(node: Any, schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None,
                  path: str = "<root>") -> List[str]:
    """Every way *node* departs from *schema*, for the keywords this schema uses:
    ``$ref`` (local), ``type``, ``const``, ``enum``, ``required``, ``properties``,
    ``additionalProperties``, ``items``, ``pattern``, ``minLength``, ``minimum``.
    A keyword outside that list is refused by :func:`_assert_supported`, so the
    schema cannot grow a rule this interpreter silently skips."""
    root = root if root is not None else schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return [f"{path}: unsupported $ref {ref!r}"]
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        return schema_errors(node, target, root, path)
    bad: List[str] = []
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[kind](node) for kind in kinds):
            return [f"{path}: {node!r} is not of type {' or '.join(kinds)}"]
    if "const" in schema and (node != schema["const"] or type(node) is not type(schema["const"])):
        bad.append(f"{path}: {node!r} is not {schema['const']!r}")
    if "enum" in schema and node not in schema["enum"]:
        bad.append(f"{path}: {node!r} is not one of {schema['enum']}")
    if isinstance(node, str):
        if "pattern" in schema and not re.search(schema["pattern"], node):
            bad.append(f"{path}: {node!r} does not match {schema['pattern']!r}")
        if len(node) < schema.get("minLength", 0):
            bad.append(f"{path}: {node!r} is shorter than {schema['minLength']}")
    if _TYPES["number"](node) and "minimum" in schema and node < schema["minimum"]:
        bad.append(f"{path}: {node!r} is below {schema['minimum']}")
    if isinstance(node, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in node:
                bad.append(f"{path}: missing required key {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, value in node.items():
            if key in properties:
                bad += schema_errors(value, properties[key], root, f"{path}/{key}")
            elif extra is False:
                bad.append(f"{path}: {key!r} is not a key this schema allows")
            elif isinstance(extra, dict):
                bad += schema_errors(value, extra, root, f"{path}/{key}")
    if isinstance(node, list) and isinstance(schema.get("items"), dict):
        for index, value in enumerate(node):
            bad += schema_errors(value, schema["items"], root, f"{path}[{index}]")
    return bad


_SUPPORTED = {
    "$schema", "$id", "$ref", "$defs", "title", "description", "type", "const", "enum",
    "required", "properties", "additionalProperties", "items", "pattern", "minLength",
    "minimum",
}


def _assert_supported(schema: Any, where: str = "<schema>") -> None:
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("properties", "$defs"):
                for name, sub in value.items():
                    _assert_supported(sub, f"{where}/{key}/{name}")
                continue
            if key not in _SUPPORTED:
                raise RecordError(f"{where}: schema keyword {key!r} is not applied by "
                                  "registry_record.schema_errors(); teach it first")
            if key in ("items", "additionalProperties"):
                _assert_supported(value, f"{where}/{key}")


# ---------------------------------------------------------------------------
# The tag's commit
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def tag_facts(repo: Path, tag: str) -> Dict[str, Any]:
    """What the tag's commit says: its SHA, its package manifest and its
    ``public_version`` (``None`` while no version has been verified).

    Raises :class:`RecordError` naming what could not be read."""
    if not re.fullmatch(rf"{TAG_PREFIX}[0-9A-Za-z.+-]+", tag or ""):
        raise RecordError(f"{tag!r} is not an {TAG_PREFIX}* tag")
    ref = f"refs/tags/{tag}"
    sha = _git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}")
    if not sha:
        raise RecordError(f"tag {tag} does not resolve in {repo}")
    manifest_text = _git(repo, "show", f"{ref}:{TAG_MANIFEST}")
    capabilities_text = _git(repo, "show", f"{ref}:{CAPABILITIES}")
    if manifest_text is None:
        raise RecordError(f"{tag} carries no {TAG_MANIFEST}")
    if capabilities_text is None:
        raise RecordError(f"{tag} carries no {CAPABILITIES}")
    try:
        manifest = json.loads(manifest_text)
    except ValueError as exc:
        raise RecordError(f"{tag}:{TAG_MANIFEST} is not JSON ({exc})")
    import yaml

    try:
        capabilities = yaml.safe_load(capabilities_text) or {}
    except yaml.YAMLError as exc:
        raise RecordError(f"{tag}:{CAPABILITIES} is not YAML ({exc})")
    package = (((capabilities.get("hosts") or {}).get("opencode") or {}).get("package") or {}) \
        if isinstance(capabilities, dict) else {}
    public = package.get("public_version") if isinstance(package, dict) else None
    if public is not None and (not isinstance(public, str) or not _VERSION.match(public)):
        raise RecordError(f"{tag}:{CAPABILITIES} has a malformed public_version {public!r}")
    return {"sha": sha.strip(), "name": manifest.get("name"),
            "version": manifest.get("version"), "public_version": public}


def _version_key(text: str) -> Tuple:
    match = _VERSION.match(text or "")
    if not match:
        return ()
    major, minor, patch, pre = match.groups()
    return (int(major), int(minor), int(patch), 0 if pre else 1, pre or "")


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------

def _criteria_names(record: Dict[str, Any]) -> List[Any]:
    criteria = record.get("criteria")
    if not isinstance(criteria, list):
        return []
    return [c.get("name") if isinstance(c, dict) else None for c in criteria]


def _network_proof_errors(proof: Any, single: str) -> List[str]:
    """Why *proof* does not show the offline run had no network, or ``[]``."""
    where = f"{OFFLINE}.network_proof"
    if not isinstance(proof, dict):
        return [f"{OFFLINE} passes without its network proof (NetworkMode, the interface "
                "list and a failing npm view, before and after)"]
    bad = []
    if proof.get("network_mode") != "none":
        bad.append(f"{where}: NetworkMode is {proof.get('network_mode')!r}, not 'none'")
    for moment in ("before", "after"):
        probe = proof.get(moment)
        if not isinstance(probe, dict):
            bad.append(f"{where}.{moment}: missing")
            continue
        interfaces = probe.get("interfaces")
        if not isinstance(interfaces, list) or not interfaces \
                or not all(isinstance(i, str) and _LOOPBACK.match(i) for i in interfaces):
            bad.append(f"{where}.{moment}: interfaces {interfaces!r} are not loopback only")
        view = probe.get("npm_view")
        if not isinstance(view, dict):
            bad.append(f"{where}.{moment}: no npm view result")
            continue
        if registry.normalize(view.get("registry") or "") != single:
            bad.append(f"{where}.{moment}: npm view asked {view.get('registry')!r}, not the "
                       f"single registry {single}")
        code = view.get("code")
        if view.get("exit") == 0 or code not in NETWORK_ERROR_CODES:
            why = ("an answer from the registry" if code == "E404" or (code or "").startswith("E4")
                   or code == "ENEEDAUTH" else "not a network failure")
            bad.append(f"{where}.{moment}: npm view exited {view.get('exit')!r} with code "
                       f"{code!r}, {why}; only a network error code proves the cut")
    payload = proof.get("payload")
    before = payload.get("before") if isinstance(payload, dict) else None
    after = payload.get("after") if isinstance(payload, dict) else None
    if not isinstance(before, dict) or not isinstance(after, dict) \
            or not before.get("location") or not before.get("digest"):
        bad.append(f"{where}.payload: the installed payload's location and digest are not "
                   "recorded before and after")
    elif (before.get("location"), before.get("digest")) != \
            (after.get("location"), after.get("digest")):
        bad.append(f"{where}.payload: the offline run moved or changed the installed payload "
                   f"({before} -> {after})")
    return bad


def validate_record(
    record: Any, *, profile: str, tag: Optional[str] = None, repo: Path = REPO_ROOT,
    file_name: Optional[str] = None, integrity: Optional[str] = None,
    digest: Optional[str] = None, rehearsal_structure_only: bool = False,
) -> List[str]:
    """Every reason *record* is not valid evidence under *profile*; ``[]`` if it is.

    With *rehearsal_structure_only*, every reason a REHEARSAL record is not
    the right shape for *profile* (module docstring); ``[]`` then never means
    release evidence.
    """
    if profile not in PROFILES:
        return [f"unknown profile {profile!r}; expected one of {PROFILES}"]
    if not isinstance(record, dict):
        return ["the record is not a JSON object"]
    schema = load_schema()
    _assert_supported(schema)
    rehearsal_only = bool(rehearsal_structure_only)
    if rehearsal_only:
        schema = json.loads(json.dumps(schema))
        schema["properties"]["rehearsal"] = {"const": True}
    errors = schema_errors(record, schema)

    if rehearsal_only and file_name is not None:
        if _FILE_NAME.match(file_name) and file_name.endswith(SUFFIXES[profile]):
            errors.append(f"{file_name} carries a release record's name; a rehearsal record "
                          "must not (name it ...-rehearsal.json, outside the records directory)")
    elif file_name is not None:
        if file_name.endswith("-rehearsal.json"):
            errors.append(f"{file_name} is a rehearsal record by name")
        elif not (_FILE_NAME.match(file_name) and file_name.endswith(SUFFIXES[profile])):
            errors.append(f"{file_name} is not a <UTC>{SUFFIXES[profile]} record")

    # -- the profile ---------------------------------------------------------
    environment = record.get("environment")
    if environment not in ENVIRONMENTS[profile]:
        errors.append(f"environment {environment!r} is not admissible under the {profile} "
                      f"profile (expected {' or '.join(ENVIRONMENTS[profile])})")
    run_id = record.get("workflow_run_id")
    if profile == "bootstrap":
        if rehearsal_only and run_id is None:
            pass  # a bootstrap rehearsal outside a workflow has no run to name
        elif not isinstance(run_id, int) or isinstance(run_id, bool) or run_id < 1:
            errors.append(f"a bootstrap record names its workflow run; workflow_run_id is {run_id!r}")
    elif run_id is not None:
        errors.append(f"a {profile} record is not a workflow run's; workflow_run_id is {run_id!r}")
    if rehearsal_only:
        if record.get("rehearsal") is not True:
            errors.append("--rehearsal-structure-only checks rehearsal records only; this one says "
                          f"\"rehearsal\": {record.get('rehearsal')!r}")
    elif record.get("rehearsal") is not False:
        errors.append(f"rehearsal evidence (\"rehearsal\": {record.get('rehearsal')!r}) is never "
                      "release evidence")

    # -- the registry --------------------------------------------------------
    single = registry.single_registry()
    for key in ("registry", "installed_registry"):
        value = record.get(key)
        if rehearsal_only and isinstance(value, str) and _REHEARSAL_REGISTRY.match(value):
            continue
        if not isinstance(value, str) or registry.normalize(value) != single:
            errors.append(f"{key} {value!r} is not the single registry {single}")

    # -- the tag binding -----------------------------------------------------
    name, version = record.get("name"), record.get("version")
    record_tag = record.get("tag")
    if tag is not None and record_tag != tag:
        errors.append(f"tag {record_tag!r} is not the release's {tag!r}")
    if record_tag != f"{TAG_PREFIX}{version}":
        errors.append(f"tag {record_tag!r} does not name version {version!r}")
    if record.get("spec") != f"{name}@{version}":
        errors.append(f"spec {record.get('spec')!r} is not {name}@{version}")
    if integrity is not None and record.get("integrity") != integrity:
        errors.append(f"integrity {record.get('integrity')!r} is not the gated {integrity!r}")
    if digest is not None and record.get("candidate_digest") != digest:
        errors.append(f"candidate_digest {record.get('candidate_digest')!r} is not {digest!r}")
    facts = None
    try:
        facts = tag_facts(Path(repo), str(tag if tag is not None else record_tag))
    except RecordError as exc:
        errors.append(f"cannot bind the record to its tag: {exc}")
    if facts:
        if record.get("sha") != facts["sha"]:
            errors.append(f"sha {record.get('sha')!r} is not {record_tag}'s commit {facts['sha']}")
        if (name, version) != (facts["name"], facts["version"]):
            errors.append(f"{name}@{version} is not the tag's package.json "
                          f"({facts['name']}@{facts['version']})")

    # -- the release kind, from the tag's commit -----------------------------
    kind = record.get("release_kind")
    if facts:
        decided = "first" if facts["public_version"] is None else "later"
        if kind != decided:
            errors.append(
                f"release_kind {kind!r}, but {record_tag}'s {CAPABILITIES} "
                + ("has no public_version: a first release"
                   if decided == "first" else
                   f"has public_version {facts['public_version']}: a later release"))
        kind = decided
    if profile == "bootstrap" and kind != "first":
        errors.append("a bootstrap record covers the first release only")
    previous = record.get("previous_pin")
    if kind == "first":
        if previous is not None:
            errors.append(f"a first release has no previous pin; previous_pin is {previous!r}")
    elif kind == "later":
        public = facts["public_version"] if facts else None
        expected = f"{name}@{public}" if public else None
        if previous is None:
            errors.append("a later release names its previous_pin")
        elif expected and previous != expected:
            errors.append(f"previous_pin {previous!r} is not {expected} (the tag's public_version)")
        if public and _version_key(public) and _version_key(version) \
                and not _version_key(public) < _version_key(version):
            errors.append(f"public_version {public} is not older than {version}")

    # -- the criteria --------------------------------------------------------
    if profile == "bootstrap":
        required = BOOTSTRAP_CRITERIA
    else:
        required = LATER_RELEASE_CRITERIA if kind == "later" else FIRST_RELEASE_CRITERIA
    names = _criteria_names(record)
    seen = set()
    for item in names:
        if item in seen:
            errors.append(f"criterion {item!r} appears more than once")
        seen.add(item)
    if RESTART not in seen:
        errors.append(f"no {RESTART}: every release records the same-pin restart")
    if profile == "bootstrap" and OFFLINE in seen:
        errors.append(f"a bootstrap record cannot carry {OFFLINE}: a runner has no way to cut "
                      "the network, so the criterion is omitted, never reported")
    if kind == "first" and UPGRADE in seen:
        errors.append(f"a first release carries no {UPGRADE}: there is no verified version to "
                      "upgrade from")
    if kind == "later" and UPGRADE not in seen:
        errors.append(f"a later release must pass {UPGRADE}")
    missing = [c for c in required if c not in seen]
    unexpected = sorted(str(c) for c in seen if c not in required)
    if missing:
        errors.append(f"missing criteria {missing}")
    if unexpected:
        errors.append(f"criteria {unexpected} are not this {profile} record's")
    for criterion in record.get("criteria") if isinstance(record.get("criteria"), list) else []:
        if not isinstance(criterion, dict):
            continue
        label = criterion.get("name")
        verdict = criterion.get("pass")
        if not isinstance(verdict, bool):
            errors.append(f"{label}: pass {verdict!r} is not a boolean")
        elif verdict is not True:
            errors.append(f"{label} did not pass")
        # A non-empty object, which is what the probe writes for every
        # criterion, a failing or unattempted one included. `in (None, "", [],
        # {})` let `false` and `0` through as "evidence": they compare unequal
        # to every member, and neither preserves any output.
        evidence = criterion.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            errors.append(f"{label}: no supporting evidence preserved (evidence is "
                          f"{evidence!r}, not a non-empty object)")
        if "network_proof" in criterion and label != OFFLINE:
            errors.append(f"{label}: network_proof belongs to {OFFLINE} only")
        if label == OFFLINE and verdict is True:
            errors += _network_proof_errors(criterion.get("network_proof"), single)
    return errors


def validate(path: Path, *, profile: str, tag: Optional[str] = None,
             repo: Path = REPO_ROOT, integrity: Optional[str] = None,
             digest: Optional[str] = None, rehearsal_structure_only: bool = False) -> List[str]:
    """:func:`validate_record` on a file, its name included."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path}: unreadable ({exc})"]
    return validate_record(record, profile=profile, tag=tag, repo=repo, file_name=path.name,
                           integrity=integrity, digest=digest,
                           rehearsal_structure_only=rehearsal_structure_only)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="registry_record.py",
        description="Validate an opencode registry-install record (step 5 or bootstrap).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="validate one record")
    check.add_argument("--profile", choices=PROFILES, required=True)
    check.add_argument("record", type=Path)
    check.add_argument("--tag", default=None,
                       help="the release tag the record must name (default: the record's own)")
    check.add_argument("--repo", type=Path, default=REPO_ROOT,
                       help="the repository whose tag decides SHA, version and release kind")
    check.add_argument("--integrity", default=None,
                       help="the gated tarball's integrity, when the caller has it")
    check.add_argument("--digest", default=None, help="the candidate digest, when known")
    check.add_argument("--rehearsal-structure-only", action="store_true",
                       help="check a rehearsal record's structure only; never release evidence")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rehearsal = args.rehearsal_structure_only
    errors = validate(args.record, profile=args.profile, tag=args.tag, repo=args.repo,
                      integrity=args.integrity, digest=args.digest,
                      rehearsal_structure_only=rehearsal)
    if rehearsal:
        print(f"NOTE: --rehearsal-structure-only: {REHEARSAL_NOTICE}", file=sys.stderr)
    if errors:
        print(f"REFUSING ({args.profile}{', rehearsal structure' if rehearsal else ''}): "
              f"{args.record}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if rehearsal:
        print(f"structurally valid {args.profile} REHEARSAL record: {args.record} — "
              f"{record['spec']} at {record['tag']} ({record['sha']}), "
              f"{len(record['criteria'])} criteria pass; {REHEARSAL_NOTICE}")
        return 0
    print(f"valid {args.profile} record: {args.record} — {record['spec']} at {record['tag']} "
          f"({record['sha']}), {record['release_kind']} release, "
          f"{len(record['criteria'])} criteria pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
