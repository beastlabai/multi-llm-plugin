#!/usr/bin/env python3
"""The codex plugin's release gate: which container rows it requires, and
whether they ran (plans/codex-native-plugin.md A4; plans/codex-plugin-publish.md).

``make codex-package-check`` used to end in an unconditional refusal, because
A4 rows 6b (the offline Git install) and 14 (the newest codex) were not wired
and a stamp over rows nobody ran would authorize a tag. This module is what
replaced the refusal. It owns three things and nothing else:

* **The tiers.** One gate runs the same zero-quota package rows on three
  codex builds: ``pinned`` (the container's lock, every package row),
  ``minimum`` (the declared plugin floor, ``hosts.codex.package.min_host_version``)
  and ``ceiling`` (npm's newest ``@openai/codex``, resolved when the gate runs —
  A4 row 14). Which rows each tier requires is DATA, the ``gate_tiers`` list on
  each ``tests/container/host_matrix.yaml`` probe row; :data:`EVERY_TIER` is the
  floor that data may not fall below.
* **The versions.** :func:`resolve_version` reads the pin, the declared floor,
  or asks npm. A failed fetch is an error naming what it could not reach, never
  a silent skip (A4 row 14).
* **The accounting.** :func:`check_tier` reads the run records the container
  harness wrote for this invocation and refuses unless every required row has a
  record newer than the tier's start, graded ``pass`` by both the container and
  the grader, on the image and codex version the tier asked for, over an
  installed tree whose digest is this checkout's. pytest's exit code alone is
  not that: a module skipped for want of a Docker daemon exits 0 too. The
  evidence it writes under ``.gate/`` (gitignored) is what ``build.py
  --write-stamp`` requires and embeds.

What the evidence says about row 1 is deliberately narrow. Upstream's
``validate_plugin.py`` has shipped in no codex since 0.156.1; the
``upstream-validator`` row records ``absent`` or runs what it finds. The
evidence carries that status verbatim, and :func:`describe` prints
"upstream validator: absent", never "validated". The compatibility layout is
otherwise held to restated documentation rules (``ingestion_rules``) and graded
by the host's own ingestion on all three builds — rows 6b, 7 and 13.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "multi-llm"
CONTAINER_DIR = SKILL_DIR / "tests" / "container"
HOST_MATRIX = CONTAINER_DIR / "host_matrix.yaml"
RECORDS_DIR = CONTAINER_DIR / "records"
CONTAINER_PACKAGE_JSON = CONTAINER_DIR / "package.json"
CAPABILITIES = SKILL_DIR / "hosts" / "capabilities.yaml"
PLUGIN_DIGEST = PACKAGE_DIR / "multi-llm" / "dist" / "tree.sha256"
PLUGIN_MANIFEST = PACKAGE_DIR / "multi-llm" / ".codex-plugin" / "plugin.json"

#: Where each tier's evidence lands. Gitignored beside the stamp: it attests to
#: the machine that ran it, exactly like the stamp it feeds.
EVIDENCE_DIR = PACKAGE_DIR / ".gate"

TIERS = ("pinned", "minimum", "ceiling")

#: The probe rows every tier must schedule, whatever `gate_tiers` says: A4 row
#: 14's list (Phase 1 item 1's two rows, rows 7 and 13 — the version read, the
#: features printout, the seed, the locator, `tools[]`, the call, `mcp get`,
#: the stdio `tools/call`), row 6b's Git install and row 1's validator search.
EVERY_TIER = (
    "floor", "features", "seed", "scripted-call", "loader-mcp-get",
    "loader-prompt-input", "loader-tools", "runnability", "git-route",
    "upstream-validator",
)

#: The `codex features list` rows the plan recorded (A4 row 14). A diff is
#: written into the evidence and the release note, never a red by itself: the
#: row is red only when the newest codex stops listing the skill or answering
#: the tool, which the loader rows grade.
EXPECTED_FEATURES = {
    "plugins": ["stable", True],
    "plugin_hooks": ["removed", False],
    "multi_agent": ["stable", True],
}

NPM_PACKAGE = "@openai/codex"
_SEMVER = re.compile(r"\d+\.\d+\.\d+")


class GateError(Exception):
    """A refusal, worded for the maintainer reading the make output."""


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    import yaml  # the dev group carries it; this runs under `uv run`

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def probe_rows(matrix: Optional[Mapping] = None) -> List[dict]:
    """The codex `package_probes` rows of host_matrix.yaml."""
    document = matrix if matrix is not None else _load_yaml(HOST_MATRIX)
    codex = ((document or {}).get("hosts") or {}).get("codex") or {}
    return [dict(row) for row in codex.get("package_probes") or []]


def required_rows(tier: str, matrix: Optional[Mapping] = None) -> List[str]:
    """The row ids ``tier`` requires, in matrix order."""
    if tier not in TIERS:
        raise GateError(f"unknown gate tier {tier!r}; the tiers are {', '.join(TIERS)}")
    return [row["id"] for row in probe_rows(matrix)
            if tier in (row.get("gate_tiers") or [])]


def schedule_problems(matrix: Optional[Mapping] = None) -> List[str]:
    """What is wrong with the `gate_tiers` data, as readable lines."""
    rows = probe_rows(matrix)
    problems: List[str] = []
    if not rows:
        return ["host_matrix.yaml declares no codex package_probes"]
    for row in rows:
        tiers = row.get("gate_tiers")
        if not isinstance(tiers, list) or not tiers:
            problems.append(f"{row.get('id')}: no gate_tiers list")
            continue
        unknown = sorted(set(map(str, tiers)) - set(TIERS))
        if unknown:
            problems.append(f"{row.get('id')}: unknown gate tier(s) {unknown}")
        if "pinned" not in tiers:
            problems.append(f"{row.get('id')}: every package row runs on the pinned tier")
    by_probe = {row.get("probe"): row for row in rows}
    for probe in EVERY_TIER:
        row = by_probe.get(probe)
        if row is None:
            problems.append(f"no package probe row runs `{probe}`, which every tier requires")
            continue
        missing = [tier for tier in TIERS if tier not in (row.get("gate_tiers") or [])]
        if missing:
            problems.append(f"{row['id']} (`{probe}`) is not scheduled on {missing}")
        # Not `blocking:` — that word is the matrix's (a recorded-only row such
        # as `features` stays non-blocking). The gate requires every row it
        # schedules to come back graded `pass`, whichever kind it is.
    return problems


# ---------------------------------------------------------------------------
# The versions
# ---------------------------------------------------------------------------

def pinned_version() -> str:
    pins = json.loads(CONTAINER_PACKAGE_JSON.read_text(encoding="utf-8"))
    return str(pins["dependencies"][NPM_PACKAGE])


def minimum_version() -> str:
    data = _load_yaml(CAPABILITIES)
    return str(data["hosts"]["codex"]["package"]["min_host_version"])


def newest_version(runner=subprocess.run) -> str:
    """npm's newest ``@openai/codex``, asked now. Raises naming the failure."""
    argv = ["npm", "view", NPM_PACKAGE, "version", "--json"]
    try:
        done = runner(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"A4 row 14: could not ask npm for the newest "
                        f"{NPM_PACKAGE} ({' '.join(argv)}): {exc}") from exc
    if done.returncode != 0:
        raise GateError(f"A4 row 14: `{' '.join(argv)}` exited {done.returncode}: "
                        f"{(done.stderr or done.stdout or '').strip()[-400:]}")
    try:
        value = json.loads(done.stdout)
    except ValueError:
        value = (done.stdout or "").strip()
    if isinstance(value, list):
        value = value[-1] if value else ""
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise GateError(f"A4 row 14: npm answered {done.stdout.strip()[:200]!r} for "
                        f"the newest {NPM_PACKAGE}, which is not a version")
    return value


def resolve_version(tier: str) -> str:
    if tier == "pinned":
        return pinned_version()
    if tier == "minimum":
        return minimum_version()
    if tier == "ceiling":
        return newest_version()
    raise GateError(f"unknown gate tier {tier!r}")


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------

def _harness():
    if str(SKILL_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_DIR))
    from tests.container import harness  # noqa: WPS433 — the record vocabulary

    return harness


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_STAMP = re.compile(r"\d{8}T\d{6}Z")


def _records_for(row_id: str, since: str, records_dir: Path) -> List[Path]:
    """Every record of ``row_id`` started at or after ``since``, oldest first."""
    found = []
    for path in sorted(records_dir.glob(f"*-{row_id}.json")):
        stamp = path.name[: -len(f"-{row_id}.json")]
        if _STAMP.fullmatch(stamp) and stamp >= since:
            found.append(path)
    return found


def _version_key(version: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", str(version))[:3])


# ---------------------------------------------------------------------------
# The session: one gate invocation's evidence, never a mix of several
# ---------------------------------------------------------------------------

SESSION_NAME = "session.json"


def begin_session(directory: Path = EVIDENCE_DIR) -> Dict[str, str]:
    """Start a gate session: earlier evidence is deleted, a fresh id written.

    ``make codex-package-check`` begins one before any tier runs, and the stamp
    accepts only evidence carrying the current id — so a tier that went green
    weeks ago, under another harness, cannot be stamped beside today's.
    """
    import uuid

    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    session = {"id": uuid.uuid4().hex,
               "started_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                   "%Y-%m-%dT%H:%M:%SZ")}
    (directory / SESSION_NAME).write_text(json.dumps(session, indent=2) + "\n",
                                          encoding="utf-8")
    return session


def current_session(directory: Path = EVIDENCE_DIR) -> Optional[str]:
    session = _read_json(directory / SESSION_NAME)
    return str(session["id"]) if isinstance(session, dict) and session.get("id") else None


def _session_for_check(directory: Path) -> str:
    """The running session's id; a tier run on its own starts one."""
    return current_session(directory) or begin_session(directory)["id"]


def checkout_digest() -> str:
    return PLUGIN_DIGEST.read_text(encoding="utf-8").strip()


def manifest_version() -> str:
    return str(json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))["version"])


def _row_problems(row_id: str, record: Mapping, bundle: Path, image: str,
                  version: str, digest: str, reasons: Sequence[str]) -> List[str]:
    problems: List[str] = []
    if record.get("outcome") != "pass":
        problems.append(f"{row_id}: outcome {record.get('outcome')!r}")
    for row in record.get("rows") or []:
        reason = str(row.get("reason") or "")
        if any(marker in reason for marker in reasons):
            problems.append(f"{row_id}: not graded by both verdicts ({reason[:200]})")
        if row.get("cli_version") != version:
            problems.append(f"{row_id}: ran codex {row.get('cli_version')!r}, the tier "
                            f"asked for {version}")
    tag = (record.get("image") or {}).get("tag")
    if tag != image:
        problems.append(f"{row_id}: ran on image {tag!r}, the tier asked for {image}")
    install = _read_json(bundle / "plugin-install.json") or {}
    if install.get("installed_digest") != digest:
        problems.append(f"{row_id}: installed tree digest "
                        f"{install.get('installed_digest')!r}, this checkout's is {digest} "
                        f"— the image was built from other bytes (rebuild it)")
    return problems


def features_diff(printout: Mapping) -> Dict[str, Any]:
    """``{name: {"expected": …, "measured": …}}`` for each recorded row that moved."""
    diff = {}
    for name, expected in EXPECTED_FEATURES.items():
        measured = printout.get(name)
        if measured != expected:
            diff[name] = {"expected": expected, "measured": measured}
    return diff


def check_tier(tier: str, image: str, since: str, version: str, *,
               records_dir: Path = RECORDS_DIR, container_dir: Path = CONTAINER_DIR,
               matrix: Optional[Mapping] = None, digest: Optional[str] = None,
               expected_version: Optional[str] = None,
               ) -> Dict[str, Any]:
    """The tier's evidence, or :class:`GateError` naming every missing row.

    ``version`` is what the caller says it ran; ``expected_version`` (default:
    :func:`resolve_version` asked now — npm, for the ceiling) is what the tier
    is FOR. They must agree, so a tier cannot be green on a codex it was not
    meant to measure (`codex-gate-derived TIER=minimum CODEX_VERSION=<the pin>`).
    """
    if not _STAMP.fullmatch(since or ""):
        raise GateError(f"--since {since!r} is not a YYYYmmddTHHMMSSZ stamp; it is "
                        "compared lexically with the record names")
    expected = expected_version if expected_version is not None else resolve_version(tier)
    if version != expected:
        raise GateError(f"tier {tier} is for codex {expected}, but the rows were run on "
                        f"{version}")
    rows = required_rows(tier, matrix)
    probe_of = {row["id"]: row.get("probe") for row in probe_rows(matrix)}
    if not rows:
        raise GateError(f"tier {tier}: host_matrix.yaml schedules no package row on it")
    digest = digest or checkout_digest()
    harness = _harness()
    reasons = (harness.PROVISIONAL_REASON, harness.UNGRADED_REASON)
    problems: List[str] = []
    evidence_rows: Dict[str, Any] = {}
    image_digests = set()
    features: Dict[str, Any] = {}
    validator: Dict[str, Any] = {}
    git_route: Dict[str, Any] = {}
    stamps: set = set()
    for row_id in rows:
        found = _records_for(row_id, since, records_dir)
        if not found:
            problems.append(f"{row_id}: no run record at or after {since} — the row "
                            f"never ran in this gate")
            continue
        path = found[-1]
        stamps.add(path.name[: -len(f"-{row_id}.json")])
        record = _read_json(path) or {}
        bundle = container_dir / str((record.get("bundle") or {}).get("root") or "")
        problems += _row_problems(row_id, record, bundle, image, version, digest, reasons)
        image_digests.add((record.get("image") or {}).get("digest"))
        evidence_rows[row_id] = {"record": path.name, "outcome": record.get("outcome")}
        probe = _read_json(bundle / f"probe-{probe_of.get(row_id)}.json") or {}
        if probe.get("probe") == "features":
            features = {"printout": probe.get("features") or {},
                        "diff": features_diff(probe.get("features") or {})}
        elif probe.get("probe") == "upstream-validator":
            validator = {"status": probe.get("status"),
                         "system_skills": probe.get("system_skills"),
                         "validators": [{k: v for k, v in run.items()
                                         if k in ("path", "sha256", "exit")}
                                        for run in probe.get("validators") or []]}
        elif probe.get("probe") == "git-route":
            install = _read_json(bundle / "plugin-install.json") or {}
            git_route = {
                "list_entry_policy": {
                    key: ((probe.get("refresh") or {}).get("list_entry") or {}).get(key)
                    for key in ("installPolicy", "authPolicy")},
                "re_add_other_ref_exit":
                    ((probe.get("re_add_other_ref") or {}).get("command") or {}).get("exit"),
                "versions_after_upgrade":
                    (probe.get("refresh") or {}).get("versions_after_upgrade"),
                "install_source": install.get("install_source"),
            }
    if len(stamps) > 1:
        # One pytest module run stamps every row it drives with the same start
        # time, so rows from two stamps are two runs mixed — possibly another
        # lane's, in this shared records directory.
        problems.append(f"tier {tier}: the newest records come from {len(stamps)} "
                        f"different runs ({', '.join(sorted(stamps))}); re-run the tier")
    image_digests.discard(None)
    if len(image_digests) > 1:
        problems.append(f"tier {tier}: rows ran on {len(image_digests)} different images")
    if problems:
        raise GateError(f"tier {tier} ({image}, codex {version}) is not green:\n"
                        + "\n".join(f"    {line}" for line in problems))
    return {
        "tier": tier, "image": image,
        "image_digest": next(iter(image_digests), None),
        "codex_version": version, "digest": digest, "version": manifest_version(),
        "since": since,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "run": next(iter(stamps), None),
        "rows": evidence_rows, "features": features,
        "upstream_validator": validator, "git_route": git_route,
    }


def write_evidence(evidence: Mapping, directory: Path = EVIDENCE_DIR) -> Path:
    """Write one tier's evidence under the running session (starting one if
    the tier runs on its own)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamped = {**evidence, "session": evidence.get("session") or _session_for_check(directory)}
    path = directory / f"{evidence['tier']}.json"
    path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_evidence(directory: Path = EVIDENCE_DIR) -> Dict[str, Any]:
    return {tier: _read_json(directory / f"{tier}.json") for tier in TIERS}


def expected_versions() -> Dict[str, str]:
    """The versions the two owned tiers must have run: read NOW, so a pin or a
    declared floor that moved after a tier ran makes its evidence stale."""
    return {"pinned": pinned_version(), "minimum": minimum_version()}


def evidence_problems(evidence: Mapping[str, Any], digest: str, version: str,
                      matrix: Optional[Mapping] = None, *,
                      session: Optional[str] = None,
                      expected: Optional[Mapping[str, str]] = None) -> List[str]:
    """Why ``evidence`` cannot back a stamp over ``digest``/``version``.

    ``session`` is the current gate session's id (``None``: there is none, which
    is itself a refusal); ``expected`` the owned tiers' versions
    (:func:`expected_versions`). The ceiling was bound to npm when it was
    checked; here it must at least not sit below either owned tier.
    """
    expected = dict(expected if expected is not None else expected_versions())
    problems: List[str] = []
    if not session:
        problems.append("no gate session: run `make codex-package-check`, which "
                        "begins one before any tier")
    for tier in TIERS:
        item = evidence.get(tier)
        if not isinstance(item, Mapping):
            problems.append(f"tier {tier}: no evidence (run `make codex-gate-{tier}`)")
            continue
        if session and item.get("session") != session:
            problems.append(f"tier {tier}: its evidence belongs to another gate session; "
                            "every tier must come from the same run")
        if item.get("digest") != digest:
            problems.append(f"tier {tier}: ran over digest {item.get('digest')!r}, the "
                            f"tree is {digest}")
        if item.get("version") != version:
            problems.append(f"tier {tier}: ran over version {item.get('version')!r}, "
                            f"the tree is {version}")
        if tier in expected and item.get("codex_version") != expected[tier]:
            problems.append(f"tier {tier}: ran codex {item.get('codex_version')!r}, the "
                            f"tier is for {expected[tier]}")
        missing = sorted(set(required_rows(tier, matrix)) - set(item.get("rows") or {}))
        if missing:
            problems.append(f"tier {tier}: no passing record for {missing}")
        if not (item.get("upstream_validator") or {}).get("status"):
            problems.append(f"tier {tier}: row 1's validator search left no status")
    ceiling = (evidence.get("ceiling") or {}).get("codex_version")
    floor = [v for v in expected.values() if v]
    if ceiling and floor and _version_key(ceiling) < max(_version_key(v) for v in floor):
        problems.append(f"tier ceiling: codex {ceiling} is below a tier it should top "
                        f"({', '.join(floor)})")
    return problems


def describe(evidence: Mapping[str, Any]) -> List[str]:
    """One line per tier, in words that never call an absence a pass."""
    lines = []
    for tier in TIERS:
        item = evidence.get(tier) or {}
        if not item:
            lines.append(f"{tier}: no evidence")
            continue
        status = (item.get("upstream_validator") or {}).get("status")
        validator = ("upstream validator: absent (nothing to run; the layout is held "
                     "to restated rules and graded by codex's own ingestion)"
                     if status == "absent" else f"upstream validator: {status}")
        diff = (item.get("features") or {}).get("diff") or {}
        features = (f"features moved: {sorted(diff)}" if diff
                    else "features as recorded")
        lines.append(f"{tier}: codex {item.get('codex_version')}, "
                     f"{len(item.get('rows') or {})} rows pass; {validator}; {features}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="release_gate.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="the gate_tiers data is complete")
    sub.add_parser("begin", help="start a gate session: delete earlier evidence")
    rows = sub.add_parser("rows", help="print the row ids a tier requires")
    rows.add_argument("tier", choices=TIERS)
    version = sub.add_parser("version", help="print the codex version a tier runs")
    version.add_argument("tier", choices=TIERS)
    check = sub.add_parser("check", help="grade a tier's run records; write its evidence")
    check.add_argument("tier", choices=TIERS)
    check.add_argument("--image", required=True)
    check.add_argument("--since", required=True, help="the tier's start, YYYYmmddTHHMMSSZ")
    check.add_argument("--version", required=True, dest="codex_version")
    sub.add_parser("summary", help="describe the evidence on disk")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "preflight":
            problems = schedule_problems()
            if problems:
                for line in problems:
                    print(f"REFUSING: {line}", file=sys.stderr)
                return 1
            print("gate rows: " + "; ".join(
                f"{tier} {len(required_rows(tier))}" for tier in TIERS))
        elif args.command == "begin":
            print(f"gate session {begin_session()['id']} (earlier evidence deleted)")
        elif args.command == "rows":
            print("\n".join(required_rows(args.tier)))
        elif args.command == "version":
            print(resolve_version(args.tier))
        elif args.command == "check":
            evidence = check_tier(args.tier, args.image, args.since, args.codex_version)
            path = write_evidence(evidence)
            print(f"tier {args.tier}: {len(evidence['rows'])} rows pass on codex "
                  f"{args.codex_version} ({args.image}); evidence {path}")
            for line in describe({args.tier: evidence}):
                print(f"  {line}")
        elif args.command == "summary":
            for line in describe(load_evidence()):
                print(line)
    except GateError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
