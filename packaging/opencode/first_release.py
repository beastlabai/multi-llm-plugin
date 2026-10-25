#!/usr/bin/env python3
"""The laptop FIRST release of ``opencode-multi-llm``: ``make opencode-publish-first``.

Run it as::

    uv run --project skills/multi-llm -- python packaging/opencode/first_release.py [--dry-run]

**Why a laptop path exists at all.** npm's trusted publishing (the OIDC route the
workflow uses, and the only route that signs provenance) is configured on an
existing package: the ``npm trust`` docs still say "The package you're
configuring must already exist" (rechecked 2026-10-25). So the first version is
uploaded from the maintainer's machine, authenticated interactively and
**without** ``--provenance``, which npm can only generate on a supported CI
provider. Every later version goes through the workflow dispatch, with
provenance. Step 4 of the publishing plan rechecks the docs right before the
first release; if npm has grown first-publish OIDC by then, this target is not
used.

**What it refuses on.** Every preflight below runs, in both modes, and each
failure is reported under its own condition name, so "the stamp is stale" is
never the same message as "the macOS leg failed":

* ``clean-tree`` — ``git status --porcelain`` is empty;
* ``stamp`` — ``build.py --check-stamp`` passes, and the stamp is the full,
  schema-2 kind that binds the gated tarball (path, integrity, sha256, files,
  digest), for this version, this HEAD and the single registry;
* ``tag`` — ``opencode-v<version>`` (the version read from ``package.json``)
  exists, is annotated (``git cat-file -t`` prints ``tag``) and peels to HEAD;
* ``origin-master`` — after ``git fetch origin``, HEAD is an ancestor of
  ``origin/master``, so an unpushed commit cannot be published;
* ``required-checks`` — ``test (<os>)`` for every ``matrix.os`` in ``test.yml``
  (derived from the file at HEAD, never typed here) and ``package``, all
  ``completed``/``success`` on HEAD, each from a GitHub Actions run of
  ``.github/workflows/test.yml`` on that SHA. Missing, pending, skipped,
  cancelled and failed all refuse, and so does any failed ``gh`` call;
* ``bump-check`` — ``build.py --check-bump`` (the shared-version check that reads
  both families' tags and npm's version list). A ``build.py`` without that flag
  refuses: the check is a precondition, not an option;
* ``npm-auth`` — ``npm whoami`` against the single registry, so a doomed upload
  is refused before it starts rather than half-way through a prompt;
* ``gated-tarball`` — the file at the stamp's path still recomputes, through
  ``build.tarball_facts``, to the stamp's integrity, sha256, inventory and
  extracted tree digest, and ``build.py --scan-tarball`` passes on it;
* ``row-9`` — ``registry_absent.mjs``, last, immediately before the upload.

**What it uploads: the exact tarball the gate built, scanned and loaded.**
``npm publish <gated>.tgz --ignore-scripts --access public --registry=<single>``.
Never the directory: the manifest's ``prepack`` hook would rebuild ``dist/``
there and npm would upload whatever that rebuild produced, bytes no gate
checked. Measured, a tarball upload runs no lifecycle scripts, and
``--ignore-scripts`` keeps any other script from repacking it. The argv is
checked before it runs (:func:`check_publish_argv`).

**After the upload.** npm's exit status is not confirmation: the registry and its
CDN can lag. The target polls ``npm view <spec> dist.integrity`` with the
explicit registry and a fresh cache, for a bounded time, until the version is
visible, and requires it to equal the gated integrity. An expired poll is "not
yet confirmed", not a mismatch: repeat the read by hand (step 4) before pushing
the tag. A different integrity is a mismatch after publication: do not push the
tag, follow Recovery. Either way the version is permanently consumed.

**Evidence.** A JSON file under ``packaging/opencode/.gate/release/``
(gitignored): the tag object ID, the commit SHA, every required check run's ID
and workflow run ID, the gated integrity, the publish time and argv, and the
confirming read. In ``release/`` because ``release_gate.py begin`` clears the
session files at the top of ``.gate/``, and this record must outlive the next
gate session.

**Every subprocess goes through one injectable runner** (:class:`Runner`), so the
suite drives each refusal with stubbed ``git``, ``gh``, ``npm`` and ``build.py``.
Nothing here publishes on import, and ``--dry-run`` stops after the preflights.

Exit codes: 0 published and confirmed (or, with ``--dry-run``, every preflight
green); 1 refused before upload; 2 usage; 3 ``npm publish`` failed; 4 published
but not yet confirmed; 5 published with a different integrity.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
BUILD = PACKAGE_DIR / "build.py"
ROW9 = PACKAGE_DIR / "registry_absent.mjs"
MANIFEST = PACKAGE_DIR / "package.json"
STAMP = PACKAGE_DIR / ".package-check.stamp"
EVIDENCE_DIR = PACKAGE_DIR / ".gate" / "release"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import registry  # noqa: E402

#: The repository whose check runs bind the release, and its default branch.
GH_REPO = "beastlabai/multi-llm-plugin"
DEFAULT_BRANCH = "master"
TAG_PREFIX = "opencode-v"
#: Where a release-gate record is committed, and how its name ends. Named only
#: to explain a clean-tree refusal that one of them caused.
RECORDS_PREFIX = "skills/multi-llm/tests/container/records/"
RECORD_SUFFIX = "-opencode-release-gate.json"

#: The CI workflow whose jobs are the required check runs, and the two jobs.
TEST_WORKFLOW = ".github/workflows/test.yml"
TEST_JOB = "test"
PACKAGE_JOB = "package"

#: The laptop path never signs provenance; the workflow's publish always does.
#: Named so the choice is a value the suite can read, not an absence.
LOCAL_PROVENANCE = False

#: Bounded npm reads, as row 9 bounds its own.
NPM_FETCH_FLAGS = (
    "--fetch-retries=2",
    "--fetch-retry-mintimeout=2000",
    "--fetch-retry-maxtimeout=10000",
    "--fetch-timeout=30000",
)

#: How long the post-publish read keeps trying before saying "not yet".
POLL_SECONDS = 300.0
POLL_INTERVAL = 10.0

#: The environment never handed to a child: the dirty-tree opt-in the Makefile
#: exports, and anything that would turn provenance on behind the argv's back.
SCRUBBED_ENV = ("MULTI_LLM_ALLOW_DIRTY", "npm_config_provenance", "NPM_CONFIG_PROVENANCE")

EXIT_OK, EXIT_REFUSED, EXIT_USAGE = 0, 1, 2
EXIT_PUBLISH_FAILED, EXIT_UNCONFIRMED, EXIT_MISMATCH = 3, 4, 5


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

@dataclass
class Completed:
    argv: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner:
    """Runs one subprocess. The suite replaces it; nothing else spawns.

    ``capture=False`` is the upload: npm's interactive login/2FA needs the
    terminal, so stdio is inherited. Every captured call gets ``stdin`` from
    ``/dev/null``, so a credential prompt from git or gh fails instead of hanging.
    """

    def __call__(self, argv: Sequence[str], *, capture: bool = True,
                 timeout: Optional[float] = 120.0) -> Completed:
        argv = [str(a) for a in argv]
        scrubbed = {name.lower() for name in SCRUBBED_ENV}
        env = {k: v for k, v in os.environ.items() if k.lower() not in scrubbed}
        env.update(GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1")
        try:
            done = subprocess.run(
                argv, cwd=REPO_ROOT, env=env, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL if capture else None,
                capture_output=capture,
            )
        except FileNotFoundError as exc:
            return Completed(argv, 127, "", f"{argv[0]}: not found ({exc})")
        except subprocess.TimeoutExpired:
            return Completed(argv, 124, "", f"{argv[0]}: no answer within {timeout:.0f} s")
        return Completed(argv, done.returncode, done.stdout or "", done.stderr or "")


class Refusal(Exception):
    """One failed precondition, under its condition name."""

    def __init__(self, condition: str, message: str):
        super().__init__(message)
        self.condition = condition
        self.message = message


def _tail(done: Completed) -> str:
    """The line worth quoting from a failed call: npm's error code when it
    printed one (its last line is only the debug-log path), else the last line."""
    text = (done.stderr or done.stdout or "").strip()
    code = re.search(r"^npm (?:ERR!|error) code (\S+)", text, re.M)
    if code:
        return f"npm error code {code.group(1)}"
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else f"exit {done.returncode}"


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The required check runs, derived from test.yml
# ---------------------------------------------------------------------------

def required_check_names(test_yml: str) -> List[str]:
    """``test (<os>)`` for every ``matrix.os`` of the ``test`` job, then
    ``package``: the check-run names GitHub gives those jobs.

    Derived, never typed, so a matrix change cannot leave this target checking a
    leg that no longer runs or skipping a new one. Refuses (:class:`Refusal`) a
    shape whose names it cannot derive — a job ``name:``, a second matrix
    dimension, ``include``/``exclude`` — rather than guessing at them.
    """
    import yaml

    try:
        workflow = yaml.safe_load(test_yml) or {}
    except yaml.YAMLError as exc:
        raise Refusal("required-checks", f"{TEST_WORKFLOW} is not readable YAML ({exc})")
    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        raise Refusal("required-checks", f"{TEST_WORKFLOW} declares no jobs")
    names: List[str] = []
    for job_id in (TEST_JOB, PACKAGE_JOB):
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            raise Refusal("required-checks", f"{TEST_WORKFLOW} has no `{job_id}` job")
        if "name" in job:
            raise Refusal("required-checks",
                          f"{TEST_WORKFLOW}: job `{job_id}` sets `name:`, so its check-run "
                          "names are not derivable here; extend required_check_names()")
        matrix = (job.get("strategy") or {}).get("matrix")
        if matrix is None:
            if job_id == TEST_JOB:
                raise Refusal("required-checks", f"{TEST_WORKFLOW}: job `test` has no matrix")
            names.append(job_id)
            continue
        if job_id != TEST_JOB:
            raise Refusal("required-checks",
                          f"{TEST_WORKFLOW}: job `{job_id}` gained a matrix; extend "
                          "required_check_names()")
        if not isinstance(matrix, dict) or set(matrix) != {"os"}:
            raise Refusal("required-checks",
                          f"{TEST_WORKFLOW}: the `test` matrix must have exactly one "
                          f"dimension, `os`; found {sorted(matrix) if isinstance(matrix, dict) else matrix!r}")
        legs = matrix["os"]
        if not isinstance(legs, list) or not legs or not all(isinstance(o, str) for o in legs):
            raise Refusal("required-checks", f"{TEST_WORKFLOW}: `matrix.os` is not a list of names")
        names.extend(f"{job_id} ({leg})" for leg in legs)
    return names


_RUN_URL = re.compile(r"/actions/runs/(\d+)/job/(\d+)")


def _json_stream(text: str) -> List[Any]:
    """Every JSON value in *text*: ``gh --paginate --jq`` prints one per line."""
    decoder, values, index = json.JSONDecoder(), [], 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return values
        value, index = decoder.raw_decode(text, index)
        values.append(value)


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------

@dataclass
class Release:
    runner: Callable[..., Completed] = field(default_factory=Runner)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    out: Callable[[str], None] = field(default=lambda line: print(line, flush=True))
    err: Callable[[str], None] = field(
        default=lambda line: print(line, file=sys.stderr, flush=True))
    stamp_path: Path = STAMP
    manifest_path: Path = MANIFEST
    evidence_dir: Path = EVIDENCE_DIR
    poll_seconds: float = POLL_SECONDS
    poll_interval: float = POLL_INTERVAL

    def __post_init__(self) -> None:
        self.facts: Dict[str, Any] = {}
        self.refusals: List[Refusal] = []

    # -- plumbing ----------------------------------------------------------

    def _run(self, argv: Sequence[str], condition: str, what: str, **kwargs) -> Completed:
        done = self.runner(list(argv), **kwargs)
        if done.returncode != 0:
            raise Refusal(condition, f"{what} failed ({_tail(done)})")
        return done

    def _git(self, *args: str, condition: str) -> str:
        return self._run(["git", *args], condition, f"`git {' '.join(args)}`").stdout.strip()

    def _check(self, condition: str, step: Callable[[], None]) -> None:
        try:
            try:
                step()
            except Refusal:
                raise
            except Exception as exc:  # a preflight that cannot run has not passed
                raise Refusal(condition, f"could not be checked ({type(exc).__name__}: {exc})")
        except Refusal as refusal:
            self.refusals.append(refusal)
            self.err(f"REFUSING ({refusal.condition}): {refusal.message}")
        else:
            self.out(f"  ok  {condition}")

    # -- the preflights ------------------------------------------------------

    def read_manifest(self) -> None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refusal("manifest", f"{self.manifest_path}: {exc}")
        if not isinstance(manifest, dict):
            raise Refusal("manifest", f"{self.manifest_path} is not a JSON object")
        name, version = manifest.get("name"), manifest.get("version")
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise Refusal("manifest", f"{self.manifest_path} names no name/version")
        self.facts.update(name=name, version=version, spec=f"{name}@{version}",
                          tag=f"{TAG_PREFIX}{version}")
        try:
            self.facts["registry"] = registry.single_registry()
        except registry.RegistryError as exc:
            raise Refusal("manifest", str(exc))
        self.facts["head"] = self._git("rev-parse", "--verify", "HEAD", condition="manifest")

    def check_clean_tree(self) -> None:
        status = self._git("status", "--porcelain", condition="clean-tree")
        if status:
            lines = status.splitlines()
            listing = "; ".join(lines[:5]) + (f"; … {len(lines) - 5} more" if len(lines) > 5 else "")
            message = f"{len(lines)} uncommitted or untracked change(s): {listing}"
            if any(RECORDS_PREFIX in line and line.rstrip().rstrip('"').endswith(RECORD_SUFFIX)
                   for line in lines):
                message += (
                    f". A release-gate record belongs in {RECORDS_PREFIX} only in the default "
                    "branch's own commit after the fast-forward (plan step 2), never in the "
                    f"tag's tree: commit it there, then check {self.facts.get('tag') or 'the tag'}'s "
                    "commit out again, clean, before `make opencode-publish-first`")
            raise Refusal("clean-tree", message)

    def check_stamp(self) -> None:
        self._run([sys.executable, str(BUILD), "--check-stamp"], "stamp",
                  "`build.py --check-stamp`", timeout=900.0)
        try:
            stamp = json.loads(self.stamp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refusal("stamp", f"{self.stamp_path}: {exc}")
        if not isinstance(stamp, dict):
            raise Refusal("stamp", f"{self.stamp_path} is not a JSON object")
        if stamp.get("schema") != 2 or stamp.get("tier") != "full":
            raise Refusal("stamp", f"{self.stamp_path} is not a full schema-2 stamp "
                          f"(schema={stamp.get('schema')!r}, tier={stamp.get('tier')!r}); "
                          "only a schema-2 stamp binds the gated tarball")
        tarball = stamp.get("tarball")
        keys = ("path", "integrity", "sha256", "files", "digest")
        if not isinstance(tarball, dict) or any(not tarball.get(k) for k in keys):
            raise Refusal("stamp", f"{self.stamp_path} carries no complete `tarball` binding "
                          f"({', '.join(keys)})")
        problems = []
        if stamp.get("version") != self.facts["version"]:
            problems.append(f"version {stamp.get('version')!r}, package.json {self.facts['version']!r}")
        if stamp.get("candidate_sha") != self.facts["head"]:
            problems.append(f"candidate_sha {stamp.get('candidate_sha')!r}, HEAD {self.facts['head']}")
        if registry.normalize(stamp.get("registry") or "") != self.facts["registry"]:
            problems.append(f"registry {stamp.get('registry')!r}, single {self.facts['registry']}")
        if tarball.get("digest") != stamp.get("digest"):
            problems.append(f"tarball digest {tarball.get('digest')} differs from the staged "
                            f"digest {stamp.get('digest')}")
        if problems:
            raise Refusal("stamp", "the stamp does not bind this release: " + "; ".join(problems))
        self.facts["stamp"] = stamp

    def check_tag(self) -> None:
        tag = self.facts["tag"]
        ref = f"refs/tags/{tag}"
        kind = self.runner(["git", "cat-file", "-t", ref])
        if kind.returncode != 0:
            raise Refusal("tag", f"{tag} does not exist. Create it on HEAD: "
                          f"git tag -a {tag} -m \"OpenCode plugin {self.facts['version']}\"")
        if kind.stdout.strip() != "tag":
            raise Refusal("tag", f"{tag} is a lightweight tag ({kind.stdout.strip() or '?'}); "
                          "the release needs an annotated one (git tag -a)")
        tagged = self._git("rev-parse", "--verify", f"{ref}^{{commit}}", condition="tag")
        if tagged != self.facts["head"]:
            raise Refusal("tag", f"{tag} names {tagged}, not HEAD {self.facts['head']}")
        self.facts["tag_object"] = self._git("rev-parse", "--verify", ref, condition="tag")

    def check_origin_master(self) -> None:
        self._run(["git", "fetch", "origin"], "origin-master", "`git fetch origin`", timeout=300.0)
        upstream = f"origin/{DEFAULT_BRANCH}"
        done = self.runner(["git", "merge-base", "--is-ancestor", "HEAD", upstream])
        if done.returncode == 1:
            raise Refusal("origin-master", f"HEAD {self.facts['head']} is not on {upstream}: "
                          "fast-forward master to this commit and push it first (step 2)")
        if done.returncode != 0:
            raise Refusal("origin-master", f"`git merge-base --is-ancestor HEAD {upstream}` "
                          f"failed ({_tail(done)})")
        self.facts["origin_master"] = self._git("rev-parse", upstream, condition="origin-master")

    def check_required_runs(self) -> None:
        sha = self.facts["head"]
        test_yml = self._git("show", f"{sha}:{TEST_WORKFLOW}", condition="required-checks")
        required = required_check_names(test_yml)
        listing = self._run(
            ["gh", "api", f"repos/{GH_REPO}/commits/{sha}/check-runs?filter=latest&per_page=100",
             "--paginate", "--jq",
             ".check_runs[] | {id, name, status, conclusion, head_sha, details_url, "
             "app: .app.slug}"],
            "required-checks", "`gh api .../check-runs`",
        )
        try:
            runs = _json_stream(listing.stdout)
        except ValueError as exc:
            raise Refusal("required-checks", f"`gh api .../check-runs` printed no JSON ({exc})")
        problems, recorded, workflow_runs = [], [], {}
        for name in required:
            mine = [r for r in runs if isinstance(r, dict) and r.get("name") == name
                    and r.get("app") == "github-actions"]
            if not mine:
                problems.append(f"{name}: missing")
                continue
            for run in mine:
                if run.get("head_sha") != sha:
                    problems.append(f"{name}: check run {run.get('id')} is for {run.get('head_sha')}")
                elif run.get("status") != "completed":
                    problems.append(f"{name}: {run.get('status') or 'pending'}")
                elif run.get("conclusion") != "success":
                    problems.append(f"{name}: {run.get('conclusion') or 'no conclusion'}")
                match = _RUN_URL.search(str(run.get("details_url") or ""))
                if not match:
                    problems.append(f"{name}: check run {run.get('id')} names no workflow run")
                    continue
                recorded.append({"name": name, "id": run.get("id"),
                                 "workflow_run_id": int(match.group(1)),
                                 "conclusion": run.get("conclusion")})
                workflow_runs.setdefault(int(match.group(1)), []).append(name)
        for run_id, names in sorted(workflow_runs.items()):
            done = self._run(
                ["gh", "api", f"repos/{GH_REPO}/actions/runs/{run_id}", "--jq",
                 "{id, path, head_sha, event, run_attempt}"],
                "required-checks", f"`gh api .../actions/runs/{run_id}`",
            )
            try:
                (info,) = _json_stream(done.stdout)
            except ValueError:
                info = None
            if not isinstance(info, dict):
                raise Refusal("required-checks", f"workflow run {run_id}: unreadable answer")
            if info.get("path") != TEST_WORKFLOW or info.get("head_sha") != sha:
                problems.append(f"{', '.join(names)}: workflow run {run_id} is "
                                f"{info.get('path')} on {info.get('head_sha')}, not "
                                f"{TEST_WORKFLOW} on {sha}")
            for entry in recorded:
                if entry["workflow_run_id"] == run_id:
                    entry.update(event=info.get("event"), run_attempt=info.get("run_attempt"))
        if problems:
            raise Refusal("required-checks", f"required check runs on {sha} are not all "
                          "green: " + "; ".join(problems))
        self.facts["required_checks"] = required
        self.facts["check_runs"] = recorded

    def check_bump(self) -> None:
        done = self.runner([sys.executable, str(BUILD), "--check-bump"], timeout=300.0)
        if done.returncode == 0:
            return
        if "--check-bump" in done.stderr and "unrecognized" in done.stderr:
            raise Refusal("bump-check", "build.py has no --check-bump: the shared-version "
                          "bump check is not in this tree, and the release cannot skip it")
        raise Refusal("bump-check", f"`build.py --check-bump` refused ({_tail(done)})")

    def check_npm_auth(self) -> None:
        done = self.runner(["npm", "whoami", f"--registry={self.facts['registry']}",
                            *NPM_FETCH_FLAGS], timeout=150.0)
        if done.returncode != 0:
            raise Refusal("npm-auth", f"not logged in to {self.facts['registry']} "
                          f"({_tail(done)}). Run: npm login --registry={self.facts['registry']}")
        self.facts["npm_user"] = done.stdout.strip()

    def check_gated_tarball(self) -> None:
        stamp = self.facts.get("stamp")
        if not stamp:
            raise Refusal("gated-tarball", "not checked: no usable stamp names the gated tarball")
        bound = stamp["tarball"]
        path = Path(bound["path"])
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.is_file():
            raise Refusal("gated-tarball", f"the gated tarball {path} is missing; re-run the "
                          "full gate (make opencode-package-check)")
        from build import PayloadError, tarball_facts

        try:
            now = tarball_facts(path)
        except PayloadError as exc:
            raise Refusal("gated-tarball", str(exc))
        problems = [
            f"{key} is now {now[key]!r}, the stamp says {bound[key]!r}"
            for key in ("integrity", "sha256", "digest") if now[key] != bound[key]
        ]
        if list(now["files"]) != sorted(bound["files"]):
            gained = sorted(set(now["files"]) - set(bound["files"]))
            lost = sorted(set(bound["files"]) - set(now["files"]))
            problems.append(f"the inventory differs (gained {gained}, lost {lost})")
        if now.get("recorded_digest") not in (None, bound["digest"]):
            problems.append(f"its dist/tree.sha256 says {now['recorded_digest']}")
        if problems:
            raise Refusal("gated-tarball", f"{path} is not the tarball the gate passed: "
                          + "; ".join(problems))
        self._run([sys.executable, str(BUILD), "--scan-tarball", str(path)], "gated-tarball",
                  "`build.py --scan-tarball`", timeout=300.0)
        self.facts["tarball"] = {**{k: now[k] for k in ("integrity", "sha256", "digest")},
                                 "path": str(path), "files": len(now["files"])}

    def check_row9(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mllm-first-release-") as scratch:
            out = Path(scratch) / "row9.json"
            done = self.runner(["node", str(ROW9), "--registry", self.facts["registry"],
                                "--out", str(out)], timeout=240.0)
            try:
                evidence = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                evidence = None
        if done.returncode != 0 or not evidence or evidence.get("pass") is not True \
                or evidence.get("outcome") != "absent":
            raise Refusal("row-9", f"{self.facts['spec']} is not provably absent from "
                          f"{self.facts['registry']} ({_tail(done)})")
        self.facts["row_9"] = evidence

    PREFLIGHTS = (
        ("clean-tree", "check_clean_tree"),
        ("stamp", "check_stamp"),
        ("tag", "check_tag"),
        ("origin-master", "check_origin_master"),
        ("required-checks", "check_required_runs"),
        ("bump-check", "check_bump"),
        ("npm-auth", "check_npm_auth"),
        ("gated-tarball", "check_gated_tarball"),
        ("row-9", "check_row9"),
    )

    def preflight(self) -> bool:
        """Every precondition, each reported under its own name. True iff all held."""
        try:
            self.read_manifest()
        except Refusal as refusal:
            self.refusals.append(refusal)
            self.err(f"REFUSING ({refusal.condition}): {refusal.message}")
            return False
        self.out(f"first release of {self.facts['spec']} to {self.facts['registry']} "
                 f"from {self.facts['head']}")
        for condition, method in self.PREFLIGHTS:
            self._check(condition, getattr(self, method))
        return not self.refusals

    # -- the upload ----------------------------------------------------------

    def publish_argv(self) -> List[str]:
        return ["npm", "publish", self.facts["tarball"]["path"], "--ignore-scripts",
                "--access", "public", f"--registry={self.facts['registry']}"]

    def check_publish_argv(self, argv: Sequence[str]) -> None:
        """The upload's argv, refused unless it is exactly the guarded shape: the
        gated FILE, lifecycle scripts off, public, the single registry, and no
        provenance token at all (the laptop cannot sign one)."""
        problems = []
        if list(argv[:2]) != ["npm", "publish"] or len(argv) < 3:
            problems.append("not an `npm publish <file>`")
        else:
            target = str(argv[2])
            if target != self.facts["tarball"]["path"] or not target.endswith(".tgz") \
                    or not Path(target).is_file():
                problems.append(f"uploads {target!r}, not the gated tarball "
                                f"{self.facts['tarball']['path']!r}")
        if "--ignore-scripts" not in argv:
            problems.append("lifecycle scripts are not disabled (--ignore-scripts)")
        if any(str(a).startswith("--provenance") for a in argv):
            problems.append("asks for provenance, which only the CI publish can sign")
        if f"--registry={self.facts['registry']}" not in argv or \
                sum(str(a).startswith("--registry") for a in argv) != 1:
            problems.append(f"does not name exactly the single registry {self.facts['registry']}")
        if "public" not in argv or "--access" not in argv:
            problems.append("is not --access public")
        if problems:
            raise Refusal("publish-argv", f"refusing to run {shlex.join(map(str, argv))}: "
                          + "; ".join(problems))

    def confirm(self) -> Dict[str, Any]:
        """Poll ``dist.integrity`` from a fresh cache until the version is visible.

        ``state`` is ``confirmed`` (it equals the gated integrity), ``mismatch``
        (the registry serves something else) or ``expired`` (never visible)."""
        expected = self.facts["tarball"]["integrity"]
        deadline = self.clock() + self.poll_seconds
        attempts = 0
        while True:
            attempts += 1
            with tempfile.TemporaryDirectory(prefix="mllm-npm-cache-") as cache:
                argv = ["npm", "view", self.facts["spec"], "dist.integrity", "--json",
                        f"--registry={self.facts['registry']}", "--cache", cache,
                        "--prefer-online", "--no-update-notifier", *NPM_FETCH_FLAGS]
                done = self.runner(argv, timeout=150.0)
            read_at = _utc()
            record = {"argv": [a if a != cache else "<fresh tmp>" for a in argv],
                      "read_at": read_at, "exit": done.returncode,
                      "output": done.stdout.strip(), "attempts": attempts}
            if done.returncode == 0:
                try:
                    served = json.loads(done.stdout)
                except ValueError:
                    served = done.stdout.strip()
                state = "confirmed" if served == expected else "mismatch"
                return {"state": state, "served": served, "expected": expected, **record}
            if self.clock() >= deadline:
                return {"state": "expired", "served": None, "expected": expected, **record}
            self.sleep(self.poll_interval)

    def write_evidence(self, publish: Dict[str, Any], confirmation: Optional[Dict[str, Any]]) -> Path:
        stamp = self.facts["stamp"]
        evidence = {
            "kind": "opencode-first-release",
            "path": "laptop",
            "provenance": LOCAL_PROVENANCE,
            "name": self.facts["name"],
            "version": self.facts["version"],
            "spec": self.facts["spec"],
            "registry": self.facts["registry"],
            "tag": {"name": self.facts["tag"], "object": self.facts["tag_object"]},
            "commit": self.facts["head"],
            "origin_master": self.facts["origin_master"],
            "required_checks": self.facts["required_checks"],
            "check_runs": self.facts["check_runs"],
            "stamp": {k: stamp.get(k) for k in ("session", "stamped_at", "digest",
                                                 "candidate_sha", "host_version", "image_id")},
            "gated_integrity": self.facts["tarball"]["integrity"],
            "tarball": self.facts["tarball"],
            "row_9": self.facts["row_9"],
            "npm_user": self.facts.get("npm_user"),
            "publish": publish,
            "confirmation": confirmation,
        }
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        stamp_name = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.evidence_dir / f"{stamp_name}-{self.facts['tag']}.json"
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        return path

    def run(self, *, dry_run: bool) -> int:
        ok = self.preflight()
        if not ok and dry_run and "registry" in self.facts:
            tarball = ((self.facts.get("stamp") or {}).get("tarball") or {}).get("path")
            self.out("dry run: once every preflight passes, the upload would be:\n  npm publish "
                     + (shlex.quote(tarball) if tarball else "<the gated .tgz the stamp names>")
                     + f" --ignore-scripts --access public --registry={self.facts['registry']}")
        if not ok:
            self.err(f"first release: {len(self.refusals)} precondition(s) failed; nothing "
                     "was uploaded.")
            return EXIT_REFUSED
        argv = self.publish_argv()
        try:
            self.check_publish_argv(argv)
        except Refusal as refusal:
            self.err(f"REFUSING ({refusal.condition}): {refusal.message}")
            return EXIT_REFUSED
        if dry_run:
            self.out(f"dry run: every preflight passed. The upload would be:\n  {shlex.join(argv)}")
            return EXIT_OK

        started = _utc()
        done = self.runner(argv, capture=False, timeout=1800.0)
        publish = {"argv": argv, "started_at": started, "finished_at": _utc(),
                   "exit": done.returncode}
        if done.returncode != 0:
            path = self.write_evidence(publish, None)
            self.err(f"npm publish exited {done.returncode}. If the failure was an ambiguous "
                     f"network error, inspect the registry before retrying (step 4): if "
                     f"{self.facts['spec']} is present, never upload it again. Evidence: {path}")
            return EXIT_PUBLISH_FAILED

        confirmation = self.confirm()
        path = self.write_evidence(publish, confirmation)
        if confirmation["state"] == "confirmed":
            self.out(f"{self.facts['spec']} is public and the registry serves the gated "
                     f"integrity. Push the tag now: git push origin {self.facts['tag']}\n"
                     f"evidence: {path}")
            return EXIT_OK
        if confirmation["state"] == "expired":
            self.err(f"NOT YET CONFIRMED: {self.facts['spec']} was uploaded but was not visible "
                     f"within {self.poll_seconds:.0f} s. Repeat the read by hand (step 4) until "
                     f"it prints {confirmation['expected']}, then push the tag:\n"
                     f"  npm view {self.facts['spec']} dist.integrity "
                     f"--registry={self.facts['registry']} --cache \"$(mktemp -d)\"\n"
                     f"evidence: {path}")
            return EXIT_UNCONFIRMED
        self.err(f"MISMATCH AFTER PUBLICATION: the registry serves {confirmation['served']!r} "
                 f"for {self.facts['spec']}, the gate recorded {confirmation['expected']!r}. Do "
                 f"NOT push {self.facts['tag']} or announce; follow Recovery. evidence: {path}")
        return EXIT_MISMATCH


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="first_release.py",
        description="The laptop first release of opencode-multi-llm (make opencode-publish-first).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="run every preflight and print the upload without running it")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS,
                        help=f"how long to wait for the registry to serve the version "
                             f"(default {POLL_SECONDS:.0f})")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return Release(poll_seconds=args.poll_seconds).run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
