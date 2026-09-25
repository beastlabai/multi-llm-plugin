"""Host-side driver: build one `docker run` per row, then read the evidence back.

This is the half of C1a's split that lives **outside** the image. The
container-marked tests *are* the driver: each builds one `docker run`, waits for
it, then asserts over the files that run left on the bind mount. Nothing in the
container decides whether a row passed — it produces evidence, and this reads it.

Three rules this file exists to hold:

* **Run flags come from `posture.yaml`, never from the Makefile.** The Makefile
  owns `docker build`; a shell-quoted per-row matrix in a make recipe could not be
  asserted against at all, whereas this can be compared field-by-field against
  `docker inspect` from the outside and `/proc/self/status` from the inside.
* **One `docker run` per row; no long-lived shared container.** That is what makes
  three separate rules hold by construction rather than by discipline: no row
  inherits another's stub session table (the stub dies with the row), one host
  process per row satisfies `reload: new-session` with no reload step, and the
  sequential budget checkpoint is simply *this* deciding whether to start the next
  container.
* **The repo tree is never mounted.** Goldens, lockfiles and run records are
  host-side writes made from what this read back out of `/work/out` once the
  container exited, so "the harness captured a golden" always lands as a
  reviewable diff.
"""

from __future__ import annotations

import fnmatch
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The bundle digest a record carries has to be the SAME walk
# `make e2e-verify-record` re-hashes with, so it is imported rather than
# reimplemented: two copies of a digest function is how a record starts
# disagreeing with its own verifier over a difference nobody chose.
from .verify_record import hash_bundle

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
#: The tracked half (C10). `runs/` is gitignored and prunable; this is not.
RECORDS = HERE / "records"
#: `runs/<utc>[-<suffix>]/` — the only entries the retention sweep may touch.
#: `README.md`, `lock-update/` and `surface/` share the directory and are not run
#: bundles, so the sweep is keyed off this shape rather than off "every child of
#: runs/" the way `make e2e-clean` is.
_RUN_BUNDLE_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[\w.-]+)?$")
#: How many bundles survive a sweep. `runs/` is gitignored, but **nothing that
#: copies a working tree consults .gitignore** — `codex plugin add` from a local
#: marketplace and a local-path `/plugin install` both take the directory as it
#: stands. Uncapped, this lane grew the checkout to 1.5 GB against 18 MB of
#: tracked files, and a plugin install would have copied all of it. `0` disables
#: the sweep; `MLLM_E2E_RUNS_KEEP` overrides it.
RUNS_KEEP_DEFAULT = 8
RECORD_SCHEMA = RECORDS / "record.schema.json"
#: Written by `verify_versions.py` in the final build layer: what the image really
#: installed, as opposed to what `cli-versions.lock` asked for. It is the only
#: honest source for a row's `cli_version` under `MLLM_PIN_MODE=latest`.
IMAGE_VERSION_REPORT = "/opt/mllm/lock/cli-versions.installed.json"
#: `hosts/capabilities.yaml`, whose `provenance` block is what a green host row
#: promotes. Read here, never written: C10 prints promotions and a human commits
#: them.
CAPABILITIES = HERE.parents[1] / "hosts" / "capabilities.yaml"
#: The skill root, for the ONE import this host-side driver makes from the
#: product itself: `sanitize_model_name`, which decides every per-model
#: result filename (`result_file_names` below). Inserted on the path lazily,
#: at the call, so importing this module costs nothing and a rig invoked
#: from outside pytest reaches the same function pytest's conftest does.
SKILL_ROOT = HERE.parents[1]
DEFAULT_IMAGE = os.environ.get("MLLM_E2E_IMAGE", "mllm-e2e:local")


def prune_runs(runs_root: Path | None = None, keep: int | None = None,
               protect: Path | None = None) -> list[Path]:
    """Delete all but the newest `keep` run bundles; return what was removed.

    Ordering is by directory NAME, which is a UTC stamp — a real chronology that
    does not move when a filesystem restamps mtimes (a copy, a restore, a
    `chmod -R`). `protect` survives however old it sorts, which is what lets the
    sweep run *during* a row rather than only between them.

    Best-effort by construction: a bundle another process still holds open can
    fail to unlink, and a retention sweep is never a reason to fail a run.
    """
    root = runs_root or RUNS
    if keep is None:
        raw = os.environ.get("MLLM_E2E_RUNS_KEEP", "").strip()
        try:
            keep = int(raw) if raw else RUNS_KEEP_DEFAULT
        except ValueError:
            keep = RUNS_KEEP_DEFAULT
    if keep <= 0 or not root.is_dir():
        return []
    kept = protect.resolve() if protect else None
    bundles = sorted(
        (p for p in root.iterdir()
         if p.is_dir() and _RUN_BUNDLE_RE.match(p.name)),
        key=lambda p: p.name,
    )
    removed: list[Path] = []
    for path in bundles[:-keep]:
        if kept is not None and path.resolve() == kept:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        removed.append(path)
    return removed


def load_posture() -> dict:
    return yaml.safe_load((HERE / "posture.yaml").read_text(encoding="utf-8"))


def load_creds_policy() -> dict:
    """C6a's mount allowlist. Paths only — this file never holds a value."""
    return yaml.safe_load((HERE / "creds.yaml").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def load_redirect() -> dict:
    """The measured per-CLI matrix (C3). Cached: it is read per cap hit."""
    return yaml.safe_load((HERE / "redirect.yaml").read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=1)
def load_host_matrix() -> dict:
    """The per-host C6c rows: model pin, recorded credential mode, expectations."""
    document = yaml.safe_load((HERE / "host_matrix.yaml").read_text(encoding="utf-8"))
    return dict((document or {}).get("hosts") or {})


@functools.lru_cache(maxsize=1)
def load_capabilities() -> dict:
    """`hosts/capabilities.yaml`, for the provenance a green row can promote."""
    document = yaml.safe_load(CAPABILITIES.read_text(encoding="utf-8"))
    return dict((document or {}).get("hosts") or {})


@functools.lru_cache(maxsize=1)
def load_record_schema() -> dict:
    return json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))


@dataclass
class Row:
    """One unit of work: a provider CLI's chain, or a (host x mode) cell."""

    row_id: str
    lane: str                      # provider | host
    cli: str | None = None
    host: str | None = None
    creds: str = "none"
    #: The row's model-selection fan-out, as `provider:model` specs.
    #:
    #: A **list**, because `providers.tmpl.yaml`'s `defaults.models` always was
    #: one and only happened to render with a single entry. A host row that
    #: fans out to five providers adds five stub-backed provider subprocesses
    #: inside the SAME host turn — no extra host turn, no extra credit — which
    #: is what makes §3.3's partition free. A scalar is still accepted and
    #: normalised to a one-element list, so every existing caller is unchanged.
    spec: str | list[str] | None = None
    runner: str | None = None
    helpers: list[str] = field(default_factory=list)
    command: str = ""
    #: Extra dispatcher invocations, run in order after ``command``. A chain row
    #: is `full`, then a standalone `ask`, then a `status` read-back; the
    #: entrypoint records each one's exit code in ``commands.json`` and stops at
    #: the first non-zero.
    commands: list[str] = field(default_factory=list)
    scenario: str | None = None
    #: Provider lane, applied-bytes row only (C4 option (a)): shadow this binary
    #: name with the edit-capable fake so one write batch can really move bytes.
    #: Every other binary stays the genuine CLI behind the wrapper.
    mock_runner: str | None = None
    row_timeout_s: int = 900
    #: Host lane: wire, stage and render the argv, then stop before the host
    #: process. The whole credentialed path except the model turn, at zero cost.
    dry_run_host: bool = False
    #: Host lane, `creds="none"`: install the shim, start a FRESH session with the
    #: host's own model pointed at the loopback stub, and type the exact
    #: `/multi-llm …` string README.md and docs/hosts.md publish. Credential-free
    #: by construction — what is under test is argument delivery, not model
    #: output. The verdict lands in the bundle as `invocation.json`.
    documented_invocation: bool = False
    #: Which install route this row exercises: ``file`` (a shim ``install-host``
    #: wrote) or ``package`` (a distribution the host's own installer resolved).
    #:
    #: **Promotion is keyed on this**, and without it the two routes would credit
    #: each other's evidence. A ``package`` row writes no shim at all, yet under a
    #: route-blind field set it would propose promotions for ``shim``,
    #: ``install_paths``, ``arguments`` and ``reload`` — describing a file at a
    #: path that row never touched. That is the same class of false promotion the
    #: ``documented_invocation`` carve-out below was added to stop after a refused
    #: command printed five of them.
    route: str = "file"
    #: Host lane, codex package route, `creds="none"` only: which zero-quota
    #: probe this row runs (`entrypoint.CODEX_PACKAGE_PROBES`, read off the
    #: matrix's `package_probes:` rows). ``None`` on every other row, and then
    #: never forwarded, so no existing row's argv moves.
    probe: str | None = None


@dataclass
class RowResult:
    row: Row
    exit_code: int | None
    wall_clock_s: float
    out_dir: Path
    timed_out: bool = False
    cap_reason: str | None = None
    stdout: str = ""
    stderr: str = ""
    #: Where C10's tracked record for this row landed, or ``None`` when the driver
    #: could not derive one (it says why in ``record-skipped.txt`` beside the
    #: bundle). What lands here first is the PROVISIONAL record — the container's
    #: verdict, no promotions; :class:`GradeLedger` rewrites the same path with the
    #: grader's verdict once pytest has finished asserting over this bundle.
    record_path: Path | None = None
    #: Earlier attempts of this SAME row that the driver discarded and re-ran,
    #: oldest first. Empty on every row that ran once — which is every row but
    #: the never-started shape, the only one this lane retries. Each entry is a
    #: complete `RowResult` whose `out_dir` points at its own shelved bundle
    #: (`runs/<utc>/<row-id>/attempt-<n>/`), so a discarded attempt is read back
    #: exactly like a graded one and lands in the record as its own row.
    attempts: list["RowResult"] = field(default_factory=list)
    #: On a DISCARDED attempt: the :func:`never_started_signature` that condemned
    #: it. `None` on the attempt that was graded, whichever number it is.
    retry_signature: dict | None = None

    @property
    def attempt(self) -> int:
        """Which attempt of its row this is — 1 unless something was discarded."""
        return len(self.attempts) + 1

    # -- the evidence, read back after the container is gone ---------------- #
    @property
    def stub_log(self) -> list[dict]:
        return _read_jsonl(self.out_dir / "stub.jsonl")

    @property
    def call_log(self) -> list[dict]:
        return _read_jsonl(self.out_dir / "calls.jsonl")

    @property
    def commands(self) -> list[dict]:
        """``[{command, exit}]`` — one entry per dispatcher invocation, in order."""
        path = self.out_dir / "commands.json"
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def receipts(self) -> list[dict]:
        directory = self.out_dir / "receipts" / self.row.row_id
        if not directory.is_dir():
            return []
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(directory.glob("*.receipt.json"))]

    @property
    def invocation(self) -> dict:
        """The documented-invocation probe's verdict, or `{}` — see the entrypoint.

        `{}` is itself a finding on a row that asked for the probe: it means the
        host process never got far enough to record one.
        """
        return _read_json(self.out_dir / "invocation.json")

    @property
    def install_source(self) -> str | None:
        """Where a package row's plugin install came from, as the bundle's
        ``plugin-install.json`` recorded it (``local-path``, ``git``), or ``None``
        when nothing recorded one. :func:`promotable_fields` keys ``package`` on
        this, never on the route word (rule j)."""
        value = _read_json(self.out_dir / "plugin-install.json").get(
            "install_source")
        return str(value) if value else None

    @property
    def host_run(self) -> dict:
        """The host process's own argv/exit/wall clock — reported SEPARATELY.

        "The host did not terminate" and "the command never ran" are different
        failures; only the receipt answers the second one.
        """
        return _read_json(self.out_dir / "host-run.json")

    @property
    def usage(self) -> dict:
        """The row's measured spend: the dedicated key's `usage` delta."""
        return _read_json(self.out_dir / "usage.json")

    @property
    def credential_census(self) -> dict:
        return _read_json(self.out_dir / "credential-census.json")

    @property
    def container_posture(self) -> dict:
        """What the entrypoint verified from INSIDE, before anything else ran."""
        return _read_json(self.out_dir / "posture-inside.json")

    @property
    def host_posture(self) -> dict:
        """`{captured, reason, mismatches}` from `docker inspect` during the run.

        The other half of the same check: `posture.yaml` is compared field by field
        against the run's real `HostConfig`, because posture is a property of the
        run *command* and the kind of thing that erodes one debugging session at a
        time.
        """
        return _read_json(self.out_dir / "posture-host.json")

    @property
    def run_states(self) -> dict:
        """`{phase: {run_id, state, exit_code}}` after the detach poll."""
        return _read_json(self.out_dir / "run-states.json")

    def phase_status(self, phase: str) -> dict:
        """`{phase}/.status.json` as exported into the bundle."""
        root = self.out_dir / "artifacts"
        for candidate in sorted(root.glob(f"*/{phase}/.status.json")):
            return _read_json(candidate)
        return {}

    def workflow_state(self) -> dict:
        root = self.out_dir / "artifacts"
        for candidate in sorted(root.glob("*/state.json")):
            return _read_json(candidate)
        return {}

    def census_delta(self) -> dict[str, tuple[str | None, str | None]]:
        """Which workspace files moved, by SHA — never what was written into them.

        The WHOLE-ROW pair: taken before `run_pre_commands` and after the
        export, so it brackets everything the container did. Left exactly as
        it was, so nothing already graded moves.
        """
        return self._census_delta("workspace-census-before.json",
                                  "workspace-census-after.json")

    def host_census_delta(self) -> dict[str, tuple[str | None, str | None]]:
        """The same, bracketing the HOST PROCESS alone.

        On a row whose earlier phases are seeded by a `pre_commands` leg —
        six of the eight claude-code phase rows — every byte the leg writes
        lands inside `census_delta`, because the whole-row `before` snapshot
        is taken right after `seed_workspace.py` and **before** the leg runs.
        Those are precisely the rows where it decides the verdict.

        Falls back to the whole-row pair when the narrower one is absent, so
        a bundle exported by an older image still grades rather than
        vanishing — the caller is told which pair it got by
        :meth:`has_host_census`.
        """
        if not self.has_host_census:
            return self.census_delta()
        return self._census_delta("workspace-census-host-before.json",
                                  "workspace-census-host-after.json")

    @property
    def has_host_census(self) -> bool:
        return ((self.out_dir / "workspace-census-host-before.json").is_file()
                and (self.out_dir / "workspace-census-host-after.json").is_file())

    def _census_delta(self, before_name: str, after_name: str
                      ) -> dict[str, tuple[str | None, str | None]]:
        before = _read_json(self.out_dir / before_name)
        after = _read_json(self.out_dir / after_name)
        delta = {}
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                delta[name] = (before.get(name), after.get(name))
        return delta


def stray_paths(names: Iterable[str], declared: Iterable[str]) -> list[str]:
    """Which of `names` no pattern in `declared` admits — sorted, never their bytes.

    Lifted verbatim out of
    `test_container_host_lane.test_no_write_landed_outside_the_declared_paths`;
    the behaviour is that test's, unchanged. It lives here because that test is
    `live`-marked beneath a module-level container gate, so it answers only inside
    a credentialed Docker run — and nobody widening a row's `paths:` glob in a PR
    will spend a host-lane run to find out they widened it too far. The offline
    both-directions pin is `tests/test_stray_write_paths.py`, which needs neither
    credential nor daemon.

    THE MATCHER IS `fnmatch`, AND ITS `*` CROSSES `/`. So `src/**` and `src/*` are
    the SAME predicate, and a declaration is judged by what it still refuses
    (`tests/conftest.py`, `tests/fixtures/data.json`, `tests/unit/deep/test_x.py`)
    rather than by how many stars it spells. Added-vs-modified is deliberately not
    part of the predicate: `census_delta` carries `(before_sha, after_sha)` and
    could express it, but a model that puts its new cases into a file that already
    exists has done the same legitimate work. Inside-vs-outside the declared trees
    is the distinction; added-vs-modified is not.
    """
    declared = list(declared)
    return sorted(name for name in names
                  if not any(fnmatch.fnmatch(name, pattern)
                             for pattern in declared))


def _read_jsonl(path: Path, torn: list[str] | None = None) -> list[dict]:
    """Every record that parses — and never a raise on the one that does not.

    These logs are appended by a process this driver may have to **kill**: a row
    that hits its cap is `docker stop`ped and, if it does not go, SIGKILLed, which
    can cut the last line in half mid-write. Those are exactly the rows whose stub
    log `classify_cap_hit` has to read, so a strict reader turns a diagnosable cap
    hit into a `JSONDecodeError` raised while *collecting the evidence for it* —
    the one moment the harness must stay readable.

    So a line that does not parse is skipped rather than fatal, and a caller that
    wants to know appends it to ``torn`` (the cap-hit path does not: a truncated
    tail is the expected shape of a killed row, not a finding). Decoding is
    ``errors="replace"`` for the same reason — a SIGKILL can land mid-codepoint.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if torn is not None:
                torn.append(line)
    return rows


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def container_name(row: Row) -> str:
    """One stable, unique name per row invocation."""
    return f"mllm-e2e-{row.row_id}-{os.getpid()}"


def fallback_binds(row: Row) -> list[tuple[Path, str]]:
    """The `:ro` credential mounts for a `--creds fallback` row (C6a rules 1-2).

    One **file** per `creds.yaml` allowlist entry for this row's host, never a home
    directory, and never anything the allowlist does not name. `~/.claude.json` is
    the entry that matters most by being absent: 87 KB of per-project history and
    MCP definitions that routinely embed third-party keys, where the host only
    needs onboarding and trust — so the entrypoint synthesizes a minimal one.

    A source that does not exist on this machine is **skipped rather than
    mounted**: docker would create a *directory* at that path in the developer's
    home, and the entrypoint refuses from inside naming the file it expected.
    """
    if row.lane != "host" or row.creds != "fallback":
        return []
    entries = (load_creds_policy().get("allowlist") or {}).get(row.host or "") or []
    binds = []
    for entry in entries:
        source = Path(str(entry["source"])).expanduser()
        if source.is_file():
            binds.append((source, str(entry["dest"])))
    return binds


def network_mode(row: Row, posture: dict) -> str:
    """Which docker `--network` this row runs under. NOT a function of the lane.

    The provider lane is `none` because its guarantee is structural: a
    mis-redirected CLI cannot reach a real endpoint when there is no route to one.
    A **`--creds none` host row inherits exactly that**, because the discovery tier
    carries no credential and needs no route — it stages the shim, crosses the
    reload boundary and asserts listed-and-invocable, and nothing in that path has
    an endpoint to talk to. Selecting the open host-lane bridge for it would leave
    a third-party binary free to reach the public internet during the tier the plan
    advertises as credential-free and zero-cost, and the row's only evidence would
    be that it did not call *our* stub — which is not the same claim.

    The credentialed host modes keep the bridge: they must reach openrouter.ai, so
    their guarantee downgrades from impossibility to detection.
    """
    network = posture["network"]
    if row.lane == "provider" or row.creds == "none":
        return str(network["provider_lane"])
    return str(network["host_lane"])


# --------------------------------------------------------------------------- #
# The host lane's fan-out, runner and helper derivation — ONE source of truth.
#
# These live here rather than in `test_container_host_lane.py` so the plain-CI
# guard in `test_redirect_matrix_claims.py` and the fixture that actually
# builds the row read the SAME functions. A guard that re-derives the partition
# asserts its own copy of it, which is how a lane ends up green over a fan-out
# nothing executes.
#
# `host_matrix.yaml` names **CLI ids only, never slugs**. The 16 provider rows
# carry eight distinct spellings of the stub model (`gpt-5.5`,
# `openai/stub-model`, `stub-model`, `stub/stub-model`, `muse-spark-1.2`,
# `haiku`, `gemini-2.5-flash`, `openai/gpt-5.6-luna`), every one of them a
# hand-typed matrix cell can get wrong — and this lane has already paid for
# exactly that once: opencode handed a bare `stub-model` raised
# `ProviderModelNotFoundError` before a single HTTP call left the process.
# Ids-only is also what gives the disjoint/union guard a single source of
# truth: it compares ids against `redirect.yaml`'s `lane: provider` rows and
# never has to parse a slug back into a CLI name.
# --------------------------------------------------------------------------- #

#: A row whose dispatch never crosses `/opt/mllm/bin` — the host's own Task
#: tool does the work — so a fan-out list has no mechanism to act through.
#: Declared explicitly rather than by omitting the key, so the guard can tell
#: "deliberately outside the partition" from "somebody forgot".
FAN_OUT_NATIVE = "native"


@functools.lru_cache(maxsize=None)
def provider_model_args() -> dict[str, str]:
    """`{cli: model_arg}` for every `lane: provider` row in `redirect.yaml`."""
    return {
        str(row["cli"]): str(row.get("model_arg") or "stub-model")
        for row in (load_redirect().get("clis") or [])
        if row.get("lane") == "provider"
    }


def provider_spec(cli: str) -> str:
    """One fan-out member's spec, in that CLI's OWN addressing scheme."""
    return f"{cli}:{provider_model_args().get(cli, 'stub-model')}"


@functools.lru_cache(maxsize=None)
def result_file_names(cli: str) -> tuple[str, ...]:
    """The EXACT filenames this fan-out member's own result can land under.

    Built from `provider_spec(cli)` — the same function that produced the
    fan-out's config — through the product's own `sanitize_model_name`, which
    is the ONE thing that decides a per-model filename: the review phases
    write `{stem}.json` (`code_review_orchestrator.save_model_result`,
    `review_orchestrator_base`) and `ask` writes `answer_{stem}.md`
    (`ask_orchestrator`). Imported rather than re-spelled as a local
    `re.sub`, because a test-side copy is how a correlation starts grading a
    filename shape the product stopped writing.

    Exact names rather than the substring test `fan_out_correlation` used to
    run: a CLI id that occurs inside a SIBLING's result filename had its
    `result` fact satisfied by that sibling's file, so a provider that
    produced nothing could be credited with another's output and the
    exactly-N rule was weaker than it claimed. `pi` sits inside
    `copilot_stub-model.json` and is the pair to watch; the two are on
    different hosts today, which makes the defect latent rather than live —
    and a partition is re-cut by editing one list in `host_matrix.yaml`,
    which is not a change anybody would expect to weaken a grader.
    """
    if str(SKILL_ROOT) not in sys.path:
        sys.path.insert(0, str(SKILL_ROOT))
    from utils.json_extractor import sanitize_model_name

    stem = sanitize_model_name(provider_spec(cli))
    return (f"{stem}.json", f"answer_{stem}.md")


def host_fan_out(host_entry: dict, matrix_row: dict) -> list[str] | str:
    """This row's effective fan-out: a list of CLI ids, or ``"native"``.

    Declared once per host and inherited by every fan-out row on it, including
    **both opencode routes** — the package row then grades the same partition
    through a different loader, which is what makes it a route test rather
    than a second, smaller partition.

    A row-level `fan_out:` is an override for a declared exception, not the
    normal spelling. The guard reduces to the HOST's declared list for the
    union and the disjointness check and separately asserts that no row's
    effective list names a CLI outside its own host's list — so an override
    can narrow a row and can never silently widen the partition or move a CLI
    between hosts.

    `remove_seed` rows inherit nothing: they are graded on the refusal landing
    *before* the first model call, which is upstream of any point a fan-out
    could act.
    """
    if matrix_row.get("remove_seed"):
        return []
    declared = matrix_row.get("fan_out", host_entry.get("fan_out"))
    if declared == FAN_OUT_NATIVE:
        return FAN_OUT_NATIVE
    return [str(cli) for cli in (declared or [])]


def host_runner_cli(host: str, host_entry: dict, matrix_row: dict) -> str:
    """The CLI id this row dispatches its write batches through.

    On a `remove_seed` row it MUST be the row's own host CLI: the row's
    `remove_seed` names a file under that CLI's seeded home, and the gate it
    trips is the executor preflighting the CONFIGURED runner — a shared runner
    would sail straight past the file the row removes and the row would assert
    nothing at all.

    Everywhere else it is a matrix key, defaulting per host. It is emphatically
    no longer the hard-coded `codex:stub-model` every non-negative row on all
    three hosts used: codex's `agent-write` renders `--sandbox
    workspace-write`, which is bubblewrap, which needs a user namespace
    Docker's default seccomp profile denies.
    """
    if matrix_row.get("remove_seed"):
        return host
    return str(matrix_row.get("runner", host_entry.get("runner")) or "")


def host_helpers(host: str, host_entry: dict, matrix_row: dict) -> list[str]:
    """Which CLIs this row needs a seeded home for.

    The union of the fan-out and the runner, plus the host itself on the
    negative rows that already needed it.

    **The runner is unioned in rather than required to be inside the
    partition.** The partition is about graded provider *coverage* and is
    disjoint by construction, so the one shared write runner can appear in at
    most one host's `fan_out`; `helpers` is about seeded *homes*, and a runner
    with no seeded home fails preflight for a reason that has nothing to do
    with the phase under test.
    """
    fan_out = host_fan_out(host_entry, matrix_row)
    members = set() if fan_out == FAN_OUT_NATIVE else set(fan_out)
    runner = host_runner_cli(host, host_entry, matrix_row)
    if runner:
        members.add(runner)
    if matrix_row.get("remove_seed"):
        members.add(host)
    return sorted(members)


# --------------------------------------------------------------------------- #
# What counts as workflow bookkeeping, and what is the phase's own output.
#
# `test_container_host_lane._source_delta` drops EVERY path under `plans/` and
# `.multi-llm/` as bookkeeping. That is right for a row whose subject is
# `apply-code-fixes` touching `src/**` and `tests/test_*.py`; it is exactly
# wrong for `apply-suggestions`, whose entire intended effect **is** an edit to
# the plan file under `plans/`. A row graded by that predicate could rewrite
# the plan, or fail to, and pass either way.
#
# The real distinction is not the `plans/` prefix. It is the plan's own OUTPUT
# DIRECTORY: `StateManager.plan_dir` is `get_output_dir(plan_path)`, which for
# `plans/auth-feature.md` is `plans/auth-feature/`, and every `{phase}/`
# artifact dir, `state.json`, `.status.json` and `report.html` lands inside it.
# Measured against a real census on 2026-09-22
# (`runs/20260920T091849Z/chain-omp`): 25 added paths, every one of them under
# `plans/auth-feature/`, and the plan file `plans/auth-feature.md` a sibling of
# that directory rather than a child of it.
#
# So: one directory level under `plans/` is a plan FILE and is gradeable; two
# or more is the workflow's own output and is bookkeeping.
# --------------------------------------------------------------------------- #

def workflow_bookkeeping(name: str) -> bool:
    """Is this census path the workflow's own record-keeping rather than work?

    `.multi-llm/**` is the run registry. `plans/<stem>/**` is that plan's
    output directory. `plans/<stem>.md` is **not** — it is the artifact
    `apply-suggestions` exists to edit.
    """
    if name.startswith(".multi-llm/"):
        return True
    if name.startswith("plans/"):
        # Inside a plan's output directory (two or more levels), not the plan
        # file itself (exactly one).
        return "/" in name[len("plans/"):]
    return False


def graded_delta(names: Iterable[str], declared: Iterable[str] | None = None
                 ) -> list[str]:
    """The census paths a phase row is answerable for, bookkeeping removed.

    ``declared`` is the row's `editable_artifacts`, and it is consulted
    **before** the bookkeeping predicate, not after it. A path the row
    explicitly declares is that row's grade by definition, wherever it happens
    to live — so the declaration outranks every prefix rule here.

    That ordering is the whole of `apply-task-suggestions`. Its one editable
    artifact is the tasks document, and `utils/tasks_file.py:137-138` puts it
    at `{plan_dir}/{sanitize_prefix(plan)}/tasks/tasks.md` — i.e.
    `plans/auth-feature/tasks/tasks.md`, two levels under `plans/` and
    therefore inside the plan's output directory. A grader that ran
    `workflow_bookkeeping` first would discard the only file that row can be
    graded on, and `bytes_must_change: true` would be unsatisfiable on it:
    header finding (3) recurring on a third row, which is exactly what §3.4's
    grader change exists to end. The tasks document is not bookkeeping merely
    for sitting beside `state.json`; it is bookkeeping for no row, and work
    for the one row that declares it.

    Undeclared paths then fall through to the prefix rules, and there the
    declaration still matters for the plan file: `finalize_tracking` writes a
    tracking back-reference into it on every phase, so on a row that does
    **not** own the plan file that change is bookkeeping — while on
    `apply-suggestions`, which does own it, the very same change is the grade.
    A path predicate alone cannot tell those apart, which is why the
    declaration is an input here rather than only a filter applied afterwards.
    """
    owned = list(declared or [])
    kept: list[str] = []
    for name in names:
        # The declaration first: an explicitly declared editable artifact is
        # gradeable even when it sits inside the plan's output directory.
        if owned and not stray_paths([name], owned):
            kept.append(name)
            continue
        if workflow_bookkeeping(name):
            continue
        is_plan_file = (name.startswith("plans/")
                        and "/" not in name[len("plans/"):])
        if is_plan_file:
            # A plan file this row does not declare: the tracking
            # back-reference, not work the row did.
            continue
        kept.append(name)
    return sorted(kept)


# --------------------------------------------------------------------------- #
# A third-party tool's cache is not a product write.
#
# `workflow_bookkeeping` above is a different category and must not absorb this
# one: it is THE WORKFLOW'S OWN record-keeping (`.multi-llm/**`, the plan's
# output directory), written by this product, in places this product chose.
# What follows is the opposite — bytes a VENDOR's CLI drops into the workspace
# because that is what that binary does, in a place nobody here chose.
#
# The case, and the only one so far: aider builds a repo map on every
# invocation and writes it to `<cwd>/.aider.tags.cache.v4/cache.db`. Measured
# across four recorded bundles — it appears on exactly the rows where aider
# ran, and nowhere else:
#
#   runs/20260922T084502Z  codex-ask, codex-review-plan, codex-full,
#                          codex-apply-executor   (aider is in codex's fan-out)
#   runs/20260920T105458Z  chain-aider   ...and on none of the other 17 rows
#   runs/20260921T224840Z  chain-aider   ...and on none of the other 7 rows
#
# On `codex-apply-executor` it was the whole census delta, and the row failed
# `AssertionError: bytes moved on a row declared 'clean-failure'`.
#
# THREE THINGS KEEP THIS FROM BEING A HOLE IN THE NET.
#
# 1. It is DECLARED, per CLI, in `redirect.yaml` — beside `credential_vars`
#    and `seed_files`, in the same written-down-rather-than-inherited style,
#    with the same rule that a row states what was measured about that binary.
#    Nothing here hardcodes a path, so adding a tool means editing the matrix
#    that already describes these binaries, on a line a reviewer reads as a
#    claim about a vendor's behaviour.
# 2. It is SCOPED TO THE ROW'S OWN FAN-OUT. `tool_artifact_paths` takes the
#    CLIs this row actually dispatches to, so aider's declaration buys nothing
#    on a row aider does not run on. The one row on either lane with a
#    byte-level applied-write grade, `claude-code-apply-native`, declares
#    `fan_out: native` and therefore gets NO exemption at all — its measured
#    delta in the same bundle is exactly `src/auth.py`, `src/security.py`,
#    `tests/test_security.py` and stays graded to the byte.
# 3. THE ROW'S OWN DECLARATION OUTRANKS IT, the same way it outranks
#    `workflow_bookkeeping` in `graded_delta`: a path a row names in
#    `editable_artifacts` is that row's grade by definition and is never
#    exempted out from under it.
#
# The glob is the version the run measured (`…cache.v4`), not a `v*` guess. An
# aider release that bumps the cache version turns these rows red with a path
# in the message, which is a re-measurement this file is meant to force — a
# wildcard would swallow it silently, and every other pin in this matrix is
# spelled the same way for the same reason.
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def tool_artifacts() -> dict[str, tuple[str, ...]]:
    """`{cli: (glob, …)}` — the caches each CLI leaves in the workspace.

    Read from `redirect.yaml`'s per-CLI rows. A CLI that declares none is
    absent rather than present-and-empty, so "nothing is exempt here" and
    "somebody wrote an empty list" are the same fact for every caller.
    """
    declared: dict[str, tuple[str, ...]] = {}
    for row in (load_redirect().get("clis") or []):
        patterns = tuple(str(p) for p in (row.get("tool_artifacts") or [])
                         if str(p).strip())
        if patterns:
            declared[str(row["cli"])] = patterns
    return declared


def tool_artifact_patterns(clis: Iterable[str] | str) -> list[str]:
    """Every declared artifact glob for exactly these CLIs — sorted, deduped.

    `clis` is the row's own effective fan-out. `FAN_OUT_NATIVE` — a string,
    not a list — yields NOTHING rather than being iterated character by
    character: a native row dispatches through the host's own subagent tool,
    crosses no `/opt/mllm/bin` shim and therefore runs none of these binaries.
    """
    if isinstance(clis, str):
        return []
    declared = tool_artifacts()
    return sorted({pattern for cli in clis
                   for pattern in declared.get(str(cli), ())})


def tool_artifact_paths(names: Iterable[str], clis: Iterable[str] | str,
                        declared: Iterable[str] | None = None) -> list[str]:
    """Which of `names` are third-party tool artifacts this row may ignore.

    The positive statement — the exempted paths, sorted — so a caller
    subtracts a named set rather than applying a filter whose effect it cannot
    print, and so the plain-CI control can assert on exactly what was forgiven.

    `declared` is the row's `editable_artifacts`, and a path it admits is
    NEVER returned here: the declaration outranks this rule as it outranks
    `workflow_bookkeeping`. Nothing is forgiven when the row declares no
    fan-out, when that fan-out is `native`, or when none of its CLIs declares
    an artifact — the default is to grade the byte.
    """
    patterns = tool_artifact_patterns(clis)
    if not patterns:
        return []
    owned = list(declared or [])
    exempt = []
    for name in names:
        if owned and not stray_paths([name], owned):
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            exempt.append(name)
    return sorted(exempt)


# --------------------------------------------------------------------------- #
# An error body is not an answer.
#
# `runs/20260922T084502Z` is the run that made this necessary. grok failed on
# all six opencode fan-out rows; four went red, and `opencode-ask` and
# `opencode-package-ask` went GREEN — because the ask lane persisted grok's
# error body as its answer. `answer_grok_stub-model.md`, 219 bytes:
#
#     {"type":"error","message":"Not signed in. To authenticate without a
#      browser, run:\n  grok login --device-code\n\n…"}
#
# The file has that provider's exact expected name, so the `result` fact — the
# one whose whole job is to prove the provider produced something — was
# satisfied by the CLI's own failure report. `ask_orchestrator.recover_answer_
# text` is why it is on disk at all: with `data` unusable it falls back to the
# provider's raw stdout, which for a failed grok run IS the envelope.
#
# THE DISCRIMINATOR IS THE CLI'S OWN, NOT A GUESS ABOUT PROSE. Two values, each
# already named and measured inside this repo's adapters:
#
#   * `type: "error"` — grok's own error discriminator. `utils/providers/
#     grok.py:186-192` states it against ten distinct injected failures (401,
#     429, 404, 400 context-length, 5xx-forever, a cut stream, a zero-byte
#     response, a 200 carrying an error object, an empty assistant message and
#     grok's pre-flight model rejection): every one of them arrives as
#     `{"type":"error","message":…}`. `utils/providers/cline.py:101-102`
#     records the identical shape for a second CLI.
#   * `is_error: true` — the result envelope claude-code and cursor-agent both
#     print, read by `utils/providers/result_envelope.py` off eleven committed
#     claude-code captures.
#
# AND THE SHAPE IS CHECKED BEFORE THE KEYS, which is what keeps this from
# turning a genuine answer red. The body must parse WHOLE as a JSON **object**.
# Measured over every per-provider result file in `runs/20260922T084502Z` (64
# files) and `runs/20260920T105458Z` (24 files): every genuine review result is
# a JSON **array** of findings, every genuine `ask` answer is prose that is not
# JSON at all (`OK`, and the 421- and 1044-byte answers), and the ONLY
# top-level object among all 88 is grok's envelope, twice. A result that merely
# mentions the word error, or that opens with a `{` inside prose, is untouched.
# --------------------------------------------------------------------------- #

def error_envelope_reason(text: str) -> str | None:
    """The CLI's own failure message when `text` IS an error envelope, else None.

    `None` means "this is a result"; a string means "this is that provider
    reporting that it failed" and the caller must not count it. The message is
    returned rather than a bare bool so the grader's failure can quote the
    reason the provider gave instead of only naming the file.
    """
    stripped = (text or "").strip()
    # Cheap and load-bearing: a prose answer is not parsed at all, so no
    # markdown answer can be reclassified by a JSON decoder's opinion of it.
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except ValueError:
        # Prose that happens to open with a brace. Not an envelope.
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "error":
        message = payload.get("message")
        return (str(message).strip() if isinstance(message, str) and message.strip()
                else 'type: "error"')
    if payload.get("is_error") is True:
        for key in ("result", "error", "api_error_status", "terminal_reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "is_error: true"
    return None


def fan_out_correlation(fan_out: Iterable[str], *, invoked: Iterable[str],
                        reached: Iterable[str],
                        produced: Iterable[str]) -> dict[str, list[str]]:
    """Which of the three facts each assigned provider failed to produce.

    `{cli: [missing fact, …]}`, empty when every assigned provider produced
    all three. The facts are the ones §3.3 names and they fail differently:
    `invocation` (a `/opt/mllm/bin` record, so the shim really ran for it),
    `stub-session` (it reached the model surface instead of dying in argument
    parsing) and `result` (a result file of its own).

    A pure function, taking the three already-extracted name sets, so the
    **negative control** can be a plain-CI fixture over recorded evidence
    rather than a keyed container row. Without that control a correlation
    that matched on the wrong field, or that iterated an empty list, reads
    exactly like complete coverage.

    The caller extracts those three sets and nothing else: the `result` fact
    is decided here, against `result_file_names(cli)` — the exact filenames
    `provider_spec(cli)` produces — rather than by asking whether the CLI id
    occurs anywhere in a filename, which credited a provider with a
    sibling's file whenever one id was a substring of another's stem.

    **`produced` is a list of files that carry an ANSWER**, and that is the
    caller's second obligation: it filters each candidate through
    :func:`error_envelope_reason` before appending its name. A provider that
    failed can still leave a file under its own exact expected name — grok did,
    on both `ask` rows of `runs/20260922T084502Z`, and those rows went green
    while grok was failing — so a name alone is not the fact. This function
    still grades names, because the name is what carries provider identity;
    what a name may no longer do is arrive here off a body the CLI wrote to
    say it had failed.

    `reached` carries NO tolerance for being empty as a whole, and that is the
    correction this predicate exists under: `stub.jsonl` records have no `cli`
    field, so a caller that read one got an empty set on every row, and a
    "grade it only if it is non-empty" rule then skipped the fact everywhere —
    silently, and exactly on the opencode `ProviderModelNotFoundError` shape
    (invoked, zero stub records) the fact was added to catch. The identity is
    the caller's join to make — `calls.jsonl`'s `exec_id`, which the wrapper
    bakes into the stub base URL — and an empty `reached` is a finding against
    every assigned provider rather than a pass.
    """
    invoked, reached, produced = set(invoked), set(reached), list(produced)
    missing: dict[str, list[str]] = {}
    for cli in fan_out:
        gaps = []
        if cli not in invoked:
            gaps.append("invocation")
        if cli not in reached:
            gaps.append("stub-session")
        expected = result_file_names(cli)
        if not any(name in expected for name in produced):
            gaps.append("result")
        if gaps:
            missing[cli] = gaps
    return missing


def row_helpers(row: Row) -> list[str]:
    """``row.helpers``, plus the host-lane runner's CLI when a row left it out.

    `host_helpers` already unions the runner in for every matrix-built row,
    for the reason its docstring gives: a runner with no seeded home fails
    preflight for a reason that has nothing to do with the phase under test.
    The three credential-free fixtures (discovery, documented-invocation, the
    codex package probes) build their `Row` by hand and name a runner without
    a helper, so `/work/homes/<runner>` was never seeded. That is not inert:
    `install-host --check` row 5 preflights the configured runner, and on
    opencode `auth list` against an empty home prints `0 credentials` — a
    CORRECT refusal of a runner whose stub provider block and stored
    credential were never placed. Measured 2026-09-25 on opencode 1.18.32:
    every discovery and documented-invocation row failed row 5
    `not_authenticated` for exactly this reason.

    Applied here, where the argv is built, so a future hand-built row cannot
    reintroduce it. Sorted and de-duplicated the way `host_helpers` returns
    its list, so every matrix-built row's argv stays byte-identical.
    """
    helpers = list(row.helpers or [])
    if row.lane == "host" and row.runner:
        runner_cli = str(row.runner).split(":", 1)[0].strip()
        if runner_cli and runner_cli not in helpers:
            helpers = sorted(set(helpers) | {runner_cli})
    return helpers


def spec_list(row: Row) -> list[str]:
    """``row.spec`` as a list, whatever shape it was written in.

    One place that normalises the scalar, so the four layers between the matrix
    row and the rendered config cannot disagree about whether `spec` is one
    value or many. That disagreement is the half-done version of §3.3's change:
    a row renders one spec while its guard asserts five, and the row looks like
    a fan-out while running `models[0]`.
    """
    if not row.spec:
        return []
    if isinstance(row.spec, str):
        return [row.spec]
    return [str(entry) for entry in row.spec if str(entry).strip()]


#: The instant every container's product renders as "now" where it stamps a
#: file (reproducible-builds' ``SOURCE_DATE_EPOCH``; 2026-01-01T00:00:00Z). The
#: tasks file's `*Generated: …*` line is hashed into `tasks_sha256` and the
#: write reports' `sha256_before`/`sha256_after`, so without a pinned value no
#: two captures of a task-phase row agree on that file's digest (C8; see
#: `tests/golden/container_bundle.py`). Changing it moves those digests in
#: every baseline at once.
SOURCE_DATE_EPOCH = "1767225600"


def build_run_argv(row: Row, out_dir: Path, posture: dict,
                   image: str = DEFAULT_IMAGE) -> list[str]:
    """Assemble the run command from posture.yaml. No flag is written twice."""
    identity = posture["identity"]
    security = posture["security"]
    limits = posture["limits"]

    argv = ["docker", "run", "--rm"]

    # A name, so a row that hits its cap can be stopped precisely rather than
    # left running with a live key on an open network. `_container_of` reads it
    # back out of this argv; without it the timeout path found nothing to stop.
    argv += ["--name", container_name(row)]

    # Non-root is not hygiene here: several agent CLIs refuse to start or change
    # sandbox behaviour under uid 0, and on the host lane a root process holds a
    # live key on an open network. The image's final USER, this flag and the tmpfs
    # uid= are three separate failures, so all three are stated.
    argv += ["--user", f"{identity['uid']}:{identity['gid']}"]

    for capability in security["cap_drop"]:
        argv += ["--cap-drop", capability]
    for option in security["security_opt"]:
        argv += ["--security-opt", option]
    if security.get("readonly_rootfs"):
        argv += ["--read-only"]

    argv += ["--pids-limit", str(limits["pids_limit"])]
    argv += ["--memory", str(limits["memory"])]
    argv += ["--cpus", str(limits["cpus"])]

    # The provider lane blocks the go-live failure structurally, and so does the
    # credential-free discovery tier. Only the credentialed host modes downgrade to
    # detection, because only they must reach openrouter.ai.
    argv += ["--network", network_mode(row, posture)]

    for mount in posture["mounts"]:
        if mount["type"] == "tmpfs":
            argv += ["--tmpfs", f"{mount['target']}:{mount['mode']}"]
        elif mount["type"] == "bind" and mount["target"] == "/work/out":
            argv += ["-v", f"{out_dir}:/work/out:rw"]
    # The default path and the whole provider lane add nothing here: this is the
    # documented fallback mode, and it is the only mode on which anything from the
    # developer's home directory enters the container.
    for source, dest in fallback_binds(row):
        argv += ["-v", f"{source}:{dest}:ro"]

    uv = posture["uv"]
    argv += ["-e", f"UV_PROJECT_ENVIRONMENT={uv['project_environment']}"]
    argv += ["-e", f"UV_CACHE_DIR={uv['cache_dir']}"]
    if uv.get("offline"):
        argv += ["-e", "UV_OFFLINE=1"]

    # Every lane, every row: a determinism pin, not a credential or a policy.
    argv += ["-e", f"SOURCE_DATE_EPOCH={SOURCE_DATE_EPOCH}"]

    # The host lane's ONE credential variable, and only ever on that lane.
    if row.lane == "host" and row.creds == "openrouter":
        argv += ["-e", "OPENROUTER_API_KEY"]
        # Present-and-EMPTY, not unset: the cookbook is explicit that an unset
        # variable lets Claude Code fall back to authenticating against Anthropic
        # directly, which would silently spend the developer's own subscription
        # and produce a green run that proves the wrong thing.
        argv += ["-e", "ANTHROPIC_API_KEY="]
        # The key's own spend limit is the only ceiling that actually holds, so
        # the entrypoint refuses a key reporting none. This forwards the
        # operator's deliberate, per-run override — never a shell-profile
        # default, which is why it is read here and passed explicitly.
        if os.environ.get("MLLM_ALLOW_UNLIMITED_KEY") == "1":
            argv += ["-e", "MLLM_ALLOW_UNLIMITED_KEY=1"]
        # Same terms for the other fail-before-spend refusal: an unreadable model
        # catalog leaves the pin unproven, and the entrypoint refuses rather than
        # spending a turn on it. Typed per run, never a profile default.
        if os.environ.get("MLLM_ALLOW_UNVERIFIED_MODEL") == "1":
            argv += ["-e", "MLLM_ALLOW_UNVERIFIED_MODEL=1"]

    # The pin override, on BOTH credentialed modes — never the openrouter one
    # alone. On that path it is the single anticipated exception (C6: the
    # reference host's row failing on the MODEL rather than on our shim). On the
    # fallback path it is the normal case, because host_matrix.yaml's pins are
    # OpenRouter slugs and vendor auth resolves ids against the host's OWN
    # catalog: forwarding it here only under `openrouter` left the entrypoint's
    # fallback branch reading a variable nothing could set, so every fallback row
    # necessarily drove its host at a slug that account does not serve, and the
    # warning it logs about that could never be acted on. Forwarded only when
    # typed, and recorded in the row's credential census either way.
    if (row.lane == "host" and row.creds in ("openrouter", "fallback")
            and os.environ.get("MLLM_HOST_MODEL")):
        argv += ["-e", f"MLLM_HOST_MODEL={os.environ['MLLM_HOST_MODEL']}"]

    argv += [image]
    argv += ["--lane", row.lane, "--row", row.row_id, "--creds", row.creds]
    if row.cli:
        argv += ["--cli", row.cli]
    if row.host:
        argv += ["--host", row.host]
    for spec in spec_list(row):
        # REPEATED `--spec`, not one comma-joined value. A spec may itself
        # carry `/` and `:` (`opencode:stub/stub-model`, `vibe:openai/
        # gpt-5.6-luna`), and the eight distinct spellings across the 16
        # provider rows are exactly the surface a second separator would have
        # to stay clear of. `--helpers` can comma-join because a CLI id is one
        # token; a spec is not.
        argv += ["--spec", spec]
    if row.runner:
        argv += ["--runner", row.runner]
    helpers = row_helpers(row)
    if helpers:
        argv += ["--helpers", ",".join(helpers)]
    if row.scenario:
        argv += ["--scenario", row.scenario]
    if row.mock_runner:
        argv += ["--mock-runner", row.mock_runner]
    if row.dry_run_host:
        argv += ["--dry-run-host"]
    if row.route != "file":
        # Passed only when it is not the default, so every existing row's argv —
        # and therefore every existing row's rendered host config — stays
        # byte-identical.
        argv += ["--route", row.route]
    if row.probe:
        argv += ["--probe", row.probe]
    if row.documented_invocation:
        argv += ["--documented-invocation"]
    if row.command:
        argv += ["--command", row.command]
    for extra in row.commands:
        argv += ["--command", extra]
    return argv


def _bytes(value) -> int:
    """`"4g"` -> bytes, the way docker's own `--memory` reads it."""
    text = str(value).strip().lower()
    units = {"b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(float(text))


def host_posture_mismatches(inspect: dict, row: Row, out_dir: Path,
                            posture: dict | None = None) -> list[str]:
    """`docker inspect`'s `HostConfig`, field by field against `posture.yaml`.

    The half of the posture check the container cannot make. From inside, a flag is
    visible only through its *effect*, and several of these have effects that look
    fine right up until the row that needed them: a `--memory` silently dropped
    reads as "no cgroup limit", an extra `-v` reads as a mount the container was
    told about. From out here it is a comparison against the declared record, so a
    dropped flag fails a named check instead of quietly widening the container.
    """
    posture = posture or load_posture()
    host_config = inspect.get("HostConfig") or {}
    config = inspect.get("Config") or {}
    identity, security = posture["identity"], posture["security"]
    limits = posture["limits"]
    bad: list[str] = []

    def check(name: str, actual, expected) -> None:
        if actual != expected:
            bad.append(f"{name}: {actual!r}, posture.yaml declares {expected!r}")

    check("Config.User", config.get("User"),
          f"{identity['uid']}:{identity['gid']}")
    check("HostConfig.CapDrop", list(host_config.get("CapDrop") or []),
          list(security["cap_drop"]))
    check("HostConfig.SecurityOpt", list(host_config.get("SecurityOpt") or []),
          list(security["security_opt"]))
    check("HostConfig.ReadonlyRootfs", bool(host_config.get("ReadonlyRootfs")),
          bool(security.get("readonly_rootfs")))
    # Not declared anywhere because it is never a value this rig sets — which is
    # exactly why it is asserted: `--privileged` undoes every line above it.
    check("HostConfig.Privileged", bool(host_config.get("Privileged")), False)
    check("HostConfig.PidsLimit", host_config.get("PidsLimit"),
          limits["pids_limit"])
    check("HostConfig.Memory", host_config.get("Memory"), _bytes(limits["memory"]))
    check("HostConfig.NanoCpus", host_config.get("NanoCpus"),
          int(float(limits["cpus"]) * 1_000_000_000))
    # The EFFECTIVE mode, per row rather than per lane: a `--creds none` discovery
    # row is required to run routeless, and "we asked for it in an argv" is not the
    # same fact as "docker gave it to us".
    check("HostConfig.NetworkMode", host_config.get("NetworkMode"),
          network_mode(row, posture))
    check("HostConfig.Tmpfs", dict(host_config.get("Tmpfs") or {}),
          {m["target"]: m["mode"] for m in posture["mounts"]
           if m["type"] == "tmpfs"})
    # Both directions, and this is the one that catches a debugging bind of the
    # repo working tree: the run's binds must be the writable artifact directory
    # plus whatever creds.yaml allowlisted for this row, and nothing else.
    check("HostConfig.Binds", sorted(host_config.get("Binds") or []),
          sorted([f"{out_dir}:/work/out:rw"]
                 + [f"{source}:{dest}:ro" for source, dest in fallback_binds(row)]))
    return bad


def _capture_inspect(name: str, out_dir: Path, stop: threading.Event,
                     state: dict, poll: float = 0.05,
                     deadline: float = 60.0) -> None:
    """Poll `docker inspect` until the container appears, then keep its record.

    It has to run *during* the row: every run is `--rm`, so the container is gone
    the moment it exits and there is nothing left to inspect afterwards. This is
    deliberately non-fatal — a row that dies in its first fraction of a second can
    lose the race, and the artifact then says `captured: false` rather than the
    driver turning a missed poll into a failed row.
    """
    end = time.time() + deadline
    while not stop.is_set() and time.time() < end:
        completed = subprocess.run(
            ["docker", "inspect", "--format", "{{json .}}", name],
            capture_output=True, text=True)
        if completed.returncode == 0 and completed.stdout.strip():
            try:
                state["inspect"] = json.loads(completed.stdout)
            except ValueError:
                return
            (out_dir / "docker-inspect.json").write_text(
                json.dumps(state["inspect"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            return
        time.sleep(poll)


#: How far above its OWN measured turn count a row has to climb before "tool loop"
#: is a defensible diagnosis, and the floor that multiple can never fall below.
#: `redirect.yaml` measured the healthy per-exec turn count of every CLI in the
#: matrix and the spread is the whole point: codex ends a session in 1 turn, muse
#: in 17 (four parallel background reminder tasks, not retries). A single global
#: constant reads muse's normal session as a loop, and `cap-hit: tool-loop` is not
#: a cosmetic label — it is the verdict that demotes a row to the fake-binary lane.
TOOL_LOOP_FACTOR = 2
TOOL_LOOP_FLOOR = 10


def measured_turns(cli: str | None) -> int | None:
    """`redirect.yaml`'s recorded per-exec turn count for one CLI, if it has one."""
    if not cli:
        return None
    for entry in load_redirect().get("clis") or []:
        if entry.get("cli") == cli and isinstance(entry.get("turns"), int):
            return int(entry["turns"])
    return None


def tool_loop_threshold(row: Row) -> int:
    """The turn count above which THIS row's cap hit is a tool loop.

    Per-row data, never one number for the matrix. A row `redirect.yaml` does not
    name — every host-lane row, whose stub turns belong to the dispatched fan-out
    rather than to one CLI session — keeps the flat floor.
    """
    measured = measured_turns(row.cli)
    if measured is None:
        return TOOL_LOOP_FLOOR
    return max(TOOL_LOOP_FLOOR, measured * TOOL_LOOP_FACTOR)


def classify_cap_hit(result: RowResult) -> str:
    """Name the cause of a cap hit from the stub log, so it is a diagnosis.

    A cap hit is a **failure signal, never a salvage mechanism**: a killed CLI has
    no parseable output, so a capped row is never graded against a golden and never
    "mostly passed". What the caps buy is that the target always terminates with a
    verdict — the failure mode this rig must not have is a `make` target that sits
    there.
    """
    calls = [r for r in result.stub_log if r.get("surface") != "health"]
    if not calls:
        # A resolver or socket wait under --network none, an unredirected surface,
        # or a prompt nobody is answering.
        return "cap-hit: no-first-call"
    turns = max((r.get("turn") or 0) for r in calls)
    if turns >= tool_loop_threshold(result.row):
        # The CLI never accepted a tool-free answer as final: C4's precondition
        # failed for this row, which demotes it to the fake-binary lane rather
        # than raising its cap. Measured against this row's own healthy turn
        # count, so a chatty-but-correct CLI is not demoted for being chatty.
        return "cap-hit: tool-loop"
    return f"cap-hit: stalled-after-{turns}"


# --------------------------------------------------------------------------- #
# THE ONE RETRYABLE SHAPE: a host turn that never started.
#
# Measured twice, in four live runs: `runs/20260922T143440Z/claude-code-review-
# plan` and `runs/20260922T154654Z/claude-code-ask`. Both rows exited 0, both
# printed a clean result envelope — `subtype: "success"`, `is_error: false`,
# `stop_reason: "end_turn"` — and both carried an EMPTY `result`. Neither ever
# invoked our dispatcher (zero receipts once the rig's own `install-host --check`
# is discounted), neither left one byte in `artifacts/`, and between them they
# spent $0.0000 and $0.0007 of the dedicated key. The agent ended its turn
# without starting the work. There is nothing in either bundle to grade.
#
# Such a row is worthless as evidence and costs essentially nothing, so running
# it a second time is nearly free — about +$0.002 for a three-row tier in the
# worst case. That is the whole justification for retrying it, and it does not
# generalise: it is a licence for exactly this shape and no other.
#
# WHAT THIS MUST NOT FIRE ON, and the reason the clauses are written the way they
# are. Also measured twice, in `runs/20260922T124956Z/claude-code-review-plan`
# and `.../claude-code-apply-native`: the agent does most of the work, returns
# from a backgrounded native `Task` batch and ends its turn early — one lost
# `validation_batch_0.json`, or a `--mark-completed` never typed. Those rows had
# already spent $0.0966 and $0.4794 and left a half-written workspace behind.
# Retrying THEM would double the tier's cost and paper over a real product bug,
# so the predicate is built out of the facts that separate the two shapes in the
# recorded data with no overlap at all:
#
#   fact                    never-started      half-finished     green (10 rows)
#   ------------------------------------------------------------------------
#   result (chars)          0, 0               53, 894           847 … 2395
#   dispatcher receipts     0, 0               6, 2              4 … 17
#   artifacts/ files        0, 0               23, 18            15 … 29
#   key spend ($)           0.0000, 0.0007     0.0966, 0.4794    0.0121 … 1.7002
#
# Three independent sources — the host's own envelope, the shim's receipts, the
# workspace — have to agree that nothing happened before a row is re-run, and the
# spend clause bounds what a false positive could ever cost.
#
# TWO FACTS DELIBERATELY LEFT OUT.
#
# * **Output tokens** (95 and 165 here, 250 on the cheaper half-finished row).
#   A threshold there would separate the shapes by 1.5x while duplicating what
#   `result` and the receipts already separate by everything; the tightest
#   number in a predicate is the first one to misfire, and this one buys nothing.
# * **The idle gap before exit** (125.8 s and 190.2 s on the half-finished rows
#   against 10-26 s on green ones). It is a real discriminator, but it needs a
#   transcript timeline to compute and it answers a question the receipt count
#   already answers exactly.
# --------------------------------------------------------------------------- #

#: The measured ceiling on what a never-started attempt may have spent. Both
#: recorded ones are under it by 14x ($0.0007 and $0.0000); the cheapest GREEN
#: claude-code row in the same four runs is $0.0121 and the cheapest
#: half-finished one is $0.0966, so nothing that did real work is anywhere near
#: it. This is the clause that bounds the blast radius: whatever else a future
#: envelope does, a row this rig re-runs on its own authority had already been
#: measured to cost a cent or less, and a row whose spend could NOT be measured
#: is never re-run at all.
NEVER_STARTED_MAX_USD = 0.01

#: The census word for a discarded attempt. Kept out of `pass` and out of
#: `failed` for the same reason `sandbox-skip` is: an attempt that never started
#: has not failed its question, it has left it unanswered — and an unanswered
#: question is not coverage either.
RETRY_OUTCOME = "retried"

#: The manifest that makes a retry impossible to miss, written into the bundle
#: root beside the graded attempt's own evidence.
RETRY_MANIFEST_NAME = "retry.json"

#: Where a discarded attempt's bundle is kept: `runs/<utc>/<row-id>/attempt-<n>/`.
#: NESTED INSIDE the row's bundle rather than beside it, so the record's digest —
#: which is `hash_bundle`'s `rglob` over exactly this directory — covers both
#: attempts. A sibling directory would leave the discarded one unhashed, i.e.
#: rewritable after the fact without the record noticing, which is the one
#: property this suite's evidence may not lose.
ATTEMPT_DIR_FMT = "attempt-{n}"


def retry_never_started_enabled() -> bool:
    """Whether the one-shot retry is armed. `MLLM_E2E_RETRY_NEVER_STARTED=0` disarms it.

    Read per call rather than at import, like every other knob here, so a test can
    set it the way the Makefile would. Detection is NOT disarmed with it: a
    disarmed lane still writes the manifest saying the shape was seen and no
    retry was run, because "we saw it and chose not to" and "we never looked" are
    different facts about the same red row.
    """
    raw = os.environ.get("MLLM_E2E_RETRY_NEVER_STARTED", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def host_result_envelope(result: RowResult) -> dict:
    """The host's own result envelope out of `host-stdout.txt`, or `{}`.

    The LAST line that parses as a JSON object carrying `type: "result"`. Both
    shapes this lane records are covered by that rule and neither is guessed at:
    claude-code with `--output-format json` writes the envelope as the whole
    file (one line), while codex and opencode write JSONL whose earlier lines are
    events rather than verdicts. `{}` means this host printed no such envelope —
    which is itself the answer for every host that does not speak this dialect.
    """
    path = result.out_dir / "host-stdout.txt"
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "result":
            return payload
    return {}


def dispatcher_receipts(result: RowResult) -> list[dict]:
    """Every receipt except the rig's own `install-host` staging invocation.

    `install-host --check` is the entrypoint staging the shim, not the host
    running our dispatcher, so counting it would let a row pass — or, here, let a
    row that touched nothing look like a row that did — on the harness's own
    work. `test_container_host_lane._dispatcher_receipts` is the same rule and
    now delegates here: two copies of it is how the grader and the retry would
    start disagreeing about whether a row did anything.
    """
    return [receipt for receipt in result.receipts
            if "install-host" not in (receipt.get("argv") or [])]


def artifact_files(result: RowResult) -> list[str]:
    """Every file the row exported under `artifacts/`, relative and sorted."""
    root = result.out_dir / "artifacts"
    if not root.is_dir():
        return []
    return sorted(str(path.relative_to(root))
                  for path in root.rglob("*") if path.is_file())


def never_started_signature(result: RowResult) -> dict | None:
    """The evidence that this row never started, or `None` — see the block above.

    A dict rather than a bool, because the manifest and the record quote it: a
    retry that cannot say what it saw is a retry nobody can audit later.
    """
    row = result.row
    # (1) The credentialed host tier only. The provider lane prints no host
    # envelope at all, and the two credential-free host tiers (`dry_run_host`,
    # `documented_invocation`) are graded on `invocation.json` rather than on a
    # dispatcher receipt and already cost nothing — re-running one buys nothing
    # and would re-run it on a rule written for a different tier's evidence.
    if row.lane != "host" or row.dry_run_host or row.documented_invocation:
        return None
    # (2) A row that hit its cap or exited non-zero is already classified —
    # `cap-hit`, `sandbox-skip`, `fail` — and each of those has a diagnosis and
    # an owner. This shape is specifically the row that exits 0 and LOOKS green.
    if result.timed_out or result.exit_code != 0:
        return None
    envelope = host_result_envelope(result)
    if not envelope:
        return None
    # (3) The host reported a clean, complete turn. Not an error, not a cap, not
    # a refusal: `end_turn` is the host saying it finished normally, which is
    # what makes an empty answer an anomaly rather than a reported failure.
    if (envelope.get("subtype") != "success"
            or envelope.get("is_error") is not False
            or envelope.get("stop_reason") != "end_turn"):
        return None
    # (4) …and said nothing at all. An empty `result` on a successful `end_turn`
    # is the anomaly itself. `""` only — a whitespace-only answer is spelled the
    # same way here, and anything with a character in it is a turn that produced
    # something and belongs to whoever grades it.
    answer = envelope.get("result")
    if not isinstance(answer, str) or answer.strip():
        return None
    # (5) THE CLAUSE THAT SEPARATES THE SHAPES. Zero dispatcher receipts means
    # the agent never touched the shim, so this row observed nothing about the
    # product. The half-finished shape carries 2 and 6.
    receipts = dispatcher_receipts(result)
    if receipts:
        return None
    # (6) The same fact from an independent source — the workspace rather than
    # the shim — so one missing receipts directory cannot on its own authorise a
    # re-run. The half-finished shape carries 18 and 23 files.
    artifacts = artifact_files(result)
    if artifacts:
        return None
    # (7) The blast-radius bound. An unmeasured spend is NOT a small spend: a row
    # whose key readback did not answer is never re-run on this rig's initiative,
    # because the one thing an automatic retry has to guarantee is a bounded bill.
    spend = result.usage.get("delta_usd")
    if not isinstance(spend, (int, float)) or isinstance(spend, bool):
        return None
    if float(spend) > NEVER_STARTED_MAX_USD:
        return None
    usage = envelope.get("usage") or {}
    return {
        "shape": "never-started",
        "envelope": {
            "type": envelope.get("type"),
            "subtype": envelope.get("subtype"),
            "is_error": envelope.get("is_error"),
            "stop_reason": envelope.get("stop_reason"),
            "result_chars": len(answer),
            "num_turns": envelope.get("num_turns"),
            # Recorded, never gated on — see the block above.
            "output_tokens": usage.get("output_tokens"),
        },
        "dispatcher_receipts": 0,
        "artifact_files": 0,
        "spend_usd": round(float(spend), 6),
        "spend_ceiling_usd": NEVER_STARTED_MAX_USD,
        "wall_clock_s": result.wall_clock_s,
    }


def never_started_reason(signature: dict | None) -> str:
    """One sentence a record can carry, built from what was actually measured."""
    signature = signature or {}
    envelope = signature.get("envelope") or {}
    return (
        "never-started: the host reported a clean turn "
        f"(subtype {envelope.get('subtype')!r}, stop_reason "
        f"{envelope.get('stop_reason')!r}, is_error "
        f"{envelope.get('is_error')!r}) with an EMPTY result, and never invoked "
        f"the dispatcher — {signature.get('dispatcher_receipts')} receipts, "
        f"{signature.get('artifact_files')} exported artifacts, "
        f"${signature.get('spend_usd')} of key spend. The attempt observed "
        f"nothing about the product, so there is nothing in it to grade")


# --------------------------------------------------------------------------- #
# C1 — the WHOLE-LANE cap, enforced here because here is where a run is started.
#
# `MLLM_E2E_MATRIX_TIMEOUT_S` is exported by the `Makefile` and read *here*. The
# per-row cap is the `subprocess.run(timeout=…)` in `run_row`, and on its own it
# bounds nothing a developer waits on: eleven chain rows at 600 s, three
# credentialed host rows at ~1400 s and a per-CLI surface probe behind them is
# most of an afternoon, and every one of those rows is individually "within its
# cap" the whole time. An export nobody reads is worse than no cap at all — it
# reads as a bound in the Makefile, in `docs/development.md`, in this directory's
# own `README.md` and in the plan while the lane runs open-ended.
#
# So the rule is the plan's, in its own words: once the cap is spent the driver
# **launches nothing further**, reports which rows ran, which failed and which
# **never started**, and the invocation exits non-zero. Three consequences are
# deliberate:
#
# * **The clock starts at the first row, not at import.** Collection, the docker
#   pre-flight and `docker build` are not the lane; charging them to it would make
#   the same cap mean something different on a cold build cache.
# * **A row already running is never killed by this cap.** That is the per-row
#   cap's job, and it is the one that leaves readable evidence (`docker stop` with
#   grace, so the entrypoint's `finally` export still runs). The overshoot is
#   therefore bounded by one row's own `row_timeout_s`, and the refusal lands on
#   the row that would have been *started* over the line.
# * **Every later row is refused individually rather than the session being torn
#   down at the first one.** Refusing costs nothing — no container is created —
#   and it is what makes `never_started` a NAMED list instead of a count nobody
#   can act on. "The lane ran out of time" and "these six rows never ran" are the
#   two facts a summary has to keep apart.
# --------------------------------------------------------------------------- #

#: The `Makefile`'s own default, restated so a driver invoked without `make` (a
#: bare `pytest -m container`) is capped identically rather than open-ended.
DEFAULT_MATRIX_TIMEOUT_S = 3600.0

#: Written beside the run bundles, like `host-lane-census.json`: the count has to
#: outlive pytest's scrollback, because the row that reports the cap is not the
#: row a developer is looking at when they come back to a red lane.
MATRIX_CENSUS_NAME = "matrix-census.json"


def matrix_timeout_s() -> float:
    """The lane cap for this invocation in seconds; ``<= 0`` means uncapped.

    Read from the environment per invocation rather than at import, so
    `MLLM_E2E_MATRIX_TIMEOUT_S=600 make e2e-container` means what it says and a
    test can set it the same way the Makefile does. An unparseable value falls
    back to the default instead of raising: a typo'd cap must not be the thing
    that fails a lane, and a lane running uncapped *because of* a typo is exactly
    the outcome this block exists to prevent.
    """
    raw = os.environ.get("MLLM_E2E_MATRIX_TIMEOUT_S")
    if raw is None or not str(raw).strip():
        return DEFAULT_MATRIX_TIMEOUT_S
    try:
        return float(str(raw).strip())
    except ValueError:
        return DEFAULT_MATRIX_TIMEOUT_S


class MatrixExpired(RuntimeError):
    """The lane's wall-clock cap is spent, so this row was never started.

    Its message **is** the census: the row that raises it is the first thing a
    developer reads, and "which ran, which failed, which never started" is the
    only summary that tells a slow lane apart from a broken one.
    """


class MatrixBudget:
    """One clock over every row of one invocation, plus the census it produces."""

    def __init__(self) -> None:
        self.timeout_s: float = 0.0
        self.started_at: float | None = None
        #: One entry per row that really ran, in the order they ran.
        self.rows: list[dict] = []
        #: Row ids refused after the cap was spent — the never-started list.
        self.refused: list[str] = []
        #: Every `runs/<utc>/` the invocation touched, so the census lands beside
        #: each module's bundles rather than only beside the first one's.
        self.roots: list[Path] = []

    def reset(self) -> None:
        """Forget the lane. For a test that drives this budget deliberately."""
        self.__init__()

    # -- the clock ---------------------------------------------------------- #
    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.time() - self.started_at

    def remaining(self) -> float | None:
        """Seconds left, or ``None`` when this invocation is uncapped."""
        if self.started_at is None or self.timeout_s <= 0:
            return None
        return self.timeout_s - self.elapsed()

    def claim(self, row_id: str, run_root: Path | None = None) -> None:
        """Admit one row, or refuse it because the lane's clock is spent.

        Called **before** the row's bundle directory exists, so a refused row
        leaves no half-made bundle behind for a later reader to mistake for a run
        that produced nothing.
        """
        if run_root is not None and run_root not in self.roots:
            self.roots.append(run_root)
        if self.started_at is None:
            # The first row starts the lane and pins its cap: one cap per
            # invocation, so a mid-run environment change cannot move the line
            # the rows before it were measured against.
            self.timeout_s = matrix_timeout_s()
            self.started_at = time.time()
            self._write()
            return
        remaining = self.remaining()
        if remaining is None or remaining > 0:
            return
        if row_id not in self.refused:
            self.refused.append(row_id)
        self._write()
        raise MatrixExpired(self.report())

    # -- the census --------------------------------------------------------- #
    def record(self, result: RowResult, outcome: str | None = None,
               reason: str | None = None) -> None:
        """File one finished row's verdict — the CONTAINER's, not the grader's.

        The same distinction `emit_record` makes: a row whose container exited 0
        can still fail an assertion over its evidence. This census answers "what
        did the driver get to start, and how long did it take", which is the
        question a cap hit raises.
        """
        if outcome is None:
            outcome, reason = row_outcome(result)
        self.note(result.row.row_id, outcome, result.wall_clock_s, reason)

    def note(self, row_id: str, outcome: str, wall_clock_s: float,
             reason: str | None = None) -> None:
        """File a row that did not come from :func:`run_row`.

        The surface probe starts its own containers — it runs a CLI's recorded
        `--help` invocation rather than a workflow, so it has no `RowResult` and
        no bundle — and a cap that could not see those rows would report a lane
        that spent half its clock outside the census.
        """
        entry = {"row": row_id, "outcome": outcome,
                 "wall_clock_s": round(float(wall_clock_s), 2)}
        if reason:
            entry["reason"] = reason
        self.rows.append(entry)
        self._write()

    def census(self) -> dict:
        ran = [entry for entry in self.rows if entry["outcome"] == "pass"]
        # `sandbox-skip` is a FOURTH named list, beside `failed`, `cap-hit` and
        # `never_started`, and it is what stops the named skip decaying back
        # into a bare red. Today the same condition reads
        # `{"row": "chain-codex", "outcome": "fail", "reason": "container exit
        # 3"}` here, which is indistinguishable from a broken write path. Dated
        # by the run directory it is written into, so a skip cannot quietly
        # outlive its blocker without that being visible in the record.
        #
        # It is kept OUT of `failed` deliberately: a row whose sandbox could not
        # start has not failed its question, it has left it unanswered, and
        # `expired`/`failed` are what a reader counts reds from. It is equally
        # kept out of `ran`, because an unanswered question is not coverage.
        skipped = [entry for entry in self.rows
                   if entry["outcome"] == "sandbox-skip"]
        # A FIFTH list, for the same reason `sandbox_skip` is a fourth: an
        # attempt this driver discarded and re-ran has neither passed nor failed
        # its question. Counting it under `failed` would report a red lane that
        # is green, and dropping it would be the lane quietly retrying — the one
        # thing a rig whose product is evidence about reliability may not do. It
        # is kept out of `ran` too, because a discarded attempt is not coverage.
        retried = [entry for entry in self.rows
                   if entry["outcome"] == RETRY_OUTCOME]
        failed = [entry for entry in self.rows
                  if entry["outcome"] not in ("pass", "sandbox-skip",
                                              RETRY_OUTCOME)]
        return {
            "cap_s": self.timeout_s,
            # The lane total, beside each row's own wall clock: it is the number
            # this directory's own `README.md` quotes and the one a later cap is
            # derived from.
            "elapsed_s": round(self.elapsed(), 2),
            "expired": bool(self.refused),
            "ran": ran,
            "failed": failed,
            "sandbox_skip": skipped,
            "retried": retried,
            "never_started": list(self.refused),
        }

    def report(self) -> str:
        census = self.census()
        lines = [
            f"matrix cap: MLLM_E2E_MATRIX_TIMEOUT_S={self.timeout_s:.0f}s is spent "
            f"after {census['elapsed_s']:.0f}s, so the driver is starting no "
            f"further `docker run` and this invocation exits non-zero.",
            f"  ran          : {len(census['ran'])}",
        ]
        lines += [f"    {entry['row']} ({entry['wall_clock_s']}s)"
                  for entry in census["ran"]]
        lines.append(f"  failed       : {len(census['failed'])}")
        lines += [f"    {entry['row']} ({entry['wall_clock_s']}s) — "
                  f"{entry.get('reason') or entry['outcome']}"
                  for entry in census["failed"]]
        lines.append(f"  sandbox skip : {len(census['sandbox_skip'])}")
        lines += [f"    {entry['row']} ({entry['wall_clock_s']}s) — "
                  f"{entry.get('reason') or entry['outcome']}"
                  for entry in census["sandbox_skip"]]
        lines.append(f"  retried      : {len(census['retried'])}")
        lines += [f"    {entry['row']} ({entry['wall_clock_s']}s) — "
                  f"{entry.get('reason') or entry['outcome']}"
                  for entry in census["retried"]]
        lines.append(f"  never started: {len(census['never_started'])}")
        lines += [f"    {row_id}" for row_id in census["never_started"]]
        for root in self.roots:
            lines.append(f"  census       : {root / MATRIX_CENSUS_NAME}")
        lines.append(
            "  This is a LANE budget, not a per-row bound: raise it deliberately "
            "(`make e2e-container MLLM_E2E_MATRIX_TIMEOUT_S=<seconds>`) or select "
            "fewer rows. A row that is individually slow is a per-row cap hit and "
            "reports itself as one.")
        return "\n".join(lines)

    def _write(self) -> None:
        """Evidence, never a failure path — an unwritable root is skipped."""
        census = json.dumps(self.census(), indent=2, sort_keys=True) + "\n"
        for root in self.roots:
            try:
                root.mkdir(parents=True, exist_ok=True)
                (root / MATRIX_CENSUS_NAME).write_text(census, encoding="utf-8")
            except OSError:
                continue


#: The lane's one budget. Module state on purpose: `make e2e-container` collects
#: the provider lane, the chain, the surface probe and the discovery tier into a
#: single pytest invocation, and the cap is over that invocation rather than over
#: whichever module happens to be running.
MATRIX = MatrixBudget()


# --------------------------------------------------------------------------- #
# C10 — provenance write-back.
#
# A row leaves its bundle under `runs/<utc>/<row-id>/`, which is gitignored,
# prunable, and erased by `make e2e-clean`. Evidence a reviewer cannot open at the
# promotion commit cannot back a claim that ships to users, so the driver derives a
# sanitized, TRACKED record from it — host-side, because the repo tree is never
# mounted and no container can write a file in it. Three rules hold it together:
#
# * **Sanitized by construction, not by review.** Every record is validated against
#   `records/record.schema.json` before it is written, and that schema closes every
#   object. A key the driver learned to log and the schema does not name is a
#   refused write, never a record that quietly became a transcript.
# * **One record per row, and its digest covers exactly that row's bundle.** The
#   plan names the file `<utc>-<host>.json`; the subject written here is the row id,
#   which begins with the host id, because `bundle.aggregate` digests one
#   `runs/<utc>/<row-id>/` tree. A record whose digest also spanned its sibling rows
#   would go MISMATCH the moment the next row wrote a file — and MISMATCH is the one
#   verdict `verify_record.py` reserves for "something rewrote the evidence".
# * **It prints promotions; it never applies them.** Promoting a capability changes
#   which code path ships to users, so the record names the `capabilities.yaml`
#   edits its evidence justifies and a human commits both in one diff.
# --------------------------------------------------------------------------- #

#: The product's D3g vocabulary (`utils/subagent_executor.WRITE_*`) mapped onto the
#: record schema's, worst-first — so a row that dispatched several batches is
#: summarized by the outcome that STOPS the loop rather than by the last one
#: recorded. `policy-violation` is deliberately absent: the schema has no spelling
#: for it, and recording it as the neighbouring `partial` would tell a reviewer a
#: batch over-applied when what really happened is that it wrote outside the
#: declared path set. It travels in the row's `reason` instead, where a free string
#: can say what it was.
#:
#: `no-op` is LAST, below `applied`, and both halves of that matter (2026-09-22).
#:
#: * **It is here at all** because it is no longer an oddity. `1633dd4`
#:   (2026-09-16) split the classifier's empty-diff branch, so a `no-op` *claim*
#:   over an empty diff is its own outcome rather than a `clean-failure`; the
#:   three `*-apply-executor` rows and every provider-lane chain row now record it
#:   as their normal outcome. Measured over `runs/20260921T224840Z`: 65 write
#:   batches, 65 of them `no-op` — so the most common write outcome this rig
#:   produces was the one with no column, and it reached the record only as prose
#:   in `reason` via the `unspellable` path below.
#: * **It is last** because this list ranks by what a reviewer most needs to see,
#:   and `no-op` is the weakest thing a batch can report: the runner claimed
#:   nothing was needed and the byte comparison agrees. Above `applied` it would
#:   MASK the finding this lane exists to catch — a stub-fed chain row that
#:   applies a byte has leaked past the redirect (`container/README.md`: "applied
#:   is a bug in the rig"), and a row with one leaked `applied` among twelve
#:   honest `no-op`s must not be summarized by the twelve. Last also means no
#:   spelling any existing record carries changes: a row only reads `no-op` when
#:   `no-op` is ALL it recorded.
WRITE_OUTCOME_SEVERITY = [
    ("partial-application", "partial"),
    ("schema-rejection", "schema-rejected"),
    ("clean-failure", "clean-failure"),
    ("applied", "applied"),
    ("no-op", "no-op"),
]


@functools.lru_cache(maxsize=None)
def image_versions(image: str = DEFAULT_IMAGE) -> dict:
    """The image's own installed-version report, read WITHOUT starting the image.

    `docker create` + `docker cp` + `docker rm`: nothing inside ever executes, so
    this stays outside C1a's "one `docker run` per row, no shared container" rule
    by construction rather than by promise.

    Non-fatal: a missing report costs the record its `cli_version` and `pin_mode`
    columns, which is a smaller loss than a driver that cannot run without them.
    The returned dict is cached and shared — read it, never mutate it.
    """
    # Unique per attempt, not just per process: a name left behind by an earlier
    # crashed run would otherwise make `create` fail and silently cost every
    # record its version column.
    name = f"mllm-e2e-versions-{os.getpid()}-{time.monotonic_ns()}"
    try:
        created = subprocess.run(["docker", "create", "--name", name, image],
                                 capture_output=True, text=True)
    except OSError:
        return {}
    if created.returncode != 0:
        return {}
    try:
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "cli-versions.installed.json"
            copied = subprocess.run(
                ["docker", "cp", f"{name}:{IMAGE_VERSION_REPORT}", str(local)],
                capture_output=True, text=True)
            if copied.returncode != 0 or not local.is_file():
                return {}
            try:
                return json.loads(local.read_text(encoding="utf-8"))
            except ValueError:
                return {}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def cli_version(cli: str | None, image: str = DEFAULT_IMAGE) -> str | None:
    """What the image really installed for one CLI id — never what it asked for.

    `cli-versions.lock` is a request; this is the build's own measurement of what
    landed, which is the only version a `MLLM_PIN_MODE=latest` record may quote.
    """
    if not cli:
        return None
    report = image_versions(image)
    for entry in list(report.get("clis") or []) + list(report.get("toolchain") or []):
        if entry.get("id") == cli and entry.get("installed"):
            return str(entry["installed"])
    return None


#: Where the in-image `sandbox_verdict` probe files its answer, beside
#: `matrix-census.json` rather than inside it: the census is the driver's
#: per-row ledger, and this is a per-*capability* measurement that several rows
#: may share.
SANDBOX_VERDICT_NAME = "sandbox-verdicts.json"

#: Inside the image the skill is installed here. The probe imports the SHIPPED
#: adapter from the SHIPPED skill rather than re-deriving codex's argv
#: test-side, which is the whole point: a second copy of
#: `codex sandbox -c 'sandbox_mode="workspace-write"' -- …` drifts from the
#: first the moment the adapter changes.
IMAGE_SKILL_DIR = "/opt/mllm/skill"

#: The skill's own interpreter, with the system one behind it. `python3` runs
#: the whole rig and carries PyYAML (`Dockerfile:218-230`), but the product is
#: installed against `UV_PROJECT_ENVIRONMENT=/opt/mllm/skill/.venv`
#: (`Dockerfile:212`), so the venv is the interpreter the adapters actually run
#: under and is tried first. Falling back rather than hard-coding one, because
#: an interpreter that is merely absent must not read as `unknown` — which on
#: this mechanism is a hard failure.
IMAGE_SKILL_PYTHON = "/opt/mllm/skill/.venv/bin/python"

_SANDBOX_PROBE_SOURCE = """
import json, sys
sys.path.insert(0, {skill_dir!r})
from utils.provider_registry import get_provider
from utils.subagent_executor import sandbox_verdict
provider = get_provider({cli!r})
if provider is None:
    print(json.dumps({{"state": "unknown",
                       "tail": "no built-in adapter for " + {cli!r}}}))
else:
    # Indexed, never a two-value unpack: `sandbox_verdict` returns a NamedTuple
    # whose third field is the fallback rung, and a stale `state, tail = …`
    # raises rather than drop it. Indexing also reads an older image's plain
    # `(state, tail)` the same way.
    verdict = sandbox_verdict(provider, {profile!r})
    print(json.dumps({{"state": verdict[0], "tail": verdict[1],
                       "rung": getattr(verdict, "rung", None)}}))
"""


def sandbox_verdict_in_image(cli: str, profile: str = "agent-write",
                             image: str = DEFAULT_IMAGE) -> tuple[str, str]:
    """``(state, tail)`` for one CLI's sandbox self-test, measured INSIDE the image.

    This is the *licence* half of the named skip, and it is deliberately
    separate from the classification half (:func:`row_outcome`). The
    classification says what happened; this says whether forgiving it is still
    permitted **on this run**.

    A free-text "skipped, dated" line is the allowlist shape that outlives its
    blocker, because nothing on a later run re-asks whether the blocker is
    still there. So the permission comes from a capability probe re-measured
    every run, run through the adapter's own argv taken from the skill
    installed in the container — not from a message, and not from a test-side
    re-derivation of the detection.

    Three states, three different consequences on the test side:

    * ``unavailable`` — the row still RUNS, and its expected failure is
      declared as a strict, typed xfail on :class:`SandboxSkipped`.
    * ``ok`` / ``not-required`` — **no skip is available at all.** The row is
      graded like every other row, and a `sandbox-skip` outcome under an `ok`
      verdict is itself a failure: the classification and the capability
      disagree and one of them is wrong.
    * ``unknown`` — a hard failure, not a licence. The probe could not answer,
      so nothing was learned about the sandbox, and an unreadable verdict must
      not become the cheap path to a green run.

    Runs in its own throwaway container with no network and no credential: the
    probe spawns bwrap and reads an exit code, and needs nothing else.
    """
    source = _SANDBOX_PROBE_SOURCE.format(
        skill_dir=IMAGE_SKILL_DIR, cli=cli, profile=profile)
    shell = (
        f'if [ -x {shlex.quote(IMAGE_SKILL_PYTHON)} ]; then '
        f'  exec {shlex.quote(IMAGE_SKILL_PYTHON)} -c "$MLLM_PROBE_SRC"; '
        f'else exec /usr/bin/python3 -c "$MLLM_PROBE_SRC"; fi'
    )
    try:
        completed = subprocess.run(
            # The source travels in the ENVIRONMENT, not in the argv: it
            # contains quotes and newlines, and a `sh -c` that had to re-quote
            # it would be one escaping bug away from probing the wrong thing
            # and reporting `unknown` — which is a hard failure on this
            # mechanism, so it must never be a rig artifact.
            ["docker", "run", "--rm", "--network", "none",
             # THE REAL BINARY, NOT THE SHIM. `PATH` is wrapper-first for the
             # whole image, so a bare `codex` in the adapter's probe argv
             # resolves to `/opt/mllm/bin/codex` — the rig's redirector, which
             # on a bare `docker run` has none of a row's tmpfs mounts and
             # dies in `seed_invocation_root` with
             # `PermissionError: /tmp/mllm-inv`. Measured 2026-09-22: that
             # crash was reported `unavailable`, which is a FALSE verdict — it
             # says the sandbox cannot start when what could not start was the
             # wrapper. The question here is whether the real CLI's own
             # sandbox comes up, so the real directory goes first.
             "-e", "PATH=/opt/mllm/real:/usr/local/bin:/usr/bin:/bin",
             # ...and a writable /tmp, because a probe that fails on the
             # filesystem has learned nothing about bubblewrap either.
             "--tmpfs", "/tmp:rw,exec,mode=1777",
             "-e", f"MLLM_PROBE_SRC={source}",
             "--entrypoint", "/bin/sh", image, "-c", shell],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "unknown", f"the probe could not be run: {exc}"
    if completed.returncode != 0:
        return "unknown", (completed.stderr or completed.stdout or "")[-500:]
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return "unknown", (completed.stdout or "")[-500:]
    return str(payload.get("state") or "unknown"), str(payload.get("tail") or "")


def record_sandbox_verdict(roots: Iterable[Path], cli: str, profile: str,
                           state: str, tail: str) -> None:
    """File one `(state, tail)` beside the census, so the licence is auditable.

    Evidence, never a failure path: an unwritable root is skipped, exactly as
    :meth:`MatrixBudget._write` does. A run whose verdict nobody can read back
    is the one case where "it was skipped for a reason" is unfalsifiable, which
    is why this is written at all.
    """
    for root in roots:
        path = Path(root) / SANDBOX_VERDICT_NAME
        try:
            existing = _read_json(path) if path.is_file() else {}
            existing[f"{cli}:{profile}"] = {"state": state, "tail": tail}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        except OSError:
            continue


def image_evidence(image: str = DEFAULT_IMAGE) -> dict:
    """`{tag, digest, platform[, pin_mode]}` from `docker image inspect`.

    Empty when the daemon cannot answer or the id is not a sha256 — the caller
    then declines to write a record at all, because an image column nobody can
    check is the one field that makes the rest of the record unfalsifiable.
    """
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .}}", image],
            capture_output=True, text=True)
    except OSError:
        return {}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    try:
        inspect = json.loads(completed.stdout)
    except ValueError:
        return {}
    digest = str(inspect.get("Id") or "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        return {}
    platform = "/".join(part for part in (inspect.get("Os"),
                                          inspect.get("Architecture"),
                                          inspect.get("Variant")) if part)
    evidence = {"tag": image, "digest": digest, "platform": platform}
    mode = image_versions(image).get("mode")
    if mode in ("pinned", "latest"):
        evidence["pin_mode"] = mode
    return evidence


def unreadable_bundle_files(out_dir: Path) -> list[Path]:
    """Bundle files this uid cannot open — the reason a digest can be missing."""
    return [path for path in sorted(out_dir.rglob("*"))
            if path.is_file() and not os.access(path, os.R_OK)]


def bundle_block(out_dir: Path, archive_url: str | None = None) -> dict | None:
    """Per-file sha256 of the ephemeral bundle plus one aggregate over the list.

    `root` is written relative to this directory so a record carries no
    machine-specific absolute path — the anchor `verify_record._resolve_root`
    tries first.

    ``None`` when a file in the bundle cannot be read from out here: the container
    writes as uid 10001 and the product's own result files are deliberately 0600,
    so the developer's uid can be locked out of part of its own evidence. That
    costs the record its digest — `verify_record.py` then reports "carries no
    bundle block" instead of a false MISMATCH — and it must never cost the row its
    verdict. Skipping the unreadable file instead is the one option not taken: a
    digest over a subset would disagree with the verifier's own walk, which is the
    verdict reserved for "something rewrote the evidence".
    """
    try:
        files, aggregate = hash_bundle(out_dir)
    except OSError:
        return None
    resolved = out_dir.resolve()
    try:
        root = str(resolved.relative_to(HERE))
    except ValueError:
        root = str(resolved)
    block = {"root": root, "files": files, "aggregate": aggregate}
    if archive_url:
        block["archive_url"] = archive_url
    return block


def declared_command(row: Row) -> str:
    """Every dispatcher invocation **the subject of the row** drives, as one string.

    The provider lane types its commands into the run argv (`row.command`); the
    host lane does **not** — the entrypoint reads them out of `host_matrix.yaml`
    by row id, so a driver that looked only at the Row would conclude that every
    host row typed nothing at all, and would then never record a `--detach` or
    `--exec-subagents` row as evidence for the capability it exercised.

    **On the host lane the PRE-COMMAND LEG is excluded, and that is the whole
    point of the branch.** `Row.commands` now carries a host row's `pre_commands`
    — dispatcher invocations *the rig* makes in the row's workspace before the
    host process starts — and the old `" ".join([row.command, *row.commands])`
    returned those instead of the row's own `command`, because a host row's
    `row.command` is empty. Two things went wrong at once and both are silent:
    the record's `mode` became the leg's mode word (`review-plan`) rather than
    the row's (`apply-suggestions`), and `--exec-subagents` typed by the RIG
    would have been promoted as evidence that the HOST exercised the executor.
    That is the same false positive the `receipts/<row>-pre/` split exists to
    prevent, one layer up: capability evidence has to come from the host's own
    invocation or it is not evidence about the host.
    """
    if row.lane == "host":
        for entry in (load_host_matrix().get(row.host or "") or {}).get("rows") or []:
            if entry.get("id") == row.row_id:
                return str(entry.get("command") or "") or row.command
        return row.command
    return " ".join([row.command, *row.commands]).strip()


class SandboxSkipped(AssertionError):
    """This row's write batches were refused because its sandbox could not start.

    Raised — and **only** raised — where :func:`row_outcome` classified the row
    `sandbox-skip`, so the test side can declare the expected failure as a
    *typed* `pytest.mark.xfail(strict=True, raises=harness.SandboxSkipped)`.

    Both halves of that are load-bearing. *Typed*, so a row failing for any
    other reason raises something else and is an ordinary red — the conflation
    an entrypoint-level skip list cannot avoid, because a row skipped before it
    runs cannot tell "the sandbox could not start" from "the write phase was
    broken some other way". *Strict*, so the day bwrap starts here the row
    passes, pytest reports XPASS, and the suite is red until the declaration is
    retired and the baseline captured.

    It is deliberately **not** `pytest.skip`: a skipped row is never executed,
    so it can never announce that its blocker has lifted. An allowlist that
    outlives its blocker is the failure mode this whole mechanism exists to
    prevent.

    It subclasses `AssertionError` rather than `Exception` so a grader that
    catches assertion failures keeps behaving as it does for every other row.
    """


def sandbox_refusals(result: RowResult) -> list[dict]:
    """Every batch record this row wrote that names a `sandbox_unavailable` refusal.

    Derived from the row's **own evidence** — the same
    `write_batch_report.json` artifacts :func:`write_outcomes` already walks —
    rather than from a second, test-side copy of the detection. The product
    already knows: `_detect_sandbox` (`utils/subagent_executor.py:1754`) runs
    the adapter's probe, `sandbox_verdict` (`:1655`) returns `unavailable`, and
    the refusal it records carries `reason_code="sandbox_unavailable"`
    (`:1814`, D3k rung 1). Re-deriving that outside the container drifts from
    it the moment the probe changes.

    Each element is the product's own record, so the caller can quote its
    reason string instead of inventing one.

    **Two sources, in order, and the second is not optional today.** Measured
    on `runs/20260921T224840Z/chain-writer-codex`: a genuinely sandbox-refused
    row's report is `{"exit": 3, "batches_dispatched": [], "tasks": []}` with
    **no `reason_code` anywhere in it** — `preflight_runner` raises
    `RunnerRejected` *before* the first batch record exists, so the code never
    reaches the file. The reports are read first because they are the declared
    source and will start hitting the day the product records it there; the
    fallback reads the product's own refusal line out of the container's
    stderr/stdout (`utils/subagent_executor.py:1806`,
    `Refusing to dispatch: … (reason: sandbox_unavailable) — …`).

    The fallback is **anchored**, not a bare string search: a write report must
    exist and must record a non-zero `exit`. A green row that merely mentions
    the string — in a prompt, a log echo or a test name — cannot ride out on
    this door.
    """
    found: list[dict] = []
    reports = sorted((result.out_dir / "artifacts").glob(
        "*/*/write_batch_report.json"))
    refused_exit = False
    for report in reports:
        payload = _read_json(report)
        if payload.get("exit") not in (0, None):
            refused_exit = True
        # The refusal can be filed on the batch itself or on one of its tasks,
        # depending on whether the preflight or the dispatch caught it; look in
        # both rather than guessing which rung produced this one.
        candidates = [payload, *(payload.get("tasks") or [])]
        for node in candidates:
            if not isinstance(node, dict):
                continue
            if str(node.get("reason_code") or "") == "sandbox_unavailable":
                found.append(node)
    if found or not refused_exit:
        return found
    for name in ("container-stderr.txt", "container-stdout.txt"):
        path = result.out_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "Refusing to dispatch" not in line:
                continue
            if "(reason: sandbox_unavailable)" not in line:
                continue
            found.append({"reason_code": "sandbox_unavailable",
                          "detail": line.strip(),
                          "source": name})
            return found
    return found


def row_outcome(result: RowResult) -> tuple[str, str | None]:
    """The driver's own verdict on a row, with the reason it is not a bare bool.

    This is the container's verdict, not the grader's: a row whose container
    exited 0 can still fail an assertion over its evidence, which is why
    :func:`emit_record` takes an explicit outcome and this is only the default.

    **`sandbox-skip` is classified here and nowhere else**, derived exactly the
    way `cap-hit` already is: from the row's own recorded evidence, in one
    place. A row whose write batches carry `reason_code: sandbox_unavailable`
    is not `fail` / `container exit 3` — that bare red is what
    `runs/20260920T105938Z/matrix-census.json` says about `chain-codex` today,
    and it reads identically to a broken write path.

    Note the ordering: the classification is *below* the cap-hit and exit-0
    branches. A row that ran out of wall clock is a cap hit whatever its
    batches recorded, and a row that exited 0 passed — the sandbox refusal
    only re-labels a **non-zero** exit whose evidence explains it.
    """
    if result.timed_out:
        return "cap-hit", result.cap_reason or classify_cap_hit(result)
    if result.exit_code == 0:
        return "pass", None
    if result.exit_code is None:
        return "not-run", "the container produced no exit status"
    refusals = sandbox_refusals(result)
    if refusals:
        # The product's own words, not the rig's paraphrase of them.
        reason = str(refusals[0].get("detail")
                     or refusals[0].get("message")
                     or "sandbox_unavailable")
        return "sandbox-skip", reason
    return "fail", f"container exit {result.exit_code}"


def write_outcomes(result: RowResult) -> set[str]:
    """Every `write_outcome` this row's batches recorded, in the product's own words."""
    seen: set[str] = set()
    for report in sorted((result.out_dir / "artifacts").glob(
            "*/*/write_batch_report.json")):
        for task in (_read_json(report).get("tasks") or []):
            if task.get("write_outcome"):
                seen.add(str(task["write_outcome"]))
    return seen


def credential_mode(row: Row, result: RowResult) -> str:
    """What credential path this row really took, read back from its own evidence.

    The operator's `--creds` is a request; `credential-census.json` is what the
    entrypoint recorded doing. They differ exactly where it matters — a `--creds
    none` row that reached the discovery tier is `discovery-only`, which is a
    fourth count and not a quiet zero.
    """
    recorded = str(result.credential_census.get("credential_mode") or "")
    if recorded in ("openrouter", "fallback"):
        return recorded
    # Both credential-free host tiers land here, and both really exercised the
    # host: `discovery.json` is the listing tier's verdict, `invocation.json` the
    # documented-invocation probe's. A row that reached either is `discovery-only`
    # rather than a quiet zero — the distinction the census exists to keep.
    if ((result.out_dir / "discovery.json").is_file()
            or (result.out_dir / "invocation.json").is_file()):
        return "discovery-only"
    return row.creds if row.creds in ("openrouter", "fallback", "none") else "none"


def cost_block(row: Row, result: RowResult) -> dict | None:
    """The row's measured spend, or an explicit admission that nothing answered.

    Never a number that was not measured: the dedicated key's `usage` delta is
    attributable to this row precisely because the key is dedicated and rows run
    one at a time, and when the readback did not answer this says
    `actual: unavailable` rather than crediting the estimate. No `estimate_*`
    fields yet — `cost_table.yaml` is Phase 4's, and a seeded constant written as
    if it were measured is the failure this block exists to prevent.
    """
    usage = result.usage
    if not usage and not (row.lane == "host" and row.creds == "openrouter"):
        return None
    delta = usage.get("delta_usd")
    if isinstance(delta, (int, float)) and not isinstance(delta, bool):
        return {"actual_usd": round(float(delta), 6),
                "actual_source": "key-usage-delta"}
    return {"actual_usd": None, "actual_source": "unavailable"}


def row_record(row: Row, result: RowResult, outcome: str | None = None,
               reason: str | None = None, image: str = DEFAULT_IMAGE) -> dict:
    """One entry of the record's `rows` array, derived from the bundle."""
    if outcome is None:
        outcome, default_reason = row_outcome(result)
        reason = reason or default_reason
    calls = [entry for entry in result.stub_log if entry.get("surface") != "health"]
    entry: dict = {
        "row_id": row.row_id,
        "outcome": outcome,
        "wall_clock_s": result.wall_clock_s,
        "turns": max([int(call.get("turn") or 0) for call in calls] or [0]),
        "stub_requests": len(calls),
        "unhandled_routes": sum(1 for call in result.stub_log
                                if call.get("unhandled")),
        "unmatched_prompts": sum(1 for call in result.stub_log
                                 if call.get("unmatched")),
        "credential_mode": credential_mode(row, result),
    }
    if row.route != "file":
        # Written only when it is not the default, so every existing record
        # stays byte-identical: a column that appeared on rows nothing changed
        # would make a route-blind run look like a route-aware one.
        entry["route"] = row.route
    recorded = write_outcomes(result)
    for product_word, schema_word in WRITE_OUTCOME_SEVERITY:
        if product_word in recorded:
            entry["write_outcome"] = schema_word
            break
    # The escape hatch STAYS, and it is narrower than it was: `no-op` left it for
    # a column of its own on 2026-09-22, which leaves `policy-violation` — a word
    # the schema declines to spell rather than one nobody thought of — plus
    # whatever the product names next. A word with no column reaching a free
    # string is the behaviour that keeps an unknown outcome in the record at all;
    # the alternative is a refused write, because the schema closes the enum.
    unspellable = sorted(recorded.difference(
        word for word, _ in WRITE_OUTCOME_SEVERITY))
    if unspellable:
        reason = "; ".join(part for part in
                           (reason, f"write batches also recorded "
                                    f"{', '.join(unspellable)}") if part)
    # THE FLAG. A row that only passed because it was re-run says so in its own
    # record entry, not only in the manifest beside the bundle: the record is
    # what a reviewer reads at the promotion commit, and position in the `rows`
    # array is not a fact anyone should have to infer a retry from.
    if result.attempts:
        reason = "; ".join(part for part in (
            reason,
            f"attempt {result.attempt} of {result.attempt}: this row was "
            f"re-run after a never-started attempt — the discarded attempt is "
            f"the preceding row entry, its bundle is "
            f"{ATTEMPT_DIR_FMT.format(n=1)}/ and "
            f"{RETRY_MANIFEST_NAME} beside it is the manifest") if part)
    if reason:
        entry["reason"] = reason
    if row.cli:
        entry["cli"] = row.cli
    if row.host:
        entry["host"] = row.host
        # The PINNED slug, straight from host_matrix.yaml — never a run-time
        # "cheapest" lookup, which is the thing that makes a cost table
        # meaningless. Only on a row that really took a credential path: a
        # discovery-only row made no model call, and a pin recorded beside one
        # reads as a model that ran.
        # THE ROW'S OWN PIN WINS over the host's, because since 2026-08-21 one
        # row carries one (`claude-code-apply-native`, see the matrix). Reading
        # only the host pin here would record the slug the row did NOT drive,
        # against a cost figure ~60x the host pin's — the exact class of quiet
        # lie a cost table exists to avoid.
        host_entry = load_host_matrix().get(row.host) or {}
        pin = ((host_entry.get("model") or {}).get("slug"))
        for matrix_row in host_entry.get("rows") or []:
            if matrix_row.get("id") == row.row_id:
                pin = ((matrix_row.get("model") or {}).get("slug")) or pin
                break
        if pin and entry["credential_mode"] in ("openrouter", "fallback"):
            entry["model"] = str(pin)
    version = cli_version(row.host or row.cli, image)
    if version:
        entry["cli_version"] = version
    # The dispatcher mode word the row drove, which is the other half of a
    # (host x mode) cell.
    mode = declared_command(row).split()
    if mode and not mode[0].startswith("-"):
        entry["mode"] = mode[0]
    cost = cost_block(row, result)
    if cost:
        entry["cost"] = cost
    return entry


def census_block(row: Row, result: RowResult, outcome: str) -> dict:
    """The credential census: how many hosts ran in each mode, each one named.

    `3/0/0/0` and `0/0/3/0` are both possible outcomes of the same command, and a
    summary that cannot tell them apart is exactly how zero host coverage passes
    for a green matrix. The lane's own count (`host-lane-census.json`, written
    beside the bundles by the host-lane module) is preferred where it exists,
    because it is the only place a host that never ran at all is named — a skipped
    host leaves no bundle to derive anything from.
    """
    lane = _read_json(result.out_dir.parent / "host-lane-census.json")
    observed: list[tuple[str, str, str | None]] = []
    for entry in lane.get("ran") or []:
        mode = str(entry.get("credential_mode") or "openrouter")
        observed.append((str(entry.get("host") or ""),
                         "discovery-only" if mode == "none" else mode, None))
    for entry in lane.get("skipped") or []:
        observed.append((str(entry.get("host") or ""), "skipped",
                         entry.get("reason")))
    if not observed:
        # No lane count beside the bundle — this row is the only host this record
        # can speak for. `none` is not one of the four counts on purpose: a row
        # that took no credential path and did not even reach the discovery tier
        # covered nothing, and `skipped` is what "covered nothing" is called.
        mode = credential_mode(row, result)
        covered = mode not in ("none",) and outcome not in ("skip", "not-run")
        observed = [(row.host or "", mode if covered else "skipped",
                     None if covered else f"row outcome {outcome}, "
                                          f"credential mode {mode}")]

    counts = {"openrouter": 0, "fallback": 0, "skipped": 0, "discovery_only": 0}
    detail: list[dict] = []
    for host, actual, why in observed:
        if not host or any(seen["host"] == host for seen in detail):
            continue
        if actual not in ("openrouter", "fallback", "skipped", "discovery-only"):
            # An unrecognized mode is counted as coverage this run did NOT have,
            # and named — the one thing a census must never do is round an
            # unreadable cell up into a mode it cannot vouch for.
            why = why or f"unrecognized credential mode {actual!r}"
            actual = "skipped"
        counts[actual.replace("-", "_")] += 1
        cell = {"host": host, "actual": actual}
        # Phase 1's MEASUREMENT of what credential path exists for this host,
        # which is what Phase 4 grades the row against — not the operator's
        # `--creds` for the invocation.
        expected = (load_host_matrix().get(host) or {}).get("credential_mode")
        if expected in ("openrouter", "fallback", "none"):
            cell["expected"] = expected
        if why:
            cell["reason"] = str(why)
        detail.append(cell)
    counts["detail"] = detail
    return counts


#: The GRADER's three states, kept apart from the container's `row_outcome`
#: vocabulary on purpose — they answer different questions and a row has one of
#: each. `ungraded` is not a synonym for `fail`: "pytest asserted nothing about
#: this row" and "pytest failed this row" are different facts, and only the second
#: one is a finding. Both withhold promotions.
GRADED_PASS = "pass"
GRADED_FAIL = "fail"
UNGRADED = "ungraded"

#: What a record says while it is still waiting for its grader. Written into the
#: row's `reason` rather than left implicit: a record left behind by an
#: interrupted session must state that its verdict is the container's alone,
#: because otherwise it reads exactly like a graded pass.
PROVISIONAL_REASON = (
    "provisional: this is the container's verdict alone — pytest had not finished "
    "grading the row when the record was written, so no promotion is emitted from it"
)

#: The other half of the same honesty: a row that ran and that no assertion ever
#: looked at is not evidence for anything, however cleanly its container exited.
UNGRADED_REASON = (
    "ungraded: no test reported a verdict on this row, so it promotes nothing"
)


def _behavioural_fields(row: Row, result: RowResult) -> set[str]:
    """The route-neutral half: what the host DID, not how it was installed.

    Both halves of each rule, because either alone is the wrong claim:
    ``--detach`` typed with no terminal worker record is a launcher that returned
    over a worker nobody watched, and a terminal record on a row that never asked
    to detach says nothing about the host's detach mechanism.
    """
    fields: set[str] = set()
    typed = declared_command(row)
    if "--detach" in typed and any(
            (state or {}).get("state") == "done"
            for state in result.run_states.values()):
        fields.add("detach")
    if "--exec-subagents" in typed:
        fields.add("subagents")
    return fields


def promotable_fields(row: Row, result: RowResult) -> set[str]:
    """Which `capabilities.yaml` fields THIS row's artifacts are evidence for.

    Deliberately narrow, and every entry names something the bundle shows rather
    than something the row implies. A receipt is a dispatcher invocation that
    really happened, which is simultaneously evidence that the shim landed where
    the host's `install_paths` say, that the host found it after its `reload`
    boundary and that it expanded our `arguments` into it. `detach` and
    `subagents` need their own signal, because a foreground row proves nothing
    about either.
    """
    if row.lane != "host" or not row.host:
        return set()
    if row.probe:
        # A zero-quota PROBE row is not a host turn. Its receipts are the
        # harness's own stdio client driving the dispatcher through the same
        # receipt-writing `uv` wrapper, so crediting them would promote
        # `package` and `multi_turn` on a row where the host took no turn at
        # all — the `documented_invocation` trap below, in another shape.
        return set()
    fields: set[str] = set()
    if row.route == "package":
        # A package row promotes `package` plus the route-NEUTRAL behavioural
        # cells, and never the four file-route ones. The behavioural cells are
        # genuinely neutral — `detach`, `subagents` and `multi_turn` are things
        # the host does once the skill is reachable, and how it became reachable
        # does not change them — while `shim`, `install_paths`, `arguments` and
        # `reload` are claims about a FILE this row never wrote.
        #
        # `package` itself needs the same two-part evidence a receipt gives the
        # file route: the payload was found and the dispatcher really ran —
        # AND the install must have come through the Git route. Rule (j):
        # local-build evidence never promotes a distribution claim, and both
        # the local-path seed and the Git install spell `route: package`, so
        # the gate keys on the RECORDED install source, not on the route word.
        # A row that recorded none (or `local-path`) earns the behavioural
        # cells only.
        if result.receipts and result.install_source == "git":
            fields.add("package")
        if len(result.receipts) >= 2:
            fields.add("multi_turn")
        fields |= _behavioural_fields(row, result)
        return fields
    if row.documented_invocation:
        # THE ONE ROW WHOSE RECEIPTS ARE NOT THE HOST'S. Its only dispatcher
        # invocation is the rig's own `install-host`, which `dispatch()` runs
        # through the same `uv` wrapper that writes receipts — so the receipt
        # rule below would credit the HOST with a call it never made, and a
        # second receipt from the same leg would promote `multi_turn` on a row
        # where the host took no turn at all. Both were observed on
        # `runs/20260824T164940Z`, where codex REFUSED the command and still
        # printed five promotions.
        #
        # This row's evidence is `invocation.json` and nothing else. A host that
        # resolved the published `/multi-llm …` string in a fresh session found
        # the shim at the path `install_paths` declares, crossed its `reload`
        # boundary to do it, and expanded our `arguments` into the turn — which
        # is a stronger demonstration of those four than a receipt, because it
        # is the spelling the documents publish. A host that did not resolve it
        # demonstrates none of them, `shim` included: an installed file at a path
        # we chose is `install-host --check`'s claim, and this lane is explicit
        # that a green `--check` is not discovery.
        if (result.invocation or {}).get("resolved"):
            fields |= {"shim", "install_paths", "arguments", "reload"}
        return fields
    if result.receipts:
        fields |= {"shim", "install_paths", "arguments", "reload"}
    if len(result.receipts) >= 2:
        fields.add("multi_turn")
    fields |= _behavioural_fields(row, result)
    return fields


def candidate_promotions(row: Row, result: RowResult) -> list[dict]:
    """The edits this row's ARTIFACTS could justify, before any verdict is applied.

    Split out from :func:`promotions` so the two questions stay separable: *what
    would this bundle support* is a property of the files on disk, and *may it be
    promoted* is a property of the two verdicts. The summary needs the first to
    say what a failing row withheld; only the second may ever reach a reviewer as
    a proposed edit.

    Only a field whose provenance is currently below `verified` appears:
    re-promoting a verified cell would be noise in the diff a reviewer has to read
    beside the evidence.
    """
    if not row.host:
        return []
    provenance = (load_capabilities().get(row.host) or {}).get("provenance") or {}
    edits = []
    for field_name in sorted(promotable_fields(row, result)):
        was = provenance.get(field_name)
        if was in ("unknown", "documented"):
            edits.append({"host": row.host, "field": field_name, "from": was,
                          "to": "verified", "evidence": row.row_id})
    return edits


def promotions(row: Row, result: RowResult, outcome: str,
               graded: str = UNGRADED) -> list[dict]:
    """The `capabilities.yaml` edits this row justifies — printed, never applied.

    **Two verdicts, and a promotion needs both.** `outcome` is the container's:
    the run exited 0. `graded` is pytest's: the assertions over the evidence that
    run left behind also passed. They are genuinely different facts — a row can
    exit 0 having produced output that then fails its own assertions, which is
    exactly how `codex-review-plan` printed five promotions on a run where the
    worker was never detached (`expect.run: done`, observed all-`None`), and how
    `claude-code-ask` printed three on a run whose host exited 1 with
    `error_max_turns`.

    `graded` therefore defaults to :data:`UNGRADED`, which promotes **nothing**.
    That default is the fix rather than a formality: the only way to emit a
    promotion is for a caller to state, in as many words, that a grader looked at
    this row and passed it. A caller that has no grader verdict to offer — the
    driver at `run_row` time, where pytest has not begun asserting yet — cannot
    accidentally promote by omission, because there is no longer a code path from
    "the container exited 0" to a proposed capability edit.
    """
    if outcome != "pass" or graded != GRADED_PASS:
        return []
    return candidate_promotions(row, result)


def _schema_errors(node, schema, path: str = "<root>") -> list[str]:
    """The sanitization half of the schema, without `jsonschema` installed.

    `jsonschema` is a dev extra and is what validates a record when it is there.
    This is the fallback, and it deliberately checks the properties C10 leans on —
    closed key sets, required keys, enums — rather than degrading to no check at
    all, because "the writer refuses any key outside this schema" is the whole
    reason a record can be trusted without being read line by line.
    """
    bad: list[str] = []
    if not isinstance(schema, dict):
        return bad
    if "const" in schema and node != schema["const"]:
        bad.append(f"{path}: {node!r} is not {schema['const']!r}")
    if "enum" in schema and node not in schema["enum"]:
        bad.append(f"{path}: {node!r} is not one of {schema['enum']}")
    if isinstance(node, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in node:
                bad.append(f"{path}: missing required key {key!r}")
        extra = schema.get("additionalProperties")
        for key, value in node.items():
            if key in properties:
                bad += _schema_errors(value, properties[key], f"{path}/{key}")
            elif extra is False:
                bad.append(f"{path}: {key!r} is not a key this schema allows")
            elif isinstance(extra, dict):
                bad += _schema_errors(value, extra, f"{path}/{key}")
    elif isinstance(node, list) and isinstance(schema.get("items"), dict):
        for index, value in enumerate(node):
            bad += _schema_errors(value, schema["items"], f"{path}[{index}]")
    return bad


def validate_record(record: dict) -> list[str]:
    """Every way this record disagrees with `records/record.schema.json`."""
    schema = load_record_schema()
    try:
        import jsonschema
    except ImportError:
        return _schema_errors(record, schema)
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(part) for part in error.path) or '<root>'}: "
            f"{error.message}"
            for error in sorted(validator.iter_errors(record),
                                key=lambda error: list(error.path))]


def build_record(row: Row, result: RowResult, posture: dict | None = None,
                 image: str = DEFAULT_IMAGE, outcome: str | None = None,
                 reason: str | None = None,
                 archive_url: str | None = None,
                 graded: str = UNGRADED) -> dict:
    """Assemble one row's record. Raises nothing; validation is the writer's job.

    `graded` is the grader's verdict and defaults to :data:`UNGRADED`, so a caller
    that has not been told a row passed its assertions writes a record with an
    empty `promotions` array.
    """
    posture = posture or load_posture()
    if outcome is None:
        outcome, default_reason = row_outcome(result)
        reason = reason or default_reason
    network = posture["network"]
    record = {
        "schema_version": 1,
        # The bundle directory's own <utc> segment, so the record and the evidence
        # it digests name the same run without a second clock being consulted.
        "started_at": result.out_dir.parent.name,
        "ended_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "lane": row.lane,
        "image": image_evidence(image),
        # Which guarantee this row actually had: `none` is structural (there is no
        # route to a real endpoint), `open` is detection-only.
        "egress": ("none" if network_mode(row, posture) == "none"
                   else str(network.get("egress") or "open")),
        # CHRONOLOGICAL, discarded attempts first. The record is evidence and
        # evidence is in the order it happened; every entry names its own
        # attempt number in `reason`, so position is never the only carrier of
        # that fact. A single-attempt row renders exactly the one-element array
        # it always did.
        "rows": [*(row_record(row, prior, "fail",
                              never_started_reason(prior.retry_signature), image)
                   for prior in result.attempts),
                 row_record(row, result, outcome, reason, image)],
        # cap-hit and skip are row verdicts; a run is pass, fail or not-run.
        "outcome": ("pass" if outcome == "pass"
                    else "not-run" if outcome == "not-run" else "fail"),
        "promotions": promotions(row, result, outcome, graded),
    }
    bundle = bundle_block(result.out_dir, archive_url)
    if bundle:
        record["bundle"] = bundle
    if row.creds in ("openrouter", "fallback", "none"):
        record["creds_mode"] = row.creds
    if row.lane == "host":
        record["census"] = census_block(row, result, outcome)
    return record


def emit_record(row: Row, result: RowResult, posture: dict | None = None,
                image: str = DEFAULT_IMAGE, outcome: str | None = None,
                reason: str | None = None, records_dir: Path | None = None,
                archive_url: str | None = None,
                graded: str = UNGRADED) -> Path | None:
    """Derive, validate and write this row's tracked record. Idempotent per row.

    Re-emitting with a graded `outcome` rewrites the same path, so the record a
    reviewer reads carries the *grader's* verdict rather than the container's exit
    code — a row whose container exited 0 can still fail an assertion over its
    evidence, and only the assertion knows that. :class:`GradeLedger` is what does
    the re-emitting; nothing else should call this with a `graded` argument,
    because the ledger is the only thing that has heard from pytest.

    The re-emission also re-digests the bundle, which matters beyond the verdict:
    a grading test may write into the bundle it is grading (the negative-auth rows
    file `negative-auth-observed.json` there), and a digest taken before the
    grader ran would then report MISMATCH — the verdict this rig reserves for
    "something rewrote the evidence" — against a bundle nobody tampered with.

    Failures are kept apart on purpose. A record that does not validate is a
    **driver bug** and raises: writing an unsanitized record is the one outcome
    C10 exists to prevent. The two **environment** conditions never fail a row
    that really ran — no image evidence means no record at all (every other column
    would be uncheckable), and an unreadable file means a record without its
    digest; both write their reason beside the bundle they are about.
    """
    record = build_record(row, result, posture, image, outcome, reason,
                          archive_url, graded)
    if not record["image"]:
        note = (f"no record written: `docker image inspect {image}` did not "
                f"answer with a sha256 id, so the record's image column could "
                f"not be filled and every other column would be uncheckable\n")
        (result.out_dir / "record-skipped.txt").write_text(note, encoding="utf-8")
        return None
    if "bundle" not in record:
        # The record still stands — it just cannot say which bytes it came from,
        # so the reason is named where the bytes are rather than inferred later
        # from `verify_record.py` reporting an absent block.
        locked = unreadable_bundle_files(result.out_dir)
        (result.out_dir / "record-unhashed.txt").write_text(
            "this bundle carries no digest in its record: it holds files this "
            "uid cannot read (the container writes as 10001, and the product's "
            "own result files are 0600)\n"
            + "".join(f"  {path}\n" for path in locked), encoding="utf-8")
    errors = validate_record(record)
    if errors:
        raise ValueError(
            f"the run record for {row.row_id} does not validate against "
            f"{RECORD_SCHEMA.name} — it is NOT written:\n  " + "\n  ".join(errors))
    directory = records_dir or RECORDS
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['started_at']}-{row.row_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# C10b — the grader's verdict, and the ordering problem it has to survive.
#
# A promotion is a claim that a capability was *demonstrated*, and demonstrating
# it takes two things that happen at two different times:
#
# 1. the container runs and exits — the driver knows this the moment `run_row`
#    returns, and it is the only verdict that existed here before;
# 2. the assertions over the evidence that run left behind pass — pytest knows
#    this later, one test at a time, after the record has already been written.
#
# The record cannot simply be written later instead: a row's bundle is the only
# durable evidence a crashed or interrupted session leaves, and a driver that
# wrote nothing until the grader finished would lose the record for exactly the
# runs that most need one. So the record is written **twice**, and the ledger
# below owns both writes:
#
# * `open()` writes it as soon as the row finishes, with the container's verdict
#   and — by construction, since `promotions()` defaults to `UNGRADED` —
#   **no promotions**, plus a `reason` that says so in as many words;
# * `note_report()` collects each pytest report and attributes it to the rows the
#   test actually read (via `item.funcargs`, so attribution is the fixture value
#   the test was handed rather than a guess from its id or from run order);
# * `resolve_all()` re-emits every row at session end with both verdicts, which is
#   where a promotion can first appear at all.
#
# The alternative designs were rejected for concrete reasons rather than taste.
# Keeping the write at row time and *applying* the grader's verdict to
# `capabilities.yaml` afterwards would break C10's rule that the harness never
# writes that file. Deriving the verdict from the record itself is circular. And
# inferring "the row passed" from pytest's exit code cannot work on a matrix: one
# red row would suppress every other row's promotions, and a lane whose failures
# are all skips would suppress none.
# --------------------------------------------------------------------------- #


@dataclass
class GradedRow:
    """One row's two verdicts, plus the record they are written into."""

    row: Row
    result: RowResult
    posture: dict | None = None
    image: str = DEFAULT_IMAGE
    records_dir: Path | None = None
    #: `nodeid`s, so a withheld promotion names the assertion that withheld it.
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    resolved: bool = False
    record_path: Path | None = None
    #: What was really written on the last emit — never what the artifacts could
    #: have supported, which is `candidate_promotions`.
    emitted: list[dict] = field(default_factory=list)

    @property
    def graded(self) -> str:
        """`fail` beats `pass`; nothing at all is :data:`UNGRADED`, not a pass."""
        if self.failed:
            return GRADED_FAIL
        if self.passed:
            return GRADED_PASS
        return UNGRADED

    def verdict(self) -> tuple[str, str | None]:
        """The row's outcome and reason with BOTH verdicts folded in.

        A grader failure over a container that exited 0 is recorded as `fail` —
        the row did not pass, and the record is the tracked artifact a reviewer
        reads instead of the pytest scrollback that scrolled away.
        """
        outcome, reason = row_outcome(self.result)
        if self.failed:
            named = ", ".join(_test_name(nodeid) for nodeid in self.failed[:3])
            if len(self.failed) > 3:
                named += f" (+{len(self.failed) - 3} more)"
            reason = "; ".join(part for part in (
                reason, f"the grader failed this row: {named}") if part)
            if outcome == "pass":
                outcome = "fail"
        elif not self.passed:
            reason = "; ".join(part for part in (reason, UNGRADED_REASON) if part)
        return outcome, reason


def _test_name(nodeid: str) -> str:
    """The readable tail of a nodeid — the part a failure message is about."""
    return nodeid.rsplit("::", 1)[-1] or nodeid


def _row_results(value) -> list[RowResult]:
    """Every `RowResult` a fixture value carries, one container level deep.

    The host-lane fixtures hand a test `(result, census, entry)`; the chain
    fixtures hand it a bare `RowResult`. Both shapes are read here rather than
    each module teaching the ledger about its own tuple.
    """
    candidates = [value]
    if isinstance(value, (tuple, list)):
        candidates += list(value)
    return [candidate for candidate in candidates
            if isinstance(candidate, RowResult)]


class GradeLedger:
    """Every host row of one invocation, and the two verdicts each one carries."""

    def __init__(self) -> None:
        self.rows: dict[str, GradedRow] = {}
        #: Insertion order, so the summary reads in the order the rows ran.
        self.order: list[str] = []
        #: Rows whose final record could not be written. A driver bug, reported
        #: rather than swallowed — see `resolve_all`.
        self.errors: list[tuple[str, str]] = []

    def reset(self) -> None:
        """Forget the invocation. For a test that drives this ledger deliberately."""
        self.__init__()

    # -- the two verdicts ---------------------------------------------------- #
    def open(self, row: Row, result: RowResult, posture: dict | None = None,
             image: str = DEFAULT_IMAGE,
             records_dir: Path | None = None) -> Path | None:
        """Register a finished row and write its PROVISIONAL record.

        Re-opening the same row id **resolves the previous entry first, then
        replaces it**: a re-run of a row is a new row result, and grading it
        against the earlier run's reports would be the same class of error this
        whole block exists to remove — while silently dropping the earlier entry
        would leave its record provisional for ever.
        """
        if row.row_id in self.rows:
            self.resolve(row.row_id)
        entry = GradedRow(row=row, result=result, posture=posture, image=image,
                          records_dir=records_dir)
        self.rows[row.row_id] = entry
        if row.row_id not in self.order:
            self.order.append(row.row_id)
        return self._emit(entry, provisional=True)

    def note(self, row_id: str, nodeid: str, passed: bool) -> None:
        entry = self.rows.get(row_id)
        if entry is None:
            return
        bucket = entry.passed if passed else entry.failed
        if nodeid not in bucket:
            bucket.append(nodeid)

    def note_report(self, item, report) -> None:
        """Attribute one pytest report to every row the test actually read.

        Attribution is by fixture VALUE (`item.funcargs`), not by the test's id or
        by whichever row ran most recently: module-scoped parametrized fixtures
        outlive their own tests — the last `discovery` instance is still alive
        while the credentialed rows run — so "the row that is currently open" would
        quietly charge one row's failures to another.

        Only a `call`-phase pass counts as a pass, because a skipped test still
        reports its setup as passed and a row every test skipped has been graded by
        nothing. A failure counts from any phase: a teardown error is a real one.
        """
        if not self.rows:
            return
        failed = bool(getattr(report, "failed", False))
        passed = bool(getattr(report, "passed", False)) and report.when == "call"
        if not failed and not passed:
            return                      # skips, and the setup/teardown passes
        seen: set[str] = set()
        for value in (getattr(item, "funcargs", None) or {}).values():
            for result in _row_results(value):
                row_id = result.row.row_id
                if row_id in seen or row_id not in self.rows:
                    continue
                seen.add(row_id)
                # The ledger keys on the row id, so a stale entry from an earlier
                # invocation of the same row must not collect this report.
                if self.rows[row_id].result is result:
                    self.note(row_id, report.nodeid, passed)

    # -- resolution ----------------------------------------------------------- #
    def resolve(self, row_id: str) -> Path | None:
        entry = self.rows.get(row_id)
        if entry is None or entry.resolved:
            return entry.record_path if entry else None
        entry.resolved = True
        return self._emit(entry, provisional=False)

    def resolve_all(self) -> list[Path]:
        """Re-emit every unresolved row with both verdicts. Never raises.

        A record that does not validate is a driver bug and `emit_record` raises
        on it — but raising from here would abandon every row after it, so the
        failure is collected, reported by `summary()` and turned into a non-zero
        exit by the caller. One unwritable record must not cost the others theirs.
        """
        written: list[Path] = []
        for row_id in list(self.order):
            entry = self.rows.get(row_id)
            if entry is None or entry.resolved:
                continue
            entry.resolved = True
            try:
                path = self._emit(entry, provisional=False)
            except Exception as exc:                     # noqa: BLE001 — reported
                self.errors.append((row_id, f"{type(exc).__name__}: {exc}"))
                continue
            if path:
                written.append(path)
        return written

    def _emit(self, entry: GradedRow, provisional: bool) -> Path | None:
        if provisional:
            outcome, reason = row_outcome(entry.result)
            reason = "; ".join(part for part in (reason, PROVISIONAL_REASON) if part)
            graded = UNGRADED
        else:
            outcome, reason = entry.verdict()
            graded = entry.graded
        entry.record_path = emit_record(
            entry.row, entry.result, entry.posture, entry.image,
            outcome=outcome, reason=reason, records_dir=entry.records_dir,
            graded=graded)
        entry.emitted = (promotions(entry.row, entry.result, outcome, graded)
                         if entry.record_path else [])
        return entry.record_path

    # -- the printed surface -------------------------------------------------- #
    def summary(self) -> list[str]:
        """What this invocation promotes, what it withheld, and why.

        PRINTED, never applied — promoting a capability changes which code path
        ships to users, so the edit stays a human commit carrying the record and
        the `capabilities.yaml` change in one diff.
        """
        if not self.rows:
            return []
        lines = ["capability promotions (printed, NEVER applied — commit the edit "
                 "beside the record it cites):"]
        promoted = [(entry, edit) for row_id in self.order
                    for entry in [self.rows[row_id]] for edit in entry.emitted]
        for entry, edit in promoted:
            lines.append(f"  {edit['host']:<12} {edit['field']:<16} "
                         f"{edit['from']} → {edit['to']}   "
                         f"evidence: {edit['evidence']}")
        if not promoted:
            lines.append("  none — no row both exited 0 and passed its grader")
        withheld = []
        for row_id in self.order:
            entry = self.rows[row_id]
            if entry.emitted:
                continue
            candidates = candidate_promotions(entry.row, entry.result)
            if not candidates:
                continue
            # The CONTAINER's own verdict, not the combined one `verdict()`
            # returns: the whole point of the line is that the two can differ, and
            # printing the combined verdict under the label "container" would hide
            # exactly the case this block exists to report.
            container, _reason = row_outcome(entry.result)
            _outcome, reason = entry.verdict()
            withheld.append((row_id, candidates, container, entry.graded, reason))
        if withheld:
            lines.append("promotions WITHHELD — these rows produced the artifacts "
                         "but did not pass, so they promote nothing:")
            for row_id, candidates, container, graded, reason in withheld:
                fields = ", ".join(edit["field"] for edit in candidates)
                lines.append(f"  {row_id}: {fields}")
                lines.append(f"    container {container}, grader {graded} — "
                             f"{reason}")
        for row_id, error in self.errors:
            lines.append(f"  RECORD NOT WRITTEN for {row_id}: {error}")
        return lines


#: The invocation's one ledger. Module state for the same reason `MATRIX` is: a
#: single pytest invocation collects several row-producing modules, and the
#: grader's reports arrive through a hook that has no other way to reach them.
GRADES = GradeLedger()


def _attempt_entry(n: int, result: RowResult, signature: dict | None,
                   bundle: str, graded: bool) -> dict:
    """One attempt's line in the manifest: what it was, and whether it counted."""
    return {
        "attempt": n,
        # Relative to the row's bundle root, so the manifest is portable
        # between checkouts exactly like the record's `bundle.root` is.
        "bundle": bundle,
        "graded": graded,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "wall_clock_s": result.wall_clock_s,
        "spend_usd": result.usage.get("delta_usd"),
        "dispatcher_receipts": len(dispatcher_receipts(result)),
        "artifact_files": len(artifact_files(result)),
        # `None` on an attempt that did NOT match the shape, which is the whole
        # point of writing it for the graded attempt too: a retry that landed on
        # the same empty turn twice is a broken row, not a flaky one, and the
        # manifest is where that reads at a glance.
        "never_started_signature": signature,
    }


def _write_retry_manifest(out_dir: Path, row: Row, attempts: list[dict],
                          state: str, graded_attempt: int | None,
                          declined: str | None = None) -> Path:
    """The bundle's own statement that a retry happened, or did not.

    Written TWICE on a retried row for the same reason the record is: once
    before the second container starts, so a session that dies mid-retry still
    leaves a bundle that explains its own `attempt-1/` directory, and once after
    it, with both attempts and the verdict. A rig that retried silently would be
    lying about its own reliability, which is the one thing this suite exists not
    to do.
    """
    path = out_dir / RETRY_MANIFEST_NAME
    payload = {
        "row_id": row.row_id,
        "host": row.host,
        "shape": "never-started",
        "detector": "harness.never_started_signature",
        "state": state,
        "max_attempts": 2,
        "graded_attempt": graded_attempt,
        "attempts": attempts,
    }
    if declined:
        payload["declined"] = declined
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _stamp_attempt(out_dir: Path, attempt: int, graded: bool) -> None:
    """Say inside one attempt's own `row-result.json` which attempt it is.

    The manifest ties the attempts together; this is so the file a reader opens
    FIRST — the row's own driver record — cannot be read as the whole row when it
    is one discarded half of it. Best-effort: a bundle whose row-result went
    missing is a worse problem than an unstamped one, and neither is a reason to
    fail a row that ran.
    """
    path = out_dir / "row-result.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    record["attempt"] = attempt
    record["graded"] = graded
    record["retry_shape"] = "never-started"
    try:
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    except OSError:
        return


def _shelve_attempt(out_dir: Path, n: int) -> Path:
    """Move a finished attempt's bundle to `<bundle>/attempt-<n>/`; return it.

    Two renames rather than a copy, so no evidence is rewritten in the process —
    the bytes a record digests have to be the bytes the container wrote. The
    intermediate sibling exists because a directory cannot be renamed into
    itself.
    """
    parent = out_dir.parent
    shelf = parent / f".{out_dir.name}.{ATTEMPT_DIR_FMT.format(n=n)}"
    if shelf.exists():
        shutil.rmtree(shelf, ignore_errors=True)
    out_dir.rename(shelf)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)
    landed = out_dir / ATTEMPT_DIR_FMT.format(n=n)
    shelf.rename(landed)
    return landed


def _retry_never_started(row: Row, first: RowResult, signature: dict,
                         posture: dict, image: str, out_dir: Path,
                         runs_root: Path, run_root: Path) -> RowResult:
    """Re-run a row that never started, ONCE, and record both attempts.

    Returns the attempt to grade: the second one when it ran, the first one when
    the retry was declined — in which case the row grades exactly as it would
    have with no retry mechanism at all, and the manifest says why.
    """
    reason = never_started_reason(signature)
    first.retry_signature = signature

    declined: str | None = None
    if not retry_never_started_enabled():
        declined = ("MLLM_E2E_RETRY_NEVER_STARTED is off, so the shape was "
                    "detected and no second attempt was run")
    else:
        # The same admission rule `MATRIX.claim` applies, read rather than
        # claimed: `claim` files a refused row under `never_started`, and this
        # row demonstrably DID start once — a census that said otherwise would
        # be describing a bundle that is sitting right there.
        remaining = MATRIX.remaining()
        if remaining is not None and remaining <= 0:
            declined = (f"the lane's wall-clock budget is spent "
                        f"({MATRIX.timeout_s:.0f}s), so no further container "
                        f"was started")
    if declined:
        MATRIX.record(first)
        _write_retry_manifest(
            out_dir, row,
            [_attempt_entry(1, first, signature, ".", graded=True)],
            state="not-retried", graded_attempt=1, declined=declined)
        return first

    # Filed BEFORE the second container starts, for the same reason every other
    # row's census entry is: the summary has to be on disk before the row that
    # overruns the cap. The `#attempt-1` suffix keeps it a row of its own rather
    # than a second entry under an id that also names the graded attempt.
    MATRIX.note(f"{row.row_id}#{ATTEMPT_DIR_FMT.format(n=1)}", RETRY_OUTCOME,
                first.wall_clock_s, reason)

    shelved = _shelve_attempt(out_dir, 1)
    first.out_dir = shelved
    _stamp_attempt(shelved, attempt=1, graded=False)
    first_entry = _attempt_entry(1, first, signature,
                                 ATTEMPT_DIR_FMT.format(n=1), graded=False)
    _write_retry_manifest(out_dir, row, [first_entry], state="retrying",
                          graded_attempt=None)

    second = _run_attempt(row, posture, image, out_dir, runs_root, run_root)
    second.attempts = [first]
    MATRIX.record(second)
    _stamp_attempt(out_dir, attempt=2, graded=True)
    # ONE SHOT. The second attempt's signature is computed and recorded, never
    # acted on: a row that never started twice running is a finding, and the
    # answer to it is a red row a human reads, not a third container.
    _write_retry_manifest(
        out_dir, row,
        [first_entry,
         _attempt_entry(2, second, never_started_signature(second), ".",
                        graded=True)],
        state="retried", graded_attempt=2)
    return second


def _run_attempt(row: Row, posture: dict, image: str, out_dir: Path,
                 runs_root: Path, run_root: Path) -> RowResult:
    """One `docker run`, from an empty bundle directory to its evidence.

    Split out of :func:`run_row` so the never-started retry runs the SAME code
    path a second time into the same directory rather than a second, subtly
    different one — a retry that ran a different command would not be a retry.
    It deliberately files nothing: the lane census and the record belong to the
    ROW, and only `run_row` knows how many attempts that row took.
    """
    # 0777 because /work/out is a host directory owned by the developer while the
    # container writes as 10001: the artifacts come back owned by 10001, readable
    # and prunable because their parent is the developer's.
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)
    # Retention lives here, not in the Makefile, so it holds for every entry
    # point — a pytest row, `make e2e-container`, a hand-driven `run_row`. The
    # bundle being written is passed as `protect` because it already exists by
    # this line and would otherwise be its own sweep's oldest candidate whenever
    # the directory was empty before it.
    prune_runs(runs_root, protect=run_root)

    argv = build_run_argv(row, out_dir, posture, image)
    (out_dir / "run-command.txt").write_text(
        " ".join(shlex.quote(a) for a in argv) + "\n", encoding="utf-8")

    # The host-side half of the posture check has to run while the container is
    # alive: `--rm` takes the record away with it. Started before the run and
    # joined after, so the poll is already going when the container appears.
    inspected: dict = {}
    stop = threading.Event()
    watcher = threading.Thread(
        target=_capture_inspect,
        args=(container_name(row), out_dir, stop, inspected), daemon=True)
    watcher.start()

    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=row.row_timeout_s)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = None
        out = _as_text(exc.stdout)
        err = _as_text(exc.stderr)
        # `docker stop` first, with grace, so the entrypoint's `finally` export
        # still runs and a capped row still leaves readable evidence — then
        # `docker kill` whatever survived the grace period.
        _stop_container(_container_of(argv))
    finally:
        stop.set()
        watcher.join(timeout=5)

    (out_dir / "posture-host.json").write_text(json.dumps({
        "captured": bool(inspected.get("inspect")),
        "reason": None if inspected.get("inspect") else (
            "`docker inspect` never caught this container — it exited before the "
            "poll reached it, and --rm took the record with it"),
        "mismatches": (host_posture_mismatches(inspected["inspect"], row, out_dir,
                                               posture)
                       if inspected.get("inspect") else []),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Persist the container's own streams beside the rest of the evidence. They
    # are the only record of a failure that happens *before* the stub or the
    # wrapper get a chance to log anything — a seeding error, a refused mount, a
    # CLI that dies on startup — and without them such a row reports "zero
    # requests reached the stub" with no way to tell why from the bundle alone.
    (out_dir / "container-stdout.txt").write_text(out or "", encoding="utf-8")
    (out_dir / "container-stderr.txt").write_text(err or "", encoding="utf-8")

    result = RowResult(row=row, exit_code=code,
                       wall_clock_s=round(time.time() - started, 2),
                       out_dir=out_dir, timed_out=timed_out, stdout=out, stderr=err)
    if timed_out:
        result.cap_reason = classify_cap_hit(result)

    # The row's own wall clock, beside its evidence rather than only in the
    # driver's memory: it is what a committed `chain_s`/`row_timeout_s` in
    # redirect.yaml is derived from (C1), and a run whose numbers live only in
    # pytest's stdout cannot be turned into a cap without re-running it.
    (out_dir / "row-result.json").write_text(json.dumps({
        "row_id": row.row_id,
        "lane": row.lane,
        "cli": row.cli,
        "host": row.host,
        "exit_code": code,
        "wall_clock_s": result.wall_clock_s,
        "row_timeout_s": row.row_timeout_s,
        "timed_out": timed_out,
        "cap_reason": result.cap_reason,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return result


def run_row(row: Row, posture: dict | None = None, image: str = DEFAULT_IMAGE,
            runs_root: Path | None = None, utc: str | None = None) -> RowResult:
    """Run one row to completion (or to its cap) and return its evidence.

    Raises :class:`MatrixExpired` — before any container exists — when the lane's
    own `MLLM_E2E_MATRIX_TIMEOUT_S` cap is already spent (C1).

    **One shape, and only one, is run twice.** A credentialed host row whose turn
    never started — a clean `end_turn` with an empty `result`, no dispatcher
    receipt, no exported artifact and a cent or less of key spend — observed
    nothing about the product, so it is discarded and re-run once. Every other
    failure is returned exactly as it happened. Both attempts land in the
    bundle: the discarded one under `attempt-1/`, both of them in `retry.json`
    beside it, and both as rows of the tracked record. See the block above
    :func:`never_started_signature` for why that shape and no other.
    """
    posture = posture or load_posture()
    utc = utc or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runs_root = runs_root or RUNS
    run_root = runs_root / utc
    # The whole-lane cap, claimed before the bundle directory is even made: a
    # refused row must leave nothing on disk that reads like a run.
    MATRIX.claim(row.row_id, run_root)
    out_dir = run_root / row.row_id

    result = _run_attempt(row, posture, image, out_dir, runs_root, run_root)

    signature = never_started_signature(result)
    if signature is not None:
        # Files its own census entries — one per attempt — because only this
        # branch knows there was more than one.
        result = _retry_never_started(row, result, signature, posture, image,
                                      out_dir, runs_root, run_root)
    else:
        # The lane census, updated per row rather than at the end: the summary
        # has to be on disk *before* the row that overruns the cap, or the one
        # invocation that needs it is the one that never wrote it.
        MATRIX.record(result)

    # C10, and only for the lane that can promote anything: a green HOST row is
    # the evidence `capabilities.yaml` promotions are made from, so that lane's
    # provenance has to outlive the gitignored bundle it was derived from. The
    # provider lane deliberately writes none — its evidence is a golden diff in
    # the same commit, and a tracked file per CLI per run would be noise around
    # the records that back a claim.
    #
    # This registers the row and writes its record PROVISIONALLY: the container's
    # verdict is in, the grader's is not, and a promotion needs both (C10b). The
    # ledger re-emits it with both once pytest has finished asserting over this
    # bundle. Emitting the final record straight from here is precisely the bug —
    # a container that exits 0 having produced output that then fails its own
    # assertions would propose capability edits nothing ever demonstrated.
    #
    # A retried row registers its GRADED attempt, which carries the discarded one
    # in `RowResult.attempts`, so the record it writes holds both.
    if row.lane == "host":
        result.record_path = GRADES.open(row, result, posture, image)

    return result


def _as_text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _container_of(argv: list[str]) -> str:
    """The --name we gave the run, so a cap hit can stop precisely that container."""
    if "--name" in argv:
        return argv[argv.index("--name") + 1]
    return ""


def _stop_container(name: str) -> None:
    """Stop a capped row's container with grace, then make sure it is gone (C1).

    `docker stop` is the graceful half: SIGTERM, then up to ten seconds. The
    entrypoint installs a SIGTERM handler for it — as PID 1 it gets no default
    disposition, so an uninstalled handler means the signal is discarded — and
    unwinds into its `finally` export, which is what leaves a capped row with
    phase dirs and a flushed stub log to triage.

    The kill is the guarantee. Docker escalates to SIGKILL itself when the grace
    period expires, but only for a stop call that reached the daemon: a `docker
    stop` that errored or hung would otherwise leave the row's container running
    for the rest of the session — on the host lane, with a live credential and a
    network route. So the state is read back and anything still running is killed
    outright. `--rm` takes the record away on exit, so an inspect that finds no
    such object is the success case, not a failure.
    """
    if not name:
        return
    try:
        subprocess.run(["docker", "stop", "--time", "10", name],
                       capture_output=True, timeout=40)
    except subprocess.TimeoutExpired:
        pass
    alive = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                           capture_output=True, text=True)
    if alive.returncode == 0 and alive.stdout.strip() == "true":
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
