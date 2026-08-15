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

import functools
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
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
RECORD_SCHEMA = RECORDS / "record.schema.json"
#: Written by `verify_versions.py` in the final build layer: what the image really
#: installed, as opposed to what `cli-versions.lock` asked for. It is the only
#: honest source for a row's `cli_version` under `MLLM_PIN_MODE=latest`.
IMAGE_VERSION_REPORT = "/opt/mllm/lock/cli-versions.installed.json"
#: `hosts/capabilities.yaml`, whose `provenance` block is what a green host row
#: promotes. Read here, never written: C10 prints promotions and a human commits
#: them.
CAPABILITIES = HERE.parents[1] / "hosts" / "capabilities.yaml"
DEFAULT_IMAGE = os.environ.get("MLLM_E2E_IMAGE", "mllm-e2e:local")


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
    spec: str | None = None
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
        """Which workspace files moved, by SHA — never what was written into them."""
        before = _read_json(self.out_dir / "workspace-census-before.json")
        after = _read_json(self.out_dir / "workspace-census-after.json")
        delta = {}
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                delta[name] = (before.get(name), after.get(name))
        return delta


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
    if row.spec:
        argv += ["--spec", row.spec]
    if row.runner:
        argv += ["--runner", row.runner]
    if row.helpers:
        argv += ["--helpers", ",".join(row.helpers)]
    if row.scenario:
        argv += ["--scenario", row.scenario]
    if row.mock_runner:
        argv += ["--mock-runner", row.mock_runner]
    if row.dry_run_host:
        argv += ["--dry-run-host"]
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
# C1 — the WHOLE-LANE cap, enforced here because here is where a run is started.
#
# `MLLM_E2E_MATRIX_TIMEOUT_S` is exported by the `Makefile` and read *here*. The
# per-row cap is the `subprocess.run(timeout=…)` in `run_row`, and on its own it
# bounds nothing a developer waits on: eleven chain rows at 600 s, three
# credentialed host rows at ~1400 s and a per-CLI surface probe behind them is
# most of an afternoon, and every one of those rows is individually "within its
# cap" the whole time. An export nobody reads is worse than no cap at all — it
# reads as a bound in the Makefile, in the README and in the plan while the lane
# runs open-ended.
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
        failed = [entry for entry in self.rows if entry["outcome"] != "pass"]
        return {
            "cap_s": self.timeout_s,
            # The lane total, beside each row's own wall clock: it is the number
            # the README quotes and the one a later cap is derived from.
            "elapsed_s": round(self.elapsed(), 2),
            "expired": bool(self.refused),
            "ran": ran,
            "failed": failed,
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
WRITE_OUTCOME_SEVERITY = [
    ("partial-application", "partial"),
    ("schema-rejection", "schema-rejected"),
    ("clean-failure", "clean-failure"),
    ("applied", "applied"),
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
    """Every dispatcher invocation this row drives, as one string.

    The provider lane types its commands into the run argv (`row.command`); the
    host lane does **not** — the entrypoint reads them out of `host_matrix.yaml`
    by row id, so a driver that looked only at the Row would conclude that every
    host row typed nothing at all, and would then never record a `--detach` or
    `--exec-subagents` row as evidence for the capability it exercised.
    """
    typed = " ".join([row.command, *row.commands]).strip()
    if typed:
        return typed
    for entry in (load_host_matrix().get(row.host or "") or {}).get("rows") or []:
        if entry.get("id") == row.row_id:
            return str(entry.get("command") or "")
    return ""


def row_outcome(result: RowResult) -> tuple[str, str | None]:
    """The driver's own verdict on a row, with the reason it is not a bare bool.

    This is the container's verdict, not the grader's: a row whose container
    exited 0 can still fail an assertion over its evidence, which is why
    :func:`emit_record` takes an explicit outcome and this is only the default.
    """
    if result.timed_out:
        return "cap-hit", result.cap_reason or classify_cap_hit(result)
    if result.exit_code == 0:
        return "pass", None
    if result.exit_code is None:
        return "not-run", "the container produced no exit status"
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
    if (result.out_dir / "discovery.json").is_file():
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
    recorded = write_outcomes(result)
    for product_word, schema_word in WRITE_OUTCOME_SEVERITY:
        if product_word in recorded:
            entry["write_outcome"] = schema_word
            break
    unspellable = sorted(recorded.difference(
        word for word, _ in WRITE_OUTCOME_SEVERITY))
    if unspellable:
        reason = "; ".join(part for part in
                           (reason, f"write batches also recorded "
                                    f"{', '.join(unspellable)}") if part)
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
        pin = ((load_host_matrix().get(row.host) or {}).get("model") or {}).get("slug")
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
    fields: set[str] = set()
    if result.receipts:
        fields |= {"shim", "install_paths", "arguments", "reload"}
    if len(result.receipts) >= 2:
        fields.add("multi_turn")
    typed = declared_command(row)
    # Both halves, because either alone is the wrong claim: `--detach` typed with
    # no terminal worker record is a launcher that returned over a worker nobody
    # watched, and a terminal record on a row that never asked to detach says
    # nothing about the host's detach mechanism.
    if "--detach" in typed and any(
            (state or {}).get("state") == "done"
            for state in result.run_states.values()):
        fields.add("detach")
    if "--exec-subagents" in typed:
        fields.add("subagents")
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
        "rows": [row_record(row, result, outcome, reason, image)],
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


def run_row(row: Row, posture: dict | None = None, image: str = DEFAULT_IMAGE,
            runs_root: Path | None = None, utc: str | None = None) -> RowResult:
    """Run one row to completion (or to its cap) and return its evidence.

    Raises :class:`MatrixExpired` — before any container exists — when the lane's
    own `MLLM_E2E_MATRIX_TIMEOUT_S` cap is already spent (C1).
    """
    posture = posture or load_posture()
    utc = utc or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = (runs_root or RUNS) / utc
    # The whole-lane cap, claimed before the bundle directory is even made: a
    # refused row must leave nothing on disk that reads like a run.
    MATRIX.claim(row.row_id, run_root)
    out_dir = run_root / row.row_id
    # 0777 because /work/out is a host directory owned by the developer while the
    # container writes as 10001: the artifacts come back owned by 10001, readable
    # and prunable because their parent is the developer's.
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)

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

    # The lane census, updated per row rather than at the end: the summary has to
    # be on disk *before* the row that overruns the cap, or the one invocation
    # that needs it is the one that never wrote it.
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
