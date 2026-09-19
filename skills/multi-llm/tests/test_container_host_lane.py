"""Host lane: does the real host still discover, load, run and survive our shim?

Three tiers, and **only the first needs nothing**:

* ``--creds none`` — stage the shim, cross the ``reload`` boundary, assert the
  command is discoverable, stop. No workflow row, no model call, no credential,
  ``--network none``. It buys the single most likely real-world break — the shim
  landing where the host no longer scans, which this design calls *the failure
  that reaches users first* — at zero cost. **These tests carry the `container`
  marker but NOT `live`**, so a keyless contributor and CI can run them.
* OpenRouter (default) and the subscription fallback — the credentialed rows:
  three per host (ask, review-plan, write) plus the two authentication negatives.
  Those carry `live` as well, and without a key each one skips **by name into a
  census** — after which the lane itself reports `not-run` and exits non-zero,
  because a host lane that was explicitly invoked and executed zero rows covered
  nothing. `--allow-empty-host-lane` is the only way to a green exit out of that
  state, and it suppresses the failure, never the census.

**A green ``install-host --check`` is not discovery.** It reads our own bytes at a
path we chose. Discovery means the *host* enumerating the command by name, which
is why the tier asks each host rather than inspecting the filesystem.

Phase 1 measured which hosts can be asked that without a model, and the answer is
one of three: ``opencode`` (via `opencode debug config`). ``codex`` and
``claude-code`` have no such surface — both were probed, `doctor` included — so
this tier reports them ``unavailable-without-credential`` rather than passing
them, and says so in the census. That distinction is the whole point: "everything
skipped" and "everything passed" are the two outcomes a lane must never conflate.

Those two hosts are not left unevidenced, though — they are evidenced one tier
up. Their credentialed ``ask`` rows already emit a ``util instructions <mode>``
receipt, which is step 3 of the shim and is obtainable only by a host that found
and read it, so ``TestDiscoveryOnTheHostsWithNoFreeSurface`` records
``discovery: listed-via-prompted-turn`` from a turn this lane already pays for.
It adds no row and costs nothing. The free tier's own verdict is untouched: the
two tiers cannot see each other's data (the module-scoped ``discovery`` fixture's
tests all run before the first credentialed row exists), and the answer to that
is to grade the credentialed evidence where it lives rather than to let the free
tier claim something it cannot observe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, requires_docker()]

HOST_MATRIX = harness.HERE / "host_matrix.yaml"

#: The `MULTI_LLM_RESULT.state` values that mean an invocation *finished*, as
#: opposed to the non-terminal ones a poll legitimately reports mid-run
#: (`nothing-to-do` from a `status` that found the worker still running,
#: `needs-human`, `error`). `partial` is terminal and is emitted at **exit 0** —
#: D6 row 3's partial aggregate — which is precisely why membership here is not on
#: its own evidence that a row succeeded, and why `expect.exit` is graded too.
TERMINAL_STATES = ("done", "complete", "partial")


def _hosts():
    data = yaml.safe_load(HOST_MATRIX.read_text(encoding="utf-8"))
    return sorted(data["hosts"].items())


HOSTS = _hosts()
IDS = [h for h, _ in HOSTS]


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module", params=HOSTS, ids=IDS)
def discovery(request, posture, utc):
    """Run one host's `--creds none` row and return its evidence."""
    host, _entry = request.param
    row = harness.Row(
        row_id=f"discovery-{host}",
        lane="host",
        host=host,
        creds="none",
        row_timeout_s=300,
    )
    result = harness.run_row(row, posture=posture, utc=utc)
    census_path = result.out_dir / "discovery.json"
    census = (json.loads(census_path.read_text(encoding="utf-8"))
              if census_path.is_file() else None)
    return result, census, dict(_entry)


class TestTheFreeTierRunsAtAll:
    def test_the_row_completed(self, discovery):
        result, _census, _entry = discovery
        assert not result.timed_out, (
            f"{result.row.host}: {result.cap_reason} after {result.wall_clock_s}s"
        )
        assert result.exit_code == 0, (
            f"{result.row.host}: exited {result.exit_code}\n"
            f"--- container stderr (tail) ---\n{result.stderr[-2000:]}"
        )

    def test_it_spent_nothing(self, discovery):
        """No model call, and therefore no cost — asserted, not assumed.

        The tier's whole claim is that it is free. A stub request would mean a
        workflow row ran, which this tier is defined not to do.
        """
        result, _census, _entry = discovery
        calls = [r for r in result.stub_log if r.get("surface") != "health"]
        assert not calls, (
            f"{result.row.host}: the discovery tier made {len(calls)} model "
            f"request(s); it is defined to make none"
        )

    def test_it_had_no_route_out(self, discovery):
        """Free is a STRUCTURAL claim on this tier, not a stub-log observation.

        An empty stub log proves only that nothing reached *our* endpoint. On the
        open host-lane bridge a third-party binary can satisfy that while talking
        to a real one, which is the whole failure mode this tier's `--network none`
        removes. Both halves of the posture check are read: docker's own view of
        the run (the flag was really applied) and the container's routing table
        (there was really no way out).
        """
        result, _census, _entry = discovery
        inside = json.loads(
            (result.out_dir / "posture-inside.json").read_text(encoding="utf-8"))
        assert inside["network"]["declared"] == "none", (
            f"{result.row.host}: the discovery row declared network "
            f"{inside['network']['declared']!r} — `--creds none` runs on the "
            f"provider lane's terms, no credential and no route"
        )
        assert not inside["network"]["default_route"], (
            f"{result.row.host}: the discovery row had a DEFAULT ROUTE via "
            f"{inside['network']['routed_via']}"
        )
        host_side = json.loads(
            (result.out_dir / "posture-host.json").read_text(encoding="utf-8"))
        # `captured: false` is a lost `docker inspect` race, which the harness is
        # explicit about not turning into a failed row; the routing table above is
        # the fact, and this is the corroboration when it is there.
        if host_side["captured"]:
            assert not host_side["mismatches"], (
                f"{result.row.host}: docker's own record of this run disagrees "
                f"with posture.yaml: {host_side['mismatches']}"
            )


class TestTheStagedCommandIsDiscoverable:
    def test_a_census_was_written(self, discovery):
        result, census, _entry = discovery
        assert census is not None, (
            f"{result.row.host}: no discovery.json — the tier produced no verdict, "
            f"which is worse than a negative one"
        )

    def test_the_host_enumerates_our_command(self, discovery):
        """The assertion, where the host offers a credential-free way to ask.

        The skip below stays a skip, and deliberately: this tier runs with no
        credential and `--network none`, so on a host with no listing surface
        there is genuinely nothing here to read. Turning it into a pass would be
        inventing a verdict, which is the one thing this module exists to stop.

        The evidence those two hosts *do* produce is a credentialed one and is
        graded where it exists — `TestDiscoveryOnTheHostsWithNoFreeSurface`, on
        the same `ask` rows this lane already runs, at no extra cost.
        """
        result, census, entry = discovery
        form = (entry.get("list_step") or {}).get("form")
        if form != "debug-config":
            pytest.skip(
                f"{result.row.host}: no credential-free listing surface exists "
                f"(measured — `doctor` included), so discovery here needs a "
                f"prompted turn and a credential. Reported as "
                f"'{census['discovery']}' in the census rather than as a pass. "
                f"The prompted-turn evidence is graded by "
                f"TestDiscoveryOnTheHostsWithNoFreeSurface in the credentialed "
                f"tier, which writes 'listed-via-prompted-turn' beside that row's "
                f"bundle."
            )
        assert census["discovery"] == "listed", (
            f"{result.row.host}: {census.get('reason')}\n"
            f"host enumerated: {census.get('commands')}"
        )


class TestTheCensusIsHonest:
    """"Everything skipped" and "everything passed" must never look alike."""

    def test_every_host_reports_a_named_state(self, discovery):
        _result, census, _entry = discovery
        assert census["discovery"] in (
            "listed", "not-listed", "failed", "unavailable-without-credential"
        ), census

    def test_no_host_is_silently_absent(self, discovery):
        _result, census, _entry = discovery
        assert census.get("host"), "a census row with no host name names nothing"


# =========================================================================== #
# The credentialed tier (C6, C6a, C6c, C8)
# =========================================================================== #

#: Which rows this tier runs, in ascending cost — and it is `host_matrix.yaml`'s
#: WHOLE credentialed set, not the cheap half of it. The `--ask` rows are the only
#: ones with no subagent batch at all, which on the reference host is the whole
#: difference between a few cents and a row whose spend scales with batch count.
#:
#: Scheduling only the ask and review-plan cells was a hole rather than a saving:
#: the matrix declares a third row per host (Phase 4's done-when), and without it
#: **write-scheduler dispatch through a real host, the one native applied-byte
#: path and both authentication negatives never execute from either host `make`
#: target** — while the lane still reports green, which is exactly the "everything
#: skipped looks like everything passed" outcome this module exists to prevent.
CREDENTIALED_ROWS = [
    ("opencode", "opencode-ask"),
    ("codex", "codex-ask"),
    ("claude-code", "claude-code-ask"),
    ("opencode", "opencode-review-plan"),
    ("codex", "codex-review-plan"),
    ("claude-code", "claude-code-review-plan"),
    # The two `--full` rows: one host turn each, and between them the ONLY
    # coverage on this lane of all eight workflow phases driven by a real host.
    # `full` runs every phase in one process with no invoker between them, so
    # this costs two rows rather than the sixteen a phase-per-row expansion would
    # need — and it needs no pre-command leg, because the chain seeds each phase
    # for the next one itself.
    #
    # THERE IS NO THIRD ONE, AND THAT IS RECORDED RATHER THAN MISSING.
    # `host_matrix.yaml`'s claude-code entry carries `full_row: not-applicable`
    # with its derivation: that host's shim forbids `--exec-subagents` and
    # `--detach`, so `full` stops at review-plan's [VALIDATION_PENDING] and then
    # at apply-suggestions' [WRITE_PHASE_NOT_EXECUTED], both survivable only by
    # an agent that re-invokes the printed `next` — which that host has measurably
    # failed to do twice.
    ("codex", "codex-full"),
    ("opencode", "opencode-full"),
    # The write rows, whose two shapes are deliberately opposite. A real executor
    # CLI pointed at a tool-call-free stub can only ever produce a CLEAN FAILURE,
    # so those rows assert that outcome and an `applied` batch there fails just as
    # loudly (bytes landing from a tool-call-free stub means the redirect leaked);
    # the reference host's native row is the ONE row on either lane where
    # `applied` bytes are a pass rather than a rig bug, and this plan's only
    # applied-byte evidence through a real host.
    ("codex", "codex-apply-executor"),
    ("opencode", "opencode-apply-executor"),
    ("claude-code", "claude-code-apply-native"),
    # Last, and the only rows that run with a seeded credential REMOVED. They are
    # cheap by construction — the point is that they are refused before the first
    # model call — and the two hosts do NOT fail alike, which is precisely why
    # both are scheduled: a green codex negative proves nothing about opencode.
    ("codex", "codex-auth-negative"),
    ("opencode", "opencode-auth-negative"),
    # THE PACKAGE ROUTE, and the only rows here that are not the file shim.
    # Everything above installs multi-llm by writing a command file the host
    # scans; these two install the npm plugin instead, which registers the same
    # command AND a `multi_llm` tool from a module. The routes are graded
    # separately because promotion is keyed on `Row.route` — crediting a plugin
    # run to the shim, or the reverse, would let either route rot behind the
    # other's evidence. Last in the list for the same reason the negatives are
    # near it: they are the newest and the cheapest to lose if the budget runs
    # out mid-lane.
    ("opencode", "opencode-package-ask"),
    ("opencode", "opencode-package-review-plan"),
]


class TestTheScheduleMatchesTheMatrix:
    """`CREDENTIALED_ROWS` is a hand-written list, so it can silently fall behind.

    The `graded` fixture already checks one direction — a listed row that the
    matrix does not name fails immediately. The other direction was unguarded and
    is the dangerous one: a row **authored into `host_matrix.yaml` and never
    added here simply does not run**, while the lane still reports green and the
    census still says every declared row executed. That is the same
    "everything skipped looks like everything passed" failure this module exists
    to prevent, one level up — the census can only count what the schedule
    declares.
    """

    def test_every_matrix_row_is_scheduled(self):
        declared = {(host, row["id"]) for host, entry in HOSTS
                    for row in (entry.get("rows") or [])}
        scheduled = set(CREDENTIALED_ROWS)
        unscheduled = sorted(declared - scheduled)
        assert not unscheduled, (
            f"host_matrix.yaml declares {unscheduled} and CREDENTIALED_ROWS does "
            f"not schedule them, so they never run and nothing reports that they "
            f"did not. Add them here, or delete them from the matrix — an "
            f"authored row nobody runs is worse than no row, because the census "
            f"counts it as covered.")

    def test_a_host_that_declares_no_full_row_has_none(self):
        """4.2's structural limit, enforced rather than only written down.

        `full_row: not-applicable` is a claim about the product (claude-code's
        shim forbids `--exec-subagents`/`--detach`, so `full` cannot complete
        there). If someone later adds the row anyway, the cell would sit beside
        it saying the opposite — and the cell is what a reader trusts.
        """
        for host, entry in HOSTS:
            has_full = any(str(row.get("id", "")).endswith("-full")
                           for row in (entry.get("rows") or []))
            declared = entry.get("full_row")
            if declared == "not-applicable":
                assert not has_full, (
                    f"{host}: declares `full_row: not-applicable` and carries a "
                    f"`--full` row anyway. One of the two is wrong, and the "
                    f"reason cell is the one a reader trusts.")
                assert entry.get("full_row_reason"), (
                    f"{host}: `full_row: not-applicable` with no "
                    f"`full_row_reason`. §6.3's rule is that a committed, "
                    f"measured 'not applicable, because …' is an acceptable "
                    f"cell; an unexplained one is not.")
            elif not has_full:
                pytest.fail(
                    f"{host}: has no `--full` row and declares no `full_row` "
                    f"verdict, so 8-of-8 phase coverage on this host is "
                    f"untested rather than recorded as impossible. Add the row, "
                    f"or add `full_row: not-applicable` with its reason.")


#: The pre-flight refusal, not a ceiling (C6). It is checked BETWEEN rows, which
#: is what turns an overshoot into one blown row instead of six.
BUDGET_USD = float(os.environ.get("MLLM_E2E_BUDGET_USD", "2.00"))

def _key_usage() -> float | None:
    """The dedicated key's own `usage`, which is this lane's measured actual.

    Attributable to one row precisely because the key is dedicated and the rows
    run one at a time. `None` means the readback was unavailable, and a row is
    then reported `actual: unavailable` rather than credited with a number
    nobody measured.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["data"]["usage"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


class _Spend:
    """The running total the sequential checkpoint consumes."""

    baseline: float | None = None
    total: float = 0.0

    @classmethod
    def check(cls) -> None:
        if cls.total > BUDGET_USD:
            pytest.exit(
                f"host lane: cumulative spend ${cls.total:.4f} crossed "
                f"MLLM_E2E_BUDGET_USD=${BUDGET_USD:.2f} — refusing to start "
                f"another row", returncode=1)


class _Census:
    """How many credentialed rows really ran, named — the lane's own coverage.

    A green pytest exit cannot tell "every row passed" from "every row skipped":
    both are exit 0. So the rows keep their own count, it is written beside the
    run bundle as `host-lane-census.json`, and `_refuse_empty_lane` below reads
    it. `3/0/0` and `0/0/3` are both possible outcomes of the same command, and a
    summary that cannot tell them apart is exactly how zero host coverage passes
    for a green matrix.
    """

    ran: list[dict] = []
    skipped: list[dict] = []

    @classmethod
    def record_ran(cls, row_id: str, host: str, mode: str = "openrouter") -> None:
        cls.ran.append({"row": row_id, "host": host, "credential_mode": mode})

    @classmethod
    def record_skipped(cls, row_id: str, host: str, reason: str) -> None:
        cls.skipped.append({"row": row_id, "host": host, "reason": reason})

    @classmethod
    def counts(cls) -> dict:
        counts = {"openrouter": 0, "fallback": 0, "skipped": len(cls.skipped)}
        for row in cls.ran:
            mode = str(row.get("credential_mode") or "openrouter")
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    @classmethod
    def status(cls) -> str:
        return "ran" if cls.ran else "not-run"

    @classmethod
    def report(cls) -> str:
        counts = cls.counts()
        lines = [
            f"  status  : {cls.status()}",
            f"  declared: {len(CREDENTIALED_ROWS)} credentialed row(s)",
            f"  counts  : openrouter={counts['openrouter']} "
            f"fallback={counts['fallback']} skipped={counts['skipped']}",
        ]
        lines += [f"  ran     : {row['row']} ({row['credential_mode']})"
                  for row in cls.ran]
        lines += [f"  skipped : {row['row']} — {row['reason']}"
                  for row in cls.skipped]
        return "\n".join(lines)

    @classmethod
    def write(cls, utc: str):
        """Beside the run bundle, so the count survives the pytest scrollback."""
        path = harness.RUNS / utc / "host-lane-census.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "lane": "host",
            "tier": "credentialed",
            "status": cls.status(),
            "declared_rows": [row for _host, row in CREDENTIALED_ROWS],
            "counts": cls.counts(),
            "ran": cls.ran,
            "skipped": cls.skipped,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _refuse_empty_lane(config, utc: str) -> None:
    """`not-run` is its own outcome, and it exits non-zero (C6).

    A missing credential makes a host's rows **skip with a named reason**, and a
    run covering only the provider lane is a valid green run. An explicitly
    invoked host lane that executed zero rows is not: it covered nothing, so it
    reports `not-run` and exits non-zero. `--allow-empty-host-lane` is the only
    way to a green exit out of that state — it suppresses the failure, never the
    census — and it exists for the contributor deliberately confirming the rig is
    wired up without a key. It is never the default, because the default has to
    make "nobody here has a key" visible.
    """
    path = _Census.write(utc)
    if _Census.ran:
        return
    if config.getoption("allow_empty_host_lane", default=False):
        return
    pytest.exit(
        "host lane: NOT-RUN — 0 of "
        f"{len(CREDENTIALED_ROWS)} declared credentialed rows executed, so this "
        "lane covered nothing. That is a distinct outcome from a pass, and it is "
        "reported as one.\n" + _Census.report() + f"\n  census  : {path}\n"
        "  Pass --allow-empty-host-lane to accept a credential-free invocation "
        "of this lane (the census is printed either way), or run the free tier: "
        "make e2e-container-host-discovery", returncode=1)


@pytest.fixture(scope="module", params=CREDENTIALED_ROWS,
                ids=[r for _h, r in CREDENTIALED_ROWS])
def graded(request, posture, utc):
    """Run one credentialed row and return (result, row entry, host entry).

    **Strictly sequential, with a budget checkpoint between rows.** The matrix is
    never parallelized: the checkpoint is the closest thing to enforcement our own
    code can honestly offer, since the only limit that actually holds is the one
    enforced by the party doing the billing.
    """
    host, row_id = request.param
    if not os.environ.get("OPENROUTER_API_KEY"):
        # A named skip, a census entry, and — unless the operator asked for an
        # empty lane — a non-zero `not-run`, because a lane that executed nothing
        # must never read as a lane that passed everything.
        _Census.record_skipped(row_id, host, "OPENROUTER_API_KEY unset")
        _refuse_empty_lane(request.config, utc)
        pytest.skip("the credentialed host lane needs OPENROUTER_API_KEY (a "
                    "DEDICATED key carrying a key-level spend limit)")
    _Spend.check()
    before = _key_usage()

    entry = dict(dict(HOSTS)[host])
    matrix_row = next((dict(r) for r in entry.get("rows") or []
                       if r.get("id") == row_id), None)
    assert matrix_row, f"host_matrix.yaml names no row {row_id!r} under {host!r}"

    # The provider fan-out identity: every model call the workflow makes is
    # dispatched by name through /opt/mllm/bin and lands on the stub. The HOST's
    # own reasoning is the only real spend on the two executor rows.
    #
    # ON A NEGATIVE AUTH ROW THE RUNNER MUST BE THE ROW'S OWN CLI. The row's
    # `remove_seed` names a file under that CLI's seeded home, and the gate it is
    # meant to trip is the executor's preflight of the CONFIGURED runner — so a
    # codex runner would sail past opencode's auth.json going missing and the row
    # would assert nothing at all.
    #
    # AND THE SLUG MUST BE IN THAT CLI'S OWN ADDRESSING SCHEME. opencode resolves
    # models as `provider/model`; handed a bare `stub-model` it parses provider
    # `stub-model` with an empty model half and raises
    # `ProviderModelNotFoundError: Model not found: stub-model/.` before a single
    # HTTP call leaves the process — which is what exited `opencode-auth-negative`
    # 3 on `runs/20260820T093838Z`, with `stub.jsonl` holding exactly two records
    # (a `/__health` and the codex fan-out's `POST /v1/responses`) and zero
    # opencode calls. The provider lane already pins the qualified form
    # (`tests/container/redirect.yaml:1024`, `model_arg: "stub/stub-model"`), so
    # this matches it rather than inventing a second spelling. codex keeps its
    # bare token, which codex resolves fine; `host_matrix.yaml` declares no runner
    # for either row, so this expression is the only place the slug is built.
    negative = bool(matrix_row.get("remove_seed"))
    negative_model = "stub/stub-model" if host == "opencode" else "stub-model"
    row = harness.Row(
        row_id=row_id, lane="host", host=host, creds="openrouter",
        helpers=sorted({"codex", host} if negative else {"codex"}),
        spec="codex:stub-model",
        runner=f"{host}:{negative_model}" if negative else "codex:stub-model",
        row_timeout_s=int(matrix_row.get("wall_clock_s") or 600) + 500,
        # Without it the worker really runs and really fails: a scenario-less
        # stub answers a bare "OK", which is not a suggestions document, and the
        # row grades the stub's default answer instead of the host.
        scenario=matrix_row.get("scenario"),
        # THE PRE-COMMAND LEG. Ordered dispatcher invocations the entrypoint runs
        # in the row's workspace BEFORE the host process starts — the rig gap
        # that kept `apply-suggestions` unrunnable on this lane, since it exits 1
        # in `_setup` against a workspace with no review-plan output and this
        # lane launches exactly one host process per row.
        #
        # `Row.commands` and the `--command` forwarding are the provider lane's
        # existing plumbing; only the host branch of the entrypoint had never
        # consumed them. The leg's receipts land in `receipts/<row>-pre/` rather
        # than `receipts/<row>/`, so no assertion in this module can mistake the
        # rig's own invocation for the host's.
        commands=[str(c) for c in (matrix_row.get("pre_commands") or [])],
        # WHICH INSTALL ROUTE THIS ROW EXERCISES, and it has to come off the
        # matrix row: `harness.Row.route` defaults to "file", `build_run_argv`
        # forwards `--route` only when it is not that, and the entrypoint's
        # `stage_host` branches on it to skip staging a file shim the plugin
        # would overwrite in memory anyway. Read it here or a row authored with
        # `route: package` runs the FILE route under a package row's name --
        # green, and grading the wrong object.
        route=str(matrix_row.get("route") or "file"),
    )
    result = harness.run_row(row, posture=posture, utc=utc)
    _Census.record_ran(row_id, host, "openrouter")
    _Census.write(utc)

    after = _key_usage()
    if before is not None and after is not None:
        _Spend.total += after - before
    return result, matrix_row, entry


@pytest.fixture
def workflow(graded):
    """`graded`, narrowed to the rows that are supposed to RUN a workflow.

    The negative auth rows are graded on a refusal — no phase completes, no
    fan-out reaches the stub, and the host's own exit is not a verdict — so every
    assertion below that reads "the workflow ran" would be asserting the opposite
    of what those rows exist to prove. They are graded by
    `TestTheNegativeAuthRowsRefuse` instead, and skipped here by name rather than
    silently excluded from the parametrization.
    """
    _result, matrix_row, _entry = graded
    if matrix_row.get("remove_seed"):
        pytest.skip(
            f"{matrix_row.get('id')}: a negative auth row asserts a REFUSAL, not "
            f"a workflow — graded by TestTheNegativeAuthRowsRefuse")
    return graded


class TestThePreCommandLegRan:
    """A row's prerequisite phase, run by the RIG before the host process starts.

    `host_matrix.yaml`'s header finding (1): this lane seeds one workspace per row
    and launches exactly ONE host process against it, and `apply-suggestions`
    exits 1 in `_setup` — "Run --review-plan first to generate validation results"
    — against a workspace with no review-plan output. The provider lane's
    applied-bytes row answers the same problem with two commands; the host lane
    had nowhere to put the first one until `pre_commands` existed.

    **This is not the host's work and is never counted as it.** The leg's receipts
    are written to `receipts/<row>-pre/`, one directory across from the
    `receipts/<row>/` that every "the agent really ran our dispatcher" assertion
    reads, so a row can never satisfy its receipt anchor on the rig's own
    invocation.

    Graded on `graded` rather than `workflow`: the negative auth rows need the leg
    too, and need it to run *before* their seeded credential is removed.
    """

    pytestmark = pytest.mark.live

    def test_every_declared_pre_command_ran_and_exited_zero(self, graded):
        result, matrix_row, _entry = graded
        declared = [str(c) for c in (matrix_row.get("pre_commands") or [])]
        if not declared:
            pytest.skip(f"{matrix_row.get('id')}: the row declares no pre_commands")
        # A ROW THAT CANNOT REACH THE LEG IS NOT A LEG FAILURE. `build_host_argv`
        # runs BEFORE the leg and raises on a row with no `args`, so such a row
        # aborts in ~1.5 s having spent nothing and having executed no
        # pre-command. Asserting the leg ran there reports the same single blocker
        # again, four more assertions per row, and buries the rows that are
        # genuinely red. The missing `args` is the finding; this is not.
        #
        # SINCE 2026-08-20 NO ROW IN THE MATRIX IS IN THAT STATE — all five that
        # were are wired — so this branch is a guard against a row losing its
        # `args` again, not a description of the lane. If it ever fires, the fix
        # is the `args` string, not this test.
        if not matrix_row.get("args"):
            pytest.skip(
                f"{matrix_row.get('id')}: declares no `args`, so the container "
                f"aborts in render_prompt before the pre-command leg is reached. "
                f"The leg is untestable here until that row is wired.")
        path = result.out_dir / "pre-commands.json"
        assert path.is_file(), (
            f"{result.row.host}: the row declares {len(declared)} pre-command(s) "
            f"and the container wrote no pre-commands.json, so the leg never ran. "
            f"Either the harness did not forward them or the host branch reached "
            f"the launch without them — and a host launched against an unprepared "
            f"workspace fails for the wrong reason.\n"
            f"--- container stderr (tail) ---\n{result.stderr[-2000:]}")
        records = json.loads(path.read_text(encoding="utf-8"))
        failed = [record for record in records if record.get("exit") != 0]
        assert not failed, (
            f"{result.row.host}: pre-command leg failed: {failed}. The host was "
            f"deliberately not launched — a prerequisite phase that did not "
            f"complete leaves the row's real mode nothing to act on.")
        assert [record.get("command") for record in records] == declared, (
            f"{result.row.host}: the leg ran "
            f"{[record.get('command') for record in records]}, and the row "
            f"declares {declared}. Order is part of the declaration: a chain that "
            f"ran a different set, or the same set in a different order, prepared "
            f"a different workspace.")

    def test_the_legs_receipts_stayed_out_of_the_hosts_directory(self, graded):
        """The separation, asserted rather than assumed.

        It is the one property that keeps this feature from being a way to pass
        the receipt anchor without a host: if the leg ever wrote into
        `receipts/<row>/`, a row whose host did nothing at all would still show a
        `review-plan` receipt with a terminal payload.
        """
        result, matrix_row, _entry = graded
        if not (matrix_row.get("pre_commands") or []):
            pytest.skip(f"{matrix_row.get('id')}: the row declares no pre_commands")
        if not matrix_row.get("args"):
            pytest.skip(
                f"{matrix_row.get('id')}: declares no `args` — see the sibling "
                f"assertion. The leg is never reached, so it left no receipts to "
                f"be in the wrong place.")
        leg = result.out_dir / "receipts" / f"{result.row.row_id}-pre"
        assert leg.is_dir() and any(leg.glob("*.receipt.json")), (
            f"{result.row.host}: the leg ran but left no receipt under {leg}. "
            f"Either MLLM_RECEIPT_DIR did not reach the dispatch or — the case "
            f"that matters — the leg wrote into the host's own receipt directory, "
            f"where its invocations would be graded as the host's.")


class TestTheAgentReallyRanOurDispatcher:
    """The anchor is the RECEIPT, never the host's prose.

    What reaches the harness is the host's event stream, in which
    `MULTI_LLM_RESULT` is at best a JSON-escaped field inside a nested tool-result
    envelope and at worst paraphrased away — measured on all three hosts. So a
    host that ran our command and narrated it badly still passes, and a host that
    never ran it produces no receipt at all and fails naming the row.
    """

    pytestmark = pytest.mark.live

    def test_a_receipt_exists_for_the_row(self, workflow):
        result, matrix_row, entry = workflow
        dispatcher = _dispatcher_receipts(result)
        assert dispatcher, (
            f"{result.row.host}: NO dispatcher receipt for row "
            f"{result.row.row_id!r}. The host never ran our command.\n"
            f"  cli version : {entry.get('cli_version', 'unrecorded')}\n"
            f"  argv        : {result.host_run.get('argv')}\n"
            f"  host exit   : {result.host_run.get('exit')} "
            f"(timed_out={result.host_run.get('timed_out')})")

    def test_enough_receipts_for_the_rows_shape(self, workflow):
        result, matrix_row, _entry = workflow
        minimum = int((matrix_row.get("expect") or {}).get("receipts_min") or 1)
        dispatcher = _dispatcher_receipts(result)
        assert len(dispatcher) >= minimum, (
            f"{result.row.host}: {len(dispatcher)} dispatcher receipt(s), "
            f"expected >= {minimum}: {[_mode_of(r) for r in dispatcher]}")

    def test_the_payload_names_the_expected_mode_and_phase(self, workflow):
        """The phase the RECEIPTS name — which is not always the phases the row runs.

        `expect.phases` is what `state.json` must record completed, and on every
        one-mode row that is also what the receipt carries, so the two are one
        list. A `--full` row breaks the identity: `full` runs all eight phases by
        calling each phase's own `main(argv)` in-process
        (`full_workflow_orchestrator.run_mode` — "the dispatcher owns one
        MULTI_LLM_RESULT line and one result context per invocation, and nesting
        it would emit a second line"), so the host makes ONE dispatcher
        invocation carrying ONE result line, whose `phase` is `modes.py`'s phase
        for the mode word: `full`.

        `expect.receipt_phase` is that row's own answer, and it NARROWS this
        assertion rather than relaxing it — the receipt must name exactly the
        declared phase. Nothing is dropped: the eight phases stay graded, one
        test down, against `state.json`'s `phases_completed`, which is the
        artifact rather than the claim.
        """
        result, matrix_row, _entry = workflow
        expect = matrix_row.get("expect") or {}
        declared = expect.get("receipt_phase")
        phases = [declared] if declared else (expect.get("phases") or [])
        payloads = [r["result"] for r in _dispatcher_receipts(result)
                    if isinstance(r.get("result"), dict)]
        assert payloads, (
            f"{result.row.host}: every dispatcher receipt carries result: null — "
            f"no invocation emitted a MULTI_LLM_RESULT line, and every mode emits "
            f"one")
        terminal = [p for p in payloads if p.get("state") in TERMINAL_STATES]
        assert terminal, f"{result.row.host}: no receipt reached a terminal state"
        for phase in phases:
            assert any(p.get("phase") == phase for p in payloads), (
                f"{result.row.host}: no receipt names phase {phase!r}; "
                f"receipts named {sorted({p.get('phase') for p in payloads})}")

    def test_every_terminal_receipt_exited_as_the_row_declares(self, workflow):
        """`expect.exit`, C8's other half — and it is the PAYLOAD's exit, not the host's.

        Three exit codes are in play on one row and they are three different
        facts: the host process's own (graded by
        `TestTheHostsOwnExitIsReportedSeparately`), the receipt's process exit,
        and `MULTI_LLM_RESULT.exit` — the dispatcher's own verdict on the
        invocation, which is the one `host_matrix.yaml` declares.

        Without this the state check above is satisfied by an invocation that
        finished **badly**: `partial` is terminal at exit 0 (D6 row 3) but the
        membership test alone accepts a payload carrying exit 2 ("nothing to do",
        i.e. the phase was already complete and this row proved nothing) or exit 3
        ("needs a human") beside it. Every workflow row in the matrix declares
        `exit: 0`; the negative auth rows name their own and are graded on a
        refusal by `TestTheNegativeAuthRowsRefuse`.

        Non-terminal payloads are deliberately not graded: a detaching row's shim
        polls `status`, and a poll that correctly reports the worker still running
        exits 2 on purpose.
        """
        result, matrix_row, _entry = workflow
        declared = (matrix_row.get("expect") or {}).get("exit")
        if declared is None or declared == "unknown":
            pytest.skip(
                f"{matrix_row.get('id')}: host_matrix.yaml declares no measured "
                f"expect.exit for this row")
        payloads = [r["result"] for r in _dispatcher_receipts(result)
                    if isinstance(r.get("result"), dict)]
        terminal = [p for p in payloads if p.get("state") in TERMINAL_STATES]
        assert terminal, (
            f"{result.row.host}: no receipt reached a terminal state, so the row "
            f"has no exit to grade against expect.exit={declared!r}")
        wrong = [(p.get("mode"), p.get("phase"), p.get("state"), p.get("exit"))
                 for p in terminal if p.get("exit") != int(declared)]
        assert not wrong, (
            f"{result.row.host}: {len(wrong)} terminal receipt(s) exited other "
            f"than the declared expect.exit={int(declared)} — "
            f"(mode, phase, state, exit): {wrong}. A terminal state is not a "
            f"verdict on its own: `partial` is emitted at exit 0, so a payload "
            f"reporting exit 2 (nothing to do — the phase was already complete) "
            f"or exit 3 (needs a human) beside it is a failed row, not a green one")

    def test_the_payload_named_as_many_artifacts_as_the_row_declares(self, workflow):
        """`artifacts_min`, per row, because one mode legitimately names none.

        MEASURED: `ask` completes with `artifacts: []` on every host, foreground
        and detached alike — the answers file is printed on its own line and
        named by `status`, but not by the ask phase's own result. A blanket
        "non-empty artifacts" rule would fail a passing row, and quietly dropping
        the rule would excuse a phase that really produced nothing. So it is a
        declared number.
        """
        result, matrix_row, _entry = workflow
        expect = matrix_row.get("expect") or {}
        minimum = expect.get("artifacts_min")
        if minimum is None:
            minimum = 1
        named = [a for r in _dispatcher_receipts(result)
                 if isinstance(r.get("result"), dict)
                 for a in (r["result"].get("artifacts") or [])]
        assert len(named) >= int(minimum), (
            f"{result.row.host}: receipts named {len(named)} artifact(s), "
            f"expected >= {minimum}")

    def test_every_named_artifact_exists_and_is_non_empty(self, workflow):
        """Named, not merely created — and read back off the bind mount."""
        result, _row, _entry = workflow
        missing = []
        for receipt in _dispatcher_receipts(result):
            payload = receipt.get("result")
            if not isinstance(payload, dict):
                continue
            for artifact in payload.get("artifacts") or []:
                exported = _exported(result, str(artifact))
                if exported is None:
                    continue          # outside the exported plan dir (e.g. $HOME)
                if not exported.is_file() or exported.stat().st_size == 0:
                    missing.append(str(artifact))
        assert not missing, (
            f"{result.row.host}: receipt named artifacts that are absent or "
            f"empty on the bind mount: {missing}")


class TestDiscoveryOnTheHostsWithNoFreeSurface:
    """Discovery on `codex` and `claude-code`, proven by the turn we already pay for.

    The free tier cannot ask these two hosts to enumerate our command: neither
    has a credential-free listing surface, measured, `doctor` included. That is a
    genuine upstream wall and this module reports it as one — `discovery:
    unavailable-without-credential`, never a pass.

    But "we cannot ask it to list" is not the same claim as "we have no evidence
    it found the shim", and the credentialed `ask` rows have been producing the
    stronger evidence all along. A `util instructions <mode>` receipt is step 3 of
    the shim, printed nowhere else: to emit that exact command the host had to
    locate the staged shim, load it, and read far enough to reach the read step.
    A host that never found the file cannot produce it. So this records
    `listed-via-prompted-turn` from evidence already on disk, at **$0.00** — it
    schedules no row, and rides on the cheapest rows in the matrix.

    THREE THINGS IT DELIBERATELY DOES NOT DO.

    * It does not touch the free tier's verdict. That tier still has no evidence,
      still skips, and still writes `unavailable-without-credential`. The two
      tiers cannot see each other's data — the module-scoped `discovery` fixture's
      tests all run before the first credentialed row exists — and the fix for
      that is not to make the free tier guess; it is to grade the credentialed
      evidence where it lives, which is here.
    * It does not claim ENUMERATION. `opencode debug config` returns a list the
      host itself compiled; this is one host, one turn, one command. The verdict
      is spelled differently for exactly that reason, and `opencode` keeps the
      stronger word.
    * It does not grade every row. One row per host, the cheapest one, so a
      review-plan or write row that fails for its own reasons does not also
      report a discovery failure it did not cause.
    """

    pytestmark = pytest.mark.live

    def test_a_prompted_turn_proves_the_host_found_our_shim(self, workflow):
        result, matrix_row, entry = workflow
        if (entry.get("list_step") or {}).get("form") != "none":
            pytest.skip(
                f"{result.row.host}: this host has a credential-free listing "
                f"surface, and discovery on it is graded there — by "
                f"TestTheStagedCommandIsDiscoverable, for free. A prompted turn "
                f"is the fallback, never the preferred evidence.")
        if not str(matrix_row.get("id") or "").endswith("-ask"):
            pytest.skip(
                f"{matrix_row.get('id')}: discovery is graded once per host, on "
                f"its `ask` row — the only row with no subagent batch, so a "
                f"failure here is about the shim being found rather than about "
                f"anything a batch did")
        evidence = [receipt for receipt in _dispatcher_receipts(result)
                    if _mode_of(receipt) == "util instructions"]
        assert evidence, (
            f"{result.row.host}: the host ran no `util instructions <mode>`, so "
            f"nothing in this row shows it read the staged shim. That command is "
            f"step 3 of the shim and is printed nowhere else; a host that never "
            f"found the file cannot emit it.\n"
            f"  receipts    : {[_mode_of(r) for r in _dispatcher_receipts(result)]}\n"
            f"  host argv   : {result.host_run.get('argv')}\n"
            f"  host exit   : {result.host_run.get('exit')}")
        (result.out_dir / "discovery-prompted.json").write_text(json.dumps({
            "host": result.row.host,
            "row": matrix_row.get("id"),
            "tier": "credentialed",
            "discovery": "listed-via-prompted-turn",
            "reason": (
                "no credential-free listing surface exists on this host "
                "(measured, `doctor` included), so the host was asked instead: "
                "it located and read the staged shim and ran the shim's own step "
                "3 command. Weaker than `listed` — one command, not an "
                "enumeration — and stronger than `unavailable-without-credential`, "
                "which is all the free tier can honestly say."),
            "evidence": [receipt.get("argv") for receipt in evidence],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TestTheWorkerNotTheLauncher:
    """A launcher can succeed over a worker that failed.

    `detach.launch` records the log and `.status.json` as artifacts and returns 0,
    so its receipt reads `state: done, exit: 0` **before the workflow has run at
    all**. On a detaching row the verdict is the worker's own `run` block.
    """

    pytestmark = pytest.mark.live

    def test_the_declared_run_state_was_reached(self, workflow):
        result, matrix_row, _entry = workflow
        expected = (matrix_row.get("expect") or {}).get("run")
        states = result.run_states
        if expected == "done":
            assert states, (
                f"{result.row.host}: expected a detached worker and found no "
                f"`.status.json` run block at all — the row was graded on a launch")
            for phase, block in states.items():
                assert block.get("state") == "done", (
                    f"{result.row.host}: {phase} run.state={block.get('state')!r}")
                assert block.get("exit_code") == 0, (
                    f"{result.row.host}: {phase} run.exit_code="
                    f"{block.get('exit_code')!r}")
        elif expected == "none":
            # An assertion, not an exemption: a foreground row that suddenly
            # grows a `run` block means a shim started detaching behind our back.
            assert not states, (
                f"{result.row.host}: this row's shim forbids --detach, yet a "
                f"worker run block appeared: {states}")
        else:
            # No silent third branch. A row whose `expect.run` is missing,
            # misspelled or newly invented would otherwise be graded by nothing
            # at all and pass vacuously, so an unrecognized value is a matrix
            # error reported here rather than a row that quietly stops counting.
            pytest.fail(
                f"{result.row.host}: row {result.row.row_id!r} declares "
                f"expect.run={expected!r}, which this test does not grade. "
                f"Use 'done' (a detached worker must reach run.state=done) or "
                f"'none' (no worker run block may appear).")

    def test_the_phase_recorded_itself_completed(self, workflow):
        """Completion, not creation: a phase dir exists before the work in it.

        The record that matters is `state.json`'s `phases_completed`, **not**
        `{phase}/.status.json`'s in-phase `state`. Until 2026-09-11 a review-plan
        row that ran to a terminal `run.state: done` with `validation.json` and
        the report on disk still left `.status.json` reading
        `state: grouping_complete` (the last checkpoint the fan-out wrote); the
        orchestrators now write `state: complete` at the end, but the workflow
        record stays the assertion target so a row captured before that change
        still grades.
        """
        result, matrix_row, _entry = workflow
        completed = (result.workflow_state().get("phases_completed") or {})
        for phase in (matrix_row.get("expect") or {}).get("phases") or []:
            if phase == "ask":
                continue          # `ask` records no workflow phase
            assert phase in completed, (
                f"{result.row.host}: state.json records no completion for "
                f"{phase!r}; it records {sorted(completed)}")


class TestTheHostsOwnExitIsReportedSeparately:
    """"The host did not terminate" and "the command never ran" differ.

    A wall-clock-killed host whose dispatcher had already finished leaves a
    complete receipt, so that row reports "command dispatched, host did not
    terminate" rather than one undifferentiated red.
    """

    pytestmark = pytest.mark.live

    def test_the_host_terminated_within_its_bound(self, workflow):
        result, _row, _entry = workflow
        record = result.host_run
        assert not record.get("timed_out"), (
            f"{result.row.host}: host hit its {record.get('bound_s')}s bound. "
            f"Dispatcher receipts present: {len(_dispatcher_receipts(result))} — "
            f"so this is 'did not terminate', not 'never ran'.")

    def test_the_host_exited_zero(self, workflow):
        result, _row, _entry = workflow
        record = result.host_run
        assert record.get("exit") == 0, (
            f"{result.row.host}: host exited {record.get('exit')}\n"
            f"--- host stderr (tail) ---\n"
            f"{(result.out_dir / 'host-stderr.txt').read_text()[-1500:]}")


class TestTheFanOutNeverWentLive:
    """With egress open, the guarantee downgrades from impossibility to detection.

    Every model call the *workflow* makes must land on the stub; only the host's
    own reasoning is real spend.
    """

    pytestmark = pytest.mark.live

    def test_the_stub_answered_the_fan_out(self, workflow):
        result, matrix_row, entry = workflow
        if _fan_out_is_native(matrix_row, entry):
            pytest.skip(
                f"{result.row.host}: this row's whole batch is dispatched through "
                f"the host's OWN subagent tool, which never crosses /opt/mllm/bin, "
                f"so there is no wrapped fan-out to answer. Its cost is real spend "
                f"and is graded by TestTheSpendWasMeasured; its bytes are graded by "
                f"TestTheWriteRowsAreGradedOnBytes.")
        calls = [r for r in result.stub_log if r.get("surface") != "health"]
        assert calls, (
            f"{result.row.host}: ZERO requests reached the stub. With egress open "
            f"that gap is exactly the signature of a provider that went live.")

    def test_no_child_carried_a_credential(self, workflow):
        """The scrub receipt: variable NAMES only, never values."""
        result, _row, _entry = workflow
        leaked = []
        for record in result.call_log:
            for name in record.get("child_env_names") or []:
                if name in ("OPENROUTER_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                    leaked.append((record.get("cli"), name))
        assert not leaked, (
            f"{result.row.host}: a provider subprocess inherited a live "
            f"credential: {leaked}")


class TestTheSpendWasMeasured:

    pytestmark = pytest.mark.live

    #: `graded`, not `workflow`: a refused row is still a row that ran, and a
    #: negative auth row that somehow spent money is a finding rather than an
    #: exemption.
    def test_the_row_recorded_an_actual(self, graded):
        result, _row, _entry = graded
        usage = result.usage
        assert usage, f"{result.row.host}: no usage readback recorded for the row"
        assert usage.get("delta_usd") is not None, (
            f"{result.row.host}: actual: unavailable — the running total is an "
            f"estimate from here on")


class TestTheWriteRowsAreGradedOnBytes:
    """The third row per host, and the model's claim is never the verdict.

    Two shapes, deliberately opposite, each declared per row rather than inferred:

    * **`clean-failure`** on the two executor hosts. A real executor CLI pointed at
      a tool-call-free stub can only ever claim success and move nothing, so that
      is what the row asserts — and an `applied` batch there fails just as loudly,
      because bytes landing from a tool-call-free stub mean the redirect leaked.
    * **`applied`** on the reference host's native row: the ONE row on either lane
      where an applied batch is a pass rather than a rig bug, and this plan's only
      applied-byte evidence through a real host. An all-clean-failure result there
      is a FAILED row — "dispatched, nothing landed" — because accepting a no-op
      is the exact false positive this rule exists to stop.
    """

    pytestmark = pytest.mark.live

    def test_the_report_exists_and_the_run_finalized(self, workflow):
        result, matrix_row, entry = workflow
        _write_outcome(matrix_row)           # skips every non-write row by name
        if _fan_out_is_native(matrix_row, entry):
            # ON THE NATIVE ROW THE ASSERTION INVERTS, and it is still an
            # assertion rather than a skip.
            #
            # `write_batch_report.json` is written ONLY by
            # `utils/write_batch_scheduler.WriteBatchScheduler`, and
            # `apply_orchestrator_base.run_write_executor` returns None — "not
            # requested, carry on natively" — unless `--exec-subagents` or
            # `--dry-run-subagents` was passed. This row's host is told by its own
            # shipped shim to "Never pass `--exec-subagents` on this host"
            # (`host_capabilities._SUBAGENTS["native"]`), and `host_matrix.yaml`
            # lists the flag under `forbidden_flags`. So the executor's per-batch
            # account cannot exist here — the native loop records prose summaries
            # through `util prior-changes append` and nothing machine readable —
            # and demanding it would fail this row for a category error rather
            # than for anything the host did.
            #
            # Its ABSENCE, though, is worth asserting: a report appearing means
            # the agent typed the flag its shim forbids, which converts this
            # matrix's only NATIVE-dispatch row into an executor row and deletes
            # the coverage it exists for. Nothing is lost by not reading the
            # report: completion is graded from `state.json` by
            # `test_the_phase_recorded_itself_completed`, and the outcome that
            # matters — whether bytes actually landed — is graded by the two
            # census assertions below, which is what a byte-level verdict is.
            stray = _write_report(result, _write_phase_dir(matrix_row))
            assert not stray, (
                f"{result.row.host}: a write_batch_report.json exists on the "
                f"NATIVE write row. Only WriteBatchScheduler writes that file and "
                f"it only runs under --exec-subagents/--dry-run-subagents, so the "
                f"host passed a flag its own shim tells it never to pass — this "
                f"row is no longer the native-dispatch row it is declared to be: "
                f"{stray.get('tasks')}")
            return
        report = _write_report(result, _write_phase_dir(matrix_row))
        assert report, (
            f"{result.row.host}: no write_batch_report.json under "
            f"{_write_phase_dir(matrix_row)}/ in the exported bundle — the write "
            f"scheduler either never dispatched a batch or never wrote its report")
        assert report.get("finalized") is True, (
            f"{result.row.host}: write_batch_report.json records "
            f"finalized={report.get('finalized')!r}. A run that dispatched every "
            f"batch and produced none of the finalization artifacts is a FAILED "
            f"run, not a partial one (stopped_at_batch="
            f"{report.get('stopped_at_batch')!r})")

    def test_every_batch_recorded_the_declared_outcome(self, workflow):
        result, matrix_row, entry = workflow
        declared = _write_outcome(matrix_row)
        if _fan_out_is_native(matrix_row, entry):
            pytest.skip(
                f"{matrix_row.get('id')}: `write_outcome` is the EXECUTOR's "
                f"per-task vocabulary and this row forbids the executor, so there "
                f"is no per-batch record to read — see the sibling assertion, "
                f"which owns the native case and grades the report's absence. "
                f"What `applied` means on this row is graded on the SHA census "
                f"below, by `test_the_bytes_moved_exactly_as_the_row_declares`.")
        report = _write_report(result, _write_phase_dir(matrix_row))
        recorded = {task.get("task_id"): task.get("write_outcome")
                    for task in report.get("tasks") or []}
        assert recorded, (
            f"{result.row.host}: the write report recorded no task at all, so "
            f"nothing was dispatched to grade")
        if declared == "applied":
            assert "applied" in recorded.values(), (
                f"{result.row.host}: every batch came back "
                f"{sorted(set(recorded.values()))} — dispatched, nothing landed. "
                f"This row is the plan's only applied-byte evidence through a real "
                f"host, so a no-op here is a failed row, never a green one")
            unsafe = {task: outcome for task, outcome in recorded.items()
                      if outcome in ("partial-application", "policy-violation")}
            assert not unsafe, (
                f"{result.row.host}: a batch stopped mid-application: {unsafe}. "
                f"The snapshots are kept — restore with `multi_llm.py util "
                f"restore-batch --phase-dir <dir> --task-id <id>`")
        else:
            wrong = {task: outcome for task, outcome in recorded.items()
                     if outcome != declared}
            assert not wrong, (
                f"{result.row.host}: expected every batch {declared!r} and got "
                f"{wrong}. An `applied` batch from a tool-call-free stub means the "
                f"redirect leaked, which fails this row exactly as loudly as a "
                f"batch that never ran")

    def test_the_bytes_moved_exactly_as_the_row_declares(self, workflow):
        """Whether these bytes changed and which paths moved — never their content."""
        result, matrix_row, _entry = workflow
        _write_outcome(matrix_row)
        expect = matrix_row.get("expect") or {}
        delta = _source_delta(result)
        if expect.get("bytes_must_change"):
            assert delta, (
                f"{result.row.host}: the SHA-256 census of /work/sample-project "
                f"came back byte-identical outside the workflow's own outputs, so "
                f"the native write row applied nothing")
        else:
            assert not delta, (
                f"{result.row.host}: bytes moved on a row declared "
                f"{_write_outcome(matrix_row)!r}: {sorted(delta)}")

    def test_no_write_landed_outside_the_declared_paths(self, workflow):
        result, matrix_row, _entry = workflow
        _write_outcome(matrix_row)
        declared = (matrix_row.get("expect") or {}).get("paths")
        if not declared:
            pytest.skip(f"{result.row.row_id}: the row declares no write paths")
        # The predicate itself lives in `harness.stray_paths` — same behaviour,
        # one importable place — because this assertion is `live`-marked beneath
        # this module's container gate and therefore only ever answers inside a
        # credentialed Docker run. `tests/test_stray_write_paths.py` pins it in
        # both directions on plain CI, where a PR that widens a `paths:` glob too
        # far is actually read.
        stray = harness.stray_paths(_source_delta(result), declared)
        assert not stray, (
            f"{result.row.host}: bytes moved outside the row's declared "
            f"{declared}: {stray}. A change outside the declared set fails the row "
            f"even when the expected file also changed")


class TestTheNegativeAuthRowsRefuse:
    """The seeded credential removed — and the two hosts do NOT fail alike.

    codex's `auth_check` fails **closed**: with its synthesized `auth.json` gone,
    `preflight_runner` raises `not_authenticated` and refuses the whole batch
    before the first stub call, so the row must fail within seconds with zero stub
    requests rather than on its wall-clock bound.

    opencode's did not when this row was first measured: a logged-out `opencode
    auth list` exits 0 printing "0 credentials", which matched none of the
    adapter's `auth_error_patterns`, so `classify_auth_probe` returned
    "authenticated" and the gate simply opened. Two things have moved since and
    neither has been re-recorded on this lane, which is why `host_matrix.yaml`
    still records the row as `exit: unknown, provenance: unmeasured` and why an
    unmeasured expectation is **recorded here, never asserted into a pass**:

    * `OpenCodeProvider.classify_auth_probe` now reads the count line
      (`_EMPTY_CREDENTIALS`): "0 credentials" with no Environment section is
      `False`, so the preflight is expected to refuse this row
      `not_authenticated` the way codex's does.
    * D3 step 2 exempts models the adapter declares credential-free
      (`credential_free_models`, today exactly `opencode/big-pickle`) from that
      refusal. This row's runner is `opencode:stub/stub-model` — a
      credential-*requiring* id by construction — so the exemption cannot turn
      the row into a tautology; `TestTheCredentialFreeDefaultPassesPreflight`
      below is the positive half, on the shipped default itself.

    When a re-recorded run promotes this row, `exit`, `provenance`, this
    docstring and the `test_the_refusal_was_named_and_fast` skip message move in
    the same edit. Both rows stay scheduled: a green codex negative proves
    nothing about opencode.
    """

    pytestmark = pytest.mark.live

    def test_no_request_reached_the_stub(self, graded):
        result, matrix_row, _entry = graded
        expect = _negative_expect(matrix_row)
        declared = expect.get("stub_requests")
        if declared is None:
            pytest.skip(
                f"{result.row.host}: this row declares no stub_requests — its gate "
                f"opens rather than refusing, so a call reaching the stub is the "
                f"documented behaviour and not a failure")
        # COUNTED FROM THE WATERMARK, NOT FROM ZERO. The row's cell means "zero
        # after the credential was removed" — the refusal lands before the first
        # model call — and this lane now runs a pre-command leg that fans out to
        # the stub *on purpose*, while the seed is still in place, through the
        # very runner this row exists to have refused. `entrypoint` writes the
        # boundary to `stub-watermark.json` at the moment the host process starts,
        # after the leg and after the removal; everything at or below it is the
        # rig's own traffic. Counting raw would grade the leg and make this cell
        # unsatisfiable the day the leg landed.
        calls = [r for r in result.stub_log if r.get("surface") != "health"]
        before = _stub_watermark(result)
        assert before is not None, (
            f"{result.row.host}: the container wrote no stub-watermark.json, so "
            f"the rig's own pre-command traffic cannot be told apart from the "
            f"host's and this row has nothing it can honestly count. The "
            f"watermark is written on every credentialed row immediately before "
            f"the host launches.")
        after = calls[before:]
        assert len(after) == int(declared), (
            f"{result.row.host}: {len(after)} request(s) reached the stub after "
            f"the credential was removed, expected {declared} "
            f"({before} earlier request(s) were the pre-command leg's and are not "
            f"counted). The refusal is supposed to land BEFORE the first model "
            f"call: {[r.get('surface') for r in after]}")

    def test_the_refusal_was_named_and_fast(self, graded):
        result, matrix_row, _entry = graded
        expect = _negative_expect(matrix_row)
        if expect.get("exit") == "unknown":
            pytest.skip(
                f"{result.row.host}: host_matrix.yaml records exit: unknown, "
                f"provenance: unmeasured for this row — {expect.get('reason')}")
        receipts = _mode_receipts(result, _row_mode(matrix_row))
        assert receipts, (
            f"{result.row.host}: no dispatcher receipt for the write mode, so the "
            f"row proves nothing about the gate: the command never ran at all")
        refusal = receipts[-1]
        assert refusal.get("exit") == int(expect["exit"]), (
            f"{result.row.host}: the dispatcher exited {refusal.get('exit')!r}, "
            f"expected {expect['exit']!r} (not_authenticated is exit 3, and a 0 "
            f"here means the batch was dispatched with no credential)")
        elapsed = (float(refusal.get("ended_at") or 0.0)
                   - float(refusal.get("started_at") or 0.0))
        within = float(expect.get("within_s") or 30)
        assert elapsed <= within, (
            f"{result.row.host}: refused after {elapsed:.1f}s, and the row "
            f"declares within {within:.0f}s. A refusal that slow did not come from "
            f"the pre-flight gate — it came from something downstream of it")
        reason = str(expect.get("reason") or "")
        assert reason in _transcript(result), (
            f"{result.row.host}: the transcript never names {reason!r}, so the row "
            f"failed for an unnamed reason. 'Refused' and 'broke' are different "
            f"outcomes and this row only passes on the first")

    def test_an_unmeasured_expectation_is_recorded_never_asserted(self, graded):
        """The honest half: evidence, filed, with no verdict invented for it."""
        result, matrix_row, _entry = graded
        expect = _negative_expect(matrix_row)
        if expect.get("exit") != "unknown":
            pytest.skip(f"{result.row.row_id}: this row's expectation is measured")
        receipts = _mode_receipts(result, _row_mode(matrix_row))
        assert receipts, (
            f"{result.row.host}: the row produced no dispatcher receipt, so it "
            f"recorded nothing — an unmeasured expectation still needs evidence "
            f"before it can ever become a measured one")
        calls = [r for r in result.stub_log if r.get("surface") != "health"]
        (result.out_dir / "negative-auth-observed.json").write_text(json.dumps({
            "row": matrix_row.get("id"),
            "host": result.row.host,
            "removed_seed": matrix_row.get("remove_seed"),
            "expect": {"exit": expect.get("exit"),
                       "reason": expect.get("reason"),
                       "provenance": expect.get("provenance")},
            "observed": {"dispatcher_exit": receipts[-1].get("exit"),
                         # Both halves, because "12 calls" and "12 calls, all of
                         # them the rig's" are different observations and only the
                         # second one says anything about the gate.
                         "stub_requests_total": len(calls),
                         "stub_requests_before_host": _stub_watermark(result),
                         "stub_requests_after_removal": (
                             None if _stub_watermark(result) is None
                             else len(calls) - _stub_watermark(result)),
                         "phases_completed": sorted(
                             result.workflow_state().get("phases_completed") or {})},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dispatcher_receipts(result) -> list[dict]:
    """Every receipt except the entrypoint's own staging invocation.

    `install-host` is the rig staging the shim, not the host running our
    dispatcher, so counting it would let a row pass on the harness's own work.
    """
    out = []
    for receipt in result.receipts:
        argv = receipt.get("argv") or []
        if "install-host" in argv:
            continue
        out.append(receipt)
    return out


def _mode_of(receipt: dict) -> str:
    """The mode word a receipt's argv dispatched, for a failure message."""
    argv = receipt.get("argv") or []
    for index, token in enumerate(argv):
        if token.endswith("multi_llm.py") and index + 1 < len(argv):
            return " ".join(argv[index + 1:index + 3])
    return "?"


def _mode_receipts(result, mode: str) -> list[dict]:
    """Every receipt whose argv dispatched `mode` itself.

    `util instructions <mode>` names the same word one argv slot later, and it is
    the read step rather than the dispatch — so the mode word is matched at its
    own position instead of anywhere in the argv.
    """
    return [receipt for receipt in _dispatcher_receipts(result)
            if _mode_of(receipt).split(" ")[0] == mode]


def _write_outcome(matrix_row: dict) -> str:
    """The row's declared `expect.write_outcome`, skipping every non-write row."""
    outcome = (matrix_row.get("expect") or {}).get("write_outcome")
    if not outcome:
        pytest.skip(f"{matrix_row.get('id')}: not a write row (no "
                    f"expect.write_outcome in host_matrix.yaml)")
    return str(outcome)


#: The phase name a write row declares -> the DIRECTORY that phase writes into.
#: Three of the four are the same word and the fourth is not: mode
#: `apply-code-fixes` records phase `apply-code-fixes` in `state.json` and in its
#: receipt payload, while `apply_code_fixes_orchestrator.phase_name` is
#: `apply-fixes`, so the artifacts land one directory over from the name the row
#: declares. Kept in step with `test_container_full_chain.WRITE_PHASE_DIRS`,
#: which maps the same four for the provider lane.
WRITE_PHASE_DIRS = {
    "apply-suggestions": "apply-suggestions",
    "apply-task-suggestions": "apply-task-suggestions",
    "implement": "implement",
    "apply-code-fixes": "apply-fixes",
}


def _write_phase_dir(matrix_row: dict) -> str:
    """Where this row's write artifacts land, not what the row calls the phase.

    Looked up rather than assumed: a row declaring `phases: [apply-code-fixes]`
    that globbed `*/apply-code-fixes/` would find nothing and report "the phase
    did not schedule anything" about a phase that scheduled everything.
    """
    phases = (matrix_row.get("expect") or {}).get("phases") or []
    phase = str(phases[0]) if phases else "apply-suggestions"
    return WRITE_PHASE_DIRS.get(phase, phase)


def _row_mode(matrix_row: dict) -> str:
    """The dispatcher mode word this row's `command` names.

    Read from the row rather than hardcoded, so a retargeted row (D3 moved the
    native write row from `apply-suggestions` to `apply-code-fixes`) takes its
    grader with it instead of quietly grading a mode it no longer runs.
    """
    command = str(matrix_row.get("command") or "").split()
    return command[0] if command else "apply-suggestions"


def _negative_expect(matrix_row: dict) -> dict:
    """The row's `expect`, skipping every row that keeps its seeded credential."""
    if not matrix_row.get("remove_seed"):
        pytest.skip(f"{matrix_row.get('id')}: not a negative auth row (it removes "
                    f"no seeded credential)")
    return matrix_row.get("expect") or {}


def _stub_watermark(result) -> int | None:
    """Non-health stub requests already logged when the host process started.

    `None` means the container wrote no watermark, which is a missing-evidence
    condition rather than a zero: reading it as 0 would silently credit the rig's
    own pre-command fan-out to the host.
    """
    path = result.out_dir / "stub-watermark.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get(
        "non_health_calls_before_host")
    return None if value is None else int(value)


def _fan_out_is_native(matrix_row: dict, entry: dict) -> bool:
    """Whether this row's batch is dispatched by the HOST rather than by us.

    Only true of the reference host's write row. Its subagent batch never crosses
    `/opt/mllm/bin` — that is what makes it real spend and what makes it the one
    row where `applied` bytes are a pass — so "every model call landed on the
    stub" has nothing to grade there. The ask and review-plan rows on the same
    host still fan out to provider CLIs by name, and are graded normally.
    """
    return (((matrix_row.get("expect") or {}).get("write_outcome") == "applied")
            and (entry.get("continuation") or {}).get("who") == "host")


def _write_report(result, phase: str) -> dict:
    """`{phase}/write_batch_report.json` as exported into the bundle."""
    root = result.out_dir / "artifacts"
    for candidate in sorted(root.glob(f"*/{phase}/write_batch_report.json")):
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _source_delta(result) -> dict:
    """The workspace's SHA census delta with the WORKFLOW's own outputs removed.

    A literally byte-identical census is not achievable and asserting one would
    fail every passing row: each row writes its phase artifacts, `state.json` and
    the tracking back-reference under `plans/`, and its run registry under
    `.multi-llm/`. What the write rows are graded on is the change set outside
    that — the bytes a write subagent moved, by SHA, never by content.
    """
    return {name: pair for name, pair in result.census_delta().items()
            if not name.startswith("plans/") and not name.startswith(".multi-llm/")}


def _transcript(result) -> str:
    """Everything the row said: the host's own streams plus the container's.

    The refusal's reason is printed prose, not a receipt field — the wrapper keeps
    the `MULTI_LLM_RESULT` payload and tees the rest to the host's stdout — so a
    named refusal is looked for where it actually lands.
    """
    chunks = []
    for name in ("host-stdout.txt", "host-stderr.txt",
                 "container-stdout.txt", "container-stderr.txt"):
        path = result.out_dir / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    # AND THE DETACHED WORKER'S OWN LOG. On a detaching row the dispatcher's
    # prose never reaches the host's stdout at all: `spawn_detached` hands the
    # worker `{phase}/orchestrator-run.log` as its stdout/stderr, which is exactly
    # where each executor host's `result_source` cell already says its verdict
    # comes from. Without
    # this, "the row failed for a named reason" would be a test of whether the
    # agent happened to run it in the foreground — and no `args` string can
    # decide that, because the shim's own step 5 adds `--detach`.
    for log in sorted((result.out_dir / "artifacts").rglob("orchestrator-run.log")):
        chunks.append(log.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _exported(result, artifact: str):
    """Map a container-absolute artifact path onto the exported bundle.

    `None` means one thing only: the artifact does not live under the plan dir
    (a receipt may legitimately name a path under `$HOME`), so the bundle says
    nothing about it. It must NEVER mean "the export produced nothing here" —
    gating the return on the parent directory existing did exactly that, turning
    a wholly failed export into a vacuous pass, since the caller reads `None` as
    "not our business" and moves on. A `/plans/` path therefore always maps to
    its expected location and the caller's `is_file()`/size check is what
    decides, present or absent.
    """
    marker = "/plans/"
    if marker not in artifact:
        return None
    tail = artifact.split(marker, 1)[1]
    return result.out_dir / "artifacts" / tail


# --------------------------------------------------------------------------- #
# D3 step 6 — the standing drift check: the `opencode/*` ids this repo ships
# against Zen's live catalog. Recurs on every opencode release.
# --------------------------------------------------------------------------- #

_SKILL_DIR = Path(__file__).resolve().parent.parent


def _shipped_opencode_ids() -> dict:
    """Every `opencode/*` id the shipped config and host data name, by source."""
    shipped: dict = {}
    providers = yaml.safe_load(
        (_SKILL_DIR / "providers.yaml").read_text(encoding="utf-8")
    )
    for model in (providers.get("providers", {}).get("opencode", {}).get("models") or []):
        if isinstance(model, str) and model.startswith("opencode/"):
            shipped.setdefault(model, set()).add("providers.yaml")
    caps = yaml.safe_load(
        (_SKILL_DIR / "hosts" / "capabilities.yaml").read_text(encoding="utf-8")
    )
    entry = (caps.get("hosts") or caps).get("opencode") or {}
    spec = str(entry.get("default_subagent_runner") or "")
    provider, _, model = spec.partition(":")
    if provider == "opencode" and model.startswith("opencode/"):
        shipped.setdefault(model, set()).add("hosts/capabilities.yaml")
    return shipped


def _live_opencode_listing():
    """`opencode models` on this lane's CLI, or a named skip when it cannot answer.

    An absent CLI or an unreachable catalog is a skip, never a failure — the
    same `known=None` rule the preflight's catalog leg pins: "not in an empty
    list" is not a fact.
    """
    binary = shutil.which("opencode")
    if not binary:
        pytest.skip("opencode is not installed on this lane")
    try:
        completed = subprocess.run(
            [binary, "models"], capture_output=True, text=True, timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"`opencode models` could not run: {exc}")
    from utils.providers.opencode import _parse_model_listing

    listing = _parse_model_listing(completed.stdout or "")
    if completed.returncode != 0 or listing is None:
        pytest.skip(
            f"`opencode models` exited {completed.returncode} without a readable "
            f"listing (catalog unreachable?): {(completed.stderr or '')[:200]!r}"
        )
    return listing


class TestTheShippedOpencodeIdsAreStillListed:
    """Zen's catalog is server-side: `opencode/sonnet`,
    `opencode/deepseek-v4-flash-free` and `opencode/hy3-free` all vanished between
    releases while the config still named them, and the host default was one of
    them. A hermetic CI run cannot see the catalog and must not try, so the diff
    lives on the lane that already runs real CLIs and is meant to run on a
    schedule. A delta fails LOUDLY, naming the id and the file that ships it —
    the runtime answer (`invalid_model_catalog` at preflight, D4 at dispatch) is
    what a user hits; this is what notices before a user does.
    """

    pytestmark = pytest.mark.live

    def test_every_shipped_opencode_id_is_in_the_live_catalog(self):
        shipped = _shipped_opencode_ids()
        assert shipped, "no opencode/* id is shipped at all — the check is vacuous"
        listing = _live_opencode_listing()
        missing = {
            model: sorted(sources)
            for model, sources in shipped.items() if model not in listing
        }
        assert not missing, (
            f"shipped opencode/* id(s) missing from `opencode models` on "
            f"opencode {_opencode_version()}: {missing}. Re-run the dated "
            f"logged-out probe in hosts/capabilities.yaml (the bump ritual), "
            f"replace or drop each id, and re-render the generated docs."
        )


def _opencode_version() -> str:
    binary = shutil.which("opencode")
    if not binary:
        return "unknown"
    try:
        return (subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL,
        ).stdout or "").strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TestTheCredentialFreeDefaultPassesPreflight:
    """The positive half of D3 step 2, on a clean environment.

    Every existing opencode row on this lane is credentialed or negative, and
    neither shape can show a credential-free default *completing*. `status`
    structurally cannot see it either — it spawns no probe. So this drives the
    executor's own `preflight_runner` on the shipped host default from a
    zero-credential HOME/XDG pair with every provider `*_API_KEY` unset, in a
    child interpreter (the caches are per process and the redirect is the
    environment), and requires it to COMPLETE rather than exit 3: the auth leg
    must read `not-required` and the catalog leg `ok`. The same child then
    preflights a credential-requiring id on the same CLI and must be refused
    `not_authenticated` — the exemption is narrow or it is a hole.
    """

    pytestmark = pytest.mark.live

    _CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
from utils import subagent_executor as executor
from utils import host_capabilities as caps
spec = caps.entry("opencode")["default_subagent_runner"]
out = {"spec": spec}
try:
    out["info"] = executor.preflight_runner(spec, key="driver.subagent_runner (host capabilities)")
except executor.RunnerRejected as exc:
    out["refused"] = {"reason_code": exc.reason_code, "message": str(exc)}
try:
    executor.preflight_runner("opencode:openrouter/deepseek/deepseek-v4-flash-0731")
    out["gated"] = "dispatched"
except executor.RunnerRejected as exc:
    out["gated"] = exc.reason_code
print(json.dumps(out))
"""

    def test_the_shipped_default_completes_preflight_with_zero_credentials(
        self, tmp_path
    ):
        if not shutil.which("opencode"):
            pytest.skip("opencode is not installed on this lane")
        env = {
            k: v for k, v in os.environ.items()
            if not k.endswith("_API_KEY") and k != "MULTI_LLM_PROVIDERS_CONFIG"
        }
        for name in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            target = tmp_path / name.lower()
            target.mkdir()
            env[name] = str(target)
        # XDG_CACHE_HOME is deliberately NOT redirected (models.json lives there).
        completed = subprocess.run(
            [sys.executable, "-c", self._CHILD, str(_SKILL_DIR)],
            capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path),
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        assert "refused" not in result, (
            f"the shipped default {result['spec']!r} was refused on a "
            f"zero-credential install: {result['refused']}"
        )
        info = result["info"]
        assert info["auth"] == "not-required", info
        assert info["model_catalog"] == "ok", info
        assert result["gated"] == "not_authenticated", result
