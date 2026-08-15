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
"""

from __future__ import annotations

import fnmatch
import json
import os

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
        """The assertion, where the host offers a credential-free way to ask."""
        result, census, entry = discovery
        form = (entry.get("list_step") or {}).get("form")
        if form != "debug-config":
            pytest.skip(
                f"{result.row.host}: no credential-free listing surface exists "
                f"(measured — `doctor` included), so discovery here needs a "
                f"prompted turn and a credential. Reported as "
                f"'{census['discovery']}' in the census rather than as a pass."
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
]

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
    negative = bool(matrix_row.get("remove_seed"))
    row = harness.Row(
        row_id=row_id, lane="host", host=host, creds="openrouter",
        helpers=sorted({"codex", host} if negative else {"codex"}),
        spec="codex:stub-model",
        runner=f"{host}:stub-model" if negative else "codex:stub-model",
        row_timeout_s=int(matrix_row.get("wall_clock_s") or 600) + 500,
        # Without it the worker really runs and really fails: a scenario-less
        # stub answers a bare "OK", which is not a suggestions document, and the
        # row grades the stub's default answer instead of the host.
        scenario=matrix_row.get("scenario"),
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
        result, matrix_row, _entry = workflow
        expect = matrix_row.get("expect") or {}
        phases = expect.get("phases") or []
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
        `{phase}/.status.json`'s in-phase `state`. Measured on both executor
        hosts: a review-plan row that ran to a terminal `run.state: done` with
        `validation.json` and the report on disk, and `state.json` recording the
        phase completed, still leaves `.status.json` reading
        `state: grouping_complete` — the in-phase checkpoint is the last
        checkpoint the fan-out wrote, not a completion flag. Asserting on it
        would fail a passing row.
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
        result, matrix_row, _entry = workflow
        _write_outcome(matrix_row)           # skips every non-write row by name
        report = _write_report(result, _write_phase(matrix_row))
        assert report, (
            f"{result.row.host}: no write_batch_report.json under "
            f"{_write_phase(matrix_row)}/ in the exported bundle — the write "
            f"scheduler either never dispatched a batch or never wrote its report")
        assert report.get("finalized") is True, (
            f"{result.row.host}: write_batch_report.json records "
            f"finalized={report.get('finalized')!r}. A run that dispatched every "
            f"batch and produced none of the finalization artifacts is a FAILED "
            f"run, not a partial one (stopped_at_batch="
            f"{report.get('stopped_at_batch')!r})")

    def test_every_batch_recorded_the_declared_outcome(self, workflow):
        result, matrix_row, _entry = workflow
        declared = _write_outcome(matrix_row)
        report = _write_report(result, _write_phase(matrix_row))
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
        stray = sorted(name for name in _source_delta(result)
                       if not any(fnmatch.fnmatch(name, pattern)
                                  for pattern in declared))
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

    opencode's does not, and that is the one measured result that contradicted the
    plan: a logged-out `opencode auth list` exits 0 printing "0 credentials", which
    matches none of the adapter's `auth_error_patterns`, so `classify_auth_probe`
    returns "authenticated" and the gate simply opens. `host_matrix.yaml` records
    that row as `exit: unknown, provenance: unmeasured`, and an unmeasured
    expectation is **recorded here, never asserted into a pass** — which is the
    whole reason both rows are scheduled: a green codex negative proves nothing
    about opencode.
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
        calls = [r for r in result.stub_log if r.get("surface") != "health"]
        assert len(calls) == int(declared), (
            f"{result.row.host}: {len(calls)} request(s) reached the stub, "
            f"expected {declared}. The refusal is supposed to land BEFORE the "
            f"first model call: {[r.get('surface') for r in calls]}")

    def test_the_refusal_was_named_and_fast(self, graded):
        result, matrix_row, _entry = graded
        expect = _negative_expect(matrix_row)
        if expect.get("exit") == "unknown":
            pytest.skip(
                f"{result.row.host}: host_matrix.yaml records exit: unknown, "
                f"provenance: unmeasured for this row — {expect.get('reason')}")
        receipts = _mode_receipts(result, "apply-suggestions")
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
        receipts = _mode_receipts(result, "apply-suggestions")
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
                         "stub_requests": len(calls),
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


def _write_phase(matrix_row: dict) -> str:
    phases = (matrix_row.get("expect") or {}).get("phases") or []
    return str(phases[0]) if phases else "apply-suggestions"


def _negative_expect(matrix_row: dict) -> dict:
    """The row's `expect`, skipping every row that keeps its seeded credential."""
    if not matrix_row.get("remove_seed"):
        pytest.skip(f"{matrix_row.get('id')}: not a negative auth row (it removes "
                    f"no seeded credential)")
    return matrix_row.get("expect") or {}


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
