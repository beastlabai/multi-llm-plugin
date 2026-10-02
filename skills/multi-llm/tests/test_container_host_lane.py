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

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
from .container import host_spend
from .container import next_command_check

pytestmark = [pytest.mark.container, requires_docker()]

HOST_MATRIX = harness.HERE / "host_matrix.yaml"

#: The `MULTI_LLM_RESULT.state` values that mean an invocation *finished*, as
#: opposed to the non-terminal ones a poll legitimately reports mid-run
#: (`nothing-to-do` from a `status` that found the worker still running,
#: `needs-human`, `error`). `partial` is terminal and is emitted at **exit 0** —
#: D6 row 3's partial aggregate — which is precisely why membership here is not on
#: its own evidence that a row succeeded, and why `expect.exit` is graded too.
TERMINAL_STATES = ("done", "complete", "partial")

#: The phases whose dispatch IS the model fan-out, so an assigned provider has
#: something to be invoked BY. Every other phase — the write phases and
#: `generate-tasks` — sends each batch through the single
#: `driver.subagent_runner` at `max_concurrent_write_subagents: 1`
#: (`subagent_batch_report.json`'s `runner_versions` carries exactly one entry
#: on every recorded write phase), and its phase directory holds batch files
#: rather than a per-provider result file. Grading a partition over one of them
#: would be grading the runner, twice removed.
FAN_OUT_PHASES = ("ask", "review-plan", "review-tasks", "code-review")


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
        # NAMED RATHER THAN DEFAULTED, even though this tier makes no model
        # call at all. The entrypoint used to fall back to a hard-coded
        # `codex:stub-model` for any row that passed no `--spec`, which is one
        # of the two places the whole lane's single-provider fan-out was
        # written down implicitly. The fallback is gone, so the inert value a
        # credential-free row renders is stated here instead of inherited from
        # a literal nobody was reading. `opencode` is the lane's declared write
        # runner (T0a); on this tier it is never dispatched.
        spec=["opencode:stub/stub-model"],
        runner="opencode:stub/stub-model",
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
    # ...and the two rows that take the package route from 2 of the file
    # route's 5 cells to 4 of 5. The package route is how opencode actually
    # ships now, so grading it on the cheap half while the file route gets the
    # write rows is the weaker half being graded. Derived cell-for-cell from
    # their file-route twins; every measured cell on them is `unmeasured`
    # until T4's first pass.
    ("opencode", "opencode-package-full"),
    ("opencode", "opencode-package-apply-executor"),
    # THE CODEX PACKAGE ROUTE's one credentialed row (plans/codex-native-
    # plugin.md, Phase 1 item 16). It mirrors `opencode-package-ask` argument
    # for argument; its zero-quota prerequisites are NOT here — they are the
    # `package_probes:` list, scheduled by the `package_probe` fixture below at
    # `--creds none`. Last, like the rows above it: newest, and the cheapest to
    # lose if the lane budget runs out.
    ("codex", "codex-package-ask"),
    # ------------------------------------------------------------------- #
    # THE claude-code PER-PHASE TIER, and it is deselected by default.
    #
    # Each row is one workflow phase, its prerequisites seeded from the rig by
    # a `pre_commands` leg so the host turn does exactly one phase natively.
    # That is `claude-code-apply-native`'s own shape — the only row in the
    # repository where applied bytes on a real host are a pass — generalized
    # along the phase axis.
    #
    # THREE ROWS, NOT EIGHT, AND THE SHORTFALL IS RECORDED RATHER THAN
    # MISSING. Of the eight chain phases, two already exist under historical
    # names (`apply-code-fixes` IS `claude-code-apply-native`, `review-plan`
    # IS `claude-code-review-plan`), and the four READ phases are not
    # authored at all: each stops at `[VALIDATION_PENDING]` and requires the
    # host to re-invoke `--reaggregate`, which this host measurably failed to
    # do on both model attempts. That failure is INTERNAL to the phase, so
    # seeding phases 1..n-1 does nothing for it — four rows built on that
    # shape would be four rows that dispatch and never complete, at
    # ~$0.25-0.60 each. They are dated `not-applicable` cells in
    # `host_matrix.yaml` instead, citing T4a as the task that would clear
    # them.
    #
    # `pytest.param(..., marks=...)` is the one mechanism that marks an
    # individual fixture param, and the `ids=` list below is DERIVED from
    # these params rather than restated beside them — two hand-kept lists
    # drift apart silently.
    pytest.param(("claude-code", "claude-code-phase-apply-suggestions"),
                 marks=pytest.mark.live_native_phase),
    pytest.param(("claude-code", "claude-code-phase-implement"),
                 marks=pytest.mark.live_native_phase),
    pytest.param(("claude-code", "claude-code-phase-apply-task-suggestions"),
                 marks=pytest.mark.live_native_phase),
    # ------------------------------------------------------------------- #
    # T10 OF plans/next-phase-command.md — THE NEXT-COMMAND LIVE CHECK, one
    # row per host and route the lane can run, and deselected by default.
    #
    # Each row runs `generate-tasks` on a fixture plan whose path holds a
    # space, then hands the command its dispatcher printed to a FRESH session
    # of the same host and route (`entrypoint.run_typed_followup`). Graded by
    # `TestTheNextCommandLiveCheck`; the generic classes above grade the phase
    # itself like any other row. A tier of its own (`live_next_command`,
    # `make e2e-container-host-next-command`) so the ordinary target's cost
    # does not move, and so the check can be re-run alone after a template
    # or instruction edit.
    pytest.param(("codex", "codex-next-command"),
                 marks=pytest.mark.live_next_command),
    pytest.param(("codex", "codex-package-next-command"),
                 marks=pytest.mark.live_next_command),
    pytest.param(("opencode", "opencode-next-command"),
                 marks=pytest.mark.live_next_command),
    pytest.param(("opencode", "opencode-package-next-command"),
                 marks=pytest.mark.live_next_command),
    pytest.param(("claude-code", "claude-code-next-command"),
                 marks=pytest.mark.live_next_command),
]


def _row_pair(param):
    """The `(host, row_id)` inside a bare tuple or a `pytest.param`.

    `CREDENTIALED_ROWS` holds both shapes on purpose: a plain tuple where the
    row needs no marker, and `pytest.param(..., marks=...)` where it does —
    that being the one mechanism that marks an individual fixture param.
    Every consumer that wants the pair goes through here rather than
    unpacking, because a `pytest.param` is not a 2-tuple and unpacking it
    raises at import.
    """
    values = getattr(param, "values", None)
    return values[0] if values is not None else param


#: Derived, never restated: `ids=` and `params=` must not be two hand-kept
#: lists. A marked param carries its value in `.values[0]`.
#:
#: The id stays the BARE row id, because `-k '[<row id>]'` — brackets included —
#: is the documented per-row selector and a bare substring once matched the
#: `skipif` keyword every item carries and silently ran the whole lane at full
#: cost (`Makefile:402`, measured 2026-08-19).
CREDENTIALED_PAIRS = [_row_pair(param) for param in CREDENTIALED_ROWS]
CREDENTIALED_IDS = [row for _host, row in CREDENTIALED_PAIRS]

#: The row ids the per-phase tier owns, for the census's tier accounting and
#: for the two collection guards. Read off the marks rather than off a name
#: pattern, so a row that loses its marker is caught by the guard rather than
#: silently rejoining the cheap target.
NATIVE_PHASE_ROWS = [
    _row_pair(param)[1] for param in CREDENTIALED_ROWS
    if any(getattr(m, "name", "") == "live_native_phase"
           for m in getattr(param, "marks", ()))
]

#: The T10 next-command rows (`live_next_command`), read off the marks the same
#: way and for the same reason.
NEXT_COMMAND_ROWS = [
    _row_pair(param)[1] for param in CREDENTIALED_ROWS
    if any(getattr(m, "name", "") == "live_next_command"
           for m in getattr(param, "marks", ()))
]

#: The rows the ordinary `make e2e-container-host` target keeps.
PROVIDER_TIER_ROWS = [row for row in CREDENTIALED_IDS
                      if row not in NATIVE_PHASE_ROWS
                      and row not in NEXT_COMMAND_ROWS]


def _invoked_tier(config) -> str:
    """Which tier this invocation selected, read off pytest's own `-m` value.

    `make e2e-container-host` types
    `-m "container and live and not live_native_phase"` and
    `make e2e-container-host-phases` types
    `-m "container and live and live_native_phase"`. Read here rather than
    guessed, because the census's whole job is to make an empty lane
    legible, and a census that reports the wrong denominator is worse than
    none.

    **`config.getoption("markexpr")`, never `sys.argv`.** `markexpr` is the
    expression pytest actually selected with, after it has merged
    `PYTEST_ADDOPTS` and the ini `addopts` into the command line. Scraping
    `sys.argv` sees only the expression someone typed on the line itself, so
    this project's own `addopts = -m 'not container and not package'`, an
    expression exported through `PYTEST_ADDOPTS`, or any wrapper that supplies
    `-m` some other way would be invisible — and the census would then report
    the wrong denominator, which is the very "unexplained gap between
    `declared` and `ran + skipped`" the tier split exists to remove. The
    config is threaded in from the fixture or hook that has it rather than
    cached in a module global, so there is no window in which it is unset.
    """
    expression = config.getoption("markexpr", default="") or ""
    # `make e2e-container-host-next-command` types
    # `-m "container and live and live_next_command"`; the ordinary target's
    # expression names the same mark only to exclude it.
    if ("live_next_command" in expression
            and "not live_next_command" not in expression):
        return "next-command"
    if "not live_native_phase" in expression:
        return "provider"
    if "live_native_phase" in expression:
        return "native-phase"
    return "all"


def _declared_for_tier(config) -> list[str]:
    tier = _invoked_tier(config)
    if tier == "provider":
        return list(PROVIDER_TIER_ROWS)
    if tier == "native-phase":
        return list(NATIVE_PHASE_ROWS)
    if tier == "next-command":
        return list(NEXT_COMMAND_ROWS)
    return list(CREDENTIALED_IDS)


def _deselected_for_tier(config) -> list[str]:
    """Named, never left as an unexplained gap.

    `declared - (ran + skipped)` with no explanation reads as rows that
    silently did not happen, which is precisely the "everything skipped looks
    like everything passed" outcome this census exists to prevent.
    """
    declared = set(_declared_for_tier(config))
    return [row for row in CREDENTIALED_IDS if row not in declared]


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
        scheduled = set(CREDENTIALED_PAIRS)
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


#: THE SPEND GUARD, in `container/host_spend.py` so it is unit-tested offline
#: (`tests/test_host_lane_budget.py`). Three limits: the key's own `limit`
#: (the only hard one — OpenRouter enforces it), the lane total
#: `MLLM_E2E_BUDGET_USD` (between rows, and mid-row as remaining headroom), and
#: each row's `cost_budget_usd × MLLM_E2E_ROW_BUDGET_HEADROOM`, enforced mid-row
#: by a poller that STOPS the row's container. The readback fails CLOSED: a
#: usage that cannot be read refuses the next row instead of crediting $0.
BUDGET_USD = host_spend.lane_budget_usd()


def _key_usage() -> float:
    """The dedicated key's own `usage`, or `host_spend.SpendUnobservable`.

    Attributable to one row precisely because the key is dedicated and the rows
    run one at a time. It used to return `None` on a network error and the row
    was credited $0 — the lane then started the next paid row having measured
    none of this one. It now retries a bounded number of times and raises.
    """
    return host_spend.read_usage()


class _Spend:
    """The running total the sequential checkpoint consumes."""

    ledger = host_spend.Ledger(BUDGET_USD)

    @classmethod
    def check(cls) -> None:
        refusal = cls.ledger.refusal()
        if refusal:
            pytest.exit(refusal, returncode=1)


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
    def record_ran(cls, row_id: str, host: str, mode: str = "openrouter",
                   attempts: int = 1) -> None:
        """File one row that ran, and how many attempts it took to get there.

        `attempts` is written only when it is not 1, so a lane with no retry in
        it renders exactly the census it always did — and a lane with one in it
        cannot render a census that hides it. This is the flag half of
        "pass with a flag": the row counts as coverage, and the count says what
        it cost to get.
        """
        entry = {"row": row_id, "host": host, "credential_mode": mode}
        if attempts > 1:
            entry["attempts"] = attempts
        cls.ran.append(entry)

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
    def report(cls, config) -> str:
        counts = cls.counts()
        lines = [
            f"  status  : {cls.status()}",
            f"  declared: {len(_declared_for_tier(config))} credentialed row(s) "
            f"in the {_invoked_tier(config)} tier "
            f"({len(_deselected_for_tier(config))} deselected)",
            f"  counts  : openrouter={counts['openrouter']} "
            f"fallback={counts['fallback']} skipped={counts['skipped']}",
        ]
        lines += [f"  ran     : {row['row']} ({row['credential_mode']})"
                  + (f" — RETRIED, graded on attempt {row['attempts']} of "
                     f"{row['attempts']}" if row.get("attempts", 1) > 1 else "")
                  for row in cls.ran]
        lines += [f"  skipped : {row['row']} — {row['reason']}"
                  for row in cls.skipped]
        return "\n".join(lines)

    @classmethod
    def write(cls, config, utc: str):
        """Beside the run bundle, so the count survives the pytest scrollback.

        Takes the pytest `Config` because the tier it records is read off
        `-m` (see `_invoked_tier`), and every caller here already has one.
        """
        path = harness.RUNS / utc / "host-lane-census.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "lane": "host",
            "tier": "credentialed",
            "status": cls.status(),
            # The census records the TIER it was invoked for. After the
            # split, counting all of CREDENTIALED_ROWS would have the
            # ordinary target report three declared rows that neither ran nor
            # skipped, and the phase target report the other twenty-one —
            # an unexplained gap between `declared` and `ran + skipped`.
            "tier_invoked": _invoked_tier(config),
            "declared_rows": _declared_for_tier(config),
            "deselected_rows": _deselected_for_tier(config),
            "counts": cls.counts(),
            "ran": cls.ran,
            "skipped": cls.skipped,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _rows_still_runnable(config) -> bool:
    """True when some declared row of this tier has a credential to run under.

    Per row, as `graded` decides it: a subscription (`fallback`) row needs no key,
    an `openrouter` row needs `OPENROUTER_API_KEY`. Decided 2026-10-02 with the
    keyless subscription run: `make e2e-container-host` no longer demands the key.
    """
    declared = set(_declared_for_tier(config))
    have_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    return any(harness.default_host_creds(host) == "fallback" or have_key
               for host, row in CREDENTIALED_PAIRS if row in declared)


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
    path = _Census.write(config, utc)
    if _Census.ran:
        return
    # A keyless run on the subscription skips the opencode rows (the first rows
    # in the schedule) BEFORE any codex or claude-code row has run, so "nothing
    # ran yet" is not "nothing will run": refuse only once no declared row is
    # left that its own credential mode could execute.
    if _rows_still_runnable(config):
        return
    if config.getoption("allow_empty_host_lane", default=False):
        return
    pytest.exit(
        "host lane: NOT-RUN — 0 of "
        f"{len(_declared_for_tier(config))} declared credentialed rows executed, so this "
        "lane covered nothing. That is a distinct outcome from a pass, and it is "
        "reported as one.\n" + _Census.report(config) + f"\n  census  : {path}\n"
        "  Pass --allow-empty-host-lane to accept a credential-free invocation "
        "of this lane (the census is printed either way), or run the free tier: "
        "make e2e-container-host-discovery", returncode=1)


#: The credential mode the credentialed rows run under, PER HOST since the
#: operator decision of 2026-10-02 (host_matrix.yaml's MODEL POLICY block).
#: `fallback` is C6a's subscription / vendor-auth mode (`creds.yaml`'s `:ro`
#: mounts, copied into the tmpfs $HOME by the entrypoint), which forwards NO
#: OpenRouter variable into the container (`harness.build_run_argv`) and drives
#: the host's `subscription.model` pin; `openrouter` is the dedicated,
#: spend-capped key. codex and claude-code take `fallback` whenever their
#: allowlisted credential file is present, else `openrouter`; opencode has no
#: subscription pin and always takes `openrouter` (`harness.default_host_creds`).
#: `MLLM_E2E_HOST_CREDS` forces one mode on every row. When the OpenRouter key is
#: also set, the fixture still reads its usage around a fallback row, so one that
#: somehow spent on the key shows up as a non-zero delta rather than as nothing.
HOST_CREDS_ENV = harness.HOST_CREDS_ENV


def _host_lane_creds(host: str) -> str:
    value = harness.default_host_creds(host)
    if value not in ("openrouter", "fallback"):
        pytest.fail(f"{HOST_CREDS_ENV}={value!r}: the credentialed rows run under "
                    f"`openrouter` or `fallback`, nothing else")
    return value


@pytest.fixture(scope="module", params=CREDENTIALED_ROWS,
                ids=CREDENTIALED_IDS)
def graded(request, posture, utc):
    """Run one credentialed row and return (result, row entry, host entry).

    **Strictly sequential, with a budget checkpoint between rows.** The matrix is
    never parallelized: the checkpoint is the closest thing to enforcement our own
    code can honestly offer, since the only limit that actually holds is the one
    enforced by the party doing the billing.
    """
    host, row_id = request.param
    # THE MODE FIRST: a subscription row needs no OpenRouter key at all, so the
    # key is required only of a row that will spend on it.
    creds = _host_lane_creds(host)
    have_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if creds == "openrouter" and not have_key:
        # A named skip, a census entry, and — unless the operator asked for an
        # empty lane — a non-zero `not-run`, because a lane that executed nothing
        # must never read as a lane that passed everything.
        _Census.record_skipped(row_id, host, "OPENROUTER_API_KEY unset")
        _refuse_empty_lane(request.config, utc)
        pytest.skip("the credentialed host lane needs OPENROUTER_API_KEY (a "
                    "DEDICATED key carrying a key-level spend limit)")
    _Spend.check()

    entry = dict(dict(HOSTS)[host])
    matrix_row = next((dict(r) for r in entry.get("rows") or []
                       if r.get("id") == row_id), None)
    assert matrix_row, f"host_matrix.yaml names no row {row_id!r} under {host!r}"
    # Refused BEFORE a token is spent: a row with no numeric budget has nothing
    # to stop it but the lane total. `test_host_lane_budget.py` catches the same
    # thing in plain CI; this is the run-time half for a matrix edited since.
    try:
        host_spend.row_ceiling(matrix_row)
    except ValueError as exc:
        pytest.fail(f"host lane: {exc}. Declare `cost_budget_usd` on the row.")
    # FAIL CLOSED on the opening readback: nothing has been spent on this row
    # yet, so stopping the lane here costs nothing and brackets nothing wrongly.
    # A subscription row with no key in the environment has no OpenRouter spend
    # to bracket: the vendor's own quota is its ceiling, which this rig cannot
    # read (the entrypoint's credential census says so).
    before = None
    if have_key:
        try:
            before = _key_usage()
        except host_spend.SpendUnobservable as exc:
            pytest.exit(f"host lane: refusing to start {row_id} — {exc}",
                        returncode=1)

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
    host_entry = entry

    # THE FAN-OUT, READ FROM THE MATRIX RATHER THAN HARD-CODED. Until 2026-09-22
    # both `spec=` and `runner=` below were the literal `"codex:stub-model"`, for
    # every row on every host, and `host_matrix.yaml` declared neither key
    # anywhere. The consequence was not small: 15 of the 16 stub-capable provider
    # CLIs never appeared in this lane at all, and a reader of "three hosts,
    # sixteen providers" would not have guessed it.
    #
    # Widening it is free. The lane's real spend is the HOST's own reasoning
    # turns; the fan-out runs real provider CLIs redirected at the stub through
    # `/opt/mllm/bin` and costs nothing. `defaults.models` was always a list and
    # only happened to render with one entry, so N members add N stub-backed
    # provider subprocesses INSIDE THE SAME host turn — no extra host turn, no
    # extra credit. (That is the hypothesis T3's one-row two-spec spike
    # measures; if host spend moves with fan-out width it is not free and the
    # partition is re-scoped before it is widened.)
    #
    # Slugs are built here, from `redirect.yaml`'s own `model_arg`, because the
    # matrix names CLI ids only — see `harness.provider_spec`.
    fan_out = harness.host_fan_out(host_entry, matrix_row)
    members = [] if fan_out == harness.FAN_OUT_NATIVE else list(fan_out)
    specs = [harness.provider_spec(cli) for cli in members]

    # THE WRITE RUNNER IS NOW A MATRIX KEY TOO (T0a). It was the same hard-coded
    # `codex:stub-model`, which means every non-negative row on all three hosts
    # dispatched its write batches through the one runner whose `agent-write`
    # renders `--sandbox workspace-write` — bubblewrap, needing a user namespace
    # Docker's default seccomp profile denies. That is the regression §1.4 of
    # `plans/chain-lane-write-runner-matrix.md` predicts for `codex-full`,
    # `opencode-full` and the two `apply-executor` rows.
    #
    # ON A NEGATIVE AUTH ROW THE RUNNER MUST STILL BE THE ROW'S OWN CLI. The
    # row's `remove_seed` names a file under that CLI's seeded home, and the
    # gate it is meant to trip is the executor's preflight of the CONFIGURED
    # runner — so a shared runner would sail past opencode's auth.json going
    # missing and the row would assert nothing at all.
    #
    # AND THE SLUG MUST BE IN THAT CLI'S OWN ADDRESSING SCHEME. opencode
    # resolves models as `provider/model`; handed a bare `stub-model` it parses
    # provider `stub-model` with an empty model half and raises
    # `ProviderModelNotFoundError: Model not found: stub-model/.` before a
    # single HTTP call leaves the process — which is what exited
    # `opencode-auth-negative` 3 on `runs/20260820T093838Z`. `harness.provider_spec`
    # reads the qualified form out of `redirect.yaml` rather than inventing a
    # second spelling.
    runner_cli = harness.host_runner_cli(host, host_entry, matrix_row)
    if negative:
        negative_model = "stub/stub-model" if host == "opencode" else "stub-model"
        runner_spec = f"{host}:{negative_model}"
    else:
        runner_spec = harness.provider_spec(runner_cli) if runner_cli else None
    # A native row, and a `remove_seed` row, fan out to nothing — but both
    # still need a spec for the rendered config to be complete. A row with an
    # EMPTY selection renders `models: []`, which deep-merges as "keep base's
    # twelve specs across five providers", i.e. the exact fall-through the
    # template's own header exists to prevent.
    #
    # One behaviour change is worth naming: `opencode-auth-negative` used to
    # render `codex:stub-model` as its fan-out (the old hard-coded literal)
    # beside an opencode runner. It now renders the opencode slug for both.
    # Inert either way — the row is graded on `stub_requests: 0` counted from
    # the watermark, and the refusal lands upstream of any model call — but
    # naming the row's own CLI is the coherent spelling, and a row whose
    # fan-out named a CLI it has nothing to do with was a leftover of the
    # literal, not a decision.
    if not specs:
        specs = [runner_spec] if runner_spec else []

    row = harness.Row(
        row_id=row_id, lane="host", host=host, creds=creds,
        # The fan-out, unioned with the runner (and the host on the negative
        # rows). A row that fans out to five providers while its helper list
        # still said `codex` ran four CLIs with no seeded home.
        helpers=harness.host_helpers(host, host_entry, matrix_row),
        spec=specs,
        runner=runner_spec,
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
    # THE MID-ROW STOP. Polls the key while the container runs and `docker
    # stop`s it — with the same grace the wall-clock cap uses, so the export
    # still runs — once the row's delta crosses the lower of its own ceiling and
    # the lane's remaining headroom. Bounded by a poll interval plus the
    # endpoint's settling lag, NOT a hard cap; see `host_spend`'s docstring.
    # It brackets BOTH attempts of a never-started retry, which reuse the name.
    watch = (host_spend.RowWatch.for_row(
        matrix_row, _Spend.ledger, before,
        stop_row=lambda: harness._stop_container(harness.container_name(row)))
        if before is not None else None)
    with (watch or contextlib.nullcontext()):
        result = harness.run_row(row, posture=posture, utc=utc)
    # `result.attempt` is 2 on a row the driver re-ran after a never-started
    # first attempt (`harness.never_started_signature`), and 1 on every other
    # row. The spend below still brackets BOTH attempts — `before` was read
    # before the first container and `after` after the last — so a retried row
    # is charged what it really cost.
    _Census.record_ran(row_id, host, creds, attempts=result.attempt)
    _Census.write(request.config, utc)

    # FAIL CLOSED on the closing readback too — but without throwing away the
    # row that already ran: it is graded, and the NEXT row is refused, because
    # the lane total is now a number nobody measured.
    after = None
    if before is not None:
        try:
            after = _key_usage()
        except host_spend.SpendUnobservable as exc:
            _Spend.ledger.unobservable = f"{row_id}'s closing readback: {exc}"
    if after is not None:
        _Spend.ledger.charge(before, after)
    if watch is not None and watch.tripped:
        # A distinct outcome, on disk beside the bundle and in the census, and
        # a failure of every test graded on this row: a row stopped mid-turn
        # proves nothing about the product either way.
        tripped = dict(watch.tripped, row=row_id,
                       delta_usd=None if after is None else round(after - before, 6))
        (result.out_dir / "budget.json").write_text(
            json.dumps(tripped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _Census.ran[-1]["outcome"] = tripped["outcome"]
        _Census.write(request.config, utc)
        if tripped["bound"] == "lane" or tripped["outcome"] != host_spend.BUDGET_EXCEEDED:
            # The lane ran out, or spend went dark mid-row: either way no
            # further row may start.
            _Spend.ledger.halted = (
                f"{row_id} was stopped mid-row: {tripped['reason']}")
        pytest.fail(f"host lane: {row_id} {tripped['outcome'].upper()} — "
                    f"{tripped['reason']}; the container was stopped mid-row. "
                    f"Evidence: {result.out_dir / 'budget.json'}")
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
        if matrix_row.get("route") == "package":
            pytest.skip(
                f"{matrix_row.get('id')}: the package route stages no shim, so "
                f"there is no step-3 `util instructions` to find; the plugin's "
                f"discovery is the tool call, graded by "
                f"TestTheCodexPackageAskUsedThePlugin")
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

    def test_every_assigned_provider_really_ran(self, workflow):
        """The runtime half of §3.3's partition — exactly N, each one named.

        The union guard in `test_redirect_matrix_claims.py` reads
        `host_matrix.yaml` and `redirect.yaml` and never starts a container,
        so it grades CONFIGURATION coverage only. Nothing else here closes the
        gap: `test_enough_receipts_for_the_rows_shape` grades
        `len(dispatcher) >= receipts_min`, a floor over a count with no
        provider identity in it, and the stub-traffic assertions grade totals
        the pre-command leg also contributes to. **A row could fan out to six
        providers, have five never invoked, and stay green on every other
        assertion in this module.**

        So, per assigned provider and counted **from the host-start
        watermark**, three facts that fail differently:

        * **an invocation** — a `calls.jsonl` record naming that CLI, which is
          what `/opt/mllm/bin` writes, so the shim really ran for it. Not a
          receipt: `MLLM_RECEIPT_DIR` is the *dispatcher* wrapper's, and no
          provider CLI writes one.
        * **a stub session** — at least one `stub.jsonl` record keyed to that
          exec, so the CLI reached the model surface instead of dying in
          argument parsing. opencode's `ProviderModelNotFoundError` failed at
          exactly this step and produced zero stub records while the row's
          total was non-zero, which is the shape a count-only assertion
          cannot see. **The stub log names no CLI** — it logs the `session`
          the wrapper minted — so this fact is a JOIN through `calls.jsonl`'s
          `exec_id`, and reading a `cli` key off a stub record (there is none)
          is how the fact reads as green for every provider at once.
        * **a usable result** — that provider's own result file, under the
          phase directories the row declares, **carrying an answer rather
          than that CLI's error envelope**. This is the exactly-N rule with
          a name attached to each of the N. The qualifier is not decoration:
          on `runs/20260922T084502Z` grok failed on all six opencode fan-out
          rows, four went red, and `opencode-ask` and `opencode-package-ask`
          went GREEN — because the ask lane persists the provider's raw
          stdout when it cannot recover text, and for a failed grok run that
          stdout IS `{"type":"error","message":"Not signed in…"}`, landing
          under grok's own exact expected filename. A name alone was
          therefore satisfiable by the CLI's own failure report, which is the
          exact opposite of the fact this bullet exists to establish, so the
          body is read through `harness.error_envelope_reason` before the
          name is counted.

        **All three are sliced at the host-start watermark, and the row is
        graded on its fan-out phases only.** The pre-command leg fans out to
        the same providers through the same shim, into the same two logs, so a
        fact counted over a whole log is a fact the rig can satisfy on the
        host's behalf — which is the whole failure this test was written
        against. A missing watermark is therefore missing evidence and fails
        here rather than defaulting to zero, and a row whose every declared
        phase dispatches through the single write runner has no fan-out to
        correlate and says so.

        **Exactly N, never at-least-K.** The fan-out is stubbed: there is no
        endpoint that can be flaky, so a wire-shape mismatch is a
        deterministic defect in the scenario's rules or in the adapter, and it
        is fixable. Grading at-least-K would convert the rig's own bugs into
        permanent silent gaps in the coverage this exists to add, and would
        make the union guard's "all 16 providers" claim unfalsifiable. A
        provider the scenario cannot answer concurrently is reassigned to
        another host's subset or recorded as a dated carve-out cell — never
        dropped by loosening this grade.
        """
        result, matrix_row, entry = workflow
        fan_out = harness.host_fan_out(entry, matrix_row)
        if fan_out == harness.FAN_OUT_NATIVE or not fan_out:
            pytest.skip(
                f"{matrix_row.get('id')}: declared outside the partition "
                f"({fan_out!r}) — a native Task dispatch never crosses "
                f"/opt/mllm/bin, so a fan-out list has no mechanism to act "
                f"through.")
        phases = [str(phase) for phase
                  in ((matrix_row.get("expect") or {}).get("phases") or [])
                  if str(phase) in FAN_OUT_PHASES]
        if not phases:
            pytest.skip(
                f"{matrix_row.get('id')}: every phase this row declares "
                f"({(matrix_row.get('expect') or {}).get('phases')}) dispatches "
                f"its batches through the ONE `driver.subagent_runner`, not "
                f"through `defaults.models` — so the assigned partition has no "
                f"mechanism to act through inside the host turn, and its only "
                f"fan-out is the pre-command leg's, which is the rig's work "
                f"and not this row's. The same host's `ask`, `review-plan` and "
                f"`full` rows grade the identical list.")
        stub_before = _stub_watermark(result)
        calls_before = _call_watermark(result)
        # THE THIRD EVIDENCE SOURCE'S WATERMARK, and it is a snapshot rather
        # than an index because result files live in a DIRECTORY, not in an
        # append-only log: `workspace-census-host-before.json` is written in
        # the same four-step block as `stub-watermark.json` — after the
        # pre-command leg, after the credential removal, immediately before
        # the host process — so every phase artifact it already lists is the
        # leg's.
        files_before = _host_start_census(result)
        assert (stub_before is not None and calls_before is not None
                and files_before is not None), (
            f"{matrix_row.get('id')}: the container wrote no host-start "
            f"watermark (stub={stub_before}, calls={calls_before} in "
            f"stub-watermark.json; workspace-census-host-before.json "
            f"{'present' if files_before is not None else 'absent'}), so "
            f"there is no line between the rig's pre-command fan-out and the "
            f"host's own dispatch. Reading it as 0 — or reading the whole "
            f"exported tree — would credit the leg's invocations, sessions "
            f"and result files to the host, which is exactly the correlation "
            f"this test exists to make: missing evidence, not zero.")
        calls = [r for r in result.call_log[calls_before:] if r.get("cli")]
        stub_after = [r for r in result.stub_log
                      if r.get("surface") != "health"][stub_before:]
        invoked = {str(r.get("cli")) for r in calls}
        # THE JOIN, because `stub.jsonl` has no `cli` field to read. The wrapper
        # mints one `exec_id` per invocation and bakes it into the base URL
        # (`wrapper.stub_base_url`), and the stub logs it straight back as the
        # record's `session` (`stub_server.session_key`). So provider identity
        # comes from the call log's `exec_id` -> `cli` map, built from the
        # POST-WATERMARK invocations alone: a session belonging to one of the
        # leg's execs then names nobody instead of naming the host's provider.
        exec_to_cli = {str(r.get("exec_id")): str(r.get("cli"))
                       for r in calls if r.get("exec_id")}
        reached = set()
        for record in stub_after:
            session = str(record.get("session") or "")
            cli = exec_to_cli.get(session)
            if cli is None:
                # The two fallback spellings `session_key` can return still
                # carry the id: a CLI that rejects the path prefix sends it as
                # the `mllm-stub-<exec-id>` credential in a session header.
                cli = next((name for exec_id, name in exec_to_cli.items()
                            if exec_id and exec_id in session), None)
            if cli:
                reached.add(cli)
        results_dir = _artifacts_root(result)
        produced: list[str] = []
        #: Files that carried a provider's own failure report instead of an
        #: answer. Kept and printed rather than dropped, so "grok wrote
        #: nothing" and "grok wrote its login error into the answer file" do
        #: not read identically in the failure.
        refused: list[str] = []
        if results_dir is not None:
            # The snapshot, re-keyed onto the exported tree's own paths once
            # (`export_artifacts` drops the plan file's parent directory), so
            # the per-file lookup below is a dict hit rather than a scan.
            prior = _exported_before_host(files_before, results_dir)
            for phase in phases:
                for directory in sorted(results_dir.glob(f"*/{phase}")):
                    # `.json` for the review phases' `<cli>_<model>.json` and
                    # `.md` for `ask`'s `answer_<cli>_<model>.md`; never the
                    # `log_<cli>_<model>.txt`, which a CLI that died in
                    # argument parsing also leaves behind.
                    for path in sorted(directory.rglob("*")):
                        # `is_file` as well as the suffix, because the slice
                        # below reads the bytes: a directory that happens to
                        # end in `.json` was a name before and would be an
                        # `IsADirectoryError` now.
                        if (path.suffix not in (".json", ".md")
                                or not path.is_file()):
                            continue
                        # AND THE SLICE. A prerequisite phase run by the leg
                        # writes ITS providers' result files into these same
                        # directories under these same names, so a name
                        # counted off the whole exported tree is a name the
                        # rig can supply on the host's behalf — the identical
                        # defect the two log watermarks close, on the one
                        # source that is a tree. A file the host-start census
                        # already lists WITH THE BYTES IT STILL HAS is the
                        # leg's; one that is absent from it, or whose sha the
                        # host moved, is the row's own.
                        key = path.relative_to(results_dir).as_posix()
                        if key in prior and prior[key] == _sha256(path):
                            continue
                        # AND THE BODY, because a provider that FAILED can
                        # still leave a file under its own exact expected
                        # name. grok did, on both `ask` rows of
                        # `runs/20260922T084502Z`: 219 bytes of
                        # `{"type":"error","message":"Not signed in…"}` in
                        # `answer_grok_stub-model.md`, written by
                        # `ask_orchestrator.recover_answer_text` falling back
                        # to the provider's raw stdout — and both rows went
                        # GREEN on this assertion while grok was failing on
                        # all six of that host's fan-out rows. The ask lane
                        # no longer reads `raw` (it reads only the reduced
                        # `answer_text`, `""` for that body — plans/ask-answer-
                        # recovery-and-false-success.md D2.3); this check does
                        # not rely on that and stays as the second gate.
                        # `harness.error_envelope_reason` reads the CLIs' own
                        # measured discriminators (`type: "error"`,
                        # `is_error: true`) off a body that parses WHOLE as a
                        # JSON object; a prose answer is never parsed, and a
                        # review result is an array.
                        reason = harness.error_envelope_reason(
                            path.read_text(encoding="utf-8", errors="replace"))
                        if reason is not None:
                            refused.append(f"{path.name} ({reason[:120]})")
                            continue
                        produced.append(path.name)
        # The predicate itself is `harness.fan_out_correlation`, a pure
        # function over the three extracted name sets — same behaviour, one
        # importable place — because this assertion is `live`-marked beneath
        # a container gate and therefore only ever answers inside a
        # credentialed Docker run. Its NEGATIVE CONTROL, without which a
        # correlation that matched on the wrong field or iterated an empty
        # list would read exactly like complete coverage, is a plain-CI
        # fixture in `tests/test_stray_write_paths.py`.
        missing = harness.fan_out_correlation(
            fan_out, invoked=invoked, reached=reached, produced=produced)
        assert not missing, (
            f"{matrix_row.get('id')}: the row declared a partition it did not "
            f"execute. Per provider, the facts it could not produce:\n"
            + "\n".join(f"    {cli}: missing {', '.join(gaps)}"
                        for cli, gaps in sorted(missing.items()))
            + f"\n  fan_out     : {sorted(fan_out)}"
            + f"\n  invoked     : {sorted(n for n in invoked if n)}"
            + f"\n  reached stub: {sorted(n for n in reached if n)}"
            + f"\n  result files: {sorted(produced)[:20]}"
            + (f"\n  refused as error payloads: {sorted(refused)[:20]}"
               if refused else "")
            + f"\n  counted from : stub.jsonl[{stub_before}:], "
              f"calls.jsonl[{calls_before}:], result files new or rewritten "
              f"since workspace-census-host-before.json, phases {phases}")

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
        if result.row.creds == "fallback":
            pytest.skip(
                f"{result.row.row_id}: --creds fallback runs on the vendor's own "
                f"subscription, whose spend this rig cannot read (the credential "
                f"census says so); the OpenRouter key is not in the container, and "
                f"when it is set in the driver's environment the fixture's own "
                f"readback of it still brackets the row")
        usage = result.usage
        assert usage, f"{result.row.host}: no usage readback recorded for the row"
        assert usage.get("delta_usd") is not None, (
            f"{result.row.host}: actual: unavailable — the running total is an "
            f"estimate from here on")


class TestTheWriteRowsAreGradedOnBytes:
    """The third row per host, and the model's claim is never the verdict.

    Two shapes, deliberately opposite, each declared per row rather than inferred:

    * **`no-op`** on the two executor hosts. A real executor CLI pointed at
      a tool-call-free stub can only ever claim no-op and move nothing, so that
      is what the row asserts — and an `applied` batch there fails just as loudly,
      because bytes landing from a tool-call-free stub mean the redirect leaked.
      CORRECTED 2026-09-22, from `clean-failure`: the word moved in the PRODUCT,
      not in the rig. `1633dd4` (2026-09-16) split the empty-diff branch so a
      `no-op` CLAIM over an empty diff records `no-op`, while an `applied` claim
      over an empty diff stays `clean-failure`. The stub's fixture is unchanged
      since `f62cc89` (2026-08-15), so the input never moved. The leak guard is
      not weaker for it: `applied` and `partial-application` are both reachable
      only on a NON-empty snapshot diff, so refusing them is what disproves a
      leak — not which of the two empty-diff words is named. See the matching
      cells in `host_matrix.yaml` and the derivation in `test_container_full_chain.py`.
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
        """Whether these bytes changed and which paths moved — never their content.

        Graded through `_graded_delta`, so a row declaring
        `editable_artifacts` is answered on the HOST-bracketed census with
        real bookkeeping filtered — and a row that declares none keeps
        `_source_delta` exactly as it was.
        """
        result, matrix_row, entry = workflow
        _write_outcome(matrix_row)
        expect = matrix_row.get("expect") or {}
        delta = _graded_delta(result, matrix_row, entry)
        if expect.get("bytes_must_change"):
            assert delta, (
                f"{result.row.host}: the SHA-256 census of /work/sample-project "
                f"came back byte-identical outside the workflow's own outputs, so "
                f"the native write row applied nothing. Graded against this "
                f"row's declared editable artifacts "
                f"{_editable_artifacts(matrix_row)} over the "
                f"{'host-bracketed' if result.has_host_census else 'whole-row'} "
                f"census.")
        else:
            assert not delta, (
                f"{result.row.host}: bytes moved on a row declared "
                f"{_write_outcome(matrix_row)!r}: {sorted(delta)}")

    def test_no_write_landed_outside_the_declared_paths(self, workflow):
        result, matrix_row, entry = workflow
        _write_outcome(matrix_row)
        # `editable_artifacts` first: it is the per-phase declaration §3.4
        # introduces, and `expect.paths` is the older spelling the rows that
        # predate it still carry. One accessor, so a row cannot declare both
        # and have the weaker one graded.
        declared = _editable_artifacts(matrix_row)
        if not declared:
            pytest.skip(f"{result.row.row_id}: the row declares no write paths")
        # The predicate itself lives in `harness.stray_paths` — same behaviour,
        # one importable place — because this assertion is `live`-marked beneath
        # this module's container gate and therefore only ever answers inside a
        # credentialed Docker run. `tests/test_stray_write_paths.py` pins it in
        # both directions on plain CI, where a PR that widens a `paths:` glob too
        # far is actually read.
        stray = harness.stray_paths(
            _graded_delta(result, matrix_row, entry), declared)
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


def _next_command_observed(graded):
    """`(result, matrix_row, observed)` for a T10 row; every other row skips.

    `observed` is `next_command_check.observe` over the row's bundle — the same
    pure function the record is written from — and it is also filed beside the
    bundle as `next-command-observed.json`, so a red row's evidence is one file a
    human can open rather than a re-derivation.
    """
    result, matrix_row, _entry = graded
    if not matrix_row.get("next_command_check"):
        pytest.skip(f"{matrix_row.get('id')}: not a T10 row (it declares no "
                    f"`next_command_check`)")
    templates = dict((harness.load_capabilities().get(result.row.host) or {})
                     .get("next_command") or {})
    observed = next_command_check.observe(
        result.out_dir, result.row.host, result.row.row_id, matrix_row,
        templates, route=result.row.route)
    (result.out_dir / "next-command-observed.json").write_text(
        json.dumps(observed, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return result, matrix_row, observed


class TestTheNextCommandLiveCheck:
    """T10 of plans/next-phase-command.md, asserted rather than read by eye.

    One phase per host and route, against the real host model with the fan-out
    stubbed. Each test is one of the plan's checks, and each one fails naming
    the evidence it read:

    * **criterion 4** — the host's final assistant message carries the command
      its dispatcher printed, as a fenced line, byte for byte, with one more
      fenced line per `optional_before` entry;
    * **claude-code's two extras** — the lookup (a `util next-command` call, or
      a phase result carrying the block, after the phase's last step), and
      whether T11's Stop hook blocked, so "copied" and "copied after the
      check" are counted apart;
    * **criterion 5** — on the two tool routes, no `multi_llm` call after the
      phase's terminal result and none whose mode is the next phase;
    * **the typed template** — the printed command, handed verbatim to a fresh
      session of the same host and route, dispatched the named mode with a plan
      argument that names the fixture and holds no quote character. This is
      the only test whose pass can promote a `next_command` cell.

    A red here is a legitimate T10 outcome — a model that does not copy the
    command, or runs it — and is recorded, never excused: the cell stays
    `documented`.
    """

    pytestmark = pytest.mark.live

    def test_the_final_message_carries_the_printed_command(self, graded):
        result, matrix_row, observed = _next_command_observed(graded)
        copy = observed["copy"]
        printed = observed.get("printed")
        assert copy.get("observable"), (
            f"{matrix_row.get('id')}: no dispatcher receipt of the session "
            f"carried a user_next_step, so there was no printed command for the "
            f"report to copy — the phase did not finish (see the receipts and "
            f"run-states.json). Criterion 4 is NOT OBSERVED on this run.")
        message = next_command_check.final_message(
            result.row.host, (result.out_dir / "host-stdout.txt").read_text(
                encoding="utf-8", errors="replace")) or ""
        if copy.get("complete"):
            assert copy.get("passes"), (
                f"{matrix_row.get('id')}: the workflow is complete, and the "
                f"final message must say {next_command_check.WORKFLOW_COMPLETE_TEXT!r} "
                f"with no fenced command (stray: {copy.get('stray_commands')}).\n"
                f"--- final message (tail) ---\n{message[-1500:]}")
            return
        assert copy.get("passes"), (
            f"{matrix_row.get('id')}: the host's final message does not carry "
            f"the printed command as a fenced line, byte for byte.\n"
            f"  printed command   : {printed.get('command')!r}\n"
            f"  command copied    : {copy.get('command_copied')} "
            f"(present anywhere: {copy.get('command_anywhere')})\n"
            f"  optional commands : {copy.get('optional_commands')}\n"
            f"  optional copied   : {copy.get('optional_copied')}\n"
            f"  near misses       : {copy.get('near_misses')}\n"
            f"--- final message (tail) ---\n{message[-2000:]}")

    def test_claude_code_looked_the_command_up(self, graded):
        result, matrix_row, observed = _next_command_observed(graded)
        if result.row.host != "claude-code":
            pytest.skip(f"{matrix_row.get('id')}: the lookup extra is "
                        f"claude-code's — the tool and file routes on the other "
                        f"hosts carry the key in the phase's own result")
        lookup = observed.get("lookup") or {}
        assert observed.get("stop_hook") in (
            "blocked", "ran", "silent", "not-installed"), (
            f"{matrix_row.get('id')}: the record does not say whether T11's Stop "
            f"hook blocked: {observed.get('stop_hook')!r}")
        assert lookup.get("lookup") in ("util-next-command", "phase-result"), (
            f"{matrix_row.get('id')}: after the phase's last dispatcher step "
            f"there is neither a `util next-command` call nor a phase result "
            f"carrying the block ({lookup}). A copy the report could not make "
            f"is a skipped CALL, not a bad copy — counted apart on purpose.")

    def test_the_host_model_did_not_run_the_printed_command(self, graded):
        result, matrix_row, observed = _next_command_observed(graded)
        if result.row.route != "package":
            pytest.skip(f"{matrix_row.get('id')}: criterion 5 is asserted on the "
                        f"two tool routes (the Codex plugin and the OpenCode "
                        f"package); this row's observation is recorded only")
        no_call = observed.get("no_call") or {}
        assert no_call.get("observable"), (
            f"{matrix_row.get('id')}: {no_call.get('reason') or no_call}. "
            f"Criterion 5 is NOT OBSERVED on this run.")
        assert no_call.get("holds"), (
            f"{matrix_row.get('id')}: the host model ran the dispatcher after "
            f"the phase's terminal result, or ran the printed next phase.\n"
            f"  next phase              : {no_call.get('phase')!r}\n"
            f"  terminal call index     : {no_call.get('terminal_call_index')}\n"
            f"  calls after terminal    : {no_call.get('calls_after_terminal')}\n"
            f"  calls naming next phase : {no_call.get('calls_naming_next_phase')}\n"
            f"  receipts naming it      : {no_call.get('receipts_naming_next_phase')}\n"
            f"  every dispatching call  : {no_call.get('calls')}")

    def test_the_typed_command_dispatched_the_named_mode(self, graded):
        result, matrix_row, observed = _next_command_observed(graded)
        typed = observed.get("typed") or {}
        assert typed.get("exercised"), (
            f"{matrix_row.get('id')}: the typed-template session never ran: "
            f"{typed.get('reason')}. This route is NOT EXERCISED on this run, "
            f"and its next_command cell stays `documented`.")
        assert typed.get("dispatched"), (
            f"{matrix_row.get('id')}: the fresh session was handed "
            f"{typed.get('typed')!r} ({typed.get('surface')}) and never "
            f"dispatched {typed.get('phase')!r}; it ran "
            f"{typed.get('modes_dispatched')} and stopped on "
            f"{typed.get('stop_reason')!r}.")
        assert typed.get("quote_free") and typed.get("names_fixture"), (
            f"{matrix_row.get('id')}: {typed.get('phase')!r} dispatched, but "
            f"the plan argument the dispatcher recorded is not the fixture "
            f"unquoted: {typed.get('plan_args')} (fixture "
            f"{(matrix_row.get('fixture_plan') or {}).get('path')!r}).")


def _dispatcher_receipts(result) -> list[dict]:
    """Every receipt except the entrypoint's own staging invocation.

    `install-host` is the rig staging the shim, not the host running our
    dispatcher, so counting it would let a row pass on the harness's own work.

    The rule itself lives in the harness now, because the driver applies it too:
    a zero-receipt row is one of the facts `harness.never_started_signature`
    re-runs a row on, and two copies of "which receipts count" is how the grader
    and the driver would start disagreeing about whether a row did anything.
    """
    return harness.dispatcher_receipts(result)


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


def _host_start_watermark(result, key: str) -> int | None:
    """One counter out of `stub-watermark.json`, or `None` if it is not there.

    `entrypoint.record_stub_watermark` writes the file at the single moment the
    boundary is unambiguous — after the pre-command leg, after the credential
    removal, immediately before the host process starts — and both counters in
    it are indexes into logs the leg also appends to.
    """
    path = result.out_dir / "stub-watermark.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get(key)
    return None if value is None else int(value)


def _stub_watermark(result) -> int | None:
    """Non-health stub requests already logged when the host process started.

    `None` means the container wrote no watermark, which is a missing-evidence
    condition rather than a zero: reading it as 0 would silently credit the rig's
    own pre-command fan-out to the host.
    """
    return _host_start_watermark(result, "non_health_calls_before_host")


def _call_watermark(result) -> int | None:
    """`calls.jsonl` records already written when the host process started.

    The byte-for-byte counterpart of :func:`_stub_watermark` on the other log
    the pre-command leg writes to: the leg's provider CLIs cross the same
    `/opt/mllm/bin` shim, so an invocation count taken over the whole file is
    the rig's fan-out plus the host's. `None` is the same missing-evidence
    condition, never a zero.
    """
    return _host_start_watermark(result, "call_records_before_host")


def _fan_out_is_native(matrix_row: dict, entry: dict) -> bool:
    """Whether this row's batch is dispatched by the HOST rather than by us.

    Only true of the reference host's write row. Its subagent batch never crosses
    `/opt/mllm/bin` — that is what makes it real spend and what makes it the one
    row where `applied` bytes are a pass — so "every model call landed on the
    stub" has nothing to grade there. The ask and review-plan rows on the same
    host still fan out to provider CLIs by name, and are graded normally.
    """
    # A row that DECLARES `fan_out: native` says the same thing in its own words:
    # T10's claude-code row runs its one task-generation subagent as a native
    # Task, so no wrapped call ever reaches the stub. Every earlier row that
    # declares it is also an `applied` write row, so none of them moves.
    if harness.host_fan_out(entry, matrix_row) == harness.FAN_OUT_NATIVE:
        return True
    return (((matrix_row.get("expect") or {}).get("write_outcome") == "applied")
            and (entry.get("continuation") or {}).get("who") == "host")


def _artifacts_root(result):
    """The exported `artifacts/` tree, or `None` when the row exported none.

    `None` rather than a missing directory, so the per-provider correlation
    can tell "this row produced no artifacts at all" (someone else's failure,
    already graded) from "this provider produced no result file" (its own).
    """
    root = result.out_dir / "artifacts"
    return root if root.is_dir() else None


def _host_start_census(result) -> dict | None:
    """`{workspace-relative path: sha256}` at the moment the host process began.

    The result-file counterpart of :func:`_stub_watermark` and
    :func:`_call_watermark`: `entrypoint.write_host_census("before")` runs in
    the same four-step block, one line after `record_stub_watermark`, so the
    three evidence sources are cut at one moment. `None` is the same
    missing-evidence condition both counters use — an absent snapshot read as
    "nothing existed yet" would hand the pre-command leg's phase artifacts to
    the host, which is the whole reason the boundary is recorded.

    The WHOLE-ROW `workspace-census-before.json` is deliberately not a
    fallback here: it is taken before the leg runs, so it lists none of the
    leg's output and would restore exactly the grade this replaces.
    """
    path = result.out_dir / "workspace-census-host-before.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _exported_before_host(census: dict, results_dir: Path) -> dict[str, str]:
    """That census re-keyed onto the exported tree, `{artifacts-relative: sha}`.

    `entrypoint._workspace_census` keys on the path relative to the workspace
    (`plans/auth-feature/review-plan/codex_stub-model.json`) while
    `export_artifacts` copies the plan's output directory, dropping the plan
    file's parent (`auth-feature/review-plan/codex_stub-model.json`).

    The DIRECTORIES are the whole of this re-key, and the `is_dir()` filter
    below is what keeps it that way now that the export also carries loose
    receipt-named files at the artifacts root (the plan file). A result file
    is only ever read out of a phase directory, so nothing that belongs in
    this index lives outside one.

    The dropped parent is recovered from the tree rather than restated from
    `entrypoint.py`'s plan path: a hard-coded `plans/` that drifted would stop
    matching, and it would stop matching SILENTLY — every result file would
    read as new and the grade would quietly revert to counting the leg's.
    Entries outside the exported directories are simply absent, which is what
    makes the lookup a slice rather than a filter over the workspace.
    """
    index: dict[str, str] = {}
    for top in sorted(path.name for path in results_dir.iterdir()
                      if path.is_dir()):
        marker = f"/{top}/"
        for key, sha in census.items():
            if key.startswith(f"{top}/"):
                index[key] = sha
                continue
            position = key.find(marker)
            if position != -1:
                index[key[position + 1:]] = sha
    return index


def _sha256(path: Path) -> str:
    """`entrypoint._sha`, on this side of the export.

    The same digest over the same bytes, so a census entry and the exported
    file compare directly and "the host rewrote the leg's result file" reads
    as the row's own evidence rather than as the leg's.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(result, phase: str) -> dict:
    """`{phase}/write_batch_report.json` as exported into the bundle."""
    root = result.out_dir / "artifacts"
    for candidate in sorted(root.glob(f"*/{phase}/write_batch_report.json")):
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _editable_artifacts(matrix_row: dict) -> list[str]:
    """What this phase is allowed — and expected — to edit, from the matrix.

    §3.4's grader change 2. It replaces "everything except the `plans/` and
    `.multi-llm/` prefixes" as the grading predicate, because
    `apply-suggestions`' entire intended effect IS an edit to a file under
    `plans/`: a row graded by the old predicate could rewrite the plan, or
    fail to, and pass either way.
    """
    declared = matrix_row.get("editable_artifacts")
    if declared:
        return [str(p) for p in declared]
    return [str(p) for p in ((matrix_row.get("expect") or {}).get("paths") or [])]


def _graded_delta(result, matrix_row: dict, host_entry: dict) -> dict:
    """The census paths this row is answerable for.

    Three corrections over `_source_delta`, and they are independent:

    1. **Bracket the host's own execution.** `host_census_delta` uses the
       narrower pair written immediately before and after the host process,
       so setup performed by `pre_commands` cannot contaminate what the host
       is credited with.
    2. **Filter real bookkeeping, not the `plans/` prefix.**
       `harness.graded_delta` drops `.multi-llm/**` and each plan's own
       OUTPUT DIRECTORY (`plans/<stem>/**`) — but never the plan FILE, unless
       this row does not declare it, in which case the change is
       `finalize_tracking`'s back-reference rather than work the row did.
    3. **Drop a third-party tool's own cache**, and only the ones the row's
       OWN fan-out can have written. `harness.tool_artifact_paths` reads the
       globs each CLI declares in `redirect.yaml` — today just aider's repo
       map, `.aider.tags.cache.v4/**`, which aider rebuilds in `cwd` on every
       invocation. It landed in the census of all four codex-host rows of
       `runs/20260922T084502Z` (aider is in codex's fan-out) and was the whole
       delta on `codex-apply-executor`, failing that row with `bytes moved on
       a row declared 'clean-failure'` over bytes no subagent wrote. Applied
       to BOTH branches below, because the rows it hits declare no
       `editable_artifacts` and therefore never reach `harness.graded_delta`.

    Rows that declare no `editable_artifacts` keep the old predicate exactly
    apart from that subtraction, so nothing already graded moves — and the
    subtraction is a no-op on a `fan_out: native` row, which is every row that
    grades applied bytes.
    """
    declared = _editable_artifacts(matrix_row)
    if not matrix_row.get("editable_artifacts"):
        delta = _source_delta(result)
    else:
        census = result.host_census_delta()
        kept = harness.graded_delta(census.keys(), declared)
        delta = {name: census[name] for name in kept}
    exempt = set(harness.tool_artifact_paths(
        delta, harness.host_fan_out(host_entry, matrix_row), declared))
    return {name: pair for name, pair in delta.items() if name not in exempt}


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

    Both halves of the map are now really produced:
    `entrypoint.export_artifacts` copies the plan's output DIRECTORY
    (`<tail>` beginning `auth-feature/`) and
    `entrypoint.export_receipt_named_plan_files` copies the receipt-named
    files beside it — the plan file itself, `<tail>` == `auth-feature.md`,
    which `apply-suggestions` names and which no directory copy could reach.
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


# --------------------------------------------------------------------------- #
# THE CODEX PACKAGE ROUTE (plans/codex-native-plugin.md, Phase 1 items 1–16)
#
# Two tiers, graded on different evidence and never on each other's:
#
# * the ZERO-QUOTA rows — `host_matrix.yaml`'s `codex.package_probes:` list,
#   each one `--creds none --route package --probe <name>` under `--network
#   none`, the host's model the loopback stub. `container`-marked, NOT `live`.
# * `codex-package-ask`, the one credentialed row, scheduled with the others in
#   `CREDENTIALED_ROWS` and graded here on the route-specific evidence
#   (`TestTheCodexPackageAskUsedThePlugin`) on top of every generic workflow
#   assertion above.
#
# THE RULE THE GRADERS KEEP (the plan's "Traps on record"): assert on the
# advertised surface and on mutated keys, never on exit status or presence.
# `codex plugin list` proves a catalog entry exists (R25); `codex doctor` cannot
# see a plugin server (R43). So the seed is graded on the config table the
# install wrote and the directory it left, the loader on `codex mcp get`, the
# advertised skill and the `tools[]` capture — and NEVER on a shim path, which a
# package row must not have at all.
#
# TOLERANT OF ROWS NOT YET RUN: every grader reads its row's bundle through the
# fixture, which SKIPS BY NAME when the plugin tree the rows install does not
# exist yet, or when the image's codex is below the plugin floor. Plain CI never
# reaches any of it — the module is `container`-marked.
# --------------------------------------------------------------------------- #

REPO_ROOT = harness.HERE.parents[3]
CODEX_PLUGIN_TREE = REPO_ROOT / "packaging" / "codex" / "multi-llm"
CODEX_ENTRY = dict(dict(HOSTS).get("codex") or {})
PACKAGE_PROBES = [dict(p) for p in (CODEX_ENTRY.get("package_probes") or [])]
PACKAGE_PROBE_IDS = [p["id"] for p in PACKAGE_PROBES]
#: The plugin floor, from the matrix cell that names it.
CODEX_PLUGIN_FLOOR = str(CODEX_ENTRY.get("plugin_floor") or "0.159.2")
CODEX_PLUGIN_ID = "multi-llm@beastlabai"
CODEX_PLUGIN_TABLE = f'[plugins."{CODEX_PLUGIN_ID}"]'


def _version_tuple(text: str | None) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(part) for part in match.groups()) if match else None


def _codex_floor_skip_reason() -> str | None:
    """`None` when the image's codex meets the plugin floor, else why not.

    Read off the image's own version report (`harness.cli_version`), which is
    the build's measurement of what landed — never the lock's request.
    """
    measured = harness.cli_version("codex")
    have = _version_tuple(measured)
    floor = _version_tuple(CODEX_PLUGIN_FLOOR)
    if have is None:
        return (f"the image reports no readable codex version ({measured!r}); "
                f"the plugin floor is {CODEX_PLUGIN_FLOOR}")
    if floor and have < floor:
        return (f"codex {measured} is below the plugin floor "
                f"{CODEX_PLUGIN_FLOOR}; every codex plugin row is skipped")
    return None


def _plugin_tree_skip_reason() -> str | None:
    if not (CODEX_PLUGIN_TREE / ".codex-plugin" / "plugin.json").is_file():
        return (f"{CODEX_PLUGIN_TREE.relative_to(REPO_ROOT)}/ does not exist yet "
                f"(plans/codex-native-plugin.md Phase 1 items 2-4 build it); the "
                f"codex package rows install it and Dockerfile §10c copies it")
    return None


def test_container_codex_meets_plugin_floor():
    """Phase 1 item 1: the image's codex is at or above the plugin floor.

    A SKIP, never a failure, below the floor — and the skip names the measured
    version and the floor, because "the plugin rows did not run" must say why
    in the one sentence a reader sees. Every plugin row's fixture asks the same
    question and skips with the same words.
    """
    reason = _codex_floor_skip_reason()
    if reason:
        pytest.skip(reason)
    assert _version_tuple(harness.cli_version("codex")) >= _version_tuple(
        CODEX_PLUGIN_FLOOR)


class TestThePackageProbesAreScheduled:
    """The probe list is read from the matrix, so it cannot fall behind it —
    but a probe the ENTRYPOINT does not know would run as a refused row, so
    the two tables are compared here."""

    def test_every_matrix_probe_is_one_the_entrypoint_runs(self):
        from .container import entrypoint

        unknown = sorted({p.get("probe") for p in PACKAGE_PROBES}
                         - set(entrypoint.CODEX_PACKAGE_PROBES))
        assert not unknown, (
            f"host_matrix.yaml's codex package_probes name {unknown}, which "
            f"entrypoint.CODEX_PACKAGE_PROBES does not run")

    def test_every_probe_row_is_package_route_and_unmeasured_or_dated(self):
        for probe in PACKAGE_PROBES:
            assert probe.get("probe"), probe
            assert str(probe.get("provenance") or "") in (
                "unmeasured", "verified", "measured-red"), probe


@pytest.fixture(scope="module", params=PACKAGE_PROBES, ids=PACKAGE_PROBE_IDS)
def package_probe(request, posture, utc):
    """Run one zero-quota codex package row and return (result, probe row)."""
    probe = request.param
    for reason in (_plugin_tree_skip_reason(), _codex_floor_skip_reason()):
        if reason:
            pytest.skip(reason)
    row = harness.Row(
        row_id=probe["id"], lane="host", host="codex", creds="none",
        # Inert on this tier (nothing dispatches a provider except the
        # sandbox-posture leg's nested codex, which the stub answers), and
        # NAMED rather than defaulted, for the discovery fixture's reason.
        spec=["opencode:stub/stub-model"],
        runner="opencode:stub/stub-model",
        scenario=probe.get("scenario"),
        route="package",
        probe=str(probe["probe"]),
        row_timeout_s=int(probe.get("wall_clock_s") or 900),
    )
    result = harness.run_row(row, posture=posture, utc=utc)
    return result, probe


def _bundle_json(result, name: str):
    path = result.out_dir / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_record(package_probe) -> dict:
    result, probe = package_probe
    record = _bundle_json(result, f"probe-{probe['probe']}.json")
    assert record is not None, (
        f"{probe['id']}: no probe-{probe['probe']}.json in {result.out_dir} — "
        f"the row did not reach its probe (see the container's stdout/stderr)")
    return record


def _tool_envelope(value):
    """The server's one envelope (``ok``/``state``/``exit``/…), decoded.

    The server replies ``{"content": [{"type": "text", "text": <envelope
    JSON>}], "isError": …}``; a probe records that inside a JSON-RPC response,
    and codex hands it back to the model as a string (the text itself, or the
    content list serialized). Every layer is unwrapped here, so a grade reads
    the envelope's fields rather than substring-searching a serialization whose
    inner JSON is escaped. ``None`` when no envelope is found.
    """
    if isinstance(value, str):
        # Codex heads the text it hands back with `Wall time: <s>
        # seconds\nOutput:\n` (measured at 0.156.1): retry from the first `{`.
        text = value
        try:
            value = json.loads(text)
        except ValueError:
            if "{" not in text:
                return None
            try:
                value = json.loads(text[text.index("{"):])
            except ValueError:
                return None
    if isinstance(value, list):
        for item in value:
            found = _tool_envelope(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    if "ok" in value and "state" in value and "exit" in value:
        return value
    for key in ("result", "content", "text", "output"):
        if key in value:
            found = _tool_envelope(value[key])
            if found is not None:
                return found
    return None


#: What a zero-quota `status` on the seeded workspace may exit with: ok, or the
#: surface's nothing-to-do (no run recorded for the plan).
_STATUS_EXITS = (0, 2)


def _assert_status_envelope(envelope, where: str, raw) -> None:
    assert envelope is not None, f"{where}: no tool envelope in {raw!r}"
    assert envelope["ok"] is True, f"{where}: not ok: {envelope}"
    assert envelope["exit"] in _STATUS_EXITS, f"{where}: {envelope}"
    assert isinstance(envelope.get("result"), dict), (
        f"{where}: the envelope carries no parsed dispatcher result "
        f"(MULTI_LLM_RESULT): {envelope}")


def _only(package_probe, *probes: str) -> None:
    _result, probe = package_probe
    if probe["probe"] not in probes:
        pytest.skip(f"{probe['id']} is not a {'/'.join(probes)} row")


class TestTheCodexPackageSeed:
    """Item 5, on EVERY package row: each container re-seeds, so each bundle
    carries its own install record."""

    def test_the_row_completed_offline(self, package_probe):
        result, probe = package_probe
        assert not result.timed_out, f"{probe['id']}: {result.cap_reason}"
        inside = _bundle_json(result, "posture-inside.json") or {}
        assert not (inside.get("network") or {}).get("default_route"), (
            f"{probe['id']}: a zero-quota package row had a route out")

    def test_the_install_wrote_its_own_table_into_a_quiet_config(self, package_probe):
        result, probe = package_probe
        install = _bundle_json(result, "plugin-install.json")
        assert install, f"{probe['id']}: no plugin-install.json"
        tables = install["config_toml"]
        assert CODEX_PLUGIN_TABLE not in tables["tables_before"]
        assert not any(t.startswith(("[plugins", "[marketplaces"))
                       for t in tables["tables_before"]), tables
        assert tables["plugin_table"], (
            f"{probe['id']}: `codex plugin add` left no {CODEX_PLUGIN_TABLE} "
            f"table — the key the install mutates is the evidence, not its "
            f"exit code")
        assert re.search(r"(?m)^\s*enabled\s*=\s*true\b", tables["plugin_table"]), (
            tables["plugin_table"])

    def test_the_install_left_a_directory_under_the_cache(self, package_probe):
        result, probe = package_probe
        install = _bundle_json(result, "plugin-install.json")
        assert install["installed_root_exists"], install
        assert "/plugins/cache/beastlabai/multi-llm/" in install["installed_root"]
        assert install["installedPath"], (
            "R40: `codex plugin add --json` carries installedPath")

    def test_the_list_document_was_committed_verbatim(self, package_probe):
        result, probe = package_probe
        listing = result.out_dir / "codex-plugin-list.json"
        assert listing.is_file(), f"{probe['id']}: no codex-plugin-list.json"
        document = json.loads(listing.read_text(encoding="utf-8"))
        assert CODEX_PLUGIN_ID in json.dumps(document)

    def test_no_file_shim_rides_beside_the_plugin(self, package_probe):
        result, probe = package_probe
        install = _bundle_json(result, "plugin-install.json")
        assert install["file_shim_present"] is False, (
            f"{probe['id']}: a package row carries the FILE shim too — two "
            f"advertised copies of the skill, and a grade that could be the "
            f"shim's")
        assert install["divergence"] is None, (
            "the installed tree was edited before the row ran; a row that "
            "passes only over an edited copy grades the edit (D3a)")


class TestTheCodexPluginFloorAndFeatures:
    def test_the_features_printout_is_the_record(self, package_probe):
        _only(package_probe, "features")
        record = _probe_record(package_probe)
        features = record["features"]
        assert features, "codex features list printed no parsable row"
        assert features.get("plugins", [None, None])[1] is True, features

    def test_the_floor_row_read_a_version(self, package_probe):
        _only(package_probe, "floor")
        record = _probe_record(package_probe)
        assert _version_tuple(record["codex_version"]["version"]) >= \
            _version_tuple(CODEX_PLUGIN_FLOOR)


class TestTheLoaderRows:
    """Item 6 — zero quota, all three."""

    def test_codex_mcp_get_echoes_the_server_from_the_cache(self, package_probe):
        _only(package_probe, "loader-mcp-get")
        record = _probe_record(package_probe)
        document = record["document"]
        assert document, f"`codex mcp get multi_llm --json` printed no JSON: " \
                         f"{record['command']}"
        text = json.dumps(document)
        assert "/plugins/cache/beastlabai/multi-llm/" in text, (
            "the server's cwd did not expand to the cache root (R11)")
        assert "tool_timeout_sec" in text

    def test_the_skill_is_advertised_and_resolves_through_the_roots(
            self, package_probe):
        _only(package_probe, "loader-prompt-input")
        skill = _probe_record(package_probe)["skill"]
        assert skill["advertised"], "multi-llm:multi-llm is not advertised"
        assert skill["resolved"], (
            f"the locator {skill['locator']!r} did not resolve through the "
            f"printed roots table {skill['roots']} (R8′: it is root-relative)")
        assert skill["resolved"].endswith("skills/multi-llm/SKILL.md")
        assert "/plugins/cache/" in skill["resolved"], (
            "the advertised skill is not the installed plugin's — a shim path "
            "would be the other route's evidence")

    def test_tools_carries_the_namespaced_tool(self, package_probe):
        _only(package_probe, "loader-tools", "scripted-call")
        exec_leg = _probe_record(package_probe)["exec"]
        assert exec_leg["declarations"], (
            "the stub saw no tools[] entry for mcp__multi_llm -> multi_llm")
        declaration = exec_leg["declarations"][0]
        assert declaration["type"] == "namespace"
        assert declaration["name"] == "mcp__multi_llm"
        names = [t.get("name") for t in declaration.get("tools") or []]
        assert "multi_llm" in names, declaration


class TestTheScriptedCallRoundTrips:
    """Item 1a: the mechanism, end to end — never a model choosing."""

    def test_the_call_was_served_and_its_output_came_back(self, package_probe):
        _only(package_probe, "scripted-call")
        exec_leg = _probe_record(package_probe)["exec"]
        assert "call" in exec_leg["scripted_phases"], exec_leg
        assert "terminal" in exec_leg["scripted_phases"], (
            "codex never sent the function_call_output back — the server's "
            "result did not re-enter the transcript")
        returned = exec_leg["returned_output"]
        _assert_status_envelope(
            _tool_envelope(returned),
            "the output that came back is not the dispatcher's result", returned)


class TestTheRunnabilityRow:
    """Item 8 — the whole chain from a Python stdio client."""

    def test_status_reaches_a_result_at_nothing_to_do(self, package_probe):
        _only(package_probe, "runnability")
        stdio = _probe_record(package_probe)["stdio"]
        assert stdio, "no installed root to drive"
        listed = [t.get("name") for t in
                  ((stdio["tools_list"].get("result") or {}).get("tools") or [])]
        assert "multi_llm" in listed
        response = stdio["calls"][0]["response"]
        assert (response.get("result") or {}).get("isError") is False, response
        _assert_status_envelope(_tool_envelope(response), "status", response)

    def test_a_partial_payload_names_the_missing_lock(self, package_probe):
        _only(package_probe, "runnability")
        record = _probe_record(package_probe)
        partial = record.get("partial_payload")
        if not partial:
            pytest.skip(f"not driven: {record['legs_not_driven']}")
        listed = [t.get("name") for t in
                  ((partial["stdio"]["tools_list"].get("result") or {})
                   .get("tools") or [])]
        assert "multi_llm" in listed, "the tool must stay listed (D3 rule 4)"
        assert partial["removed"] in json.dumps(partial["stdio"]["calls"][0])


class TestTheSeptember24FixesHoldInTheImage:
    """The fixes of 2026-09-24 (plans/codex-native-plugin.md, "Fixes for the
    issues rows 1b and 1-14 found"), each asserted on the row that found the
    issue. These are the plan's own green branch, so a red here is a
    regression of a fix, not a recorded answer."""

    def test_the_cd_workspace_is_the_anchor_via_turn_metadata(self, package_probe):
        _only(package_probe, "anchor")
        reading = _probe_record(package_probe)["reverified"]["cd_flag"]
        assert reading["anchor_source"] == "turn-metadata", reading
        assert reading["anchor_is_expected"], reading
        assert reading["child_cwd"]["shell"] == reading["expected_anchor"], reading
        assert reading["pwd_at_launch"] != [reading["expected_anchor"]], (
            "the leg must launch with PWD naming a DIFFERENT checkout", reading)

    def test_a_pwd_less_launch_anchors_via_metadata(self, package_probe):
        _only(package_probe, "anchor")
        reading = _probe_record(package_probe)["reverified"]["without_pwd"]
        assert reading["pwd_at_launch"] == [None], reading
        assert reading["anchor_source"] == "turn-metadata", reading
        assert reading["state"] != "needs-human", reading

    def test_no_bytecode_lands_in_the_plugin_cache(self, package_probe):
        _only(package_probe, "anchor")
        record = _probe_record(package_probe)
        assert record["plugin_pycache_writes"] == [], record["plugin_pycache_writes"]
        assert record["init_exec_effects"]["plugins_subtree_changed"] == []

    def test_sandboxed_sessions_reach_the_server_under_the_shipped_default(
            self, package_probe):
        _only(package_probe, "sandbox-posture")
        record = _probe_record(package_probe)
        for key in ("read_only", "workspace_write"):
            leg = record[key]
            assert leg["server_got_tools_call"], (key, leg.get("tool_call_item"))
            assert not leg["requires_approval_refusal"], key

    def test_read_only_refuses_an_apply_mode_before_any_spawn(self, package_probe):
        _only(package_probe, "sandbox-posture")
        leg = _probe_record(package_probe)["refusal_read_only"]
        envelope = leg["envelope"]
        assert leg["server_got_tools_call"], leg.get("tool_call_item")
        assert envelope["state"] == "needs-human" and envelope["exit"] is None
        assert "read-only" in envelope["message"], envelope["message"]
        assert leg["dispatcher_processes"] == [], leg["dispatcher_processes"]
        assert leg["porcelain_byte_identical"], (leg["porcelain_before"],
                                                 leg["porcelain_after"])

    def test_the_lever_dispatches_while_the_session_shell_stays_sandboxed(
            self, package_probe):
        _only(package_probe, "sandbox-posture")
        leg = _probe_record(package_probe)["lift_read_only"]
        assert leg["spawned_dispatcher"], leg["dispatcher_processes"]
        assert "read-only" not in str(leg["envelope"].get("message") or ""), (
            leg["envelope"])
        # Recorded, not required: whether the session's own exec dies on bwrap.
        assert "session_shell" in leg

    def test_workspace_write_does_not_refuse_an_apply_mode(self, package_probe):
        _only(package_probe, "sandbox-posture")
        leg = _probe_record(package_probe)["apply_workspace_write"]
        assert leg["spawned_dispatcher"], leg["dispatcher_processes"]
        assert "sandbox posture" not in str(leg["envelope"].get("message") or "")

    def test_the_shell_ladder_refuses_with_no_child(self, package_probe):
        _only(package_probe, "environment")
        leg = _probe_record(package_probe)["shell_ladder_refusal"]
        assert leg["envelope"]["ok"] is False and leg["envelope"]["exit"] is None
        assert leg["message_names_bash_and_path"], leg["envelope"]["message"]
        assert leg["login_shell"] is False and leg["shell"] is None
        assert leg["environment_keys"] == sorted(
            ["platform", "shell", "login_shell", "profile_shell", "interpreter",
             "missing"])
        assert leg["dispatcher_processes"] == [], leg["dispatcher_processes"]

    def test_a_missing_python3_is_named_by_the_check(self, package_probe):
        _only(package_probe, "environment")
        leg = _probe_record(package_probe)["interpreter_absent"]
        assert leg["python3_resolves"] is None
        assert leg["exec"]["declarations"] == [], "tools[] must carry no multi_llm"
        assert not leg["exec"]["multi_llm_in_tools"]
        checks = leg["install_host_check"]
        assert checks, "install-host codex --check was not run"
        for name, check in checks.items():
            assert check["names_missing_interpreter"], (name, check["output"])

    def test_every_leg_is_driven(self, package_probe):
        _result, probe = package_probe
        if probe["probe"] not in ("anchor", "environment", "progress-cancellation",
                                  "plugin-data", "sandbox-posture", "timeout"):
            pytest.skip(f"{probe['id']} names no leg list")
        record = _probe_record(package_probe)
        assert record["legs_not_driven"] == {}, record["legs_not_driven"]


class TestTheRecordedProbes:
    """Items 7(a), 9, 10, 11, 12(a), 13, 14: done when RECORDED (item 15)."""

    def test_the_probe_record_exists(self, package_probe):
        _result, probe = package_probe
        if probe.get("blocking"):
            pytest.skip(f"{probe['id']} is a blocking row, graded above")
        record = _probe_record(package_probe)
        assert record["probe"] == probe["probe"]
        for name in probe.get("records") or []:
            assert (package_probe[0].out_dir / name).is_file(), (
                f"{probe['id']}: declared record {name} is missing")


@pytest.fixture
def codex_package_ask(graded):
    result, matrix_row, entry = graded
    if matrix_row.get("id") != "codex-package-ask":
        pytest.skip(f"{matrix_row.get('id')} is not the codex package row")
    for reason in (_plugin_tree_skip_reason(), _codex_floor_skip_reason()):
        if reason:
            pytest.skip(reason)
    return result, matrix_row, entry


class TestTheCodexPackageAskUsedThePlugin:
    """Item 16's route-specific half; the workflow half is graded above."""

    # `live`, like every other class graded off the credentialed fixture: the
    # lane selects `-m "container and live"`, so without it these four never
    # ran on the only invocation that can produce their evidence.
    pytestmark = pytest.mark.live

    def test_the_plugin_was_installed_and_no_shim_staged(self, codex_package_ask):
        result, _row, _entry = codex_package_ask
        install = _bundle_json(result, "plugin-install.json")
        assert install and install["installed_root_exists"], install
        assert install["file_shim_present"] is False

    def test_the_key_was_exported_into_an_unshadowed_profile(self, codex_package_ask):
        result, _row, _entry = codex_package_ask
        export = _bundle_json(result, "profile-export.json")
        assert export, "no profile-export.json: D3a's second carrier was not written"
        assert export["form"] == "export"
        assert export["exported"] == ["OPENROUTER_API_KEY"]
        assert export["shadowed_by"] is None

    def test_the_host_called_the_tool_rather_than_a_shell_line(
            self, codex_package_ask):
        result, _row, _entry = codex_package_ask
        stdout = (result.out_dir / "host-stdout.txt")
        assert stdout.is_file(), "no host-stdout.txt"
        tool_items = []
        for line in stdout.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item") or {}
            if item.get("type") in ("command_execution", "agent_message",
                                    "reasoning", None):
                continue
            if "multi_llm" in json.dumps(item):
                tool_items.append(item.get("type"))
        assert tool_items, (
            "the codex session never emitted a multi_llm tool item — it "
            "narrated or ran a shell line instead, which is the route's red "
            "(measured-red, not unmeasured, per the matrix header)")

    def test_the_servers_child_took_the_key_from_the_profile(
            self, codex_package_ask):
        """D3a: the login-shell child derives the key from `.profile`, never
        from the session's environment. Presence booleans only
        (`entrypoint.wrap_profile_with_read_probe`); no value is recorded."""
        result, _row, _entry = codex_package_ask
        path = result.out_dir / "profile-reads.jsonl"
        assert path.is_file(), "no profile-reads.jsonl: no login shell read .profile"
        reads = [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
        child = [r for r in reads if "mcp/server.py" in (r.get("parent") or "")]
        assert child, (
            f"no login shell whose parent is mcp/server.py read .profile: "
            f"{[r.get('parent') for r in reads]}")
        assert all(r["key_before_profile"] is False for r in child), child
        assert all(r["key_after_profile"] is True for r in child), child
