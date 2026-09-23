"""Provider lane, the whole chain: one `full --quick --yes --exec-subagents` per CLI.

`test_container_provider_lane.py` runs **one phase** per CLI and answers the
question "does the real binary still accept our argv and emit a stream we
parse?". This module runs the chain the product actually ships —
review-plan -> apply-suggestions -> generate-tasks -> review-tasks ->
apply-task-suggestions -> implement -> review-code -> apply-code-fixes, then a
standalone `ask` and a `status` read-back — and answers a different question:
*does every phase's parser, validator, batcher and scheduler still agree with a
real CLI's output, end to end, in one process?*

Three things about this row are not optional, and each is a plan finding rather
than a preference:

**`--exec-subagents`, and `--yes` does not imply it.** `full` runs every phase
inside one process, so nothing sits between a phase and the next one to execute
the batches a write phase emits. `full_workflow_orchestrator._unexecuted_batches`
detects exactly that and the run ends at `[WRITE_PHASE_NOT_EXECUTED]` with exit
3 — a chain run without the flag does not "mostly pass", it stops at the first
apply phase. :class:`TestTheFlagIsDoingTheWork` is that negative control.

**Both driver keys, not just the obvious one.** `tests/container/providers.tmpl.yaml`
renders `driver.subagent_runner` *and* `driver.subagent_runner_cheap`, because an
unset cheap key keeps base's `claude-code:haiku` and adds a second binary to
every row. Only `claude-code`, `codex` and `opencode` declare `agent-write`, so
the other **13** rows name their write runner in `redirect.yaml`'s
`chain_write_runner` and carry it as the row's one allowed helper. (The count
read "eight" until 2026-09-21 and was wrong by five: it predates `omp` and the
other late arrivals. Counted from `redirect.yaml`, the reviewer axis is 16 rows
and 13 of them declare a helper.)
``tests/test_redirect_matrix_claims.py`` cross-checks that key against the
adapters themselves, in **plain CI** — a drift guard living in this module would
only run where Docker runs, which is not where the drift gets introduced. Its
name deliberately does *not* start with `test_container_`: that prefix is what
the image's own glob and the marker-gating check select on, and a plain-CI guard
must not be swept into either.

**Every write batch must land on `no-op`, and that is asserted in both
directions.** The stub emits no tool call, so no redirected CLI on this lane can
produce a byte change: the four write phases complete on the documented no-op
(`status: ok`, `write_outcome: no-op`, zero files touched, loop continues,
finalization in full, phase completed at exit 0). A batch that reads `applied`
here is a bug in the rig, not a pass — and so is a write phase that resolves to
the "nothing to apply" skip, which is why the fixture chain is chosen to give
each apply phase real batches to emit rather than a vacuous green.

*CORRECTED 2026-09-22, FROM `clean-failure`, AND THE WORD MOVED IN THE PRODUCT
RATHER THAN IN THE RIG.* `1633dd4` (2026-09-16, "Tell a no-op from a failure, and
track the files the write executor verified") split
`utils/write_batch_scheduler`'s empty-diff branch, which until then had ONE
verdict for three claims. An `applied`-or-no-status claim over an empty diff is
still `clean-failure` at `status: ok`; a `failed` claim or a non-zero exit over
an empty diff is `clean-failure` at `STATUS_BATCH_FAILED`; and a **`no-op` claim
over an empty diff is now `WRITE_NO_OP` at `status: ok`**. Why, in the constant's
own docstring (`utils/subagent_executor.WRITE_NO_OP`): a clean failure is
*success* claimed with nothing moved, so labelling an honest "already present"
batch `(batch failed — …)` in the prior-changes log told the NEXT batch, and the
reader, that the work had failed. What this lane's stub CLAIMS never changed —
all four fixtures under `fixtures/e2e/responses/write_batches/` read
`{"status": "no-op", …}` and have since `f62cc89` (2026-08-15), one commit under
`git log --follow` and untouched by today's tree — so this docstring was written
to the pre-`1633dd4` verdict of an unchanged claim and outlived it by six days.
Measured on `runs/20260921T224840Z`: 130 write batches across every chain CLI,
`write_outcome: no-op` on all 130, in `write_batch_report.json` and in
`subagent_batch_report.json` alike.

*THE REDIRECT-LEAK CLAIM SURVIVES THE RENAME*, which is what makes the word safe
to move rather than load-bearing to keep. `WRITE_NO_OP` is assigned ONLY inside
the classifier's `if not changed:` branch — the same branch both
`WRITE_CLEAN_FAILURE` assignments live in — while `WRITE_APPLIED` and
`WRITE_PARTIAL` are reachable only where the pre/post snapshot diff is NON-empty.
What disproves a leak is therefore the row refusing `applied` and
`partial-application`, never which of the two empty-diff words it names, and
`no-op` excludes both exactly as strictly. It is in fact the STRICTER of the two
here: `no-op` requires the runner's claim and the byte verdict to AGREE, where
`clean-failure` at `status: ok` is success CLAIMED with nothing moved — a
claim/byte mismatch a tool-call-free stub whose document says `no-op` cannot
produce.

**Every structural assertion above stays green through output-shape drift, so the
row is also graded against a normalized golden bundle (C8).** "Phases completed",
"batches recorded no-op" and "the census moved nothing" are all satisfied
by a run whose parsed suggestions silently halved: the phases still complete, the
batches still dispatch, and no byte moves either way. So each row's exported
artifacts are folded into the bundle `tests/golden/container_bundle.py` defines
and diffed against `tests/golden/container/<cli>/bundle.json` at a threshold of
**zero**, with `tests/golden/container/expected_changes.yaml` — which ships empty —
as the only way to declare a difference. Per CLI, because the parse under test is
per CLI; captured by `tests/container/seed_container_golden.py` on the host from a
green run, never by this module, because a guard that can rewrite its own baseline
is not a guard.

**...which is why the applied branch gets its own row, right here.** Requiring
`no-op` from every chain batch and calling `applied` a rig bug leaves the
*other* half of the write scheduler — the one that decides bytes really landed —
with no in-container coverage at all, so a wrapper or executor regression that
prevents real file edits would pass this module in full. C4's option (a) is that
coverage: :class:`TestOneWriteBatchReallyMovedBytes` runs `apply-suggestions`
**once**, for the whole lane rather than per CLI, with `driver.subagent_runner`
pointed at the edit-capable fake (`tests/mocks/mock_llm.py` under
`MOCK_LLM_EDITS`) instead of at a redirected CLI. The two rows are each other's
negative control: the same scheduler, the same pre-batch snapshot, the same
`write_batch_report.json`, and each fails if it lands on the other's outcome.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
from .golden import container_bundle
from .golden.bundle_compare import (
    AllowlistError,
    format_unexplained,
    load_allowlist,
    partition,
)

pytestmark = [pytest.mark.container, requires_docker()]

REDIRECT = harness.HERE / "redirect.yaml"

#: In-image path. The image ships tests/fixtures/e2e/ because the stub resolves
#: a scenario's `fixture:` paths relative to the scenario file.
SCENARIO = "/opt/mllm/skill/tests/fixtures/e2e/scenarios/container_full_chain.yaml"

PLAN = "plans/auth-feature.md"

#: The chain, then the two read-only follow-ups Phase 3 asks for. `status
#: --view workflow` and not `--view all`: the run view reports "no run recorded"
#: as exit 2 for a foreground run, which is the documented non-terminal code and
#: not a failure — asserting on it would make the row fail for being green.
CHAIN_COMMANDS = [
    f"full {PLAN} --quick --yes --exec-subagents",
    f"ask {PLAN} 'What is the weakest part of the rate limiter?' --quick --yes",
    f"status {PLAN} --view workflow",
]

#: Every phase `full --quick --yes` runs, in order, as `state.json` spells them.
#: Note the one place mode word != phase name: mode `review-code`, phase
#: `code-review`.
CHAIN_PHASES = (
    "review-plan",
    "apply-suggestions",
    "generate-tasks",
    "review-tasks",
    "apply-task-suggestions",
    "implement",
    "code-review",
    "apply-code-fixes",
)

#: The four write phases and the report each one must leave behind.
WRITE_PHASE_DIRS = {
    "apply-suggestions": "apply-suggestions",
    "apply-task-suggestions": "apply-task-suggestions",
    "implement": "implement",
    "apply-code-fixes": "apply-fixes",
}


def _rows():
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return [r for r in data["clis"] if r.get("lane") == "provider"]


VERIFIED = _rows()
IDS = [r["cli"] for r in VERIFIED]
BY_CLI = {r["cli"]: r for r in VERIFIED}


def _spec(cli: str) -> str:
    return f"{cli}:{BY_CLI[cli].get('model_arg', 'stub-model')}"


def _runner_cli(entry: dict) -> str:
    """The binary the executor dispatches every subagent through, for this row."""
    return entry.get("chain_write_runner") or entry["cli"]


# --------------------------------------------------------------------------- #
# The writer axis
#
# The reviewer axis above asks one question per row and grades two things with
# it: *does this CLI's reviewer parse still hold across eight phases*, and *does
# the write path still work*. Those want different parametrizations, and
# conflating them is why codex's write path was graded 14 times while opencode's
# and claude-code's were graded once each — and why "non-write CLI as reviewer +
# opencode as writer" had never run at all.
#
# So: three more rows, the reviewer held FIXED (`pi`, pinned once in
# `redirect.yaml`'s `chain_writer_axis:` block so it cannot drift per row) and
# the WRITE RUNNER varied. Same `full --quick --yes --exec-subagents`, same
# scenario, same fixture plan. The axis is read from that block rather than
# from `clis:` for four reasons that each break a different consumer — see
# §5.1 of `plans/chain-lane-write-runner-matrix.md` and the block's own comment.
# --------------------------------------------------------------------------- #

def _writer_axis() -> list[dict]:
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return list(data.get("chain_writer_axis") or [])


WRITER_AXIS = _writer_axis()
WRITER_IDS = [f"writer-{entry['writer']}" for entry in WRITER_AXIS]

#: Each write-capable CLI's own containment flag, as §0 of the plan measures
#: them off the shipped adapters. This is Tier 1 of the writer axis's grade:
#: without it, "all three write-capable CLIs run all phases" would mean only
#: that three binaries spawned, returned the canned document and were recorded
#: `no-op` — plumbing coverage, and the same "nothing was really graded"
#: criticism the reviewer axis already attracts. (The word was `clean-failure`
#: here until 2026-09-22; `1633dd4` moved it in the classifier, not in the rig —
#: the module docstring carries the derivation. Note the collision that
#: correction creates and does not resolve: `no-op` is now BOTH the verdict this
#: lane earns and the informal word for a test that grades nothing, so the
#: criticism is spelled out longhand above rather than named.)
#:
#: codex's is the only OS boundary of the three: `--sandbox workspace-write` is
#: bubblewrap. opencode's `--agent build` and claude-code's
#: `--permission-mode acceptEdits` are application-level, which is exactly why
#: neither has anything that can fail to start.
WRITER_CONTAINMENT_FLAGS = {
    "codex": ("--sandbox", "workspace-write"),
    "opencode": ("--agent", "build"),
    "claude-code": ("--permission-mode", "acceptEdits"),
}


def _chain_row(entry: dict, *, exec_subagents: bool = True) -> harness.Row:
    cli = entry["cli"]
    runner = _runner_cli(entry)
    helpers = [] if runner == cli else [runner]
    commands = list(CHAIN_COMMANDS)
    if not exec_subagents:
        commands = [commands[0].replace(" --exec-subagents", "")]
    return harness.Row(
        row_id=f"chain-{cli}" if exec_subagents else f"chain-noexec-{cli}",
        lane="provider",
        cli=cli,
        spec=[_spec(cli)],
        # BOTH driver keys come from this one value (seed_workspace defaults
        # `--runner-cheap` to `--runner`), which is what keeps base's
        # claude-code:haiku out of every row.
        runner=_spec(runner),
        helpers=helpers,
        command=commands[0],
        commands=commands[1:],
        scenario=SCENARIO,
        row_timeout_s=int(entry.get("chain_row_timeout_s",
                                    max(600, int(entry.get("row_timeout_s", 300)) * 2))),
    )


# --------------------------------------------------------------------------- #
# The chain row
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


#: One in-image `agent-write` verdict per CLI, measured on first use and shared
#: by every row that dispatches its writes through that CLI.
#:
#: The verdict is a property of the IMAGE and the ADAPTER, never of the row, and
#: each measurement is a container start: the 13 reviewer rows that borrow one
#: shared helper would otherwise probe the same binary 13 times to learn the same
#: thing. Cached here rather than per row, and measured for EVERY row's effective
#: write runner rather than only the declared ones — the `ok`/`not-required`
#: branch is what makes a `sandbox-skip` outcome on a capable host a failure
#: rather than a free pass, and that check needs a verdict on rows that never
#: declared one.
_SANDBOX_VERDICTS: dict[str, tuple[str, str]] = {}


def _sandbox_verdict(cli: str) -> tuple[str, str]:
    """``(state, tail)`` for this CLI's `agent-write` sandbox, inside the image."""
    if cli not in _SANDBOX_VERDICTS:
        _SANDBOX_VERDICTS[cli] = harness.sandbox_verdict_in_image(
            cli, "agent-write")
    return _SANDBOX_VERDICTS[cli]


@pytest.fixture(scope="module", params=VERIFIED, ids=IDS)
def chain(request, posture, utc):
    """Run one CLI's whole chain once and share its evidence across the asserts.

    The row's **effective write runner** is probed inside the image on this same
    run, exactly as the writer axis probes its writer. A reviewer row dispatches
    four write batches too, so a row whose runner cannot start its sandbox is
    refused here as surely as on the other axis: `chain-codex` needs the same
    licence — a strict, typed xfail — or it is the bare red §3.2 forbids, while
    `matrix-census.json` calls the very same evidence `sandbox-skip`.
    """
    entry = request.param
    result = harness.run_row(_chain_row(entry), posture=posture, utc=utc)
    runner = _runner_cli(entry)
    state, tail = _sandbox_verdict(runner)
    harness.record_sandbox_verdict(
        [result.out_dir, result.out_dir.parent], runner, "agent-write",
        state, tail)
    return result


def _writer_row(entry: dict) -> harness.Row:
    """One writer-axis row: the pinned reviewer, this entry's write runner.

    The reviewer half is what `cli` and `spec` name — it is the CLI that
    answers the eight-phase fan-out — and the WRITER half is `runner`. That
    split is also what makes the three `row_id`s and their baseline paths
    distinct: one shared reviewer cannot collide them, and none of the three
    collides with the reviewer axis's own `chain-pi`.
    """
    writer, reviewer = entry["writer"], entry["reviewer"]
    review_entry = BY_CLI[reviewer]
    return harness.Row(
        row_id=f"chain-writer-{writer}",
        lane="provider",
        cli=reviewer,
        spec=[_spec(reviewer)],
        runner=_spec(writer),
        # The writer is a second real binary in the container, so it is the
        # row's one declared helper — exactly as a reviewer row declares its
        # borrowed `chain_write_runner`. A writer that IS the reviewer would
        # need none, which cannot happen here: the pin is constrained to a CLI
        # declaring no `agent-write`.
        helpers=[] if writer == reviewer else [writer],
        command=CHAIN_COMMANDS[0],
        commands=CHAIN_COMMANDS[1:],
        scenario=SCENARIO,
        # Priced off the reviewer's own measured chain, which bounds the row
        # above: the writer's four batches run inside it.
        row_timeout_s=int(review_entry.get(
            "chain_row_timeout_s",
            max(600, int(review_entry.get("row_timeout_s", 300)) * 2))),
    )


class WriterRun:
    """One writer-axis row's evidence, plus the sandbox verdict that licenses it.

    The two are carried together on purpose. The *classification* — did this
    row's batches record `sandbox_unavailable` — is `harness.row_outcome`'s,
    derived from the row's own evidence in one place. The *licence* to forgive
    it is a capability re-measured inside the image on this same run. Keeping
    them in one object is what makes "the classification and the capability
    disagree" an assertable condition rather than two facts nobody compares.
    """

    def __init__(self, entry: dict, result: harness.RowResult,
                 verdict: tuple[str, str]):
        self.entry = entry
        self.result = result
        self.verdict = verdict

    @property
    def writer(self) -> str:
        return str(self.entry["writer"])

    @property
    def outcome(self) -> str:
        return harness.row_outcome(self.result)[0]


@pytest.fixture(scope="module", params=WRITER_AXIS, ids=WRITER_IDS)
def writer_chain(request, posture, utc):
    """Run one writer's whole chain once, and measure its sandbox on the same run."""
    entry = request.param
    row = _writer_row(entry)
    result = harness.run_row(row, posture=posture, utc=utc)
    # Measured for EVERY writer, not only the ones declaring a skip: the
    # `ok`/`not-required` branch is what makes a `sandbox-skip` outcome on a
    # capable host a failure rather than a free pass, and that check needs a
    # verdict on rows that never declared one.
    verdict = harness.sandbox_verdict_in_image(entry["writer"], "agent-write")
    harness.record_sandbox_verdict(
        [result.out_dir, result.out_dir.parent], entry["writer"],
        "agent-write", *verdict)
    return WriterRun(entry, result, verdict)


def _audit_lines(result: harness.RowResult) -> list[dict]:
    """Every `subagent_audit.jsonl` record this row's phases wrote, in order."""
    lines: list[dict] = []
    for path in sorted(_artifacts(result).glob("*/subagent_audit.jsonl")):
        lines += harness._read_jsonl(path)
    return lines


def _argv_carries(argv, flag: str, value: str) -> bool:
    """Did this argv really pass ``flag value`` — adjacent, or as ``flag=value``?

    Adjacency and not "the flag is in the list and the value is somewhere in
    the line", because both evidence sources carry a SECOND argv for the same
    flag: opencode's read dispatch is `--agent plan` beside the write's
    `--agent build`, and codex's read probe is `--sandbox danger-full-access`
    beside the write's `--sandbox workspace-write`. A loose match reads the
    write's containment off the read's line.
    """
    tokens = [str(token) for token in (argv or [])]
    for index, token in enumerate(tokens):
        if token == f"{flag}={value}":
            return True
        if token == flag and index + 1 < len(tokens) and tokens[index + 1] == value:
            return True
    return False


def _calls_by(result: harness.RowResult, cli: str) -> list[dict]:
    """Every `/opt/mllm/bin` call record this row's CLI really crossed."""
    return [record for record in result.call_log if record.get("cli") == cli]


class TestTheWriterAxisGradesTheWritePath:
    """Three rows, reviewer fixed, write runner varied (G1).

    The reviewer axis's 16 rows grade a parse; these grade a write path. They
    are the rows that make "all three write-capable CLIs run all phases" a
    measurement rather than a sentence.
    """

    #: The two Tier 1 tests, which a sandbox refusal must NOT excuse (§8).
    #: Every other test here answers "did the chain run", which a refused row
    #: did not; these two answer "was the write contained", which a refused row
    #: still has evidence for — and is the only grade `chain-writer-codex`
    #: gets, since it has no golden here or anywhere. Named as data rather than
    #: as a marker so the licence and its exception are read in one place.
    GRADED_ON_REFUSAL = frozenset({
        "test_the_rendered_write_argv_carried_the_containment_flag",
        "test_the_shim_receipt_agrees_the_writer_really_ran",
    })

    @pytest.fixture(autouse=True)
    def _sandbox_policy(self, request, writer_chain):
        """Where the licence to forgive an unrunnable sandbox is decided.

        Not a `pytest.skip`: a skipped row is never executed, so it can never
        announce that its blocker has lifted. Not a test-side environment
        gate either: that re-derives, badly, what the dispatch already knows.
        The permission comes from `sandbox_verdict('agent-write')` run inside
        the image on THIS run, and it is spent as a strict, typed xfail.
        """
        state, tail = writer_chain.verdict
        declared = writer_chain.entry.get("sandbox_skip_when")
        if state == "unknown":
            pytest.fail(
                f"{writer_chain.writer}: the in-image sandbox probe could not "
                f"answer ({tail!r}). `unknown` is a hard failure, not a "
                f"licence — nothing was learned about the sandbox, and an "
                f"unreadable verdict must not become the cheap path to a "
                f"green run."
            )
        if state == "unavailable":
            assert declared, (
                f"{writer_chain.writer}: the image's own probe says its "
                f"agent-write sandbox is unavailable, but redirect.yaml's "
                f"chain_writer_axis entry declares no sandbox_skip_when. "
                f"Declare it or fix the sandbox; an undeclared refusal is a "
                f"bare red by another name."
            )
            if request.node.function.__name__ in self.GRADED_ON_REFUSAL:
                # The two Tier 1 tests are the exception, and deliberately so:
                # §8 grades `chain-writer-codex` on PROPERTIES because it has
                # no golden anywhere, and these are the properties. They are
                # written to pass on a refused row, so marking them
                # strict-xfail here would turn the grading §8 requires into an
                # XPASS red — the licence would cancel the one thing it exists
                # to leave running.
                return
            request.node.add_marker(pytest.mark.xfail(
                strict=True,
                raises=harness.SandboxSkipped,
                reason=(f"{writer_chain.writer} agent-write sandbox "
                        f"unavailable in-image: {tail.strip()[:200]}"),
            ))
        # `ok` / `not-required`: NO marker is added at all. The row is graded
        # like every other row, and the check below turns the disagreement
        # into a failure rather than a silent pass.

    def _refuse_if_skipped(self, writer_chain) -> None:
        """Raise the typed failure the xfail above is declared against."""
        state = writer_chain.verdict[0]
        outcome, reason = harness.row_outcome(writer_chain.result)
        if outcome != "sandbox-skip":
            return
        assert state == "unavailable", (
            f"{writer_chain.writer}: the row's own batches recorded "
            f"reason_code=sandbox_unavailable, but the image's probe answered "
            f"{state!r} on this run. The classification and the capability "
            f"disagree, and one of them is wrong — this is a failure, not a "
            f"skip."
        )
        raise harness.SandboxSkipped(
            f"{writer_chain.writer}: {reason}"
        )

    def test_the_row_did_not_hit_its_cap(self, writer_chain):
        self._refuse_if_skipped(writer_chain)
        assert not writer_chain.result.timed_out, (
            f"{writer_chain.writer}: {writer_chain.result.cap_reason} after "
            f"{writer_chain.result.wall_clock_s}s"
        )

    def test_the_row_exited_zero(self, writer_chain):
        self._refuse_if_skipped(writer_chain)
        assert writer_chain.result.exit_code == 0, (
            f"{writer_chain.writer}: exited {writer_chain.result.exit_code}\n"
            f"--- container stderr (tail) ---\n"
            f"{writer_chain.result.stderr[-3000:]}"
        )

    def test_state_json_records_every_phase_completed(self, writer_chain):
        self._refuse_if_skipped(writer_chain)
        state = _state(writer_chain.result)
        completed = set(state.get("phases_completed") or {})
        missing = [p for p in CHAIN_PHASES if p not in completed]
        assert not missing, (
            f"{writer_chain.writer}: {missing} not recorded completed "
            f"(recorded: {sorted(completed)})"
        )

    def test_the_rendered_write_argv_carried_the_containment_flag(
            self, writer_chain):
        """Tier 1 of Q4: the PROFILE is graded, not just the exit code.

        Without this the axis would assert only that three binaries spawned,
        returned the canned document and were recorded `no-op` (`clean-failure`
        until 2026-09-22 — see the module docstring) — which passes just as well
        when every real CLI's edit tooling is broken.

        Read from the dispatch's own `subagent_audit.jsonl` line. That line's
        argv is a re-render through the same builder
        (`write_batch_scheduler._render_argv`, "never executed here"), so on
        its own it proves the profile-to-flag mapping and NOT the process that
        ran — which is why the shim-receipt cross-check below is not
        belt-and-braces.

        **No `_refuse_if_skipped` here, unlike the exit/phase/bundle tests.**
        A sandbox refusal excuses "the chain ran"; it does not excuse "the
        write was contained", and §8 grades `chain-writer-codex` on exactly
        these properties because it has no golden to be compared against. What
        the refusal changes is only *which* evidence exists — see
        `_grade_the_refused_containment`.
        """
        flag, value = WRITER_CONTAINMENT_FLAGS[writer_chain.writer]
        if writer_chain.outcome == "sandbox-skip":
            self._grade_the_refused_containment(writer_chain, flag, value)
            return
        lines = _audit_lines(writer_chain.result)
        assert lines, (
            f"{writer_chain.writer}: no subagent_audit.jsonl records under "
            f"{_artifacts(writer_chain.result)} — the axis graded nothing"
        )
        carrying = [
            line for line in lines
            if _argv_carries(line.get("argv"), flag, value)
        ]
        assert carrying, (
            f"{writer_chain.writer}: no audit record's rendered argv carried "
            f"{flag} {value}. Recorded argvs: "
            f"{[line.get('argv') for line in lines[:4]]}"
        )

    def _grade_the_refused_containment(self, writer_chain, flag, value) -> None:
        """Tier 1 on a row that refused before it rendered anything.

        `preflight_runner` raises `RunnerRejected` **above**
        `AuditLog.dispatched`, so a refused row leaves no rendered argv at all
        — measured on `runs/20260921T224840Z/chain-writer-codex`: no
        `subagent_audit.jsonl` under its artifacts and a
        `write_batch_report.json` of `{"exit": 3, "batches_dispatched": [],
        "tasks": []}` (the same shape `harness.sandbox_refusals` documents).
        Asserting the audit argv here would fail the row for obeying its own
        refusal.

        Two properties survive, and both are graded rather than assumed:

        * the refusal itself, in the product's words (`reason_code:
          sandbox_unavailable`), and
        * the POLICY it was refused on, off the `/opt/mllm/bin` call log:
          codex's probe crosses the shim as `codex sandbox -c
          sandbox_mode="workspace-write"`, naming the same policy
          `--sandbox workspace-write` would have dispatched under.

        The policy is read off the writer's **last** call and not off any call
        naming it, because the read-only probe names its own policy in the
        same log: the refusal ends the run where it is raised, so the write
        profile's probe is by construction the last thing that CLI did. A row
        refused on some other policy — or after the write probe passed — puts
        a different argv last and fails here.
        """
        refusals = harness.sandbox_refusals(writer_chain.result)
        assert refusals, (
            f"{writer_chain.writer}: the row is classified `sandbox-skip`, but "
            f"nothing in its own evidence records reason_code="
            f"sandbox_unavailable. A refusal nobody wrote down is not a graded "
            f"property."
        )
        calls = _calls_by(writer_chain.result, writer_chain.writer)
        assert calls, (
            f"{writer_chain.writer}: refused with reason_code="
            f"sandbox_unavailable, but /opt/mllm/bin never saw it run at all — "
            f"a refusal no process was measured for."
        )
        last = " ".join(str(a) for a in (calls[-1].get("argv") or []))
        assert value in last, (
            f"{writer_chain.writer}: the last thing it did before the row "
            f"refused was {calls[-1].get('argv')!r}, which does not name "
            f"{value!r} — so nothing shows the refusal was taken on the "
            f"containment {flag} {value} dispatches under."
        )

    def test_the_bundle_reproduces_its_writer_baseline(self, writer_chain):
        """The writer axis's own golden, in the `writer-<cli>` namespace.

        One bundle per ROW, not per CLI: `capture_bundle` folds a single row's
        exported `out_dir`, so this is the reviewer's eight phases *and* the
        writer's four `write_batch_report.json` artifacts, interleaved as the
        row ran them. Same shape as a reviewer bundle, under a different key.

        `writer-codex` is exempt **by declaration** and not by an empty
        directory: it has no golden anywhere, here or on any host, so it is
        graded on properties only (§8) until a permissive-kernel host captures
        one. The exemption covers absence, never staleness — a
        `writer-codex/` that does exist is graded in full, right here.

        The declaration alone is not enough to spend it: the blocker has to
        still be there on THIS run. So the exemption is also keyed on the
        in-image verdict, and the day the probe answers `ok` the row runs
        green and the missing baseline becomes a failure carrying the capture
        command (§8) — not a standing allowance waiting for someone to
        retire the declaration by hand.
        """
        self._refuse_if_skipped(writer_chain)
        cli = writer_chain.writer
        path = container_bundle.baseline_path(cli, container_bundle.AXIS_WRITER)
        declared = writer_chain.entry.get("sandbox_skip_when")
        blocked = writer_chain.verdict[0] == "unavailable"
        if not path.is_file() and declared and blocked:
            pytest.skip(
                f"{cli} is a declared uncovered writer, and this image's own "
                f"agent-write probe still answers `unavailable`: no writer "
                f"baseline has ever been captured, here or anywhere. Graded "
                f"on properties only until a permissive-userns host runs the "
                f"row green."
            )
        assert path.is_file(), (
            f"{cli}: no writer-axis golden at {path}. This row is ungraded "
            f"until one exists — capture it from THIS green run with:\n"
            f"  {RESEED.format(cli=cli).replace('--cli', '--writer')}\n"
            f"then read the diff before committing it."
        )
        baseline_doc = container_bundle.load_baseline(
            cli, container_bundle.AXIS_WRITER)
        manifest = baseline_doc.get("manifest") or {}
        assert manifest.get("axis") == container_bundle.AXIS_WRITER, (
            f"{cli}: {path} records axis={manifest.get('axis')!r} — a reviewer "
            f"bundle filed under the writer namespace grades the wrong thing"
        )
        assert manifest.get("reviewer") == writer_chain.entry["reviewer"], (
            f"{cli}: {path} was captured against reviewer "
            f"{manifest.get('reviewer')!r}, but this axis pins "
            f"{writer_chain.entry['reviewer']!r}. A bundle captured against a "
            f"different reviewer is not comparable to its siblings."
        )
        stale = container_bundle.stale_hash_pointers(baseline_doc)
        assert not stale, (
            f"{cli}: {path} was captured while file hashes were erased "
            f"({len(stale)} `<str>` pointers, first {stale[0]}); the rig now "
            f"pins SOURCE_DATE_EPOCH and compares them by value, so it can "
            f"never reproduce — re-capture it with:\n"
            f"  {RESEED.format(cli=cli).replace('--cli', '--writer')}"
        )
        produced = container_bundle.capture_bundle(writer_chain.result.out_dir)
        diffs = container_bundle.diff_bundles(
            container_bundle.compare_payload(baseline_doc), produced)
        assert not diffs, (
            f"{cli}: the writer-axis bundle no longer reproduces its "
            f"baseline:\n" + "\n".join(str(d) for d in diffs[:20])
        )

    def test_the_shim_receipt_agrees_the_writer_really_ran(self, writer_chain):
        """The cross-check the audit line cannot make about itself.

        The audit line is a re-render; this is the `/opt/mllm/bin` call log,
        which is written by the wrapper the moment a real process crosses it.
        A row whose audit says `--agent build` and whose call log never names
        opencode has graded a string, not a dispatch.

        **The flag is cross-checked here, not only the name.** The name alone
        leaves the re-render ungraded in the one way that matters: an executed
        argv that dropped the containment flag, or carried a different value
        for it, still puts the CLI in this log. `calls.jsonl` records the argv
        (`tests/container/wrapper.py`'s `write_call_record`), so the flag the
        audit claims is required on a process that really crossed the shim.

        A refused row is graded too (no `_refuse_if_skipped`), on the converse
        property: the product's refusal ends "Nothing was dispatched", and
        this is the log that can contradict it.
        """
        flag, value = WRITER_CONTAINMENT_FLAGS[writer_chain.writer]
        named = {record.get("cli") for record in writer_chain.result.call_log}
        assert writer_chain.writer in named, (
            f"{writer_chain.writer}: the row's rendered argv claims its "
            f"containment flag, but /opt/mllm/bin never saw it run "
            f"(call log named {sorted(n for n in named if n)})"
        )
        calls = _calls_by(writer_chain.result, writer_chain.writer)
        contained = [c for c in calls if _argv_carries(c.get("argv"), flag, value)]
        if writer_chain.outcome == "sandbox-skip":
            assert not contained, (
                f"{writer_chain.writer}: the row is classified `sandbox-skip` "
                f"and the product's refusal says nothing was dispatched, but "
                f"/opt/mllm/bin saw {flag} {value} cross it "
                f"{len(contained)}x. Either the refusal is mislabelled or a "
                f"write ran under it."
            )
            return
        assert contained, (
            f"{writer_chain.writer}: the rendered audit argv carried "
            f"{flag} {value}, but no process crossing /opt/mllm/bin did — the "
            f"re-render and the dispatch disagree. Call log argvs: "
            f"{[c.get('argv') for c in calls[:6]]}"
        )


def _artifacts(result: harness.RowResult) -> Path:
    return result.out_dir / "artifacts" / "auth-feature"


def _state(result: harness.RowResult) -> dict:
    path = _artifacts(result) / "state.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class _ReviewerSandboxPolicy:
    """The reviewer axis's half of §3.2's named skip, shared by its five classes.

    The writer axis grades a write path and the reviewer axis grades a parse,
    but both dispatch four write batches through a real runner, so both need
    the same licence when that runner's sandbox cannot start. Without it
    `chain-codex` fails every assertion below as an ordinary red while
    `matrix-census.json`, reading the same evidence, calls the row
    `sandbox-skip` — the split §3.2 names as "not green, not a bare red".

    Same mechanism, same three states, and deliberately not a second copy of
    the detection: the classification is `harness.row_outcome`'s, from the
    row's own batch records, and the licence is an in-image
    `sandbox_verdict('agent-write')` re-measured on this run.
    """

    @pytest.fixture(autouse=True)
    def _sandbox_policy(self, request, chain):
        """Where the licence to forgive an unrunnable sandbox is decided.

        Not a `pytest.skip`: a skipped row is never executed, so it can never
        announce that its blocker has lifted. The permission comes from the
        probe the `chain` fixture ran inside the image, and it is spent as a
        strict, typed xfail on this row's own test item.
        """
        entry = BY_CLI[chain.row.cli]
        runner = _runner_cli(entry)
        state, tail = _sandbox_verdict(runner)
        declared = entry.get("sandbox_skip_when")
        if state == "unknown":
            pytest.fail(
                f"{chain.row.cli}: the in-image sandbox probe could not answer "
                f"for its write runner {runner} ({tail!r}). `unknown` is a hard "
                f"failure, not a licence — nothing was learned about the "
                f"sandbox, and an unreadable verdict must not become the cheap "
                f"path to a green run."
            )
        if state == "unavailable":
            assert declared, (
                f"{chain.row.cli}: the image's own probe says its write "
                f"runner {runner}'s agent-write sandbox is unavailable, but "
                f"redirect.yaml's row declares no sandbox_skip_when. Declare "
                f"it or fix the sandbox; an undeclared refusal is a bare red "
                f"by another name."
            )
            request.node.add_marker(pytest.mark.xfail(
                strict=True,
                raises=harness.SandboxSkipped,
                reason=(f"{chain.row.cli}: write runner {runner}'s agent-write "
                        f"sandbox unavailable in-image: {tail.strip()[:200]}"),
            ))
        # `ok` / `not-required`: NO marker is added at all. The row is graded
        # like every other row, and the check below turns the disagreement into
        # a failure rather than a silent pass.

    def _refuse_if_skipped(self, chain) -> None:
        """Raise the typed failure the xfail above is declared against."""
        state = _sandbox_verdict(_runner_cli(BY_CLI[chain.row.cli]))[0]
        outcome, reason = harness.row_outcome(chain)
        if outcome != "sandbox-skip":
            return
        assert state == "unavailable", (
            f"{chain.row.cli}: the row's own batches recorded "
            f"reason_code=sandbox_unavailable, but the image's probe answered "
            f"{state!r} on this run. The classification and the capability "
            f"disagree, and one of them is wrong — this is a failure, not a "
            f"skip."
        )
        raise harness.SandboxSkipped(f"{chain.row.cli}: {reason}")


class TestTheChainRan(_ReviewerSandboxPolicy):
    def test_the_row_did_not_hit_its_cap(self, chain):
        self._refuse_if_skipped(chain)
        assert not chain.timed_out, (
            f"{chain.row.cli}: {chain.cap_reason} after {chain.wall_clock_s}s "
            f"(cap {chain.row.row_timeout_s}s)"
        )

    def test_every_dispatcher_command_exited_zero(self, chain):
        """Which invocation failed, not one number for three of them."""
        self._refuse_if_skipped(chain)
        failed = [record for record in chain.commands if record["exit"] != 0]
        assert not failed, (
            f"{chain.row.cli}: {failed}\n"
            f"--- container stderr (tail) ---\n{chain.stderr[-3000:]}"
        )
        assert len(chain.commands) == len(CHAIN_COMMANDS), (
            f"{chain.row.cli}: {len(chain.commands)} of {len(CHAIN_COMMANDS)} "
            f"commands ran"
        )

    def test_the_row_exited_zero(self, chain):
        self._refuse_if_skipped(chain)
        assert chain.exit_code == 0, (
            f"{chain.row.cli}: exited {chain.exit_code}\n"
            f"--- container stderr (tail) ---\n{chain.stderr[-3000:]}"
        )

    def test_no_prompt_fell_through_to_a_default(self, chain):
        """`on_unmatched: fail`: every model turn resolved through a pattern.

        Zero `unmatched: true` records is what makes "this row exercised the
        fixture it claims to cover" an assertion rather than an inference — a
        generic answer is shaped enough to look like a passing phase.
        """
        self._refuse_if_skipped(chain)
        unmatched = [r["prompt_sha"] for r in chain.stub_log if r.get("unmatched")]
        assert not unmatched, (
            f"{chain.row.cli}: {len(unmatched)} prompt(s) matched no pattern: "
            f"{unmatched} — add the turn to container_full_chain.yaml"
        )

    def test_no_route_went_unserved(self, chain):
        self._refuse_if_skipped(chain)
        unhandled = [(r["method"], r["path"]) for r in chain.stub_log
                     if r.get("unhandled")]
        assert not unhandled, (
            f"{chain.row.cli}: the stub does not serve {unhandled} — add a "
            f"handler and the route to this CLI's redirect.yaml http_log"
        )

    def test_the_cli_actually_reached_the_stub(self, chain):
        self._refuse_if_skipped(chain)
        calls = [r for r in chain.stub_log if r.get("surface") != "health"]
        assert calls, f"{chain.row.cli}: zero requests reached the stub"


class TestEveryPhaseFinished(_ReviewerSandboxPolicy):
    def test_state_json_records_every_phase_completed(self, chain):
        """The artifact, not the exit code.

        A phase directory is created *before* the work inside it happens, and
        two of the four write phases mark themselves completed when they *emit*
        their batch document. `state.json`'s `phases_completed` is the record
        that survives both.
        """
        self._refuse_if_skipped(chain)
        state = _state(chain)
        completed = set(state.get("phases_completed") or {})
        missing = [p for p in CHAIN_PHASES if p not in completed]
        assert not missing, (
            f"{chain.row.cli}: {missing} not recorded completed in state.json "
            f"(recorded: {sorted(completed)})"
        )

    def test_no_phase_was_skipped(self, chain):
        """A skipped apply phase is a vacuous green, not a pass.

        The one legitimate skip on this chain is consolidation, which
        `full_workflow_orchestrator` records under its own step name rather than
        as a workflow phase.
        """
        self._refuse_if_skipped(chain)
        state = _state(chain)
        skipped = set(state.get("phases_skipped") or {})
        assert not (skipped & set(CHAIN_PHASES)), (
            f"{chain.row.cli}: phases skipped: {sorted(skipped & set(CHAIN_PHASES))}"
        )

    def test_the_workflow_reports_itself_complete(self, chain):
        self._refuse_if_skipped(chain)
        state = _state(chain)
        full = state.get("full_workflow") or {}
        assert full.get("current_phase") == "complete", (
            f"{chain.row.cli}: full_workflow = {full}"
        )
        assert full.get("failed_phase") is None


class TestEveryWriteBatchWasANoOp(_ReviewerSandboxPolicy):
    """C4's option (b) verdict, asserted in both directions.

    A canned answer emits no edit tool call, so no redirected CLI here can move
    a byte. The row must record that outcome and only that outcome: `applied`
    would mean the rig invented a change, and an empty batch list would mean the
    phase resolved to the "nothing to apply" skip while still reporting green.

    RENAMED 2026-09-22 from `TestEveryWriteBatchWasACleanFailure`, following the
    word the classifier now uses. Nothing selected on either name — no skip list,
    golden, matrix cell or doc named the class or its methods — and the guard is
    unchanged in strength; the module docstring carries the derivation.
    """

    def test_each_write_phase_emitted_real_batches(self, chain):
        self._refuse_if_skipped(chain)
        for phase, directory in WRITE_PHASE_DIRS.items():
            report = _artifacts(chain) / directory / "write_batch_report.json"
            assert report.is_file(), (
                f"{chain.row.cli}: {phase} left no write_batch_report.json at "
                f"{report} — the phase did not schedule anything"
            )
            tasks = json.loads(report.read_text(encoding="utf-8")).get("tasks") or []
            assert tasks, (
                f"{chain.row.cli}: {phase} scheduled zero batches, so its green "
                f"is vacuous — the fixture chain must give it something to apply"
            )

    def test_every_batch_recorded_no_op_and_touched_nothing(self, chain):
        """`no-op` means the runner CLAIMED no-op and the bytes AGREE.

        Both halves, which is why the word is not interchangeable with the other
        empty-diff verdict. See the module docstring for why it moved on
        2026-09-22 and why the redirect-leak claim survived the move.
        """
        self._refuse_if_skipped(chain)
        for phase, directory in WRITE_PHASE_DIRS.items():
            report = _artifacts(chain) / directory / "write_batch_report.json"
            tasks = json.loads(report.read_text(encoding="utf-8")).get("tasks") or []
            for task in tasks:
                assert task.get("status") == "ok", (
                    f"{chain.row.cli}/{phase}/{task.get('task_id')}: "
                    f"status={task.get('status')} reason={task.get('reason')}"
                )
                assert task.get("write_outcome") == "no-op", (
                    f"{chain.row.cli}/{phase}/{task.get('task_id')}: "
                    f"write_outcome={task.get('write_outcome')!r}, expected "
                    f"'no-op' — the runner claimed 'no-op' AND the pre/post byte "
                    f"snapshot agrees, which is the only verdict a tool-call-free "
                    f"stub can earn here.\n"
                    f"  'applied'/'partial-application' means THE REDIRECT "
                    f"LEAKED: both are reachable only where the snapshot diff is "
                    f"non-empty, so bytes really moved and no canned fixture on "
                    f"this lane can have moved them.\n"
                    f"  'clean-failure' means the opposite kind of break — "
                    f"success CLAIMED over an empty diff — so the fixture's own "
                    f"'no-op' status stopped reaching the classifier (a wrapper "
                    f"or parser dropped it), not that a write escaped.\n"
                    f"  reason={task.get('reason')!r}"
                )
                assert not task.get("files_touched"), (
                    f"{chain.row.cli}/{phase}/{task.get('task_id')}: claims to "
                    f"have touched {task.get('files_touched')}"
                )

    def test_the_workspace_source_tree_is_byte_identical(self, chain):
        """The census, which is independent of anything a batch claimed.

        Scoped to the code the write phases were asked to edit. The orchestrators
        legitimately write elsewhere — the plan file gains its `TASKS_FILE` and
        `IMPL_SUMMARY` markers, and every phase dir is new — so a whole-workspace
        equality assertion would fail on a correct run.
        """
        self._refuse_if_skipped(chain)
        moved = sorted(name for name in chain.census_delta()
                       if name.startswith(("src/", "tests/")))
        assert not moved, (
            f"{chain.row.cli}: the write phases moved bytes in {moved}, which no "
            f"canned answer on this lane can do"
        )


class TestTheRowRanExactlyOneCli(_ReviewerSandboxPolicy):
    def test_the_call_log_names_only_this_row_s_binaries(self, chain):
        self._refuse_if_skipped(chain)
        entry = BY_CLI[chain.row.cli]
        expected = {chain.row.cli} | set(chain.row.helpers)
        seen = {record["cli"] for record in chain.call_log}
        seen.discard("uv")
        assert seen == expected, (
            f"{chain.row.cli}: call log names {sorted(seen)}, expected exactly "
            f"{sorted(expected)} (chain_write_runner="
            f"{entry.get('chain_write_runner')})"
        )

    def test_the_stub_saw_only_this_row_s_models(self, chain):
        """An exact allowed set, not a count.

        `side_model_ids` is what keeps this an assertion rather than a budget:
        grok's session-title call hardcodes `grok-build` instead of the `-m`
        value, and gemini puts the model in the PATH rather than the body, so the
        stub records its own `stub-model` default for it. Both are declared;
        anything else is base's selection leaking into the row.
        """
        self._refuse_if_skipped(chain)
        entry = BY_CLI[chain.row.cli]
        allowed = {
            _spec(chain.row.cli).split(":", 1)[1],
            _spec(_runner_cli(entry)).split(":", 1)[1],
            "stub-model",  # the stub's own default for a body carrying no model
            *entry.get("side_model_ids", []),
        }
        models = {r.get("model") for r in chain.stub_log
                  if r.get("surface") not in (None, "health")}
        models.discard(None)
        assert models <= allowed, (
            f"{chain.row.cli}: stub saw model ids {sorted(models - allowed)} that "
            f"this row did not pin ({sorted(allowed)}), so the selection override "
            f"leaked. If the CLI legitimately sends one, declare it in "
            f"redirect.yaml's side_model_ids."
        )
        assert models, f"{chain.row.cli}: the stub recorded no model id at all"


# --------------------------------------------------------------------------- #
# The golden gate (C8)
# --------------------------------------------------------------------------- #

#: The re-seed command, printed by every failure that a deliberate change would
#: resolve. Named in full rather than described, because the alternative to
#: running it is editing a 5,000-line JSON by hand.
RESEED = ("uv run --project skills/multi-llm -- python "
          "skills/multi-llm/tests/container/seed_container_golden.py --cli {cli}")


@pytest.fixture(scope="module")
def chain_bundle(chain):
    """This row's normalized bundle, captured once from what the container left.

    A capped row is **never** graded against a golden (C1): a killed CLI's last
    artifacts are half-written by definition, and `test_the_row_did_not_hit_its_cap`
    has already failed the row with the classified reason. Skipping here reports
    the cap once instead of four more times, each with a misleading diff.
    """
    if chain.timed_out:
        pytest.skip(f"{chain.row.cli}: capped ({chain.cap_reason}) — a capped row "
                    f"is never compared against a golden")
    if harness.row_outcome(chain)[0] == "sandbox-skip":
        # Graded by `_refuse_if_skipped` in each test BODY, which raises the
        # typed failure the row's xfail is declared against. Capturing a bundle
        # from a row whose write phases were refused would fail here, in SETUP,
        # where a marker added by an autouse fixture does not apply — and a
        # setup error is the bare red the declaration exists to prevent.
        return {}
    bundle = container_bundle.capture_bundle(chain.out_dir)
    # A file this uid cannot read is folded into a `<unreadable>` marker rather
    # than raised, which would otherwise reach the comparison as a few hundred
    # diffs against holes. Named here instead, once, with the cause: the container
    # writes as 10001 and the product's own result files are 0600.
    assert "<unreadable>" not in json.dumps(bundle), (
        f"{chain.row.cli}: the exported bundle holds files this uid cannot read, "
        f"so a golden comparison would grade holes: "
        f"{harness.unreadable_bundle_files(chain.out_dir)}"
    )
    return bundle


def _baseline(cli: str) -> dict:
    """This CLI's committed baseline, or one assertion naming how to make it.

    One accessor for all four assertions below, so a missing file fails the row
    once with the re-seed command rather than raising `FileNotFoundError` three
    more times underneath it.
    """
    path = container_bundle.baseline_path(cli)
    assert path.is_file(), (
        f"{cli}: no container golden at {path}. This row is ungraded until one "
        f"exists — capture it from THIS green run with:\n"
        f"  {RESEED.format(cli=cli)}\n"
        f"then read the diff before committing it."
    )
    return container_bundle.load_baseline(cli)


def _allowlist(cli: str) -> list[dict]:
    """This CLI's entries, through the shared loader plus the per-CLI rule."""
    try:
        entries = load_allowlist(container_bundle.ALLOWLIST_PATH)
    except AllowlistError as exc:  # keep this guard's failures assertion-shaped
        raise AssertionError(str(exc)) from exc
    container_bundle.validate_entries(entries, container_bundle.ALLOWLIST_PATH)
    return container_bundle.entries_for(cli, entries)


class TestTheNormalizedBundleReproducesItsBaseline(_ReviewerSandboxPolicy):
    """C8's provider-lane verdict: zero diff outside the container allowlist.

    Everything else in this module asserts a *property* of the run — a phase
    completed, a batch recorded an outcome, a census stayed still. This asserts
    the run's whole normalized shape, which is the only assertion that fails when
    a CLI upgrade changes what we parse out of its stream without changing
    whether the chain finishes. The two are complements: the properties say what
    must be true, the bundle says nothing else may quietly become true.
    """

    def test_a_baseline_exists_for_this_cli(self, chain, chain_bundle):
        self._refuse_if_skipped(chain)
        _baseline(chain.row.cli)  # the assertion lives in the accessor

    def test_the_baseline_shape_is_current(self, chain, chain_bundle):
        """A baseline from an older bundle shape cannot be diffed meaningfully."""
        self._refuse_if_skipped(chain)
        baseline = _baseline(chain.row.cli)
        assert (baseline.get("bundle_version")
                == container_bundle.CONTAINER_BUNDLE_VERSION), (
            f"{chain.row.cli}: the baseline was captured at bundle_version "
            f"{baseline.get('bundle_version')}, the capture code is at "
            f"{container_bundle.CONTAINER_BUNDLE_VERSION} — re-capture it:\n"
            f"  {RESEED.format(cli=chain.row.cli)}"
        )
        stale = container_bundle.stale_hash_pointers(baseline)
        assert not stale, (
            f"{chain.row.cli}: the baseline was captured while file hashes were "
            f"erased ({len(stale)} `<str>` pointers, first {stale[0]}); the rig "
            f"now pins SOURCE_DATE_EPOCH and compares them by value, so it can "
            f"never reproduce — re-capture it:\n"
            f"  {RESEED.format(cli=chain.row.cli)}"
        )
        manifest = baseline.get("manifest") or {}
        assert manifest.get("cli") == chain.row.cli, (
            f"{chain.row.cli}: {container_bundle.baseline_path(chain.row.cli)} "
            f"records cli={manifest.get('cli')!r} — a baseline filed under the "
            f"wrong CLI grades this row against another binary's output"
        )

    def test_the_bundle_reproduces_the_baseline(self, chain, chain_bundle):
        self._refuse_if_skipped(chain)
        baseline = container_bundle.compare_payload(_baseline(chain.row.cli))
        diffs = container_bundle.diff_bundles(baseline, chain_bundle)
        allowed, unexplained, _matched = partition(
            diffs, _allowlist(chain.row.cli))
        assert not unexplained, (
            f"{chain.row.cli}: the chain's normalized bundle no longer matches "
            f"its baseline. Either this is the regression the gate exists for, or "
            f"it is intentional — in which case declare it in "
            f"{container_bundle.ALLOWLIST_PATH.name} with a `cli`, a `pointer` and "
            f"a `rationale`, or re-capture with\n  "
            f"{RESEED.format(cli=chain.row.cli)}\n"
            + format_unexplained(unexplained, container_bundle.ALLOWLIST_PATH)
        )
        assert len(allowed) == len(diffs)

    def test_no_allowlist_entry_for_this_cli_is_stale(self, chain, chain_bundle):
        """The second rule: an entry that matches nothing has to go.

        A stale exception is an unmaintained blind spot, and the only way to tell
        one from a live one is to require it to still be doing work.
        """
        self._refuse_if_skipped(chain)
        entries = _allowlist(chain.row.cli)
        if not entries:
            return  # the shipping state: nothing excused, nothing to go stale
        baseline = container_bundle.compare_payload(_baseline(chain.row.cli))
        _allowed, _unexplained, matched = partition(
            container_bundle.diff_bundles(baseline, chain_bundle), entries)
        stale = [pointer for pointer, hits in matched.items() if hits == 0]
        assert not stale, (
            f"{chain.row.cli}: {container_bundle.ALLOWLIST_PATH.name} entries that "
            f"no longer match anything (delete them, or the file decays into a "
            f"blanket ignore): {stale}"
        )


# --------------------------------------------------------------------------- #
# The applied-bytes row (C4 option (a)) — the chain's other direction
# --------------------------------------------------------------------------- #

#: The row's write runner: an adapter that declares `agent-write`, whose BINARY
#: (`claude`) the entrypoint shadows with `tests/mocks/mock_llm.py`. The spec has
#: to name a real adapter — `driver.subagent_runner` is validated fail-fast
#: against the built-in adapters and the profile the phase needs — so the fake
#: enters as that adapter's binary rather than as a provider id of its own.
MOCK_RUNNER_CLI = "claude-code"
MOCK_RUNNER_BINARY = "claude"

#: The plan fixture `seed_workspace.py` copies into the sample project **byte for
#: byte**, which is what makes this row's `before` hash a known constant rather
#: than whatever the census happened to record.
PLAN_FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "e2e" / "plans"
                / "auth-feature.md")

#: The verdicts the review-plan fan-out gets on this scenario. Their
#: `group_hash`es *are* the `expected_ids` the apply batches carry (content
#: hashes of this repo's own fixtures, identical on every row), and D3n's
#: spot-check is that the runner attests to them: a batch that moves bytes while
#: attesting to none is a **partial application**, not a success. Read out of the
#: fixture rather than restated here — a fixture edit moves the hashes, and a
#: hard-coded copy would turn that into an exit-3 partial application two phases
#: away from the file that changed.
VALIDATION_FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "e2e"
                      / "responses" / "validation"
                      / "plan_suggestions_valid.json")

#: What the fake appends to the plan on each write dispatch. `patch` (append) and
#: not `create`: the mock applies its edits on **every** invocation, so a
#: fixed-content `create` would leave every batch after the first with an empty
#: diff and record it as `clean-failure` — an empty-diff outcome, the very kind
#: this row exists to be distinguishable from.
#:
#: The word here stayed `clean-failure` through the 2026-09-22 correction, and
#: the difference is the point. `mock_llm.write_batch_result` derives
#: `{"status": "applied"}` from a non-empty edit LIST, not from bytes that moved,
#: so a re-`create` of identical content still claims `applied` — and an
#: `applied` claim over an empty diff is the branch `1633dd4` left at
#: `clean-failure`. The chain rows above land on `no-op` instead because their
#: stub CLAIMS `no-op`. Two different empty-diff verdicts, one per claim.
APPLIED_MARKER = "\n<!-- applied by the edit-capable fake (mock_llm.py) -->\n"


def _applied_group_ids() -> list[str]:
    data = json.loads(VALIDATION_FIXTURE.read_text(encoding="utf-8"))
    return [str(group["group_hash"]) for group in data.get("groups") or []]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _applied_row() -> harness.Row:
    """One row for the whole lane, and two invocations rather than `full`.

    `apply-suggestions` needs a completed review-plan to have anything to apply,
    so the row runs that first — and the two are deliberately **separate
    dispatcher invocations** rather than one `full`, because the environment that
    makes the fake edit has to apply to the second one and not to the first. The
    mock applies `MOCK_LLM_EDITS` on every invocation it makes, so a read task run
    with the variable set would both move bytes outside any batch and answer a
    validation prompt with a write-batch document (rung 6, never retried). The
    entrypoint's per-command `NAME=value` prefix is what scopes it.
    """
    # Read out of the matrix rather than named as a string: a CLI that leaves the
    # provider lane fails here instead of running an unredirected fan-out.
    cli = BY_CLI["codex"]["cli"]
    edits = [{"path": PLAN, "mode": "patch", "content": APPLIED_MARKER}]
    write_env = " ".join([
        "MOCK_LLM_EDITS=" + shlex.quote(json.dumps(edits, separators=(",", ":"))),
        "MOCK_LLM_WRITE_IDS=" + shlex.quote(
            json.dumps(_applied_group_ids(), separators=(",", ":"))),
    ])
    return harness.Row(
        row_id="applied-bytes",
        lane="provider",
        cli=cli,
        spec=_spec(cli),
        # The fan-out stays a real redirected CLI answering from the stub; only
        # the EXECUTOR's runner is the fake. Which CLI answered the fan-out is
        # irrelevant to the branch under test, which is why this is one row for
        # the lane and not one per CLI.
        runner=_spec(MOCK_RUNNER_CLI),
        # No `helpers`: the fake is not the genuine CLI behind C2's wrapper and
        # reads no config root of its own, so this row runs exactly one real
        # binary — its own — and its call log says so.
        mock_runner=MOCK_RUNNER_BINARY,
        command=f"review-plan {PLAN} --quick --yes --exec-subagents",
        commands=[f"{write_env} apply-suggestions {PLAN} --yes --exec-subagents"],
        scenario=SCENARIO,
        row_timeout_s=600,
    )


@pytest.fixture(scope="module")
def applied(posture, utc):
    """Run the applied-bytes row once and share its evidence."""
    return harness.run_row(_applied_row(), posture=posture, utc=utc)


def _applied_tasks(result: harness.RowResult) -> list[dict]:
    report = _artifacts(result) / "apply-suggestions" / "write_batch_report.json"
    assert report.is_file(), (
        f"applied-bytes: apply-suggestions left no write_batch_report.json at "
        f"{report} — the phase did not schedule anything"
    )
    return json.loads(report.read_text(encoding="utf-8")).get("tasks") or []


class TestOneWriteBatchReallyMovedBytes:
    """C4's option (a): the `applied` branch, end to end, in the container.

    Every chain row above is required to record `no-op` and treats `applied` as a
    rig bug — correctly, because a canned answer emits no edit tool call. That
    leaves the branch where bytes actually land untested in-container, and it is
    the branch a wrapper or executor regression breaks *silently*: the scheduler
    would keep reporting the documented no-op and every chain row would stay
    green. So this row inverts every one of those assertions against the same
    scheduler, and it fails just as loudly if it lands on `clean-failure`.

    That last word is deliberately NOT the chain rows' word, and it did not move
    on 2026-09-22 when theirs did. The fake claims `applied` (derived from its
    edit list, never from the bytes), so a dispatch that edited nothing lands on
    the `applied`-claim-plus-empty-diff branch — `clean-failure` — while the
    chain rows, whose stub claims `no-op`, land on `no-op`. The two rows are still
    each other's negative control; the empty-diff side of the pair simply has two
    spellings, one per claim.
    """

    def test_the_row_did_not_hit_its_cap(self, applied):
        assert not applied.timed_out, (
            f"{applied.cap_reason} after {applied.wall_clock_s}s "
            f"(cap {applied.row.row_timeout_s}s)"
        )

    def test_both_invocations_exited_zero(self, applied):
        failed = [record for record in applied.commands if record["exit"] != 0]
        assert not failed, (
            f"applied-bytes: {failed}\n"
            f"--- container stderr (tail) ---\n{applied.stderr[-3000:]}"
        )
        assert len(applied.commands) == 2, (
            f"applied-bytes: {len(applied.commands)} of 2 commands ran — "
            f"apply-suggestions has nothing to apply without review-plan"
        )
        assert applied.exit_code == 0, (
            f"applied-bytes: exited {applied.exit_code}\n"
            f"--- container stderr (tail) ---\n{applied.stderr[-3000:]}"
        )

    def test_the_phase_scheduled_real_batches(self, applied):
        tasks = _applied_tasks(applied)
        assert tasks, (
            "applied-bytes: apply-suggestions scheduled zero batches, so nothing "
            "was dispatched to the fake at all — a green here would prove only "
            "that the phase resolved to the 'nothing to apply' skip"
        )

    def test_every_batch_recorded_applied_and_named_the_plan(self, applied):
        """The scheduler's verdict, not the runner's claim.

        `write_outcome` comes from the pre/post byte comparison (D3g), so this is
        the assertion that distinguishes "the fake said `applied`" from "the
        scheduler observed the bytes move". `files_touched` is the observed set
        for the same reason, which is why it is compared against the one file the
        batch declared rather than against what the fake reported.
        """
        for task in _applied_tasks(applied):
            assert task.get("status") == "ok", (
                f"applied-bytes/{task.get('task_id')}: "
                f"status={task.get('status')} reason={task.get('reason')}"
            )
            assert task.get("write_outcome") == "applied", (
                f"applied-bytes/{task.get('task_id')}: "
                f"write_outcome={task.get('write_outcome')!r}, expected "
                f"'applied'. `clean-failure` here means the fake never edited "
                f"anything — the runner it shadows, MOCK_LLM_EDITS reaching the "
                f"child, or the snapshot comparison itself"
            )
            touched = [Path(str(p)).as_posix() for p in
                       (task.get("files_touched") or [])]
            assert touched and all(p.endswith(PLAN) for p in touched), (
                f"applied-bytes/{task.get('task_id')}: files_touched={touched}, "
                f"expected exactly the plan the batch declared ({PLAN})"
            )

    def test_the_plan_file_bytes_really_moved(self, applied):
        """The census, which is independent of anything the batch claimed.

        The `before` hash is asserted exactly rather than merely compared: the
        sample project's plan is `tests/fixtures/e2e/plans/auth-feature.md` copied
        byte for byte, so a `before` that is anything else means this row edited a
        file the fixture chain never seeded and the `after` proves nothing.
        """
        delta = applied.census_delta()
        moved = delta.get(PLAN)
        assert moved is not None, (
            f"applied-bytes: {PLAN} is byte-identical after a run whose every "
            f"batch claims to have edited it — the census delta was "
            f"{sorted(delta)}"
        )
        before, after = moved
        assert before == _sha256(PLAN_FIXTURE.read_bytes()), (
            f"applied-bytes: the seeded plan hashed {before}, and the fixture "
            f"{PLAN_FIXTURE} hashes {_sha256(PLAN_FIXTURE.read_bytes())} — the "
            f"row edited something the chain never seeded"
        )
        assert after and after != before, (
            f"applied-bytes: {PLAN} went {before} -> {after}"
        )

    def test_no_source_file_was_touched(self, applied):
        """The fake edits what the batch DECLARED, and nothing else.

        The same scoped census the chain rows use, asserted for the opposite
        reason: there the write phases must move nothing, here they must move
        exactly one file and it is not one of these. A byte moving under `src/`
        would mean the write-path policy's post-run sweep let an undeclared write
        through.
        """
        moved = sorted(name for name in applied.census_delta()
                       if name.startswith(("src/", "tests/")))
        assert not moved, (
            f"applied-bytes: the write phase moved bytes in {moved}, outside the "
            f"single file its batches declared ({PLAN})"
        )


# --------------------------------------------------------------------------- #
# The negative control
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entry", [BY_CLI["codex"]], ids=["codex"])
def test_the_same_chain_without_exec_subagents_finishes_no_phase(
    entry, posture, utc
):
    """`--yes` does not imply `--exec-subagents`, and the difference is total.

    **Measured, and it stops one phase EARLIER than the plan predicted.** The
    plan expects `full_workflow_orchestrator._unexecuted_batches` to end the run
    at `[WRITE_PHASE_NOT_EXECUTED]` on the first *apply* phase. That is what
    happens on a host with a native subagent channel. This lane has none — the
    container's `full` is the whole invoker — so the run stops at **review-plan**,
    which emits `[VALIDATION_BATCHES_PENDING]` and exits 0 without recording
    itself completed. `_verify` catches exactly that ("exited 0 but state.json
    records no completion") and the run ends at exit 3 with **zero** phases
    recorded. The write-phase guard is never reached because nothing gets that
    far.

    The property under test is unchanged and is if anything stronger: without the
    flag, this chain finishes nothing. One row is enough — the mechanism is one
    branch of one function, identical on every CLI.
    """
    result = harness.run_row(_chain_row(entry, exec_subagents=False),
                             posture=posture, utc=utc)
    assert not result.timed_out, f"{result.cap_reason} after {result.wall_clock_s}s"
    assert result.exit_code == 3, (
        f"expected exit 3, got {result.exit_code}\n"
        f"--- container stderr (tail) ---\n{result.stderr[-3000:]}"
    )
    state = _state(result)
    completed = set(state.get("phases_completed") or {})
    assert not completed, (
        f"without --exec-subagents the chain recorded {sorted(completed)} as "
        f"completed; nothing should finish, because nothing executes the "
        f"subagent batches each phase emits"
    )
    full = state.get("full_workflow") or {}
    assert full.get("failed_phase") == "review-plan", (
        f"expected the run to stop at review-plan's pending validation batches, "
        f"got {full}"
    )
    haystack = result.stdout + result.stderr
    assert "VALIDATION_BATCHES_PENDING" in haystack, (
        "exit 3 for some other reason than an unexecuted subagent batch"
    )
    assert "records no completion for 'review-plan'" in haystack, (
        "the phase-vs-artifact check (D2i) is what must have refused, not a crash"
    )
