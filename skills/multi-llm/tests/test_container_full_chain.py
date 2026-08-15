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
the other eight rows name their write runner in `redirect.yaml`'s
`chain_write_runner` and carry it as the row's one allowed helper.
``tests/test_redirect_matrix_claims.py`` cross-checks that key against the
adapters themselves, in **plain CI** — a drift guard living in this module would
only run where Docker runs, which is not where the drift gets introduced. Its
name deliberately does *not* start with `test_container_`: that prefix is what
the image's own glob and the marker-gating check select on, and a plain-CI guard
must not be swept into either.

**Every write batch must land on `clean-failure`, and that is asserted in both
directions.** The stub emits no tool call, so no redirected CLI on this lane can
produce a byte change: the four write phases complete on the documented no-op
(`status: ok`, `write_outcome: clean-failure`, zero files touched, loop
continues, finalization in full, phase completed at exit 0). A batch that reads
`applied` here is a bug in the rig, not a pass — and so is a write phase that
resolves to the "nothing to apply" skip, which is why the fixture chain is chosen
to give each apply phase real batches to emit rather than a vacuous green.

**Every structural assertion above stays green through output-shape drift, so the
row is also graded against a normalized golden bundle (C8).** "Phases completed",
"batches recorded clean-failure" and "the census moved nothing" are all satisfied
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
`clean-failure` from every chain batch and calling `applied` a rig bug leaves the
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
        spec=_spec(cli),
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


@pytest.fixture(scope="module", params=VERIFIED, ids=IDS)
def chain(request, posture, utc):
    """Run one CLI's whole chain once and share its evidence across the asserts."""
    return harness.run_row(_chain_row(request.param), posture=posture, utc=utc)


def _artifacts(result: harness.RowResult) -> Path:
    return result.out_dir / "artifacts" / "auth-feature"


def _state(result: harness.RowResult) -> dict:
    path = _artifacts(result) / "state.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheChainRan:
    def test_the_row_did_not_hit_its_cap(self, chain):
        assert not chain.timed_out, (
            f"{chain.row.cli}: {chain.cap_reason} after {chain.wall_clock_s}s "
            f"(cap {chain.row.row_timeout_s}s)"
        )

    def test_every_dispatcher_command_exited_zero(self, chain):
        """Which invocation failed, not one number for three of them."""
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
        unmatched = [r["prompt_sha"] for r in chain.stub_log if r.get("unmatched")]
        assert not unmatched, (
            f"{chain.row.cli}: {len(unmatched)} prompt(s) matched no pattern: "
            f"{unmatched} — add the turn to container_full_chain.yaml"
        )

    def test_no_route_went_unserved(self, chain):
        unhandled = [(r["method"], r["path"]) for r in chain.stub_log
                     if r.get("unhandled")]
        assert not unhandled, (
            f"{chain.row.cli}: the stub does not serve {unhandled} — add a "
            f"handler and the route to this CLI's redirect.yaml http_log"
        )

    def test_the_cli_actually_reached_the_stub(self, chain):
        calls = [r for r in chain.stub_log if r.get("surface") != "health"]
        assert calls, f"{chain.row.cli}: zero requests reached the stub"


class TestEveryPhaseFinished:
    def test_state_json_records_every_phase_completed(self, chain):
        """The artifact, not the exit code.

        A phase directory is created *before* the work inside it happens, and
        two of the four write phases mark themselves completed when they *emit*
        their batch document. `state.json`'s `phases_completed` is the record
        that survives both.
        """
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
        state = _state(chain)
        skipped = set(state.get("phases_skipped") or {})
        assert not (skipped & set(CHAIN_PHASES)), (
            f"{chain.row.cli}: phases skipped: {sorted(skipped & set(CHAIN_PHASES))}"
        )

    def test_the_workflow_reports_itself_complete(self, chain):
        state = _state(chain)
        full = state.get("full_workflow") or {}
        assert full.get("current_phase") == "complete", (
            f"{chain.row.cli}: full_workflow = {full}"
        )
        assert full.get("failed_phase") is None


class TestEveryWriteBatchWasACleanFailure:
    """C4's option (b) verdict, asserted in both directions.

    A canned answer emits no edit tool call, so no redirected CLI here can move
    a byte. The row must record that outcome and only that outcome: `applied`
    would mean the rig invented a change, and an empty batch list would mean the
    phase resolved to the "nothing to apply" skip while still reporting green.
    """

    def test_each_write_phase_emitted_real_batches(self, chain):
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

    def test_every_batch_recorded_clean_failure_and_touched_nothing(self, chain):
        for phase, directory in WRITE_PHASE_DIRS.items():
            report = _artifacts(chain) / directory / "write_batch_report.json"
            tasks = json.loads(report.read_text(encoding="utf-8")).get("tasks") or []
            for task in tasks:
                assert task.get("status") == "ok", (
                    f"{chain.row.cli}/{phase}/{task.get('task_id')}: "
                    f"status={task.get('status')} reason={task.get('reason')}"
                )
                assert task.get("write_outcome") == "clean-failure", (
                    f"{chain.row.cli}/{phase}/{task.get('task_id')}: "
                    f"write_outcome={task.get('write_outcome')!r}, expected "
                    f"'clean-failure'. `applied` on this lane is a bug in the "
                    f"rig, not a pass: the stub emits no edit tool call."
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
        moved = sorted(name for name in chain.census_delta()
                       if name.startswith(("src/", "tests/")))
        assert not moved, (
            f"{chain.row.cli}: the write phases moved bytes in {moved}, which no "
            f"canned answer on this lane can do"
        )


class TestTheRowRanExactlyOneCli:
    def test_the_call_log_names_only_this_row_s_binaries(self, chain):
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


class TestTheNormalizedBundleReproducesItsBaseline:
    """C8's provider-lane verdict: zero diff outside the container allowlist.

    Everything else in this module asserts a *property* of the run — a phase
    completed, a batch recorded an outcome, a census stayed still. This asserts
    the run's whole normalized shape, which is the only assertion that fails when
    a CLI upgrade changes what we parse out of its stream without changing
    whether the chain finishes. The two are complements: the properties say what
    must be true, the bundle says nothing else may quietly become true.
    """

    def test_a_baseline_exists_for_this_cli(self, chain, chain_bundle):
        _baseline(chain.row.cli)  # the assertion lives in the accessor

    def test_the_baseline_shape_is_current(self, chain, chain_bundle):
        """A baseline from an older bundle shape cannot be diffed meaningfully."""
        baseline = _baseline(chain.row.cli)
        assert (baseline.get("bundle_version")
                == container_bundle.CONTAINER_BUNDLE_VERSION), (
            f"{chain.row.cli}: the baseline was captured at bundle_version "
            f"{baseline.get('bundle_version')}, the capture code is at "
            f"{container_bundle.CONTAINER_BUNDLE_VERSION} — re-capture it:\n"
            f"  {RESEED.format(cli=chain.row.cli)}"
        )
        manifest = baseline.get("manifest") or {}
        assert manifest.get("cli") == chain.row.cli, (
            f"{chain.row.cli}: {container_bundle.baseline_path(chain.row.cli)} "
            f"records cli={manifest.get('cli')!r} — a baseline filed under the "
            f"wrong CLI grades this row against another binary's output"
        )

    def test_the_bundle_reproduces_the_baseline(self, chain, chain_bundle):
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
#: diff and record it as `clean-failure` — the very outcome this row exists to be
#: distinguishable from.
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

    Every chain row above is required to record `clean-failure` and treats
    `applied` as a rig bug — correctly, because a canned answer emits no edit tool
    call. That leaves the branch where bytes actually land untested in-container,
    and it is the branch a wrapper or executor regression breaks *silently*: the
    scheduler would keep reporting the documented no-op and every chain row would
    stay green. So this row inverts every one of those assertions against the same
    scheduler, and it fails just as loudly if it lands on `clean-failure`.
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
