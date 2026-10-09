"""Provider lane: `--improve-plan` and `--improve-code` against real CLIs.

The two composites (`improve-plan`: review-plan then apply-suggestions;
`improve-code`: review-code then apply-code-fixes) are pinned in plain CI by
`test_improve_orchestrator.py` and `test_improve_steps.py`, and every one of
those tests stubs `FullWorkflow.run_mode`: no real phase ever runs under a
composite there. The chain rows in `test_container_full_chain.py` run all eight
phases for real, but only under `full`. So nothing graded what a user of these
two modes actually gets: one unattended invocation whose review fans out to a
real CLI, whose validation and write batches dispatch through a real runner, and
which stops after its two steps.

These rows close that gap at zero cost, on the provider lane's terms: the
loopback stub, `--network none`, no credential. Each row runs, in one
container:

    improve-plan plans/auth-feature.md --quick --yes --exec-subagents
    improve-code plans/auth-feature.md --quick --yes --exec-subagents
    status plans/auth-feature.md --view workflow

and the assertions read the evidence the run left behind, never its exit code
alone:

* **exactly two phases per composite.** `state.json` records the four phases
  the two composites own, in order, and none of the four they do not. The stub
  log says the same thing from the other side: the turns of the phases the
  composites own were served, and no turn of `generate-tasks`, `review-tasks`,
  `apply-task-suggestions`, `implement` or `ask` ever reached the stub;
* **the write path really ran.** Both apply phases scheduled batches, through
  the row's own CLI and its containment flag, and every batch is the lane's
  documented `no-op` (the stub emits no tool call; see the chain module's
  docstring for why `applied` here would be a redirect leak);
* **each composite answers for itself on its result line.** The dispatcher's
  receipt for each invocation records the composite's own mode word and a
  `user_next_step` that answers as the phase the composite ends on.

The `[ALREADY_COMPLETE]` re-run (exit 2) is deliberately not a command here:
the entrypoint stops a row at its first non-zero exit and the census would file
the row as `fail` for obeying its own contract. It is graded on the host
instead, in `test_e2e_improve_modes.py`, where nothing reads exit 2 as a broken
container.

Why these two CLIs and not the whole reviewer axis: what is new here is the
composite's orchestration, which does not change per reviewer CLI, while each
CLI's parse is already graded per CLI by the chain rows. `opencode` and
`claude-code` are the two edit-capable CLIs whose write containment is
application-level, so both their write paths run inside the image; codex's
`--sandbox workspace-write` cannot start here (the chain's declared
`sandbox_skip_when`), and a codex row would only re-grade that refusal.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
from . import test_container_full_chain as chain_lane

pytestmark = [pytest.mark.container, requires_docker()]

PLAN = chain_lane.PLAN

#: The same scenario the chain rows answer from: every turn the composites take
#: is a turn `full` takes too, and `on_unmatched: fail` makes a turn it does not
#: know a failed row rather than a default answer.
SCENARIO = chain_lane.SCENARIO

COMMANDS = [
    f"improve-plan {PLAN} --quick --yes --exec-subagents",
    f"improve-code {PLAN} --quick --yes --exec-subagents",
    f"status {PLAN} --view workflow",
]

#: Each composite's two phases, as `state.json` spells them, in run order.
OWNED_PHASES = ("review-plan", "apply-suggestions", "code-review",
                "apply-code-fixes")

#: The workflow phases neither composite runs.
NOT_OWNED_PHASES = tuple(p for p in chain_lane.CHAIN_PHASES
                         if p not in OWNED_PHASES)

#: The two apply phases and the directory each writes its report into.
APPLY_DIRS = {"apply-suggestions": "apply-suggestions",
              "apply-code-fixes": "apply-fixes"}

#: The scenario fixtures (paths as the stub logs them) each side of the claim
#: rests on. Served: one per turn kind the composites own. Never served: one per
#: turn kind of a phase they do not own.
SERVED = {
    "../responses/review_plan/raw_suggestions.json": "review-plan fan-out",
    "../responses/validation/plan_suggestions_valid.json": "review-plan validation",
    "../responses/write_batches/apply_suggestions.json": "apply-suggestions batch",
    "../responses/code_review/container_issues.json": "review-code fan-out",
    "../responses/validation/code_review_valid.json": "code-review validation",
    "../responses/write_batches/apply_code_fixes.json": "apply-code-fixes batch",
}
NEVER_SERVED = {
    "../responses/generate_tasks/tasks_v1.json": "generate-tasks",
    "../responses/review_tasks/coverage_suggestions.json": "review-tasks fan-out",
    "../responses/validation/task_suggestions_valid.json": "review-tasks validation",
    "../responses/write_batches/apply_task_suggestions.json": "apply-task-suggestions",
    "../responses/write_batches/implement.json": "implement",
    "../responses/ask/plan_question.md": "ask",
}

IMPROVE_CLIS = ("opencode", "claude-code")


def _entries() -> list[dict]:
    data = yaml.safe_load(chain_lane.REDIRECT.read_text(encoding="utf-8"))
    by_cli = {r["cli"]: r for r in data["clis"] if r.get("lane") == "provider"}
    return [by_cli[cli] for cli in IMPROVE_CLIS]


ENTRIES = _entries()


def _row(entry: dict) -> harness.Row:
    cli = entry["cli"]
    spec = f"{cli}:{entry.get('model_arg', 'stub-model')}"
    return harness.Row(
        row_id=f"improve-{cli}",
        lane="provider",
        cli=cli,
        spec=[spec],
        # Both driver keys from this one value (seed_workspace defaults the
        # cheap runner to it), as on the chain rows: the reviewer is its own
        # write runner, so the row runs exactly one binary.
        runner=spec,
        helpers=[],
        command=COMMANDS[0],
        commands=COMMANDS[1:],
        scenario=SCENARIO,
        # Four phases where the chain runs eight, so the chain's own cap bounds
        # it from above.
        row_timeout_s=int(entry.get(
            "chain_row_timeout_s",
            max(600, int(entry.get("row_timeout_s", 300)) * 2))),
    )


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module", params=ENTRIES, ids=list(IMPROVE_CLIS))
def improve(request, posture, utc):
    """Run one CLI's improve row once and share its evidence."""
    return harness.run_row(_row(request.param), posture=posture, utc=utc)


def _state(result: harness.RowResult) -> dict:
    return chain_lane._state(result)


def _receipts_by_mode(result: harness.RowResult) -> list[tuple[str, dict]]:
    """``(mode word, result line)`` per dispatcher invocation, in run order."""
    out = []
    for receipt in sorted(result.receipts, key=lambda r: r.get("seq", 0)):
        argv = [str(a) for a in receipt.get("argv") or []]
        mode = next((argv[i + 1] for i, token in enumerate(argv[:-1])
                     if token.endswith("multi_llm.py")), "")
        out.append((mode, receipt.get("result") or {}))
    return out


def _served_fixtures(result: harness.RowResult) -> set[str]:
    return {str(r["fixture"]) for r in result.stub_log if r.get("fixture")}


class TestTheRowRan:
    def test_the_row_did_not_hit_its_cap(self, improve):
        assert not improve.timed_out, (
            f"{improve.row.cli}: {improve.cap_reason} after "
            f"{improve.wall_clock_s}s (cap {improve.row.row_timeout_s}s)"
        )

    def test_every_command_exited_zero(self, improve):
        failed = [record for record in improve.commands if record["exit"] != 0]
        assert not failed, (
            f"{improve.row.cli}: {failed}\n"
            f"--- container stderr (tail) ---\n{improve.stderr[-3000:]}"
        )
        assert [r["command"] for r in improve.commands] == COMMANDS, (
            f"{improve.row.cli}: ran {[r['command'] for r in improve.commands]}"
        )

    def test_no_prompt_fell_through_and_no_route_went_unserved(self, improve):
        unmatched = [r["prompt_sha"] for r in improve.stub_log if r.get("unmatched")]
        unhandled = [(r["method"], r["path"]) for r in improve.stub_log
                     if r.get("unhandled")]
        assert not unmatched, (
            f"{improve.row.cli}: {len(unmatched)} prompt(s) matched no pattern "
            f"in container_full_chain.yaml: {unmatched}"
        )
        assert not unhandled, f"{improve.row.cli}: unserved routes {unhandled}"

    def test_the_call_log_names_only_this_row_s_cli(self, improve):
        seen = {record.get("cli") for record in improve.call_log}
        seen.discard("uv")
        assert seen == {improve.row.cli}, (
            f"{improve.row.cli}: call log names {sorted(s for s in seen if s)}"
        )


class TestEachCompositeRanItsTwoPhases:
    def test_state_json_records_the_four_owned_phases_in_order(self, improve):
        completed = _state(improve).get("phases_completed") or {}
        missing = [p for p in OWNED_PHASES if p not in completed]
        assert not missing, (
            f"{improve.row.cli}: {missing} not recorded completed "
            f"(recorded: {sorted(completed)})"
        )
        stamps = [datetime.fromisoformat(str(completed[p])) for p in OWNED_PHASES]
        assert stamps == sorted(stamps), (
            f"{improve.row.cli}: completion order "
            f"{dict(zip(OWNED_PHASES, map(str, stamps)))} is not "
            f"review, apply, review, apply"
        )

    def test_no_phase_outside_the_two_composites_ran(self, improve):
        state = _state(improve)
        touched = (set(state.get("phases_completed") or {})
                   | set(state.get("phases_skipped") or {})
                   | set(state.get("phases_started") or {}))
        strays = sorted(touched & set(NOT_OWNED_PHASES))
        assert not strays, (
            f"{improve.row.cli}: a composite ran or recorded {strays}, which "
            f"belong to neither improve-plan nor improve-code"
        )

    def test_neither_owned_phase_was_skipped(self, improve):
        skipped = set(_state(improve).get("phases_skipped") or {})
        assert not (skipped & set(OWNED_PHASES)), (
            f"{improve.row.cli}: skipped {sorted(skipped & set(OWNED_PHASES))} "
            f"— a skipped apply phase is a vacuous green"
        )

    @pytest.mark.parametrize("key", ["improve_plan", "improve_code"])
    def test_each_composite_records_itself_complete(self, improve, key):
        record = _state(improve).get(key) or {}
        assert record.get("current_phase") == "complete", (
            f"{improve.row.cli}: state.json {key} = {record}"
        )
        assert record.get("failed_phase") is None, record

    def test_the_stub_served_the_owned_turns_and_no_other_phase(self, improve):
        served = _served_fixtures(improve)
        missing = {path: what for path, what in SERVED.items()
                   if path not in served}
        assert not missing, (
            f"{improve.row.cli}: no turn reached the stub for "
            f"{sorted(missing.values())} (served: {sorted(served)})"
        )
        extra = {path: what for path, what in NEVER_SERVED.items()
                 if path in served}
        assert not extra, (
            f"{improve.row.cli}: the stub answered a turn of "
            f"{sorted(extra.values())}, which neither composite owns"
        )


class TestTheWritePathRan:
    @pytest.mark.parametrize("phase", sorted(APPLY_DIRS))
    def test_each_apply_phase_scheduled_real_no_op_batches(self, improve, phase):
        report = (chain_lane._artifacts(improve) / APPLY_DIRS[phase]
                  / "write_batch_report.json")
        assert report.is_file(), (
            f"{improve.row.cli}: {phase} left no write_batch_report.json — it "
            f"scheduled nothing"
        )
        tasks = json.loads(report.read_text(encoding="utf-8")).get("tasks") or []
        assert tasks, f"{improve.row.cli}: {phase} scheduled zero batches"
        for task in tasks:
            where = f"{improve.row.cli}/{phase}/{task.get('task_id')}"
            assert task.get("status") == "ok", (
                f"{where}: status={task.get('status')} reason={task.get('reason')}"
            )
            assert task.get("write_outcome") == "no-op", (
                f"{where}: write_outcome={task.get('write_outcome')!r}; the "
                f"stub emits no tool call, so only `no-op` is earnable here "
                f"(see test_container_full_chain's docstring)"
            )
            assert not task.get("files_touched"), task.get("files_touched")

    def test_no_report_for_a_write_phase_the_composites_do_not_own(self, improve):
        artifacts = chain_lane._artifacts(improve)
        strays = [d for d in ("apply-task-suggestions", "implement")
                  if (artifacts / d / "write_batch_report.json").exists()]
        assert not strays, f"{improve.row.cli}: write reports under {strays}"

    @pytest.mark.parametrize("phase", sorted(APPLY_DIRS))
    def test_the_write_dispatch_carried_the_containment_flag(self, improve, phase):
        flag, value = chain_lane.WRITER_CONTAINMENT_FLAGS[improve.row.cli]
        audit = chain_lane._artifacts(improve) / APPLY_DIRS[phase] / "subagent_audit.jsonl"
        lines = harness._read_jsonl(audit)
        assert lines, f"{improve.row.cli}: no subagent_audit.jsonl at {audit}"
        assert any(chain_lane._argv_carries(line.get("argv"), flag, value)
                   for line in lines), (
            f"{improve.row.cli}/{phase}: no audited write argv carried "
            f"{flag} {value}: {[line.get('argv') for line in lines[:3]]}"
        )
        crossed = [c for c in improve.call_log if c.get("cli") == improve.row.cli
                   and chain_lane._argv_carries(c.get("argv"), flag, value)]
        assert crossed, (
            f"{improve.row.cli}: the audit claims {flag} {value}, but no "
            f"process crossing /opt/mllm/bin carried it"
        )

    def test_the_workspace_source_tree_is_byte_identical(self, improve):
        moved = sorted(name for name in improve.census_delta()
                       if name.startswith(("src/", "tests/")))
        assert not moved, (
            f"{improve.row.cli}: the apply phases moved bytes in {moved}, which "
            f"no canned answer on this lane can do"
        )


class TestEachResultLineAnswersForItsComposite:
    def test_one_receipt_per_command_in_order(self, improve):
        modes = [mode for mode, _result in _receipts_by_mode(improve)]
        assert modes == ["improve-plan", "improve-code", "status"], (
            f"{improve.row.cli}: dispatcher receipts name {modes}"
        )

    @pytest.mark.parametrize("index,mode", [(0, "improve-plan"), (1, "improve-code")])
    def test_the_result_line_names_the_composite_and_finished(self, improve,
                                                              index, mode):
        _word, result = _receipts_by_mode(improve)[index]
        assert result.get("mode") == mode, result
        assert result.get("exit") == 0, result
        assert result.get("state") == "done", result
        assert result.get("next") is None, (
            f"{improve.row.cli}: {mode} finished in one invocation, so no "
            f"follow-up call is owed; next={result.get('next')!r}"
        )

    def test_improve_plan_answers_as_apply_suggestions(self, improve):
        """The next step after the phase it ends on: generate-tasks."""
        _word, result = _receipts_by_mode(improve)[0]
        step = result.get("user_next_step")
        assert isinstance(step, dict), (
            f"{improve.row.cli}: improve-plan's result line carries no "
            f"user_next_step: {result}"
        )
        assert step.get("phase") == "generate-tasks", step
        assert step.get("complete") is False, step
        assert step.get("command"), step

    def test_improve_code_answers_as_apply_code_fixes(self, improve):
        """No required phase follows apply-code-fixes, so the answer is complete."""
        _word, result = _receipts_by_mode(improve)[1]
        step = result.get("user_next_step")
        assert isinstance(step, dict), (
            f"{improve.row.cli}: improve-code's result line carries no "
            f"user_next_step: {result}"
        )
        assert step.get("phase") is None and step.get("complete") is True, step
