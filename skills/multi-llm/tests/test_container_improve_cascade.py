"""Provider lane: `--improve-plan --cascade` against a real CLI, two stages.

`--cascade` runs one review-then-apply round per model, one model after the
other; from stage 2 on, the reviewer is told what the earlier stages rejected
and changed (the forward ledger). `tests/test_e2e_improve_cascade.py` grades
that every PR with the fake binaries; this row runs it in the image, at zero
cost, on the provider lane's terms: the loopback stub, `--network none`, no
credential. One container runs:

    improve-plan plans/auth-feature.md --cascade \
        --models claude-code:haiku claude-code:sonnet --yes --exec-subagents
    status plans/auth-feature.md --view workflow

and the assertions read what the run left behind:

* **two stages, archived.** The composite's run dir names one cascade; its
  `record.json` is finished with both stages done in the typed order; each
  stage's `artifacts/` was published once, every file its `.complete.json`
  lists still hashes to the recorded value, and the archived review round
  carries the stage's `cascade` block;
* **the per-round table is the archive's.** `summary.md` has one row per
  stage, and its findings and verdict columns are the archived `grouped.json`
  and `validation.json`, recounted here;
* **the later stage's reviewer saw the ledger.** The row's scenario
  (`fixtures/e2e/container_scenarios/container_improve_cascade.yaml`) answers a
  reviewer prompt that carries `## Earlier review rounds` through a rule of its
  own, whose fixture string the stub logs as written. So the stub log itself
  says that stage 2's reviewer turn held the ledger, that stage 1's did not,
  and that the two turns came from different models;
* **`state.json`** records the two phases the composite owns and none other;
* **the write path ran** through the row's own CLI: stage 1's apply scheduled
  batches, each the lane's documented `no-op` (the stub emits no tool call);
* **one result line** for the composite, finished, owing nothing.

Why one CLI: what is new is the cascade's orchestration, which does not change
per reviewer CLI, and it needs two models of the same CLI the stub serves.
`claude-code` resolves `haiku` and `sonnet` itself and its write containment is
application-level, so its write path runs inside the image. The finished
cascade's re-run (exit 2) is graded on the host only, for the reason
`tests/test_container_improve_modes.py` gives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
from . import test_container_full_chain as chain_lane
from . import test_container_improve_modes as improve_lane

pytestmark = [pytest.mark.container, requires_docker()]

PLAN = chain_lane.PLAN
CLI = "claude-code"
#: Weakest first; the first is also the runner every subagent dispatches to.
STAGES = (f"{CLI}:haiku", f"{CLI}:sonnet")

#: The cascade scenario, as the image holds it (`tests/fixtures/e2e` ships whole).
SCENARIO = ("/opt/mllm/skill/tests/fixtures/e2e/container_scenarios/"
            "container_improve_cascade.yaml")

COMMANDS = [
    f"improve-plan {PLAN} --cascade --models {' '.join(STAGES)} --yes --exec-subagents",
    # `--view workflow`, as the improve rows read it back: the default view exits
    # 2 when no run is live, and the entrypoint files a non-zero command as a
    # failed row.
    f"status {PLAN} --view workflow",
]

#: The two spellings the stub logs for a plan-review turn: the chain rule's (a
#: prompt without the ledger) and the cascade scenario's front rule (a prompt
#: carrying `## Earlier review rounds`).
FIRST_STAGE_REVIEW = "../responses/review_plan/raw_suggestions.json"
LEDGER_REVIEW = "../container_scenarios/../responses/review_plan/raw_suggestions.json"
SERVED = {
    FIRST_STAGE_REVIEW: "stage 1's review-plan fan-out",
    LEDGER_REVIEW: "stage 2's review-plan fan-out, the ledger in its prompt",
    "../responses/validation/plan_suggestions_valid.json": "review-plan validation",
    "../responses/write_batches/apply_suggestions.json": "apply-suggestions batch",
}
NEVER_SERVED = dict(improve_lane.NEVER_SERVED) | {
    "../responses/code_review/container_issues.json": "review-code fan-out",
    "../responses/write_batches/apply_code_fixes.json": "apply-code-fixes batch",
}

OWNED_PHASES = ("review-plan", "apply-suggestions")
NOT_OWNED_PHASES = tuple(p for p in chain_lane.CHAIN_PHASES if p not in OWNED_PHASES)
#: `review_plan/raw_suggestions.json` raises four findings: four groups on a
#: one-model stage.
FINDINGS = 4


def _entry() -> dict:
    data = yaml.safe_load(chain_lane.REDIRECT.read_text(encoding="utf-8"))
    return next(r for r in data["clis"] if r["cli"] == CLI and r.get("lane") == "provider")


def _row() -> harness.Row:
    entry = _entry()
    return harness.Row(
        row_id=f"improve-cascade-{CLI}",
        lane="provider",
        cli=CLI,
        spec=list(STAGES),
        runner=STAGES[0],
        helpers=[],
        command=COMMANDS[0],
        commands=COMMANDS[1:],
        scenario=SCENARIO,
        # Two review-then-apply rounds where the improve rows run two
        # composites; the chain's cap, with headroom for the second model.
        row_timeout_s=max(900, int(entry.get("chain_row_timeout_s", 600))),
    )


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module")
def cascade(posture, utc):
    """Run the cascade row once and share its evidence."""
    return harness.run_row(_row(), posture=posture, utc=utc)


def _plan_dir(result: harness.RowResult) -> Path:
    return chain_lane._artifacts(result)


def _cascade_dir(result: harness.RowResult) -> Path:
    run_dir = _plan_dir(result) / "improve-plan"
    pointer = run_dir / "cascade-current.json"
    assert pointer.is_file(), f"no cascade pointer at {pointer}"
    return run_dir / json.loads(pointer.read_text(encoding="utf-8"))["dir"]


def _record(result: harness.RowResult) -> dict:
    return json.loads((_cascade_dir(result) / "record.json").read_text(encoding="utf-8"))


def _table_rows(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells and cells[0].isdigit():
            rows[int(cells[0])] = cells
    return rows


def _review_turns(result: harness.RowResult, fixture: str) -> list[dict]:
    return [r for r in result.stub_log if r.get("fixture") == fixture]


class TestTheRowRan:
    def test_the_row_did_not_hit_its_cap(self, cascade):
        assert not cascade.timed_out, (
            f"{cascade.cap_reason} after {cascade.wall_clock_s}s "
            f"(cap {cascade.row.row_timeout_s}s)")

    def test_every_command_exited_zero(self, cascade):
        failed = [record for record in cascade.commands if record["exit"] != 0]
        assert not failed, (
            f"{failed}\n--- container stderr (tail) ---\n{cascade.stderr[-3000:]}")
        assert [r["command"] for r in cascade.commands] == COMMANDS, cascade.commands

    def test_no_prompt_fell_through_and_no_route_went_unserved(self, cascade):
        unmatched = [r["prompt_sha"] for r in cascade.stub_log if r.get("unmatched")]
        unhandled = [(r["method"], r["path"]) for r in cascade.stub_log if r.get("unhandled")]
        assert not unmatched, (
            f"{len(unmatched)} prompt(s) matched no pattern in "
            f"container_improve_cascade.yaml: {unmatched}")
        assert not unhandled, f"unserved routes {unhandled}"

    def test_the_call_log_names_only_this_row_s_cli(self, cascade):
        seen = {record.get("cli") for record in cascade.call_log}
        seen.discard("uv")
        assert seen == {CLI}, f"call log names {sorted(s for s in seen if s)}"


class TestTwoStagesArchived:
    def test_the_record_is_finished_with_both_stages_done_in_order(self, cascade):
        run_dir = _plan_dir(cascade) / "improve-plan"
        assert len(list((run_dir / "cascades").iterdir())) == 1, "more than one cascade"
        record = _record(cascade)
        assert record["state"] == "finished" and record["mode"] == "improve-plan", record
        assert [s["model"] for s in record["stages"]] == list(STAGES), record["stages"]
        assert [s["state"] for s in record["stages"]] == ["done", "done"], record["stages"]
        assert (_cascade_dir(cascade) / "start.md").is_file()

    @pytest.mark.parametrize("index", [1, 2])
    def test_each_stage_was_published_once_and_its_marks_hold(self, cascade, index):
        record = _record(cascade)
        stage = record["stages"][index - 1]
        home = _cascade_dir(cascade) / f"stage-{index}"
        artifacts = home / "artifacts"
        assert not (home / "artifacts.partial").exists()
        mark = json.loads((artifacts / ".complete.json").read_text(encoding="utf-8"))
        assert (mark["cascade_id"], mark["stage"], mark["model"], mark["round_id"]) == (
            record["cascade_id"], index, stage["model"], stage["bound_round"]), mark
        assert mark["files"], mark
        for rel, digest in mark["files"].items():
            data = (artifacts / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, f"stage {index}: {rel} moved"
        status = json.loads((artifacts / "review-plan" / ".status.json").read_text(
            encoding="utf-8"))
        block = status.get("cascade") or {}
        assert (block.get("cascade_id"), block.get("stage"), block.get("model")) == (
            record["cascade_id"], index, stage["model"]), block
        manifest = json.loads((home / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        assert (manifest["stage"], manifest["model"]) == (index, stage["model"]), manifest

    def test_stage_2s_frozen_ledger_is_stage_1s(self, cascade):
        inputs = _cascade_dir(cascade) / "stage-2" / "inputs"
        ledger = (inputs / "ledger.md").read_text(encoding="utf-8")
        assert ledger.startswith("## Earlier review rounds"), ledger[:200]
        assert "1 automated review round by" in ledger, ledger[:400]
        assert (_cascade_dir(cascade) / "stage-1" / "inputs" / "ledger.md").read_text(
            encoding="utf-8") == "", "stage 1 has no earlier round"

    def test_the_table_is_the_archive_recounted(self, cascade):
        record = _record(cascade)
        rows = _table_rows((_cascade_dir(cascade) / "summary.md").read_text(encoding="utf-8"))
        assert sorted(rows) == [1, 2], rows
        for stage in record["stages"]:
            index = stage["index"]
            review = _cascade_dir(cascade) / f"stage-{index}" / "artifacts" / "review-plan"
            groups = json.loads((review / "grouped.json").read_text(encoding="utf-8"))["groups"]
            verdicts = json.loads((review / "validation.json").read_text(
                encoding="utf-8"))["groups"]
            valid = sum(v.get("status") == "valid" for v in verdicts)
            invalid = sum(v.get("status") == "invalid" for v in verdicts)
            cells = rows[index]
            assert cells[1] == stage["model"], cells
            assert cells[2:5] == [str(len(groups)), str(valid), str(invalid)], (index, cells)
            assert len(groups) == FINDINGS and valid == FINDINGS, (
                f"stage {index}: {len(groups)} groups, {valid} valid: the verdict sheet "
                f"did not join this stage's groups")
            for key, value in (("findings", len(groups)), ("valid", valid),
                               ("invalid", invalid)):
                assert stage["counts"][key] == value, (index, key, stage["counts"])
            # The last column is Cost (after Held back, Judge and Time).
            assert cells[-1] == "not recorded" or cells[-1].startswith("$"), cells


class TestTheLaterStageSawTheLedger:
    def test_the_stub_served_both_reviews_each_from_its_own_rule(self, cascade):
        served = {str(r["fixture"]) for r in cascade.stub_log if r.get("fixture")}
        missing = {what for path, what in SERVED.items() if path not in served}
        assert not missing, f"no turn reached the stub for {sorted(missing)}: {sorted(served)}"
        extra = {what for path, what in NEVER_SERVED.items() if path in served}
        assert not extra, f"the stub answered a turn of {sorted(extra)}"

    def test_the_two_review_turns_came_from_the_two_stage_models(self, cascade):
        first = {r.get("model") for r in _review_turns(cascade, FIRST_STAGE_REVIEW)}
        later = {r.get("model") for r in _review_turns(cascade, LEDGER_REVIEW)}
        assert first and later, (first, later)
        assert first.isdisjoint(later), (
            f"one model served both stages' reviews ({first} / {later}): stage 2 did "
            f"not review with its own model")


class TestTheCompositeRanItsTwoPhases:
    def test_state_json_records_the_owned_phases_and_no_other(self, cascade):
        state = chain_lane._state(cascade)
        completed = set(state.get("phases_completed") or {})
        assert set(OWNED_PHASES) <= completed, sorted(completed)
        touched = (completed | set(state.get("phases_skipped") or {})
                   | set(state.get("phases_started") or {}))
        assert not (touched & set(NOT_OWNED_PHASES)), sorted(touched & set(NOT_OWNED_PHASES))
        record = state.get("improve_plan") or {}
        assert record.get("current_phase") == "complete", record
        assert record.get("failed_phase") is None, record

    def test_stage_1s_apply_scheduled_real_no_op_batches(self, cascade):
        report = (_cascade_dir(cascade) / "stage-1" / "artifacts" / "apply-suggestions"
                  / "write_batch_report.json")
        tasks = json.loads(report.read_text(encoding="utf-8")).get("tasks") or []
        assert tasks, "stage 1's apply scheduled zero batches"
        for task in tasks:
            assert task.get("status") == "ok", task
            assert task.get("write_outcome") == "no-op", task
            assert not task.get("files_touched"), task

    def test_the_write_dispatch_carried_the_containment_flag(self, cascade):
        flag, value = chain_lane.WRITER_CONTAINMENT_FLAGS[CLI]
        audit = (_cascade_dir(cascade) / "stage-1" / "artifacts" / "apply-suggestions"
                 / "subagent_audit.jsonl")
        lines = harness._read_jsonl(audit)
        assert lines, f"no subagent_audit.jsonl at {audit}"
        assert any(chain_lane._argv_carries(line.get("argv"), flag, value) for line in lines)


class TestTheResultLine:
    def test_one_receipt_per_command_in_order(self, cascade):
        modes = [mode for mode, _result in improve_lane._receipts_by_mode(cascade)]
        assert modes == ["improve-plan", "status"], modes

    def test_the_composite_finished_owing_nothing(self, cascade):
        _mode, result = improve_lane._receipts_by_mode(cascade)[0]
        assert (result.get("mode"), result.get("state"), result.get("exit")) == (
            "improve-plan", "done", 0), result
        assert result.get("next") is None, result
        artifacts = [str(a) for a in result.get("artifacts") or []]
        assert any(a.endswith("summary.md") and "/cascades/" in a for a in artifacts), artifacts
        step = result.get("user_next_step") or {}
        assert step.get("phase") == "generate-tasks", step
