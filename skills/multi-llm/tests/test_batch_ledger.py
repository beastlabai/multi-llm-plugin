"""``batch_ledger``: one write batch's records, read together
(docs/development/write-scheduler.md, "A batch's records").

Pinned here: the verdict stamp a result file carries and the resume skip's
reading of it back are one round trip, over every ``TaskOutcome`` field (a
new field fails until :data:`~utils.batch_ledger.VERDICT_FIELDS` says what the
stamp does with it); the files a phase changed are read by one function whose
call states the filter; and the records a characterisation run leaves agree
on whether each batch finished and what its checks reported, so a record
that says otherwise is named, and so is a row, a record or a finding one of
them lacks.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils.batch_ledger import (  # noqa: E402
    VERDICT_FIELDS,
    BatchLedger,
    outcome_from_verdict,
    touched_files,
    verdict_record,
)
from utils.status_file import status_path  # noqa: E402
from utils.subagent_executor import (  # noqa: E402
    STATUS_OK,
    STATUS_PARTIAL,
    WRITE_APPLIED,
    WRITE_NO_OP,
    WRITE_PARTIAL,
    TaskOutcome,
    batch_report_path,
)
from utils.write_safety import AuditLog  # noqa: E402

from tests.test_write_executor import FakeState, env  # noqa: E402,F401


# ===========================================================================
# The verdict stamp's round trip
# ===========================================================================

#: The stamped fields a resume skip reads back for one kind only: a check's
#: findings ride a verification-only (implement) task, declined and skipped
#: items an apply batch.
_IMPLEMENT_ONLY = {"check_findings", "verification_only"}
_APPLY_ONLY = {"declined", "skipped_items"}


def _stamped():
    return {name for name, how in VERDICT_FIELDS.items() if how == "stamped"}


def _full_outcome(kind: str) -> TaskOutcome:
    """An ``ok`` outcome whose every stamped field holds something."""
    return TaskOutcome(
        task_id="T001" if kind == "implement" else "apply-fix-batch-0",
        kind=kind, runner="codex:gpt", status=STATUS_OK, profile="agent-write",
        write_outcome=WRITE_NO_OP if kind == "implement" else WRITE_APPLIED,
        reason="its check reported problems",
        files_touched=["src/a.py"],
        sha256_before={"src/a.py": "0" * 64},
        sha256_after={"src/a.py": "1" * 64},
        verification_only=kind == "implement",
        check_findings=["lint fails"] if kind == "implement" else None,
        batch_check=["its own test fails"],
        batch_check_accepted={"T000": ["known"]},
        batch_check_repeats={"T002": ["same"]},
        batch_check_note="ran npm test",
        runner_notes="the validator said so",
        declined={"g1": "the plan forbids it."} if kind != "implement" else {},
        skipped_items={"g2": "you chose to keep the batch."} if kind != "implement" else {},
        unapplied_ids=["g3"] if kind != "implement" else [],
    )


def _task(outcome: TaskOutcome):
    return {"task_id": outcome.task_id, "expected_ids": ["g1", "g2", "g3"]}


def test_every_task_outcome_field_says_what_the_stamp_does_with_it():
    fields = {field.name for field in dataclasses.fields(TaskOutcome)}
    assert set(VERDICT_FIELDS) == fields, (
        "a TaskOutcome field needs a VERDICT_FIELDS row: \"stamped\" when the "
        "verdict carries it and a resume skip reads it back, else why not; "
        f"missing {sorted(fields - set(VERDICT_FIELDS))}, "
        f"stale {sorted(set(VERDICT_FIELDS) - fields)}"
    )


@pytest.mark.parametrize("kind", ["implement", "apply_code_fixes"])
def test_every_stamped_field_survives_the_round_trip(kind):
    outcome = _full_outcome(kind)
    task = _task(outcome)
    verdict = verdict_record(
        outcome, task=task, inputs_fingerprint=lambda _task: "fp",
        tree_fingerprint=lambda: "tree",
    )
    data = {"status": "applied", "ids": ["g1"], "summary": "S",
            "verdict": json.loads(json.dumps(verdict))}

    def no_fallback():
        raise AssertionError("the verdict carries files_touched")

    back = outcome_from_verdict(
        data, {}, task, kind=kind, runner="host", profile="agent-write",
        output_path="/r.json", files_on_record=no_fallback,
    )
    scoped = _APPLY_ONLY if kind == "implement" else _IMPLEMENT_ONLY
    for name in sorted(_stamped() - scoped):
        assert getattr(back, name) == getattr(outcome, name), name
    assert back.skipped is True and back.ok


def test_the_stamp_keeps_its_key_order():
    """The stamp is part of the result file's bytes: its keys keep the order
    the scheduler always wrote them in."""
    outcome = _full_outcome("apply_code_fixes")
    verdict = verdict_record(
        outcome, task=_task(outcome), inputs_fingerprint=lambda _task: "fp",
        tree_fingerprint=lambda: "tree", sandbox_rung="plugin-landlock",
    )
    assert list(verdict) == [
        "write_outcome", "status", "ok", "files_touched", "sha256_before",
        "sha256_after", "reason", "batch_check", "batch_check_accepted",
        "batch_check_repeats", "batch_check_note", "runner_notes", "declined",
        "unapplied_ids", "skipped_items", "task_id", "expected_ids",
        "inputs_fingerprint", "sandbox_rung",
    ]
    check = _full_outcome("implement")
    assert list(verdict_record(
        check, task=None, inputs_fingerprint=lambda _task: "fp",
        tree_fingerprint=lambda: "tree",
    ))[:11] == [
        "write_outcome", "status", "ok", "files_touched", "sha256_before",
        "sha256_after", "reason", "verification_only", "outcome_label", "check",
        "tree_fingerprint",
    ]


class TestTheStampReadsWhatCostsIOOnlyWhenItNeedsIt:

    def _record(self, outcome, task):
        calls = []
        verdict_record(
            outcome, task=task,
            inputs_fingerprint=lambda _task: calls.append("inputs") or "fp",
            tree_fingerprint=lambda: calls.append("tree") or "tree",
        )
        return calls

    def test_a_plain_applied_batch_reads_no_tree(self):
        outcome = _full_outcome("apply_code_fixes")
        assert self._record(outcome, _task(outcome)) == ["inputs"]

    def test_a_check_with_findings_reads_the_tree_first(self):
        outcome = _full_outcome("implement")
        assert self._record(outcome, _task(outcome)) == ["tree", "inputs"]

    def test_a_failed_check_that_moved_nothing_reads_the_tree(self):
        outcome = _full_outcome("implement")
        outcome.status, outcome.files_touched = "failed", []
        assert self._record(outcome, None) == ["tree"]

    def test_a_failed_check_not_run_again_reads_none(self):
        outcome = _full_outcome("implement")
        outcome.status, outcome.files_touched, outcome.not_rerun = "failed", [], True
        assert self._record(outcome, None) == []


def test_an_older_result_file_falls_back_to_the_audit_line_then_the_reply():
    task = {"task_id": "apply-batch-0", "expected_ids": ["s0"]}
    recorded = {"write_outcome": WRITE_APPLIED, "reason": "from the audit",
                "observed": {"plans/p.md": {"sha256_before": "a", "sha256_after": "b"}}}
    back = outcome_from_verdict(
        {"status": "no-op", "ids": ["s0"]}, recorded, task,
        kind="apply_suggestions", runner="host", profile="agent-write",
        output_path="/r.json", files_on_record=lambda: ["from/the/reply.md"],
    )
    assert back.write_outcome == WRITE_APPLIED
    assert back.reason == "from the audit"
    assert back.files_touched == ["from/the/reply.md"]
    assert back.sha256_before == {"plans/p.md": "a"}
    assert back.sha256_after == {"plans/p.md": "b"}
    bare = outcome_from_verdict(
        {"status": "no-op", "ids": []}, {}, task,
        kind="apply_suggestions", runner="host", profile="agent-write",
        output_path="/r.json", files_on_record=lambda: [],
    )
    assert bare.write_outcome == WRITE_NO_OP


# ===========================================================================
# What a phase changed
# ===========================================================================

class TestTouchedFiles:

    ROWS = [
        {"status": "ok", "files_touched": ["a.py", "b.py"]},
        {"status": "partial_application", "write_outcome": WRITE_PARTIAL,
         "files_touched": ["c.py"]},
        {"status": "failed", "write_outcome": "clean-failure", "files_touched": ["d.py"]},
        {"status": "ok", "files_touched": ["b.py", "", 7, "e.py"]},
        {"status": "ok", "files_touched": "f.py"},
        "not a row",
    ]

    def test_the_ok_rows_only(self):
        assert touched_files(self.ROWS, include_partial=False) == ["a.py", "b.py", "e.py"]

    def test_with_the_partial_applications(self):
        assert touched_files(self.ROWS, include_partial=True) == [
            "a.py", "b.py", "c.py", "e.py"]

    def test_outcomes_read_by_their_own_fields(self):
        ok = TaskOutcome(task_id="T1", kind="implement", runner="r", status=STATUS_OK,
                         files_touched=["x.py"])
        partial = TaskOutcome(task_id="T2", kind="implement", runner="r",
                              status=STATUS_PARTIAL, write_outcome=WRITE_PARTIAL,
                              files_touched=["y.py"])
        assert touched_files([ok, partial], include_partial=False) == ["x.py"]
        assert touched_files([ok, partial], include_partial=True) == ["x.py", "y.py"]
        assert touched_files([{"status": "ok", "files_touched": ["w.py"]}, ok],
                             include_partial=False) == ["w.py", "x.py"]


# ===========================================================================
# The records agree
# ===========================================================================

REASON = "The plan forbids this change."


def _apply_run(env, *, second_fails=False, resume=False, state=None):
    def respond(task, prompt):
        if task["task_id"] == "apply-batch-0":
            env.plan.write_text(env.plan.read_text() + "s1\n")
            return {"status": "applied", "ids": ["s1"],
                    "declined": [{"id": "s0", "reason": REASON}],
                    "files_touched": ["plans/p.md"], "summary": "applied s1"}
        if second_fails:
            return {"status": "failed", "ids": [], "files_touched": [],
                    "summary": "could not apply s2"}
        env.plan.write_text(env.plan.read_text() + "s2\n")
        return {"status": "applied", "ids": ["s2"],
                "files_touched": ["plans/p.md"], "summary": "applied s2"}

    state = state or FakeState(env.tmp_path / "state.json")
    tasks = [env.apply_task(0, expected_ids=["s0", "s1"]),
             env.apply_task(1, expected_ids=["s2"])]
    result = env.scheduler(tasks, respond=respond, state=state, resume=resume).run()
    return result, state


def _on_disk(env):
    return json.loads((env.tmp_path / "state.json").read_text(encoding="utf-8"))


def _implement_task(env, task_id, declared, deps=()):
    return env.apply_task(
        int(task_id[1:]) - 1, kind="implement", task_id=task_id,
        expected_ids=[task_id], output_name=f"implement_batch_{task_id}.json",
        declared_files=declared, depends_on=list(deps), prior_changes_id=task_id,
    )


#: The implement run's findings: T002's check reports REPEAT after T001's own
#: check reported it beside OWN, so REPEAT moves to T001's repeats.
REPEAT = "node scripts/e2e.mjs: login page returns 500"
OWN = "node scripts/unit.mjs: test_a expected 1 got 0"


def _implement_run(env):
    """T001 applied while its own check fails, and T002 a check that
    reported problems, one of them T001's too: every check fact a mirror
    row and ``state.json`` keep is on record."""
    def respond(task, prompt):
        if task["task_id"] == "T001":
            (env.root / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
            return {"status": "applied", "ids": ["T001"],
                    "files_touched": ["src/a.py"], "summary": "edited a.py",
                    "check": {"passed": False, "findings": [REPEAT, OWN]}}
        return {"status": "no-op", "ids": ["T002"], "files_touched": [],
                "summary": "ran the check",
                "check": {"passed": False, "findings": [REPEAT]}}

    state = FakeState(env.tmp_path / "state.json")
    return env.scheduler(
        [_implement_task(env, "T001", [str(env.root / "src" / "a.py")]),
         _implement_task(env, "T002", [])],
        kind="implement", respond=respond, state=state,
    ).run()


def _edit_json(path, change):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    change(data)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _edit_mirror(env, change):
    _edit_json(status_path(str(env.phase_dir)),
               lambda data: change(data["progress"]["write_subagents"]))


def _mirror_row(mirror, task_id):
    return next(row for row in mirror["tasks"] if row["task_id"] == task_id)


class TestTheRecordsOfARunAgree:

    def test_a_finished_apply_run(self, env):
        result, _state = _apply_run(env)
        assert result.exit_code == 0, result.messages
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []

    def test_a_stopped_apply_run(self, env):
        result, _state = _apply_run(env, second_fails=True)
        assert result.exit_code == 3
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []

    def test_a_resume_that_keeps_every_batch(self, env):
        _result, state = _apply_run(env)
        resumed, _state = _apply_run(env, resume=True, state=state)
        assert resumed.exit_code == 0, resumed.messages
        assert all(outcome.skipped for outcome in resumed.outcomes)
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []

    def test_an_implement_run_with_a_check_that_reported_problems(self, env):
        def respond(task, prompt):
            if task["task_id"] == "T001":
                (env.root / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
                return {"status": "applied", "ids": ["T001"],
                        "files_touched": ["src/a.py"], "summary": "edited a.py",
                        "check": {"passed": True, "findings": []}}
            return {"status": "no-op", "ids": ["T002"], "files_touched": [],
                    "summary": "ran the check",
                    "check": {"passed": False, "findings": ["lint fails"]}}

        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            [_implement_task(env, "T001", [str(env.root / "src" / "a.py")]),
             _implement_task(env, "T002", [])],
            kind="implement", respond=respond, state=state,
        ).run()
        assert [o.outcome_label for o in result.outcomes] == ["applied", "check-findings"]
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []

    def test_an_implement_run_whose_own_check_repeats_a_check(self, env):
        _implement_run(env)
        recorded = _on_disk(env)
        assert recorded["task_check_findings"] == {"T002": [REPEAT]}
        assert recorded["task_batch_check"] == {"T001": [OWN]}
        assert recorded["task_batch_check_repeats"] == {"T001": {"T002": [REPEAT]}}
        assert BatchLedger(env.phase_dir).disagreements(recorded) == []

    def test_an_older_mirror_with_its_lists_beside_its_rows(self, env):
        _implement_run(env)

        def older(mirror):
            # What the writer recorded beside the rows before they carried
            # everything its readers rebuild.
            mirror["failed_task_ids"] = []
            mirror["check_findings_task_ids"] = ["T002"]
            mirror["batch_check_task_ids"] = ["T001"]

        _edit_mirror(env, older)
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []


class TestADisagreementIsNamed:

    def test_a_state_record_that_says_otherwise(self, env):
        _result, _state = _apply_run(env)
        recorded = _on_disk(env)
        recorded["task_status"]["apply-batch-1"] = "failed"
        assert BatchLedger(env.phase_dir).disagreements(recorded) == [
            "apply-batch-1: write_batch_report.json says ok, "
            "state.json task_status says failed",
        ]

    def test_a_verdict_that_says_otherwise(self, env):
        result, _state = _apply_run(env)
        path = Path(result.outcomes[0].output_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["verdict"]["ok"] = False
        path.write_text(json.dumps(data), encoding="utf-8")
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            f"apply-batch-0: write_batch_report.json says ok, {path.name}'s "
            f"verdict says ok=False",
        ]

    def test_a_processed_item_that_says_otherwise(self, env):
        _result, _state = _apply_run(env, second_fails=True)
        recorded = _on_disk(env)
        recorded["processed_apply-suggestions"]["s2"]["status"] = "applied"
        assert BatchLedger(env.phase_dir).disagreements(recorded) == [
            "apply-batch-1: write_batch_report.json says not ok, "
            "processed item s2 says applied",
        ]

    @pytest.mark.parametrize("record", ["executor", "mirror"])
    def test_a_missing_peer_row_is_named(self, env, record):
        _implement_run(env)
        drop = lambda document: document.update(  # noqa: E731
            tasks=[row for row in document["tasks"] if row["task_id"] != "T002"])
        if record == "executor":
            path = batch_report_path(str(env.phase_dir), "implement")
            _edit_json(path, drop)
            where = Path(path).name
        else:
            _edit_mirror(env, drop)
            where = ".status.json write_subagents"
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            f"T002: {where} has no row"]

    def test_a_missing_mirror_row_is_named_for_every_row(self, env):
        _implement_run(env)
        _edit_mirror(env, lambda mirror: mirror.update(tasks=[]))
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            "T001: .status.json write_subagents has no row",
            "T002: .status.json write_subagents has no row",
        ]

    def test_a_missing_record_is_named_once(self, env):
        _implement_run(env)
        path = Path(batch_report_path(str(env.phase_dir), "implement"))
        path.unlink()
        Path(status_path(str(env.phase_dir))).unlink()
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            f"{path.name}: missing or unreadable",
            ".status.json write_subagents: missing or unreadable",
        ]

    @pytest.mark.parametrize("record, change, line", [
        ("mirror", lambda m: _mirror_row(m, "T002").pop("check_findings"),
         f"T002: write_batch_report.json says check findings {[REPEAT]!r}, "
         ".status.json write_subagents says []"),
        ("mirror", lambda m: _mirror_row(m, "T001").pop("batch_check_findings"),
         f"T001: write_batch_report.json says own-check findings {[OWN]!r}, "
         ".status.json write_subagents says []"),
        ("mirror", lambda m: m.pop("batch_check_repeats"),
         "T001: write_batch_report.json says repeated findings ['T002'], "
         ".status.json write_subagents says []"),
        ("state", lambda s: s.pop("task_check_findings"),
         f"T002: write_batch_report.json says check findings {[REPEAT]!r}, "
         "state.json task_check_findings says []"),
        ("state", lambda s: s.pop("task_batch_check"),
         f"T001: write_batch_report.json says own-check findings {[OWN]!r}, "
         "state.json task_batch_check says []"),
        ("state", lambda s: s.pop("task_batch_check_repeats"),
         f"T001: write_batch_report.json says repeated findings "
         f"{ {'T002': [REPEAT]}!r}, state.json task_batch_check_repeats says {{}}"),
    ], ids=["mirror-check", "mirror-own-check", "mirror-repeats",
            "state-check", "state-own-check", "state-repeats"])
    def test_missing_findings_are_named(self, env, record, change, line):
        _implement_run(env)
        recorded = _on_disk(env)
        if record == "mirror":
            _edit_mirror(env, change)
        else:
            change(recorded)
        assert BatchLedger(env.phase_dir).disagreements(recorded) == [line]

    def test_an_older_mirrors_list_its_rows_contradict(self, env):
        _implement_run(env)
        _edit_mirror(env, lambda mirror: mirror.update(check_findings_task_ids=[]))
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            ".status.json write_subagents: check_findings_task_ids [] "
            "against its rows' ['T002']",
        ]

    def test_a_missing_state_record_is_named(self, env):
        _apply_run(env)
        recorded = _on_disk(env)
        del recorded["task_status"]["apply-batch-1"]
        del recorded["processed_apply-suggestions"]["s0"]
        assert BatchLedger(env.phase_dir).disagreements(recorded) == [
            "apply-batch-0: processed item s0 has no record",
            "apply-batch-1: state.json task_status has no entry",
        ]

    def test_a_missing_audit_line_is_named(self, env):
        # The simulated invoker records no attempts, so the rows say a
        # runner ran them here; the audit has their completed lines.
        _apply_run(env)
        report = env.phase_dir / "write_batch_report.json"
        _edit_json(report, lambda data: [
            row.update(attempt_count=1) for row in data["tasks"]])
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == []
        audit = Path(AuditLog(str(env.phase_dir)).path)
        audit.write_text("".join(
            line + "\n" for line in audit.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") != "completed"
        ), encoding="utf-8")
        assert BatchLedger(env.phase_dir).disagreements(_on_disk(env)) == [
            "apply-batch-0: the audit has no completed line",
            "apply-batch-1: the audit has no completed line",
        ]

    def test_no_report_is_named(self, tmp_path):
        assert BatchLedger(tmp_path).disagreements() == [
            "write_batch_report.json: missing or unreadable"]


def test_the_ledger_reads_the_reports_rows_and_date(env):
    _apply_run(env)
    ledger = BatchLedger(env.phase_dir)
    assert [row["task_id"] for row in ledger.rows()] == ["apply-batch-0", "apply-batch-1"]
    assert ledger.touched(include_partial=False) == ["plans/p.md"]
    assert ledger.written_at() is not None
    assert BatchLedger(env.tmp_path / "nowhere").rows() is None
