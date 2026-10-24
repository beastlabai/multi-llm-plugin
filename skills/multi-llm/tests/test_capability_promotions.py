"""A capability promotion needs BOTH verdicts (C10b).

A green host-lane row is the only thing that promotes a `hosts/capabilities.yaml`
field from `documented` to `verified`, and a `verified` capability changes which
code path ships to users. "Green" has to mean both halves of what a row produces:

* the **container** exited 0 — the driver knows this the moment `run_row` returns;
* the **grader** (pytest, asserting over the bundle that run left behind) passed —
  which is known later, one test at a time, after the record was already written.

They come apart in practice, which is the regression these tests pin. Two rows
from one real credentialed run:

* `codex-review-plan` exited 0 and then failed
  `test_the_declared_run_state_was_reached` — the worker was never detached, so
  there was no `orchestrator-run.log` and no `.lock` and every run state read
  `None`. It printed promotions for `arguments`, `install_paths`, `multi_turn`,
  `reload` and `shim` anyway.
* `claude-code-ask` exited 0 and then failed `test_the_host_exited_zero` — the
  host exited 1 with `subtype: error_max_turns` at `num_turns: 26` against
  `--max-turns 25`. It printed promotions for `install_paths`, `reload` and
  `shim`.

Both are proposals to mark a capability *verified* on the strength of a row that
demonstrated nothing. This module needs no credential and no Docker daemon: it
drives the host-side record path over a synthetic bundle, which is exactly the
half of the harness where the promotion is decided.

**The filename is deliberately not `test_container_*`** — the same reason
`test_redirect_matrix_claims.py` gives. That glob is what the build context ships
into the image and what the gating check enumerates as container-marked; a
plain-CI module answering to it would be shipped for nothing and would fail that
marker assertion. A guard that only runs where Docker runs is a guard that does
not run on the PR where someone re-couples promotions to the container's exit
code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .container import harness

#: A host whose `capabilities.yaml` provenance is `documented` across the board,
#: so every field this bundle is evidence for is genuinely promotable — the
#: capability data is stubbed anyway, but the id is a real one so the test reads
#: like the run it came from.
HOST = "codex"
ROW_ID = "codex-review-plan"
UTC = "20260101T000000Z"

#: What the stubbed `capabilities.yaml` says today. `shim_max_lines` is already
#: `verified` on purpose: a promotion of a verified cell is noise in the diff and
#: must not be proposed, graded row or not.
PROVENANCE = {
    "shim": "documented",
    "install_paths": "documented",
    "arguments": "documented",
    "reload": "documented",
    "multi_turn": "documented",
    "shim_max_lines": "verified",
}

#: The five fields two dispatcher receipts are evidence for — the same five the
#: failing `codex-review-plan` row proposed.
PROMOTABLE = ["arguments", "install_paths", "multi_turn", "reload", "shim"]

IMAGE = {"tag": "mllm-e2e:local", "digest": "sha256:" + "ab" * 32,
         "platform": "linux/amd64", "pin_mode": "pinned"}


def _write_bundle(root: Path, row_id: str = ROW_ID, receipts: int = 2) -> Path:
    """A minimal run bundle: N dispatcher receipts and the driver's own files.

    A receipt is the harness's promotable evidence — a dispatcher invocation that
    really happened — and two of them are what `multi_turn` needs.
    """
    out_dir = root / "runs" / UTC / row_id
    receipt_dir = out_dir / "receipts" / row_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, receipts + 1):
        (receipt_dir / f"{index:03d}.receipt.json").write_text(json.dumps({
            "argv": ["python", "multi_llm.py", "review-plan", "/work/plan.md"],
            "exit": 0,
            "result": {"mode": "review-plan", "phase": "review-plan",
                       "state": "done", "exit": 0, "artifacts": []},
        }) + "\n", encoding="utf-8")
    (out_dir / "row-result.json").write_text("{}\n", encoding="utf-8")
    return out_dir


def _row(row_id: str = ROW_ID) -> harness.Row:
    return harness.Row(row_id=row_id, lane="host", host=HOST, creds="openrouter",
                       command="review-plan /work/plan.md")


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """The harness with every daemon- and repo-dependent read stubbed out.

    `image_evidence` and `cli_version` would otherwise shell out to Docker;
    `load_capabilities` and `load_host_matrix` are lru_cached reads of tracked
    files, and pinning them keeps this test about the promotion gate rather than
    about today's contents of either file.
    """
    monkeypatch.setattr(harness, "image_evidence",
                        lambda image=harness.DEFAULT_IMAGE: dict(IMAGE))
    monkeypatch.setattr(harness, "cli_version",
                        lambda cli, image=harness.DEFAULT_IMAGE: None)
    monkeypatch.setattr(harness, "load_host_matrix", lambda: {})
    monkeypatch.setattr(harness, "load_capabilities",
                        lambda: {HOST: {"provenance": dict(PROVENANCE)}})
    return tmp_path / "records"


@pytest.fixture
def ledger(rig):
    """The invocation-wide ledger, emptied around each test that drives it.

    Deliberately separate from `rig`: the tests in
    :class:`TestAPromotionNeedsBothVerdicts` are written entirely in the
    vocabulary that existed *before* this fix, so they must not touch the ledger
    at all — that is what makes them fail on the old code for the real reason
    (an edit was proposed) instead of erroring on a name it does not have yet.
    """
    harness.GRADES.reset()
    yield harness.GRADES
    harness.GRADES.reset()


@pytest.fixture
def opened(ledger, rig, tmp_path):
    """One finished row registered with the ledger, exactly as `run_row` does."""
    out_dir = _write_bundle(tmp_path)
    row = _row()
    result = harness.RowResult(row=row, exit_code=0, wall_clock_s=12.5,
                               out_dir=out_dir)
    path = harness.GRADES.open(row, result, posture=harness.load_posture(),
                               records_dir=rig)
    return row, result, path


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fields(record: dict) -> list[str]:
    return sorted(edit["field"] for edit in record["promotions"])


class TestAPromotionNeedsBothVerdicts:
    """The gate itself, at the two call sites that decide.

    Written in the vocabulary that existed before this fix — `promotions(row,
    result, outcome)` and `emit_record(row, result, posture)` — so these fail
    against the old code because it *proposes an edit*, not because it is missing
    a name.
    """

    def test_the_containers_verdict_alone_promotes_nothing(self, rig, tmp_path):
        """`pass` with no grader verdict is the state the bug shipped in."""
        out_dir = _write_bundle(tmp_path)
        row = _row()
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        assert harness.promotions(row, result, "pass") == [], (
            "a row whose container exited 0 promotes nothing on its own: the "
            "grader has not looked at the evidence yet")

    def test_the_drivers_own_record_proposes_no_edit(self, rig, tmp_path):
        """The pre-fix call site, verbatim: the driver emitting at row time.

        `run_row` used to write the row's final record right here, with the
        container's verdict and nothing else, and this is where five proposed
        capability edits appeared for a row no assertion had yet read.
        """
        out_dir = _write_bundle(tmp_path)
        row = _row()
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        path = harness.emit_record(row, result, harness.load_posture(),
                                   records_dir=rig)
        assert path is not None
        assert _record(path)["promotions"] == [], (
            "a record written before the grader ran must propose no capability "
            "edit: the container's exit code is not evidence that the row's "
            "assertions passed")

    def test_a_grader_failure_promotes_nothing(self, rig, tmp_path):
        out_dir = _write_bundle(tmp_path)
        row = _row()
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        assert harness.promotions(row, result, "pass", harness.GRADED_FAIL) == []

    def test_both_verdicts_promote_exactly_the_evidenced_fields(self, rig,
                                                                tmp_path):
        """The other direction, so the fix cannot be "promote nothing, ever"."""
        out_dir = _write_bundle(tmp_path)
        row = _row()
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        edits = harness.promotions(row, result, "pass", harness.GRADED_PASS)
        assert sorted(edit["field"] for edit in edits) == PROMOTABLE
        assert {edit["to"] for edit in edits} == {"verified"}
        assert {edit["host"] for edit in edits} == {HOST}


class TestTheRecordTheDriverWritesIsProvisional:
    """The row's record exists immediately; its promotions do not."""

    def test_the_row_gets_a_record_as_soon_as_it_finishes(self, opened):
        _row_, _result, path = opened
        assert path is not None and path.is_file(), (
            "a finished row must leave its tracked record even if the session "
            "never gets to grade it — the bundle it was derived from is "
            "gitignored and prunable")

    def test_that_record_carries_no_promotion_and_says_why(self, opened):
        _row_, _result, path = opened
        record = _record(path)
        assert record["promotions"] == [], (
            "the record written before pytest has graded the row must propose no "
            "capability edit at all")
        assert "provisional" in (record["rows"][0].get("reason") or ""), (
            "a record left behind by an interrupted session has to state that its "
            "verdict is the container's alone; otherwise it reads as a graded pass")


class TestTheGraderResolvesTheRecord:
    """`resolve_all` is where a promotion can first appear — or be withheld."""

    def test_a_row_the_grader_failed_promotes_nothing(self, opened):
        """The `codex-review-plan` case, end to end over the record path."""
        _row_, _result, path = opened
        harness.GRADES.note(
            ROW_ID,
            "tests/test_container_host_lane.py::TestTheWorkerNotTheLauncher::"
            "test_the_declared_run_state_was_reached", passed=False)
        harness.GRADES.resolve_all()
        record = _record(path)
        assert record["promotions"] == [], (
            "the container exited 0 but the grader failed the row, so it "
            "demonstrated nothing and may propose no capability edit")
        assert record["outcome"] == "fail"
        assert record["rows"][0]["outcome"] == "fail"
        assert "test_the_declared_run_state_was_reached" in \
            record["rows"][0]["reason"], (
                "the record is the tracked artifact a reviewer reads instead of "
                "the pytest scrollback, so it has to name what failed the row")

    def test_a_grader_failure_beats_a_grader_pass_on_the_same_row(self, opened):
        """Most of a row's tests passing is not most of a pass."""
        _row_, _result, path = opened
        harness.GRADES.note(ROW_ID, "…::test_a_receipt_exists_for_the_row",
                            passed=True)
        harness.GRADES.note(ROW_ID, "…::test_the_host_exited_zero", passed=False)
        harness.GRADES.resolve_all()
        assert _record(path)["promotions"] == []

    def test_a_row_that_passed_both_verdicts_promotes(self, opened):
        """The discovery tier's behaviour, which must keep working."""
        _row_, _result, path = opened
        harness.GRADES.note(ROW_ID, "…::test_the_row_completed", passed=True)
        harness.GRADES.resolve_all()
        record = _record(path)
        assert _fields(record) == PROMOTABLE
        assert record["outcome"] == "pass"
        assert all(edit["from"] == "documented" and edit["to"] == "verified"
                   for edit in record["promotions"])
        assert all(edit["evidence"] == ROW_ID for edit in record["promotions"])

    def test_an_already_verified_field_is_not_re_promoted(self, opened):
        _row_, _result, path = opened
        harness.GRADES.note(ROW_ID, "…::test_the_row_completed", passed=True)
        harness.GRADES.resolve_all()
        assert "shim_max_lines" not in _fields(_record(path))

    def test_a_row_nothing_graded_promotes_nothing(self, opened):
        """"No test asserted anything about this row" is not a pass either."""
        _row_, _result, path = opened
        harness.GRADES.resolve_all()
        record = _record(path)
        assert record["promotions"] == []
        assert "ungraded" in record["rows"][0]["reason"]

    def test_a_failed_container_promotes_nothing_even_if_the_grader_is_happy(
            self, rig, tmp_path):
        out_dir = _write_bundle(tmp_path)
        row = _row()
        result = harness.RowResult(row=row, exit_code=1, wall_clock_s=1.0,
                                   out_dir=out_dir)
        path = harness.GRADES.open(row, result, posture=harness.load_posture(),
                                   records_dir=rig)
        # Every assertion over this bundle "passed" only because they all skip on
        # a row that never produced anything; the container's exit stands.
        harness.GRADES.note(row.row_id, "…::test_that_skipped_everything",
                            passed=True)
        harness.GRADES.resolve_all()
        record = _record(path)
        assert record["promotions"] == []
        assert record["outcome"] == "fail"

    def test_the_resolved_record_still_validates_against_the_schema(self, opened):
        row, result, path = opened
        harness.GRADES.note(ROW_ID, "…::test_the_row_completed", passed=True)
        harness.GRADES.resolve_all()
        assert harness.validate_record(_record(path)) == []

    def test_the_resolved_record_digests_the_bundle_as_it_finally_stands(
            self, opened):
        """A grading test may write into the bundle it is grading.

        `test_an_unmeasured_expectation_is_recorded_never_asserted` files
        `negative-auth-observed.json` beside the evidence. A digest taken before
        the grader ran reports MISMATCH against a bundle nobody tampered with —
        the one verdict `verify_record.py` reserves for rewritten evidence.
        """
        _row_, result, path = opened
        (result.out_dir / "negative-auth-observed.json").write_text(
            "{}\n", encoding="utf-8")
        harness.GRADES.note(ROW_ID, "…::test_it", passed=True)
        harness.GRADES.resolve_all()
        record = _record(path)
        files, aggregate = harness.hash_bundle(result.out_dir)
        assert record["bundle"]["files"] == files
        assert record["bundle"]["aggregate"] == aggregate


class TestTheVerdictIsAttributedToTheRightRow:
    """Attribution is by fixture VALUE, never by id or by whatever ran last."""

    @pytest.fixture
    def two_rows(self, rig, tmp_path):
        rows = {}
        for row_id in ("codex-review-plan", "claude-code-ask"):
            out_dir = _write_bundle(tmp_path, row_id=row_id)
            row = harness.Row(row_id=row_id, lane="host", host=HOST,
                              creds="openrouter", command="review-plan /work/p.md")
            result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                       out_dir=out_dir)
            rows[row_id] = (row, result,
                            harness.GRADES.open(row, result,
                                                posture=harness.load_posture(),
                                                records_dir=rig))
        return rows

    def test_only_the_row_the_test_read_collects_the_failure(self, two_rows):
        failing, _result, _path = two_rows["codex-review-plan"]
        other = two_rows["claude-code-ask"]
        item = _FakeItem({"graded": (two_rows["codex-review-plan"][1], {}, {})})
        harness.GRADES.note_report(item, _FakeReport("call", "failed", "…::t"))
        harness.GRADES.resolve_all()
        assert _record(two_rows[failing.row_id][2])["promotions"] == []
        assert _fields(_record(other[2])) == [], (
            "the untouched row collected no verdict of its own, so it is "
            "ungraded rather than passed — and ungraded promotes nothing")

    def test_a_skipped_test_is_not_a_pass(self, two_rows):
        row, result, path = two_rows["codex-review-plan"]
        item = _FakeItem({"discovery": (result, {}, {})})
        harness.GRADES.note_report(item, _FakeReport("call", "skipped", "…::t"))
        harness.GRADES.resolve_all()
        record = _record(path)
        assert record["promotions"] == [], (
            "a tier whose every assertion skipped has graded nothing, and "
            "'nothing objected' is not the same claim as 'a test passed'")
        assert "ungraded" in record["rows"][0]["reason"]

    def test_a_passing_call_report_is_a_pass(self, two_rows):
        row, result, path = two_rows["codex-review-plan"]
        item = _FakeItem({"discovery": (result, {}, {})})
        harness.GRADES.note_report(item, _FakeReport("setup", "passed", "…::t"))
        harness.GRADES.note_report(item, _FakeReport("call", "passed", "…::t"))
        harness.GRADES.resolve_all()
        assert _fields(_record(path)) == PROMOTABLE

    def test_a_setup_pass_alone_is_not_a_pass(self, two_rows):
        """Every skipped test still reports a passing setup phase."""
        row, result, path = two_rows["codex-review-plan"]
        item = _FakeItem({"discovery": (result, {}, {})})
        harness.GRADES.note_report(item, _FakeReport("setup", "passed", "…::t"))
        harness.GRADES.resolve_all()
        assert _record(path)["promotions"] == []

    def test_a_teardown_failure_still_fails_the_row(self, two_rows):
        row, result, path = two_rows["codex-review-plan"]
        item = _FakeItem({"discovery": (result, {}, {})})
        harness.GRADES.note_report(item, _FakeReport("call", "passed", "…::t"))
        harness.GRADES.note_report(item, _FakeReport("teardown", "failed", "…::t"))
        harness.GRADES.resolve_all()
        assert _record(path)["promotions"] == []


class TestThePrintedSummary:
    """The harness PRINTS promotions and never applies them."""

    def test_a_withheld_promotion_names_the_row_and_the_assertion(self, opened):
        _row_, _result, _path = opened
        harness.GRADES.note(ROW_ID, "…::test_the_declared_run_state_was_reached",
                            passed=False)
        harness.GRADES.resolve_all()
        text = "\n".join(harness.GRADES.summary())
        assert "WITHHELD" in text and ROW_ID in text
        assert "test_the_declared_run_state_was_reached" in text
        assert "container pass, grader fail" in text, (
            "the line's whole point is that the two verdicts DIFFER on this row, "
            "so it has to print the container's own rather than the combined one")
        assert "documented → verified" not in text, (
            "a failing row must not print the edit it would have proposed as if "
            "it were one")

    def test_a_promoted_row_prints_the_edit(self, opened):
        _row_, _result, _path = opened
        harness.GRADES.note(ROW_ID, "…::test_the_row_completed", passed=True)
        harness.GRADES.resolve_all()
        text = "\n".join(harness.GRADES.summary())
        assert "documented → verified" in text
        assert "NEVER applied" in text

    def test_the_harness_never_writes_capabilities_yaml(self, opened):
        """The rule C10 states, pinned rather than trusted."""
        before = harness.CAPABILITIES.read_bytes()
        harness.GRADES.note(ROW_ID, "…::test_the_row_completed", passed=True)
        harness.GRADES.resolve_all()
        assert harness.CAPABILITIES.read_bytes() == before


class _FakeReport:
    """The three fields `note_report` reads off a pytest report."""

    def __init__(self, when: str, outcome: str, nodeid: str) -> None:
        self.when = when
        self.outcome = outcome
        self.nodeid = nodeid

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"


class _FakeItem:
    def __init__(self, funcargs: dict) -> None:
        self.funcargs = funcargs


# =========================================================================== #
# The wiring, in a real pytest session.
#
# Everything above drives the ledger directly. This drives `conftest.py`'s two
# hooks — which is the half that can silently stop working without a single
# assertion above changing colour — by running a nested pytest session whose row
# is synthetic and whose grading test is told to pass or to fail.
# =========================================================================== #

SKILL_DIR = harness.HERE.parents[1]

_NESTED_CONFTEST = '''
import sys
sys.path.insert(0, {skill_dir!r})

# The two hooks under test, imported from the suite's own conftest so this really
# pins the shipped wiring rather than a copy of it.
from tests.conftest import (          # noqa: F401
    pytest_runtest_makereport,
    pytest_sessionfinish,
)
'''

_NESTED_TEST = '''
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, {skill_dir!r})
from tests.container import harness

RECORDS = Path({records!r})
BUNDLE = Path({bundle!r})
VERDICT = os.environ["MLLM_FAKE_ROW_VERDICT"]

harness.image_evidence = lambda image=harness.DEFAULT_IMAGE: {image!r}
harness.cli_version = lambda cli, image=harness.DEFAULT_IMAGE: None
harness.load_host_matrix = lambda: {{}}
harness.load_capabilities = lambda: {{{host!r}: {{"provenance": {provenance!r}}}}}


@pytest.fixture(scope="module")
def graded():
    """Stands in for `run_row`: a finished row, registered with the ledger."""
    harness.GRADES.reset()
    row = harness.Row(row_id={row_id!r}, lane="host", host={host!r},
                      creds="openrouter", command="review-plan /work/plan.md")
    result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                               out_dir=BUNDLE)
    harness.GRADES.open(row, result, posture=harness.load_posture(),
                        records_dir=RECORDS)
    return result, {{}}, {{}}


def test_the_row_did_what_it_claimed(graded):
    assert VERDICT == "pass", "the grader failed this row"
'''


def _run_nested(tmp_path: Path, verdict: str) -> tuple[int, str, Path]:
    """One nested pytest session over a synthetic row; returns its record path."""
    session = tmp_path / f"nested-{verdict}"
    session.mkdir()
    records = session / "records"
    bundle = _write_bundle(session)
    (session / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (session / "conftest.py").write_text(
        _NESTED_CONFTEST.format(skill_dir=str(SKILL_DIR)), encoding="utf-8")
    (session / "test_nested_row.py").write_text(_NESTED_TEST.format(
        skill_dir=str(SKILL_DIR), records=str(records), bundle=str(bundle),
        image=IMAGE, host=HOST, provenance=PROVENANCE, row_id=ROW_ID),
        encoding="utf-8")
    # UTF-8 both ways: the promotion lines carry `→`, which a Windows pipe's
    # default code page cannot encode.
    env = dict(os.environ, MLLM_FAKE_ROW_VERDICT=verdict, PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-s", "-q",
         "-c", str(session / "pytest.ini"), str(session)],
        cwd=str(session), env=env, capture_output=True, encoding="utf-8")
    return (completed.returncode, completed.stdout + completed.stderr,
            records / f"{UTC}-{ROW_ID}.json")


class TestTheConftestHooksCarryTheVerdictBack:
    """The wiring, exercised by a real session rather than described."""

    def test_a_failing_assertion_withholds_the_promotion(self, tmp_path):
        code, output, record_path = _run_nested(tmp_path, "fail")
        assert code != 0, output
        assert record_path.is_file(), (
            f"the nested session left no record at all:\n{output}")
        record = _record(record_path)
        assert record["promotions"] == [], (
            f"the nested row's assertion FAILED and the harness still proposed "
            f"{_fields(record)}:\n{output}")
        assert record["outcome"] == "fail"
        assert "test_the_row_did_what_it_claimed" in record["rows"][0]["reason"]
        assert "WITHHELD" in output

    def test_a_passing_assertion_emits_the_promotion(self, tmp_path):
        code, output, record_path = _run_nested(tmp_path, "pass")
        assert code == 0, output
        record = _record(record_path)
        assert _fields(record) == PROMOTABLE, output
        assert "documented → verified" in output
