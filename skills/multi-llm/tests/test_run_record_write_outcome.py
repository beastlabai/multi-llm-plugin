"""The record's `write_outcome` column, and the word that had no spelling for it.

A row's tracked record (C10) summarizes every write batch the row dispatched in
one column, mapping the product's D3g vocabulary (`utils/subagent_executor`'s
`WRITE_*`) onto the record schema's closed enum. The schema closes that enum and
`harness.emit_record` validates against it before writing, so the two halves are
one change: a word in the map that the enum does not name is a **refused record**,
not a record missing a column.

`no-op` had neither half until 2026-09-22, and by then it was not an oddity.
`1633dd4` (2026-09-16) split the classifier's empty-diff branch — a `no-op` claim
over an empty diff became its own outcome rather than a `clean-failure` — and the
container lane's three `*-apply-executor` rows plus every provider-lane chain row
produce it as their NORMAL outcome. Measured over the bundles on disk: 65 of 65
write batches in `runs/20260921T224840Z`, 51 of 51 in `runs/20260922T124956Z`,
`no-op` and nothing else. Every one of those rows was recorded with no
`write_outcome` column at all; the word reached the record only as prose in the
free-text `reason`, through the `unspellable` path meant for words the schema
declines to spell. The most common write outcome the rig produces was the one a
reviewer could not filter, sort or count.

So the two properties this module pins are the two a careless edit would lose:

* **The column exists and is correct**, including the ORDER. The map is
  worst-first and the loop takes the first match, so placement decides what a
  mixed row reports. `no-op` sits last, below `applied`: a stub-fed chain row that
  applies a byte has leaked past the redirect, and one leaked `applied` among
  twelve honest `no-op`s must not be summarized by the twelve.
* **The escape hatch survives.** `policy-violation` is deliberately unspelled —
  recording it as the neighbouring `partial` would say a batch over-applied when
  it really wrote outside the declared path set — so it still travels in `reason`.

**The filename is deliberately not `test_container_*`** — the same reason
`test_capability_promotions.py` and `test_e2e_runs_retention.py` give. That glob
is what the build context ships into the image and what the marker-gating check
enumerates; this is the host-side record path over a synthetic bundle, with no
container, no credential and no Docker daemon in it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .container import harness

#: Enough of an image block to satisfy the schema without a daemon: `build_record`
#: would otherwise shell out to `docker image inspect`, and a record with no image
#: evidence is not written at all.
IMAGE = {"tag": "mllm-e2e:local", "digest": "sha256:" + "cd" * 32,
         "platform": "linux/amd64", "pin_mode": "pinned"}

#: A provider-lane chain row, which is where `no-op` is the documented outcome:
#: the stub emits no tool call, so no redirected CLI on that lane can move a byte.
ROW_ID = "chain-aider"
UTC = "20260101T000000Z"


def _bundle(root: Path, batches: dict[str, list[str]],
            row_id: str = ROW_ID) -> Path:
    """A run bundle carrying the write batch reports and nothing else.

    `batches` maps a phase directory to the `write_outcome` each of its tasks
    recorded — the same `artifacts/<plan>/<phase>/write_batch_report.json` shape
    `harness.write_outcomes` globs for, with the task fields it actually reads.
    """
    out_dir = root / "runs" / UTC / row_id
    for phase, outcomes in batches.items():
        phase_dir = out_dir / "artifacts" / "auth-feature" / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "write_batch_report.json").write_text(json.dumps({
            "report_version": 1,
            "phase": phase,
            "exit": 0,
            "tasks": [{"task_id": f"{phase}-batch-{index}",
                       "status": "ok",
                       "write_outcome": outcome,
                       "files_touched": []}
                      for index, outcome in enumerate(outcomes)],
        }) + "\n", encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "row-result.json").write_text("{}\n", encoding="utf-8")
    return out_dir


def _row(row_id: str = ROW_ID) -> harness.Row:
    return harness.Row(row_id=row_id, lane="provider", cli="aider",
                       command="full /work/plan.md")


def _entry(tmp_path: Path, *outcomes: str, reason: str | None = None,
           phases: dict[str, list[str]] | None = None) -> dict:
    """The record's `rows[]` entry for a row whose batches recorded `outcomes`."""
    row = _row()
    out_dir = _bundle(tmp_path, phases or {"apply-suggestions": list(outcomes)})
    result = harness.RowResult(row=row, exit_code=0, wall_clock_s=9.5,
                               out_dir=out_dir)
    return harness.row_record(row, result, "pass", reason)


@pytest.fixture
def daemonless(monkeypatch):
    """`build_record` without the two reads that need Docker or today's matrix."""
    monkeypatch.setattr(harness, "image_evidence",
                        lambda image=harness.DEFAULT_IMAGE: dict(IMAGE))
    monkeypatch.setattr(harness, "cli_version",
                        lambda cli, image=harness.DEFAULT_IMAGE: None)
    monkeypatch.setattr(harness, "load_host_matrix", lambda: {})


def _record(tmp_path: Path, *outcomes: str) -> dict:
    row = _row()
    out_dir = _bundle(tmp_path, {"apply-suggestions": list(outcomes)})
    result = harness.RowResult(row=row, exit_code=0, wall_clock_s=9.5,
                               out_dir=out_dir)
    return harness.build_record(row, result, harness.load_posture())


class TestTheCommonOutcomeHasAColumn:
    """The gap itself: an all-`no-op` row used to be recorded as a row with no
    write outcome at all."""

    def test_an_all_no_op_row_carries_the_column(self, tmp_path):
        entry = _entry(tmp_path, "no-op", "no-op", "no-op", "no-op")
        assert entry.get("write_outcome") == "no-op", (
            "every batch this row dispatched correctly did nothing, which is the "
            "container lane's documented outcome — a record that omits the "
            "column cannot be told from one whose row dispatched no write at all")

    def test_the_word_no_longer_arrives_as_prose(self, tmp_path):
        entry = _entry(tmp_path, "no-op", "no-op")
        assert "no-op" not in (entry.get("reason") or ""), (
            "the `unspellable` path is for words the schema declines to spell; "
            "`no-op` has a column now, so it must not also be narrated in the "
            "free-text reason a reviewer cannot filter on")

    def test_that_record_validates_against_the_tracked_schema(self, daemonless,
                                                              tmp_path):
        """The half a naive severity-list edit would miss.

        Adding `("no-op", "no-op")` to the map without widening the enum makes
        `emit_record` RAISE and write nothing — worse than the gap it closes.
        """
        record = _record(tmp_path, "no-op", "no-op", "no-op")
        assert record["rows"][-1]["write_outcome"] == "no-op"
        assert harness.validate_record(record) == []

    def test_the_writer_really_writes_it(self, daemonless, tmp_path):
        """End to end through `emit_record`, which validates before writing."""
        row = _row()
        out_dir = _bundle(tmp_path, {"apply-suggestions": ["no-op"] * 4,
                                     "implement": ["no-op"] * 9})
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=9.5,
                                   out_dir=out_dir)
        path = harness.emit_record(row, result, harness.load_posture(),
                                   records_dir=tmp_path / "records")
        assert path is not None and path.is_file()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["rows"][-1]["write_outcome"] == "no-op"

    def test_a_row_that_dispatched_no_write_still_has_no_column(self, tmp_path):
        """Absent has to keep meaning "no write batch ran".

        A read-only row (`ask`, `review-plan`) carries no write outcome, and
        giving it one — `no-op` most temptingly — would claim it dispatched a
        write batch that did nothing.
        """
        entry = _entry(tmp_path, phases={})
        assert "write_outcome" not in entry


class TestWhereNoOpSitsInTheOrder:
    """Worst-first, first match wins — so placement is the whole decision."""

    def test_one_applied_batch_outranks_a_dozen_no_ops(self, tmp_path):
        entry = _entry(tmp_path, *(["no-op"] * 12 + ["applied"]))
        assert entry["write_outcome"] == "applied", (
            "on a stub-fed lane an applied byte is a redirect leak — the finding "
            "this lane exists to catch. Ranking `no-op` above `applied` would "
            "summarize that row as one where nothing happened")

    def test_a_clean_failure_outranks_a_no_op(self, tmp_path):
        entry = _entry(tmp_path, "no-op", "clean-failure", "no-op")
        assert entry["write_outcome"] == "clean-failure", (
            "a clean failure is success CLAIMED with nothing moved; a no-op is a "
            "claim and a byte verdict that agree. The disagreement is the one a "
            "reviewer has to see")

    def test_a_partial_application_still_wins_outright(self, tmp_path):
        entry = _entry(tmp_path, *(["no-op"] * 8 + ["partial-application"]))
        assert entry["write_outcome"] == "partial", (
            "the worst outcome still stops the loop first; nothing about adding "
            "a fifth word may reorder the four above it")

    def test_no_op_is_reported_only_when_it_is_all_there_was(self, tmp_path):
        """The back-compatibility property, stated as behaviour.

        Because `no-op` is last, no row that records any other word changes
        spelling — so no already-written record is re-described by this change.
        """
        for louder in ("applied", "clean-failure", "schema-rejection",
                       "partial-application"):
            entry = _entry(tmp_path / louder, "no-op", louder)
            assert entry["write_outcome"] != "no-op", (
                f"a row that also recorded {louder!r} must not read `no-op`")


class TestTheUnspellableWordsStillTravelInProse:
    """`policy-violation` is absent from the enum ON PURPOSE, and stays absent."""

    def test_it_gets_no_column(self, tmp_path):
        entry = _entry(tmp_path, "policy-violation")
        assert "write_outcome" not in entry, (
            "recording a containment failure as the neighbouring `partial` would "
            "tell a reviewer a batch over-applied when what really happened is "
            "that it wrote outside the declared path set")

    def test_it_names_itself_in_the_free_text_reason(self, tmp_path):
        entry = _entry(tmp_path, "policy-violation",
                       reason="the grader failed this row: …::test_it")
        assert "policy-violation" in entry["reason"]
        assert "the grader failed this row" in entry["reason"], (
            "the prose is APPENDED to whatever reason the row already had, never "
            "substituted for it")

    def test_such_a_record_still_validates(self, daemonless, tmp_path):
        record = _record(tmp_path, "policy-violation")
        assert harness.validate_record(record) == [], (
            "a word with no column must degrade to prose rather than to a "
            "refused write: the schema closes the enum, so an unmapped word "
            "written into the column would cost the row its whole record")

    def test_a_no_op_row_that_also_hit_one_reports_both_ways(self, tmp_path):
        """The mixed case, which is the reason the two paths are independent."""
        entry = _entry(tmp_path, "no-op", "policy-violation")
        assert entry["write_outcome"] == "no-op"
        assert "policy-violation" in entry["reason"]


class TestTheTwoVocabulariesStayCoupled:
    """The map and the enum are one change; drift between them is a refused
    record, which is why it is checked here rather than discovered on a red lane.
    """

    def test_every_word_the_map_emits_is_a_word_the_schema_admits(self):
        enum = set(harness.load_record_schema()["properties"]["rows"]["items"]
                   ["properties"]["write_outcome"]["enum"])
        emitted = {schema_word for _, schema_word
                   in harness.WRITE_OUTCOME_SEVERITY}
        assert emitted <= enum, (
            f"the driver can emit {sorted(emitted - enum)}, which "
            f"`record.schema.json` does not name — every row recording it would "
            f"have its record REFUSED, not shortened")

    def test_the_schema_admits_no_word_the_map_cannot_emit(self):
        enum = set(harness.load_record_schema()["properties"]["rows"]["items"]
                   ["properties"]["write_outcome"]["enum"])
        emitted = {schema_word for _, schema_word
                   in harness.WRITE_OUTCOME_SEVERITY}
        assert enum <= emitted, (
            f"{sorted(enum - emitted)} is spelled in the schema but nothing "
            f"writes it — a column a reviewer can never see is a claim about "
            f"coverage the rig does not have")

    def test_every_product_word_but_the_deliberate_one_is_mapped(self):
        """`no-op` was in this gap for six days; nothing else is in it now."""
        from utils import subagent_executor as executor

        product = {executor.WRITE_APPLIED, executor.WRITE_CLEAN_FAILURE,
                   executor.WRITE_NO_OP, executor.WRITE_PARTIAL,
                   executor.WRITE_SCHEMA_REJECTED,
                   executor.WRITE_POLICY_VIOLATION}
        mapped = {product_word for product_word, _
                  in harness.WRITE_OUTCOME_SEVERITY}
        assert product - mapped == {executor.WRITE_POLICY_VIOLATION}, (
            f"unmapped product write outcomes: {sorted(product - mapped)}. A new "
            f"one reaches the record only as prose in `reason`; that is the "
            f"documented fate of `policy-violation` alone, and a deliberate "
            f"decision each time")

    def test_the_map_names_only_words_the_product_still_uses(self):
        from utils import subagent_executor as executor

        product = {executor.WRITE_APPLIED, executor.WRITE_CLEAN_FAILURE,
                   executor.WRITE_NO_OP, executor.WRITE_PARTIAL,
                   executor.WRITE_SCHEMA_REJECTED,
                   executor.WRITE_POLICY_VIOLATION}
        mapped = {product_word for product_word, _
                  in harness.WRITE_OUTCOME_SEVERITY}
        assert mapped <= product, (
            f"{sorted(mapped - product)} is mapped but no longer a word the "
            f"write executor records — a rename would leave the row summarized "
            f"by whatever it fell through to")
