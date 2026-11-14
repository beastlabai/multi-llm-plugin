"""A cascade's forward ledger: entries from the archives, budget, rendering, freezing.

``utils/cascade_ledger.py`` builds what later rounds are told about earlier
ones (what was rejected or held back, what was changed) from the archived
stages alone, renders the four ledger files, and freezes them per stage in
``stage-<k>/inputs/`` with the manifest last. The archives are made by the
real close-out (``utils/cascade_archive.py``) over the worlds of
``tests/support/cascade_world.py``.

Contract: docs/development/workflows.md, "A cascade (``--cascade``)".
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cascade_archive as archive  # noqa: E402
from utils import cascade_ledger as ledger  # noqa: E402
from utils import cascade_provenance as provenance  # noqa: E402
from tests.support.cascade_world import (  # noqa: E402
    PLAN_TEXT,
    CascadeWorld,
    groups_of,
    ids,
    member,
    tree,
)

PLAN, CODE = "improve-plan", "improve-code"
LEDGERS = (provenance.REVIEWER_LEDGER, provenance.VALIDATOR_LEDGER,
           provenance.JUDGE_LEDGER, provenance.FULL_LEDGER)


def close(world: CascadeWorld, stage: int):
    return archive.close_out(
        plan_path=world.plan, mode=world.mode, cascade_dir=world.cascade_dir,
        cascade_id=world.cascade_id, stage=stage, model=world.models[stage - 1],
        round_id=None, recheck=lambda: True)


def inputs(world: CascadeWorld, stage: int) -> Path:
    return world.cascade_dir / f"stage-{stage}" / "inputs"


def stage_one(world: CascadeWorld):
    """Stage 1 of improve-plan with one finding per outcome, closed out."""
    world.freeze(1)
    groups = groups_of([
        member("Add a total column", "The plan never lists the total column.", "high"),
        member("Rename the module", "Rename report.py to export.py.", "low",
               reference="## Step 2: Delimiter"),
        member("Document the delimiter", "Say the delimiter is a comma.", "medium"),
        member("Add charts", "Add charts to the weekly report.", "medium"),
        member("Quote fields", "Quote fields that contain commas.", "medium"),
        member("Columns are listed", "Step 1 lists every column already.", "low"),
        member("Broken batch", "A group whose validation failed.", "low"),
    ])
    sha = world.write_review(1, groups, verdicts=[
        ("valid", "real gap"), ("invalid", "the module name is out of scope"),
        ("needs-human-decision", "ask"), ("needs-human-decision", "ask"),
        ("valid", "fine"), ("valid", "fine"), ("validation_failed", "batch failed"),
    ], started=0, completed=5)
    g = ids(groups)
    world.write_apply(1, sha, batches=[
        ("apply-batch-0", [g[0], g[2]], [str(world.plan)]),
        ("apply-batch-1", [g[4], g[5]], [str(world.plan)]),
    ], outcomes={g[0]: ("applied", ""), g[2]: ("applied", ""),
                 g[4]: ("declined", "conflicts with Step 2"), g[5]: ("no-op", "")},
        decisions={g[2]: ("approved", "judge_auto_decide", "a real gap"),
                   g[3]: ("skipped", "judge_auto_decide", "charts are not in the goal")},
        at=10, plan_edit="\n## Step 3: Totals\nAdd a total column.\n")
    close(world, 1)
    return groups


R_CHARTS = ("- R1-2 [MEDIUM] (openrouter:cheap) Add charts — Add charts to the weekly "
            "report. — reason: charts are not in the goal")
R_QUOTE = ("- R1-3 [MEDIUM] (openrouter:cheap) Quote fields — Quote fields that contain "
           "commas. — reason: conflicts with Step 2")
R_RENAME = ("- R1-1 [LOW] (openrouter:cheap) Rename the module — Rename report.py to "
            "export.py. — reason: the module name is out of scope")
A_TOTAL = "- A1-1 [HIGH] (openrouter:cheap) Add a total column — at: ## Step 1: Columns"
A_DELIM = "- A1-2 [MEDIUM] (openrouter:cheap) Document the delimiter — at: ## Step 1: Columns"


# ---------------------------------------------------------------------------
# Freezing a stage's inputs
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_stage_one_has_empty_ledgers_and_the_plan_as_it_is(self, tmp_path):
        world = CascadeWorld(tmp_path)
        manifest_path = world.freeze(1)
        directory = inputs(world, 1)
        assert manifest_path == directory / provenance.MANIFEST_NAME
        for name in LEDGERS + (provenance.CHANGES_SO_FAR,):
            assert (directory / name).read_bytes() == b"", name
        assert (directory / provenance.PLAN_BEFORE).read_bytes() == PLAN_TEXT.encode()
        manifest = provenance.load_manifest(manifest_path)
        assert (manifest["cascade_id"], manifest["mode"], manifest["stage"], manifest["of"],
                manifest["model"]) == (world.cascade_id, PLAN, 1, 3, "openrouter:cheap")
        assert sorted(manifest["files"]) == sorted(LEDGERS + (provenance.PLAN_BEFORE,
                                                              provenance.CHANGES_SO_FAR))
        block = provenance.block_from_manifest(manifest_path)
        assert provenance.ledger_text(block, provenance.REVIEWER_LEDGER) == ""

    def test_improve_code_freezes_the_ledgers_alone(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        manifest = provenance.load_manifest(world.freeze(1))
        assert sorted(manifest["files"]) == sorted(LEDGERS)
        assert sorted(p.name for p in inputs(world, 1).iterdir()) == sorted(
            LEDGERS + (provenance.MANIFEST_NAME,))

    def test_a_frozen_stage_is_returned_untouched(self, tmp_path):
        world = CascadeWorld(tmp_path)
        first = world.freeze(1)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in inputs(world, 1).iterdir()}
        world.plan.write_text(PLAN_TEXT + "\nedited after the stage started\n", encoding="utf-8")
        assert world.freeze(1) == first
        assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in inputs(world, 1).iterdir()} == before

    def test_inputs_without_a_manifest_are_rebuilt(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        good = tree(inputs(world, 2))
        (inputs(world, 2) / provenance.MANIFEST_NAME).unlink()
        (inputs(world, 2) / provenance.REVIEWER_LEDGER).write_text("half", encoding="utf-8")
        (inputs(world, 2) / "stray.tmp").write_text("x", encoding="utf-8")
        world.freeze(2)
        assert tree(inputs(world, 2)) == good

    def test_a_changed_frozen_file_stops(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        (inputs(world, 2) / provenance.JUDGE_LEDGER).write_text("edited", encoding="utf-8")
        with pytest.raises(provenance.CascadeProvenanceError):
            world.freeze(2)

    def test_changes_so_far_is_the_cascade_start_against_the_plan_now(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        diff = (inputs(world, 2) / provenance.CHANGES_SO_FAR).read_text(encoding="utf-8")
        assert diff.startswith("--- a/export.md (cascade start)\n+++ b/export.md (stage 2 start)\n")
        assert "+## Step 3: Totals" in diff
        assert (inputs(world, 2) / provenance.PLAN_BEFORE).read_bytes() == world.plan.read_bytes()

    def test_a_later_stage_needs_start_md(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        (world.cascade_dir / "start.md").unlink()
        with pytest.raises(ledger.ArchiveRefusal) as caught:
            world.freeze(2)
        assert "start.md" in caught.value.message
        assert not (inputs(world, 2) / provenance.MANIFEST_NAME).exists()

    def test_a_skipped_stage_contributes_nothing(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)  # stage 1 ran no review with a result: no archive
        world.freeze(2)
        for name in LEDGERS:
            assert (inputs(world, 2) / name).read_bytes() == b""

    def test_an_earlier_archive_that_does_not_check_out_stops(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        records = world.cascade_dir / "stage-1" / "artifacts" / "records.json"
        records.write_text("{}", encoding="utf-8")
        with pytest.raises(ledger.ArchiveRefusal) as caught:
            world.freeze(2)
        assert "changed since it was published" in caught.value.message

    def test_another_cascades_archive_stops(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        mark_path = world.cascade_dir / "stage-1" / "artifacts" / ".complete.json"
        mark = json.loads(mark_path.read_text(encoding="utf-8"))
        mark["cascade_id"] = "someone-else"
        mark_path.write_text(json.dumps(mark), encoding="utf-8")
        with pytest.raises(ledger.ArchiveRefusal) as caught:
            world.freeze(2)
        assert "belongs to cascade 'someone-else'" in caught.value.message


# ---------------------------------------------------------------------------
# The entries
# ---------------------------------------------------------------------------

class TestEntries:
    def test_one_entry_per_source_and_none_for_failed_or_no_op(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        built = ledger.build_ledger(world.cascade_dir, 1, PLAN)
        assert built.stages == (1,)
        assert [e.line() for e in built.rejected] == [R_RENAME, R_CHARTS, R_QUOTE]
        assert [e.line() for e in built.applied] == [A_TOTAL, A_DELIM]
        text = ledger.render_full(built)
        assert "Broken batch" not in text and "Columns are listed" not in text

    def test_a_scope_deferral_is_held_back(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("Also export PDF", "Add a PDF export beside the CSV.", "low"),
                            member("Fix the header", "The header misses a column.", "high")])
        g = ids(groups)
        sha = world.write_review(1, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                 started=0, completed=1, scope_policy={
                                     "review_policy": "preserve", "review_policy_source": "explicit",
                                     "recorded_at": "2026-11-13T09:00:00",
                                     "deferrals": [{"group_id": g[0], "theme": "PDF",
                                                    "title": "Also export PDF", "importance": "LOW",
                                                    "label": "Extends scope",
                                                    "reason": "the goal names CSV only"}]})
        world.write_apply(1, sha, batches=[("apply-batch-0", [g[1]], ["p"])],
                          outcomes={g[1]: ("applied", "")}, at=2)
        close(world, 1)
        built = ledger.build_ledger(world.cascade_dir, 1, PLAN)
        assert [e.line() for e in built.rejected] == [
            "- R1-1 [LOW] (openrouter:cheap) Also export PDF — Add a PDF export beside the "
            "CSV. — reason: held back by the scope policy (valid, not applied; raise it "
            "again only if the scope now allows it): the goal names CSV only"]
        # Never worded as a rejection: later reviewers may raise it once the
        # scope allows it.
        assert "reject" not in built.rejected[0].reason.lower()
        assert [e.id for e in built.applied] == ["A1-1"]

    def test_improve_code_applied_entries_name_the_batchs_files(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        world.freeze(1)
        groups = groups_of([
            member("Off by one", "The loop skips the last row.", "high", file="src/a.py"),
            member("Leak", "The file is never closed.", "medium", file="src/b.py"),
            member("Typo", "A typo in a message.", "low", file="src/c.py"),
        ], code=True)
        sha = world.write_review(1, groups, verdicts=[("valid", "x")] * 3,
                                 started=0, completed=1)
        g = ids(groups)
        world.write_apply(1, sha, batches=[
            ("apply-fix-batch-0", [g[0]], ["src/a.py"]),
            ("apply-fix-batch-1", [g[1]], ["src/b.py"]),
            ("apply-fix-batch-2", [g[2]], ["src/c.py"]),
        ], outcomes={g[0]: ("applied", ""), g[1]: ("applied", ""), g[2]: ("applied", "")},
            at=3, native=False, report_rows=[
                {"task_id": "apply-fix-batch-0", "status": "ok",
                 "files_touched": ["src/a.py", "src/a_util.py"]},
                {"task_id": "apply-fix-batch-1", "status": "failed",
                 "write_outcome": "partial", "files_touched": ["src/b.py"]},
            ])
        close(world, 1)
        built = ledger.build_ledger(world.cascade_dir, 1, CODE)
        assert [(e.id, e.at) for e in built.applied] == [
            ("A1-1", "src/a.py, src/a_util.py"),
            ("A1-2", "src/b.py"),
            ("A1-3", "src/c.py"),
        ]

    def test_long_fields_are_clipped(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        long = "word " * 120
        groups = groups_of([member("Long one", long.strip(), "medium")])
        sha = world.write_review(1, groups, verdicts=[("invalid", long.strip())],
                                 started=0, completed=1)
        world.write_apply(1, sha, batches=[], outcomes={}, at=2)
        close(world, 1)
        entry = ledger.build_ledger(world.cascade_dir, 1, PLAN).rejected[0]
        assert len(entry.description) <= ledger.DESCRIPTION_LIMIT
        assert len(entry.reason) <= ledger.REASON_LIMIT
        assert entry.description.endswith("…") and entry.reason.endswith("…")

    def test_the_ledger_reads_the_archives_alone(self, tmp_path):
        """The live dirs and ``state.json`` move on after a close-out; the
        ledger of the closed stage does not."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        before = ledger.render_full(ledger.build_ledger(world.cascade_dir, 1, PLAN))
        import shutil

        shutil.rmtree(world.review_dir)
        shutil.rmtree(world.apply_dir)
        world.state_path.write_text("{}", encoding="utf-8")
        assert ledger.render_full(ledger.build_ledger(world.cascade_dir, 1, PLAN)) == before

    def test_two_stages_number_each_stage_from_one(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        groups = groups_of([member("Name the encoding", "Say the CSV is UTF-8.", "high"),
                            member("Add charts again", "Charts would really help.", "low")])
        sha = world.write_review(2, groups, verdicts=[
            ("valid", "x"), ("invalid", "repeats R1-2 with nothing new", "R1-2")],
            started=20, completed=25)
        g = ids(groups)
        world.write_apply(2, sha, batches=[("apply-batch-0", [g[0]], ["p"])],
                          outcomes={g[0]: ("applied", "")}, at=30)
        close(world, 2)
        built = ledger.build_ledger(world.cascade_dir, 2, PLAN)
        assert [e.id for e in built.rejected] == ["R1-1", "R1-2", "R1-3", "R2-1"]
        assert [e.id for e in built.applied] == ["A1-1", "A1-2", "A2-1"]
        assert built.stages == (1, 2)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_the_reviewer_block_word_for_word(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        changes = inputs(world, 2) / provenance.CHANGES_SO_FAR
        expected = "\n".join([
            "## Earlier review rounds",
            "",
            "This plan has already been through 1 automated review round by",
            "other models. Their results are below.",
            "",
            "Rejected or held back in earlier rounds. Do not raise these again unless you",
            "have an argument the recorded reason does not answer; if you do, start the",
            "finding's description with \"Re-raises R<n>-<m>:\" and say what is new.",
            R_CHARTS, R_QUOTE, R_RENAME,
            "",
            "Changed by earlier rounds. These edits were made by other models, not by the",
            "author. Review them as you review the rest of the work: they can be wrong, and",
            "you may propose correcting or reverting them.",
            A_TOTAL, A_DELIM,
            "",
            "Every change earlier rounds made to the plan, as a unified diff against the "
            f"plan before the first round: {os.path.abspath(changes)}",
        ]) + "\n"
        assert (inputs(world, 2) / provenance.REVIEWER_LEDGER).read_text("utf-8") == expected

    def test_the_validator_and_judge_blocks(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.freeze(2)
        validator = (inputs(world, 2) / provenance.VALIDATOR_LEDGER).read_text("utf-8")
        assert validator == "\n".join([
            "Rejected or held back in earlier rounds:", R_CHARTS, R_QUOTE, R_RENAME, "",
            "Changed by earlier rounds:", A_TOTAL, A_DELIM]) + "\n"
        judge = (inputs(world, 2) / provenance.JUDGE_LEDGER).read_text("utf-8")
        assert judge == "\n".join([
            "Rejected or held back in earlier rounds:", R_CHARTS, R_QUOTE, R_RENAME]) + "\n"
        full = (inputs(world, 2) / provenance.FULL_LEDGER).read_text("utf-8")
        assert full == "\n".join([
            "# Cascade ledger", "", "Archived rounds: 1.", "",
            "## Rejected or held back in earlier rounds", "", R_RENAME, R_CHARTS, R_QUOTE, "",
            "## Changed by earlier rounds", "", A_TOTAL, A_DELIM]) + "\n"

    def test_improve_code_names_the_code_and_no_diff(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        world.freeze(1)
        groups = groups_of([member("Leak", "Never closed.", file="src/b.py")], code=True)
        sha = world.write_review(1, groups, verdicts=[("invalid", "it is closed")],
                                 started=0, completed=1)
        world.write_apply(1, sha, batches=[], outcomes={}, at=2)
        close(world, 1)
        world.freeze(2)
        block = (inputs(world, 2) / provenance.REVIEWER_LEDGER).read_text("utf-8")
        assert "This code has already been through 1 automated review round by" in block
        assert "unified diff" not in block
        assert "Changed by earlier rounds. These edits" in block and "- (none)" in block

    def test_rendering_is_deterministic(self, tmp_path):
        first = CascadeWorld(tmp_path / "one")
        second = CascadeWorld(tmp_path / "two")
        for world in (first, second):
            stage_one(world)
            world.freeze(2)
        for name in (provenance.VALIDATOR_LEDGER, provenance.JUDGE_LEDGER,
                     provenance.FULL_LEDGER):
            assert (inputs(first, 2) / name).read_bytes() == (inputs(second, 2) / name).read_bytes()
        built = ledger.build_ledger(first.cascade_dir, 1, PLAN)
        assert ledger.render_reviewer(built, full_path=Path("/x/full.md")) == \
            ledger.render_reviewer(ledger.build_ledger(first.cascade_dir, 1, PLAN),
                                   full_path=Path("/x/full.md"))


def _entry(kind, stage, number, importance, *, model="m"):
    letter = "R" if kind == ledger.REJECTED else "A"
    return ledger.LedgerEntry(
        id=f"{letter}{stage}-{number}", kind=kind, stage=stage, number=number, model=model,
        title=f"Title {letter}{stage}-{number}", importance=importance,
        description="A description." if kind == ledger.REJECTED else "",
        reason="A reason." if kind == ledger.REJECTED else "",
        at="## Step 1" if kind == ledger.APPLIED else "")


class TestBudget:
    def _ledger(self):
        rejected = (
            _entry(ledger.REJECTED, 1, 1, "LOW"),
            _entry(ledger.REJECTED, 1, 2, "HIGH"),
            _entry(ledger.REJECTED, 1, 3, "MEDIUM"),
            _entry(ledger.REJECTED, 2, 1, "HIGH"),
            _entry(ledger.REJECTED, 2, 2, "MEDIUM"),
        )
        applied = (_entry(ledger.APPLIED, 1, 1, "HIGH"), _entry(ledger.APPLIED, 2, 1, "LOW"))
        return ledger.Ledger(mode=PLAN, upto_stage=2, stages=(1, 2),
                             rejected=rejected, applied=applied)

    def test_uncapped_blocks_list_every_entry_by_priority(self):
        text = ledger.render_validator(self._ledger())
        ids_shown = [line.split()[1] for line in text.splitlines() if line.startswith("- ")]
        assert ids_shown == ["R2-1", "R1-2", "R2-2", "R1-3", "R1-1", "A2-1", "A1-1"]
        assert "left out" not in text

    def test_over_the_cap_the_most_important_entries_stay(self):
        built = self._ledger()
        lines = {e.id: e.line() for e in built.entries}
        expected = "\n".join([
            ledger.REJECTED_HEADING, lines["R2-1"], lines["R1-2"], lines["R2-2"], "",
            ledger.APPLIED_HEADING, "- (none)", "",
            "4 more entries are left out here to keep this block short.",
        ]) + "\n"
        assert ledger.render_validator(built, limit=len(expected)) == expected
        assert ledger.render_judge(built, limit=len(expected)).splitlines()[1:4] == [
            lines["R2-1"], lines["R1-2"], lines["R2-2"]]

    def test_applied_entries_come_after_every_rejected_one(self):
        """The applied list is cut first, its oldest stage first."""
        built = self._ledger()
        section = "## Step 1: a section whose name is long enough to matter here"
        applied = tuple(
            ledger.LedgerEntry(id=e.id, kind=e.kind, stage=e.stage, number=e.number,
                               model=e.model, title=e.title, importance=e.importance,
                               at=section)
            for e in built.applied)
        longer = ledger.Ledger(mode=PLAN, upto_stage=2, stages=(1, 2),
                               rejected=built.rejected, applied=applied)
        uncapped = ledger.render_validator(longer)
        overflow = "1 more entry is left out here to keep this block short."
        limit = len(uncapped) - (len(applied[0].line()) + 1) + len(overflow) + 2
        assert limit < len(uncapped)
        text = ledger.render_validator(longer, limit=limit)
        assert applied[1].line() in text and applied[0].line() not in text
        assert all(e.line() in text for e in built.rejected)
        assert text.endswith(overflow + "\n")
        assert len(text) <= limit

    def test_the_reviewer_overflow_names_the_full_ledger_and_the_validator_no_path(self, tmp_path):
        built = self._ledger()
        full = tmp_path / "stage-3" / "inputs" / "ledger-full.md"
        changes = tmp_path / "stage-3" / "inputs" / "changes-so-far.diff"
        reviewer = ledger.render_reviewer(built, full_path=full, changes_path=changes, limit=1000)
        assert len(reviewer) <= 1000
        assert "left out here to keep this block short; every entry is in " \
               f"{os.path.abspath(full)}." in reviewer
        assert reviewer.rstrip("\n").endswith(os.path.abspath(changes))
        validator = ledger.render_validator(built, limit=400)
        assert len(validator) <= 400
        assert "more entries are left out" in validator
        assert str(tmp_path) not in validator and "ledger-full" not in validator

    def test_the_inline_cap_is_eight_thousand(self):
        assert ledger.LEDGER_INLINE_MAX_CHARS == 8000
        many = tuple(_entry(ledger.REJECTED, 1, n, "MEDIUM") for n in range(1, 400))
        built = ledger.Ledger(mode=PLAN, upto_stage=1, stages=(1,), rejected=many, applied=())
        text = ledger.render_validator(built)
        assert len(text) <= ledger.LEDGER_INLINE_MAX_CHARS
        assert "more entries are left out" in text
        assert ledger.render_full(built).count("\n- R1-") == len(many)

    def test_an_empty_ledger_renders_nothing(self):
        empty = ledger.Ledger(mode=PLAN, upto_stage=0, stages=(), rejected=(), applied=())
        assert ledger.render_reviewer(empty, full_path=Path("/x")) == ""
        assert ledger.render_validator(empty) == ""
        assert ledger.render_judge(empty) == ""
        assert ledger.render_full(empty) == ""
