"""Tests for prior_changes utility — JSONL-based prior batch changes tracking."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.prior_changes import (
    DEFAULT_TEXT,
    ENTRY_ID_DIGEST_CHARS,
    FILENAME,
    MAX_CHARS,
    PHASE_ALIASES,
    _jsonl_path,
    _load_entries,
    append,
    canonical_phase,
    entries as read_entries,
    entry_id,
    render,
    stable_key,
)

UTIL_PATH = str(Path(__file__).parent.parent / "utils" / "prior_changes.py")


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run the prior_changes CLI utility and return the completed process.

    The child bootstraps its streams to UTF-8, so the parent must decode with
    UTF-8 too — `text=True` alone would use the locale codec (cp1252 on
    Windows) and mangle non-ASCII output such as the em dash in DEFAULT_TEXT.
    """
    return subprocess.run(
        [sys.executable, UTIL_PATH, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=stdin,
    )


def _append(output_dir: str, entry_id: str, phase: str, summary: str) -> subprocess.CompletedProcess:
    """Shorthand for the append command via CLI."""
    return _run_cli(
        "append",
        "--output-dir", output_dir,
        "--id", entry_id,
        "--phase", phase,
        "--summary", summary,
    )


def _read(output_dir: str) -> subprocess.CompletedProcess:
    """Shorthand for the read command via CLI."""
    return _run_cli("read", "--output-dir", output_dir)


def _clear(output_dir: str) -> subprocess.CompletedProcess:
    """Shorthand for the clear command via CLI."""
    return _run_cli("clear", "--output-dir", output_dir)


class TestPriorChanges:
    """Core tests for prior_changes utility."""

    def test_read_empty_returns_default(self, tmp_path: Path) -> None:
        """No file present -> returns the default 'first batch' text."""
        result = _read(str(tmp_path))
        assert result.returncode == 0
        assert result.stdout.strip() == DEFAULT_TEXT

    def test_append_creates_file(self, tmp_path: Path) -> None:
        """First append creates the JSONL file with exactly one entry."""
        result = _append(str(tmp_path), "step_1", "review", "Initial review")
        assert result.returncode == 0

        jsonl_file = tmp_path / FILENAME
        assert jsonl_file.exists()

        entries = _load_entries(jsonl_file)
        assert len(entries) == 1
        assert entries[0]["id"] == "step_1"
        assert entries[0]["phase"] == "review"
        assert entries[0]["summary"] == "Initial review"
        assert "timestamp" in entries[0]

    def test_append_then_read(self, tmp_path: Path) -> None:
        """Append 2 entries with different IDs, read returns formatted numbered list."""
        _append(str(tmp_path), "step_1", "review", "First review pass")
        _append(str(tmp_path), "step_2", "apply-suggestions", "Applied formatting fixes")

        result = _read(str(tmp_path))
        assert result.returncode == 0
        output = result.stdout.strip()

        assert "1. **Batch 1** (step_1): First review pass" in output
        assert "2. **Batch 2** (step_2): Applied formatting fixes" in output

    def test_read_ordering(self, tmp_path: Path) -> None:
        """Entries returned in append order. Batch numbers sequential from 1."""
        _append(str(tmp_path), "z_last", "phase_a", "Summary Z")
        _append(str(tmp_path), "a_first", "phase_b", "Summary A")
        _append(str(tmp_path), "m_middle", "phase_c", "Summary M")

        result = _read(str(tmp_path))
        output = result.stdout.strip()
        lines = output.splitlines()

        assert lines[0].startswith("1. **Batch 1** (z_last):")
        assert lines[1].startswith("2. **Batch 2** (a_first):")
        assert lines[2].startswith("3. **Batch 3** (m_middle):")

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        """Clear removes the JSONL file entirely."""
        _append(str(tmp_path), "step_1", "review", "Something")
        assert (tmp_path / FILENAME).exists()

        result = _clear(str(tmp_path))
        assert result.returncode == 0
        assert not (tmp_path / FILENAME).exists()

    def test_read_after_clear(self, tmp_path: Path) -> None:
        """Clear then read returns the default text."""
        _append(str(tmp_path), "step_1", "review", "Something")
        _clear(str(tmp_path))

        result = _read(str(tmp_path))
        assert result.stdout.strip() == DEFAULT_TEXT

    def test_append_idempotent_same_id_phase(self, tmp_path: Path) -> None:
        """Same id + phase combination is a no-op; no duplicate entry created."""
        _append(str(tmp_path), "step_1", "review", "First summary")
        _append(str(tmp_path), "step_1", "review", "Duplicate summary")

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 1
        assert entries[0]["summary"] == "First summary"

    def test_append_same_id_different_phase(self, tmp_path: Path) -> None:
        """Same id but different phase creates a separate entry."""
        _append(str(tmp_path), "step_1", "review", "Review summary")
        _append(str(tmp_path), "step_1", "apply-suggestions", "Apply summary")

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 2
        assert entries[0]["phase"] == "review"
        assert entries[1]["phase"] == "apply-suggestions"

    def test_read_ordering_deterministic(self, tmp_path: Path) -> None:
        """Entries appear in append order regardless of ID alphabetical order."""
        ids_in_order = ["zzz", "aaa", "mmm", "bbb"]
        for i, eid in enumerate(ids_in_order):
            _append(str(tmp_path), eid, f"phase_{i}", f"Summary for {eid}")

        result = _read(str(tmp_path))
        lines = result.stdout.strip().splitlines()

        for i, eid in enumerate(ids_in_order):
            assert lines[i].startswith(f"{i + 1}. **Batch {i + 1}** ({eid}):")

    def test_append_via_summary_stdin(self, tmp_path: Path) -> None:
        """Append using --summary-stdin flag, piping summary via stdin."""
        result = _run_cli(
            "append",
            "--output-dir", str(tmp_path),
            "--id", "step_stdin",
            "--phase", "review",
            "--summary-stdin",
            stdin="Summary from stdin pipe",
        )
        assert result.returncode == 0

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 1
        assert entries[0]["summary"] == "Summary from stdin pipe"

    def test_append_rejects_both_summary_flags(self, tmp_path: Path) -> None:
        """Providing both --summary and --summary-stdin exits with error."""
        result = _run_cli(
            "append",
            "--output-dir", str(tmp_path),
            "--id", "step_1",
            "--phase", "review",
            "--summary", "inline",
            "--summary-stdin",
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr

    def test_append_rejects_neither_summary_flag(self, tmp_path: Path) -> None:
        """Providing neither --summary nor --summary-stdin exits with error."""
        result = _run_cli(
            "append",
            "--output-dir", str(tmp_path),
            "--id", "step_1",
            "--phase", "review",
        )
        assert result.returncode != 0
        assert "required" in result.stderr

    def test_read_corrupted_jsonl_returns_default(self, tmp_path: Path) -> None:
        """Corrupted JSON lines in the file cause read to return the default text."""
        jsonl_file = tmp_path / FILENAME
        jsonl_file.write_text("this is not json\n{broken\n", encoding="utf-8")

        result = _read(str(tmp_path))
        assert result.stdout.strip() == DEFAULT_TEXT

    def test_resume_does_not_duplicate_entries(self, tmp_path: Path) -> None:
        """Re-appending an existing entry is a no-op; new entries still append."""
        _append(str(tmp_path), "step_1", "review", "Batch 1")
        _append(str(tmp_path), "step_2", "review", "Batch 2")
        _append(str(tmp_path), "step_3", "review", "Batch 3")

        # Re-append batch 1 — should be skipped
        _append(str(tmp_path), "step_1", "review", "Batch 1 again")
        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 3

        # Append truly new batch 4
        _append(str(tmp_path), "step_4", "review", "Batch 4")
        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 4

    def test_full_batch_loop_integration(self, tmp_path: Path) -> None:
        """Full read -> substitute -> append cycle simulating orchestrator flow."""
        prompt_template = "Prior changes:\n{prior_changes_context}\n\nNow do the next thing."

        # Step 1: Read returns default (empty state)
        result = _read(str(tmp_path))
        prior_text = result.stdout.strip()
        assert prior_text == DEFAULT_TEXT

        # Step 2: Substitute into prompt template
        prompt = prompt_template.replace("{prior_changes_context}", prior_text)
        assert DEFAULT_TEXT in prompt

        # Step 3: Append batch result
        _append(str(tmp_path), "step_1", "apply-suggestions", "Added error handling guidance")

        # Step 4: Read returns formatted list
        result = _read(str(tmp_path))
        prior_text = result.stdout.strip()
        assert "**Batch 1** (step_1): Added error handling guidance" in prior_text

        # Step 5: Substitute again, verify prompt has prior changes
        prompt = prompt_template.replace("{prior_changes_context}", prior_text)
        assert "Added error handling guidance" in prompt
        assert DEFAULT_TEXT not in prompt

        # Step 6: Append second batch
        _append(str(tmp_path), "step_2", "apply-suggestions", "Refactored validation logic")

        # Step 7: Read returns both entries in order
        result = _read(str(tmp_path))
        output = result.stdout.strip()
        lines = output.splitlines()
        assert len(lines) == 2
        assert "step_1" in lines[0]
        assert "step_2" in lines[1]


class TestPriorChangesIntegration:
    """Integration tests simulating orchestrator flow patterns."""

    def test_multi_batch_run_accumulates_context(self, tmp_path: Path) -> None:
        """Simulate 3+ batch run; each read includes all prior summaries in order."""
        summaries = [
            ("step_1", "review", "Reviewed architecture"),
            ("step_2", "apply-suggestions", "Applied naming conventions"),
            ("step_3", "review", "Reviewed test coverage"),
            ("step_4", "apply-suggestions", "Added missing edge-case tests"),
        ]

        for i, (eid, phase, summary) in enumerate(summaries):
            # Read before each batch — should show all prior entries
            result = _read(str(tmp_path))
            output = result.stdout.strip()
            if i == 0:
                assert output == DEFAULT_TEXT
            else:
                # All prior entries should be present
                for j in range(i):
                    assert summaries[j][2] in output

            # Append current batch
            _append(str(tmp_path), eid, phase, summary)

        # Final read should include all entries in order
        result = _read(str(tmp_path))
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 4
        for idx, (_, _, summary) in enumerate(summaries):
            assert summary in lines[idx]

    def test_failure_then_resume_preserves_history(self, tmp_path: Path) -> None:
        """Simulate failure mid-run; resume preserves entries + new ones append."""
        # Batches before failure
        _append(str(tmp_path), "step_1", "review", "Pre-failure review")
        _append(str(tmp_path), "step_2", "apply-suggestions", "Pre-failure apply")

        entries_before = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries_before) == 2

        # Simulate failure — nothing happens to the file

        # Resume: re-append existing entries (should be no-ops)
        _append(str(tmp_path), "step_1", "review", "Pre-failure review")
        _append(str(tmp_path), "step_2", "apply-suggestions", "Pre-failure apply")

        entries_after_resume = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries_after_resume) == 2  # No duplicates

        # New entries after resume
        _append(str(tmp_path), "step_3", "review", "Post-resume review")
        entries_final = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries_final) == 3

    def test_fresh_resets_then_rebuilds(self, tmp_path: Path) -> None:
        """Clear then rebuild from scratch — starts over cleanly."""
        _append(str(tmp_path), "step_1", "review", "Old entry")
        _clear(str(tmp_path))

        result = _read(str(tmp_path))
        assert result.stdout.strip() == DEFAULT_TEXT

        _append(str(tmp_path), "step_new_1", "review", "Fresh start")
        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 1
        assert entries[0]["id"] == "step_new_1"

    def test_resume_preserves_prior_changes(self, tmp_path: Path) -> None:
        """Non-fresh rerun does NOT clear — existing entries remain."""
        _append(str(tmp_path), "step_1", "review", "Preserved entry")
        _append(str(tmp_path), "step_2", "apply-suggestions", "Also preserved")

        # Simulate resume (no clear call) — just read
        result = _read(str(tmp_path))
        output = result.stdout.strip()
        assert "Preserved entry" in output
        assert "Also preserved" in output

        # Append more
        _append(str(tmp_path), "step_3", "review", "New after resume")
        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 3

    def test_fresh_clears_prior_changes(self, tmp_path: Path) -> None:
        """Fresh run clears the file, wiping all prior entries."""
        _append(str(tmp_path), "step_1", "review", "Will be cleared")
        _append(str(tmp_path), "step_2", "review", "Also cleared")
        assert len(_load_entries(_jsonl_path(str(tmp_path)))) == 2

        _clear(str(tmp_path))

        assert not (tmp_path / FILENAME).exists()
        result = _read(str(tmp_path))
        assert result.stdout.strip() == DEFAULT_TEXT

    def test_fresh_clears_via_orchestrator_base(self, tmp_path: Path) -> None:
        """Fresh mode through ApplyOrchestratorBase._setup removes prior_changes.jsonl.

        Regression guard: ensures that the orchestrator base class fresh-start
        path actually deletes the prior changes file, not just the standalone
        utility clear helper.
        """
        import argparse

        from utils.apply_orchestrator_base import ApplyOrchestratorBase

        # Build a minimal concrete subclass with required class attributes.
        class _StubOrchestrator(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

        # Set up a plan file and output directory that match the orchestrator's
        # derive_prefix / find_output_dir conventions.
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# stub plan\n", encoding="utf-8")

        out_dir = tmp_path / "plan"  # sanitize_prefix("plan.md") -> "plan"
        out_dir.mkdir()

        # Seed prior_changes.jsonl inside the output directory.
        _append(str(out_dir), "step_1", "review", "Will be cleared by fresh")
        assert (out_dir / FILENAME).exists()

        # Construct the orchestrator with --fresh enabled.
        args = argparse.Namespace(
            plan_file=str(plan_file),
            fresh=True,
            yes=False,
            force=False,
            approve_all=False,
            skip_human_review=False,
            skip=False,
            verbose=False,
        )
        orch = _StubOrchestrator(args)
        orch._setup()

        # The file must have been removed through the base-class fresh path.
        assert not (out_dir / FILENAME).exists()


class TestPriorChangesEdgeCases:
    """Edge case tests for prior_changes utility."""

    def test_bounded_history_long_run(self, tmp_path: Path) -> None:
        """30+ entries: read stays within MAX_CHARS, compressed summary preserves count."""
        for i in range(35):
            _append(str(tmp_path), f"step_{i}", f"phase_{i}", f"Summary for batch {i}")

        result = _read(str(tmp_path))
        output = result.stdout.strip()

        # Output must be within bounds
        assert len(output) <= MAX_CHARS

        # Should contain compressed summary for earlier batches
        assert "... and 20 earlier batches applied miscellaneous changes" in output

        # Should still show the last 15 entries with correct numbering
        lines = output.splitlines()
        # First line is the compressed summary
        assert lines[0].startswith("... and ")
        # Remaining lines should be the last 15 entries (batch 21 through 35)
        for idx, line in enumerate(lines[1:], start=21):
            assert line.startswith(f"{idx}. **Batch {idx}**")

    def test_special_characters_in_summary(self, tmp_path: Path) -> None:
        """Multiline summaries and shell metacharacters stored correctly."""
        special_summaries = [
            'Summary with $dollar and `backticks`',
            'Summary with "double quotes" and \'single quotes\'',
            'Summary with backslash \\ and pipe |',
            'Line one\nLine two\nLine three',
        ]

        for i, summary in enumerate(special_summaries):
            # Use stdin for reliable special character handling
            result = _run_cli(
                "append",
                "--output-dir", str(tmp_path),
                "--id", f"special_{i}",
                "--phase", "review",
                "--summary-stdin",
                stdin=summary,
            )
            assert result.returncode == 0

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 4

        assert entries[0]["summary"] == 'Summary with $dollar and `backticks`'
        assert entries[1]["summary"] == 'Summary with "double quotes" and \'single quotes\''
        assert entries[2]["summary"] == 'Summary with backslash \\ and pipe |'
        # stdin.read().strip() will trim trailing newline but multiline content is preserved
        assert "Line one\nLine two\nLine three" in entries[3]["summary"]

    def test_summary_at_200_char_limit(self, tmp_path: Path) -> None:
        """Summary exactly at 200 characters works fine."""
        summary_200 = "A" * 200
        assert len(summary_200) == 200

        _append(str(tmp_path), "step_limit", "review", summary_200)

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 1
        assert entries[0]["summary"] == summary_200
        assert len(entries[0]["summary"]) == 200

    def test_summary_exceeding_200_chars(self, tmp_path: Path) -> None:
        """Summary > 200 chars is stored as-is (truncation is caller responsibility)."""
        summary_long = "B" * 500
        assert len(summary_long) == 500

        _append(str(tmp_path), "step_long", "review", summary_long)

        entries = _load_entries(_jsonl_path(str(tmp_path)))
        assert len(entries) == 1
        assert entries[0]["summary"] == summary_long
        assert len(entries[0]["summary"]) == 500


class TestEntryId:
    """The id `append` is keyed on: unique per batch AND stable across a redispatch.

    Both halves fail silently when they are wrong, in opposite directions — a
    colliding id DROPS an entry (append is idempotent on `(id, phase)`, so the
    second batch under a shared key is skipped at exit 0), an unstable id DOUBLES
    one. Which is why the id is a content key plus a digest of the batch's own
    members, and never a loop index.
    """

    def test_the_key_alone_is_not_the_id(self) -> None:
        assert entry_id("src/a.py", ["i1"], fallback="b") != "src/a.py"
        assert entry_id("src/a.py", ["i1"], fallback="b").startswith("src/a.py#")

    def test_batches_sharing_a_key_get_different_ids(self) -> None:
        first = entry_id("src/a.py", ["i1", "i2", "i3"], fallback="batch-0")
        second = entry_id("src/a.py", ["i4", "i5", "i6"], fallback="batch-1")
        assert first != second

    def test_the_same_members_always_produce_the_same_id(self) -> None:
        assert entry_id("s", ["a", "b"], fallback="x") == entry_id(
            "s", ["a", "b"], fallback="y"
        )

    def test_member_order_is_part_of_the_identity(self) -> None:
        assert entry_id("s", ["a", "b"], fallback="x") != entry_id(
            "s", ["b", "a"], fallback="x"
        )

    def test_a_keyless_batch_still_gets_a_content_id(self) -> None:
        """Both invokers must spell this the same, so the base is a constant."""
        assert entry_id("", ["a"], fallback="batch-7") == entry_id(
            "   ", ["a"], fallback="batch-9"
        )
        assert entry_id("", ["a"], fallback="batch-7").startswith("batch#")

    def test_the_fallback_is_only_for_a_batch_with_no_members(self) -> None:
        assert entry_id("", [], fallback="batch-3") == "batch-3"
        assert entry_id("", ["", "  ", None], fallback="batch-3") == "batch-3"
        assert entry_id("sec", [], fallback="batch-3") == "sec"

    def test_the_digest_is_the_declared_width(self) -> None:
        _, _, digest = entry_id("k", ["a"], fallback="x").partition("#")
        assert len(digest) == ENTRY_ID_DIGEST_CHARS
        assert all(c in "0123456789abcdef" for c in digest)

    def test_stable_key_takes_the_first_non_empty_candidate(self) -> None:
        assert stable_key([None, "", "  ", "g1", "id"], "f0") == "g1"
        assert stable_key([None, ""], "f0") == "f0"

    def test_every_batch_on_one_key_lands_in_the_log(self, tmp_path: Path) -> None:
        """The user-visible failure, at the level `append` sees it.

        Nineteen batches on one file key: with the file key as the id exactly one
        entry survives, so every batch after the first reads a context that
        pretends its predecessors never ran.
        """
        out = str(tmp_path)
        batches = [[f"fix-{n}-{k}" for k in range(3)] for n in range(19)]

        collided = sum(
            append(out, "src/index.mjs", "apply-fixes", f"batch {n}")
            for n, _ in enumerate(batches)
        )
        assert collided == 1  # the bug, pinned
        assert len(read_entries(out)) == 1

        keyed = sum(
            append(
                out,
                entry_id("src/index.mjs", members, fallback=f"batch-{n}"),
                "apply-fixes",
                f"batch {n}",
            )
            for n, members in enumerate(batches)
        )
        assert keyed == 19
        assert len(read_entries(out)) == 20  # the collided one plus all nineteen

    def test_a_redispatched_batch_is_still_recorded_once(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        members = ["a", "b"]
        eid = entry_id("sec", members, fallback="batch-0")
        assert append(out, eid, "apply-suggestions", "first") is True
        assert append(out, eid, "apply-suggestions", "again") is False
        assert len(read_entries(out)) == 1


class TestRenderPhaseFiltering:
    """`render(output_dir, phase=...)` prefers a phase's own entries (N8).

    The log is one file shared by every write phase a plan ever runs; without
    filtering, a phase's inter-task context competes for its budget with
    however many batches earlier phases logged. `phase=None` stays today's
    unfiltered tail.
    """

    def test_no_phase_argument_is_unfiltered(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        append(out, "s1", "apply-suggestions", "Applied A")
        append(out, "s2", "implement", "Did B")

        assert render(out) == render(out, phase=None)
        output = render(out)
        assert "Applied A" in output
        assert "Did B" in output
        assert "other phases" not in output

    def test_prefers_same_phase_entries_over_other_phases(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        append(out, "s1", "apply-suggestions", "Applied A")
        append(out, "s2", "apply-task-suggestions", "Applied B")
        append(out, "t1", "implement", "T001: did the thing")
        append(out, "t2", "implement", "T002: did another thing")

        output = render(out, phase="implement")

        assert "T001: did the thing" in output
        assert "T002: did another thing" in output
        assert "Applied A" not in output
        assert "Applied B" not in output
        assert "... and 2 earlier batches from other phases" in output

    def test_no_same_phase_entries_falls_back_to_unfiltered(self, tmp_path: Path) -> None:
        """A phase's first-ever dispatch has no entries of its own to prefer."""
        out = str(tmp_path)
        append(out, "s1", "apply-suggestions", "Applied A")
        append(out, "s2", "apply-task-suggestions", "Applied B")

        filtered = render(out, phase="implement")
        assert filtered == render(out)
        assert "Applied A" in filtered
        assert "Applied B" in filtered
        assert "other phases" not in filtered

    def test_same_phase_numbering_is_scoped_to_that_phase(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        append(out, "s1", "apply-suggestions", "Applied A")
        append(out, "s2", "apply-suggestions", "Applied B")
        append(out, "t1", "implement", "T001 done")

        output = render(out, phase="implement")
        lines = output.splitlines()
        assert lines[0] == "1. **Batch 1** (t1): T001 done"
        assert lines[1] == "... and 2 earlier batches from other phases"

    def test_phase_scoped_tail_respects_the_entry_cap(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        for i in range(7):
            append(out, f"o{i}", "apply-suggestions", f"other {i}")
        for i in range(20):
            append(out, f"i{i}", "implement", f"T{i:03d} summary")

        output = render(out, phase="implement")

        # DEFAULT_TAIL (15) caps same-phase entries; the oldest 5 of the 20
        # implement entries are summarized, independent of the 7 other-phase
        # entries, which are reported separately.
        assert "... and 5 earlier batches applied miscellaneous changes" in output
        assert "... and 7 earlier batches from other phases" in output
        assert "T004 summary" not in output
        assert "T019 summary" in output

    def test_phase_scoped_output_still_respects_max_chars(self, tmp_path: Path) -> None:
        out = str(tmp_path)
        for i in range(5):
            append(out, f"o{i}", "apply-suggestions", "x" * 300)
        for i in range(20):
            append(out, f"i{i}", "implement", "y" * 300)

        output = render(out, phase="implement")

        assert "... and 5 earlier batches from other phases" in output
        assert "earlier batches applied miscellaneous changes" in output
        assert len(output) <= MAX_CHARS

    def test_one_oversized_entry_cannot_escape_the_reader_budget(self, tmp_path: Path) -> None:
        append(str(tmp_path), "T001", "implement", "x" * 10000)
        output = render(str(tmp_path), phase="implement", max_chars=120)
        assert len(output) <= 120
        assert "T001" in output
        assert output.endswith("... [truncated]")

    def test_empty_log_and_tiny_budget_stay_bounded(self, tmp_path: Path) -> None:
        assert len(render(str(tmp_path), max_chars=4)) <= 4
        append(str(tmp_path), "T001", "implement", "x" * 10000)
        for limit in (0, 1, 4):
            assert len(render(str(tmp_path), max_chars=limit)) <= limit

    def test_a_reader_can_widen_the_budget_the_write_batch_fixed(
        self, tmp_path: Path
    ) -> None:
        """The code-review prompts summarize the whole phase, not a batch tail.

        At the write-batch budget the oldest entries fall off the top — which
        is how the entry that motivated showing reviewers this log at all
        (``accel tuned 28->32``, the phase's third task) was trimmed out of a
        thirteen-entry implement log.
        """
        out = str(tmp_path)
        append(out, "T003", "implement", "accel tuned 28->32 for <4s")
        for i in range(4, 21):
            append(out, f"T{i:03d}", "implement", "y" * 300)

        assert "accel tuned" not in render(out, phase="implement")
        assert "accel tuned" in render(
            out, phase="implement", tail=200, max_chars=20000
        )


class TestApplyCodeFixesPhaseAliasing:
    """F6: the write executor appends under ``apply-code-fixes`` (its mode/
    ``PHASE_BY_WRITE_KIND`` word) while ``instructions/apply-code-fixes.md``
    step 5.7 mandates ``--phase "apply-fixes"`` (the phase-directory word) for
    the native leg's ``util prior-changes append``. Both must be treated as one
    phase — by ``append``'s dedup, by ``render``'s same-phase preference, and
    by the CLI (``cmd_append``), which is what the native leg actually
    invokes — or a run that starts on one leg and resumes on the other
    re-appends every batch instead of recognising the duplicate, and reads a
    prior-changes context that calls its own predecessors "other phases".
    """

    def test_canonical_phase_maps_the_mode_word_to_the_dir_word(self) -> None:
        assert canonical_phase("apply-code-fixes") == "apply-fixes"
        assert canonical_phase("apply-fixes") == "apply-fixes"
        # Unrelated phases pass through unchanged.
        assert canonical_phase("implement") == "implement"
        assert canonical_phase(None) is None
        assert PHASE_ALIASES == {"apply-code-fixes": "apply-fixes"}

    def test_append_dedupes_across_both_spellings_in_process(
        self, tmp_path: Path
    ) -> None:
        """The write executor's own call: `_prior_changes.append(out, id, self.phase, ...)`."""
        out = str(tmp_path)
        # First dispatch, as the executor writes it today (self.phase ==
        # PHASE_BY_WRITE_KIND["apply_code_fixes"] == "apply-code-fixes").
        assert append(out, "index.html#abc", "apply-code-fixes", "Fix A") is True
        # A resumed batch re-dispatched under the native leg's documented
        # spelling must be recognised as the same entry, not appended again.
        assert append(out, "index.html#abc", "apply-fixes", "Fix A (retry)") is False

        stored = read_entries(out)
        assert len(stored) == 1
        # Stored under the canonical (phase-directory) spelling either way.
        assert stored[0]["phase"] == "apply-fixes"
        assert stored[0]["summary"] == "Fix A"

    def test_append_dedupes_across_both_spellings_reversed_order(
        self, tmp_path: Path
    ) -> None:
        out = str(tmp_path)
        assert append(out, "index.html#abc", "apply-fixes", "Fix A") is True
        assert append(out, "index.html#abc", "apply-code-fixes", "Fix A (retry)") is False

        stored = read_entries(out)
        assert len(stored) == 1
        assert stored[0]["phase"] == "apply-fixes"

    def test_cli_append_normalizes_the_stored_phase(self, tmp_path: Path) -> None:
        """`util prior-changes append` (the native leg's own command)."""
        out = str(tmp_path)
        result = _append(out, "index.html#abc", "apply-code-fixes", "Fix A")
        assert result.returncode == 0
        stored = read_entries(out)
        assert len(stored) == 1
        assert stored[0]["phase"] == "apply-fixes"

    def test_cli_append_and_in_process_append_dedupe_across_the_two_legs(
        self, tmp_path: Path
    ) -> None:
        """The exact round trip: executor dispatches a batch, then the same
        batch is re-dispatched (e.g. on --resume) through the native leg's
        CLI, or vice versa — neither leg should double the entry.
        """
        out = str(tmp_path)
        # Executor leg (in-process, mode word).
        assert append(out, "index.html#abc", "apply-code-fixes", "Fix A") is True
        # Native leg (CLI, phase-dir word) resumes the same batch.
        result = _append(out, "index.html#abc", "apply-fixes", "Fix A (retry)")
        assert "already exists, skipping" in result.stderr

        stored = read_entries(out)
        assert len(stored) == 1
        assert stored[0]["summary"] == "Fix A"

        # And the reverse direction: native leg first, executor resumes.
        assert append(out, "index.html#def", "apply-fixes", "Fix B") is True
        assert append(out, "index.html#def", "apply-code-fixes", "Fix B (retry)") is False
        assert len(read_entries(out)) == 2

    def test_render_prefers_same_phase_entries_under_either_spelling(
        self, tmp_path: Path
    ) -> None:
        out = str(tmp_path)
        append(out, "index.html#abc", "apply-fixes", "Restored accel 28")
        append(out, "index.html#def", "apply-code-fixes", "OOB teleport resync")
        append(out, "s1", "apply-suggestions", "unrelated phase")

        # Both spellings see the SAME phase-scoped entries — one phase, not
        # two, however the caller spells it.
        by_dir_word = render(out, phase="apply-fixes")
        by_mode_word = render(out, phase="apply-code-fixes")
        assert by_dir_word == by_mode_word
        assert "Restored accel 28" in by_dir_word
        assert "OOB teleport resync" in by_dir_word
        assert "unrelated phase" not in by_dir_word
        assert "... and 1 earlier batches from other phases" in by_dir_word
