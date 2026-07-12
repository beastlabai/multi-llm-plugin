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
    FILENAME,
    MAX_CHARS,
    _jsonl_path,
    _load_entries,
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
            marks_phase_completed = False

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
