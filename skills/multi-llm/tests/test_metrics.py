"""Tests for metrics utility."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    _compute_eta,
    finish_phase,
    format_duration,
    format_number,
    generate_all_phases_report,
    generate_phase_report,
    record_metric,
    start_phase,
)


def _create_state(tmp_path, extra=None):
    """Create a minimal valid state.json for testing."""
    state = {"schema_version": "1.0", "plan_path": "/test/plan.md", "task_status": {}}
    if extra:
        state.update(extra)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state, indent=2))
    return state_file


class TestRecordMetric:
    """Tests for record_metric function."""

    def test_record_creates_metrics_key(self, tmp_path):
        """First record creates metrics.{phase} in state."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task 1",
            token_count=1000,
        )

        state = json.loads(state_file.read_text())
        assert "metrics" in state
        assert "implement" in state["metrics"]
        assert len(state["metrics"]["implement"]) == 1
        assert state["metrics"]["implement"][0]["label"] == "Task 1"
        assert state["metrics"]["implement"][0]["token_count"] == 1000

    def test_record_appends_to_existing(self, tmp_path):
        """Subsequent records append to the array."""
        state_file = _create_state(tmp_path)

        record_metric(state_path=state_file, phase="implement", label="Task 1")
        record_metric(state_path=state_file, phase="implement", label="Task 2")
        record_metric(state_path=state_file, phase="implement", label="Task 3")

        state = json.loads(state_file.read_text())
        entries = state["metrics"]["implement"]
        assert len(entries) == 3
        assert entries[0]["label"] == "Task 1"
        assert entries[1]["label"] == "Task 2"
        assert entries[2]["label"] == "Task 3"

    def test_record_with_null_values(self, tmp_path):
        """Missing metric args store without those keys (not null)."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="review-plan",
            label="Review step",
            # All optional args left as None
        )

        state = json.loads(state_file.read_text())
        entry = state["metrics"]["review-plan"][0]

        # Required fields present
        assert "label" in entry
        assert "timestamp" in entry

        # Optional fields should be ABSENT, not set to null
        assert "subagent_type" not in entry
        assert "token_count" not in entry
        assert "tool_uses" not in entry
        assert "duration_ms" not in entry

    def test_record_multiple_phases(self, tmp_path):
        """Recording to different phases creates separate arrays."""
        state_file = _create_state(tmp_path)

        record_metric(state_path=state_file, phase="implement", label="Impl task")
        record_metric(state_path=state_file, phase="code-review", label="Review task")
        record_metric(state_path=state_file, phase="apply-fixes", label="Fix task")

        state = json.loads(state_file.read_text())
        assert len(state["metrics"]) == 3
        assert len(state["metrics"]["implement"]) == 1
        assert len(state["metrics"]["code-review"]) == 1
        assert len(state["metrics"]["apply-fixes"]) == 1
        assert state["metrics"]["implement"][0]["label"] == "Impl task"
        assert state["metrics"]["code-review"][0]["label"] == "Review task"
        assert state["metrics"]["apply-fixes"][0]["label"] == "Fix task"

    def test_record_preserves_existing_state(self, tmp_path):
        """Recording metrics doesn't clobber other state.json fields."""
        state_file = _create_state(
            tmp_path,
            extra={
                "human_decisions": {"group-1": "approved"},
                "processed_items": ["item-a", "item-b"],
                "plan_hash": "abc123",
            },
        )

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task 1",
            token_count=500,
        )

        state = json.loads(state_file.read_text())

        # Original fields preserved
        assert state["schema_version"] == "1.0"
        assert state["plan_path"] == "/test/plan.md"
        assert state["task_status"] == {}
        assert state["human_decisions"] == {"group-1": "approved"}
        assert state["processed_items"] == ["item-a", "item-b"]
        assert state["plan_hash"] == "abc123"

        # Metrics also present
        assert "metrics" in state
        assert len(state["metrics"]["implement"]) == 1

    def test_record_auto_generates_timestamp(self, tmp_path):
        """Verify timestamp is auto-generated and within a few seconds of now."""
        state_file = _create_state(tmp_path)

        before = datetime.now()
        record_metric(state_path=state_file, phase="implement", label="Task 1")
        after = datetime.now()

        state = json.loads(state_file.read_text())
        entry = state["metrics"]["implement"][0]
        assert "timestamp" in entry

        # Parse the ISO timestamp
        ts = datetime.fromisoformat(entry["timestamp"])
        assert before <= ts <= after


class TestGeneratePhaseReport:
    """Tests for generate_phase_report and generate_all_phases_report."""

    def test_report_single_phase(self, tmp_path):
        """Generates correct markdown table with totals."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task A",
            subagent_type="general-purpose",
            token_count=5000,
            tool_uses=12,
            duration_ms=60000,
        )
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task B",
            subagent_type="general-purpose",
            token_count=3000,
            tool_uses=8,
            duration_ms=45000,
        )

        report = generate_phase_report(state_file, "implement")

        # Should contain markdown table structure
        assert "## Resource Usage" in report
        assert "| Task | Subagent Type | Tokens | Tool Uses | Duration |" in report

        # Should contain entries
        assert "Task A" in report
        assert "Task B" in report
        assert "general-purpose" in report
        assert "5,000" in report
        assert "3,000" in report

        # Should contain totals
        assert "**Total**" in report
        assert "**8,000**" in report  # 5000 + 3000
        assert "**20**" in report  # 12 + 8
        assert "**1m 45s**" in report  # 60000 + 45000 = 105000ms

    def test_report_empty_phase(self, tmp_path):
        """Returns empty string when no metrics exist."""
        state_file = _create_state(tmp_path)

        report = generate_phase_report(state_file, "nonexistent-phase")
        assert report == ""

    def test_report_all_phases(self, tmp_path):
        """Aggregates across phases correctly."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Impl task",
            token_count=10000,
            tool_uses=20,
            duration_ms=120000,
        )
        record_metric(
            state_path=state_file,
            phase="code-review",
            label="Review task",
            token_count=5000,
            tool_uses=10,
            duration_ms=60000,
        )

        report = generate_all_phases_report(state_file)

        # Should contain summary table
        assert "### Resource Usage" in report
        assert "| Phase | Subagent Calls | Tokens | Tool Uses | Duration |" in report

        # Should contain both phases
        assert "implement" in report
        assert "code-review" in report

        # Should contain grand totals
        assert "**Total**" in report
        assert "**15,000**" in report  # 10000 + 5000
        assert "**30**" in report  # 20 + 10
        assert "**3m 0s**" in report  # 120000 + 60000 = 180000ms
        assert "**2**" in report  # 2 total subagent calls

    def test_report_single_entry(self, tmp_path):
        """Table with 1 row; total row matches the single entry exactly."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Only task",
            subagent_type="general-purpose",
            token_count=7500,
            tool_uses=15,
            duration_ms=83000,
        )

        report = generate_phase_report(state_file, "implement")

        # Entry values
        assert "Only task" in report
        assert "7,500" in report
        assert "15" in report
        assert "1m 23s" in report

        # Total row should match the single entry
        assert "**7,500**" in report
        assert "**15**" in report
        assert "**1m 23s**" in report

    def test_report_with_mixed_null_values(self, tmp_path):
        """Some entries have tokens but no duration, others vice versa."""
        state_file = _create_state(tmp_path)

        # Entry with tokens but no duration
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Token only",
            token_count=4000,
        )
        # Entry with duration but no tokens
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Duration only",
            duration_ms=30000,
        )

        report = generate_phase_report(state_file, "implement")

        # Verify "-" renders for missing values
        lines = report.split("\n")
        data_lines = [l for l in lines if l.startswith("| ") and "---" not in l
                      and "Task |" not in l and "Total" not in l]

        # First entry should have "-" for duration
        assert len(data_lines) == 2
        # "Token only" row should contain "4,000" and have "-" in duration column
        token_line = [l for l in data_lines if "Token only" in l][0]
        assert "4,000" in token_line
        # Duration column should be "-" (rendered for None)
        # The last column before the final | is the duration
        token_cols = [c.strip() for c in token_line.split("|") if c.strip()]
        assert token_cols[-1] == "-"  # duration column is "-"

        # "Duration only" row should contain "0m 30s" and have "-" for tokens
        dur_line = [l for l in data_lines if "Duration only" in l][0]
        assert "0m 30s" in dur_line
        dur_cols = [c.strip() for c in dur_line.split("|") if c.strip()]
        assert dur_cols[2] == "-"  # tokens column is "-"

        # Totals should only sum non-null values
        total_line = [l for l in lines if "**Total**" in l][0]
        assert "**4,000**" in total_line  # only the one entry with tokens
        assert "**0m 30s**" in total_line  # only the one entry with duration

    def test_report_phases_sorted_alphabetically(self, tmp_path):
        """All-phases report lists phases in alphabetical order."""
        state_file = _create_state(tmp_path)

        # Record in non-alphabetical order
        record_metric(state_path=state_file, phase="code-review", label="R1")
        record_metric(state_path=state_file, phase="apply-fixes", label="A1")
        record_metric(state_path=state_file, phase="implement", label="I1")

        report = generate_all_phases_report(state_file)

        # Find phase positions in the report
        pos_apply = report.index("apply-fixes")
        pos_impl = report.index("implement")
        pos_review = report.index("code-review")

        # Should be alphabetical: apply-fixes < code-review < implement
        assert pos_apply < pos_review < pos_impl


class TestFormatters:
    """Tests for format_duration and format_number."""

    def test_format_duration(self):
        """Standard duration formatting: 83000 -> '1m 23s', 0 -> '0s', None -> '-'."""
        assert format_duration(83000) == "1m 23s"
        assert format_duration(0) == "0s"
        assert format_duration(None) == "-"

    def test_format_duration_edge_cases(self):
        """Edge cases for duration formatting around minute boundaries."""
        assert format_duration(59999) == "0m 59s"  # Just under 1 minute
        assert format_duration(60000) == "1m 0s"   # Exactly 1 minute
        assert format_duration(3600000) == "60m 0s" # Exactly 1 hour

    def test_format_number(self):
        """Standard number formatting: 12450 -> '12,450', 0 -> '0', None -> '-'."""
        assert format_number(12450) == "12,450"
        assert format_number(0) == "0"
        assert format_number(None) == "-"


class TestMetricsCLI:
    """Tests for metrics CLI interface."""

    def test_cli_record(self, tmp_path):
        """Subprocess invocation via record subcommand updates state file."""
        state_file = _create_state(tmp_path)

        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", str(state_file),
                "--phase", "implement",
                "--label", "CLI task",
                "--tokens", "2000",
                "--tool-uses", "5",
                "--duration-ms", "30000",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Verify state file was updated
        state = json.loads(state_file.read_text())
        assert "metrics" in state
        assert "implement" in state["metrics"]
        assert len(state["metrics"]["implement"]) == 1
        entry = state["metrics"]["implement"][0]
        assert entry["label"] == "CLI task"
        assert entry["token_count"] == 2000
        assert entry["tool_uses"] == 5
        assert entry["duration_ms"] == 30000

    def test_cli_report(self, tmp_path):
        """Subprocess invocation via report subcommand outputs markdown."""
        state_file = _create_state(tmp_path)

        # First record some metrics
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Report task",
            token_count=8000,
            tool_uses=20,
            duration_ms=90000,
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "report",
                "--state-file", str(state_file),
                "--phase", "implement",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Should output markdown report
        assert "## Resource Usage" in result.stdout
        assert "Report task" in result.stdout
        assert "8,000" in result.stdout
        assert "1m 30s" in result.stdout

    def test_cli_record_missing_required_args(self):
        """Missing --state-file, --phase, or --label exits with non-zero code."""
        # Missing --label
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", "/tmp/fake.json",
                "--phase", "implement",
                # --label is missing
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        # argparse exits with 2 for missing required args
        assert result.returncode != 0

        # Missing --phase
        result2 = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", "/tmp/fake.json",
                "--label", "test",
                # --phase is missing
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result2.returncode != 0

        # Missing --state-file
        result3 = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--phase", "implement",
                "--label", "test",
                # --state-file is missing
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result3.returncode != 0

    def test_cli_record_nonexistent_state_file(self):
        """Nonexistent state file path exits with code 1."""
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", "/tmp/nonexistent_state_12345.json",
                "--phase", "implement",
                "--label", "test",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_cli_report_nonexistent_state_file(self):
        """Nonexistent state file path exits with code 1."""
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "report",
                "--state-file", "/tmp/nonexistent_state_12345.json",
                "--phase", "implement",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr


class TestMetricsEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_report_large_dataset(self, tmp_path):
        """50+ entries in a phase; verify correct formatting and totals."""
        state_file = _create_state(tmp_path)

        expected_total_tokens = 0
        expected_total_tool_uses = 0
        expected_total_duration_ms = 0
        num_entries = 55

        for i in range(num_entries):
            tokens = 1000 + i * 100
            tools = 5 + i
            duration = 10000 + i * 1000
            expected_total_tokens += tokens
            expected_total_tool_uses += tools
            expected_total_duration_ms += duration

            record_metric(
                state_path=state_file,
                phase="implement",
                label=f"Task {i + 1}",
                token_count=tokens,
                tool_uses=tools,
                duration_ms=duration,
            )

        report = generate_phase_report(state_file, "implement")

        # Should have correct number of data rows (entries + header + separator + total + footer)
        lines = report.split("\n")
        data_lines = [
            l for l in lines
            if l.startswith("| ")
            and "---" not in l
            and "Task |" not in l  # header
            and "**Total**" not in l
        ]
        assert len(data_lines) == num_entries

        # Verify totals are correct
        assert f"**{format_number(expected_total_tokens)}**" in report
        assert f"**{format_number(expected_total_tool_uses)}**" in report
        assert f"**{format_duration(expected_total_duration_ms)}**" in report

    def test_record_special_characters_in_label(self, tmp_path):
        """Labels with pipes, markdown chars don't break table formatting."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task | with | pipes",
            token_count=1000,
        )
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Task **bold** and `code`",
            token_count=2000,
        )

        report = generate_phase_report(state_file, "implement")

        # Pipes should be escaped so they don't break the markdown table
        assert "\\|" in report

        # The report should still be valid (each data row starts and ends with |)
        lines = report.split("\n")
        data_lines = [
            l for l in lines
            if l.startswith("| ")
            and "---" not in l
            and "Task |" not in l  # header uses "Task" column name
            and "**Total**" not in l
        ]

        for line in data_lines:
            # Each line should start and end with |
            assert line.startswith("|")
            assert line.endswith("|")

        # Totals should still be correct
        assert "**3,000**" in report  # 1000 + 2000


class _ClockController:
    """Helper that makes ``datetime.now()`` return a controllable value."""

    def __init__(self, monkeypatch, start: datetime):
        self._now = start
        self._monkeypatch = monkeypatch
        self._install()

    def _install(self):
        controller = self

        class _FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: ARG003
                return controller._now

        # Patch the datetime symbol the metrics module uses
        import utils.metrics as metrics_mod
        self._monkeypatch.setattr(metrics_mod, "datetime", _FakeDateTime)

    def advance(self, seconds: float):
        from datetime import timedelta
        self._now = self._now + timedelta(seconds=seconds)

    def set(self, value: datetime):
        self._now = value


class TestEta:
    """Tests for ETA computation and stderr output in ``record_metric``."""

    def test_eta_per_item_only(self, tmp_path, capsys):
        """No ``start_phase`` called: only per-item ETA is shown."""
        state_file = _create_state(tmp_path)

        # Record 3 batches of 8, each with a 60s duration
        for i in range(1, 4):
            record_metric(
                state_path=state_file,
                phase="apply-suggestions",
                label=f"Batch {i}",
                duration_ms=60000,
                total_batches=8,
                batch_index=i,
            )

        err = capsys.readouterr().err
        # Last line should reference batch 3/8 and per-item ~5m
        last_lines = [l for l in err.splitlines() if l.startswith("[ETA]")]
        assert last_lines, f"Expected [ETA] line on stderr, got: {err!r}"
        last = last_lines[-1]
        assert "batch 3/8" in last
        assert "per-item ~" in last
        assert "wall-clock" not in last
        # 5 remaining * 60s avg = 300s = 5m
        assert "~5m 0s" in last

    def test_eta_wall_clock_only(self, tmp_path, capsys, monkeypatch):
        """``start_phase`` called, no duration_ms: only wall-clock ETA shown."""
        state_file = _create_state(tmp_path)
        clock = _ClockController(monkeypatch, datetime(2026, 1, 1, 12, 0, 0))

        start_phase(state_file, phase="review-tasks", total_batches=4)

        # Advance 60s and record batch 1 (no duration_ms)
        clock.advance(60)
        record_metric(
            state_path=state_file,
            phase="review-tasks",
            label="Batch 1",
            total_batches=4,
            batch_index=1,
        )

        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        assert "batch 1/4" in last
        assert "wall-clock ~" in last
        assert "per-item" not in last
        # 60s elapsed / 1 done * 3 remaining = 180s = 3m
        assert "~3m 0s" in last

    def test_eta_both_signals_picks_min(self, tmp_path, capsys, monkeypatch):
        """When both signals are available, ETA = min(per_item, wall_clock)."""
        state_file = _create_state(tmp_path)
        clock = _ClockController(monkeypatch, datetime(2026, 1, 1, 12, 0, 0))

        # 4 batches in parallel — start_phase + advance only 30s of wall clock
        # but each batch reports 5 minutes of subagent work.
        start_phase(state_file, phase="review-plan", total_batches=4)

        # Wall-clock advance: 30s total
        clock.advance(30)
        record_metric(
            state_path=state_file,
            phase="review-plan",
            label="Batch 1",
            duration_ms=300000,  # 5m per-item
            total_batches=4,
            batch_index=1,
        )

        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        # per-item ETA = 5m * 3 = 15m
        # wall-clock ETA = 30s / 1 * 3 = 90s = 1m 30s
        # min should drive the "remaining" segment to wall-clock.
        assert "per-item ~15m 0s" in last
        assert "wall-clock ~1m 30s" in last
        assert "~1m 30s remaining" in last

    def test_eta_complete_at_last_batch(self, tmp_path, capsys):
        """When batch_index == total_batches, ALL BATCHES COMPLETE is shown."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=10000,
            total_batches=2,
            batch_index=1,
        )
        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 2",
            duration_ms=10000,
            total_batches=2,
            batch_index=2,
        )

        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        assert "batch 2/2" in last
        assert "ALL BATCHES COMPLETE" in last
        assert "remaining" not in last

    def test_eta_suppressed_with_no_eta_flag(self, tmp_path, capsys):
        """``print_eta=False`` suppresses the ETA line."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=10000,
            total_batches=4,
            batch_index=1,
            print_eta=False,
        )

        err = capsys.readouterr().err
        assert "[ETA]" not in err

    def test_start_then_finish_returns_elapsed(self, tmp_path, monkeypatch, capsys):
        """``finish_phase`` returns the elapsed ms since ``start_phase``."""
        state_file = _create_state(tmp_path)
        clock = _ClockController(monkeypatch, datetime(2026, 1, 1, 12, 0, 0))

        start_phase(state_file, phase="implement", total_batches=3)
        clock.advance(2.5)  # 2500ms
        elapsed = finish_phase(state_file, phase="implement")
        assert elapsed is not None
        assert elapsed >= 2500
        # progress entry should be removed
        state = json.loads(state_file.read_text())
        assert "implement" not in state.get("metrics_progress", {})

    def test_start_replaces_existing_progress(self, tmp_path, monkeypatch):
        """Calling ``start_phase`` twice overwrites the prior entry."""
        state_file = _create_state(tmp_path)
        clock = _ClockController(monkeypatch, datetime(2026, 1, 1, 12, 0, 0))

        start_phase(state_file, phase="implement", total_batches=4)
        first_state = json.loads(state_file.read_text())
        first_started = first_state["metrics_progress"]["implement"]["started_at"]
        assert first_state["metrics_progress"]["implement"]["total_batches"] == 4

        clock.advance(60)
        start_phase(state_file, phase="implement", total_batches=8)
        second_state = json.loads(state_file.read_text())
        second_started = second_state["metrics_progress"]["implement"]["started_at"]
        assert second_state["metrics_progress"]["implement"]["total_batches"] == 8
        assert second_started != first_started
        # Single entry, not appended
        assert isinstance(second_state["metrics_progress"]["implement"], dict)

    def test_finish_when_never_started(self, tmp_path):
        """``finish_phase`` returns None if the phase was never started."""
        state_file = _create_state(tmp_path)
        result = finish_phase(state_file, phase="never-started")
        assert result is None

    def test_record_with_eta_flags_only_one_provided(self, tmp_path, capsys):
        """Both ``--total-batches`` and ``--batch-index`` are required for ETA."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=10000,
            total_batches=4,  # batch_index missing
        )
        err = capsys.readouterr().err
        assert "[ETA]" not in err

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=10000,
            batch_index=1,  # total_batches missing
        )
        err = capsys.readouterr().err
        assert "[ETA]" not in err

    def test_record_eta_zero_durations(self, tmp_path, capsys):
        """Zero-duration entries don't crash; per-item ETA = 0."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=0,
            total_batches=2,
            batch_index=1,
        )
        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        assert "[ETA]" in last
        # 0 duration averages to 0 -> ~0s remaining
        assert "per-item ~0s" in last

    def test_record_eta_resilient_to_missing_progress_key(self, tmp_path, capsys):
        """Missing ``metrics_progress`` key: wall-clock omitted, per-item shown."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Batch 1",
            duration_ms=30000,
            total_batches=3,
            batch_index=1,
        )

        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        assert "per-item ~" in last
        assert "wall-clock" not in last

    def test_eta_colorizes_remaining_when_enabled(self):
        """When ``colorize=True``, remaining segment is wrapped in ANSI codes."""
        entries = [{"label": "Batch 1", "duration_ms": 60000}]
        result = _compute_eta(
            entries=entries,
            progress=None,
            phase="implement",
            batch_index=1,
            total_batches=4,
            this_duration_ms=60000,
            colorize=True,
        )
        assert "\033[1;36m~" in result["line"]
        assert "remaining\033[0m" in result["line"]

    def test_eta_colorizes_complete_when_enabled(self):
        """When ``colorize=True`` and remaining=0, the complete segment is green."""
        entries = [{"label": "Batch 1", "duration_ms": 60000}]
        result = _compute_eta(
            entries=entries,
            progress=None,
            phase="implement",
            batch_index=2,
            total_batches=2,
            this_duration_ms=60000,
            colorize=True,
        )
        assert "\033[1;32mALL BATCHES COMPLETE\033[0m" in result["line"]

    def test_eta_no_color_when_disabled(self):
        """When ``colorize=False`` (default), no ANSI codes are emitted."""
        entries = [{"label": "Batch 1", "duration_ms": 60000}]
        result = _compute_eta(
            entries=entries,
            progress=None,
            phase="implement",
            batch_index=1,
            total_batches=4,
            this_duration_ms=60000,
        )
        assert "\033[" not in result["line"]

    def test_compute_eta_helper_returns_dict(self):
        """``_compute_eta`` returns a dict with the expected keys."""
        entries = [
            {"label": "Batch 1", "duration_ms": 60000},
            {"label": "Batch 2", "duration_ms": 60000},
        ]
        result = _compute_eta(
            entries=entries,
            progress=None,
            phase="implement",
            batch_index=2,
            total_batches=4,
            this_duration_ms=60000,
        )
        assert set(result.keys()) >= {
            "per_item_ms", "wall_clock_ms", "eta_ms",
            "elapsed_ms", "done", "total", "remaining", "line",
        }
        # 60s avg * 2 remaining = 120000ms
        assert result["per_item_ms"] == 120000
        assert result["wall_clock_ms"] is None
        assert result["eta_ms"] == 120000
        assert result["remaining"] == 2
        assert result["line"].startswith("[ETA] implement: batch 2/4 done in 1m 0s")

    def test_eta_single_batch_phase(self, tmp_path, capsys):
        """A 1-batch phase: ``batch 1/1 done in Xs · ALL BATCHES COMPLETE``."""
        state_file = _create_state(tmp_path)

        record_metric(
            state_path=state_file,
            phase="implement",
            label="Only batch",
            duration_ms=5000,
            total_batches=1,
            batch_index=1,
        )

        err = capsys.readouterr().err
        last = [l for l in err.splitlines() if l.startswith("[ETA]")][-1]
        assert "batch 1/1" in last
        assert "ALL BATCHES COMPLETE" in last
        assert "remaining" not in last

    def test_cli_start_finish_record(self, tmp_path):
        """End-to-end CLI: start → record → finish writes/reads state correctly."""
        state_file = _create_state(tmp_path)

        # start
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "start",
                "--state-file", str(state_file),
                "--phase", "implement",
                "--total-batches", "3",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        state = json.loads(state_file.read_text())
        assert state["metrics_progress"]["implement"]["total_batches"] == 3

        # record with ETA flags
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", str(state_file),
                "--phase", "implement",
                "--label", "Batch 1",
                "--duration-ms", "10000",
                "--total-batches", "3",
                "--batch-index", "1",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # ETA line should land on stderr
        assert "[ETA] implement: batch 1/3" in result.stderr

        # record with --no-eta suppresses the ETA line
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "record",
                "--state-file", str(state_file),
                "--phase", "implement",
                "--label", "Batch 2",
                "--duration-ms", "10000",
                "--total-batches", "3",
                "--batch-index", "2",
                "--no-eta",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "[ETA]" not in result.stderr

        # finish clears progress
        result = subprocess.run(
            [
                sys.executable, "-m", "utils.metrics", "finish",
                "--state-file", str(state_file),
                "--phase", "implement",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        state = json.loads(state_file.read_text())
        assert "implement" not in state.get("metrics_progress", {})
