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
    finish_phase_in_state,
    format_duration,
    format_number,
    generate_all_phases_report,
    generate_phase_report,
    record_metric,
    record_metric_in_state,
    start_phase,
    start_phase_in_state,
)


def _create_state(tmp_path, extra=None):
    """Create a minimal valid state.json for testing."""
    state = {"schema_version": "1.0", "plan_path": "/test/plan.md", "task_status": {}}
    if extra:
        state.update(extra)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
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

        state = json.loads(state_file.read_text(encoding="utf-8"))
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

        state = json.loads(state_file.read_text(encoding="utf-8"))
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

        state = json.loads(state_file.read_text(encoding="utf-8"))
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

        state = json.loads(state_file.read_text(encoding="utf-8"))
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

        state = json.loads(state_file.read_text(encoding="utf-8"))

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

        state = json.loads(state_file.read_text(encoding="utf-8"))
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
        assert "| Task | Executor | Tokens | Tool Uses | Duration |" in report

        # Should contain entries
        assert "Task A" in report
        assert "Task B" in report
        # D2h: the flag keeps its name, the recorded value becomes the
        # host-neutral executor role.
        assert "agent" in report
        assert "general-purpose" not in report
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
            encoding="utf-8",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Verify state file was updated
        state = json.loads(state_file.read_text(encoding="utf-8"))
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
            encoding="utf-8",
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
            encoding="utf-8",
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
            encoding="utf-8",
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
            encoding="utf-8",
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
            encoding="utf-8",
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
            encoding="utf-8",
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


class TestTheInMemoryMutators:
    """F4: a holder of a live ``StateManager`` dict has to record metrics
    without becoming a second writer of ``state.json`` — the two race, and the
    ``StateManager``'s own wholesale ``save()`` always wins."""

    def test_recording_into_a_dict_writes_no_file(self, tmp_path):
        state_file = _create_state(tmp_path)
        state = json.loads(state_file.read_text(encoding="utf-8"))

        record_metric_in_state(state, "implement", "T001", duration_ms=1200)

        assert state["metrics"]["implement"][0]["label"] == "T001"
        # Nothing reached disk: persisting is the dict owner's job.
        assert "metrics" not in json.loads(state_file.read_text(encoding="utf-8"))

    def test_a_null_metrics_key_is_replaced_rather_than_indexed(self):
        """Real state files carry `metrics: null`, and `"metrics" not in state`
        is False for one — the old guard would have raised on it."""
        state = {"metrics": None}

        record_metric_in_state(state, "implement", "T001")

        assert state["metrics"]["implement"][0]["label"] == "T001"

    def test_the_eta_line_is_returned_and_never_printed(self, capsys):
        state = {}
        start_phase_in_state(state, "implement", 4)

        line = record_metric_in_state(
            state, "implement", "T001", duration_ms=1000,
            total_batches=4, batch_index=1,
        )

        assert line.startswith("[ETA] implement: task 1/4")
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_no_eta_without_both_halves_of_the_batch_position(self):
        state = {}
        assert record_metric_in_state(state, "implement", "T001") is None
        assert record_metric_in_state(
            state, "implement", "T002", total_batches=4
        ) is None

    def test_finishing_a_phase_that_never_started_leaves_the_dict_alone(self):
        state = {"metrics_progress": {"code-review": {"started_at": "x"}}}

        assert finish_phase_in_state(state, "implement") is None
        assert state["metrics_progress"] == {"code-review": {"started_at": "x"}}

    def test_the_closing_line_is_returned_rather_than_printed(self, capsys):
        """F7: the in-memory twin printed the closing `[ETA] … phase complete`
        line to stderr itself, so a scheduler whose every other ETA line went to
        the phase log had this one land somewhere else. It is returned now,
        beside the elapsed ms, exactly as `record_metric_in_state` returns its
        own line."""
        state = {}
        start_phase_in_state(state, "implement", 3)

        finished = finish_phase_in_state(state, "implement")
        assert finished is not None
        elapsed_ms, line = finished
        assert isinstance(elapsed_ms, int)
        assert line.startswith("[ETA] implement:") and "phase complete" in line
        # Nothing reached the streams: the caller decides where the line goes.
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        assert "metrics_progress" not in state

    def test_the_cli_leg_still_prints_the_closing_line_to_stderr(
        self, tmp_path, capsys
    ):
        """The file-level pair owns stderr; only its return value changed."""
        state_file = _create_state(tmp_path)
        start_phase(state_file, "implement", 1)

        elapsed = finish_phase(state_file, "implement")

        assert isinstance(elapsed, int)
        err = capsys.readouterr().err
        assert "[ETA] implement:" in err and "phase complete" in err

    def test_the_file_level_pair_still_round_trips_through_the_mutators(
        self, tmp_path
    ):
        """The CLI leg is now a read/mutate/write wrapper; its observable
        behaviour must not have moved."""
        state_file = _create_state(tmp_path)

        start_phase(state_file, "implement", 2)
        assert json.loads(state_file.read_text())["metrics_progress"][
            "implement"]["total_batches"] == 2

        record_metric(state_path=state_file, phase="implement", label="T001",
                      duration_ms=1000, print_eta=False)
        assert len(json.loads(state_file.read_text())["metrics"]["implement"]) == 1

        assert finish_phase(state_file, "implement") is not None
        assert "metrics_progress" not in json.loads(state_file.read_text())
        # A second finish is the documented graceful no-op.
        assert finish_phase(state_file, "implement") is None


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
        """No ``start_phase`` called: only the batch-duration signal exists,
        and the line reports its result — never a labelled intermediate."""
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
        # Last line should reference batch 3/8 and ~5m remaining
        last_lines = [l for l in err.splitlines() if l.startswith("[ETA]")]
        assert last_lines, f"Expected [ETA] line on stderr, got: {err!r}"
        last = last_lines[-1]
        assert "batch 3/8" in last
        assert "per-item" not in last
        assert "wall-clock" not in last
        # 5 remaining * 60s avg = 300s = 5m
        assert "~5m 0s remaining" in last

    def test_eta_wall_clock_only(self, tmp_path, capsys, monkeypatch):
        """``start_phase`` called, no duration_ms: only the wall-clock signal
        exists, and the line reports its result — never a labelled
        intermediate."""
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
        assert "per-item" not in last
        assert "wall-clock" not in last
        # 60s elapsed / 1 done * 3 remaining = 180s = 3m
        assert "~3m 0s remaining" in last

    def test_eta_both_signals_picks_min(self, tmp_path, capsys, monkeypatch):
        """When both signals are available, the printed ETA is
        min(per_item_ms, wall_clock_ms) — neither intermediate is printed,
        only the number a reader acts on."""
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
        # min drives the printed "remaining" segment to wall-clock, and
        # neither intermediate appears in the line.
        assert "per-item" not in last
        assert "wall-clock" not in last
        assert "~1m 30s remaining" in last

    def test_eta_complete_at_last_batch(self, tmp_path, capsys):
        """When batch_index == total_batches, the line says everything it
        counts is complete (implement counts tasks, fix round 22)."""
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
        assert "task 2/2" in last
        assert "ALL TASKS COMPLETE" in last
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
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert "implement" not in state.get("metrics_progress", {})

    def test_start_replaces_existing_progress(self, tmp_path, monkeypatch):
        """Calling ``start_phase`` twice overwrites the prior entry."""
        state_file = _create_state(tmp_path)
        clock = _ClockController(monkeypatch, datetime(2026, 1, 1, 12, 0, 0))

        start_phase(state_file, phase="implement", total_batches=4)
        first_state = json.loads(state_file.read_text(encoding="utf-8"))
        first_started = first_state["metrics_progress"]["implement"]["started_at"]
        assert first_state["metrics_progress"]["implement"]["total_batches"] == 4

        clock.advance(60)
        start_phase(state_file, phase="implement", total_batches=8)
        second_state = json.loads(state_file.read_text(encoding="utf-8"))
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
        """Zero-duration entries don't crash; the batch-rate estimate is 0."""
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
        assert "~0s remaining" in last

    def test_record_eta_resilient_to_missing_progress_key(self, tmp_path, capsys):
        """Missing ``metrics_progress`` key: wall-clock signal absent, the
        batch-duration signal alone still drives the printed ETA."""
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
        assert "~" in last and "remaining" in last
        assert "per-item" not in last
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
        assert "\033[1;32mALL TASKS COMPLETE\033[0m" in result["line"]

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
        assert result["line"].startswith("[ETA] implement: task 2/4 done in 1m 0s")

    def test_eta_single_batch_phase(self, tmp_path, capsys):
        """A 1-task implement: ``task 1/1 done in Xs · ALL TASKS COMPLETE``."""
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
        assert "task 1/1" in last
        assert "ALL TASKS COMPLETE" in last
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
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        state = json.loads(state_file.read_text(encoding="utf-8"))
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
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # ETA line should land on stderr
        assert "[ETA] implement: task 1/3" in result.stderr

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
            encoding="utf-8",
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
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert "implement" not in state.get("metrics_progress", {})


# --------------------------------------------------------------------------- #
# One event, one entry (review-run-scope-salvage-and-failure-visibility §7.1)
# --------------------------------------------------------------------------- #

PHASE = "apply-suggestions"
LABEL = "Batch 4 (step_4)"


def _record(state, label=LABEL, batch_index=4, **values):
    return record_metric_in_state(
        state, PHASE, label, subagent_type="general-purpose",
        total_batches=8, batch_index=batch_index, **values,
    )


def _entries(state, phase=PHASE):
    return state["metrics"][phase]


def _measured(entry):
    return {k: entry[k] for k in ("token_count", "tool_uses", "duration_ms")
            if k in entry}


def _phase_token_total(state):
    """The token total a phase report would print (None when nothing was
    measured): a fabricated zero would print `0` here, a free run."""
    total = None
    for entry in state["metrics"][PHASE]:
        if entry.get("token_count") is not None:
            total = (total or 0) + entry["token_count"]
    return total


class TestTheRecordingIdentity:
    """Each start stamps a run id; each entry carries it and its index."""

    def test_every_start_stamps_a_fresh_run_id(self):
        state = {}
        start_phase_in_state(state, PHASE, 8)
        first = state["metrics_progress"][PHASE]["run_id"]
        start_phase_in_state(state, PHASE, 8)
        second = state["metrics_progress"][PHASE]["run_id"]

        assert len(first) == 32 and int(first, 16) >= 0
        assert first != second

    def test_an_entry_carries_the_run_id_and_its_batch_index(self):
        state = {}
        start_phase_in_state(state, PHASE, 8)
        _record(state, batch_index=1, duration_ms=1000)

        entry = _entries(state)[0]
        assert entry["run_id"] == state["metrics_progress"][PHASE]["run_id"]
        assert entry["batch_index"] == 1

    def test_the_cli_leg_persists_the_run_id_across_calls(self, tmp_path):
        """The native leg records through separate processes, so the id has to
        live in `state.json`, and the second identical record has to collapse
        there too."""
        state_file = _create_state(tmp_path)

        def run(*args):
            return subprocess.run(
                [sys.executable, "-m", "utils.metrics", *args,
                 "--state-file", str(state_file)],
                cwd=Path(__file__).parent.parent,
                capture_output=True, text=True, encoding="utf-8",
            )

        assert run("start", "--phase", PHASE, "--total-batches", "8").returncode == 0
        record = ("record", "--phase", PHASE, "--label", LABEL,
                  "--duration-ms", "5000", "--total-batches", "8", "--batch-index", "1")
        assert run(*record).returncode == 0
        second = run(*record)

        assert second.returncode == 0
        assert "[METRICS_DUPLICATE]" in second.stderr
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert len(state["metrics"][PHASE]) == 1
        assert state["metrics"][PHASE][0]["run_id"] == \
            state["metrics_progress"][PHASE]["run_id"]


class TestASecondRecordOfTheSameEvent:
    """T7.1: collapse a second record of the same event, and nothing else."""

    @pytest.fixture
    def state(self):
        state = {}
        start_phase_in_state(state, PHASE, 8)
        for index in (1, 2, 3):
            _record(state, label=f"Batch {index}", batch_index=index)
        return state

    def test_an_identical_duplicate_is_skipped_with_a_notice(self, state, capsys):
        _record(state, token_count=1200, tool_uses=7, duration_ms=5000)
        capsys.readouterr()

        line = _record(state, token_count=1200, tool_uses=7, duration_ms=5000)

        assert line is None
        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" in err and LABEL in err
        assert "[METRICS_INDEX_MISMATCH]" not in err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        assert _measured(batch_4[0]) == {
            "token_count": 1200, "tool_uses": 7, "duration_ms": 5000}

    @pytest.mark.parametrize("empty", [{}, {"token_count": 0, "tool_uses": 0,
                                            "duration_ms": 0}],
                             ids=["no-numbers", "all-zeros"])
    def test_real_numbers_survive_an_empty_record_landing_first(
        self, state, capsys, empty
    ):
        _record(state, **empty)
        capsys.readouterr()

        line = _record(state, token_count=1200, duration_ms=5000)

        assert line is not None and line.startswith("[ETA]")
        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        # The zeros are gone, not merged in beside the real numbers.
        assert _measured(batch_4[0]) == {"token_count": 1200, "duration_ms": 5000}
        # The executor role survives the replacement, in place.
        assert batch_4[0]["subagent_type"] == "agent"
        assert _entries(state).index(batch_4[0]) == 3

    @pytest.mark.parametrize("empty", [{}, {"token_count": 0, "tool_uses": 0,
                                            "duration_ms": 0}],
                             ids=["no-numbers", "all-zeros"])
    def test_real_numbers_survive_an_empty_record_landing_second(
        self, state, capsys, empty
    ):
        _record(state, token_count=1200, duration_ms=5000)
        capsys.readouterr()

        assert _record(state, **empty) is None

        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        assert _measured(batch_4[0]) == {"token_count": 1200, "duration_ms": 5000}

    @pytest.mark.parametrize("first,second", [
        ({"token_count": 0, "tool_uses": 0, "duration_ms": 0}, {}),
        ({}, {"token_count": 0, "tool_uses": 0, "duration_ms": 0}),
    ], ids=["zeros-then-none", "none-then-zeros"])
    def test_the_incidents_zero_and_none_pair_keeps_the_one_without_numbers(
        self, state, capsys, first, second
    ):
        """`EV/state.json`: two `Batch 4 (step_4)` entries 45 ms apart, one all
        zeros and one with no numbers. Whichever lands first, one entry
        remains, and it is the one that fabricated nothing."""
        _record(state, **first)
        _record(state, **second)

        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        assert _measured(batch_4[0]) == {}
        assert _phase_token_total(state) is None

    def test_a_subset_agreeing_on_what_it_carries_is_skipped(self, state, capsys):
        _record(state, token_count=1200, tool_uses=7, duration_ms=5000)

        assert _record(state, duration_ms=5000) is None

        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert _measured(batch_4[0]) == {
            "token_count": 1200, "tool_uses": 7, "duration_ms": 5000}

    def test_a_superset_agreeing_on_every_shared_field_replaces(self, state, capsys):
        _record(state, duration_ms=5000)

        _record(state, token_count=1200, duration_ms=5000)

        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        assert _measured(batch_4[0]) == {"token_count": 1200, "duration_ms": 5000}

    def test_a_conflict_keeps_the_first_and_appends_nothing(self, state, capsys):
        _record(state, token_count=1200, duration_ms=5000)
        capsys.readouterr()

        assert _record(state, token_count=900, duration_ms=5000) is None

        err = capsys.readouterr().err
        assert "[METRICS_CONFLICT]" in err
        # Both payloads are named, so the reader can tell which one was dropped.
        assert '"token_count": 1200' in err and '"token_count": 900' in err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 1
        assert batch_4[0]["token_count"] == 1200

    def test_the_same_label_at_another_index_is_another_event(self, state, capsys):
        _record(state, duration_ms=5000)
        _record(state, batch_index=5, duration_ms=6000)

        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" not in err
        assert len([e for e in _entries(state) if e["label"] == LABEL]) == 2

    def test_a_rerun_records_the_same_label_again(self, state, capsys):
        """A rerun starts the phase again, so it is a new run with its own
        measurement, never a duplicate of the last run's."""
        _record(state, duration_ms=5000)
        start_phase_in_state(state, PHASE, 8)

        _record(state, batch_index=4, duration_ms=5000)

        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" not in err
        batch_4 = [e for e in _entries(state) if e["label"] == LABEL]
        assert len(batch_4) == 2
        assert batch_4[0]["run_id"] != batch_4[1]["run_id"]

    def test_a_resumed_execution_leaves_the_interrupted_runs_entries_alone(
        self, capsys
    ):
        state = {}
        start_phase_in_state(state, PHASE, 4)
        _record(state, label="Batch 1", batch_index=1, duration_ms=1000)
        _record(state, label="Batch 2", batch_index=2, duration_ms=2000)
        interrupted = json.loads(json.dumps(_entries(state)))

        # --resume: the phase starts again, and batch 2 is re-dispatched.
        start_phase_in_state(state, PHASE, 4)
        _record(state, label="Batch 2", batch_index=2, duration_ms=2500)
        _record(state, label="Batch 3", batch_index=3, duration_ms=3000)

        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" not in err and "[METRICS_CONFLICT]" not in err
        entries = _entries(state)
        assert entries[:2] == interrupted
        assert [e["label"] for e in entries[2:]] == ["Batch 2", "Batch 3"]
        assert [e["duration_ms"] for e in entries[2:]] == [2500, 3000]

    def test_a_legacy_entry_without_a_run_id_is_never_matched(self, capsys):
        legacy = {"label": LABEL, "timestamp": "2026-09-25T13:40:00",
                  "subagent_type": "agent", "batch_index": 4, "token_count": 0}
        state = {"metrics": {PHASE: [dict(legacy)]}}
        start_phase_in_state(state, PHASE, 8)

        _record(state, token_count=1200)

        assert "[METRICS_" not in capsys.readouterr().err
        entries = _entries(state)
        assert len(entries) == 2
        assert entries[0] == legacy

    def test_a_record_with_no_started_phase_appends_unchecked(self, capsys):
        state = {}

        _record(state, token_count=0)
        _record(state, token_count=0)
        _record(state, token_count=5, batch_index=9)

        assert "[METRICS_" not in capsys.readouterr().err
        entries = _entries(state)
        assert len(entries) == 3
        assert all("run_id" not in e for e in entries)

    def test_a_finished_phase_no_longer_collapses(self, capsys):
        """`finish` pops the progress entry, so a record after it has no run to
        belong to and appends as it always did."""
        state = {}
        start_phase_in_state(state, PHASE, 1)
        _record(state, batch_index=1, duration_ms=1000)
        finish_phase_in_state(state, PHASE)

        _record(state, batch_index=1, duration_ms=1000)

        assert "[METRICS_" not in capsys.readouterr().err
        assert len(_entries(state)) == 2


class TestTheBatchIndexCheck:
    """T7.2: `--batch-index` against this run's own high-water mark."""

    @staticmethod
    def _run(indices, state=None, start=True):
        state = {} if state is None else state
        if start:
            start_phase_in_state(state, PHASE, 10)
        for index in indices:
            _record(state, label=f"Batch {index}", batch_index=index,
                    duration_ms=1000)
        return state

    def test_consecutive_indices_are_silent(self, capsys):
        self._run([1, 2, 3, 4])

        assert "[METRICS_INDEX_MISMATCH]" not in capsys.readouterr().err

    def test_one_skipped_batch_warns_once_and_the_next_is_silent(self, capsys):
        self._run([1, 2, 4, 5, 6])

        err = capsys.readouterr().err
        assert err.count("[METRICS_INDEX_MISMATCH]") == 1
        line = next(ln for ln in err.splitlines() if "[METRICS_INDEX_MISMATCH]" in ln)
        # Both numbers are named: what was recorded and what was expected.
        assert "batch 4" in line and "next index is 3" in line

    def test_a_duplicate_submission_does_not_make_the_next_batch_warn(self, capsys):
        state = self._run([1, 2])
        _record(state, label="Batch 2", batch_index=2, duration_ms=1000)
        _record(state, label="Batch 3", batch_index=3, duration_ms=1000)

        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" in err
        assert "[METRICS_INDEX_MISMATCH]" not in err

    def test_a_rerun_does_not_warn(self, capsys):
        state = self._run([1, 2, 3])
        self._run([1, 2, 3], state=state)

        assert "[METRICS_INDEX_MISMATCH]" not in capsys.readouterr().err

    def test_a_resumed_run_whose_first_index_is_4_does_not_warn(self, capsys):
        """A resumed scheduler run's index counts the batches the resume
        skipped, so its first record legitimately starts past 1."""
        state = self._run([1, 2])
        self._run([4, 5, 6], state=state)

        assert "[METRICS_INDEX_MISMATCH]" not in capsys.readouterr().err

    @pytest.mark.parametrize("indices", [[1, 2, 2], [1, 2, 3, 1]],
                             ids=["repeated", "backward"])
    def test_a_repeated_or_backward_index_within_a_run_warns(self, capsys, indices):
        """Distinct batches (distinct labels) handed an index already used, or
        one behind the mark: the shifted-by-one labels of the audited run."""
        state = {}
        start_phase_in_state(state, PHASE, 10)
        for position, index in enumerate(indices, start=1):
            _record(state, label=f"Task {position}", batch_index=index,
                    duration_ms=1000)

        err = capsys.readouterr().err
        assert "[METRICS_DUPLICATE]" not in err
        assert err.count("[METRICS_INDEX_MISMATCH]") == 1
        # A warning, not a rejection: the record is still there.
        assert len(_entries(state)) == len(indices)

    def test_no_started_phase_is_never_checked(self, capsys):
        self._run([1, 5, 2], start=False)

        assert "[METRICS_INDEX_MISMATCH]" not in capsys.readouterr().err
