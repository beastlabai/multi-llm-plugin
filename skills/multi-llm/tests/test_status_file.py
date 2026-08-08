"""Tests for _write_status and _update_status checkpoint helpers.

Plus the detach consumer sweep: ``{phase}/.status.json`` is not a free path. It is
today's orchestrator checkpoint, and three live consumers read specific keys out
of it — ``ask_orchestrator``'s slug-collision guard reads ``question``
byte-for-byte, its stale-answers warning reads ``plan_hash``, and
``build_unsanitize_map`` reads ``models_requested``/``models_completed`` to restore
colon-bearing model specs during reaggregation. A heartbeat with a different shape
written to that file would corrupt all three, so the detach fields are **additive**
(namespaced under one reserved ``run`` object) and every one of those consumers is
asserted to survive interleaved heartbeats here.
"""

import glob
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from review_plan_orchestrator import _write_status, _update_status
from utils import status_file
from utils.json_extractor import build_unsanitize_map


class TestWriteStatus:
    """Tests for _write_status."""

    def test_creates_status_file(self, tmp_path):
        """Test that .status.json is created with correct data."""
        _write_status(str(tmp_path), {"state": "models_running", "phase": "review-plan"})
        status_path = tmp_path / ".status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["state"] == "models_running"
        assert data["phase"] == "review-plan"
        assert "updated_at" in data

    def test_sets_updated_at_automatically(self, tmp_path):
        """Test that updated_at is set automatically."""
        _write_status(str(tmp_path), {"state": "test"})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert "updated_at" in data

    def test_overwrites_existing_file(self, tmp_path):
        """Test that writing again overwrites the previous status."""
        _write_status(str(tmp_path), {"state": "first"})
        _write_status(str(tmp_path), {"state": "second"})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["state"] == "second"

    def test_no_crash_on_invalid_dir(self):
        """Test that writing to an invalid directory doesn't raise."""
        # Should not raise even with a non-existent directory
        _write_status("/nonexistent/path/that/does/not/exist", {"state": "test"})

    def test_no_temp_files_left_on_success(self, tmp_path):
        """Test that no temp files are left after successful write."""
        _write_status(str(tmp_path), {"state": "test"})
        temp_files = list(tmp_path.glob(".status_*.tmp"))
        assert len(temp_files) == 0

    def test_atomic_write_valid_json(self, tmp_path):
        """Test the file contains valid JSON after write."""
        _write_status(str(tmp_path), {"state": "test", "models": ["a", "b"]})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["models"] == ["a", "b"]


class TestUpdateStatus:
    """Tests for _update_status."""

    def test_creates_file_if_not_exists(self, tmp_path):
        """Test that _update_status creates .status.json if it doesn't exist."""
        _update_status(str(tmp_path), {"state": "models_complete"})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["state"] == "models_complete"

    def test_merges_with_existing_data(self, tmp_path):
        """Test that updates are merged with existing data."""
        _write_status(str(tmp_path), {"state": "models_running", "phase": "review-plan", "models_requested": ["a", "b"]})
        _update_status(str(tmp_path), {"state": "models_complete", "models_completed": ["a", "b"]})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["state"] == "models_complete"
        assert data["phase"] == "review-plan"  # preserved from original
        assert data["models_requested"] == ["a", "b"]  # preserved
        assert data["models_completed"] == ["a", "b"]  # added

    def test_deep_merges_output_files(self, tmp_path):
        """Test that output_files dict is deep-merged, not replaced."""
        _write_status(str(tmp_path), {"state": "running", "output_files": {"a.json": True}})
        _update_status(str(tmp_path), {"output_files": {"b.json": True}})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["output_files"]["a.json"] is True
        assert data["output_files"]["b.json"] is True

    def test_no_crash_on_corrupted_file(self, tmp_path):
        """Test that corrupted .status.json doesn't crash — falls back to fresh write."""
        status_path = tmp_path / ".status.json"
        status_path.write_text("not valid json{{{", encoding="utf-8")
        _update_status(str(tmp_path), {"state": "recovered"})
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["state"] == "recovered"

    def test_no_crash_on_invalid_dir(self):
        """Test that update to non-existent directory doesn't raise."""
        _update_status("/nonexistent/path", {"state": "test"})

    def test_preserves_started_at(self, tmp_path):
        """Test that started_at from initial write is preserved through updates."""
        _write_status(str(tmp_path), {"state": "models_running", "started_at": "2024-01-01T00:00:00"})
        _update_status(str(tmp_path), {"state": "models_complete"})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["started_at"] == "2024-01-01T00:00:00"


# ===========================================================================
# The detach fields are additive (D4a)
# ===========================================================================

def _checkpoint(phase_dir: Path, **extra) -> dict:
    """A realistic pre-detach checkpoint, with the three load-bearing keys."""
    data = {
        "phase": "ask",
        "state": "models_running",
        "started_at": "2026-01-01T00:00:00",
        "models_requested": ["cursor-agent:auto", "gemini:gemini-2.5-flash"],
        "models_completed": [],
        "models_failed": {},
        "error": None,
        "question": "Is the rollback strategy sufficient — really?",
        "plan_hash": "deadbeefcafe",
    }
    data.update(extra)
    _write_status(str(phase_dir), data)
    return data


def _run_record(run_id: str = "run-a", state: str = "running") -> dict:
    return {
        "run_id": run_id,
        "pid": os.getpid(),
        "process_start_time": "12345",
        "detached": True,
        "host": "codex",
        "state": state,
        "exit_code": None,
        "error": None,
        "heartbeat_at": status_file.now_iso(),
        "heartbeat_interval_s": 10.0,
        "resolutions": [],
    }


class TestDetachFieldsAreAdditive:
    """A heartbeat must not cost the checkpoint any of its keys."""

    def test_heartbeat_round_trips_without_dropping_checkpoint_keys(self, tmp_path):
        original = _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        for _ in range(5):
            assert status_file.update_run(
                str(tmp_path), {"heartbeat_at": status_file.now_iso()},
                expect_run_id="run-a",
            )
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["question"] == original["question"]
        assert data["plan_hash"] == original["plan_hash"]
        assert data["models_requested"] == original["models_requested"]
        assert data["state"] == "models_running", "the phase machine is untouched"
        assert data["run"]["state"] == "running", "the process machine is separate"
        assert data["run"]["run_id"] == "run-a"

    def test_run_state_is_not_state(self, tmp_path):
        """Two state machines on different axes; a crashed fan-out is both."""
        _checkpoint(tmp_path, state="models_running")
        status_file.claim_run(str(tmp_path), _run_record())
        status_file.update_run(
            str(tmp_path), {"state": "failed", "exit_code": 1, "error": "boom"},
            expect_run_id="run-a",
        )
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["state"] == "models_running"
        assert data["run"]["state"] == "failed"

    def test_a_phase_checkpoint_write_does_not_erase_the_run(self, tmp_path):
        """`write_status` replaces the document — except the reserved objects.

        The fan-out phase writes `state: models_running` from inside the very run
        that is heartbeating, so an unguarded whole-document write would erase the
        run record milliseconds after the worker published it.
        """
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        status_file.update_progress(str(tmp_path), {"step": "fanout"})
        _write_status(str(tmp_path), {"phase": "ask", "state": "models_running"})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["run"]["run_id"] == "run-a"
        assert data["progress"]["step"] == "fanout"

    def test_run_and_progress_merge_key_by_key(self, tmp_path):
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        status_file.update_progress(str(tmp_path), {"step": "fanout",
                                                   "models_total": 3})
        status_file.update_progress(str(tmp_path), {"models_done": 2})
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["progress"] == {"step": "fanout", "models_total": 3,
                                   "models_done": 2}
        assert data["run"]["heartbeat_interval_s"] == 10.0

    def test_claim_replaces_a_predecessor_run_wholesale(self, tmp_path):
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record("run-old", "done"))
        status_file.update_run(str(tmp_path), {"exit_code": 0},
                               expect_run_id="run-old")
        status_file.claim_run(str(tmp_path), _run_record("run-new", "starting"))
        data = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert data["run"]["run_id"] == "run-new"
        assert data["run"]["state"] == "starting"
        assert data["run"]["exit_code"] is None, (
            "the new run must not inherit its predecessor's exit code"
        )
        assert data["question"], "the checkpoint keys still survive a takeover"

    def test_resolutions_are_appended_for_the_audit_trail(self, tmp_path):
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        status_file.append_resolution(
            str(tmp_path), run_id="run-a", site="D6-row-3",
            decision="auto-resumed 1 missing model",
        )
        status_file.append_resolution(
            str(tmp_path), run_id="run-a", site="D6-row-4",
            decision="ran consolidation once",
        )
        run = status_file.read_run(str(tmp_path))
        assert [entry["site"] for entry in run["resolutions"]] == [
            "D6-row-3", "D6-row-4"
        ]
        assert all(entry["at"] for entry in run["resolutions"])

    def test_status_file_stays_invisible_to_reaggregation(self, tmp_path):
        """Dot-prefixed on purpose: `glob` skips it, so it is never a model result."""
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        assert glob.glob(str(tmp_path / "*.json")) == []


class TestConsumersSurviveInterleavedHeartbeats:
    """The three readers named in D4a, each driven over a heartbeating file."""

    def test_ask_resume_still_matches_the_stored_question(self, tmp_path):
        from ask_orchestrator import resolve_question_dir
        from utils.output_handler import slugify_question

        question = "Is the rollback strategy sufficient — really?"
        slug = slugify_question(question)
        question_dir = tmp_path / slug
        question_dir.mkdir()
        _checkpoint(question_dir, question=question)
        status_file.claim_run(str(question_dir), _run_record())
        for _ in range(3):
            status_file.update_run(
                str(question_dir), {"heartbeat_at": status_file.now_iso()},
                expect_run_id="run-a",
            )

        resolved, is_resume = resolve_question_dir(str(tmp_path), slug, question)
        assert Path(resolved) == question_dir
        assert is_resume is True, (
            "a heartbeat that dropped `question` would route this run to a "
            "<slug>-2 sibling instead of resuming"
        )
        # And a genuinely different question still gets its own dir.
        other, other_resume = resolve_question_dir(
            str(tmp_path), slug, "a completely different question"
        )
        assert other.endswith("-2") and other_resume is False

    def test_unsanitize_map_still_restores_colon_bearing_specs(self, tmp_path):
        _checkpoint(
            tmp_path,
            models_requested=["cursor-agent:auto", "gemini:gemini-2.5-flash"],
            models_completed=["cursor-agent:auto"],
        )
        status_file.claim_run(str(tmp_path), _run_record())
        status_file.update_run(
            str(tmp_path), {"heartbeat_at": status_file.now_iso()},
            expect_run_id="run-a",
        )
        mapping = build_unsanitize_map(str(tmp_path))
        assert mapping["cursor-agent_auto"] == "cursor-agent:auto"
        assert mapping["gemini_gemini-2_5-flash"] == "gemini:gemini-2.5-flash"

    def test_reaggregation_still_resolves_exactly_the_model_results(self, tmp_path):
        """A live log, two rotated logs and a heartbeating checkpoint are invisible."""
        from utils import detach

        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record())
        (tmp_path / "cursor-agent_auto.json").write_text("[]", encoding="utf-8")
        (tmp_path / "gemini_gemini-2_5-flash.json").write_text("[]", encoding="utf-8")
        (tmp_path / "grouped.json").write_text("[]", encoding="utf-8")
        (tmp_path / "report_data.json").write_text("{}", encoding="utf-8")
        # Executor bookkeeping that lands in a REVIEW phase dir: the batch report
        # (D3i) and `--dry-run-subagents`'s preflight (D3l). Both are objects, so a
        # missing exclude pattern crashes reaggregate rather than degrading quietly.
        (tmp_path / "subagent_batch_report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "subagent_dry_run.json").write_text("{}", encoding="utf-8")
        # The write scheduler's two, which normally live in an apply/implement dir.
        (tmp_path / "write_batch_report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "write_tasks.json").write_text("{}", encoding="utf-8")
        # `.jsonl` never matches the `*.json` glob, so the audit log and the
        # prior-changes log are invisible by construction — pinned, not assumed.
        (tmp_path / "subagent_audit.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "prior_changes.jsonl").write_text("{}\n", encoding="utf-8")
        Path(detach.log_path(str(tmp_path))).write_text("live\n", encoding="utf-8")
        for run_id in ("run-old-1", "run-old-2"):
            Path(detach.rotated_log_path(str(tmp_path), run_id)).write_text(
                "rotated\n", encoding="utf-8"
            )
        status_file.update_run(
            str(tmp_path), {"heartbeat_at": status_file.now_iso()},
            expect_run_id="run-a",
        )

        # The exact collection reaggregate_from_existing_files performs — read from
        # the module rather than restated, so a new phase-dir artifact that forgets
        # its exclude pattern fails HERE instead of crashing a user's reaggregate.
        exclude = _reaggregation_exclude_patterns()
        result_files = sorted(
            os.path.basename(f)
            for f in glob.glob(str(tmp_path / "*.json"))
            if not any(os.path.basename(f).startswith(ex)
                       or ex in os.path.basename(f) for ex in exclude)
        )
        assert result_files == ["cursor-agent_auto.json",
                               "gemini_gemini-2_5-flash.json"]


def _reaggregation_exclude_patterns():
    """The live ``exclude_patterns`` list from ``reaggregate_from_existing_files``.

    Parsed out of the source rather than restated: the pitfall this guards is a new
    non-model JSON file landing in a phase dir without being added to the list, and
    a test carrying its own copy of the list cannot detect that.
    """
    import ast
    import inspect

    from utils import review_orchestrator_base

    source = inspect.getsource(
        review_orchestrator_base.reaggregate_from_existing_files
    )
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "exclude_patterns"
                        for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("exclude_patterns not found in the reaggregation helper")


class TestRunCompareAndSet:
    """Every run write after the claim is compare-and-set on ``run_id`` (D4b)."""

    def test_a_displaced_run_cannot_resurrect_itself(self, tmp_path):
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record("run-old"))
        status_file.claim_run(str(tmp_path), _run_record("run-new"))
        assert not status_file.update_run(
            str(tmp_path), {"heartbeat_at": status_file.now_iso()},
            expect_run_id="run-old", quiet=True,
        )
        assert status_file.read_run(str(tmp_path))["run_id"] == "run-new"

    def test_a_closed_out_run_stays_closed(self, tmp_path):
        """A worker paused past its heartbeat window may not overwrite the record."""
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record("run-a"))
        assert status_file.update_run(
            str(tmp_path), {"state": "failed", "exit_code": None,
                            "error": "heartbeat_timeout"},
            expect_run_id="run-a",
        )
        # The worker wakes up and tries to carry on, then to finish cleanly.
        assert not status_file.update_run(
            str(tmp_path), {"state": "running"}, expect_run_id="run-a", quiet=True,
        )
        assert not status_file.update_run(
            str(tmp_path), {"state": "done", "exit_code": 0},
            expect_run_id="run-a", quiet=True,
        )
        run = status_file.read_run(str(tmp_path))
        assert run["state"] == "failed" and run["error"] == "heartbeat_timeout"

    @pytest.mark.parametrize(
        "start,attempt,allowed",
        [
            (None, "starting", True),
            (None, "running", True),
            ("starting", "running", True),
            ("running", "done", True),
            ("running", "failed", True),
            ("running", "aborted", True),
            ("running", "starting", False),
            ("done", "running", False),
            ("done", "failed", False),
            ("aborted", "done", False),
            ("failed", "done", False),
        ],
    )
    def test_only_documented_transitions_are_merged(self, tmp_path, start,
                                                    attempt, allowed):
        _checkpoint(tmp_path)
        if start is not None:
            status_file.claim_run(str(tmp_path), _run_record("run-a", start))
        else:
            status_file.claim_run(str(tmp_path), {"run_id": "run-a"})
        assert status_file.update_run(
            str(tmp_path), {"state": attempt}, expect_run_id="run-a", quiet=True,
        ) is allowed
        expected = attempt if allowed else start
        assert status_file.read_run(str(tmp_path)).get("state") == expected

    def test_heartbeat_writes_need_no_state_and_never_transition(self, tmp_path):
        _checkpoint(tmp_path)
        status_file.claim_run(str(tmp_path), _run_record("run-a", "done"))
        assert status_file.update_run(
            str(tmp_path), {"heartbeat_at": status_file.now_iso()},
            expect_run_id="run-a",
        )
        assert status_file.read_run(str(tmp_path))["state"] == "done"

    def test_is_terminal_reads_the_documented_vocabulary(self):
        assert status_file.TERMINAL_RUN_STATES == ("done", "failed", "aborted")
        assert status_file.is_terminal({"state": "done"})
        assert not status_file.is_terminal({"state": "running"})
        assert not status_file.is_terminal({})
        assert not status_file.is_terminal(None)
