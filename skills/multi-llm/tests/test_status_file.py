"""Tests for _write_status and _update_status checkpoint helpers."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from review_plan_orchestrator import _write_status, _update_status


class TestWriteStatus:
    """Tests for _write_status."""

    def test_creates_status_file(self, tmp_path):
        """Test that .status.json is created with correct data."""
        _write_status(str(tmp_path), {"state": "models_running", "phase": "review-plan"})
        status_path = tmp_path / ".status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text())
        assert data["state"] == "models_running"
        assert data["phase"] == "review-plan"
        assert "updated_at" in data

    def test_sets_updated_at_automatically(self, tmp_path):
        """Test that updated_at is set automatically."""
        _write_status(str(tmp_path), {"state": "test"})
        data = json.loads((tmp_path / ".status.json").read_text())
        assert "updated_at" in data

    def test_overwrites_existing_file(self, tmp_path):
        """Test that writing again overwrites the previous status."""
        _write_status(str(tmp_path), {"state": "first"})
        _write_status(str(tmp_path), {"state": "second"})
        data = json.loads((tmp_path / ".status.json").read_text())
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
        data = json.loads((tmp_path / ".status.json").read_text())
        assert data["models"] == ["a", "b"]


class TestUpdateStatus:
    """Tests for _update_status."""

    def test_creates_file_if_not_exists(self, tmp_path):
        """Test that _update_status creates .status.json if it doesn't exist."""
        _update_status(str(tmp_path), {"state": "models_complete"})
        data = json.loads((tmp_path / ".status.json").read_text())
        assert data["state"] == "models_complete"

    def test_merges_with_existing_data(self, tmp_path):
        """Test that updates are merged with existing data."""
        _write_status(str(tmp_path), {"state": "models_running", "phase": "review-plan", "models_requested": ["a", "b"]})
        _update_status(str(tmp_path), {"state": "models_complete", "models_completed": ["a", "b"]})
        data = json.loads((tmp_path / ".status.json").read_text())
        assert data["state"] == "models_complete"
        assert data["phase"] == "review-plan"  # preserved from original
        assert data["models_requested"] == ["a", "b"]  # preserved
        assert data["models_completed"] == ["a", "b"]  # added

    def test_deep_merges_output_files(self, tmp_path):
        """Test that output_files dict is deep-merged, not replaced."""
        _write_status(str(tmp_path), {"state": "running", "output_files": {"a.json": True}})
        _update_status(str(tmp_path), {"output_files": {"b.json": True}})
        data = json.loads((tmp_path / ".status.json").read_text())
        assert data["output_files"]["a.json"] is True
        assert data["output_files"]["b.json"] is True

    def test_no_crash_on_corrupted_file(self, tmp_path):
        """Test that corrupted .status.json doesn't crash — falls back to fresh write."""
        status_path = tmp_path / ".status.json"
        status_path.write_text("not valid json{{{")
        _update_status(str(tmp_path), {"state": "recovered"})
        data = json.loads(status_path.read_text())
        assert data["state"] == "recovered"

    def test_no_crash_on_invalid_dir(self):
        """Test that update to non-existent directory doesn't raise."""
        _update_status("/nonexistent/path", {"state": "test"})

    def test_preserves_started_at(self, tmp_path):
        """Test that started_at from initial write is preserved through updates."""
        _write_status(str(tmp_path), {"state": "models_running", "started_at": "2024-01-01T00:00:00"})
        _update_status(str(tmp_path), {"state": "models_complete"})
        data = json.loads((tmp_path / ".status.json").read_text())
        assert data["started_at"] == "2024-01-01T00:00:00"
