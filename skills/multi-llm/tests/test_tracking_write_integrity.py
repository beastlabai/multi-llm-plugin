"""Tracking preserves real paths and the previous state on write failure."""

import json
import os
import subprocess

import finalize_tracking
from utils.git_utils import get_modified_files, get_staged_files


def test_staged_unicode_and_escaped_paths_are_not_duplicated(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    names = ["café.py"]
    if os.name != "nt":
        names.append('quote"name.py')
    for name in names:
        (tmp_path / name).write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", *names], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    assert set(get_staged_files()) == set(names)
    assert set(get_modified_files()) == set(names)
    assert finalize_tracking.get_all_modified_files() == set(names)


def test_hash_restamp_failure_preserves_entire_state(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    plan = tmp_path / "plan.md"
    original = json.dumps({"plan_hash": "old", "tracked_files": ["a.py"]})
    state_path.write_text(original)
    plan.write_text("# New plan")

    def fail_replace(*_):
        raise OSError("disk failure")

    monkeypatch.setattr(finalize_tracking.os, "replace", fail_replace)
    finalize_tracking._restamp_plan_hash(state_path, plan)
    assert state_path.read_text() == original
    assert list(tmp_path.glob(".tracking_*.tmp")) == []
