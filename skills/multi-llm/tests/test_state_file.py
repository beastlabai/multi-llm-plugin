"""``utils/state_file.py``: ``state.json``'s lock, readers, one writer and contract."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import typing
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from utils import metrics  # noqa: E402
from utils import state_file  # noqa: E402
from utils import state_manager as state_manager_mod  # noqa: E402
from utils.schema_validator import validate_state_file  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

_JSON_TYPES = {str: "string", bool: "boolean", dict: "object", list: "array"}

#: What the strict reader refuses: no JSON, not UTF-8, another top-level value.
_NOT_AN_OBJECT = {
    "not-json": b"{not json",
    "not-utf8": b"\xff\xfe{}",
    "a-list": b"[]",
    "a-string": b'"text"',
}


def _schema() -> dict:
    return json.loads((SKILL_ROOT / "schemas" / "state_file.schema.json").read_text(encoding="utf-8"))


def _json_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    return {str: "string", dict: "object", list: "array", float: "number",
            type(None): "null"}[type(value)]


def _schema_for(schema: dict, key: str) -> dict:
    if key in schema["properties"]:
        return schema["properties"][key]
    matched = [spec for pattern, spec in schema["patternProperties"].items()
               if re.match(pattern, key)]
    assert matched, f"{key}: neither a property nor a per-phase family"
    return matched[0]


class TestTheContract:
    """``StateDoc`` and the schema list the same keys with the same types."""

    def test_every_key_is_a_schema_property_and_back(self):
        assert set(state_file.StateDoc.__annotations__) == set(_schema()["properties"])

    def test_the_types_agree(self):
        properties = _schema()["properties"]
        hints = typing.get_type_hints(state_file.StateDoc)
        for key, hint in hints.items():
            origin = typing.get_origin(hint) or hint
            assert properties[key].get("type") == _JSON_TYPES[origin], key

    def test_the_per_phase_families_are_the_pattern_properties(self):
        patterns = set(_schema()["patternProperties"])
        assert patterns == {f"^{prefix}" for prefix in state_file.STATE_KEY_PREFIXES}


def _written_by_the_state_manager(tmp_path: Path) -> dict:
    """The ``state.json`` a ``StateManager`` saves after its writers ran: its
    methods and the ``*_in_state`` writers its callers apply to it."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    manager = StateManager(plan)
    manager.record_commit_range({"base": None, "head": "0" * 40}, refresh_plan_hash=True)
    manager.record_task_titles(
        {"T001": "Code", "T002": "Manual", "T003": "After"}, human_ids=["T002"],
        depends_on={"T001": [], "T002": ["T001"], "T003": ["T002"]})
    manager.update_task_status("T001", "completed")
    manager.update_task_status("T002", "skipped", reason="deferred — manual task")
    manager.track_file("src/a.py", "created", "T001")
    manager.mark_phase_started("implement")
    manager.mark_phase_completed("review-plan")
    manager.mark_phase_skipped("apply-suggestions", "User chose to skip")
    manager.save_human_task_strategy("skip-continue")
    manager.record_human_task_answers({"T002": "skip"})
    manager.record_dependency_override("T003", ["T002"])
    manager.record_human_decision("code-review", "g1", "approved", reason="ok")
    manager.record_judge_verdicts("code-review", {"g2": {"verdict": "approve"}}, unresolved=["g3"])
    manager.record_processing_progress("apply-fixes", 2, 1, 1, 2)
    manager.mark_item_processed("apply-fixes", "g1", "applied", {"task_id": "apply-fix-batch-0"})
    manager.start_metrics_phase("implement", 3)
    manager.record_metric("implement", "T001", duration_ms=1000, token_count=10, tool_uses=2)
    manager.set_metrics_left_out("implement", {"T002": "manual"})
    manager.set_metrics_set_aside("implement", {"T003": "waits on T002"})
    manager.set_metrics_total("implement", 3)
    manager.refresh_plan_hash()
    state = manager.state
    state_manager_mod.set_task_check_findings_in_state(state, "T001", ["a finding"])
    state_manager_mod.set_task_batch_check_in_state(state, "T001", ["check failed"])
    state_manager_mod.set_task_own_text_in_state(state, "T001", "own text")
    state_manager_mod.set_task_own_check_note_in_state(state, "T001", "a note")
    state_manager_mod.set_task_own_check_ran_in_state(state, "T001", True)
    state_manager_mod.set_task_batch_check_accepted_in_state(state, "T001", {"C1": ["accepted"]})
    state_manager_mod.set_task_batch_check_repeats_in_state(state, "T001", {"T004": ["repeat"]})
    state_manager_mod.set_task_check_tree_in_state(state, "T001", "f" * 40)
    state_manager_mod.mark_tasks_reset_in_state(state, ["T001"])
    manager.save()
    return json.loads(manager.state_file.read_text(encoding="utf-8"))


class TestTheWrittenFileKeepsTheContract:
    """The declarations checked against what the writers actually save, not
    only against each other."""

    @pytest.fixture(scope="class")
    def written(self, tmp_path_factory):
        return _written_by_the_state_manager(tmp_path_factory.mktemp("written"))

    def test_every_written_key_is_declared(self, written):
        declared = set(state_file.StateDoc.__annotations__)
        undeclared = sorted(
            key for key in written
            if key not in declared and not key.startswith(state_file.STATE_KEY_PREFIXES)
        )
        assert undeclared == []

    def test_every_written_value_has_the_schema_type(self, written):
        schema = _schema()
        for key, value in written.items():
            assert _json_type(value) == _schema_for(schema, key).get("type"), key

    def test_the_manual_task_ids_are_part_of_it(self, written):
        assert written["implement_human_task_ids"] == ["T002"]

    def test_the_saved_file_validates(self, written):
        assert validate_state_file(written) == (True, [])


class TestReaders:
    @pytest.mark.parametrize("content", [None, "{not json", "[1, 2]", b"\xff\xfe{}"])
    def test_read_document_is_none_for_anything_but_an_object(self, tmp_path, content):
        path = tmp_path / "state.json"
        if isinstance(content, bytes):
            path.write_bytes(content)
        elif content is not None:
            path.write_text(content, encoding="utf-8")
        assert state_file.read_document(path) is None

    def test_read_document_returns_the_object(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text('{"a": 1}', encoding="utf-8")
        assert state_file.read_document(path) == {"a": 1}

    def test_the_strict_reader_raises(self, tmp_path):
        path = tmp_path / "state.json"
        with pytest.raises(FileNotFoundError):
            state_file.read_document_strict(path)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            state_file.read_document_strict(path)
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="not an object"):
            state_file.read_document_strict(path)

    @pytest.mark.parametrize("content", sorted(_NOT_AN_OBJECT))
    def test_a_file_with_no_object_is_a_state_document_error(self, tmp_path, content):
        path = tmp_path / "state.json"
        path.write_bytes(_NOT_AN_OBJECT[content])
        with pytest.raises(state_file.StateDocumentError) as raised:
            state_file.read_document_strict(path)
        assert isinstance(raised.value, ValueError)
        assert raised.value.path == str(path)
        assert str(raised.value) == f"{path}: {raised.value.detail}"
        assert str(path) not in raised.value.detail


def _implement_phase(tmp_path: Path, state_bytes: bytes):
    """A materialized implement phase whose ``state.json`` holds *state_bytes*."""
    from utils.write_phase_bridge import materialize_write_tasks, write_task_file_for_phase

    plan = tmp_path / "plans" / "feature.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feature\n", encoding="utf-8")
    out = tmp_path / "plans" / "feature"
    phase_dir = out / "implement"
    phase_dir.mkdir(parents=True)
    batches = [{"batch_index": 0, "can_parallelize": False, "tasks": [
        {"id": "T001", "title": "Foundation", "prompt": "# Implement\n\n1. Do it.\n",
         "files_to_create": ["src/a.py"], "depends_on": [], "executor_role": "agent"},
    ]}]
    tasks = materialize_write_tasks(
        kind="implement", batches=batches, phase_dir=str(phase_dir), plan_file=str(plan))
    write_task_file_for_phase(
        kind="implement", tasks=tasks, phase_dir=str(phase_dir), plan_file=str(plan))
    state_path = out / "state.json"
    state_path.write_bytes(state_bytes)
    return state_path, phase_dir


class TestEveryStrictReaderCallerRefusesAsBefore:
    """A ``state.json`` holding no object reaches each caller of the strict
    reader as that caller's own handled refusal, never a traceback, and
    nothing is written. The callers: ``util batch-done`` (exit 3), ``util
    metrics`` (exit 1), ``commit_apply.save_guard`` (``ValueError``);
    ``util write-prompt``'s re-check is in ``test_commit_write_policy.py``."""

    @pytest.mark.parametrize("content", sorted(_NOT_AN_OBJECT))
    @pytest.mark.parametrize("status", ["completed", "failed"])
    def test_batch_done_is_exit_3_not_a_json_object(self, tmp_path, capsys, content, status):
        from utils import batch_done

        state_path, phase_dir = _implement_phase(tmp_path, _NOT_AN_OBJECT[content])
        argv = ["--phase-dir", str(phase_dir), "--task-id", "T001", "--status", status]
        if status == "completed":
            argv += ["--own-check", "passed"]
        else:
            argv += ["--reason", "the subagent failed"]
        assert batch_done.main(argv) == 3
        err = capsys.readouterr().err
        assert f"ERROR: {state_path} is not a JSON object (" in err
        assert "Traceback" not in err
        assert state_path.read_bytes() == _NOT_AN_OBJECT[content]

    @pytest.mark.parametrize("content", sorted(_NOT_AN_OBJECT))
    @pytest.mark.parametrize("command", [
        ["record", "--phase", "implement", "--label", "T001"],
        ["start", "--phase", "implement", "--total-batches", "2"],
        ["finish", "--phase", "implement"],
        ["report", "--all-phases"],
    ], ids=lambda argv: argv[0])
    def test_metrics_is_exit_1_naming_the_file(self, tmp_path, capsys, content, command):
        state_path = tmp_path / "state.json"
        state_path.write_bytes(_NOT_AN_OBJECT[content])
        argv = [command[0], "--state-file", str(state_path), *command[1:]]
        assert metrics.main(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith(f"Error: {state_path}: ")
        assert state_path.read_bytes() == _NOT_AN_OBJECT[content]

    @pytest.mark.parametrize("content", sorted(_NOT_AN_OBJECT))
    def test_save_guard_raises_value_error(self, tmp_path, content):
        from utils.commit_apply import save_guard

        state_path = tmp_path / "state.json"
        state_path.write_bytes(_NOT_AN_OBJECT[content])
        with pytest.raises(ValueError):
            save_guard(state_path, {"head": "0" * 40})
        assert state_path.read_bytes() == _NOT_AN_OBJECT[content]


class TestTheWriter:
    def test_indent_two_json_and_the_text_returned(self, tmp_path):
        path = tmp_path / "sub" / "state.json"
        text = state_file.write_document(path, {"b": [1], "a": "é"})
        assert text == json.dumps({"b": [1], "a": "é"}, indent=2)
        assert path.read_bytes() == text.encode("utf-8")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_owner_only_without_fsync(self, tmp_path, monkeypatch):
        def no_fsync(_descriptor):
            raise AssertionError("state.json is written without fsync")

        monkeypatch.setattr(os, "fsync", no_fsync)
        previous = os.umask(0o022)
        try:
            path = tmp_path / "state.json"
            state_file.write_document(path, {})
            path.chmod(0o644)
            state_file.write_document(path, {"a": 1})
        finally:
            os.umask(previous)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.parametrize("prefix", [state_file.STATE_TEMP_PREFIX, ".tracking_"])
    def test_a_failed_write_leaves_the_file_and_no_temp(self, tmp_path, monkeypatch, prefix):
        path = tmp_path / "state.json"
        path.write_text('{"old": true}', encoding="utf-8")
        sources = []

        def refuse(source, _destination, **_held_folder):
            # The rename is relative to the output folder held (dir_fd).
            sources.append(os.path.basename(source))
            raise OSError("disk failure")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError):
            state_file.write_document(path, {"new": True}, temp_prefix=prefix)
        monkeypatch.undo()
        assert sources and sources[0].startswith(prefix) and sources[0].endswith(".tmp")
        assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


class TestUpdate:
    def _seed(self, tmp_path, document):
        path = tmp_path / "state.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_a_change_is_written_under_the_lock(self, tmp_path):
        path = self._seed(tmp_path, {"a": 1})
        with state_file.update(path) as document:
            assert state_file._HELD_STATE_LOCKS, "the body runs under the file lock"
            document["b"] = 2
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
        assert not state_file._HELD_STATE_LOCKS

    def test_no_change_writes_nothing(self, tmp_path, monkeypatch):
        path = self._seed(tmp_path, {"a": {"b": [1]}})

        def no_replace(*_args):
            raise AssertionError("an unchanged document is not written")

        monkeypatch.setattr(os, "replace", no_replace)
        with state_file.update(path) as document:
            document["a"]["b"] = [1]

    def test_an_exception_in_the_body_writes_nothing(self, tmp_path):
        path = self._seed(tmp_path, {"a": 1})
        before = path.read_bytes()
        with pytest.raises(RuntimeError):
            with state_file.update(path) as document:
                document["a"] = 2
                raise RuntimeError("stop")
        assert path.read_bytes() == before

    def test_a_missing_file_raises_before_the_body(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            with state_file.update(tmp_path / "state.json"):
                raise AssertionError("not reached")

    @pytest.mark.parametrize("text", ["[]", "{not json", b"\xff\xfe"])
    def test_on_invalid_turns_the_read_into_the_callers_refusal(self, tmp_path, text):
        path = tmp_path / "state.json"
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")
        before = path.read_bytes()

        class Refused(Exception):
            pass

        with pytest.raises(Refused) as caught:
            with state_file.update(path, on_invalid=lambda exc: Refused(exc.detail)):
                raise AssertionError("not reached")
        assert isinstance(caught.value.__cause__, state_file.StateDocumentError)
        assert path.read_bytes() == before
        assert not state_file._HELD_STATE_LOCKS

    def test_on_invalid_leaves_the_bodys_own_errors_alone(self, tmp_path):
        """Only the read's error is the caller's refusal: one the body
        raises (a nested reader's, say) passes through as it was."""
        path = self._seed(tmp_path, {"a": 1})
        before = path.read_bytes()
        raised = state_file.StateDocumentError(tmp_path / "other.json", "not an object")
        with pytest.raises(state_file.StateDocumentError) as caught:
            with state_file.update(path, on_invalid=lambda exc: AssertionError(exc)) as document:
                document["a"] = 2
                raise raised
        assert caught.value is raised
        assert path.read_bytes() == before

    def test_on_invalid_is_not_called_for_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            with state_file.update(tmp_path / "state.json",
                                   on_invalid=lambda exc: AssertionError(exc)):
                raise AssertionError("not reached")


def test_metrics_keeps_the_old_names_for_the_same_objects():
    assert metrics.state_file_lock is state_file.locked
    assert metrics.StateLockTimeout is state_file.StateLockTimeout
    assert metrics.STATE_LOCK_FILENAME == state_file.STATE_LOCK_FILENAME
