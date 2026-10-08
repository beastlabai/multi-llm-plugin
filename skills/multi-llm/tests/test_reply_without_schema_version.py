#!/usr/bin/env python3
"""A reply that left out only its ``schema_version`` is read as the one
version there is (fix round 28).

Live run 12 (2026-10-08), Claude Code racing 03: the host copied the
task-generation subagent's JSON into ``task_generation_result.json`` without
its ``"schema_version": "1.0"``; the first ``--reaggregate`` failed with
``<root>: missing required field 'schema_version'`` and the host had to add
the key by hand before the second one passed. The key has one value
(``generated_tasks.schema.json``'s ``const``), so an absent one is now read as
that value, with one plain line saying so, wherever such a reply is read: the
native consume path (which writes the key into the file), the executor's
parse lane, ``task_is_complete``, and the judge verdict files a host writes
the same way. A present key with another value is still the schema's error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from tests.test_generate_tasks_orchestrator import (  # noqa: E402
    RESULT_NAME,
    _materialize,
    _phase_dir,
    _plan,
    _tasks_md,
    _valid_document,
)
import generate_tasks_orchestrator as gto  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.human_decisions import ingest_verdicts  # noqa: E402
from utils.schema_validator import supply_schema_version  # noqa: E402


def _without_version() -> dict:
    document = _valid_document()
    document.pop("schema_version")
    return document


class TestRacing03OnTheNativePath:

    def test_the_reaggregate_reads_it_as_version_one(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(_without_version()), encoding="utf-8")
        capsys.readouterr()
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        out = capsys.readouterr().out
        assert (f'[generate-tasks] {RESULT_NAME} has no "schema_version"; it was '
                'read as version "1.0", the only version of this document, and '
                "the key was written into the file before validating.") in out
        assert _tasks_md(plan).exists()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert list(on_disk)[0] == "schema_version"
        assert on_disk["schema_version"] == "1.0"

    def test_a_reply_with_its_version_says_nothing(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(_valid_document()), encoding="utf-8")
        capsys.readouterr()
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        assert "schema_version" not in capsys.readouterr().out

    def test_another_version_is_still_schema_invalid(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        document = _valid_document()
        document["schema_version"] = "2.0"
        (_phase_dir(plan) / RESULT_NAME).write_text(json.dumps(document), encoding="utf-8")
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 3
        assert "schema_invalid" in capsys.readouterr().err
        assert not _tasks_md(plan).exists()

    def test_a_result_on_disk_without_it_is_no_stale_result(self, tmp_path):
        path = tmp_path / RESULT_NAME
        path.write_text(json.dumps(_without_version()), encoding="utf-8")
        assert gto.stale_result_problems(str(path)) == []


class TestTheExecutorLanes:

    def test_the_parse_lane_supplies_it_and_says_so(self, capsys):
        capture = executor._Capture(
            stdout=json.dumps(_without_version()), stderr="", exit_code=0, duration=1.0,
        )

        class _Provider:
            name = "stub"

            def parse_output(self, stdout, stderr=""):
                return {"success": True, "data": stdout}

        status, error, data = executor._parse_and_validate(
            _Provider(), capture, kind="task_generation", prefer_arrays=False,
        )
        assert (status, error) == (executor.STATUS_OK, "")
        assert data["schema_version"] == "1.0"
        assert ('[exec-subagents] task_generation: the reply has no '
                '"schema_version"; it was read as version "1.0", the only '
                "version of this document, before validating.") in capsys.readouterr().out

    @pytest.mark.parametrize("document", [
        {"verdict": "looks good"},
        # A fragment the extractor served from inside a broken reply: one task.
        {"id": "T001", "title": "Countdown", "depends_on": []},
    ])
    def test_a_reply_that_lacks_more_than_the_version_keeps_its_errors(
        self, capsys, document,
    ):
        before = json.dumps(document)
        assert executor.supply_result_version("task_generation", document) is None
        assert json.dumps(document) == before
        assert any("schema_version" in error
                   for error in executor.validate_result("task_generation", document))
        executor._report_normalized("task_generation", [], None)
        assert capsys.readouterr().out == ""

    def test_a_file_without_it_counts_as_done(self, tmp_path):
        (tmp_path / RESULT_NAME).write_text(json.dumps(_without_version()), encoding="utf-8")
        task = {"task_id": "task_generation", "kind": "task_generation",
                "output_name": RESULT_NAME}
        assert executor.task_is_complete(task, str(tmp_path)) is True

    @pytest.mark.parametrize("kind", ["validation", "consolidation", "salvage", "implement"])
    def test_no_other_kind_is_given_a_version(self, kind):
        data = {"groups": []}
        assert executor.supply_result_version(kind, data) is None
        assert data == {"groups": []}


GROUP = "a1b2c3d4e5f60718"


class TestJudgeVerdictFiles:

    def _write(self, tmp_path, document) -> Path:
        path = tmp_path / "human_decision_batch_0.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _verdicts(self):
        return [{"group_id": GROUP, "decision": "approve", "reason": "clear",
                 "salvaged_description": None, "dropped": None}]

    def test_a_host_copy_without_it_is_ingested(self, tmp_path, capsys):
        self._write(tmp_path, {"verdicts": self._verdicts()})
        result = ingest_verdicts(phase_dir=str(tmp_path), expected_group_ids=[GROUP])
        assert result.ok and GROUP in result.verdicts
        assert ('[human-decision] human_decision_batch_0.json has no '
                '"schema_version"; it was read as version "1.0", the only '
                "version of this document.") in capsys.readouterr().err

    def test_a_verdict_file_that_lacks_more_says_nothing_of_a_version(
        self, tmp_path, capsys,
    ):
        self._write(tmp_path, {"verdicts": [{"group_id": GROUP, "decision": "approve"}]})
        result = ingest_verdicts(phase_dir=str(tmp_path), expected_group_ids=[GROUP])
        assert not result.ok
        assert any("schema_version" in error for error in result.errors)
        assert "[human-decision]" not in capsys.readouterr().err

    def test_another_version_is_still_rejected(self, tmp_path):
        self._write(tmp_path, {"schema_version": "9.9", "verdicts": self._verdicts()})
        result = ingest_verdicts(phase_dir=str(tmp_path), expected_group_ids=[GROUP])
        assert not result.ok and result.errors

    def test_the_executor_judge_lane_too(self, tmp_path):
        self._write(tmp_path, {"verdicts": self._verdicts()})
        task = {"task_id": "judge-0", "kind": "human_decision_judge",
                "output_name": "human_decision_batch_0.json"}
        assert executor.task_is_complete(task, str(tmp_path)) is True


class TestSupplySchemaVersion:

    def test_supplied_first_and_nothing_else_moves(self):
        document = {"verdicts": []}
        assert supply_schema_version(document, "human_decision_verdicts.schema.json") == "1.0"
        assert list(document) == ["schema_version", "verdicts"]
        document = _without_version()
        assert supply_schema_version(document, "generated_tasks.schema.json") == "1.0"
        assert list(document) == ["schema_version", "plan_preamble", "tasks"]

    @pytest.mark.parametrize("document", [
        {"schema_version": "2.0", "tasks": []}, {"schema_version": None}, [], "x", None,
    ])
    def test_a_present_key_or_a_non_object_is_left_alone(self, document):
        before = json.dumps(document)
        assert supply_schema_version(document, "generated_tasks.schema.json") is None
        assert json.dumps(document) == before

    def test_a_schema_without_a_const_version_supplies_nothing(self):
        document = {"tasks": []}
        assert supply_schema_version(document, "write_batch_result.schema.json") is None
        assert document == {"tasks": []}
