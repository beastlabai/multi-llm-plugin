"""``util check-reply``: a persisted validation reply that does not parse is caught.

Issue 11 of the 2026-10-01 three-host run: Sonnet validation subagents returned
JSON missing its closing ``]`` on 3 of 14 batches, and the host hand-repaired
one. The check is what lets the instructions say "re-ask once, never repair".
"""

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.check_reply import check_task_reply, main  # noqa: E402

HASH_A, HASH_B = "aaaa000000000000", "bbbb000000000000"


def _task(tmp_path, **extra):
    task = {
        "task_id": "validation-batch-0",
        "kind": "validation",
        "output_name": "validation_batch_0.json",
        "group_ids": [HASH_A, HASH_B],
    }
    task.update(extra)
    return task


def _reply(*hashes, status="valid"):
    return {
        "groups": [
            {"group_index": i, "group_hash": h, "status": status,
             "reason": "r", "confidence": 0.9}
            for i, h in enumerate(hashes)
        ],
        "metadata": {"model": "m", "schema_version": "2.1"},
    }


def _write(tmp_path, document):
    text = document if isinstance(document, str) else json.dumps(document)
    (tmp_path / "validation_batch_0.json").write_text(text, encoding="utf-8")


class TestCheckTaskReply:
    def test_a_complete_reply_passes(self, tmp_path):
        _write(tmp_path, _reply(HASH_A, HASH_B))
        assert check_task_reply(_task(tmp_path), str(tmp_path)) == []

    def test_a_bare_list_is_tolerated_as_the_merge_tolerates_it(self, tmp_path):
        _write(tmp_path, _reply(HASH_A, HASH_B)["groups"])
        assert check_task_reply(_task(tmp_path), str(tmp_path)) == []

    def test_a_reply_missing_its_closing_bracket_is_named_not_repaired(self, tmp_path):
        """The failure the run saw, byte for byte: the last `]` is gone."""
        whole = json.dumps(_reply(HASH_A, HASH_B))
        cut = whole.index("]")
        broken = whole[:cut] + whole[cut + 1:]
        _write(tmp_path, broken)
        problems = check_task_reply(_task(tmp_path), str(tmp_path))
        assert len(problems) == 1 and "not valid JSON" in problems[0]
        # Advisory only: the check never touches the file.
        assert (tmp_path / "validation_batch_0.json").read_text() == broken

    def test_a_missing_or_empty_file_is_reported(self, tmp_path):
        assert "no reply file" in check_task_reply(_task(tmp_path), str(tmp_path))[0]
        _write(tmp_path, "  \n")
        assert "empty" in check_task_reply(_task(tmp_path), str(tmp_path))[0]

    def test_a_missing_and_a_foreign_hash_are_named(self, tmp_path):
        _write(tmp_path, _reply(HASH_A, "cccc000000000000"))
        problems = " | ".join(check_task_reply(_task(tmp_path), str(tmp_path)))
        assert f"missing group_hash: {HASH_B}" in problems
        assert "foreign group_hash: cccc000000000000" in problems

    def test_an_unknown_status_is_reported(self, tmp_path):
        _write(tmp_path, _reply(HASH_A, HASH_B, status="maybe"))
        assert "status 'maybe'" in check_task_reply(_task(tmp_path), str(tmp_path))[0]

    def test_a_non_group_kind_is_only_checked_for_being_json(self, tmp_path):
        _write(tmp_path, {"anything": 1})
        assert check_task_reply(_task(tmp_path, kind="consolidation"), str(tmp_path)) == []
        _write(tmp_path, '{"anything": ')
        assert check_task_reply(_task(tmp_path, kind="consolidation"), str(tmp_path))


class TestMain:
    def _tasks_file(self, tmp_path):
        document = {"batches": [_task(tmp_path)]}
        path = tmp_path / "validation_tasks.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    def test_exit_0_and_ok_line_for_a_good_reply(self, tmp_path, capsys):
        _write(tmp_path, _reply(HASH_A, HASH_B))
        assert main(["--tasks-file", self._tasks_file(tmp_path)]) == 0
        assert "[REPLY_OK] validation-batch-0" in capsys.readouterr().out

    def test_exit_1_and_invalid_line_for_a_bad_one(self, tmp_path, capsys):
        _write(tmp_path, '{"groups": [')
        assert main(["--tasks-file", self._tasks_file(tmp_path)]) == 1
        assert "[REPLY_INVALID] validation-batch-0:" in capsys.readouterr().out

    def test_an_unknown_task_id_is_an_error(self, tmp_path, capsys):
        assert main(["--tasks-file", self._tasks_file(tmp_path),
                     "--task-id", "nope"]) == 2

    def test_it_is_a_registered_util_command(self):
        from multi_llm import UTIL_COMMANDS
        assert UTIL_COMMANDS["check-reply"][0] == "utils.check_reply"
