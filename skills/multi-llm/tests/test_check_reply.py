"""``util check-reply``: a persisted validation reply that does not parse is caught.

Issue 11 of the 2026-10-01 three-host run: Sonnet validation subagents returned
JSON missing its closing ``]`` on 3 of 14 batches, and the host hand-repaired
one. The check is what lets the instructions say "re-ask once, never repair".
"""

import json
import os
import subprocess
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

    def test_a_reply_keyed_by_group_id_passes_as_the_merge_takes_it(self, tmp_path):
        """Adversarial review, 2026-10-02: the merge joins on `group_hash or
        group_id`, so a check that demanded `group_hash` re-asked the subagent
        for a reply reaggregation would have used, then called it failed."""
        from utils.validation import merge_batched_validation_results

        reply = _reply(HASH_A, HASH_B)
        for entry in reply["groups"]:
            entry["group_id"] = entry.pop("group_hash")
        _write(tmp_path, reply)
        assert check_task_reply(_task(tmp_path), str(tmp_path)) == []

        merged = merge_batched_validation_results(
            str(tmp_path),
            {"batches": [{"batch_index": 0, "group_indices": [0, 1],
                          "group_ids": [HASH_A, HASH_B]}]},
            2,
        )
        assert [r["status"] for r in merged] == ["valid", "valid"]

    def test_a_reply_with_no_id_at_all_is_still_reported(self, tmp_path):
        reply = _reply(HASH_A, HASH_B)
        for entry in reply["groups"]:
            del entry["group_hash"]
        _write(tmp_path, reply)
        problems = " | ".join(check_task_reply(_task(tmp_path), str(tmp_path)))
        assert "groups[0] has no group_hash (or group_id)" in problems
        assert f"missing group_hash: {HASH_A}, {HASH_B}" in problems

    def test_an_unknown_status_is_reported(self, tmp_path):
        _write(tmp_path, _reply(HASH_A, HASH_B, status="maybe"))
        assert "status 'maybe'" in check_task_reply(_task(tmp_path), str(tmp_path))[0]

    def test_a_non_group_kind_is_only_checked_for_being_json(self, tmp_path):
        _write(tmp_path, {"anything": 1})
        assert check_task_reply(_task(tmp_path, kind="consolidation"), str(tmp_path)) == []
        _write(tmp_path, '{"anything": ')
        assert check_task_reply(_task(tmp_path, kind="consolidation"), str(tmp_path))



class TestOneCheckForBothLanes:
    """The native lane's check and the executor lane's are one function
    (`schema_validator.validation_reply_problems`). They differ only in what
    the merge recovers from by leaving a group `validation_failed`: the native
    lane re-asks over it, the executor lane leaves it to the merge, since a
    schema failure is terminal for a validation batch."""

    @staticmethod
    def _executor(document):
        from utils import subagent_executor
        return subagent_executor.validate_result("validation", document)

    def test_an_unknown_status_is_re_asked_natively_and_left_to_the_merge(self, tmp_path):
        from utils.schema_validator import VALIDATION_VERDICTS

        reply = _reply(HASH_A, HASH_B)
        reply["groups"][0]["status"] = "approved"
        _write(tmp_path, reply)
        assert check_task_reply(_task(tmp_path), str(tmp_path)) == [
            "groups[0] status 'approved' is not one of " + ", ".join(VALIDATION_VERDICTS)]
        assert self._executor(reply) == []

    @pytest.mark.parametrize("document, problem", [
        ({"groups": None}, 'reply has no "groups" list'),
        (42, 'reply has no "groups" list'),
        ({"groups": ["x"]}, "groups[0] is not an object"),
        ({"groups": [{"group_hash": HASH_A, "status": ""}]}, "groups[0] has no status"),
        ({"groups": [{"status": "valid"}]}, "groups[0] has no group_hash (or group_id)"),
    ])
    def test_what_the_merge_cannot_use_fails_on_both(self, tmp_path, document, problem):
        _write(tmp_path, document)
        assert problem in check_task_reply(_task(tmp_path), str(tmp_path))
        assert problem in self._executor(document)

    def test_coverage_is_checked_natively_only(self):
        from utils.schema_validator import validation_reply_problems

        reply = _reply(HASH_A)
        assert validation_reply_problems(reply, group_ids=[HASH_A, HASH_B]) == [
            f"missing group_hash: {HASH_B}"]
        assert validation_reply_problems(
            reply, group_ids=[HASH_A, HASH_B], recoverable=False) == []


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

    def test_it_runs_as_a_direct_script(self, tmp_path):
        """`python utils/check_reply.py` is a supported entry point: its import
        shim loads schema_validator as a top-level module, so every module that
        one imports must load outside the package too."""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        done = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "utils" / "check_reply.py"), "--help"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
        )
        assert done.returncode == 0, done.stderr
        assert "--tasks-file" in done.stdout


class TestRevalidationRunsTheCheckToo:
    """Adversarial review, 2026-10-02: only the three review cores told the host
    to check a validation reply. Revalidation in apply-suggestions and
    apply-code-fixes dispatches the same subagents with the same failure mode,
    and `check_reply` already handles kind `revalidation`."""

    @staticmethod
    def _paragraph(text: str) -> str:
        start = text.index("**Check each reply right after you write it.**")
        end = text.index("leave its file as it is and say so.", start)
        return " ".join(text[start:end].split())

    def test_the_revalidation_runbook_carries_the_review_paragraph(self):
        """Both apply cores send revalidation to one runbook,
        ``references/revalidation.md``, which a host reads as it is: the same
        paragraph, its command behind the ``<prefix>`` that file defines and its
        path under the ``{review}`` dir the core names."""
        instructions = SKILL_ROOT / "instructions"
        review = self._paragraph(
            (instructions / "review-plan.md").read_text(encoding="utf-8"))
        text = (SKILL_ROOT / "references" / "revalidation.md").read_text(
            encoding="utf-8")
        assert self._paragraph(text) == review.replace(
            'uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python '
            '"{SKILL_DIR}/multi_llm.py" util check-reply',
            "<prefix> util check-reply",
        ).replace(
            '"{plan}/review-plan/validation_tasks.json"',
            '"{plan}/{review}/revalidation_tasks.json"')
        # Above both the single- and the multi-batch path.
        assert text.index("util check-reply") < text.index(
            "### Single batch: `[REVALIDATION_PENDING]`")
        for name, phase in (("apply-suggestions.md", "review-plan"),
                            ("apply-code-fixes.md", "code-review")):
            core = (instructions / name).read_text(encoding="utf-8")
            trigger = next(line for line in core.splitlines()
                           if "references/revalidation.md" in line)
            assert "[REVALIDATION_PENDING]" in trigger, name
            assert f"`{phase}` as its `{{review}}`" in trigger, name

    def test_a_revalidation_task_is_checked_like_a_validation_one(self, tmp_path):
        _write(tmp_path, _reply(HASH_A))
        problems = check_task_reply(_task(tmp_path, kind="revalidation"), str(tmp_path))
        assert problems == [f"missing group_hash: {HASH_B}"]


class TestARepliesNotWrittenYet:
    """Fix round 28 (live run 12, Claude Code racing 01): the host wrote wave
    1's four replies, started wave 2's three subagents and ran check-reply
    over the whole file while they were still running. It printed three
    ``[REPLY_INVALID] … no reply file`` lines and exited 1, for replies that
    were simply not there yet."""

    def _tasks_file(self, tmp_path, count):
        batches = [
            _task(tmp_path, task_id=f"validation-batch-{n}",
                  output_name=f"validation_batch_{n}.json")
            for n in range(count)
        ]
        path = tmp_path / "validation_tasks.json"
        path.write_text(json.dumps({"batches": batches}), encoding="utf-8")
        return str(path)

    def _write_batch(self, tmp_path, n, document):
        text = document if isinstance(document, str) else json.dumps(document)
        (tmp_path / f"validation_batch_{n}.json").write_text(text, encoding="utf-8")

    def test_racing_01_wave_2_still_running(self, tmp_path, capsys):
        tasks_file = self._tasks_file(tmp_path, 7)
        for n in range(4):
            self._write_batch(tmp_path, n, _reply(HASH_A, HASH_B))
        assert main(["--tasks-file", tasks_file]) == 0
        out = capsys.readouterr().out
        assert "[REPLY_INVALID]" not in out
        assert out.count("[REPLY_OK]") == 4
        missing = [line for line in out.splitlines() if line.startswith("[REPLY_MISSING]")]
        assert [line.split(":")[0] for line in missing] == [
            f"[REPLY_MISSING] validation-batch-{n}" for n in (4, 5, 6)]
        assert all("not a bad reply" in line and "nothing to re-ask" in line
                   for line in missing)
        assert str(tmp_path / "validation_batch_4.json") in missing[0]

    def test_one_alone_by_its_task_id(self, tmp_path, capsys):
        tasks_file = self._tasks_file(tmp_path, 2)
        assert main(["--tasks-file", tasks_file, "--task-id", "validation-batch-1"]) == 0
        assert capsys.readouterr().out.startswith("[REPLY_MISSING] validation-batch-1:")

    def test_a_bad_reply_beside_a_missing_one_still_exits_1(self, tmp_path, capsys):
        tasks_file = self._tasks_file(tmp_path, 2)
        self._write_batch(tmp_path, 0, '{"groups": [')
        assert main(["--tasks-file", tasks_file]) == 1
        out = capsys.readouterr().out
        assert "[REPLY_INVALID] validation-batch-0:" in out
        assert "[REPLY_MISSING] validation-batch-1:" in out

    def test_something_else_at_the_path_is_still_invalid(self, tmp_path, capsys):
        tasks_file = self._tasks_file(tmp_path, 1)
        (tmp_path / "validation_batch_0.json").mkdir()
        assert main(["--tasks-file", tasks_file]) == 1
        assert "[REPLY_INVALID] validation-batch-0: no reply file" in capsys.readouterr().out

    @pytest.mark.parametrize("name", [
        "instructions/review-plan.md", "instructions/review-code.md",
        "instructions/review-tasks.md",
        # The apply cores' revalidation runbook, which both name by a trigger.
        "references/revalidation.md",
    ])
    def test_every_runbook_checks_the_whole_file_after_the_subagents_return(self, name):
        text = " ".join((SKILL_ROOT / name).read_text(encoding="utf-8").split())
        assert ("(omit `--task-id` to check every batch, and only once every "
                "batch's subagent has returned: a reply not written yet prints "
                "`[REPLY_MISSING]`, which is no bad reply and no re-ask)") in text
