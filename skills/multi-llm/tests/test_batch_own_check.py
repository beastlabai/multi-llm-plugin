"""A code batch that reports a failing check is never a clean success.

Fix round 14 (codex racing run 5, 2026-10-06): apply-code-fixes batch 1's
runner wrote in its free-text ``notes`` that "the bundled test fails earlier
because it expects a two-sample move to be rejected"; afterwards
``node tests/race-logic.test.mjs`` failed, yet the batch was recorded
``ok (applied)`` and the summary, the result and the host all said every fix
was applied. ``driver.verify_command`` was unset, so only a log line nobody
relayed hinted at it.

Deterministic, never a reading of ``notes``: every batch that edits code is
asked (``WRITE_CHECK_CONTRACT``) to report the checks it ran in the same
lenient ``check`` object a check task answers with, and one that reports a
failing check is recorded applied *with* that finding — on the row, the
verdict stamp, summary.md's ``## Check Findings``, the batch log line, the
``[BATCH_CHECK_FAILED]`` line, the status mirror and the result notice. A
code phase that finished with no ``driver.verify_command`` says, on the same
surfaces, that nothing ran the repository's tests and which setting to add.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import test_write_executor
from tests.test_write_executor import FakeState
from utils import subagent_executor as executor
from utils.subagent_tasks import (
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.write_batch_scheduler import (
    BATCH_CHECK_MARKER,
    CHECK_RESULT_CONTRACT,
    WRITE_CHECK_CONTRACT,
    unverified_line,
    verify_command_candidate,
    with_result_contract,
    write_result_notice,
)

env = test_write_executor.env

FAILING = (
    "node tests/race-logic.test.mjs: AssertionError: a skipped gate must be "
    "rejected"
)


# ---------------------------------------------------------------------------
# The contract and the reader
# ---------------------------------------------------------------------------

class TestTheContractAsksABatchForTheChecksItRan:

    @staticmethod
    def _task(kind, declared):
        return {"task_id": "x", "expected_ids": ["x"], "declared_files": declared}

    @pytest.mark.parametrize("kind", ["apply_code_fixes", "implement"])
    def test_a_code_batch_is_asked(self, kind):
        text = with_result_contract(kind, self._task(kind, ["src/a.py"]), "Do it.")
        assert WRITE_CHECK_CONTRACT in text
        assert "## Checks you ran (required when you ran any)" in text
        assert CHECK_RESULT_CONTRACT not in text

    @pytest.mark.parametrize("kind", ["apply_suggestions", "apply_task_suggestions"])
    def test_a_plan_text_batch_is_not(self, kind):
        text = with_result_contract(kind, self._task(kind, ["plans/p.md"]), "Do it.")
        assert WRITE_CHECK_CONTRACT not in text

    def test_a_check_task_keeps_its_own_contract(self):
        text = with_result_contract("implement", self._task("implement", []), "Check.")
        assert CHECK_RESULT_CONTRACT in text
        assert WRITE_CHECK_CONTRACT not in text


class TestTheReader:

    @pytest.mark.parametrize("reply", [
        {"status": "applied"},
        {"status": "applied", "check": None},
        {"status": "applied", "check": {}},
        {"status": "applied", "check": {"passed": "maybe"}},
        "not an object",
    ])
    def test_no_usable_report_is_no_finding(self, reply):
        """A batch reports `check` only when it ran one: saying nothing
        usable is not a finding here, as it is for a check task."""
        assert executor.batch_check_findings(reply) is None

    def test_a_pass_is_empty_and_a_failure_is_its_findings(self):
        assert executor.batch_check_findings(
            {"check": {"passed": True, "findings": []}}) == []
        assert executor.batch_check_findings(
            {"check": {"passed": "false", "findings": FAILING}}) == [FAILING]
        assert executor.batch_check_findings({"check": {"passed": False}}) == [
            executor.CHECK_FAILED_WITHOUT_FINDINGS]

    def test_notes_are_never_read(self):
        reply = {"status": "applied", "notes": "the bundled test fails"}
        assert executor.batch_check_findings(reply) is None

    # Fix round 15 (review of round 14, R3): the check task's fail-closed
    # reading made a passing code batch "not a clean finish".

    @pytest.mark.parametrize("findings", [
        ["npm test: 12 passed"], "npm test: 12 passed", ["all 3 suites green"],
    ])
    def test_a_clear_pass_is_a_pass_whatever_it_lists(self, findings):
        """`passed: true` is the batch's own verdict on its checks; what it
        lists beside it is informational, never a failing check."""
        assert executor.batch_check_findings(
            {"check": {"passed": True, "findings": findings}}) == []
        # A check TASK keeps the fail-closed reading: a finding beside a
        # pass still counts there.
        assert executor.check_findings_from(
            {"check": {"passed": True, "findings": findings}})

    def test_a_check_that_could_not_run_says_so_not_that_it_fails(self):
        assert executor.batch_check_findings(
            {"check": {"passed": False, "could_not_run": True, "findings": []}}
        ) == [executor.BATCH_CHECK_COULD_NOT_RUN]
        assert executor.batch_check_findings(
            {"check": {"could_not_run": "yes", "findings": ["npx: not found"]}}
        ) == [f"{executor.BATCH_CHECK_COULD_NOT_RUN}: npx: not found"]


# ---------------------------------------------------------------------------
# The executor leg
# ---------------------------------------------------------------------------

def _fix_task(env, index=0, path="src/a.py"):
    return make_write_task(
        kind="apply_code_fixes",
        task_id=f"apply-fix-batch-{index}",
        prompt=f"Apply fix {index} to {env.root / path}\n\n{PRIOR_CHANGES_TOKEN}\n",
        inputs=build_inputs(plan_file=str(env.plan)),
        batch_index=index,
        prior_changes_id=f"{path}#{index:08d}",
        phase_dir=str(env.phase_dir),
        declared_files=[str(env.root / path)],
        expected_ids=[f"f{index}"],
        output_name=write_output_name("apply_code_fixes", index),
    )


def _reply(ids, files, check=None):
    out = {"status": "applied", "ids": ids, "files_touched": files,
           "summary": "rank cars by unwrapped progress",
           "notes": "Direct progress checks passed; the bundled test fails."}
    if check is not None:
        out["check"] = check
    return out


class TestAFailingCheckOfAnAppliedBatch:

    def _run(self, env, check, *, state=None, summary=None, lines=None):
        env.edits([{"path": "src/a.py", "mode": "patch", "content": "a = 2\n"}],
                  result=_reply(["f0"], ["src/a.py"], check))
        return env.scheduler(
            [_fix_task(env)], kind="apply_code_fixes", state=state,
            summary_file=summary,
            log=(lines.append if lines is not None else None),
        ).run()

    def test_it_is_applied_with_its_finding_on_every_surface(self, env):
        from utils import run_status

        lines = []
        state = FakeState(env.tmp_path / "state.json")
        result = self._run(
            env, {"passed": False, "findings": [FAILING]}, state=state, lines=lines)

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.batch_check == [FAILING]
        # The batch is applied, the phase completes: a finding is not a stop.
        assert result.exit_code == 0 and result.finalized is True
        assert result.batch_check_findings() == {"apply-fix-batch-0": [FAILING]}

        # The rows, both reports, and the verdict stamp a resume reads.
        for name in ("write_batch_report.json", "subagent_batch_report.json"):
            row = json.loads((env.phase_dir / name).read_text())["tasks"][0]
            assert row["batch_check"] == {"passed": False, "findings": [FAILING]}, name
        stamped = json.loads(Path(outcome.output_path).read_text())["verdict"]
        assert stamped["batch_check"] == {"passed": False, "findings": [FAILING]}

        # The batch log line and the result's own lines.
        assert any(
            "apply-fix-batch-0 applied, but its own report says a check it ran fails" in line
            and FAILING in line for line in lines
        ), lines
        blob = "\n".join(result.messages)
        assert (
            f"{BATCH_CHECK_MARKER} apply-fix-batch-0: its edits were applied, "
            f"but its own report says a check it ran on them fails "
            f"(1 problem(s)): \"{FAILING}\""
        ) in blob
        assert "Not a clean finish: 1 batch(es) applied their edits, but their own " \
               "report says a check they ran fails: apply-fix-batch-0 (\"" in blob

        # The status mirror, the progress line and the notice a poll relays.
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        write = mirror["progress"]["write_subagents"]
        assert write["batch_check_task_ids"] == ["apply-fix-batch-0"]
        assert write["tasks"][0]["batch_check_findings"] == [FAILING]
        view = {"progress": mirror["progress"]}
        assert "own check fails apply-fix-batch-0" in ", ".join(
            run_status._mirror_text("write_subagents", write))
        notice = run_status._write_result_notice([view])
        assert notice.startswith("not a clean finish: 1 batch(es) applied their edits")
        assert FAILING[:60] in notice
        assert write_result_notice(result).startswith("not a clean finish:")

    def test_a_passing_check_is_recorded_and_reads_clean(self, env):
        result = self._run(env, {"passed": True, "findings": []},
                           state=FakeState(env.tmp_path / "state.json"))
        assert result.outcomes[0].batch_check == []
        assert result.batch_check_findings() == {}
        row = json.loads((env.phase_dir / "write_batch_report.json").read_text())["tasks"][0]
        assert row["batch_check"] == {"passed": True, "findings": []}
        assert BATCH_CHECK_MARKER not in "\n".join(result.messages)
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        assert "batch_check_task_ids" not in mirror["progress"]["write_subagents"]

    def test_a_pass_listing_what_it_ran_reads_clean(self, env):
        result = self._run(env, {"passed": True, "findings": ["npm test: 12 passed"]},
                           state=FakeState(env.tmp_path / "state.json"))
        assert result.outcomes[0].batch_check == []
        assert result.batch_check_findings() == {}
        blob = "\n".join(result.messages)
        assert BATCH_CHECK_MARKER not in blob and "Not a clean finish" not in blob

    def test_a_check_that_could_not_run_is_named_so_on_every_surface(self, env):
        lines = []
        summary = env.phase_dir / "summary.md"
        result = self._run(
            env, {"passed": False, "could_not_run": True,
                  "findings": ["npx: command not found"]},
            state=FakeState(env.tmp_path / "state.json"), lines=lines,
            summary=str(summary),
        )
        outcome = result.outcomes[0]
        assert outcome.ok and outcome.batch_check == [
            f"{executor.BATCH_CHECK_COULD_NOT_RUN}: npx: command not found"]
        blob = "\n".join(result.messages)
        assert (
            f"{BATCH_CHECK_MARKER} apply-fix-batch-0: its edits were applied, "
            f"but its own report says its check could not run: npx: command not found"
        ) in blob, blob
        assert "fails" not in blob.split(BATCH_CHECK_MARKER, 1)[1].splitlines()[0]
        assert "Not a clean finish: 1 batch(es) applied their edits, but their own " \
               "report says their check could not run: apply-fix-batch-0" in blob, blob
        assert any(
            "applied, but its own report says its check could not run" in line
            for line in lines
        ), lines
        assert "a failing check" not in write_result_notice(result)
        assert "### apply-fix-batch-0: its check could not run" in summary.read_text()
        from utils import run_status

        write = json.loads((env.phase_dir / ".status.json").read_text())[
            "progress"]["write_subagents"]
        text = ", ".join(run_status._mirror_text("write_subagents", write))
        assert "own check could not run apply-fix-batch-0" in text, text
        assert "own check fails" not in text

    def test_a_reply_that_reports_no_check_leaves_every_row_as_it_was(self, env):
        result = self._run(env, None, state=FakeState(env.tmp_path / "state.json"))
        assert result.outcomes[0].batch_check is None
        row = json.loads((env.phase_dir / "write_batch_report.json").read_text())["tasks"][0]
        assert "batch_check" not in row

    def test_a_resume_keeps_what_the_earlier_run_s_check_reported(self, env):
        first = self._run(env, {"passed": False, "findings": [FAILING]})
        assert first.outcomes[0].batch_check == [FAILING]
        env.edits([], result=None)
        resumed = env.scheduler(
            [_fix_task(env)], kind="apply_code_fixes", resume=True,
        ).run()
        outcome = resumed.outcomes[0]
        assert outcome.skipped is True
        assert outcome.batch_check == [FAILING]
        assert resumed.batch_check_findings() == {"apply-fix-batch-0": [FAILING]}


class TestACorrectedAttemptIsNotAFinding:
    """Run 6, Codex racing 06r: T006 reported `check.passed: false` with
    the one finding "The initial `node --check -` invocation used CommonJS
    mode and failed on top-level await; the corrected `node
    --input-type=module --check` syntax check passed." — a check that
    passed. The plugin recorded "applied, its check fails" and the host
    relayed a failing check that was not real. The contract now asks for
    the last run of each check, and every surface says the claim is the
    batch's own report and quotes it; a real failure is still recorded."""

    T006 = (
        "The initial `node --check -` invocation used CommonJS mode and failed "
        "on top-level await; the corrected `node --input-type=module --check` "
        "syntax check passed."
    )

    def test_the_contract_asks_for_the_last_run_of_each_check(self):
        assert "Report the last run of each check." in WRITE_CHECK_CONTRACT
        assert "an attempt you corrected is not a finding" in WRITE_CHECK_CONTRACT
        assert ("`true` only when the last run of every check you ran passed"
                in WRITE_CHECK_CONTRACT)
        # A check task may correct only its own wrong form of the command
        # the task names, never swap the check for another.
        assert "Report the last run of each check." in CHECK_RESULT_CONTRACT
        assert "ran it as the task names it" in CHECK_RESULT_CONTRACT
        assert "the wrong form is not a finding" in CHECK_RESULT_CONTRACT

    def test_the_corrected_run_is_the_same_check_in_full(self):
        """Fix round 17 (review of round 16): the code batch's contract let
        "the corrected run" be the reported one without pinning it to the
        command, so `npm test` failing and `npm test -- src/x.test.js` (or
        `--passWithNoTests`) passing was a pass within the letter. The check
        task's contract pins "as the task names it"; this one now says the
        same: the full command the task or the repository names, never a
        narrower one, and a narrower pass leaves the failure to report."""
        text = " ".join(WRITE_CHECK_CONTRACT.split())
        assert (
            "The corrected run is the same check over the same scope: the full "
            "command the task or the repository names, with only what you ran "
            "wrongly put right."
        ) in text
        assert "never a narrower one" in text
        assert "`--passWithNoTests`" in text
        assert (
            "If the only run that passed was narrower, the failing run is the "
            "last run of that check: report it."
        ) in text
        schema = json.loads(
            (Path(__file__).parent.parent / "schemas" / "write_batch_result.schema.json")
            .read_text(encoding="utf-8"))
        check = schema["properties"]["check"]["properties"]
        for key in ("passed", "findings"):
            assert "never a narrower one" in check[key]["description"], key

    def test_a_reported_failure_is_quoted_as_the_batch_s_own_words(self, env):
        lines = []
        summary = env.phase_dir / "summary.md"
        result = TestAFailingCheckOfAnAppliedBatch()._run(
            env, {"passed": False, "findings": [self.T006]},
            state=FakeState(env.tmp_path / "state.json"), lines=lines,
            summary=str(summary),
        )
        # Never second-guessed: still applied with a failing check.
        assert result.outcomes[0].batch_check == [self.T006]
        quoted = f'"{self.T006}"'
        blob = "\n".join(result.messages)
        assert (
            f"{BATCH_CHECK_MARKER} apply-fix-batch-0: its edits were applied, "
            f"but its own report says a check it ran on them fails "
            f"(1 problem(s)): {quoted}"
        ) in blob, blob
        notice = write_result_notice(result)
        assert notice.startswith(
            "not a clean finish: 1 batch(es) applied their edits, but their "
            "own report says a check they ran fails: apply-fix-batch-0 (\""
        ), notice
        assert any(
            "applied, but its own report says a check it ran fails (" + quoted
            in line for line in lines
        ), lines
        text = summary.read_text(encoding="utf-8")
        assert "each one's own closing report says a check it ran failed" in text
        assert "in its own words" in text


class TestAnImplementTaskWhoseOwnTestFails:
    """The same for implement's code tasks — and it holds nothing back: a
    check *task* with findings holds its dependents, a write task whose own
    test fails does not."""

    def _task(self, env, task_id, path, deps=()):
        return env.apply_task(
            int(task_id[1:]) - 1, kind="implement", task_id=task_id,
            expected_ids=[task_id], output_name=f"implement_batch_{task_id}.json",
            declared_files=[str(env.root / path)], depends_on=list(deps),
            prior_changes_id=task_id,
        )

    def test_the_dependents_run_and_the_summary_lists_the_failure(self, env):
        env.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps(
            [{"path": "src/a.py", "mode": "patch", "content": "a = 3\n"}]))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002"]))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": False, "findings": [FAILING]}))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Apply batch 0")
        state = FakeState(env.tmp_path / "state.json")
        summary = env.out_dir / "implement" / "summary.md"
        result = env.scheduler(
            [self._task(env, "T001", "src/a.py"),
             self._task(env, "T002", "src/a.py", deps=["T001"])],
            kind="implement", state=state, summary_file=str(summary),
        ).run()

        assert result.exit_code == 0, result.messages
        assert [o.task_id for o in result.outcomes] == ["T001", "T002"]
        assert result.blocked_by_check == {}
        assert "task_check_findings" not in state.state, "never a hold"
        assert result.batch_check_findings() == {"T001": [FAILING]}
        text = summary.read_text(encoding="utf-8")
        assert "| T001 | applied, its check fails |" in text
        assert "## Check Findings" in text
        assert "### T001: a check it ran fails" in text
        assert f"- {FAILING}" in text
        assert "| T002 | applied |" in text


    def test_a_stop_prints_the_marker_line_too(self, env):
        """Fix round 15 (review of round 14): `[BATCH_CHECK_FAILED]` was
        printed only on a run that finished; a stop after the batch printed
        `[CHECK_FINDINGS]` (round 14) but not this one."""
        env.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps(
            [{"path": "src/a.py", "mode": "patch", "content": "a = 3\n"}]))
        env.monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", "Apply batch 0")
        env.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002"]))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": False, "findings": [FAILING]}))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Apply batch 0")
        env.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Apply batch 1")
        result = env.scheduler(
            [self._task(env, "T001", "src/a.py"),
             self._task(env, "T002", "src/b.py", deps=["T001"])],
            kind="implement", state=FakeState(env.tmp_path / "state.json"),
        ).run()

        assert result.exit_code == 3
        assert result.stop and result.stop["task_id"] == "T002", result.messages
        blob = "\n".join(result.messages)
        assert f"{BATCH_CHECK_MARKER} T001: its edits were applied" in blob, blob
        # Before the stop's own block, so the ERROR block stays in one piece.
        assert blob.index(BATCH_CHECK_MARKER) < blob.index("ERROR: implement stopped")


# ---------------------------------------------------------------------------
# No driver.verify_command: one plain line, naming the setting
# ---------------------------------------------------------------------------

class TestTheCandidate:

    def test_package_json_test_script_by_its_lockfile(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps(
            {"scripts": {"test": "node --test"}}), encoding="utf-8")
        assert verify_command_candidate(str(tmp_path)) == (
            "npm test", "package.json defines scripts.test")
        (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")
        assert verify_command_candidate(str(tmp_path))[0] == "pnpm test"

    def test_npm_s_placeholder_is_no_test(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"scripts": {
            "test": 'echo "Error: no test specified" && exit 1'}}), encoding="utf-8")
        assert verify_command_candidate(str(tmp_path)) is None

    @pytest.mark.parametrize("name,text,expected", [
        ("Makefile", "build:\n\tcc x.c\ntest: build\n\t./t\n", "make test"),
        ("Cargo.toml", "[package]\nname = 'x'\n", "cargo test"),
        ("go.mod", "module x\n", "go test ./..."),
        ("pyproject.toml", "[tool.pytest.ini_options]\n", "python -m pytest"),
    ])
    def test_other_manifests(self, tmp_path, name, text, expected):
        (tmp_path / name).write_text(text, encoding="utf-8")
        assert verify_command_candidate(str(tmp_path))[0] == expected

    def test_nothing_obvious_is_none_and_the_line_says_what_to_set(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\tcc x.c\n", encoding="utf-8")
        assert verify_command_candidate(str(tmp_path)) is None
        line = unverified_line(str(tmp_path))
        assert line.startswith(
            "Nothing ran this repository's tests: driver.verify_command is not set.")
        assert "under driver: in .multi-llm/providers.yaml" in line

    def test_the_candidate_is_never_run_or_written(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps(
            {"scripts": {"test": "touch ran"}}), encoding="utf-8")
        before = sorted(p.name for p in tmp_path.iterdir())
        line = unverified_line(str(tmp_path))
        assert 'add verify_command: "npm test" under driver: in ' \
               '.multi-llm/providers.yaml (package.json defines scripts.test).' in line
        assert sorted(p.name for p in tmp_path.iterdir()) == before


class TestACodePhaseWithNoVerifyCommand:

    def test_the_result_says_nothing_ran_the_tests_and_names_the_setting(self, env):
        (env.root / "package.json").write_text(json.dumps(
            {"scripts": {"test": "node --test"}}), encoding="utf-8")
        env.edits([{"path": "src/a.py", "mode": "patch", "content": "a = 2\n"}],
                  result=_reply(["f0"], ["src/a.py"]))
        summary = env.phase_dir / "summary.md"
        result = env.scheduler(
            [_fix_task(env)], kind="apply_code_fixes",
            state=FakeState(env.tmp_path / "state.json"), summary_file=str(summary),
        ).run()

        assert result.exit_code == 0, result.messages
        assert result.unverified == unverified_line(str(env.root))
        assert '"npm test"' in result.unverified
        assert result.unverified in result.messages
        report = json.loads((env.phase_dir / "write_batch_report.json").read_text())
        assert report["unverified"] == result.unverified
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        assert mirror["progress"]["write_subagents"]["unverified"] == result.unverified
        assert "**Verification**: not configured (`driver.verify_command` is " \
               "unset). Nothing ran this repository's tests" in summary.read_text()
        assert write_result_notice(result) == result.unverified

    def test_a_plan_text_phase_owes_no_such_line(self, env):
        env.edits([{"path": "plans/p.md", "mode": "patch", "content": "x\n"}],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "s"})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 0, result.messages
        assert result.unverified is None


# ---------------------------------------------------------------------------
# The phase's own result: implement's result line, apply-code-fixes' report
# ---------------------------------------------------------------------------

SIMPLE_PLAN = """# Simple plan

## Overview
Two edits.

## Tasks

### T001: Edit: change the module
Change it.
- Depends on: none
- Files to modify: src/main.py

### T002: Edit: change it again
Change it again.
- Depends on: T001
- Files to modify: src/main.py
"""


class TestImplementsResultLineSaysTheCheckFails:

    def test_the_notice_names_the_failing_check_and_the_missing_setting(
        self, tmp_path, monkeypatch, capsys,
    ):
        import multi_llm
        from tests.test_implement_stop_account import make_repo
        from utils import provider_registry as registry
        from utils.cli_common import parse_result_line

        repo = make_repo(tmp_path, monkeypatch, SIMPLE_PLAN)
        try:
            monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
                {"passed": False, "findings": [FAILING]}))
            monkeypatch.setenv(
                "MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Title: Edit: change the module")
            assert multi_llm.main([
                "implement", str(repo.plan), "--yes", "--exec-subagents",
            ]) == 0
            out = capsys.readouterr().out
        finally:
            registry._config = None
            registry._config_key = None
        assert f"{BATCH_CHECK_MARKER} T001: its edits were applied" in out
        notice = parse_result_line(out)["notice"]
        assert notice.startswith(
            "not a clean finish: 1 batch(es) applied their edits, but their own "
            "report says a check they ran fails: T001 (\"")
        assert "Nothing ran this repository's tests: driver.verify_command is not set." in notice
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert "### T001: a check it ran fails" in summary


class TestTheApplyReport:
    """apply-code-fixes' own summary.md and RESULT block (the base writes
    both, from the scheduler's rows)."""

    def _orchestrator(self, tmp_path, rows, unverified=None):
        from tests.test_apply_orchestrator_base import _orchestrator_on

        class Result:
            outcomes = []
            report_path = None
            stopped_at_batch = None
            verification = None

            def report_rows(self):
                return [dict(row) for row in rows]

        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_kind = "apply_code_fixes"
        orch.write_tasks = [{"task_id": row["task_id"], "expected_ids": ["a"]}
                            for row in rows]
        orch.write_result = Result()
        orch.write_result.unverified = unverified
        return orch

    ROWS = [
        {"task_id": "apply-batch-0", "status": "ok", "write_outcome": "applied",
         "expected_ids": ["a"], "duration_seconds": 1.0, "runner": "codex:x",
         "batch_check": {"passed": False, "findings": [FAILING]}},
        {"task_id": "apply-batch-1", "status": "ok", "write_outcome": "applied",
         "expected_ids": ["a"], "duration_seconds": 1.0, "runner": "codex:x"},
    ]

    def test_summary_md_never_reads_the_failing_batch_as_a_clean_success(
        self, tmp_path,
    ):
        orch = self._orchestrator(tmp_path, self.ROWS, unverified=unverified_line(None))
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "### apply-batch-0 — ok (applied, its check fails)" in text
        assert "### apply-batch-1 — ok (applied)" in text
        assert "## Check Findings" in text
        assert "### apply-batch-0: a check it ran fails" in text
        assert f"- {FAILING}" in text
        assert "- Verification: not configured (`driver.verify_command` is " \
               "unset). Nothing ran this repository's tests" in text

    def test_the_result_block_says_it_too(self, tmp_path, capsys):
        orch = self._orchestrator(tmp_path, self.ROWS, unverified=unverified_line(None))
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "Not a clean finish: 1 batch(es) applied their edits, but their own " \
               "report says a check they ran fails: apply-batch-0" in out
        assert "apply-batch-0: ok (applied, its check fails)" in out
        assert "apply-batch-1: ok (applied)," in out
        assert "Nothing ran this repository's tests" in out

    def test_a_clean_run_reads_as_before(self, tmp_path, capsys):
        rows = [dict(self.ROWS[1])]
        orch = self._orchestrator(tmp_path, rows)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "## Check Findings" not in text and "its check fails" not in text
        orch.print_write_outcome_summary()
        assert "Not a clean finish" not in capsys.readouterr().out


@pytest.mark.parametrize("core", ["implement.md", "apply-code-fixes.md"])
def test_the_native_instructions_record_and_report_a_batch_s_own_check(core):
    """The native leg's half: the host records the reply's `check` with
    `util batch-done --check`, lists a failing one under Check Findings and
    says so in its report, and names the setting nothing ran the tests for."""
    text = (Path(__file__).parent.parent / "instructions" / core).read_text(
        encoding="utf-8")
    assert "`[BATCH_CHECK_FAILED]`" in text
    assert "`--check findings` and one `--finding \"{finding}\"` per entry" in text
    assert "## Check Findings" in text
    assert "`verify_command` under `driver:` in `.multi-llm/providers.yaml`" in text


@pytest.mark.parametrize("core", ["implement.md", "apply-code-fixes.md"])
def test_the_native_host_reads_a_clear_pass_as_a_pass(core):
    """Fix round 15: the executor reads `passed: true` as a pass whatever is
    listed beside it, so the native host is told the same."""
    text = (Path(__file__).parent.parent / "instructions" / core).read_text(
        encoding="utf-8")
    assert ('add `--check passed` when it says `"passed": true` (what it lists '
            'beside a pass, such as `npm test: 12 passed`, is not a failing '
            'check)') in text
