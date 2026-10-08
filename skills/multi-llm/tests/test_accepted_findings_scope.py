"""What an accepted check's re-run may hide, and where it is on record.

Fix round 22 took out of a released task's own-check findings those that only
repeat what the user accepted for a check (`split_accepted_findings`, live
run 9, OpenCode accept 02). The review of that round (fix round 23, records
#1-#3 and #6) found it hid more than that:

* a count after `exited with` / `returned` was read as an exit code and
  dropped, so "exited with 9 failures in tests/test_auth.py" matched an
  accepted "exited with 3 failures in tests/test_auth.py";
* it ran on every implement task's own check, also a task that does not
  depend on the accepted check (its prompt never said the problems were
  accepted) and a task whose own job is to fix them;
* with two accepted findings, a finding straddling both matched the two read
  one after the other, and any slice of three words from the middle of one
  matched it;
* the split findings lived only in the result row and the printed line:
  `state.json` and `status` showed the task as plainly `completed`.

Now numbers are words (only a whole exit clause is set aside), the split
runs only for a task whose prompt carried the accepted-checks section and
whose own text does not name the check's command or ask to fix its findings,
a finding matches one accepted finding (all of it, or a run of its words from
its start), and the split is on record in `state.json`'s
`task_batch_check_accepted` on both legs, which `status` names.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401 - `repo` is a fixture
    ALLOW,
    CHECK_GATE,
    DENY,
    NOTE,
    PLAN,
    _policy,
    _statuses,
)
from tests.test_implement_stop_account import _implement, make_repo  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_ACCEPTED_MARKER,
    BATCH_CHECK_MARKER,
    accepted_findings_for,
    split_accepted_findings,
)

#: T002's finding as OpenCode's check recorded it in run 9.
ACCEPTED = "node scripts/check.mjs: retries must be 3 (found 2)"
#: What a task's own check reported when it ran the same check again.
RERUN = "node scripts/check.mjs: check failed: retries must be 3 (found 2) (exit 1)"
#: The Claude Code host's wording of the same T002 finding (run 9).
NATIVE = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must be "
    "3 (found 2)'. src/config.json has retries=2 (set by T001), but the check "
    "requires 3."
)
#: The native host's run-11 report: same diagnostic, with its attribution.
ANNOTATED = (
    "node scripts/check.mjs exited 1: 'check failed: retries must be 3 (found 2)' "
    "(the problem the user accepted from T002)"
)


# -- #1: numbers are words ----------------------------------------------------


class TestNumbersAreWords:
    """Only a whole exit clause is set aside: never a count."""

    @pytest.mark.parametrize("accepted, rerun", [
        # The reviewer's scenario.
        ("pytest tests/: exited with 3 failures in tests/test_auth.py",
         "pytest tests/: exited with 9 failures in tests/test_auth.py"),
        ("npm test: returned 1 failure in auth.test.js",
         "npm test: returned 4 failures in auth.test.js"),
        ("pytest tests/: exited 1 with 3 failures in tests/test_auth.py",
         "pytest tests/: exited 1 with 9 failures in tests/test_auth.py"),
        ("npm test exited with 2 failing tests in auth.test.js",
         "npm test exited with 5 failing tests in auth.test.js"),
    ])
    def test_another_count_is_never_the_accepted_finding(self, accepted, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [accepted]})
        assert own == [rerun] and matched == {}

    @pytest.mark.parametrize("accepted, rerun", [
        (ACCEPTED, RERUN),
        (ACCEPTED, "node scripts/check.mjs exited 1: retries must be 3 (found 2)"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2), exit code 1"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2); exited with status 2"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2) and exited 1"),
        (ACCEPTED, "node scripts/check.mjs returned 1: retries must be 3 (found 2)"),
        (NATIVE, RERUN),
        ("pytest tests/: exited with 3 failures in tests/test_auth.py",
         "pytest tests/: exited with 3 failures in tests/test_auth.py (exit code 1)"),
    ])
    def test_a_whole_exit_clause_is_still_set_aside(self, accepted, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [accepted]})
        assert own == [] and matched == {"T002": [rerun]}


# -- #3: one accepted finding, from its start ----------------------------------


class TestOneAcceptedFinding:

    def test_a_finding_straddling_two_accepted_findings_is_the_tasks_own(self):
        accepted = {"T002": [
            "lint: unused variable x in a.js", "lint: missing semicolon in b.js",
        ]}
        own, matched = split_accepted_findings(["lint: x in a.js missing"], accepted)
        assert own == ["lint: x in a.js missing"] and matched == {}

    def test_a_slice_from_the_middle_of_an_accepted_finding_is_the_tasks_own(self):
        accepted = {"T002": ["eslint src/: no-unused-vars x in src/a.js line 10 col 4"]}
        own, matched = split_accepted_findings(["eslint src/: x in src/a.js"], accepted)
        assert own == ["eslint src/: x in src/a.js"] and matched == {}

    def test_the_start_of_an_accepted_finding_still_matches(self):
        accepted = {"T002": ["eslint src/: no-unused-vars x in src/a.js line 10 col 4"]}
        rerun = "eslint src/: no-unused-vars x in src/a.js"
        assert split_accepted_findings([rerun], accepted) == ([], {"T002": [rerun]})

    def test_each_of_two_accepted_findings_matches_on_its_own(self):
        accepted = {"T002": [
            "npm test: test parses the config failed",
            "npm test: test greets the user failed",
        ]}
        found = ["npm test: test greets the user failed (exit 1)",
                 "npm test: test parses the config failed"]
        assert split_accepted_findings(found, accepted) == ([], {"T002": found})


class TestAcceptedFindingAnnotations:

    @pytest.mark.parametrize("finding", [
        ANNOTATED,
        ANNOTATED.replace("the problem", "the findings") + ".",
    ])
    def test_the_attribution_keeps_the_original_report(self, finding):
        assert split_accepted_findings([finding], {"T002": [NATIVE]}) == (
            [], {"T002": [finding]})

    @pytest.mark.parametrize("finding", [
        ANNOTATED.replace("found 2", "found 4"),
        ANNOTATED.replace("must be 3", "must be 9"),
        ANNOTATED.replace("scripts/check.mjs", "scripts/other.mjs"),
        ANNOTATED.replace("retries must be 3", "retries must not be 3"),
        ANNOTATED.replace("from T002", "from T003"),
        ANNOTATED + "; also found a missing token",
        ANNOTATED.replace(" (the problem", "; missing token (the problem"),
        ANNOTATED.replace("from T002)", "from T002; missing token)"),
        ANNOTATED.replace("the user accepted", "the user has not accepted"),
    ])
    def test_new_or_unattributed_failures_remain_the_tasks_own(self, finding):
        assert split_accepted_findings([finding], {"T002": [NATIVE]}) == (
            [finding], {})

    def test_citing_another_accepted_check_cannot_match_the_first(self):
        finding = ANNOTATED.replace("from T002", "from T003")
        accepted = {"T002": [NATIVE], "T003": ["npm test: missing token"]}
        assert split_accepted_findings([finding], accepted) == ([finding], {})

    def test_matching_shared_diagnostics_are_attributed_to_the_cited_check(self):
        finding = ANNOTATED.replace("from T002", "from T003")
        accepted = {"T002": [NATIVE], "T003": [NATIVE]}
        assert split_accepted_findings([finding], accepted) == ([], {"T003": [finding]})

    def test_each_new_finding_is_still_reported(self):
        extra = "npm test: missing token"
        assert split_accepted_findings([ANNOTATED, extra], {"T002": [NATIVE]}) == (
            [extra], {"T002": [ANNOTATED]})


# -- #2: which task's check may repeat the accepted findings ------------------


def _task(task_id, title, description, *, depends_on=(), criteria=()):
    """A materialized implement task, its prompt in the template's shape."""
    criteria_block = (
        "## Acceptance Criteria\nVerify these.\n\n"
        + "".join(f"- [ ] {item}\n" for item in criteria)
        if criteria else ""
    )
    return {
        "task_id": task_id, "task_title": title,
        "depends_on": list(depends_on), "declared_files": ["src/main.py"],
        "prompt": (
            "You are implementing a specific task from a larger plan.\n\n"
            "## Plan Context\nThe release check `node scripts/check.mjs` gates "
            "the greeting module; fix nothing it reports by hand.\n\n"
            f"## Task\nTitle: {title}\nDescription: {description}\n\n"
            "## Relevant Files to Review\nsrc/main.py\n\n"
            f"{criteria_block}\n## Changes Applied in Prior Batches\n"
            "{prior_changes_context}\n"
        ),
    }


def _state():
    return {
        "task_status": {"T002": "completed", "T003": "completed"},
        "task_status_reasons": {
            "T002": "accepted — the user answered --human-task T002=done"},
        "task_check_findings": {"T002": [ACCEPTED]},
        "implement_task_depends_on": {
            "T001": [], "T002": ["T001"], "T003": ["T001"],
        },
    }


class TestWhichTask:
    """`accepted_findings_for`: the accepted findings a task's own check may
    repeat — those of the accepted checks its prompt was told about, unless
    its own text makes them its job."""

    #: Run 9's T007, word for word: it names T002 and says "resolved", and
    #: is the task the split is for.
    RELEASE_NOTE = (
        "Create `docs/release.md` with release notes for check-gate 1.0: a "
        "first line naming the version, then one line stating the outcome of "
        "the release check (T002) as recorded: that it passed, or the problems "
        "it reported and how they were resolved."
    )

    def test_a_released_task_may_repeat_them(self):
        task = _task("T007", "Write the release note", self.RELEASE_NOTE,
                     depends_on=["T002"],
                     criteria=["docs/release.md states the release check's outcome truthfully."])
        assert accepted_findings_for([task], task, _state()) == {"T002": [ACCEPTED]}

    def test_a_task_that_does_not_depend_on_the_check_may_not(self):
        task = _task("T006", "Print the build stamp", "Print it.", depends_on=["T003"])
        assert accepted_findings_for([task], task, _state()) == {}

    @pytest.mark.parametrize("title, description, criteria", [
        ("Make the release check pass", "Change src/main.py.", ()),
        ("Fix the release check", "Fix the problems T002 reported.", ()),
        ("Run the suite and fix failures", "Run it; fix what fails.", ()),
        ("Set retries", "Change src/main.py so `node scripts/check.mjs` reports nothing.", ()),
        ("Set retries", "Change src/main.py.", ("`node scripts/check.mjs` exits 0.",)),
        ("Repair the config", "Resolve the findings of T002.", ()),
    ])
    def test_a_task_whose_job_is_the_check_may_not(self, title, description, criteria):
        task = _task("T008", title, description, depends_on=["T002"], criteria=criteria)
        assert accepted_findings_for([task], task, _state()) == {}

    def test_a_task_told_not_to_fix_them_may_repeat_them(self):
        task = _task("T007", "Write the release note",
                     "Write the note. Do not fix the problems T002 reported.",
                     depends_on=["T002"])
        assert accepted_findings_for([task], task, _state()) == {"T002": [ACCEPTED]}

    def test_the_plan_context_naming_the_command_is_not_the_tasks_own_text(self):
        # `_task`'s plan context names `node scripts/check.mjs` and says "fix".
        task = _task("T007", "Write the release note", "Write the note.",
                     depends_on=["T002"])
        assert "node scripts/check.mjs" in task["prompt"]
        assert accepted_findings_for([task], task, _state()) == {"T002": [ACCEPTED]}

    def test_through_a_finished_task(self):
        """Transitive, over the recorded decomposition, as the section is."""
        task = _task("T009", "Write the changelog", "Write it.", depends_on=["T007"])
        state = _state()
        state["implement_task_depends_on"]["T007"] = ["T002"]
        assert accepted_findings_for([task], task, state) == {"T002": [ACCEPTED]}


# -- #2 and #6 on the executor leg --------------------------------------------

#: The check-gate fixture plus a task whose own job is T002's check.
FIXER = "Edit: make the release check pass"
SCOPE_PLAN = PLAN + f"""
### T008: {FIXER}
Change src/main.py so `node scripts/check.mjs` reports nothing.
- Depends on: T002
- Files to modify: src/main.py
"""


@pytest.fixture
def scope_repo(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, SCOPE_PLAN)
    _policy(obj, DENY)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 9)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    yield obj
    registry._config = None
    registry._config_key = None


def _every_edit_reports(repo, findings) -> None:
    """Every code task's reply carries the same own `check`."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
        {"passed": False, "findings": list(findings)}))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Title: Edit:")


def _first_stop(repo, capsys) -> None:
    """T002 reports ACCEPTED, T005 is refused by the write policy, T007 and
    T008 wait on T002."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": [ACCEPTED]}))
    assert _implement(repo) == 3
    capsys.readouterr()
    repo.log.unlink(missing_ok=True)


def _lines(text: str, marker: str) -> dict:
    return {
        line.split(":", 1)[0].split()[-1]: line
        for line in text.splitlines() if line.startswith(marker)
    }


def _status_text(repo, capsys, *host) -> str:
    from utils import run_status

    capsys.readouterr()
    run_status.main(["--plan-file", str(repo.plan), *host])
    captured = capsys.readouterr()
    return captured.out + captured.err


class TestTheExecutorLeg:

    def test_only_the_released_task_repeats_the_accepted_findings(self, scope_repo, capsys):
        repo = scope_repo
        _first_stop(repo, capsys)
        _policy(repo, ALLOW)
        _every_edit_reports(repo, [RERUN])
        rc = _implement(repo, "--resume", "--human-task", "T002=done",
                        "--human-task", "T005=retry")
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert rc == 0, text[-3000:]
        statuses = _statuses(repo)
        assert all(statuses[t] == "completed" for t in ("T005", "T006", "T007", "T008"))
        # T007 was told the findings were accepted; T005 and T006 were not
        # (they do not depend on T002), and T008's job is T002's check.
        assert set(_lines(text, BATCH_CHECK_ACCEPTED_MARKER)) == {"T007"}, text[-3000:]
        assert set(_lines(text, BATCH_CHECK_MARKER)) == {"T005", "T006", "T008"}, text[-3000:]
        state = get_or_create_state(repo.plan).state
        assert state["task_batch_check"] == {
            "T005": [RERUN], "T006": [RERUN], "T008": [RERUN]}
        # #6: the accepted re-run is on record beside the check findings.
        assert state["task_batch_check_accepted"] == {"T007": {"T002": [RERUN]}}
    def test_status_names_the_accepted_rerun(self, scope_repo, capsys):
        repo = scope_repo
        _first_stop(repo, capsys)
        _policy(repo, ALLOW)
        _every_edit_reports(repo, [RERUN])
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", f"Title: {NOTE}\n")
        assert _implement(repo, "--resume", "--human-task", "T002=done",
                          "--human-task", "T005=retry") == 0
        capsys.readouterr()
        text = _status_text(repo, capsys)
        clause = "completed, its check repeated T002's accepted findings: T007"
        # The run's progress line and the workflow trailer.
        assert text.count(clause) == 2, text
        assert "own check fails" not in text, text


# -- #2 and #6 on the native leg ----------------------------------------------


class TestTheNativeLeg:

    def _util(self, capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _accept(self, repo, capsys) -> str:
        """The native first run: T002 reports NATIVE, T005-T008 are left;
        then the accept emit hands them out."""
        import implement_orchestrator

        _policy(repo, ALLOW)
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(capsys, "batch-done", "--phase-dir", phase,
                              "--task-id", task_id, "--status", "completed")[0] == 0
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T002", "--status", "completed", "--check", "findings",
                          "--finding", NATIVE)[0] == 0
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", "--resume",
             "--human-task", "T002=done"]
        ) == 0
        capsys.readouterr()
        return phase

    def _own_check(self, capsys, phase, task_id, finding=RERUN):
        return self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", task_id,
            "--status", "completed", "--own-check", "findings", "--finding", finding)

    @pytest.mark.parametrize("finding", [RERUN, ANNOTATED])
    def test_batch_done_splits_only_the_released_tasks_check(self, scope_repo, capsys, finding):
        repo = scope_repo
        phase = self._accept(repo, capsys)
        for task_id in ("T005", "T006", "T008"):
            code, out, err = self._own_check(capsys, phase, task_id, finding)
            assert code == 0, err
            assert BATCH_CHECK_ACCEPTED_MARKER not in out, (task_id, out)
            assert any(line.startswith(f"{BATCH_CHECK_MARKER} {task_id}:")
                       for line in out.splitlines()), (task_id, out)
        code, out, err = self._own_check(capsys, phase, "T007", finding)
        assert code == 0, err
        assert any(line.startswith(f"{BATCH_CHECK_ACCEPTED_MARKER} T007:")
                   for line in out.splitlines()), out
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_batch_check"] == {
            "T005": [finding], "T006": [finding], "T008": [finding]}
        assert state["task_batch_check_accepted"] == {"T007": {"T002": [finding]}}

    def test_status_names_the_accepted_rerun(self, scope_repo, capsys):
        repo = scope_repo
        phase = self._accept(repo, capsys)
        assert self._own_check(capsys, phase, "T007")[0] == 0
        text = _status_text(repo, capsys, "--host", "claude-code")
        clause = "completed, its check repeated T002's accepted findings: T007"
        # The native implement block and the workflow trailer.
        assert text.count(clause) == 2, text

    def test_a_later_record_of_the_task_clears_it(self, scope_repo, capsys):
        repo = scope_repo
        phase = self._accept(repo, capsys)
        assert self._own_check(capsys, phase, "T007")[0] == 0
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T007", "--status", "completed", "--own-check", "passed")[0] == 0
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "task_batch_check_accepted" not in state
