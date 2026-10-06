"""The counts implement prints: what a run dispatches, on both legs.

Fix round 18, from live run 7 on the check-gate fixture (a check T002 that
reports problems, T007 behind it, T005 refused by the write policy, T006
after T005):

* Claude Code stop 01 (#2): the native emit said "Organized into 3 execution
  batches: 5 task(s) to dispatch (T007 waits on a check's verdict)" and "pass
  --total-batches 5". The ``[ETA]`` line counts one ``util metrics record``
  per dispatched task, so five was the right unit, but T002 reported problems,
  T007 was held, and the line stopped at "batch 4/5" for good. The executor's
  own ``[ETA]`` lines did the same with an accepted or a held task.
* Codex and OpenCode stop resumes (#3): every resume printed "Organized into 4
  execution batches: 4 task(s) to dispatch" — also when it dispatched nothing
  (01b, 01c) and when it dispatched two (fix 02) — and the status poll read
  "implement 1/4 batches ok, 0/4 items applied" for a resume that ran nothing.

These tests drive the real orchestrators against the mock runner, through the
fixture of ``test_check_gate_fixture.py``.
"""

from __future__ import annotations

import json
import re

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    ALLOW,
    DENY,
    FINDINGS,
    _check,
    _first_run,
    _policy,
    repo,
)
from tests.test_implement_stop_account import _dispatched_titles, _implement
from utils.state_manager import get_or_create_state

ETA = re.compile(r"\[ETA\] implement: batch (\d+)/(\d+) done")


def _etas(text: str) -> list:
    """Every ``[ETA] implement: batch N/M`` line's ``(N, M)``, in order."""
    return [(int(a), int(b)) for a, b in ETA.findall(text)]


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _emit(repo, capsys, *flags):
    import implement_orchestrator

    assert implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(open(path, encoding="utf-8").read()), out


class TestTheNativeEtaEndsAtTheTasksDispatched:
    """The host passes the emit's `tasks_to_dispatch` to `util metrics start`
    and `record`; `util batch-done` takes out of that total each handed-out
    task its record keeps from being dispatched."""

    def _start(self, capsys, handoff) -> str:
        state_file = handoff["state_file"]
        code, _out, err = _util(
            capsys, "metrics", "start", "--state-file", state_file,
            "--phase", "implement", "--total-batches", str(handoff["tasks_to_dispatch"]),
        )
        assert code == 0, err
        return state_file

    def _done(self, repo, capsys, task_id, *extra) -> str:
        code, out, err = _util(
            capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", task_id, *extra,
        )
        assert code == 0, err
        return out

    def _record(self, capsys, state_file, handoff, task_id, index) -> tuple:
        code, _out, err = _util(
            capsys, "metrics", "record", "--state-file", state_file,
            "--phase", "implement", "--label", f"{task_id}: task",
            "--subagent-type", "agent", "--duration-ms", "1000",
            "--total-batches", str(handoff["tasks_to_dispatch"]),
            "--batch-index", str(index),
        )
        assert code == 0, err
        return _etas(err)[-1], err

    def test_a_task_a_check_holds_leaves_the_total(self, repo, capsys):
        handoff, out = _emit(repo, capsys)
        assert handoff["tasks_to_dispatch"] == 5
        assert "pass --total-batches 5" in out
        state_file = self._start(capsys, handoff)

        self._done(repo, capsys, "T001", "--status", "completed")
        assert self._record(capsys, state_file, handoff, "T001", 1)[0] == (1, 5)
        blocked = self._done(
            repo, capsys, "T002", "--status", "completed", "--check", "findings",
            "--finding", FINDINGS[0])
        assert "[CHECK_BLOCKED] T007" in blocked
        assert self._record(capsys, state_file, handoff, "T002", 2)[0] == (2, 4)
        self._done(repo, capsys, "T003", "--status", "completed")
        assert self._record(capsys, state_file, handoff, "T003", 3)[0] == (3, 4)
        self._done(repo, capsys, "T004", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T004", 4)
        assert last == (4, 4), err
        assert "ALL BATCHES COMPLETE" in err

        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 4
        assert progress["withdrawn"] == {"T007": "T002"}

    def test_the_tasks_after_a_failed_one_leave_it_too(self, repo, capsys):
        _policy(repo, ALLOW)
        handoff, _out = _emit(repo, capsys)
        assert handoff["tasks_to_dispatch"] == 7
        state_file = self._start(capsys, handoff)
        for index, task_id in enumerate(("T001", "T002", "T003", "T004"), 1):
            extra = ("--check", "passed") if task_id == "T002" else ()
            self._done(repo, capsys, task_id, "--status", "completed", *extra)
            self._record(capsys, state_file, handoff, task_id, index)
        self._done(repo, capsys, "T005", "--status", "failed", "--reason", "no stamp")
        assert self._record(capsys, state_file, handoff, "T005", 5)[0] == (5, 6)
        self._done(repo, capsys, "T007", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T007", 6)
        assert last == (6, 6) and "ALL BATCHES COMPLETE" in err, err

    def test_a_task_write_prompt_refused_leaves_it_with_its_dependents(
        self, repo, capsys,
    ):
        _policy(repo, ALLOW)
        handoff, _out = _emit(repo, capsys)
        state_file = self._start(capsys, handoff)
        _policy(repo, DENY)
        for index, task_id in enumerate(("T001", "T002", "T003", "T004"), 1):
            extra = ("--check", "passed") if task_id == "T002" else ()
            self._done(repo, capsys, task_id, "--status", "completed", *extra)
            self._record(capsys, state_file, handoff, task_id, index)
        code, _out, err = _util(
            capsys, "write-prompt", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T005")
        assert code == 3 and "[WRITE_PROMPT_REFUSED]" in err
        self._done(
            repo, capsys, "T005", "--status", "failed", "--reason",
            "refused before dispatch: the write policy (driver.write_paths) "
            "does not allow a file it declares")
        self._done(repo, capsys, "T007", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T007", 5)
        assert last == (5, 5) and "ALL BATCHES COMPLETE" in err, err

    def test_recording_the_check_again_puts_its_held_tasks_back(self, repo, capsys):
        handoff, _out = _emit(repo, capsys)
        self._start(capsys, handoff)
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 4
        self._done(repo, capsys, "T002", "--status", "completed", "--check", "passed")
        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 5
        assert "withdrawn" not in progress

    def test_without_metrics_start_nothing_is_stored(self, repo, capsys):
        _emit(repo, capsys)
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        assert "metrics_progress" not in get_or_create_state(repo.plan).state


#: A task held by two causes at once: T004 waits on the check T002 and on
#: the code task T003 (fix round 19, review of round 18, #3).
TWO_CAUSES = """# Two causes

## Overview
A task after a check and after a code task.

## Tasks

### T001: Edit: create the module
Create it.
- Depends on: none
- Files to modify: src/main.py

### T002: Check the module
Run the release check. Read-only.
- Depends on: T001
- Files to modify: none

### T003: Edit: add the greeting module
Add the greeting.
- Depends on: T001
- Files to modify: src/main.py

### T004: Edit: wire both together
Wire them.
- Depends on: T002, T003
- Files to modify: src/main.py
"""


class TestATaskKeptOutForTwoReasons:
    """Review of round 18, #3: `util batch-done` put a task back in the
    `[ETA]` total when the record that took it out was replaced, although a
    second cause still kept it from running, so the native line ended
    "N/N+1" again."""

    @staticmethod
    def _repo(tmp_path, monkeypatch):
        from tests.test_implement_stop_account import make_repo

        return make_repo(tmp_path, monkeypatch, TWO_CAUSES)

    @staticmethod
    def _progress(repo) -> tuple:
        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        return progress["total_batches"], progress.get("withdrawn")

    def _started(self, repo, capsys):
        handoff, _out = _emit(repo, capsys)
        assert handoff["tasks_to_dispatch"] == 4
        code, _out, err = _util(
            capsys, "metrics", "start", "--state-file", handoff["state_file"],
            "--phase", "implement", "--total-batches", "4")
        assert code == 0, err
        self._done(repo, capsys, "T001", "--status", "completed")

    @staticmethod
    def _done(repo, capsys, task_id, *extra):
        code, out, err = _util(
            capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", task_id, *extra)
        assert code == 0, err
        return out

    def test_a_failed_task_retried_leaves_it_out_while_the_check_holds_it(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        self._started(repo, capsys)
        self._done(repo, capsys, "T003", "--status", "failed", "--reason", "boom")
        assert self._progress(repo) == (3, {"T004": "T003"})
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        assert self._progress(repo) == (3, {"T004": "T003"})
        # The retried T003 succeeds; T002's findings still hold T004.
        self._done(repo, capsys, "T003", "--status", "completed")
        assert self._progress(repo) == (3, {"T004": "T002"})
        # The check passes on a re-run: nothing keeps T004 out any more.
        self._done(repo, capsys, "T002", "--status", "completed", "--check", "passed")
        assert self._progress(repo) == (4, None)

    def test_a_check_that_passes_leaves_it_out_while_its_other_upstream_failed(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        self._started(repo, capsys)
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        assert self._progress(repo) == (3, {"T004": "T002"})
        self._done(repo, capsys, "T003", "--status", "failed", "--reason", "boom")
        assert self._progress(repo) == (3, {"T004": "T002"})
        # The check passes; T003 is still failed, so the host skips T004.
        self._done(repo, capsys, "T002", "--status", "completed", "--check", "passed")
        assert self._progress(repo) == (3, {"T004": "T003"})


class TestTheExecutorEtaEndsAtTheTasksDispatched:

    def test_the_accept_path_counts_only_the_released_task(self, repo, capsys):
        """Codex and OpenCode accept 02: "batch 1/2 … remaining", the
        accepted check T002 counted in the total."""
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        text = capsys.readouterr()
        text = text.out + text.err
        assert _dispatched_titles(repo) == ["Edit: write the release note"]
        assert _etas(text) == [(1, 1)], text
        assert "ALL BATCHES COMPLETE" in text

    def test_a_task_a_check_holds_leaves_the_total(self, repo, capsys):
        _policy(repo, ALLOW)
        _check(repo, passed=False)
        assert _implement(repo) == 3
        text = capsys.readouterr()
        text = text.out + text.err
        etas = _etas(text)
        assert etas[-1] == (6, 6), etas
        assert etas[0] == (1, 7), etas  # T007 counted until T002 reported
        assert "ALL BATCHES COMPLETE" in text


def _organized(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith("Organized into ")]


def _run(repo, capsys, *flags) -> tuple:
    code = _implement(repo, *flags)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestTheExecutorSaysWhatItsRunDispatches:
    """Run 7 #3: the executor leg's line counted every task a resume took up.
    It is the scheduler's now, printed once per run, and counts what the run
    dispatches, beside the `resume: kept …` line that names what it kept."""

    def test_the_first_run_counts_the_tasks_before_the_refused_one(self, repo, capsys):
        text = _first_run(repo, capsys)
        # T005 is refused in its batch and the run stops there: T006 and
        # T007 (in later batches of this decomposition) are never reached.
        assert _organized(text) == [
            "Organized into 3 execution batches: 4 task(s) to dispatch"]
        assert text.count("[ETA] implement: batch ") == 4

    def test_a_resume_that_changes_nothing_says_it_dispatches_nothing(
        self, repo, capsys,
    ):
        """01b (`--resume`) and 01c (plain): "4 task(s) to dispatch" before
        "dispatched 0"."""
        _first_run(repo, capsys)
        for flags in (("--resume",), ()):
            code, text = _run(repo, capsys, *flags)
            assert code == 3, text
            assert _organized(text) == [
                "Organized into 0 execution batches: 0 task(s) to dispatch"], text
            assert "dispatched 0; refused before dispatch 1 (T005)" in text
        assert _dispatched_titles(repo) == []

    def test_the_fix_path_counts_the_check_and_its_waiter(self, repo, capsys):
        """Fix 02: the files changed, so T002 runs again and T007 waits on
        it; T005 and T006 are skipped by the answer. It said 4, ran 2."""
        _first_run(repo, capsys)
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "retries = 3\n", encoding="utf-8")
        _check(repo, passed=True)
        code, text = _run(repo, capsys, "--resume", "--human-task", "T005=skip")
        assert code == 0, text
        assert _organized(text) == [
            "Organized into 2 execution batches: 2 task(s) to dispatch "
            "(T007 waits on a check's verdict)"], text
        assert "dispatched 2" in text
        assert _dispatched_titles(repo) == [
            "Check the module", "Edit: write the release note"]

    def test_the_accept_path_counts_the_released_task(self, repo, capsys):
        _first_run(repo, capsys)
        code, text = _run(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip")
        assert code == 0, text
        assert _organized(text) == [
            "Organized into 1 execution batch: 1 task(s) to dispatch"], text
        assert "dispatched 1" in text

    def test_a_policy_widened_since_counts_everything_left(self, repo, capsys):
        _first_run(repo, capsys)
        _policy(repo, ALLOW)
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "retries = 3\n", encoding="utf-8")
        _check(repo, passed=True)
        code, text = _run(repo, capsys, "--resume")
        assert code == 0, text
        assert _organized(text) == [
            "Organized into 4 execution batches: 4 task(s) to dispatch "
            "(T007 waits on a check's verdict)"], text
        assert len(_dispatched_titles(repo)) == 4


#: Three tasks of one batch on disjoint files, the middle one on a file the
#: policy denies, and a fourth after them (fix round 19, review of round 18,
#: #4).
PARALLEL = """# Parallel

## Overview
Three independent edits and one after them.

## Tasks

### T001: Edit: add the first module
Add it.
- Depends on: none
- Files to modify: src/first.py

### T002: Edit: add the vendored stamp
Add it.
- Depends on: none
- Files to modify: src/vendor/stamp.js

### T003: Edit: add the third module
Add it.
- Depends on: none
- Files to modify: src/third.py

### T004: Edit: wire them together
Wire them.
- Depends on: T001, T002, T003
- Files to modify: src/main.py
"""


class TestARefusedTaskInAGroupThatRunsTogether:
    """Review of round 18, #4: with `write_ceiling()` above 1 a batch's
    disjoint tasks are dispatched together (`admit_disjoint_groups`) and the
    loop stops only once the group settled, so the siblings of a task the
    write policy refuses run; the `Organized into …` line stopped counting
    at the refused task."""

    def test_the_line_counts_the_siblings_dispatched_beside_it(
        self, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_implement_stop_account import make_repo
        from utils.write_batch_scheduler import WriteBatchScheduler

        repo = make_repo(tmp_path, monkeypatch, PARALLEL)
        _policy(repo, DENY)
        monkeypatch.setattr(WriteBatchScheduler, "write_ceiling", lambda self: 3)
        code, text = _run(repo, capsys)
        assert code == 3, text
        assert "T002 refused before dispatch" in text, text
        dispatched = sorted(set(_dispatched_titles(repo)))
        assert dispatched == [
            "Edit: add the first module", "Edit: add the third module"], dispatched
        assert _organized(text) == [
            "Organized into 1 execution batch: 2 task(s) to dispatch"], text

    def test_one_at_a_time_the_line_still_stops_at_the_refused_task(
        self, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_implement_stop_account import make_repo

        repo = make_repo(tmp_path, monkeypatch, PARALLEL)
        _policy(repo, DENY)
        code, text = _run(repo, capsys)
        assert code == 3, text
        assert sorted(set(_dispatched_titles(repo))) == ["Edit: add the first module"]
        assert _organized(text) == [
            "Organized into 1 execution batch: 1 task(s) to dispatch"], text


class TestAFailedCheckAndAManualTask:
    """The stop-account fixture: T002 a check that fails, T004 after it, T005
    a manual task the executor never runs."""

    def test_the_manual_task_is_not_counted_and_the_unchanged_failure_stops(
        self, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_implement_stop_account import _check_fails, make_repo

        stop = make_repo(tmp_path, monkeypatch)
        _check_fails(stop)
        code, text = _run(stop, capsys)
        assert code == 3, text
        assert _organized(text) == [
            "Organized into 3 execution batches: 4 task(s) to dispatch "
            "(T004 waits on a check's verdict)"], text
        stop.log.unlink(missing_ok=True)
        code, text = _run(stop, capsys, "--resume")
        assert code == 3, text
        # T002 failed on these very files: not run again, and the run stops.
        assert _organized(text) == [
            "Organized into 0 execution batches: 0 task(s) to dispatch"], text
        assert _dispatched_titles(stop) == []


class TestTheStatusProgressLineNamesWhatAResumeKept:
    """Run 7 #3 (OpenCode 01b/01c): "implement 1/4 batches ok, 0/4 items
    applied" for a resume that dispatched nothing — the one batch "ok" was
    the check T002 an earlier run finished, kept on unchanged files."""

    def _progress(self, repo, capsys) -> str:
        code = multi_llm.main([
            "status", str(repo.plan), "--view", "run", "--phase", "implement"])
        captured = capsys.readouterr()
        line = [
            l for l in captured.out.splitlines() if "progress:" in l and "implement " in l
        ]
        assert line, (code, captured.out)
        result = json.loads(
            [l for l in captured.out.splitlines()
             if l.startswith("MULTI_LLM_RESULT:")][-1].split(" ", 1)[1])
        return line[0], result

    def test_a_resume_that_kept_the_check_says_so(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo, "--resume") == 3
        capsys.readouterr()
        line, result = self._progress(repo, capsys)
        # T007, held by T002's findings again, is out of the total (fix
        # round 19), as it is out of the `[ETA]` line's.
        assert (
            "implement 1/3 batches ok (1 kept from an earlier run: T002), "
            "0/3 items applied"
        ) in line, line
        # The OpenCode route shows the poll's own rendered lines.
        assert any(
            "(1 kept from an earlier run: T002)" in text
            for text in result.get("progress_lines") or []
        ), result

    def test_a_first_run_has_nothing_kept_to_name(self, repo, capsys):
        _first_run(repo, capsys)
        line, _result = self._progress(repo, capsys)
        assert "kept from an earlier run" not in line, line


class TestTheNativeResumeEmitCountsWhatItHandsOut:
    """The native leg's resume emits already agreed in run 7 (01b 0/0/0, fix
    02 2/2/2, accept 02 1/1/1); pinned beside the executor's, so both legs
    say the same for the same story."""

    def _record_first_run(self, repo, capsys):
        _emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                  task_id, "--status", "completed")
        _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
              "--status", "completed", "--check", "findings", "--finding", FINDINGS[0])

    def test_an_unchanged_resume_hands_out_nothing_and_says_so(self, repo, capsys):
        self._record_first_run(repo, capsys)
        handoff, out = _emit(repo, capsys, "--resume")
        assert _organized(out) == [
            "Organized into 0 execution batches: 0 task(s) to dispatch"], out
        assert handoff["tasks_to_dispatch"] == 0
        assert "pass --total-batches 0" in out
        assert "Nothing to dispatch" in out


def _status_progress(repo, capsys) -> tuple:
    """``status --view run``'s implement progress line, and the result line's
    ``progress_lines`` (what the OpenCode poll relays)."""
    code = multi_llm.main([
        "status", str(repo.plan), "--view", "run", "--phase", "implement"])
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "progress:" in l and "implement " in l]
    assert lines, (code, out)
    result = json.loads(
        [l for l in out.splitlines()
         if l.startswith("MULTI_LLM_RESULT:")][-1].split(" ", 1)[1])
    return lines[0], result.get("progress_lines") or []


class TestEverySurfaceCountsTheSameTasks:
    """Fix round 19 (review of round 18, #1): the executor's `[ETA]` line left
    out a task a check holds (`_eta_total`) while the run mirror kept it, so
    one run read "batch 6/6 … ALL BATCHES COMPLETE" in its log and
    "implement 6/7 batches ok, 5/7 items applied" on `status --view run` and
    the OpenCode poll line."""

    def test_a_task_a_check_holds_is_out_of_both_totals(self, repo, capsys):
        _policy(repo, ALLOW)
        _check(repo, passed=False)
        assert _implement(repo) == 3
        text = capsys.readouterr()
        assert _etas(text.out + text.err)[-1] == (6, 6)
        line, polled = _status_progress(repo, capsys)
        assert "implement 6/6 batches ok, 5/6 items applied" in line, line
        # Still named on the line, with the check that holds it.
        assert "not run (upstream check T002 reported problems) T007" in line, line
        assert any("implement 6/6 batches ok" in row for row in polled), polled
        mirror = json.loads((repo.phase_dir / ".status.json").read_text())
        counts = mirror["progress"]["write_subagents"]
        assert (counts["total"], counts["items_total"]) == (6, 6), counts

    def test_a_manual_task_answered_done_is_out_of_both_totals(
        self, tmp_path, monkeypatch, capsys,
    ):
        """No runner runs a manual task: a clean run whose manual T005 was
        answered done read "implement 4/5 batches ok" beside "batch 4/4 …
        ALL BATCHES COMPLETE"."""
        import implement_orchestrator
        from tests.test_implement_stop_account import _check_passes, make_repo

        stop = make_repo(tmp_path, monkeypatch)
        _check_passes(stop)
        assert implement_orchestrator.main([
            "--plan-file", str(stop.plan), "--no-confirm", "--exec-subagents",
            "--force-strategy", "pause-and-ask", "--human-task", "T005=done",
        ]) == 0
        text = capsys.readouterr()
        assert _etas(text.out + text.err)[-1] == (4, 4)
        line, _polled = _status_progress(stop, capsys)
        assert "implement 4/4 batches ok, 3/4 items applied" in line, line
