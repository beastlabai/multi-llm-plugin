#!/usr/bin/env python3
"""A plain ``--implement`` on an implement that started and did not finish.

Codex racing run 6, 06r (2026-10-06): the host left mid-call, the run was
aborted at 3/10 with T001, T002 and T010 recorded completed, and the re-run —
the same ``--implement plans/racing-game.md`` — dispatched those three again.
They came back ``no-op`` (a minute and three model calls) and their "applied"
rows were lost from the summary, because only ``--resume`` skipped a finished
task. A run without ``--resume`` is the start-over spelling only for an
implement that finished; one that started and did not finish now continues as
a resume and says so in one line, and ``--fresh`` is how to start over.

Both legs: the executor (``--exec-subagents``, the tool routes) and the native
emit Claude Code's host dispatches from.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import implement_orchestrator  # noqa: E402
from test_write_executor_orchestrators import (  # noqa: E402,F401  (fixture)
    IMPLEMENT_WRITE_IDS,
    _phase_dir,
    _seed_generate_tasks,
    repo,
)
from utils.write_phase_bridge import WRITE_TASK_FILENAME  # noqa: E402

CONTINUED = "resume: implement started earlier and did not finish"


def _stop_after_t001(repo) -> None:
    """Run 1: T001 applied, T002 a partial application (the loop stops)."""
    _seed_generate_tasks(repo)
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
    assert implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
    ]) == 3


def _second_run(repo, capsys, *flags) -> tuple:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
    log = repo.tmp_path / "calls-2.jsonl"
    repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
    capsys.readouterr()
    code = implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents", *flags,
    ])
    captured = capsys.readouterr()
    return code, captured.out + captured.err, log


class TestTheExecutorLeg:

    def test_a_plain_rerun_continues_and_never_dispatches_a_finished_task(
        self, repo, capsys
    ):
        from utils.state_manager import get_or_create_state

        _stop_after_t001(repo)
        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert "kept 1 task(s) an earlier run finished (T001)" in text, text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T002", "T003"}
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "completed", "T003": "completed",
        }
        assert state.is_phase_completed("implement")

    def test_fresh_starts_over(self, repo, capsys):
        _stop_after_t001(repo)
        code, text, log = _second_run(repo, capsys, "--fresh")
        assert code == 0, text
        assert CONTINUED not in text
        assert "T001" in _ids_in(log.read_text(encoding="utf-8"))

    def test_resume_wins_over_fresh(self, repo, capsys):
        _stop_after_t001(repo)
        code, text, log = _second_run(repo, capsys, "--resume", "--fresh")
        assert code == 0, text
        assert "T001" not in _ids_in(log.read_text(encoding="utf-8"))

    def test_a_finished_implement_still_starts_over(self, repo, capsys):
        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 0
        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED not in text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T001", "T002", "T003"}

    def test_a_task_list_regenerated_since_is_not_continued(self, repo, capsys):
        """The ids an earlier implement recorded may name other tasks once
        generate-tasks wrote a new list: that run starts over, as before."""
        from utils.state_manager import get_or_create_state

        _stop_after_t001(repo)
        state = get_or_create_state(repo.plan)
        state.state.setdefault("phases_completed", {})["generate-tasks"] = (
            "2999-01-01T00:00:00"
        )
        state.save()
        code, text, log = _second_run(repo, capsys)
        assert CONTINUED not in text
        assert "T001" in _ids_in(log.read_text(encoding="utf-8"))

    def test_an_explicit_task_rerun_is_left_alone(self, repo, capsys):
        """4ca1f34's `--task` path: it reruns the task it names, no resume."""
        _stop_after_t001(repo)
        code, text, log = _second_run(repo, capsys, "--task", "T001")
        assert CONTINUED not in text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T001"}


def _ids_in(calls: str) -> set:
    """The tasks the mock runner was dispatched for, read off each prompt's
    result contract (``- `ids`: this task's id, `T002`, when …``)."""
    found = set()
    for line in calls.splitlines():
        prompt = json.loads(line).get("prompt") or ""
        found.update(re.findall(r"`ids`: this task's id, `(T\d{3})`", prompt))
    return found


class TestTheNativeEmit:
    """The emit Claude Code's host dispatches from (`implement` without
    `--exec-subagents`), with `util batch-done` as its per-task record."""

    def _emit(self, repo, capsys, *flags):
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", *flags])
        captured = capsys.readouterr()
        document = json.loads((_phase_dir(repo) / WRITE_TASK_FILENAME).read_text())
        return code, captured.out + captured.err, [
            t["task_id"] for t in document["batches"]]

    def _record(self, repo, task_id):
        from utils import batch_done, prior_changes
        from utils.output_handler import get_output_dir

        assert batch_done.main([
            "--phase-dir", str(_phase_dir(repo)), "--task-id", task_id,
            "--status", "completed",
        ]) == 0
        prior_changes.append(str(get_output_dir(repo.plan)), task_id, "implement",
                             f"did {task_id}")

    def test_a_plain_emit_after_a_recorded_task_hands_out_only_the_rest(
        self, repo, capsys
    ):
        _seed_generate_tasks(repo)
        code, text, first = self._emit(repo, capsys)
        assert code == 0, text
        assert first == ["T001", "T002", "T003"]
        self._record(repo, "T001")
        capsys.readouterr()
        code, text, second = self._emit(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert second == ["T002", "T003"]

    def test_a_plain_emit_with_nothing_recorded_is_unchanged(self, repo, capsys):
        _seed_generate_tasks(repo)
        assert self._emit(repo, capsys)[0] == 0
        code, text, second = self._emit(repo, capsys)
        assert code == 0, text
        assert CONTINUED not in text
        assert second == ["T001", "T002", "T003"]

    def test_fresh_hands_out_every_task_again(self, repo, capsys):
        _seed_generate_tasks(repo)
        assert self._emit(repo, capsys)[0] == 0
        self._record(repo, "T001")
        capsys.readouterr()
        code, text, second = self._emit(repo, capsys, "--fresh")
        assert code == 0, text
        assert CONTINUED not in text
        assert second == ["T001", "T002", "T003"]


#: A `codex` that is the mock runner, except that T002's first dispatch
#: blocks: the run is then mid-task when the abort lands, as Codex racing
#: 06's was (T003 in flight after T001, T002 and T010).
SLOW_T002_DISPATCH = '''#!/bin/sh
case " $* " in
  *" exec "*)
    tmp="{marker}.stdin.$$"
    cat > "$tmp"
    if [ ! -e "{marker}.blocked" ] && ( grep -q "id, .T002." "$tmp" \\
        || printf '%s' "$*" | grep -q "id, .T002." ); then
      touch "{marker}.blocked"
      sleep 120
    fi
    exec "{mock}" "$@" < "$tmp"
    ;;
esac
exec "{mock}" "$@"
'''


class TestAHostThatLeftMidCall:
    """Codex racing 06, end to end on the real dispatcher: the host's tool
    server aborts the detached run it was watching when the host leaves
    (`status --abort --run-id`, which is what both tool routes send on
    cancellation), mid-task. What must be left behind: a run recorded
    aborted — not crashed — with the finished task kept, a `status` and a
    `util next-command --blocked implement` (the tool routes' resume lookup)
    that both name `--implement <plan> --resume`, and a plain `--implement`
    that carries on from there."""

    def test_the_abort_leaves_a_resumable_run_and_names_the_way_forward(
        self, repo, tmp_path, capsys
    ):
        import os
        import subprocess
        import time

        import pytest

        from test_write_executor_orchestrators import MOCK_LLM, UV_STUB
        from utils import detach, next_phase, run_status, status_file
        from utils.state_manager import get_or_create_state

        if os.name == "nt":  # pragma: no cover - POSIX double-spawn path
            pytest.skip("the POSIX double-spawn path")
        _seed_generate_tasks(repo)
        marker = tmp_path / "dispatch"
        codex = tmp_path / "bin" / "codex"
        # The mock reads which CLI it plays from its own name.
        real = tmp_path / "real-bin"
        real.mkdir()
        (real / "codex").symlink_to(MOCK_LLM)
        codex.unlink()
        codex.write_text(SLOW_T002_DISPATCH.format(
            marker=marker, mock=real / "codex"), encoding="utf-8")
        codex.chmod(0o755)
        stub = tmp_path / "uv-stub"
        stub.write_text(UV_STUB, encoding="utf-8")
        stub.chmod(0o755)
        skill_root = Path(__file__).parent.parent
        env = {
            **os.environ,
            detach.UV_ENV: str(stub),
            "MULTI_LLM_HEARTBEAT_INTERVAL_S": "1",
            "MOCK_LLM_EDITS": json.dumps([
                {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
            ]),
            "MOCK_LLM_WRITE_IDS": IMPLEMENT_WRITE_IDS,
        }
        launch = subprocess.run(
            [sys.executable, str(skill_root / "multi_llm.py"), "implement",
             str(repo.plan), "--detach", "--exec-subagents", "--host", "codex"],
            cwd=str(repo.root), env=env, capture_output=True, text=True, timeout=120,
        )
        assert launch.returncode == 0, launch.stdout + launch.stderr

        phase_dir = _phase_dir(repo)
        deadline = time.time() + 120
        while not Path(f"{marker}.blocked").exists() and time.time() < deadline:
            time.sleep(0.2)
        assert Path(f"{marker}.blocked").exists(), (phase_dir / "orchestrator-run.log").read_text()

        # The host leaves: its tool server aborts the run it claimed.
        assert run_status.main(["--plan-file", str(repo.plan), "--abort"]) == 0
        capsys.readouterr()

        run = status_file.read_run(str(phase_dir))
        assert run["state"] == "aborted", run
        assert run["exit_code"] == 3
        log = (phase_dir / "orchestrator-run.log").read_text(encoding="utf-8")
        assert "Traceback" not in log, log
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T001") == "completed"
        assert state.get_task_status("T002") != "completed"
        assert not state.is_phase_completed("implement")

        # (b) the way forward, from `status` and from the tool routes' lookup.
        code = run_status.main(["--plan-file", str(repo.plan), "--host", "codex"])
        out = capsys.readouterr().out
        assert code == 3, out
        assert "resume with:" in out and "--resume" in out, out
        assert next_phase.main([
            str(repo.plan), "--blocked", "implement", "--host", "codex"]) == 0
        answered = capsys.readouterr().out
        assert f"--implement {repo.plan} --resume" in answered, answered

        # And the same `--implement` the host first sent carries on.
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", env["MOCK_LLM_EDITS"])
        code, text, calls = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert _ids_in(calls.read_text(encoding="utf-8")) == {"T002", "T003"}


# --- fix round 17: a later cycle, a start-over's earlier records ------------

def _finish_once(repo) -> None:
    """Cycle 1 on the executor leg: every task done, implement completed."""
    _seed_generate_tasks(repo)
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
    assert implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents",
    ]) == 0


def _stop_again_after_t001(repo, capsys, *flags) -> str:
    """A later cycle on the executor leg that stops at T002 (T003 not reached)."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
    capsys.readouterr()
    code = implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents", *flags,
    ])
    captured = capsys.readouterr()
    assert code == 3, captured.out + captured.err
    return captured.out + captured.err


class TestALaterCycleOnTheExecutorLeg:
    """Review of fix round 16 (abort #1): the executor leg never recorded
    implement started, so once an implement had finished, a later run that
    stopped left the old completion standing and the next plain
    ``--implement`` started over again; and a start-over left the earlier
    cycle's ``completed`` statuses and result files behind, so a later
    ``--resume`` kept T003 on the evidence of a cycle that ran before."""

    def test_a_stopped_second_cycle_is_continued_by_a_plain_run(self, repo, capsys):
        from utils.state_manager import get_or_create_state

        _finish_once(repo)
        _stop_again_after_t001(repo, capsys)
        state = get_or_create_state(repo.plan)
        assert state.is_phase_started("implement")
        assert not state.is_phase_completed("implement")
        # T003 was not reached in this cycle: nothing says it is done.
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "failed", "T003": "pending",
        }

        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert "kept 1 task(s) an earlier run finished (T001)" in text, text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T002", "T003"}
        assert get_or_create_state(repo.plan).is_phase_completed("implement")

    def test_a_third_cycle_started_with_fresh_is_continued_too(self, repo, capsys):
        """Cycle 2 finishes again (a plain run after a finished one starts
        over); cycle 3 is `--fresh` and stops; cycle 4, plain, continues it."""
        _finish_once(repo)
        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED not in text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T001", "T002", "T003"}

        _stop_again_after_t001(repo, capsys, "--fresh")
        log.unlink()  # the call log is appended to by every run
        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T002", "T003"}

    def test_a_resume_never_keeps_a_task_on_an_earlier_cycles_result(
        self, repo, capsys
    ):
        from utils.write_safety import TaskSnapshot

        _finish_once(repo)
        phase_dir = _phase_dir(repo)
        assert (phase_dir / "implement_batch_T003.json").is_file()
        assert TaskSnapshot.load(str(phase_dir), "T003") is not None

        text = _stop_again_after_t001(repo, capsys)
        assert "start over:" in text, text
        # Cycle 1's result and rollback point for T003 are set aside, never
        # deleted, and never read as this cycle's.
        assert not (phase_dir / "implement_batch_T003.json").exists()
        assert list(phase_dir.glob("implement_batch_T003.json.rerun-*"))
        assert TaskSnapshot.load(str(phase_dir), "T003") is None

        code, text, log = _second_run(repo, capsys, "--resume")
        assert code == 0, text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T002", "T003"}


    def test_a_resume_that_dispatches_on_a_finished_implement_records_it_started(
        self, repo, capsys
    ):
        """A `--resume` of a finished implement that hands a task out again
        (here T003, back to `failed` on record) and stops: implement is
        started again, so the next plain run continues it."""
        from utils.state_manager import get_or_create_state

        _finish_once(repo)
        state = get_or_create_state(repo.plan)
        state.update_task_status("T003", "failed", "an earlier run's failure")
        state.save()
        assert get_or_create_state(repo.plan).is_phase_completed("implement")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
        capsys.readouterr()
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume",
        ]) == 3
        state = get_or_create_state(repo.plan)
        assert state.is_phase_started("implement")
        assert not state.is_phase_completed("implement")

        code, text, log = _second_run(repo, capsys)
        assert code == 0, text
        assert CONTINUED in text, text
        assert _ids_in(log.read_text(encoding="utf-8")) == {"T003"}


class TestALaterCycleOnTheNativeLeg:
    """The same three cycles on the emit Claude Code's host dispatches from,
    with `util batch-done` as the per-task record."""

    _emit = TestTheNativeEmit._emit
    _record = TestTheNativeEmit._record

    def _finish_natively(self, repo, capsys):
        from test_stop_results_surface import multi_llm_util
        from utils.state_manager import get_or_create_state

        _seed_generate_tasks(repo)
        code, text, handed = self._emit(repo, capsys)
        assert code == 0, text
        for task_id in handed:
            self._record(repo, task_id)
        state = get_or_create_state(repo.plan)
        assert multi_llm_util([
            "finalize-tracking", "--state-file", str(state.state_file),
            "--mark-completed",
        ]) == 0
        capsys.readouterr()
        assert get_or_create_state(repo.plan).is_phase_completed("implement")

    def test_a_stopped_second_cycle_hands_out_what_it_did_not_reach(
        self, repo, capsys
    ):
        from utils import prior_changes
        from utils.output_handler import get_output_dir
        from utils.state_manager import get_or_create_state

        self._finish_natively(repo, capsys)
        code, text, handed = self._emit(repo, capsys)
        assert code == 0, text
        assert CONTINUED not in text
        assert handed == ["T001", "T002", "T003"]
        assert get_or_create_state(repo.plan).implementation_task_statuses() == {
            "T001": "pending", "T002": "pending", "T003": "pending",
        }
        # The host runs T001 again, records it and leaves.
        from utils import batch_done

        assert batch_done.main([
            "--phase-dir", str(_phase_dir(repo)), "--task-id", "T001",
            "--status", "completed",
        ]) == 0
        prior_changes.append(str(get_output_dir(repo.plan)), "T001", "implement",
                             "did T001 again")
        capsys.readouterr()

        code, text, handed = self._emit(repo, capsys)
        # T002 and T003 are in the prior-changes log from cycle 1: that is
        # not this cycle's run of them, so the resume is not refused.
        assert code == 0, text
        assert "RESUME_REFUSED" not in text, text
        assert CONTINUED in text, text
        assert "kept 1 task(s) an earlier run finished (T001)" in text, text
        assert handed == ["T002", "T003"]

    def test_a_task_logged_in_this_cycle_but_not_recorded_is_still_refused(
        self, repo, capsys
    ):
        from utils import prior_changes
        from utils.output_handler import get_output_dir

        self._finish_natively(repo, capsys)
        assert self._emit(repo, capsys)[0] == 0
        prior_changes.append(str(get_output_dir(repo.plan)), "T001", "implement",
                             "did T001 again")
        capsys.readouterr()
        code = implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--resume"])
        text = "".join(capsys.readouterr())
        assert code == 3, text
        assert "[RESUME_REFUSED]" in text, text
        assert "T001" in text and "T002 (" not in text and "T003 (" not in text, text

    def test_an_interrupted_task_rerun_is_handed_out_again(self, repo, capsys):
        """4ca1f34's `--task` rerun of a completed task, the host gone before
        its `batch-done`: the plain emit continues it and hands T001 out,
        instead of refusing it on the log line of the run before."""
        _finish_once(repo)
        capsys.readouterr()
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--yes", "--task", "T001"]) == 0
        capsys.readouterr()
        code, text, handed = self._emit(repo, capsys)
        assert code == 0, text
        assert "RESUME_REFUSED" not in text, text
        assert CONTINUED in text, text
        assert handed == ["T001"]


class TestAStartOverClearsTheEarlierPrompts:
    """Review of fix round 16 (noise #4): a start-over left the earlier run's
    `prompts/<id>.prompt.txt` and `<id>.attempt-N.prompt.txt`, so a task the
    new run never dispatched still had a prompt file in its folder."""

    def test_a_task_the_start_over_did_not_dispatch_keeps_no_prompt(
        self, repo, capsys
    ):
        _finish_once(repo)
        prompts = _phase_dir(repo) / "prompts"
        assert (prompts / "T003.prompt.txt").is_file()
        (prompts / "T003.attempt-2.prompt.txt").write_text("re-ask", encoding="utf-8")
        (prompts / "T003.tree.json").write_text("{}", encoding="utf-8")
        (prompts / "notes.txt").write_text("keep", encoding="utf-8")

        _stop_again_after_t001(repo, capsys)
        assert (prompts / "T001.prompt.txt").is_file()
        assert (prompts / "T002.prompt.txt").is_file()
        assert not (prompts / "T003.prompt.txt").exists()
        assert not (prompts / "T003.attempt-2.prompt.txt").exists()
        assert (prompts / "T003.tree.json").read_text(encoding="utf-8") == "{}"
        assert (prompts / "notes.txt").read_text(encoding="utf-8") == "keep"

    def test_the_native_start_over_clears_them_too(self, repo, capsys):
        _finish_once(repo)
        prompts = _phase_dir(repo) / "prompts"
        (prompts / "T002.tree.json").write_text("{}", encoding="utf-8")
        capsys.readouterr()
        code, text, handed = TestTheNativeEmit._emit(None, repo, capsys)
        assert code == 0, text
        assert handed == ["T001", "T002", "T003"]
        assert sorted(p.name for p in prompts.iterdir()) == ["T002.tree.json"]

    def test_a_planted_symlink_is_never_followed(self, repo, capsys, tmp_path):
        import os

        _finish_once(repo)
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "T009.prompt.txt"
        victim.write_text("not the plugin's", encoding="utf-8")
        prompts = _phase_dir(repo) / "prompts"
        os.symlink(victim, prompts / "T008.prompt.txt")
        (prompts / "T007.prompt.txt.d").mkdir()
        (prompts / "T006.prompt.txt").mkdir()
        _stop_again_after_t001(repo, capsys)
        assert victim.read_text(encoding="utf-8") == "not the plugin's"
        assert (prompts / "T008.prompt.txt").is_symlink()
        assert (prompts / "T006.prompt.txt").is_dir()

        # And a `prompts/` that is itself a symlink is left alone entirely.
        moved = _phase_dir(repo) / "prompts.real"
        prompts.rename(moved)
        os.symlink(outside, prompts)
        capsys.readouterr()
        code, text, _handed = TestTheNativeEmit._emit(None, repo, capsys, "--fresh")
        assert code == 0, text
        assert victim.read_text(encoding="utf-8") == "not the plugin's"
