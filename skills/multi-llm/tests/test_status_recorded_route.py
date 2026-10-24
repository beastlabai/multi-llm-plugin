"""A run's commands are spelled by the route that started it (fix round 24).

Live run 10 (FINDINGS-racing-codex.md issue 2, FINDINGS-stop-codex.md issue
1): `status --host codex` run from a shell, where `MULTI_LLM_ROUTE` is unset,
printed the resume and manual-task commands of a run the Codex plugin's tool
had started as "Use the multi-llm skill with exactly these arguments: …", the
file shim's sentence, where the user types `$multi-llm:multi-llm …`.

The tool servers set `MULTI_LLM_ROUTE=package` on every child, so the
detached worker records the route it was started on (`run.route`), and a
reader given no route of its own — a shell's `status`, `util next-command` —
spells that run's commands in the route's spelling: the template and the
flags that route composes. A reader on a tool route still hands its server
the dispatcher command, a run with no record keeps today's default, and a
run another host started, or a Claude Code reader (whose route is its own
skill root), ignore the record.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import multi_llm
from utils import cli_common, detach, next_phase, process_control, run_status, status_file
from utils.state_manager import get_or_create_state

import tests.test_check_gate_fixture as gate
from tests.test_check_gate_fixture import _first_run, repo  # noqa: F401
from tests.test_manual_tasks_on_completion import workdir  # noqa: F401

TYPED = "plans/p.md"
CODEX_TOOL = "$multi-llm:multi-llm "
CODEX_FILE = "Use the multi-llm skill with exactly these arguments: "


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "plans" / "p.md"
    path.parent.mkdir()
    path.write_text("# Plan\n\n## Step 1\n\nBody.\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                  "commit", "-q", "-m", "plan"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True,
                       capture_output=True)
    return path


def _failed_run(plan: Path, *, mode: str, directory: str, flags, host: str,
                route=None, run_id: str = "run-0001"):
    phase = plan.parent / plan.stem / directory
    phase.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id, "state": "failed", "exit_code": 3,
        "error": "needs_human (exit 3)", "detached": True, "mode": mode,
        "host": host, "replay_flags": list(flags), "plan_arg": TYPED,
        "started_at": "2026-10-07T10:00:00",
    }
    if route:
        record["route"] = route
    status_file.claim_run(str(phase), record)
    return phase


def _resume_line(capsys, *argv) -> str:
    capsys.readouterr()
    assert run_status.main(["--plan-file", TYPED, "--view", "run", *argv]) == 3
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if "resume with:" in line]
    assert len(lines) == 1, out
    return lines[0].split("resume with: ", 1)[1]


class TestTheWorkerRecordsItsRoute:

    def _start(self, tmp_path, monkeypatch, host):
        tmp_path = tmp_path / f"repo-{len(list(tmp_path.iterdir()))}"
        tmp_path.mkdir()
        monkeypatch.setattr(
            process_control, "install_worker_signal_handlers", lambda _cb: None)
        monkeypatch.setattr(detach, "_ACTIVE", None)
        monkeypatch.setattr(detach, "_sweep_live_clis", lambda: None)
        monkeypatch.setattr(detach, "_restore_codex_trust", lambda: None)
        monkeypatch.chdir(tmp_path)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase = tmp_path / "plan" / "implement"
        phase.mkdir(parents=True)
        handle = detach.RunHandle(
            str(phase), run_id="run-route", mode="implement",
            plan_file=str(plan), host=host, detached=False,
        )
        handle.start()
        try:
            return status_file.read_run(str(phase))
        finally:
            handle.finish(0)

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_a_tool_route_run_records_package(self, tmp_path, monkeypatch, host):
        monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        assert self._start(tmp_path, monkeypatch, host)["route"] == "package"

    def test_a_file_route_run_records_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MULTI_LLM_ROUTE", raising=False)
        assert "route" not in self._start(tmp_path, monkeypatch, "codex")

    def test_a_value_the_host_does_not_honour_is_not_recorded(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        assert "route" not in self._start(tmp_path, monkeypatch, "claude-code")
        monkeypatch.setenv("MULTI_LLM_ROUTE", "plugin")
        assert "route" not in self._start(tmp_path, monkeypatch, "codex")


class TestStatusSpellsTheRunsOwnRoute:

    def test_a_codex_tool_run_read_from_a_shell(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"], host="codex",
                    route="package")
        assert _resume_line(capsys, "--host", "codex") == (
            f"{CODEX_TOOL}--apply-code-fixes {TYPED} --resume")

    def test_the_flags_dropped_are_the_ones_that_route_composes(
        self, plan, capsys,
    ):
        """OpenCode's two routes share one template; the tool route composes
        `--max-detached` (and `--if-not-running`), so a run it started never
        prints them back as typed."""
        flags = ["--quick", "--host", "opencode", "--detach", "--max-detached",
                 "2", "--if-not-running", "--yes"]
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=flags, host="opencode", route="package")
        assert _resume_line(capsys, "--host", "opencode") == (
            f"/multi-llm --review-plan {TYPED} --quick --reaggregate")

    def test_with_no_record_the_default_stands(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"], host="codex")
        assert _resume_line(capsys, "--host", "codex") == (
            f"{CODEX_FILE}--apply-code-fixes {TYPED} --resume")

    def test_a_reader_on_the_tool_route_keeps_the_dispatcher_command(
        self, plan, capsys, monkeypatch,
    ):
        monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"], host="codex",
                    route="package")
        line = _resume_line(capsys, "--host", "codex")
        # The dispatcher's own prefix, which quotes a Windows script path.
        assert f"{cli_common.dispatcher_prefix()} apply-code-fixes" in line, line

    def test_another_hosts_record_is_not_this_hosts_route(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "opencode", "--detach"], host="opencode",
                    route="package")
        assert _resume_line(capsys, "--host", "codex") == (
            f"{CODEX_FILE}--apply-code-fixes {TYPED} --resume")

    def test_claude_code_takes_its_route_from_its_skill_root(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "claude-code"], host="claude-code",
                    route="package")
        assert _resume_line(capsys, "--host", "claude-code") == (
            f"/multi-llm:multi-llm --apply-code-fixes {TYPED} --resume")

    def test_each_run_is_spelled_by_its_own_route(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"], host="codex",
                    route="package")
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=["--host", "codex", "--detach"], host="codex",
                    run_id="run-0002")
        capsys.readouterr()
        run_status.main(["--plan-file", TYPED, "--view", "run", "--host", "codex"])
        lines = [line.strip().split("resume with: ", 1)[1]
                 for line in capsys.readouterr().out.splitlines()
                 if "resume with:" in line]
        assert sorted(lines) == sorted([
            f"{CODEX_TOOL}--apply-code-fixes {TYPED} --resume",
            f"{CODEX_FILE}--review-plan {TYPED} --reaggregate",
        ]), lines


class TestTheStopsCommandsOutsideTheRunView:
    """The executor stop of the check-gate fixture, recorded as the Codex
    tool route's detached worker records it."""

    def _stop(self, repo, capsys, route="package"):
        _first_run(repo, capsys)
        record = {
            "run_id": "run-tool", "state": "failed", "exit_code": 3,
            "error": "implement stopped at T005", "detached": True,
            "mode": "implement", "host": "codex",
            "replay_flags": ["--host", "codex", "--yes", "--exec-subagents",
                             "--detach"],
            "plan_arg": "plans/checked.md", "started_at": "2026-10-07T10:00:00",
        }
        if route:
            record["route"] = route
        status_file.claim_run(str(repo.phase_dir), record)

    def _util(self, capsys, *argv):
        capsys.readouterr()
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_next_command_blocked_reads_the_recorded_route(self, repo, capsys):
        self._stop(repo, capsys)
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--blocked", "implement",
            "--host", "codex")
        assert code == 0, err
        assert (f"[RESUME_COMMAND] {CODEX_TOOL}--implement plans/checked.md "
                "--resume") in out.splitlines(), out

    def test_next_command_after_reads_it_too(self, repo, capsys):
        self._stop(repo, capsys)
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement",
            "--host", "codex")
        assert code == 0, err
        lines = out.splitlines()
        assert (f"[NEXT_COMMAND] {CODEX_TOOL}--implement plans/checked.md "
                "--resume") in lines, out
        answers = [line for line in lines if line.startswith("[NEXT_COMMAND_ANSWERS] ")]
        assert answers and all(
            line.split(" ", 1)[1].startswith(CODEX_TOOL) for line in answers), answers

    def test_with_no_record_util_next_command_keeps_the_default(self, repo, capsys):
        self._stop(repo, capsys, route=None)
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--blocked", "implement",
            "--host", "codex")
        assert code == 0, err
        assert (f"[RESUME_COMMAND] {CODEX_FILE}--implement plans/checked.md "
                "--resume") in out.splitlines(), out

    def test_status_resume_and_trailer_read_it(self, repo, capsys):
        self._stop(repo, capsys)
        capsys.readouterr()
        run_status.main(["--plan-file", "plans/checked.md", "--host", "codex"])
        out = capsys.readouterr().out
        resumes = [line for line in out.splitlines() if "resume with:" in line]
        assert resumes and all(CODEX_TOOL in line for line in resumes), out
        assert CODEX_FILE not in out, out

    def test_a_shell_reader_with_the_route_set_by_hand_is_unchanged(
        self, repo, capsys, monkeypatch,
    ):
        """A recorded file route (none) under a reader on the tool route:
        that reader's own route decides, as before."""
        self._stop(repo, capsys, route=None)
        monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--blocked", "implement",
            "--host", "codex")
        assert code == 0, err
        assert (f"[RESUME_COMMAND] {CODEX_TOOL}--implement plans/checked.md "
                "--resume") in out.splitlines(), out


class TestTheRecordedRouteLookup:

    def test_the_newest_run_of_the_host_wins(self, plan):
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=[], host="codex", route="package")
        phase = _failed_run(plan, mode="apply-suggestions",
                            directory="apply-suggestions", flags=[], host="codex",
                            run_id="run-0002")
        status_file.update_run(str(phase), {"started_at": "2026-10-07T11:00:00"},
                               expect_run_id="run-0002")
        assert next_phase.recorded_route(plan, "codex") is None
        assert next_phase.recorded_route(plan, "codex", "review-plan") == "package"
        assert next_phase.recorded_route(plan, "opencode") is None

    def test_a_planted_value_is_never_a_route(self, plan):
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=[], host="codex", route="$(rm -rf ~)")
        assert next_phase.recorded_route(plan, "codex") is None


class TestThePlansManualTasksAndNextStep:
    """A complete workflow with two manual tasks deferred
    (`test_manual_tasks_on_completion`), whose last phase the Codex tool
    route ran detached."""

    def _plan(self, workdir, route="package", state="done", exit_code=0):
        from tests.test_manual_tasks_on_completion import _complete_plan

        plan = _complete_plan(workdir)
        phase = plan.parent / plan.stem / "apply-fixes"
        phase.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": "run-fixes", "state": state, "exit_code": exit_code,
            "detached": True, "mode": "apply-code-fixes", "host": "codex",
            "replay_flags": ["--host", "codex", "--detach"],
            "started_at": "2026-10-07T10:00:00",
        }
        if route:
            record["route"] = route
        status_file.claim_run(str(phase), record)
        return plan

    def test_the_manual_note_in_plain_status(self, workdir, capsys):
        from tests.test_manual_note_command_last import ADVICE

        plan = self._plan(workdir)
        capsys.readouterr()
        run_status.main(["--plan-file", str(plan), "--host", "codex"])
        out = capsys.readouterr().out
        assert f"{ADVICE}: {CODEX_TOOL}--implement " in out, out
        assert CODEX_FILE not in out, out

    def test_the_terminal_polls_next_step_block(self, workdir, capsys):
        plan = self._plan(workdir)
        capsys.readouterr()
        assert run_status.main(["--plan-file", str(plan), "--host", "codex",
                                "--view", "run", "--phase", "apply-code-fixes"]) == 0
        out = capsys.readouterr().out
        commands = [line for line in out.splitlines()
                    if line.startswith("[MANUAL_TASK_COMMAND] ")]
        assert commands and commands[0].startswith(
            f"[MANUAL_TASK_COMMAND] {CODEX_TOOL}--implement "), out

    def test_with_no_record_both_keep_the_default(self, workdir, capsys):
        plan = self._plan(workdir, route=None)
        capsys.readouterr()
        assert run_status.main(["--plan-file", str(plan), "--host", "codex",
                                "--view", "run", "--phase", "apply-code-fixes"]) == 0
        out = capsys.readouterr().out
        assert f"[MANUAL_TASK_COMMAND] {CODEX_FILE}--implement " in out, out
