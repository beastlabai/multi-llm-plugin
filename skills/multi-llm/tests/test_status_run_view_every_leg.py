"""`status --view run` on every leg (fix round 18, live run 7, run-7 #8).

Four things the run view got wrong in the seventh live run:

* Codex racing: its `resume with:` line was the raw
  `uv run … multi_llm.py apply-code-fixes <abs plan> --host codex --detach
  --resume`, while every other next-step line of the same view (the manual
  task note, the next phase) was the host's own spelling.
* Claude Code racing 06 and the stop path: the native implement block said
  `tasks: 6/10 done; …`, not the `all tasks:` tally (in the same words) the
  tool routes print.
* Claude Code racing 06, mid-run: a running native implement read "no run
  record — the host ran these tasks itself", the same words as a stopped one.
* The apply phases the native host ran did not appear at all, while on the
  tool routes, whose apply runs leave a run record, every one did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parent.parent
for entry in (str(SKILL_DIR), str(REPO_ROOT / "packaging")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import multi_llm  # noqa: E402
from common import surface  # noqa: E402
from utils import cli_common, run_status, status_file  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)

from tests.test_native_implement_report import (  # noqa: E402,F401
    PLAN_ARG,
    _emit,
    _util,
    repo,
)

TYPED = "plans/p.md"


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
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


def _failed_run(plan: Path, *, mode: str, directory: str, flags):
    phase = plan.parent / plan.stem / directory
    phase.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(phase), {
        "run_id": f"run-{mode}-0001", "state": "failed", "exit_code": 3,
        "error": "needs_human (exit 3)", "detached": True, "mode": mode,
        "replay_flags": list(flags), "plan_arg": TYPED,
    })
    if directory == "apply-fixes":
        # A write run that dispatched a batch, so its stop offers `--resume`.
        status_file.update_status(str(phase), {"progress": {"write_subagents": {
            "kind": "apply_code_fixes", "total": 2, "batches_dispatched": [0, 1],
            "tasks": [
                {"task_id": "apply-fix-batch-0", "status": "ok"},
                {"task_id": "apply-fix-batch-1", "status": "partial_application"},
            ],
            "failed_task_ids": ["apply-fix-batch-1"],
        }}})


def _resume_line(capsys, *argv) -> str:
    assert run_status.main(["--plan-file", TYPED, "--view", "run", *argv]) == 3
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if "resume with:" in line]
    assert len(lines) == 1, out
    return lines[0].split("resume with: ", 1)[1]


class TestResumeWithIsTheHostsSpelling:

    def test_codex_reads_the_way_its_user_types_it(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"])
        assert _resume_line(capsys, "--host", "codex") == (
            "Use the multi-llm skill with exactly these arguments: "
            "--apply-code-fixes plans/p.md --resume"
        )

    def test_claude_code_and_opencode_too(self, plan, capsys):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "claude-code", "--detach"])
        assert _resume_line(capsys, "--host", "opencode") == (
            "/multi-llm --apply-code-fixes plans/p.md --resume")
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "claude-code"])
        assert _resume_line(capsys, "--host", "claude-code") == (
            "/multi-llm:multi-llm --apply-code-fixes plans/p.md --resume")

    def test_the_flags_the_user_typed_ride_along_and_no_other(self, plan, capsys):
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=["--quick", "--host", "opencode", "--detach",
                           "--max-detached", "2", "--yes", "--internal-x"])
        assert _resume_line(capsys, "--host", "opencode") == (
            "/multi-llm --review-plan plans/p.md --quick --max-detached 2 "
            "--reaggregate")

    def test_a_tool_route_keeps_the_dispatcher_command_its_server_re_spells(
        self, plan, capsys, monkeypatch,
    ):
        """The tool's server re-spells every dispatcher command in the output
        and hands it back as a tool call in `options`: it needs the command."""
        monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "codex", "--detach"])
        line = _resume_line(capsys, "--host", "codex")
        assert "multi_llm.py apply-code-fixes" in line and line.endswith(
            "--host codex --detach --resume"), line

    def test_the_flags_dropped_are_the_ones_the_tool_routes_reserve(self):
        """The two lists of what a host composes on every call cannot drift."""
        reserved = surface._reserved_flags("codex")
        names = list(reserved["exact"]) + list(reserved["named_exception"])
        for name in names:
            assert cli_common.drop_host_composed_flags([name]) == [], name
            assert cli_common.drop_host_composed_flags([f"{name}=x", "--quick"]) == [
                "--quick"], name
        for prefix in reserved["prefixes"]:
            assert cli_common.drop_host_composed_flags([prefix + "worker"]) == []
        spellings = {
            spelling for names_, _value in cli_common._HOST_COMPOSED.values()
            for spelling in names_
        }
        assert {s for s in spellings
                if not s.startswith(cli_common.INTERNAL_FLAG_PREFIX)} == set(names)


def _template_composed(host: str) -> set:
    """What *host*'s own template bakes into its dispatcher command, plus
    `--detach` when its *run detached* sentence passes it, as parser dests."""
    import re

    from utils import host_capabilities

    path = SKILL_DIR / ("SKILL.md" if host == "claude-code"
                        else f"hosts/{host}/shim.md.tmpl")
    (baked,) = re.findall(r'<mode> "<plan-path>" (.*?) \[flags…\]',
                          path.read_text(encoding="utf-8"))
    dests = {
        action.dest: action
        for action in cli_common.build_parent_parser()._actions
        for _name in action.option_strings
    }
    by_name = {name: dest for dest, action in dests.items()
               for name in action.option_strings}
    out = {by_name[word] for word in baked.split() if word.startswith("--")}
    if host_capabilities.verb_sentence(host, "detach").startswith(
            "**Run detached** — pass `--detach`"):
        out.add("detach")
    return out


class TestEachHostDropsOnlyWhatItsOwnShimComposes:
    """Fix round 19: the tool routes' reserved set was dropped on every file
    route, so a Claude Code run started with `--exec-subagents --yes` printed
    a resume without them (the native leg, not the one that stopped), and the
    shims lost `--interactive`, which replaces their baked `--yes`."""

    @pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
    def test_the_set_is_what_the_hosts_template_bakes(self, host):
        assert set(cli_common.file_route_composed_flags(host)) == (
            _template_composed(host)), host

    def test_every_host_is_held_to_its_template(self):
        from utils import host_capabilities

        assert set(host_capabilities.host_ids()) == {
            "claude-code", "codex", "opencode"}

    def test_claude_code_keeps_the_leg_and_the_approval_the_user_chose(
        self, plan, capsys,
    ):
        _failed_run(plan, mode="apply-code-fixes", directory="apply-fixes",
                    flags=["--host", "claude-code", "--exec-subagents", "--yes",
                           "--models", "a:b"])
        assert _resume_line(capsys, "--host", "claude-code") == (
            "/multi-llm:multi-llm --apply-code-fixes plans/p.md "
            "--exec-subagents --yes --models a:b --resume")

    @pytest.mark.parametrize("host,prefix", [
        ("codex", "Use the multi-llm skill with exactly these arguments: "),
        ("opencode", "/multi-llm "),
    ])
    def test_a_shim_keeps_interactive_and_drops_what_it_bakes(
        self, plan, capsys, host, prefix,
    ):
        _failed_run(plan, mode="review-plan", directory="review-plan",
                    flags=["--host", host, "--interactive", "--exec-subagents",
                           "--quick"])
        assert _resume_line(capsys, "--host", host) == (
            f"{prefix}--review-plan plans/p.md --interactive --quick --reaggregate")


def _failed_ask(plan: Path, question: str) -> None:
    from utils.output_handler import output_dir_path, slugify_question

    ask_dir = output_dir_path(plan) / "ask" / (
        slugify_question(question) + "-20261006-000000")
    ask_dir.mkdir(parents=True)
    status_file.claim_run(str(ask_dir), {
        "run_id": "run-ask-0001", "state": "failed", "exit_code": 1,
        "error": "boom", "detached": True, "mode": "ask",
        "replay_flags": ["--host", "claude-code", "--quick"], "plan_arg": TYPED,
    })
    status_file.update_status(str(ask_dir), {"question": question})


class TestTheAskQuestionIsNeverDropped:
    """Fix round 19: the control-character filter ran over the question
    positional, so a question with a newline or a tab vanished from the
    host-spelled line, and the command printed did not parse."""

    @pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
    @pytest.mark.parametrize("question", [
        "Is the cache safe?\nWhat about eviction?", "Is it safe?\tAnd fast?"])
    def test_one_it_cannot_spell_keeps_the_dispatcher_command(
        self, plan, capsys, host, question,
    ):
        _failed_ask(plan, question)
        assert run_status.main(["--plan-file", TYPED, "--view", "run", "--host",
                                host, "--run-id", "run-ask-0001"]) == 3
        out = capsys.readouterr().out
        assert "resume with: " in out, out
        command = out.split("resume with: ", 1)[1]
        assert "multi_llm.py ask " in command, out
        assert f"'{question}' --host claude-code --quick --force" in command, out

    def test_a_one_line_question_is_spelled_the_hosts_way(self, plan, capsys):
        _failed_ask(plan, "Is the cache safe?")
        assert _resume_line(capsys, "--host", "claude-code", "--run-id",
                            "run-ask-0001") == (
            '/multi-llm:multi-llm --ask plans/p.md "Is the cache safe?" '
            "--quick --force")


def _status_text(capsys, *flags) -> str:
    multi_llm.main(["status", PLAN_ARG, "--host", "claude-code", "--view", "run", *flags])
    return capsys.readouterr().out


class TestTheNativeImplementBlock:

    def test_a_running_one_says_it_is_in_progress(self, repo, capsys):
        handoff, _out = _emit(capsys)
        state_file = handoff["state_file"]
        assert _util(capsys, "metrics", "start", "--state-file", state_file,
                     "--phase", "implement", "--total-batches", "4")[0] == 0
        assert _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                     "--task-id", "T001", "--status", "completed")[0] == 0
        out = _status_text(capsys)
        head = next(line for line in out.splitlines() if line.startswith("  implement  "))
        assert head.startswith("  implement  in progress on the host (started "), out
        assert "the host runs these tasks itself" in head
        tally = next(line for line in out.splitlines() if "all tasks:" in line)
        assert tally.split("all tasks:", 1)[1].strip().startswith("1/6 done; completed: T001")
        assert "pending: T002" in tally  # a live run's tasks wait, not "not run"

    def test_one_the_host_left_says_it_is_not_finished(self, repo, capsys):
        handoff, _out = _emit(capsys)
        state_file = handoff["state_file"]
        _util(capsys, "metrics", "start", "--state-file", state_file,
              "--phase", "implement", "--total-batches", "4")
        _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
              "--task-id", "T001", "--status", "completed")
        _util(capsys, "metrics", "finish", "--state-file", state_file,
              "--phase", "implement")
        out = _status_text(capsys)
        assert ("  implement  not finished on the host (not marked completed) — "
                "no run record, since the host ran these tasks itself") in out, out
        assert "not run: T002" in out

    def test_its_tally_is_the_tool_routes_all_tasks_line(self, repo, capsys):
        """Same label, same words as the executor routes' tally."""
        _emit(capsys)
        _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
              "--task-id", "T001", "--status", "completed")
        out = _status_text(capsys)
        tally = run_status.implement_tally_text(
            run_status.implement_task_tally(Path(repo.plan)))
        assert f"      all tasks:      {tally}" in out, out
        assert not any(line.strip().startswith("tasks:") for line in out.splitlines())


def _native_apply(plan: Path, *, metrics_start: bool = True) -> Path:
    """What the native emit and the host's loop leave for apply-suggestions."""
    out = plan.parent / plan.stem
    phase_dir = out / "apply-suggestions"
    phase_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        {"section_key": "s1", "prompt": "Apply.", "suggestions": [
            {"group_id": "g1", "title": "One"}]},
        {"section_key": "s2", "prompt": "Apply.", "suggestions": [
            {"group_id": "g2", "title": "Two"}]},
    ]
    tasks = materialize_write_tasks(
        kind="apply_suggestions", batches=batches, phase_dir=str(phase_dir),
        plan_file=str(plan))
    write_task_file_for_phase(
        kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
        plan_file=str(plan))
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.mark_phase_started("apply-suggestions")
    state.save()
    state_file = out / "state.json"
    if metrics_start:
        assert multi_llm.main(["util", "metrics", "start", "--state-file",
                               str(state_file), "--phase", "apply-suggestions",
                               "--total-batches", "2"]) == 0
    return state_file


def _apply_block(capsys, *flags) -> list:
    run_status.main(["--plan-file", TYPED, "--view", "run", "--host",
                     "claude-code", *flags])
    out = capsys.readouterr().out
    lines = out.splitlines()
    at = next((i for i, line in enumerate(lines)
               if line.startswith("  apply-suggestions  ")), None)
    if at is None:
        return [out]
    return lines[at:at + 2]


class TestTheNativeApplyPhases:

    def test_a_running_one_is_listed_with_its_batches(self, plan, capsys):
        state_file = _native_apply(plan)
        assert multi_llm.main(["util", "metrics", "record", "--state-file",
                               str(state_file), "--phase", "apply-suggestions",
                               "--label", "Batch 1", "--batch-index", "1",
                               "--total-batches", "2", "--no-eta"]) == 0
        capsys.readouterr()
        head, batches = _apply_block(capsys)
        assert head.startswith(
            "  apply-suggestions  in progress on the host (started "), head
        assert "no run record, since the host runs these batches itself" in head
        assert batches.strip() == "batches:        1/2 recorded by this run"

    def test_a_finished_one_reads_done(self, plan, capsys):
        state_file = _native_apply(plan)
        multi_llm.main(["util", "metrics", "finish", "--state-file",
                        str(state_file), "--phase", "apply-suggestions"])
        state = StateManager(plan)
        state.mark_phase_completed("apply-suggestions")
        state.save()
        capsys.readouterr()
        head = _apply_block(capsys)[0]
        assert head == (
            "  apply-suggestions  done on the host (marked completed) — no run "
            "record, since the host ran these batches itself; read from state.json")

    def test_json_carries_them(self, plan, capsys):
        _native_apply(plan)
        capsys.readouterr()
        run_status.main(["--plan-file", TYPED, "--view", "run", "--json"])
        payload = json.loads(capsys.readouterr().out)
        entry = payload["apply_phases"][0]
        assert entry["phase"] == "apply-suggestions"
        assert entry["completed"] is False and entry["in_progress_since"]
        assert entry["batches_total"] == 2

    def test_an_executor_run_keeps_its_own_view(self, plan, capsys):
        """A run the write executor drove has its report and run view."""
        _native_apply(plan)
        phase_dir = plan.parent / plan.stem / "apply-suggestions"
        (phase_dir / "write_batch_report.json").write_text("{}", encoding="utf-8")
        capsys.readouterr()
        block = _apply_block(capsys)
        assert "on the host" not in block[0], block

    def test_the_phase_selector_narrows_it(self, plan, capsys):
        _native_apply(plan)
        capsys.readouterr()
        assert "on the host" not in _apply_block(capsys, "--phase", "implement")[0]
        assert "on the host" in _apply_block(capsys, "--phase", "apply-suggestions")[0]
