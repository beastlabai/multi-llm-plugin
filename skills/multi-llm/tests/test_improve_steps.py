"""``util improve-steps``: the native leg's planner, and its parity with the executor.

Plan improve-plan-and-improve-code §2.4 (T4) and the planner, parity and
native-leg items of T11. The planner calls the executor leg's own functions
(``improve_workflow.read_records``, ``plan_steps``, ``preflight``), so most
tests here pin what it prints for a state and hold it against what the driver
forwards for the same state. The executor-leg fixtures (``stub``, the state
writers) are ``tests/test_improve_orchestrator.py``'s.

The last section drives the native leg end to end: the dispatcher calls in the
order the cores make them, each phase without ``--exec-subagents`` (the host's
simulated subagents as ``SkillRunner.simulate_subagents`` plays them), then the
same through the Codex file shim's baked ``--host codex --yes
--exec-subagents``.

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_improve_steps.py -v
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import full_workflow_orchestrator as fwo  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from harness import SkillRunner  # noqa: E402
from utils import cli_common, detach, host_capabilities, next_phase, status_file  # noqa: E402
from utils import improve_steps  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_improve_orchestrator import (  # noqa: E402,F401
    APPLY_FLAGS,
    APPLY_PHASE,
    CODE,
    ENTRY,
    LISTING_NAME,
    MODES,
    PLAN,
    REFUSAL,
    REVIEW_PHASE,
    STATE_FAULTS,
    T,
    _args,
    _dead_pid,
    _live_run,
    _out,
    _plan,
    _write_manifest,
    _write_state,
    stub,
)

SKILL_ROOT = Path(__file__).parent.parent
PLUGIN = "/multi-llm:multi-llm "
APPLY_DIR = {PLAN: "apply-suggestions", CODE: "apply-fixes"}

#: The Codex file shim's baked dispatcher tokens (§3, criterion 1).
CODEX_SHIM = ("--host", "codex", "--yes", "--exec-subagents")

#: The cross-cutting flags the parity matrix types on both legs.
CROSS = ("--host", "codex", "--exec-subagents", "--dry-run-subagents",
         "--skip-prerequisites")

#: No step's argv and no continuation ever carries one of these.
NEVER_FORWARDED = ("--host", "--skip-prerequisites", "--detach", "--if-not-running",
                   "--max-detached")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository holding the plan, as the cwd, with no host declared."""
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _typed(plan: Path) -> str:
    """The plan as the user typed it: relative to the cwd (the repository)."""
    return os.path.relpath(plan)


def _shown(path: Path) -> str:
    """A path as a message names it: relative to the cwd, ``/``-separated on every OS."""
    return _typed(path).replace(os.sep, "/")


def _run(capsys, plan, mode, *argv) -> Tuple[int, str, str]:
    """``util improve-steps`` in-process; ``(exit, stdout, stderr)``."""
    target = plan if isinstance(plan, str) else _typed(plan)
    code = improve_steps.main([target, "--mode", mode, *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _json(capsys, plan, mode, *argv) -> Tuple[int, Optional[Dict[str, Any]], str]:
    code, out, err = _run(capsys, plan, mode, *argv, "--json")
    return code, (json.loads(out) if out.strip() else None), err


def _flags(out: str, index: str) -> List[str]:
    """The tokens of step *index*'s ``[IMPROVE_STEP_FLAGS]`` line, unquoted."""
    prefix = f"{improve_steps.FLAGS_MARKER} {index} "
    found = [line for line in out.splitlines() if line.startswith(prefix)]
    assert len(found) == 1, (index, out)
    return shlex.split(found[0])[2:]


def _rerun_line(out: str) -> str:
    found = [line for line in out.splitlines()
             if line.startswith(improve_steps.RERUN_MARKER + " ")]
    assert len(found) == 1, out
    return found[0][len(improve_steps.RERUN_MARKER) + 1:]


def _parse_text(out: str) -> Dict[str, Any]:
    """The text block read back into the ``--json`` object's facts."""
    notices: List[str] = []
    steps: List[Dict[str, Any]] = []
    fresh: List[str] = []
    flags: Dict[str, List[str]] = {}
    already = rerun = None
    for line in out.splitlines():
        if line.startswith(improve_steps.STEP_MARKER + " "):
            body = line[len(improve_steps.STEP_MARKER) + 1:]
            index, step, rest = body.split(" ", 2)
            decision, _, reason = rest.partition(" ")
            if reason:
                assert reason.startswith("(") and reason.endswith(")"), line
                reason = reason[1:-1]
            steps.append({"index": index, "step": step, "decision": decision,
                          "reason": reason or None})
        elif line.startswith(improve_steps.FRESH_MARKER + " "):
            fresh.append(line.split(" ", 1)[1])
        elif line.startswith(improve_steps.FLAGS_MARKER + " "):
            tokens = shlex.split(line)
            flags[tokens[1]] = tokens[2:]
        elif line.startswith(improve_steps.ALREADY_COMPLETE_MARKER + " "):
            already = line[len(improve_steps.ALREADY_COMPLETE_MARKER) + 1:]
        elif line.startswith(improve_steps.RERUN_MARKER + " "):
            rerun = line[len(improve_steps.RERUN_MARKER) + 1:]
        else:
            assert not steps, f"a notice after the step blocks: {line!r}"
            notices.append(line)
    for step in steps:
        step["fresh"] = step["index"] in fresh
        step["argv"] = flags[step["index"]]
    return {"already_complete": already, "notices": notices, "steps": steps,
            "rerun": rerun}


def _tree(root: Path) -> Dict[str, Any]:
    """Every file and dir under *root* but ``.git``, with its bytes and stamp."""
    seen: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        stat = path.lstat()
        seen[str(path.relative_to(root))] = (
            (path.read_bytes(), stat.st_mtime_ns) if path.is_file() else "dir"
        )
    return seen


class NoStdin:
    """A stdin the planner must never read."""

    def read(self, *a, **k):
        pytest.fail("stdin was read")

    readline = read

    def __iter__(self):
        pytest.fail("stdin was read")


def _review_files(plan: Path, actionable: int, consolidated: bool = False) -> None:
    """review-plan's files with *actionable* groups judged valid/needs-human."""
    phase_dir = _out(plan) / "review-plan"
    phase_dir.mkdir(parents=True, exist_ok=True)
    statuses = ["valid", "needs-human-decision"]
    (phase_dir / "validation.json").write_text(json.dumps({"groups": [
        {"group_index": i, "status": statuses[i % 2]} for i in range(actionable)
    ]}), encoding="utf-8")
    grouped = json.dumps({"groups": []})
    (phase_dir / "grouped.json").write_text(grouped, encoding="utf-8")
    if consolidated:
        import hashlib

        (phase_dir / "consolidated.json").write_text(json.dumps({"metadata": {
            "grouped_hash": hashlib.sha256(grouped.encode()).hexdigest(),
        }}), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The util route, the parser and the help
# ---------------------------------------------------------------------------

class TestTheRoute:
    def test_it_is_a_util_command_listed_under_util_help(self, capsys):
        module, attr, _summary = multi_llm.UTIL_COMMANDS["improve-steps"]
        assert (module, attr) == ("utils.improve_steps", "main")
        assert multi_llm.main(["util", "--help"]) == 0
        out = capsys.readouterr().out
        assert "improve-steps" in out
        assert "util improve-steps <plan> --mode improve-plan|improve-code" in out

    def test_the_dispatcher_routes_it_in_process(self, repo, capsys):
        plan = _plan(repo, name="x")
        assert multi_llm.main(
            ["util", "improve-steps", "plans/x.md", "--mode", PLAN]) == 0
        out = capsys.readouterr().out
        body, result = out.rsplit(cli_common.RESULT_MARKER, 1)
        assert "[IMPROVE_STEP] 1 review-plan run (not reviewed yet)" in body
        assert json.loads(result)["mode"] == "util improve-steps"
        assert plan.is_file()

    @pytest.mark.parametrize("mode", MODES)
    def test_its_flags_are_the_composites_own(self, mode):
        """The planner's namespace is the one ``skip_reason``, ``step_flags``
        and ``continuation_flags`` read on the executor leg: same option
        strings, dests, nargs and defaults (less ``--plan-file``)."""
        def own(parser):
            return {
                (tuple(a.option_strings), a.dest, a.nargs, repr(a.default),
                 type(a).__name__)
                for a in parser._actions
                if a.option_strings and a.dest not in (
                    "help", "plan_file", "mode", "json")
            }

        planner = improve_steps.build_parser(mode)
        composite = iw.build_parser(mode)
        assert own(planner) == own(composite)
        assert planner.get_default("consolidate") is None, (
            "the planner builds no ImproveWorkflow, so it sets neither default")

    def test_improve_plan_refuses_the_scope_flags_as_its_composite_does(
        self, repo, capsys
    ):
        _plan(repo, name="x")
        with pytest.raises(SystemExit) as excinfo:
            improve_steps.main(["plans/x.md", "--mode", PLAN, "--files", "a.py"])
        assert excinfo.value.code == 2
        assert "unrecognized arguments: --files a.py" in capsys.readouterr().err
        # Through the dispatcher a parser's refusal is exit 3, never 2.
        assert multi_llm.main(["util", "improve-steps", "plans/x.md", "--mode", PLAN,
                               "--files", "a.py"]) == 3

    def test_a_missing_mode_is_a_usage_error(self, repo, capsys):
        _plan(repo, name="x")
        assert multi_llm.main(["util", "improve-steps", "plans/x.md"]) == 3
        assert "--mode" in capsys.readouterr().err

    def test_help_names_the_markers_and_needs_no_mode(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            improve_steps.main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        for marker in ("[IMPROVE_STEP]", "[IMPROVE_STEP_FRESH]", "[IMPROVE_STEP_FLAGS]",
                       "[IMPROVE_RERUN]", "[ALREADY_COMPLETE]", "--files-from"):
            assert marker in out
        assert "util improve-steps plans/my-plan.md --mode improve-plan" in out


# ---------------------------------------------------------------------------
# 2. Exit codes and the refusals
# ---------------------------------------------------------------------------

class TestExitCodes:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_step_to_run_is_exit_0(self, repo, capsys, mode):
        plan = _plan(repo)
        code, out, err = _run(capsys, plan, mode)
        assert code == 0, err
        assert "[IMPROVE_RERUN] " in out

    @pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
    @pytest.mark.parametrize("mode, line", [
        (PLAN, "[ALREADY_COMPLETE] improve-plan: review-plan completed "
               "2026-10-01T12:00:00, apply-suggestions completed "
               "2026-10-01T12:05:00; nothing to do. --force runs review-plan again."),
        (CODE, "[ALREADY_COMPLETE] improve-code: review-code completed "
               "2026-10-01T12:00:00, apply-code-fixes completed "
               "2026-10-01T12:05:00; nothing to do. --force runs review-code again."),
    ], ids=MODES)
    def test_a_finished_plan_is_exit_2_with_the_pinned_line(
        self, repo, capsys, mode, line, as_json
    ):
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)})
        if as_json:
            code, payload, _err = _json(capsys, plan, mode)
            assert code == 2
            assert payload["already_complete"] == line[len("[ALREADY_COMPLETE] "):]
            assert payload["rerun"] is None and payload["rerun_argv"] is None
            assert [s["decision"] for s in payload["steps"]] == ["skip", "skip"]
        else:
            code, out, _err = _run(capsys, plan, mode)
            assert code == 2
            assert out.splitlines()[-1] == line
            assert "[IMPROVE_RERUN]" not in out

    @pytest.mark.parametrize("via", ["main", "dispatcher"])
    @pytest.mark.parametrize("as_json", [[], ["--json"]], ids=["text", "json"])
    def test_an_empty_models_value_is_exit_3_row_15(self, repo, capsys, via, as_json):
        """The ``util`` route skips the mode route's row-15 check, so without
        its own call ``--models ""`` would plan the default model list."""
        _plan(repo, name="x")
        argv = ["plans/x.md", "--mode", PLAN, "--models", "", *as_json]
        if via == "main":
            assert improve_steps.main(argv) == 3
            out = capsys.readouterr()
            assert out.out == ""
        else:
            assert multi_llm.main(["util", "improve-steps", *argv]) == 3
            out = capsys.readouterr()
            body = out.out.rsplit(cli_common.RESULT_MARKER, 1)[0]
            assert body == ""
        assert "--models was passed an empty value" in out.err

    def test_an_unknown_host_is_exit_3(self, repo, capsys):
        plan = _plan(repo)
        code, out, err = _run(capsys, plan, PLAN, "--host", "nope")
        assert (code, out) == (3, "")
        assert "unknown host 'nope'" in err

    def test_a_missing_plan_is_exit_1_as_on_the_executor_leg(self, repo, capsys):
        code, out, err = _run(capsys, "plans/missing.md", PLAN)
        assert (code, out) == (1, "")
        assert "ERROR: Plan file not found: plans/missing.md" in err


class TestInteractiveIsRefused:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("spelling", ["--interactive", "-i", "--interac"])
    @pytest.mark.parametrize("beside", [[], ["--agent-decide"], ["--claude-decide"],
                                        ["--let-claude-decide"], ["--no-confirm"],
                                        ["--json"], ["--yes"]],
                             ids=["alone", "agent-decide", "claude-decide",
                                  "let-claude-decide", "no-confirm", "json", "yes"])
    def test_every_spelling_is_exit_3_with_the_pinned_line(
        self, repo, capsys, mode, spelling, beside
    ):
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        before = _tree(repo)
        code, out, err = _run(capsys, plan, mode, spelling, *beside)
        assert (code, out) == (3, "")
        assert err.strip().splitlines() == [REFUSAL[mode]]
        assert _tree(repo) == before, "a refusal writes nothing"

    def test_the_accepted_spellings_alone_plan_normally(self, repo, capsys):
        plan = _plan(repo)
        for flag in ("--agent-decide", "--claude-decide", "--let-claude-decide",
                     "--no-confirm"):
            code, out, err = _run(capsys, plan, PLAN, flag)
            assert code == 0, err
            assert _flags(out, "2") == APPLY_FLAGS

    def test_through_the_dispatcher_too(self, repo, capsys):
        _plan(repo, name="x")
        assert multi_llm.main(
            ["util", "improve-steps", "plans/x.md", "--mode", CODE, "-i"]) == 3
        assert REFUSAL[CODE] in capsys.readouterr().err.splitlines()


class TestALiveRun:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("where", ["review", "apply", "full", "own"])
    @pytest.mark.parametrize("flags", [[], ["--force"], ["--resume"], ["--json"]],
                             ids=["bare", "force", "resume", "json"])
    def test_a_live_owner_is_exit_3_naming_its_commands(
        self, repo, capsys, mode, where, flags
    ):
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        dirs = {"review": REVIEW_PHASE[mode], "apply": APPLY_DIR[mode],
                "full": "full", "own": mode}
        lock = _live_run(_out(plan) / dirs[where], "run-live-0001", "review-plan")
        try:
            before = _tree(repo)
            code, out, err = _run(capsys, plan, mode, *flags)
            after = _tree(repo)
        finally:
            lock.release()
        assert (code, out) == (3, "")
        assert f"ERROR: {mode}: a live run owns " in err
        assert f"{dirs[where]}/ (run run-live-0001, pid {os.getpid()}, running" in err
        assert "Watch it: " in err and "--run-id run-live-0001" in err
        assert "Abort it: " in err and "--abort --run-id run-live-0001" in err
        assert after == before, "read-only: nothing written, no lock file made"

    def test_a_dead_holder_blocks_nothing(self, repo, capsys):
        plan = _plan(repo)
        review_dir = _out(plan) / "review-plan"
        review_dir.mkdir(parents=True)
        status_file.claim_run(str(review_dir), {
            "run_id": "run-dead-0001", "pid": _dead_pid(),
            "process_start_time": "12345", "state": "running", "detached": True,
            "heartbeat_at": "2026-10-01T11:00:00",
        })
        code, _out_text, err = _run(capsys, plan, PLAN)
        assert code == 0, err
        assert status_file.read_run(str(review_dir))["state"] == "running", (
            "classify_run(close_out=False) closed nothing out")


class TestAnUnreadableStateFile:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("fault", list(STATE_FAULTS))
    @pytest.mark.parametrize("as_json", [[], ["--json"]], ids=["text", "json"])
    def test_it_is_refused_and_left_as_it_was(
        self, repo, capsys, mode, fault, as_json
    ):
        plan = _plan(repo)
        state_file = _out(plan) / "state.json"
        state_file.parent.mkdir(parents=True)
        text, detail = STATE_FAULTS[fault]
        state_file.write_text(text, encoding="utf-8")
        before = _tree(repo)
        code, out, err = _run(capsys, plan, mode, *as_json)
        assert (code, out) == (3, ""), "nothing on stdout, --json or not"
        [line] = [l for l in err.splitlines() if l.startswith("ERROR:")]
        assert line.startswith(f"ERROR: {mode}: cannot read {_shown(state_file)} (")
        assert detail in line
        assert line.endswith(
            "; nothing was run. Repair the file, or move it aside to start the "
            "plan's records over, then run the command again.")
        assert "Traceback" not in err
        assert _tree(repo) == before

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("shape", ["no-phase-keys", "missing"])
    def test_missing_keys_and_a_missing_file_plan_as_no_records(
        self, repo, capsys, mode, shape
    ):
        plan = _plan(repo)
        if shape == "no-phase-keys":
            _out(plan).mkdir(parents=True)
            (_out(plan) / "state.json").write_text('{"tracked_files": []}')
        code, out, err = _run(capsys, plan, mode)
        assert code == 0, err
        review = mode_table.resolve(mode).sequence[0]
        assert f"[IMPROVE_STEP] 1 {review} run (not reviewed yet)" in out


# ---------------------------------------------------------------------------
# 3. What it prints, state by state
# ---------------------------------------------------------------------------

class TestANewRound:
    def test_force_on_a_finished_plan_prints_the_plans_example_block(
        self, repo, capsys
    ):
        plan = _plan(repo, name="x")
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(5)})
        code, out, err = _run(capsys, "plans/x.md", PLAN, "--force")
        assert code == 0, err
        assert out.splitlines() == [
            "[IMPROVE_STEP] 1 review-plan run (new round, --force)",
            "[IMPROVE_STEP_FRESH] 1",
            "[IMPROVE_STEP_FLAGS] 1 --yes --force --rerun-all",
            "[IMPROVE_STEP] 1b consolidate after-review (runs if the review "
            "recommends it)",
            "[IMPROVE_STEP_FLAGS] 1b --consolidate --yes",
            "[IMPROVE_STEP] 2 apply-suggestions run (review not applied yet)",
            "[IMPROVE_STEP_FLAGS] 2 --yes --agent-decide --no-confirm",
            "[IMPROVE_RERUN] /multi-llm:multi-llm --improve-plan plans/x.md --resume",
        ]

    @pytest.mark.parametrize("mode", MODES)
    def test_force_resume_prints_the_same_step_2_line(self, repo, capsys, mode):
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)},
                     started={APPLY_PHASE[mode]: T(1)})
        code, out, _err = _run(capsys, plan, mode, "--force", "--resume")
        assert code == 0
        assert _flags(out, "1") == ["--yes", "--force", "--rerun-all"]
        assert _flags(out, "2") == APPLY_FLAGS, "no --resume after a review that ran"
        assert _rerun_line(out).endswith(" --resume"), "the continuation drops --force"
        assert "--force" not in _rerun_line(out)


class TestScope:
    def _reviewed(self, repo, scope=None, apply_done=False):
        plan = _plan(repo)
        completed = {"code-review": T(0)}
        if apply_done:
            completed["apply-code-fixes"] = T(5)
        extra = {"review_scope_files": list(scope)} if scope is not None else None
        _write_state(plan, completed=completed, extra=extra)
        return plan

    @pytest.mark.parametrize("resume", [[], ["--resume"]], ids=["bare", "resume"])
    def test_a_typed_scope_over_the_default_one_reviews_again(
        self, repo, capsys, resume
    ):
        plan = self._reviewed(repo, apply_done=True)
        code, out, _err = _run(capsys, plan, CODE, "--files", "a.py", *resume)
        assert code == 0, "never exit 2: the scope asks for another review"
        lines = out.splitlines()
        assert lines[0] == (
            "[IMPROVE_SCOPE_CHANGED] improve-code: the code review on record "
            "reviewed the default scope (tracked_files, else the git diff; no "
            "review_scope_files recorded); this run asks for 1 files "
            "(--files/--files-from). Reviewing again (as --force)."
        )
        assert lines[1:3] == [
            "[IMPROVE_STEP] 1 review-code run (review scope changed)",
            "[IMPROVE_STEP_FRESH] 1",
        ]
        assert _flags(out, "1") == ["--yes", "--force", "--files", "a.py"]
        assert _flags(out, "2") == APPLY_FLAGS

    def test_the_same_set_in_another_order_skips_the_review(self, repo, capsys):
        plan = self._reviewed(repo, scope=["a.py", "b.py"])
        _code, out, _err = _run(capsys, plan, CODE, "--files", "b.py", "--files", "a.py")
        assert "[IMPROVE_STEP] 1 review-code skip (completed 2026-10-01T12:00:00)" in out
        _code, out, _err = _run(capsys, plan, CODE, "--files", "a.py")
        assert "[IMPROVE_STEP] 1 review-code run (review scope changed)" in out
        assert "2 files given with --files (state.json review_scope_files)" in out

    def test_an_unfinished_review_resumes_its_recorded_scope(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, extra={"review_scope_files": ["b.py", "a.py"]})
        code, payload, _err = _json(capsys, plan, CODE)
        assert code == 0
        assert payload["steps"][0]["argv"] == [
            "--yes", "--files", "b.py", "--files", "a.py"]
        assert payload["rerun_argv"] == ["--resume"], "the record is never carried"

    @pytest.mark.parametrize("apply_done, code", [(False, 0), (True, 2)],
                             ids=["apply-left", "finished"])
    def test_files_from_stdin_on_a_completed_review_is_a_note(
        self, repo, capsys, monkeypatch, apply_done, code
    ):
        monkeypatch.setattr(sys, "stdin", NoStdin())
        plan = self._reviewed(repo, apply_done=apply_done)
        got, out, _err = _run(capsys, plan, CODE, "--files-from", "-")
        assert got == code
        lines = out.splitlines()
        assert lines[0] == (
            "NOTE: improve-code does not compare --files-from - with the scope on "
            "record; the completed code review stands. Pass the list with --files "
            "or --files-from <file>, or add --force, to review it."
        )
        assert lines[1].startswith("[IMPROVE_STEP] 1 review-code skip (completed ")

    @pytest.mark.parametrize("extra", [[], ["--files", "a.py"]], ids=["alone", "beside-files"])
    @pytest.mark.parametrize("force", [False, True], ids=["fresh", "force"])
    @pytest.mark.parametrize("as_json", [[], ["--json"]], ids=["text", "json"])
    def test_files_from_stdin_for_a_review_planned_run_is_exit_1(
        self, repo, capsys, monkeypatch, extra, force, as_json
    ):
        monkeypatch.setattr(sys, "stdin", NoStdin())
        plan = _plan(repo)
        if force:
            _write_state(plan, completed={"code-review": T(0),
                                          "apply-code-fixes": T(5)})
        before = _tree(repo)
        code, out, err = _run(capsys, plan, CODE, "--files-from", "-", *extra,
                              *(["--force"] if force else []), *as_json)
        assert (code, out) == (1, "")
        assert err.splitlines() == [
            "ERROR: --files-from - reads stdin, which a --detach worker does not "
            "inherit; write the list to a file and pass its path instead"
        ]
        assert _tree(repo) == before

    @pytest.mark.parametrize("scope, says", [
        (["--files-from", "missing.txt"], "--files-from missing.txt: cannot read it"),
        (["--files", "/elsewhere/a.py"], "--files path is outside the repository"),
    ], ids=["unreadable-list", "outside"])
    def test_a_scope_that_cannot_be_read_is_exit_1(self, repo, capsys, scope, says):
        plan = _plan(repo)
        before = _tree(repo)
        code, out, err = _run(capsys, plan, CODE, *scope)
        assert (code, out) == (1, "")
        assert says in err
        assert _tree(repo) == before


class TestLeftoverValidation:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_review_stopped_before_its_regroup_is_fresh(self, repo, capsys, mode):
        plan = _plan(repo)
        _write_state(plan, completed={APPLY_PHASE[mode]: T(5)})
        review_dir = _out(plan) / REVIEW_PHASE[mode]
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "validation.json").write_text("[]", encoding="utf-8")
        code, out, _err = _run(capsys, plan, mode)
        assert code == 0
        review = mode_table.resolve(mode).sequence[0]
        assert f"[IMPROVE_STEP] 1 {review} run (review unfinished)" in out
        assert "[IMPROVE_STEP_FRESH] 1" in out.splitlines()
        assert _flags(out, "1") == (["--yes", "--force"] if mode == CODE else ["--yes"])
        assert _flags(out, "2") == APPLY_FLAGS
        # The same stopped round re-run with --force is another round.
        _code, out, _err = _run(capsys, plan, mode, "--force")
        assert _flags(out, "1") == ["--yes", "--force", "--rerun-all"]

    def test_a_retired_round_does_not_count(self, repo, capsys):
        plan = _plan(repo)
        review_dir = _out(plan) / "code-review"
        review_dir.mkdir(parents=True)
        (review_dir / "validation.20261001T120000.stale.json").write_text("[]")
        _code, out, _err = _run(capsys, plan, CODE)
        assert "[IMPROVE_STEP_FRESH]" not in out
        assert _flags(out, "1") == ["--yes"]

    def test_a_stale_review_is_fresh_and_gets_no_force(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "implement": T(5)})
        review_dir = _out(plan) / "code-review"
        review_dir.mkdir(parents=True)
        (review_dir / "validation.json").write_text("[]")
        _code, out, _err = _run(capsys, plan, CODE)
        assert ("[IMPROVE_STEP] 1 review-code run (reviewed before the latest "
                "implement)") in out
        assert "[IMPROVE_STEP_FRESH] 1" in out.splitlines()
        assert _flags(out, "1") == ["--yes"], "a stale review forces itself"


class TestConsolidation:
    def test_an_open_recommendation_reads_run(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)})
        _review_files(plan, 15)
        code, out, _err = _run(capsys, plan, PLAN)
        assert code == 0
        assert "[IMPROVE_STEP] 1b consolidate run" in out.splitlines()
        assert _flags(out, "1b") == ["--consolidate", "--yes"]
        _code, payload, _err = _json(capsys, plan, PLAN)
        assert payload["steps"][1]["reason"] is None

    @pytest.mark.parametrize("actionable, consolidated, reason", [
        (15, True, "already ran on these review groups"),
        (14, False, "not recommended by review-plan"),
        (0, False, "not recommended by review-plan"),
    ], ids=["consolidated", "fourteen", "zero"])
    def test_no_pass_reads_skip_with_the_executors_detail(
        self, repo, capsys, actionable, consolidated, reason
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)})
        _review_files(plan, actionable, consolidated)
        _code, out, _err = _run(capsys, plan, PLAN)
        assert f"[IMPROVE_STEP] 1b consolidate skip ({reason})" in out.splitlines()

    def test_a_review_that_runs_reads_after_review(self, repo, capsys):
        plan = _plan(repo)
        _review_files(plan, 15)
        _code, out, _err = _run(capsys, plan, PLAN)
        assert ("[IMPROVE_STEP] 1b consolidate after-review (runs if the review "
                "recommends it)") in out.splitlines()

    def test_a_finished_run_prints_no_1b_with_a_recommendation_open(
        self, repo, capsys
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(5)})
        _review_files(plan, 15)
        code, payload, _err = _json(capsys, plan, PLAN)
        assert code == 2
        assert [s["index"] for s in payload["steps"]] == ["1", "2"]

    def test_improve_code_prints_no_1b(self, repo, capsys):
        plan = _plan(repo)
        _code, payload, _err = _json(capsys, plan, CODE)
        assert [s["index"] for s in payload["steps"]] == ["1", "2"]


class TestTheApplyStep:
    def test_an_apply_in_progress_gets_resume_untyped(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)},
                     started={"apply-suggestions": T(5)})
        _code, out, _err = _run(capsys, plan, PLAN)
        assert ("[IMPROVE_STEP] 2 apply-suggestions run (apply in progress for "
                "this review)") in out
        assert _flags(out, "2") == APPLY_FLAGS + ["--resume"]
        _code, out, _err = _run(capsys, plan, PLAN, "--resume")
        assert _flags(out, "2").count("--resume") == 1

    def test_an_executor_batch_on_record_resumes_too(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        state = StateManager(plan)
        state.update_task_status("apply-fix-batch-1", "completed")
        state.save()
        _code, out, _err = _run(capsys, plan, CODE)
        assert _flags(out, "2")[:4] == APPLY_FLAGS + ["--resume"]

    def test_the_recorded_base_ref_reaches_an_apply_only_rerun(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        _write_manifest(plan, "origin/release")
        _code, out, _err = _run(capsys, plan, CODE)
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", "origin/release"]
        _code, out, _err = _run(capsys, plan, CODE, "--base-ref", "main")
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", "main"], (
            "a typed ref wins and is not doubled")
        _code, out, _err = _run(capsys, plan, CODE, "--force")
        assert "--base-ref" not in _flags(out, "2"), (
            "when step 1 runs nothing is read from the manifest")


# ---------------------------------------------------------------------------
# 4. The host of [IMPROVE_RERUN]
# ---------------------------------------------------------------------------

def _spelled(host: str, args: str, route: Optional[str] = None) -> str:
    return host_capabilities.next_command_template(host, route).replace("{args}", args)


class TestTheHost:
    def test_the_codex_shims_baked_tokens_parse_and_exit_0(self, repo, capsys):
        _plan(repo, name="x")
        code, out, err = _run(capsys, "plans/x.md", PLAN, *CODEX_SHIM)
        assert code == 0, err
        assert _flags(out, "1") == ["--yes", "--exec-subagents"]
        assert _flags(out, "2") == APPLY_FLAGS + ["--exec-subagents"]
        assert _rerun_line(out) == _spelled("codex", "--improve-plan plans/x.md --resume")

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_each_hosts_default_spelling(self, repo, capsys, host):
        _plan(repo, name="x")
        _code, out, _err = _run(capsys, "plans/x.md", CODE, "--host", host, "--quick")
        assert _rerun_line(out) == _spelled(
            host, "--improve-code plans/x.md --quick --resume")

    def test_no_host_is_claude_codes_spelling(self, repo, capsys):
        _plan(repo, name="x")
        _code, out, _err = _run(capsys, "plans/x.md", PLAN)
        assert _rerun_line(out) == f"{PLUGIN}--improve-plan plans/x.md --resume"

    def test_the_environment_declares_a_host_when_no_flag_does(
        self, repo, capsys, monkeypatch
    ):
        monkeypatch.setenv("MULTI_LLM_HOST", "codex")
        _plan(repo, name="x")
        _code, out, _err = _run(capsys, "plans/x.md", PLAN)
        assert _rerun_line(out) == _spelled("codex", "--improve-plan plans/x.md --resume")

    def test_a_shell_reads_the_route_the_composites_run_was_started_on(
        self, repo, capsys
    ):
        plan = _plan(repo, name="x")
        run_dir = _out(plan) / PLAN
        run_dir.mkdir(parents=True)
        status_file.claim_run(str(run_dir), {
            "run_id": "run-0001", "state": "failed", "exit_code": 3,
            "detached": True, "mode": PLAN, "host": "codex", "route": "package",
            "started_at": "2026-10-07T10:00:00",
        })
        _code, out, _err = _run(capsys, "plans/x.md", PLAN, "--host", "codex")
        assert _rerun_line(out) == _spelled(
            "codex", "--improve-plan plans/x.md --resume", "package")

    @pytest.mark.parametrize("mode", MODES)
    def test_rerun_never_carries_a_host_token(self, repo, capsys, mode):
        plan = _plan(repo)
        _code, payload, _err = _json(capsys, plan, mode, *CODEX_SHIM,
                                     "--dry-run-subagents", "--models", "a,b")
        assert payload["rerun_argv"] == ["--models", "a,b", "--resume"]
        for token in ("--host", "--yes", "--exec-subagents", "--dry-run-subagents"):
            assert token not in shlex.split(payload["rerun"].split(":", 1)[-1])
            assert token not in payload["rerun_argv"]


# ---------------------------------------------------------------------------
# 5. Token-safe rendering
# ---------------------------------------------------------------------------

class TestTokenSafeRendering:
    NAMES = ["a b.py", "it's.py", 'say "hi".py']

    def _typed_scope(self, repo):
        listing = repo / LISTING_NAME
        listing.write_text("c.py\n", encoding="utf-8")
        argv: List[str] = []
        for name in self.NAMES:
            argv += ["--files", name]
        argv += ["--files-from", str(listing)]
        return argv

    def test_each_flags_line_splits_back_to_the_argv(self, repo, capsys):
        plan = _plan(repo, name="spaced plan")
        argv = self._typed_scope(repo)
        code, out, _err = _run(capsys, plan, CODE, *argv, "--base-ref", "it's ref")
        assert code == 0
        assert _flags(out, "1") == ["--yes", *argv, "--base-ref", "it's ref"]
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", "it's ref"]
        _code, payload, _err = _json(capsys, plan, CODE, *argv, "--base-ref", "it's ref")
        assert payload["steps"][0]["argv"] == ["--yes", *argv, "--base-ref", "it's ref"]
        assert payload["rerun_argv"] == [*argv, "--base-ref", "it's ref", "--resume"]

    def test_each_rerun_token_is_its_quote_plan_arg_spelling(self, repo, capsys):
        plan = _plan(repo, name="spaced plan")
        argv = self._typed_scope(repo)
        _code, out, _err = _run(capsys, plan, CODE, *argv)
        flags = [*argv, "--resume"]
        expected = " ".join(
            ["--improve-code", next_phase.quote_plan_arg(_typed(plan)),
             *(next_phase.quote_plan_arg(t) for t in flags)])
        assert _rerun_line(out) == f"{PLUGIN}{expected}"
        assert '"a b.py"' in _rerun_line(out) and "'say \"hi\".py'" in _rerun_line(out)

    def test_phase_command_leaves_todays_tokens_bare(self):
        """Every token today's callers pass has no whitespace and no quote, so
        their output does not change."""
        flags = ["--resume", "--human-task", "T003=done", "--human-task",
                 "{task_id}=skip", "apply-batch-3=skip"]
        command = next_phase.phase_command("plans/p.md", "claude-code", None,
                                           "implement", *flags)
        assert command == f"{PLUGIN}--implement plans/p.md {' '.join(flags)}"
        spaced = next_phase.phase_command("plans/p.md", "claude-code", None,
                                          "improve-code", "--files", "a b.py")
        assert spaced == f'{PLUGIN}--improve-code plans/p.md --files "a b.py"'


# ---------------------------------------------------------------------------
# 6. --json: the pinned shape, and agreement with the text block
# ---------------------------------------------------------------------------

def _state_fresh(repo, mode):
    return _plan(repo), ()


def _state_reviewed(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
    return plan, ()


def _state_applying(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)},
                 started={APPLY_PHASE[mode]: T(5)})
    return plan, ()


def _state_new_round(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0), APPLY_PHASE[mode]: T(5)})
    return plan, ("--force", "--quick")


def _state_older_apply(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={APPLY_PHASE[mode]: T(-5), REVIEW_PHASE[mode]: T(0)})
    return plan, ("--models", "a,b")


def _state_leftover(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={APPLY_PHASE[mode]: T(5)})
    review_dir = _out(plan) / REVIEW_PHASE[mode]
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "validation.json").write_text("[]", encoding="utf-8")
    return plan, ("--resume",)


def _state_consolidate(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
    if mode == PLAN:
        _review_files(plan, 15)
    return plan, ()


def _state_scope_changed(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0), APPLY_PHASE[mode]: T(5)})
    return plan, (("--files", "a.py") if mode == CODE else ("--force",))


def _state_base_ref(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
    if mode == CODE:
        _write_manifest(plan, "origin/release")
    return plan, ()


#: The parity matrix: one state builder per row, each run for both modes.
MATRIX = {
    "fresh": _state_fresh,
    "reviewed": _state_reviewed,
    "applying": _state_applying,
    "new-round": _state_new_round,
    "older-apply": _state_older_apply,
    "leftover-validation": _state_leftover,
    "consolidate": _state_consolidate,
    "scope-changed": _state_scope_changed,
    "base-ref": _state_base_ref,
}


def _state_finished(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0), APPLY_PHASE[mode]: T(5)})
    return plan, ()


def _state_files_from_stdin(repo, mode):
    plan = _plan(repo)
    _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
    return plan, (("--files-from", "-") if mode == CODE else ())


AGREEMENT = {**MATRIX, "finished": _state_finished,
             "files-from-stdin": _state_files_from_stdin}


class TestTheJsonObject:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("state", sorted(AGREEMENT))
    def test_it_has_exactly_the_pinned_keys_and_agrees_with_the_text(
        self, repo, capsys, monkeypatch, mode, state
    ):
        monkeypatch.setattr(sys, "stdin", NoStdin())
        plan, argv = AGREEMENT[state](repo, mode)
        text_code, out, _err = _run(capsys, plan, mode, *argv)
        json_code, payload, _err = _json(capsys, plan, mode, *argv)
        assert text_code == json_code and text_code in (0, 2)

        assert tuple(payload) == improve_steps.JSON_KEYS
        assert "schema_version" not in payload
        assert payload["mode"] == mode and payload["plan"] == _typed(plan)
        assert isinstance(payload["notices"], list)
        assert all(isinstance(n, str) for n in payload["notices"])
        assert payload["already_complete"] is None or isinstance(
            payload["already_complete"], str)
        assert payload["rerun"] is None or isinstance(payload["rerun"], str)
        assert (payload["rerun"] is None) == (payload["rerun_argv"] is None)
        for step in payload["steps"]:
            assert tuple(step) == improve_steps.STEP_KEYS
            assert all(isinstance(step[k], str) for k in ("index", "step", "decision"))
            assert step["reason"] is None or isinstance(step["reason"], str)
            assert isinstance(step["fresh"], bool)
            assert all(isinstance(t, str) for t in step["argv"])

        parsed = _parse_text(out)
        assert payload["steps"] == parsed["steps"]
        assert payload["notices"] == parsed["notices"]
        assert payload["already_complete"] == parsed["already_complete"]
        assert payload["rerun"] == parsed["rerun"]
        assert (text_code == 2) == (payload["already_complete"] is not None)
        if payload["rerun"] is not None:
            assert payload["rerun"].endswith(" ".join(
                next_phase.quote_plan_arg(t) for t in payload["rerun_argv"]))


# ---------------------------------------------------------------------------
# 7. Parity with the executor leg
# ---------------------------------------------------------------------------

def _without_host(argv: List[str]) -> List[str]:
    """*argv* with its one ``--host <id>`` pair removed (the only normalization)."""
    out = list(argv)
    if "--host" in out:
        index = out.index("--host")
        del out[index:index + 2]
    assert "--host" not in out
    return out


class TestParityWithTheExecutor:
    @pytest.mark.parametrize("cross", [(), CROSS], ids=["plain", "cross-cutting"])
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("state", sorted(MATRIX))
    def test_the_planners_argv_is_what_the_driver_forwards(
        self, repo, capsys, stub, mode, state, cross
    ):
        plan, argv = MATRIX[state](repo, mode)
        argv = (*argv, *cross)
        code, payload, err = _json(capsys, plan, mode, *argv)
        assert code == 0, err

        # The driver's hook answers the same run/skip on the same records.
        args = _args(mode, plan, *argv)
        records = iw.read_records(mode, plan, args)
        workflow = iw.ImproveWorkflow(_args(mode, plan, *argv), mode, records=records)
        hook = {
            step.phase: ("skip" if workflow.skip_reason(step) else "run")
            for step in workflow.steps() if step.phase
        }
        planned = {mode_table.phase_for(s["step"]): s["decision"]
                   for s in payload["steps"] if s["step"] != iw.CONSOLIDATE}
        assert planned == hook
        assert payload["rerun_argv"] == workflow.resume_flags()

        calls = stub()
        assert ENTRY[mode].main(["--plan-file", str(plan), *argv]) == 0
        forwarded = {}
        for word, extra, _held in calls:
            key = iw.CONSOLIDATE if "--consolidate" in extra else word
            forwarded[key] = _without_host(extra)
        for step in payload["steps"]:
            if step["decision"] == "run":
                assert forwarded.pop(step["step"]) == step["argv"], step
        assert not forwarded, f"the driver ran steps the planner did not: {forwarded}"

        for step in payload["steps"]:
            for token in step["argv"]:
                assert token not in NEVER_FORWARDED and not token.startswith(
                    "--internal-"), (step, token)
            if cross:
                assert step["argv"][-2:] == ["--exec-subagents", "--dry-run-subagents"]
        for token in payload["rerun_argv"]:
            assert token not in NEVER_FORWARDED and token not in (
                "--yes", "--exec-subagents", "--dry-run-subagents")

    @pytest.mark.parametrize("mode", MODES)
    def test_typed_route_flags_are_never_printed(self, repo, capsys, mode):
        plan = _plan(repo)
        code, payload, err = _json(
            capsys, plan, mode, "--host", "codex", "--detach", "--if-not-running",
            "--max-detached", "2", "--internal-worker", "--internal-run-id", "r-1",
            "--skip-prerequisites", "--yes",
        )
        assert code == 0, err
        for step in payload["steps"]:
            assert not set(step["argv"]) & set(NEVER_FORWARDED)
            assert not any(t.startswith("--internal-") for t in step["argv"])
            assert step["argv"].count("--yes") == 1
        assert payload["rerun_argv"] == ["--resume"]

    def test_both_legs_plan_through_the_one_skip_reason(
        self, repo, capsys, stub, monkeypatch
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(5)})
        code, _out_text, _err = _run(capsys, plan, PLAN)
        assert code == 2
        monkeypatch.setattr(iw, "skip_reason", lambda *a, **k: None)
        code, payload, _err = _json(capsys, plan, PLAN)
        assert code == 0
        assert [s["decision"] for s in payload["steps"]] == ["run", "after-review", "run"]
        calls = stub()
        assert ENTRY[PLAN].main(["--plan-file", str(plan)]) == 0
        assert [w for w, _a, _h in calls] == ["review-plan", "apply-suggestions"]

    def test_the_planner_builds_no_improve_workflow(self, repo, capsys, monkeypatch):
        def refuse(*a, **k):
            pytest.fail("the planner built a FullWorkflow")

        monkeypatch.setattr(fwo.FullWorkflow, "__init__", refuse)
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        for argv in ((), ("--force",)):
            code, _out_text, err = _run(capsys, plan, CODE, *argv)
            assert code == 0, err
        _write_state(plan, completed={"code-review": T(0), "apply-code-fixes": T(5)})
        assert _run(capsys, plan, CODE)[0] == 2


# ---------------------------------------------------------------------------
# 8. The native leg, end to end
# ---------------------------------------------------------------------------
#
# The dispatcher calls in the order the cores make them, through
# ``multi_llm.py`` as a host runs it, against the mock provider binaries. On
# Claude Code each phase runs without ``--exec-subagents`` and the host's
# subagents are ``SkillRunner.simulate_subagents``; on the Codex file shim each
# call carries the shim's baked ``--host codex --yes --exec-subagents``.

#: Two reviewers, so "every model answered `[]`" is more than one slot.
REVIEWERS = ("--models", "cursor-agent:auto", "gemini:gemini-2.5-flash")

PLAN_BODY = """# Upload Plan

## Overview
Upload the user's files to the archive.

## Step 1
Retry a failed upload.
"""

SUGGESTION = {
    "title": "Bound the upload retry",
    "desc": "Step 1 retries a failed upload without saying how often. Name the "
            "number of attempts and the backoff between them.",
    "type": "clarification",
    "reference": "## Step 1",
    "importance": "MEDIUM",
}

CODE_ISSUES = [
    {
        "title": "Unchecked upload status",
        "desc": "The client ignores the HTTP status of the upload response.",
        "importance": "medium",
        "file": "src/upload/client.py",
        "line_range": [10, 12],
        "type": "bug",
        "anchor_text": "def send(path):",
    },
    {
        "title": "Retry loop has no bound",
        "desc": "The retry loop never stops on a permanent failure.",
        "importance": "high",
        "file": "src/upload/retry.py",
        "line_range": [4, 9],
        "type": "bug",
        "anchor_text": "while True:",
    },
]


def _verdicts(status: str):
    """A ``result_for`` that judges every group of a validation task *status*."""
    def result_for(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "groups": [
                {"group_index": index, "group_hash": group_hash, "status": status,
                 "reason": f"simulated: {status}", "confidence": 0.9}
                for index, group_hash in zip(
                    task.get("group_indices", []), task.get("group_ids", []))
            ],
            "metadata": {"model": "harness", "timestamp": "1970-01-01T00:00:00",
                         "schema_version": "2.1"},
        }

    return result_for


def validation_owed(out: str) -> bool:
    """Whether a review launch stopped at its validation hand-off (review-plan
    prints ``[VALIDATION_PENDING]``, review-code ``[VALIDATION_BATCHES_PENDING]``)."""
    return any(marker in out for marker in (
        "[VALIDATION_PENDING]", "[VALIDATION_BATCHES_PENDING]"))


class NativeLeg:
    """One plan driven through the dispatcher as a host's core drives it."""

    def __init__(self, tmp_path: Path, mode: str, host: Tuple[str, ...]):
        self.mode = mode
        self.host = tuple(host)
        self.runner = SkillRunner(tmp_path / "harness")
        self.plan = tmp_path / "upload-plan.md"
        self.plan.write_text(PLAN_BODY, encoding="utf-8")
        review, apply = mode_table.resolve(mode).sequence
        self.review, self.apply = review, apply
        self.review_dir = _out(self.plan) / mode_table.phase_for(review)
        if mode == CODE:
            self._implemented()

    def _implemented(self) -> None:
        """What review-code needs on record: an implement that tracked files."""
        state = StateManager(self.plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("implement")
        state.set("head_at_start", "abc123def")
        for issue in CODE_ISSUES:
            state.track_file(issue["file"], "created", "T001")
        state.save()
        implement = _out(self.plan) / "implement"
        implement.mkdir(parents=True, exist_ok=True)
        (implement / "summary.json").write_text(json.dumps(
            {"status": "completed", "tasks_completed": 2, "tasks_total": 2}),
            encoding="utf-8")

    def respond(self, response: Any) -> None:
        self.runner.set_config({"response": response})

    def call(self, *argv: str, timeout: int = 120):
        return self.runner._run_command(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"), *argv],
            self.plan, timeout, False,
        )

    def ok(self, result) -> str:
        assert result.exit_code == 0, (
            f"exit {result.exit_code}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
        return result.stdout

    def steps(self, *typed: str):
        """Section 1: ``util improve-steps`` with the host's ``--host``."""
        return self.call("util", "improve-steps", str(self.plan), "--mode", self.mode,
                         *self.host, *typed)

    def phase(self, word: str, *flags: str):
        return self.call(word, "--plan-file", str(self.plan), *self.host, *flags)

    def run(self, *typed: str, verdict: str = "valid") -> str:
        """Sections 1-3: plan, then run each ``run`` step's phase with its line."""
        out = self.ok(self.steps(*typed))
        assert f"[IMPROVE_STEP] 1 {self.review} run (" in out
        assert f"[IMPROVE_STEP] 2 {self.apply} run (" in out
        assert "[IMPROVE_STEP] 1b consolidate run" not in out
        launched = self.phase(self.review, *_flags(out, "1"))
        first = self.ok(launched)
        if validation_owed(first):
            assert "--exec-subagents" not in self.host
            self.runner.simulate_subagents(
                self.review_dir / "validation_tasks.json",
                result_for=_verdicts(verdict))
            last = self.ok(self.phase(self.review, "--reaggregate"))
        else:
            last = first
        result = cli_common.parse_result_line(last)
        assert "consolidation_recommended" not in result, "1b would run here"
        self.ok(self.phase(self.apply, *_flags(out, "2")))
        if "--exec-subagents" not in self.host:
            self.ok(self.phase(self.apply, "--mark-completed"))
        return out

    def state(self) -> Dict[str, Any]:
        return json.loads((_out(self.plan) / "state.json").read_text(encoding="utf-8"))

    def stamp(self, word: str) -> str:
        return self.state()["phases_completed"][mode_table.phase_for(word)]

    def assert_finished(self) -> str:
        """Both phases on record for this review, and a re-plan exits 2."""
        from datetime import datetime

        assert datetime.fromisoformat(self.stamp(self.apply)) > datetime.fromisoformat(
            self.stamp(self.review))
        again = self.steps()
        assert again.exit_code == 2, again.stdout + again.stderr
        assert f"[ALREADY_COMPLETE] {self.mode}: {self.review} completed " in again.stdout
        assert f"[IMPROVE_STEP] 2 {self.apply} skip (completed " in again.stdout
        return again.stdout

    def lookup(self, after: str, host: str = "claude-code") -> str:
        result = self.call("util", "next-command", str(self.plan), "--after", after,
                           "--host", host)
        return self.ok(result)


CLAUDE_CODE = ("--host", "claude-code")


@pytest.fixture
def no_host(monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_host")
class TestEmptyApplyOutcomesOnTheNativeLeg:
    """Zero findings, every finding invalid and an empty ``--force`` round each
    leave the apply recorded completed, newer than the review, and a second
    ``util improve-steps`` exits 2 (§2.3, *Empty apply outcomes*; §2.4)."""

    @pytest.mark.parametrize("mode", MODES)
    def test_zero_findings_from_a_review_whose_every_model_returned_nothing(
        self, tmp_path, mode
    ):
        leg = NativeLeg(tmp_path, mode, CLAUDE_CODE)
        leg.respond([])
        out = leg.run(*REVIEWERS)
        assert _flags(out, "1") == ["--yes", *REVIEWERS]
        groups = json.loads((leg.review_dir / "grouped.json").read_text())["groups"]
        assert groups == []
        leg.assert_finished()

    @pytest.mark.parametrize("mode", MODES)
    def test_every_finding_judged_invalid(self, tmp_path, mode):
        leg = NativeLeg(tmp_path, mode, CLAUDE_CODE)
        leg.respond([SUGGESTION] if mode == PLAN else CODE_ISSUES)
        leg.run(REVIEWERS[0], REVIEWERS[1], verdict="invalid")
        validation = json.loads((leg.review_dir / "validation.json").read_text())
        statuses = {g["status"] for g in validation["groups"]}
        assert statuses == {"invalid"}, statuses
        leg.assert_finished()

    @pytest.mark.parametrize("mode", MODES)
    def test_an_empty_force_round_over_an_older_apply_completion(self, tmp_path, mode):
        leg = NativeLeg(tmp_path, mode, CLAUDE_CODE)
        # Round 1: a real review with findings, validated, and its apply on record.
        leg.respond([SUGGESTION] if mode == PLAN else CODE_ISSUES)
        first = leg.ok(leg.phase(leg.review, "--yes", REVIEWERS[0], REVIEWERS[1]))
        assert validation_owed(first)
        leg.runner.simulate_subagents(leg.review_dir / "validation_tasks.json")
        leg.ok(leg.phase(leg.review, "--reaggregate"))
        state = StateManager(leg.plan)
        state.mark_phase_completed(mode_table.phase_for(leg.apply))
        state.save()
        assert leg.steps().exit_code == 2
        old_apply = leg.stamp(leg.apply)

        # Round 2: --force, every model answers `[]`. The planner's step-1 line
        # clears the completion and runs the review (never its exit-2 guard).
        leg.respond([])
        out = leg.run("--force", *REVIEWERS)
        assert _flags(out, "1") == ["--yes", *REVIEWERS, "--force", "--rerun-all"]
        assert _flags(out, "2")[:3] == APPLY_FLAGS and "--resume" not in _flags(out, "2")
        assert json.loads((leg.review_dir / "validation.json").read_text()) == []
        from datetime import datetime

        assert datetime.fromisoformat(leg.stamp(leg.apply)) > datetime.fromisoformat(
            old_apply)
        leg.assert_finished()


@pytest.mark.usefixtures("no_host")
class TestTheNativeLegsClosingLookup:
    """Each core's step 4 looks up ``--after <its own mode>``. These answers
    come through ``next_phase``'s ``start = mode.last_phase`` branch (§2.6,
    T5): until it lands ``--after improve-*`` answers nothing."""

    def test_an_improve_plan_run_names_generate_tasks(self, tmp_path):
        leg = NativeLeg(tmp_path, PLAN, CLAUDE_CODE)
        leg.respond([])
        leg.run(*REVIEWERS)
        leg.assert_finished()
        out = leg.lookup(PLAN)
        assert f"[NEXT_COMMAND] {PLUGIN}--generate-tasks {leg.plan}" in out.splitlines()

    def test_an_improve_code_run_is_complete_and_names_no_generate_tasks(
        self, tmp_path
    ):
        leg = NativeLeg(tmp_path, CODE, CLAUDE_CODE)
        leg.respond([])
        leg.run(*REVIEWERS)
        leg.assert_finished()
        out = leg.lookup(CODE)
        assert "[WORKFLOW_COMPLETE]" in out.splitlines()
        assert "--generate-tasks" not in out
        # The same state looked up with the other core's word walks on from
        # apply-suggestions: the wrong answer the first lookup keeps out.
        other = leg.lookup(PLAN)
        assert f"[NEXT_COMMAND] {PLUGIN}--generate-tasks {leg.plan}" in other.splitlines()


@pytest.mark.usefixtures("no_host")
class TestTheCodexFileShimLeg:
    """§3 and criterion 1's *File-shim route*: every call carries the shim's
    baked ``--host codex --yes --exec-subagents``, so each phase finishes in
    its one call and nothing else records it."""

    def test_both_phases_record_themselves_from_the_phase_calls_alone(self, tmp_path):
        leg = NativeLeg(tmp_path, PLAN, CODEX_SHIM)
        leg.respond([])
        out = leg.run(*REVIEWERS)
        assert _flags(out, "1") == ["--yes", *REVIEWERS, "--exec-subagents"]
        assert _rerun_line(out) == _spelled(
            "codex", f"--improve-plan {leg.plan} {' '.join(REVIEWERS)} --resume")
        state = leg.state()
        assert {"review-plan", "apply-suggestions"} <= set(state["phases_completed"])
        again = leg.assert_finished()
        assert "[IMPROVE_RERUN]" not in again

    def test_its_lookup_is_spelled_for_the_codex_shim(self, tmp_path):
        leg = NativeLeg(tmp_path, PLAN, CODEX_SHIM)
        leg.respond([])
        leg.run(*REVIEWERS)
        out = leg.lookup(PLAN, host="codex")
        assert ("[NEXT_COMMAND] "
                + _spelled("codex", f"--generate-tasks {leg.plan}")) in out.splitlines()
