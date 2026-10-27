"""The commit spelling at every site that prints a command (plans/review-commits.md
§2.8, T8b).

The two phase front doors re-record their run as the phase they front
(``review-commits`` as ``review-code``, ``apply-commit-fixes`` as
``apply-code-fixes``), so every builder that reads the record would print
``apply-code-fixes <brief> --base-ref <sha> --head-ref <sha>``: a fronted mode
word plus three flags the front doors refuse. The hook
(``utils/commit_spelling.py``) spells them back, and this file holds every
site §2.8 lists to it:

* ``multi_llm``'s ``user_next_step`` after a foreground ``--review-commits``
  is ``--apply-commit-fixes <pinned>``, under a why naming both reports;
* ``next_phase``'s builder, ``resume_command``, ``phase_command`` and
  ``util next-command`` (``--after`` / ``--blocked``, with the pinned range);
* ``run_status``'s resume line, next step, ``native_resume_command``, the
  native ``resume with:`` line and ``_composite_resume_tokens``;
* the row-14 stop (``FullWorkflow._phase_failed``) and so ``record_next``.

A root range's spelling is the brief path, and that printed command passes the
dispatcher's ``[COMMIT_FLAG_CONFLICT]`` scan and normalizes to ``--plan-file``.
A direct ``review-code --head-ref`` run is never mapped (§2.5), and the lines a
user reads call a brief "Commit brief", a plan "Plan".

Every fixture lives under ``tmp_path``: a seeded brief and ``state.json``
(the hook reads nothing else), or, for the dispatcher legs, a throwaway git
repository with the fronted mains monkeypatched, so nothing calls a model.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apply_code_fixes_orchestrator  # noqa: E402
import code_review_orchestrator  # noqa: E402
import full_workflow_orchestrator as fwo  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, improve_workflow, next_phase, run_status, status_file  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.commit_review import BRIEF_END_MARKER, BRIEF_MARKER, output_dir_for_brief  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

BASE = "a1b2c3d4e5f6" + "0" * 28
HEAD = "e4f5a6b7c8d9" + "1" * 28
PINNED = "a1b2c3d4e5f6..e4f5a6b7c8d9"
FOLDER = "plans/commit-reviews"
SLUG = "a1b2c3d4e5f6-e4f5a6b7c8d9"
ROOT_SLUG = "root-e4f5a6b7c8d9"
REL = f"{FOLDER}/{SLUG}.md"
ROOT_REL = f"{FOLDER}/{ROOT_SLUG}.md"
PLAN_REL = "plans/foo.md"

#: The six host/route rows (tests/test_next_phase.py's table), with the text
#: before ``{args}``.
ROUTES = [
    ("claude-code", None, "/multi-llm:multi-llm "),
    ("claude-code", "skill", "/multi-llm "),
    ("codex", None, "Use the multi-llm skill with exactly these arguments: "),
    ("codex", "package", "$multi-llm:multi-llm "),
    ("opencode", None, "/multi-llm "),
    ("opencode", "package", "/multi-llm "),
]
CC = "/multi-llm:multi-llm "

#: What a printed command never holds (§2.8).
SYNTHESISED = ("--plan-file", "--base-ref", "--head-ref", "--focus", "--context-plan",
               "--commit-stage", "--revs")

ONE_GROUP = {"format_version": 2, "groups": [
    {"group_id": "g1", "theme": "t", "suggestions": [{"title": "s", "importance": "HIGH"}]},
]}


def _clean(command: str) -> None:
    tokens = shlex.split(command)
    for flag in SYNTHESISED:
        assert not any(t == flag or t.startswith(flag + "=") for t in tokens), command
    for word in ("review-code", "apply-code-fixes", "--review-code", "--apply-code-fixes"):
        assert word not in tokens, command


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(root), check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def here(tmp_path, monkeypatch):
    """A git repository (the pinned spelling names its brief through the
    repository root) as the cwd, with no host or route in the environment."""
    root = Path(os.path.realpath(tmp_path))
    _git(root, "init", "--quiet", "-b", "main")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "-m", "init")
    monkeypatch.chdir(root)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    cli_common.clear_invocation()
    yield root
    cli_common.clear_invocation()


def _state(path: Path, *, completed=(), tasks=None, files=None) -> None:
    for name, content in (files or {}).items():
        target = output_dir_for_brief(path) / "code-review"
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(content), encoding="utf-8")
    state = StateManager(path)
    for phase in completed:
        state.mark_phase_completed(phase)
    for task_id, status in (tasks or {}).items():
        state.update_task_status(task_id, status)
    state.save()


def _inject(path: Path, key: str, value) -> None:
    state_file = output_dir_for_brief(path) / "state.json"
    document = json.loads(state_file.read_text(encoding="utf-8"))
    document[key] = value
    state_file.write_text(json.dumps(document), encoding="utf-8")


def _seed(root: Path, *, base=BASE, **state) -> Path:
    """A commit brief whose ``state.json`` records ``commit_range``."""
    slug = f"{base[:12]}-{HEAD[:12]}" if base else f"root-{HEAD[:12]}"
    brief = root / FOLDER / f"{slug}.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text(f"{BRIEF_MARKER}\n# Commit brief\n{BRIEF_END_MARKER}\n", encoding="utf-8")
    _state(brief, **state)
    _inject(brief, "commit_range", {
        "base": base, "head": HEAD, "commits": [], "brief": f"{FOLDER}/{slug}.md",
        "phrase": None, "window": None, "notes": [], "header_sha256": "0" * 64,
        "resolved_at": "2026-10-26T10:00:00Z",
    })
    return brief


def _plan(root: Path, *, head_ref: bool = True, **state) -> Path:
    """A plan, after a direct ``review-code --head-ref`` run when *head_ref*."""
    plan = root / PLAN_REL
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# foo\n", encoding="utf-8")
    _state(plan, **state)
    if head_ref:
        _inject(plan, "head_ref", {"base": BASE, "head": HEAD, "drifted": [],
                                   "resolved_at": "2026-10-26T10:00:00Z"})
    return plan


REVIEWED = {"completed": ("code-review",), "files": {"grouped.json": ONE_GROUP}}
APPLYING = {"completed": ("code-review",), "files": {"grouped.json": ONE_GROUP},
            "tasks": {"apply-fix-batch-0": "completed"}}
APPLIED = {"completed": ("code-review", "apply-code-fixes"),
           "files": {"grouped.json": ONE_GROUP}}


def _reports_named(why: str, slug: str = SLUG) -> None:
    assert f"{FOLDER}/{slug}/code-review/report.md" in why
    assert f"{FOLDER}/{slug}/code-review/report.html" in why
    assert "--improve-commits" in why


def _dispatch_printed(command: str, monkeypatch, capsys):
    """Run a printed dispatcher command's tokens through ``multi_llm.main``
    with every target recording its argv and running nothing."""
    tokens = shlex.split(command)
    start = next(i for i, token in enumerate(tokens)
                 if os.path.basename(token) == "multi_llm.py")
    seen = []

    def loader(module_path, attr):
        def target(argv):
            seen.append({"module": module_path, "argv": list(argv)})
            return 0
        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)
    code = multi_llm.main(tokens[start + 1:])
    captured = capsys.readouterr()
    return code, seen, captured.out, captured.err


# ===========================================================================
# multi_llm: the user_next_step after a foreground commit mode
# ===========================================================================


def _commit(root: Path, path: str, text: str, message: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(here):
    """Three commits; ``last 2 commits`` is HEAD~2..HEAD."""
    _commit(here, "a.py", "a = 1\n", "c1: start")
    _commit(here, "b.py", "b = 2\n", "c2: add b")
    _commit(here, "a.py", "a = 3\n", "c3: change a")
    return here


def _range(root: Path):
    base = _git(root, "rev-parse", "HEAD~2").strip()
    head = _git(root, "rev-parse", "HEAD").strip()
    return base, head, f"{base[:12]}..{head[:12]}", f"{base[:12]}-{head[:12]}"


def _main(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestTheDispatchersNextStep:
    def test_after_a_foreground_review_commits_it_is_the_apply_naming_both_reports(
        self, repo, monkeypatch, capsys
    ):
        seen = {}

        def review(argv):
            brief = cli_common.flag_value(argv, "--plan-file")
            seen["invocation"] = dict(cli_common.invocation() or {})
            _state(Path(os.path.abspath(brief)), **REVIEWED)
            return 0

        monkeypatch.setattr(code_review_orchestrator, "main", review)
        code, out, err = _main(["review-commits", "last", "2", "commits",
                                "--host", "claude-code"], capsys)
        assert code == 0, err
        _base, _head, pinned, slug = _range(repo)
        # Recorded as the fronted phase, answered in the commit spelling.
        assert seen["invocation"]["mode"] == "review-code"
        step = parse_result_line(out)["user_next_step"]
        assert step["phase"] == "apply-code-fixes"
        assert step["command"] == f"{CC}--apply-commit-fixes {pinned}"
        _clean(step["command"])
        _reports_named(step["why"], slug)
        lines = out.splitlines()
        command_line = lines.index(f"[NEXT_COMMAND] {CC}--apply-commit-fixes {pinned}")
        assert lines[command_line - 1].startswith("[NEXT_COMMAND_WHY] ")
        assert f"{pinned}" in step["markdown"] and "report.html" in step["markdown"]

    def test_after_a_foreground_apply_commit_fixes_the_workflow_is_complete(
        self, repo, monkeypatch, capsys
    ):
        monkeypatch.setattr(code_review_orchestrator, "main", lambda argv: _state(
            Path(os.path.abspath(cli_common.flag_value(argv, "--plan-file"))), **REVIEWED) or 0)
        assert _main(["review-commits", "last", "2", "commits"], capsys)[0] == 0

        def apply(argv):
            _state(Path(os.path.abspath(cli_common.flag_value(argv, "--plan-file"))),
                   completed=("apply-code-fixes",))
            return 0

        monkeypatch.setattr(apply_code_fixes_orchestrator, "main", apply)
        _base, _head, pinned, _slug = _range(repo)
        code, out, err = _main(["apply-commit-fixes", pinned, "--exec-subagents"], capsys)
        assert code == 0, err
        step = parse_result_line(out)["user_next_step"]
        assert step["complete"] is True and step["command"] is None
        assert "[WORKFLOW_COMPLETE]" in out

    def test_a_foreground_improve_commits_finds_its_brief_from_the_pinned_range(
        self, repo, monkeypatch, capsys
    ):
        """improve-commits is recorded with ``--revs <pinned>``, not a plan:
        the next step still answers, for the brief that spelling names."""
        def entry_point(mode, argv):
            _state(Path(os.path.abspath(cli_common.flag_value(argv, "--plan-file"))),
                   **APPLIED)
            return 0

        monkeypatch.setattr(improve_workflow, "entry_point", entry_point)
        code, out, err = _main(["improve-commits", "last", "2", "commits"], capsys)
        assert code == 0, err
        step = parse_result_line(out).get("user_next_step")
        assert step is not None and step["complete"] is True
        assert "[WORKFLOW_COMPLETE]" in out

    def test_a_continuation_in_the_commit_spelling_names_the_fronted_phase_and_brief(
        self, here
    ):
        brief = _seed(here)
        target = multi_llm._continuation_target(
            cli_common.dispatcher_command("review-commits", PINNED, "--reaggregate"))
        assert target is not None
        mode, plan = target
        assert mode.name == "review-code"
        assert os.path.realpath(plan) == os.path.realpath(brief)
        root = _seed(here, base=None)
        mode, plan = multi_llm._continuation_target(
            cli_common.dispatcher_command("review-commits", ROOT_REL, "--reaggregate"))
        assert mode.name == "review-code" and plan == ROOT_REL
        assert os.path.realpath(plan) == os.path.realpath(root)
        # Not a continuation verb: no target, as before.
        assert multi_llm._continuation_target(
            cli_common.dispatcher_command("review-commits", PINNED, "--resume")) is None


# ===========================================================================
# next_phase: the builder, resume_command, phase_command, util next-command
# ===========================================================================


class TestTheBuilder:
    def test_after_review_code_on_a_brief_every_route_names_the_apply(self, here):
        brief = _seed(here, **REVIEWED)
        for host, route, prefix in ROUTES:
            answer = next_phase.next_phase_commands(REL, host, route, "review-code",
                                                    plan_path=brief)
            assert answer.phase == "apply-code-fixes"
            assert answer.command == f"{prefix}--apply-commit-fixes {PINNED}", host
            _clean(answer.command)
            _reports_named(answer.why)
            payload = answer.to_dict()
            assert payload["why"] == answer.why
            assert f"--apply-commit-fixes {PINNED}" in payload["markdown"]

    def test_a_root_range_spells_the_brief_path_after_the_commit_mode(self, here):
        brief = _seed(here, base=None, **REVIEWED)
        answer = next_phase.next_phase_commands(ROOT_REL, "claude-code", None,
                                                "review-code", plan_path=brief)
        assert answer.command == f"{CC}--apply-commit-fixes {ROOT_REL}"
        _clean(answer.command)
        _reports_named(answer.why, ROOT_SLUG)

    def test_an_apply_in_progress_resumes_as_the_commit_mode(self, here):
        brief = _seed(here, **APPLYING)
        answer = next_phase.next_phase_commands(REL, "claude-code", None, "review-code",
                                                plan_path=brief)
        assert answer.command == f"{CC}--apply-commit-fixes {PINNED} --resume"
        assert answer.why is None

    def test_after_the_apply_and_after_improve_commits_the_workflow_is_complete(self, here):
        brief = _seed(here, **APPLIED)
        for finished in ("apply-code-fixes", "improve-commits"):
            answer = next_phase.next_phase_commands(REL, "claude-code", None, finished,
                                                    plan_path=brief)
            assert answer.complete is True and answer.command is None

    def test_util_next_command_takes_the_pinned_range_and_the_commit_word(
        self, here, capsys
    ):
        _seed(here, **REVIEWED)
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert next_phase.main([PINNED, "--after", "review-commits",
                                    "--host", "claude-code"]) == 0
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert f"[NEXT_COMMAND] {CC}--apply-commit-fixes {PINNED}" in out.splitlines()
        assert ctx.user_next_step["command"] == f"{CC}--apply-commit-fixes {PINNED}"


class TestResumeAndPhaseCommands:
    def test_resume_command_answers_for_either_word(self, here):
        brief = _seed(here, **APPLYING)
        for word in ("apply-code-fixes", "apply-commit-fixes"):
            for host, route, prefix in ROUTES:
                command = next_phase.resume_command(REL, host, route, word, plan_path=brief)
                assert command == f"{prefix}--apply-commit-fixes {PINNED} --resume", word

    def test_the_blocked_lookup_takes_the_pinned_range(self, here, capsys):
        _seed(here, **APPLYING)
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert next_phase.main([PINNED, "--blocked", "apply-commit-fixes",
                                    "--host", "claude-code"]) == 0
        finally:
            cli_common.deactivate_result()
        expected = f"{CC}--apply-commit-fixes {PINNED} --resume"
        assert f"[RESUME_COMMAND] {expected}" in capsys.readouterr().out.splitlines()
        resume = ctx.user_resume
        assert resume["command"] == expected
        assert resume["mode"] == "apply-commit-fixes"
        assert resume["phase"] == "apply-code-fixes"
        assert resume["flags"] == ["--resume"]
        assert "To go on with apply-commit-fixes:" in resume["markdown"]

    def test_phase_command_spells_every_flag_list_back(self, here):
        brief = _seed(here)
        assert next_phase.phase_command(
            REL, "claude-code", None, "apply-code-fixes",
            "--resume", "--human-task", "apply-fix-batch-0=skip",
        ) == f"{CC}--apply-commit-fixes {PINNED} --resume --human-task apply-fix-batch-0=skip"
        # The composite's continuation ([IMPROVE_RERUN]): nothing the front
        # door synthesised or the brief inherits.
        command = next_phase.phase_command(
            REL, "codex", "package", "improve-commits", "--models", "a", "--files", "a b.py",
            "--base-ref", BASE, "--head-ref", HEAD, "--plan-file", REL, "--focus", "x",
            "--context-plan=docs/x.md", "--resume",
        )
        assert command == (f"$multi-llm:multi-llm --improve-commits {PINNED} --models a "
                           f'--files "a b.py" --resume')
        # With the brief named by plan_path, the plan argument may be anything.
        assert next_phase.phase_command(
            PINNED, "claude-code", None, "review-code", "--resume", plan_path=brief,
        ) == f"{CC}--review-commits {PINNED} --resume"
        # A mode that takes no range keeps the brief path, which it takes as a plan.
        assert next_phase.phase_command(REL, "claude-code", None, "review-plan") == (
            f"{CC}--review-plan {REL}")


# ===========================================================================
# run_status: resume line, next step, native resume, composite tokens
# ===========================================================================


def _run(path: Path, *, mode: str, directory: str, flags, state="failed", exit_code=3,
         plan_arg=REL):
    phase = output_dir_for_brief(path) / directory
    phase.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(phase), {
        "run_id": f"run-{mode}-0001", "state": state, "exit_code": exit_code,
        "error": "needs_human (exit 3)" if exit_code else None, "detached": True,
        "mode": mode, "replay_flags": list(flags), "plan_arg": plan_arg,
    })
    return phase


def _resume_line(capsys, plan_arg, *argv) -> str:
    assert run_status.main(["--plan-file", plan_arg, "--view", "run", *argv]) == 3
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if "resume with:" in line]
    assert len(lines) == 1, out
    return lines[0].split("resume with: ", 1)[1]


class TestStatus:
    def test_a_stopped_review_resumes_as_review_commits(self, here, capsys):
        brief = _seed(here)
        _run(brief, mode="review-code", directory="code-review",
             flags=["--base-ref", BASE, "--head-ref", HEAD, "--models", "a", "--host",
                    "claude-code"])
        for typed in (REL, PINNED):
            line = _resume_line(capsys, typed, "--host", "claude-code")
            assert line == f"{CC}--review-commits {PINNED} --models a --reaggregate", typed
        # The dispatcher spelling (a tool route's), from the same record.
        view = {"run": {"mode": "review-code", "replay_flags": [
            "--base-ref", BASE, "--head-ref", HEAD, "--models", "a"]}}
        command = run_status._resume_command(view, brief)
        assert command == cli_common.dispatcher_command(
            "review-commits", PINNED, "--models", "a", "--reaggregate")
        _clean(command)

    def test_a_stopped_apply_resumes_as_apply_commit_fixes(self, here, capsys):
        brief = _seed(here, **APPLYING)
        _run(brief, mode="apply-code-fixes", directory="apply-fixes",
             flags=["--plan-file", REL, "--host", "opencode", "--detach"])
        assert _resume_line(capsys, REL, "--host", "opencode") == (
            f"/multi-llm --apply-commit-fixes {PINNED} --resume")

    def test_a_stopped_improve_commits_resumes_with_its_continuation(self, here):
        brief = _seed(here)
        view = {"run": {"mode": "improve-commits",
                        "replay_flags": ["--models", "a", "--base-ref", BASE]}}
        command = run_status._resume_command(view, brief)
        assert command == cli_common.dispatcher_command(
            "improve-commits", PINNED, "--models", "a", "--resume")
        spelled = run_status._resume_command(
            view, brief, run_status.user_speller("claude-code", REL, plan_path=brief))
        assert spelled == f"{CC}--improve-commits {PINNED} --models a --resume"

    def test_composite_resume_tokens_never_carry_the_front_doors_flags(self):
        flags = ["--models", "a", "--base-ref", BASE]
        assert run_status._composite_resume_tokens("improve-commits", flags) == [
            "--models", "a", "--resume"]
        # improve-code keeps a typed --base-ref, as today.
        assert run_status._composite_resume_tokens("improve-code", flags) == [
            "--models", "a", "--base-ref", BASE, "--resume"]

    def test_the_native_resume_and_its_resume_with_line(self, here):
        brief = _seed(here, **APPLYING)
        assert run_status.native_resume_command("apply-code-fixes", brief) == (
            cli_common.dispatcher_command("apply-commit-fixes", PINNED, "--resume"))
        spell = run_status.user_speller("claude-code", PINNED, plan_path=brief)
        resume = run_status.native_resume_command("apply-code-fixes", brief, spell)
        assert resume == f"{CC}--apply-commit-fixes {PINNED} --resume"
        lines = run_status.render_state_apply_lines(
            {"phase_dir": "apply-fixes", "completed": False}, resume=resume)
        assert f"      resume with: {CC}--apply-commit-fixes {PINNED} --resume" in lines

    def test_a_terminal_poll_of_the_review_names_the_apply(self, here, capsys):
        brief = _seed(here, **REVIEWED)
        _run(brief, mode="review-code", directory="code-review",
             flags=["--base-ref", BASE, "--head-ref", HEAD], state="done", exit_code=0)
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main(["--plan-file", PINNED, "--view", "run",
                                    "--phase", "code-review", "--host", "claude-code"]) == 0
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert f"[NEXT_COMMAND] {CC}--apply-commit-fixes {PINNED}" in out.splitlines()
        assert ctx.user_next_step["command"] == f"{CC}--apply-commit-fixes {PINNED}"
        _reports_named(ctx.user_next_step["why"])


# ===========================================================================
# The row-14 stop, and the dispatcher reading a printed command back
# ===========================================================================


def _stop(brief: Path, *typed: str, pending=None):
    args = improve_workflow.parse_args("improve-commits", ["--plan-file", str(brief), *typed])
    workflow = improve_workflow.ImproveWorkflow(args, "improve-commits")
    ctx = cli_common.activate_result(cli_common.ResultContext(mode="improve-commits"))
    if pending:
        cli_common.record_next(pending)
    try:
        code = workflow._phase_failed(
            fwo.Step("apply-code-fixes", "Step 2: apply-commit-fixes",
                     lambda: ("failed", "", 1)),
            "exited 1", 1)
    finally:
        cli_common.deactivate_result()
    return code, ctx


class TestTheRow14Stop:
    def test_the_stop_and_next_name_the_commit_continuation(self, here, capsys):
        brief = _seed(here)
        code, ctx = _stop(brief, "--models", "a", "--base-ref", BASE)
        assert code == 3
        expected = cli_common.dispatcher_command("improve-commits", PINNED, "--models", "a",
                                                 "--resume")
        assert ctx.next == expected
        assert expected in [line.strip() for line in capsys.readouterr().err.splitlines()]
        _clean(expected)

    def test_a_follow_up_the_phase_printed_reads_in_the_commit_spelling_too(
        self, here, capsys
    ):
        """An apply that left its batches unexecuted prints its own follow-up
        (``_unexecuted_write_phase``), which the stop names and records first."""
        brief = _seed(here)
        code, ctx = _stop(brief, pending=cli_common.dispatcher_command(
            "apply-code-fixes", brief, "--yes"))
        assert code == 3
        expected = cli_common.dispatcher_command("apply-commit-fixes", PINNED, "--yes")
        assert ctx.next == expected
        err = [line.strip() for line in capsys.readouterr().err.splitlines()]
        assert expected in err
        assert cli_common.dispatcher_command("improve-commits", PINNED, "--resume") in err
        # Another plan's follow-up is left as it is.
        other = cli_common.dispatcher_command("apply-code-fixes", here / "other.md", "--yes")
        assert _stop(brief, pending=other)[1].next == other

    def test_a_root_ranges_stop_names_the_brief_and_the_dispatcher_takes_it(
        self, here, monkeypatch, capsys
    ):
        brief = _seed(here, base=None)
        code, ctx = _stop(brief)
        assert code == 3
        assert ctx.next == cli_common.dispatcher_command("improve-commits", str(brief),
                                                         "--resume")
        capsys.readouterr()
        code, seen, out, err = _dispatch_printed(ctx.next, monkeypatch, capsys)
        assert code == 0, err
        assert "[COMMIT_FLAG_CONFLICT]" not in out + err
        [call] = seen
        assert call["module"] == "improve_commits_orchestrator"
        assert cli_common.flag_value(call["argv"], "--plan-file") == str(brief)
        assert "--revs" not in call["argv"]


class TestAPrintedCommandRunsAsTyped:
    def test_a_root_ranges_next_step_passes_the_conflict_scan(self, here, monkeypatch, capsys):
        brief = _seed(here, base=None, **APPLYING)
        command = run_status._resume_command(
            {"run": {"mode": "apply-code-fixes", "replay_flags": ["--plan-file", ROOT_REL]}},
            Path(ROOT_REL))
        assert command == cli_common.dispatcher_command("apply-commit-fixes", ROOT_REL,
                                                        "--resume")
        code, seen, out, err = _dispatch_printed(command, monkeypatch, capsys)
        assert code == 0, err
        assert "[COMMIT_FLAG_CONFLICT]" not in out + err
        [call] = seen
        assert call["module"] == "apply_commit_fixes_orchestrator"
        assert cli_common.flag_value(call["argv"], "--plan-file") == ROOT_REL
        assert "--revs" not in call["argv"] and "--resume" in call["argv"]
        assert os.path.realpath(ROOT_REL) == os.path.realpath(brief)

    def test_a_pinned_next_step_is_a_range(self, here, monkeypatch, capsys):
        _seed(here, **APPLYING)
        command = run_status.native_resume_command("apply-code-fixes", Path(REL))
        code, seen, out, err = _dispatch_printed(command, monkeypatch, capsys)
        assert code == 0, err
        [call] = seen
        assert cli_common.flag_value(call["argv"], "--revs") == PINNED
        assert "--plan-file" not in call["argv"]


# ===========================================================================
# A direct --head-ref run is never mapped (§2.5, "Printed commands")
# ===========================================================================


class TestADirectHeadRefRun:
    def test_its_next_step_and_resume_lines_stay_review_codes(self, here, capsys):
        plan = _plan(here, **REVIEWED)
        answer = next_phase.next_phase_commands(PLAN_REL, "claude-code", None,
                                                "review-code", plan_path=plan)
        assert answer.command == f"{CC}--apply-code-fixes {PLAN_REL}"
        assert answer.why is None
        view = {"run": {"mode": "review-code",
                        "replay_flags": ["--base-ref", BASE, "--head-ref", HEAD]}}
        assert run_status._resume_command(view, plan) == cli_common.dispatcher_command(
            "review-code", str(plan), "--base-ref", BASE, "--head-ref", HEAD, "--reaggregate")
        _run(plan, mode="review-code", directory="code-review",
             flags=["--base-ref", BASE, "--head-ref", HEAD], plan_arg=PLAN_REL)
        assert _resume_line(capsys, PLAN_REL, "--host", "claude-code") == (
            f"{CC}--review-code {PLAN_REL} --base-ref {BASE} --head-ref {HEAD} --reaggregate")
        assert next_phase.phase_command(PLAN_REL, "claude-code", None, "apply-code-fixes",
                                        "--resume") == (
            f"{CC}--apply-code-fixes {PLAN_REL} --resume")

    def test_it_calls_the_plan_a_plan(self, here):
        plan = _plan(here)
        assert code_review_orchestrator._plan_file_label(plan) == "Plan file"
        assert run_status._document_word(plan) == "plan"


# ===========================================================================
# "Commit brief", never "plan" (§2.8)
# ===========================================================================


def _apply_orchestrator(path: Path):
    orchestrator = apply_code_fixes_orchestrator.ApplyCodeFixesOrchestrator.__new__(
        apply_code_fixes_orchestrator.ApplyCodeFixesOrchestrator)
    orchestrator.args = SimpleNamespace(plan_file=str(path))
    return orchestrator


class TestCommitBriefNeverPlan:
    def test_review_codes_own_lines(self, here, capsys):
        brief = _seed(here)
        plan = _plan(here, head_ref=False)
        assert code_review_orchestrator._plan_file_label(brief) == "Commit brief"
        assert code_review_orchestrator._plan_file_label(plan) == "Plan file"
        # The brief deleted, its state kept: the error names what is missing.
        brief.unlink()
        with pytest.raises(SystemExit):
            asyncio.run(code_review_orchestrator.main(["--plan-file", REL]))
        assert f"ERROR: Commit brief not found: {REL}" in capsys.readouterr().out
        (here / PLAN_REL).unlink()
        with pytest.raises(SystemExit):
            asyncio.run(code_review_orchestrator.main(["--plan-file", PLAN_REL]))
        assert f"ERROR: Plan file not found: {PLAN_REL}" in capsys.readouterr().out

    def test_applys_own_lines(self, here, capsys):
        brief = _seed(here)
        plan = _plan(here, head_ref=False)
        assert _apply_orchestrator(brief)._document_title() == "Commit brief"
        assert _apply_orchestrator(brief)._document_title("Plan file") == "Commit brief"
        assert _apply_orchestrator(plan)._document_title() == "Plan"
        assert _apply_orchestrator(plan)._document_title("Plan file") == "Plan file"
        brief.unlink()
        code = apply_code_fixes_orchestrator.main(["--plan-file", REL])
        assert code == 1
        assert f"ERROR: Commit brief not found: {REL}" in capsys.readouterr().err

    def test_status_names_the_commit_brief(self, here, capsys):
        brief = _seed(here)
        plan = _plan(here, head_ref=False)
        assert run_status.main(["--plan-file", REL, "--view", "run"]) == 2
        assert "no run recorded for this commit brief" in capsys.readouterr().out
        assert run_status.main(["--plan-file", PLAN_REL, "--view", "run"]) == 2
        assert "no run recorded for this plan" in capsys.readouterr().out
        assert run_status.main(["--plan-file", PINNED, "--abort"]) == 2
        assert ("Nothing to abort: no run recorded for this commit brief."
                in capsys.readouterr().out)
        assert brief.is_file() and plan.is_file()

    def test_the_composites_report_names_the_commit_brief(self, here, capsys):
        brief = _seed(here)
        args = improve_workflow.parse_args("improve-commits", ["--plan-file", str(brief)])
        improve_workflow.ImproveWorkflow(args, "improve-commits").print_report()
        assert f"**Commit brief**: {brief}" in capsys.readouterr().out
        plan = _plan(here, head_ref=False)
        args = improve_workflow.parse_args("improve-code", ["--plan-file", str(plan)])
        improve_workflow.ImproveWorkflow(args, "improve-code").print_report()
        assert f"**Plan**: {plan}" in capsys.readouterr().out
