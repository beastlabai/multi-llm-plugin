"""A native leg's continuation ends with the plan as the user typed it.

Racing re-run 4, Claude Code phase 03 (2026-10-05): the host ran
``generate-tasks plans/racing-game.md``, the native leg handed back its one
follow-up (``… generate-tasks /tmp/…/plans/racing-game.md --reaggregate``,
fully resolved by design: it is a dispatcher command the host runs, not one the
user types), the host ran it verbatim as the shim says, and the run that
finished the phase printed ``[NEXT_COMMAND] /multi-llm:multi-llm --implement
/tmp/…/plans/racing-game.md``. A next-phase command carries the plan as its
source supplied it, and the source of a continuation is the leg that handed it
back: the hand-off records the typed plan in the phase dir's ``.status.json``
(``plan_arg``) and the continuation prints that.

Every native leg that hands back a continuation on the claude-code host is
driven here end to end through the dispatcher: ``generate-tasks`` and the three
review phases' validation ``--reaggregate``. The write phases hand back none
(their closing call is the host's own ``--mark-completed`` or ``util
next-command``, typed with the plan the user gave). The rest pins the edges:
consolidation's continuation, an absolute plan typed by the user, a spelling
that no longer names the plan, an invocation that is not a continuation, a
detached continuation, and that every builder's follow-up parses as one.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from harness import SkillRunner  # noqa: E402
from utils import cli_common, detach, status_file  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.next_phase import (  # noqa: E402
    NEXT_COMMAND_MARKER,
    NEXT_COMMAND_OPTIONAL_MARKER,
)
from utils.output_handler import phase_dir_path  # noqa: E402

from tests.test_generate_tasks_orchestrator import (  # noqa: E402
    PLAN_BODY,
    _valid_document,
)
from tests.test_next_command_emitters import (  # noqa: E402
    PLUGIN,
    _run,
    _target,
    build,
    workdir,  # noqa: F401 - the fixture
)
from tests.test_next_phase import (  # noqa: E402
    P,
    RP_VALID,
    UP_TO_TASKS,
    S,
)

SKILL_ROOT = Path(__file__).parent.parent
TYPED = "plans/auth.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _dispatch(argv, repo, env):
    """``multi_llm.py <argv…>`` from *repo*, as the host's shell runs it."""
    result = subprocess.run(
        [sys.executable, str(SKILL_ROOT / "multi_llm.py"), *argv],
        cwd=str(repo), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    payload = parse_result_line(result.stdout)
    assert payload is not None, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result, payload


def _verbatim(command):
    """The dispatcher argv of a printed follow-up, exactly as printed.

    Everything after ``multi_llm.py``: the ``uv run … --project … -- python``
    prefix only locates the interpreter, which the test supplies itself.
    """
    tokens = shlex.split(command)
    names = [os.path.basename(token) for token in tokens]
    return tokens[names.index("multi_llm.py") + 1:]


def _printed_commands(stdout):
    """Every command a ``[NEXT_COMMAND]``/``[NEXT_COMMAND_OPTIONAL]`` line printed."""
    commands = []
    for line in stdout.splitlines():
        for marker in (NEXT_COMMAND_MARKER, NEXT_COMMAND_OPTIONAL_MARKER):
            if line.startswith(marker + " "):
                commands.append(line[len(marker) + 1:])
    return commands


@pytest.fixture
def repo(tmp_path, skill_runner, mock_provider, fixture_manager):
    """A git repository with ``plans/auth.md`` and one source file, plus the
    mock providers' environment (the fixture manager made it a work tree)."""
    mock_provider.set_scenario("happy_path")
    skill_runner.extra_env.update(mock_provider.get_env())
    (tmp_path / "plans").mkdir()
    (tmp_path / TYPED).write_text(PLAN_BODY, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path, skill_runner._build_env()


def _generate_tasks_leg(repo_dir, env):
    """The generate-tasks native leg: materialize, reply, run ``next`` as printed."""
    _out, first = _dispatch(
        ["generate-tasks", TYPED, "--host", "claude-code", "--yes"], repo_dir, env
    )
    phase = repo_dir / "plans" / "auth" / "generate-tasks"
    SkillRunner.simulate_subagents(
        phase / "task_generation_tasks.json", lambda _task: _valid_document()
    )
    return first, _dispatch(_verbatim(first["next"]), repo_dir, env)


def _review_leg(repo_dir, env, mode, phase_name, *flags):
    """A review phase's native leg: fan out, validate, run ``next`` as printed."""
    _out, first = _dispatch(
        [mode, TYPED, "--models", "cursor-agent", "--host", "claude-code", *flags],
        repo_dir, env,
    )
    SkillRunner.simulate_subagents(
        repo_dir / "plans" / "auth" / phase_name / "validation_tasks.json"
    )
    return first, _dispatch(_verbatim(first["next"]), repo_dir, env)


# ---------------------------------------------------------------------------
# Every native leg that hands back a continuation, end to end
# ---------------------------------------------------------------------------

LEGS = {
    # mode: (phase dir, the primary next-phase command's flag)
    "generate-tasks": ("generate-tasks", "--implement"),
    "review-plan": ("review-plan", "--apply-suggestions"),
    "review-tasks": ("review-tasks", "--implement"),
    "review-code": ("code-review", "--apply-code-fixes"),
}


class TestEachNativeLegEndsWithTheTypedPlan:
    @pytest.mark.parametrize("mode", list(LEGS))
    def test_the_continuation_run_as_printed_names_the_typed_plan(self, repo, mode):
        repo_dir, env = repo
        phase_name, primary = LEGS[mode]
        absolute = str(repo_dir / TYPED)
        if mode == "generate-tasks":
            first, (finished, result) = _generate_tasks_leg(repo_dir, env)
        elif mode == "review-tasks":
            _generate_tasks_leg(repo_dir, env)
            first, (finished, result) = _review_leg(repo_dir, env, mode, phase_name)
        elif mode == "review-code":
            (repo_dir / "src" / "main.py").write_text("print(2)\n", encoding="utf-8")
            first, (finished, result) = _review_leg(
                repo_dir, env, mode, phase_name,
                "--skip-prerequisites", "--files", "src/main.py",
            )
        else:
            first, (finished, result) = _review_leg(repo_dir, env, mode, phase_name)

        # The hand-off: a dispatcher command, fully resolved by design, and the
        # spelling the user typed recorded beside the checkpoint's plan_file.
        assert first["state"] == "done" and first["next"], first
        assert "--reaggregate" in _verbatim(first["next"])
        assert absolute in _verbatim(first["next"])
        assert status_file.read_status(
            str(repo_dir / "plans" / "auth" / phase_name)
        )["plan_arg"] == TYPED

        # The continuation, run exactly as printed, finishes the phase and hands
        # the user the plan as they typed it — in the block and on the result.
        assert finished.returncode == 0, finished.stderr
        step = result["user_next_step"]
        assert step["command"] == f"{PLUGIN}{primary} {TYPED}"
        commands = [step["command"]] + [
            entry["command"] for entry in step.get("optional_before", [])
        ]
        assert _printed_commands(finished.stdout) == commands
        for command in commands:
            assert command.endswith(f" {TYPED}"), command
            assert absolute not in command, command


# ---------------------------------------------------------------------------
# The edges, through the dispatcher with stubbed phase targets
# ---------------------------------------------------------------------------

def _hand_off(plan_arg_for_next, *verbs, mode="review-plan"):
    """A first leg that hands back ``<mode> <plan> <verbs…>`` as its ``next``,
    in the phase dir the real one would have created."""

    def first_leg(argv):
        plan = Path(os.path.abspath(cli_common.flag_value(argv, "--plan-file")))
        modes = cli_common.modes_table()
        phase_dir_path(plan, modes.resolve(mode).phase).mkdir(
            parents=True, exist_ok=True
        )
        cli_common.record_next(cli_common.retry_command(
            mode, plan_arg_for_next(plan), *verbs, flags=["--host", "claude-code"],
        ))

    return first_leg


class TestTheRecordedSpelling:
    def test_consolidation_hands_on_the_typed_plan_too(
        self, workdir, capsys, monkeypatch
    ):
        build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch, _hand_off(str, "--reaggregate-consolidation"))
        _code, _out, _err, first = _run(["review-plan", P], capsys)
        _target(monkeypatch)
        code, out, err, result = _run(_verbatim(first["next"]), capsys)
        assert code == 0, err
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {P}"
        )
        assert _printed_commands(out) == [f"{PLUGIN}--apply-suggestions {P}"]

    def test_an_absolute_plan_the_user_typed_stays_absolute(
        self, workdir, capsys, monkeypatch
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch, _hand_off(str, "--reaggregate"))
        _code, _out, _err, first = _run(["review-plan", str(plan)], capsys)
        _target(monkeypatch)
        _code, _out, _err, result = _run(_verbatim(first["next"]), capsys)
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )

    def test_a_spelling_that_names_another_file_from_here_is_never_printed(
        self, workdir, capsys, monkeypatch
    ):
        """Resolved from this cwd, the recorded spelling must name this plan;
        when it does not (the shell moved), the plan prints as received."""
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch, _hand_off(str, "--reaggregate"))
        _code, _out, _err, first = _run(["review-plan", P], capsys)
        elsewhere = workdir / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        _target(monkeypatch)
        _code, _out, _err, result = _run(_verbatim(first["next"]), capsys)
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )

    def test_an_invocation_that_continues_nothing_keeps_its_own_spelling(
        self, workdir, capsys, monkeypatch
    ):
        """A recorded spelling is the continuation's source and nobody else's:
        a fresh run typed with the absolute path prints the absolute path."""
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch, _hand_off(str, "--reaggregate"))
        _run(["review-plan", P], capsys)
        _target(monkeypatch)
        _code, _out, _err, result = _run(["review-plan", str(plan)], capsys)
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )

    def test_a_follow_up_naming_another_plan_records_nothing(
        self, workdir, capsys, monkeypatch
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        other = workdir / "plans" / "other.md"
        other.write_text("# other\n", encoding="utf-8")
        _target(monkeypatch, _hand_off(lambda _plan: other, "--reaggregate"))
        _run(["review-plan", P], capsys)
        assert "plan_arg" not in status_file.read_status(
            str(phase_dir_path(other, "review-plan"))
        )
        assert "plan_arg" not in status_file.read_status(
            str(phase_dir_path(plan, "review-plan"))
        )

    def test_a_detached_continuation_ships_the_typed_plan_to_its_worker(
        self, workdir, capsys, monkeypatch
    ):
        """The worker's block and the terminal poll print ``MULTI_LLM_PLAN_ARG``;
        a continuation launched with the printed absolute path ships the
        spelling the dispatcher recovered for it."""
        build(workdir, S(UP_TO_TASKS))
        _target(monkeypatch, _hand_off(str, "--reaggregate", mode="generate-tasks"))
        _code, _out, _err, first = _run(["generate-tasks", P], capsys)
        shipped = {}
        _target(
            monkeypatch,
            lambda argv: shipped.setdefault("plan_arg", detach._invoker_plan_arg(argv)),
        )
        _run(_verbatim(first["next"]), capsys)
        assert shipped["plan_arg"] == P


class TestEveryBuilderHandsBackAContinuation:
    """The writer recognises a hand-off by parsing the command it printed, so
    each builder's output must parse to its own mode and plan."""

    PLAN = "/repo/plans/my plan.md"

    @pytest.mark.parametrize("command,mode", [
        (cli_common.retry_command(
            "generate-tasks", PLAN, "--reaggregate", flags=["--host", "claude-code"],
        ), "generate-tasks"),
        (detach.resolved_reaggregate_command({
            "reaggregate_mode": "review-code", "plan_file": PLAN,
            "replay_flags": ["--quick"],
        }), "review-code"),
        (cli_common.retry_command(
            "review-plan", PLAN, "--reaggregate-consolidation",
        ), "review-plan"),
        ("PYTHONUNBUFFERED=1 " + cli_common.dispatcher_command(
            "review-tasks", "--plan-file", PLAN, "--reaggregate",
        ), "review-tasks"),
    ], ids=["retry", "status-rebuilt", "consolidation", "plan-file-flag"])
    def test_it_parses_to_its_mode_and_plan(self, command, mode):
        target = multi_llm._continuation_target(command)
        assert target is not None, command
        assert target[0].name == mode
        assert target[1] == self.PLAN

    @pytest.mark.parametrize("command", [
        detach.status_command_for(PLAN, phase="review-plan"),
        cli_common.retry_command("review-plan", PLAN, "--force"),
        cli_common.dispatcher_command("ask", PLAN, "why?", "--reaggregate"),
        "not a 'command",
    ], ids=["status-poll", "force-retry", "ask", "unparseable"])
    def test_anything_else_is_no_continuation(self, command):
        assert multi_llm._continuation_target(command) is None

