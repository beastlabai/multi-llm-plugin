"""A workflow command typed in a subdirectory of the repository runs from its root.

Live run 11, Claude Code racing 03 (2026-10-08): the host had ``cd``'d into the
phase folder ``plans/racing-game/generate-tasks`` to look at its files, then ran
``generate-tasks plans/racing-game.md --host claude-code --yes --reaggregate``
there; the dispatcher answered ``Plan file not found: plans/racing-game.md``
(resolved under the phase folder) and the host re-ran it from the root. Every
path a workflow command names is spelled from the repository root, so when the
relative ones name nothing from the cwd and each names something inside the
root of the repository the cwd is in, the dispatcher runs the whole command
from that root and says so on one stderr line: every mode, ``--reaggregate``
included, and every ``util`` command that takes such a path. Nothing moves when
a path names something from the cwd, is missing from the root too, or resolves
outside the repository, outside a repository, or while ``GIT_DIR`` redirects
git. ``main`` moves back to the caller's cwd on its way out.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import modes  # noqa: E402
import multi_llm  # noqa: E402
from harness import SkillRunner  # noqa: E402
from utils import cli_common, detach  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

from tests.test_generate_tasks_orchestrator import (  # noqa: E402
    PLAN_BODY,
    _valid_document,
)

SKILL_ROOT = Path(__file__).parent.parent
PLAN = "plans/x.md"
NOTICE = "multi-llm: running from the repository root"


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a plan, its folder's usual files, and the run-11 cwd:
    the plan's ``generate-tasks`` phase folder."""
    root = Path(os.path.realpath(tmp_path)) / "repo"
    (root / "plans" / "x" / "generate-tasks").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / PLAN).write_text("# Plan\n\nBody\n", encoding="utf-8")
    folder = root / "plans" / "x"
    (folder / "state.json").write_text("{}\n", encoding="utf-8")
    for phase in ("review-plan", "implement"):
        (folder / phase).mkdir()
    (folder / "review-plan" / "orchestrator_output.json").write_text("{}\n", encoding="utf-8")
    (folder / "review-plan" / "validation_tasks.json").write_text("{}\n", encoding="utf-8")
    (folder / "generate-tasks" / "task_generation_result.json").write_text("{}\n", encoding="utf-8")
    (folder / "tasks").mkdir()
    (folder / "tasks" / "tasks.md").write_text("# Tasks\n\n### T001: one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", "GIT_DIR", "GIT_WORK_TREE",
                 "GIT_COMMON_DIR", detach.PLAN_ARG_ENV, detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    here = folder / "generate-tasks"
    monkeypatch.chdir(here)
    return root, here


def _stub_targets(monkeypatch):
    """Every mode's and util's target records the cwd it ran in and its argv."""
    seen = []

    def loader(module_path, attr):
        def target(argv):
            seen.append({"cwd": os.getcwd(), "argv": list(argv)})
            return 0

        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)
    return seen


def _run(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


PLAN_MODES = [
    mode.name for mode in modes.MODES
    if modes.POS_PLAN in mode.positionals and not mode.unavailable
]


def _mode_argv(mode, plan=PLAN):
    argv = [mode, plan]
    if mode == "ask":
        argv.append("Is the plan complete?")
    return argv + ["--host", "claude-code"]


class TestEveryModeRunsFromTheRoot:
    @pytest.mark.parametrize("mode", PLAN_MODES)
    def test_a_root_relative_plan_typed_in_the_phase_folder(
        self, repo, mode, monkeypatch, capsys,
    ):
        root, here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(_mode_argv(mode), capsys)
        assert code == 0, err
        [call] = seen
        assert call["cwd"] == str(root)
        # The plan is forwarded as typed, and names the file from there.
        assert cli_common.flag_value(call["argv"], "--plan-file") == PLAN
        assert err.count(NOTICE) == 1
        assert f"{NOTICE} {root}, where {PLAN} is" in err
        assert f"not under the current directory {here})" in err
        # The caller keeps its own cwd.
        assert os.getcwd() == str(here)

    @pytest.mark.parametrize("mode, flag", [
        ("generate-tasks", "--reaggregate"),
        ("review-plan", "--reaggregate"),
        ("review-code", "--reaggregate"),
        ("review-plan", "--reaggregate-consolidation"),
        ("implement", "--resume"),
    ])
    def test_a_continuation_typed_in_the_phase_folder(
        self, repo, mode, flag, monkeypatch, capsys,
    ):
        root, _here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(_mode_argv(mode) + ["--yes", flag], capsys)
        assert code == 0, err
        assert [call["cwd"] for call in seen] == [str(root)]
        assert flag in seen[0]["argv"]

    def test_the_plan_flag_spelled_out_counts_too(self, repo, monkeypatch, capsys):
        root, _here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(["review-plan", f"--plan-file={PLAN}", "--host", "claude-code"], capsys)
        assert code == 0, err
        assert seen[0]["cwd"] == str(root)


UTIL_CALLS = {
    "metrics": ["record", "--state-file", "plans/x/state.json", "--phase", "implement",
                "--label", "T001"],
    "prior-changes": ["read", "--output-dir", "plans/x"],
    "report": ["regenerate-decisions", "--phase-dir", "plans/x/review-plan",
               "--state-file", "plans/x/state.json", "--apply-phase", "apply-suggestions"],
    "decisions": ["--output-file", "plans/x/review-plan/orchestrator_output.json"],
    "update-tasks": ["--plan-file", PLAN, "--tasks-file",
                     "plans/x/generate-tasks/task_generation_result.json"],
    "prereq": ["--plan-file", PLAN, "--status"],
    "finalize-tracking": ["--state-file", "plans/x/state.json", "--mark-completed"],
    "backup": [PLAN],
    "restore-batch": ["--phase-dir", "plans/x/implement", "--all-failed", "--dry-run"],
    "runs": ["--plan-file", PLAN],
    "next-command": [PLAN, "--after", "review-plan"],
    "check-tasks": ["plans/x/tasks/tasks.md"],
    "check-reply": ["--tasks-file", "plans/x/review-plan/validation_tasks.json"],
    "write-prompt": ["--phase-dir", "plans/x/implement", "--task-id", "T001"],
    "batch-done": ["--phase-dir", "plans/x/implement", "--task-id", "T001",
                   "--status", "completed"],
}


class TestEveryUtilThatTakesAPathRunsFromTheRoot:
    def test_the_table_covers_every_util_command_but_those_that_take_none(self):
        # report-check and wakeup-check read a Claude Code hook's input on stdin.
        assert set(UTIL_CALLS) == set(multi_llm.UTIL_COMMANDS) - {
            "report-check", "wakeup-check", "instructions"}

    @pytest.mark.parametrize("name", sorted(UTIL_CALLS))
    def test_a_root_relative_path_typed_in_the_phase_folder(
        self, repo, name, monkeypatch, capsys,
    ):
        root, here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(["util", name, *UTIL_CALLS[name]], capsys)
        assert code == 0, err
        assert [call["cwd"] for call in seen] == [str(root)]
        assert seen[0]["argv"] == UTIL_CALLS[name]
        assert err.count(NOTICE) == 1
        assert os.getcwd() == str(here)

    def test_next_command_s_plan_after_its_options(self, repo, monkeypatch, capsys):
        root, _here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(
            ["util", "next-command", "--after", "review-plan", "--host", "claude-code", PLAN],
            capsys,
        )
        assert code == 0, err
        assert seen[0]["cwd"] == str(root)

    def test_the_real_lookup_answers_as_it_does_from_the_root(self, repo, capsys):
        """Unstubbed: the phase-folder answer is the root answer plus the one line."""
        root, here = repo
        argv = ["util", "next-command", PLAN, "--after", "review-plan", "--host", "claude-code"]
        code, out, err = _run(argv, capsys)
        assert code == 0, err
        assert f"{NOTICE} {root}" in err
        os.chdir(root)
        try:
            root_code, root_out, root_err = _run(argv, capsys)
        finally:
            os.chdir(here)
        assert (code, out) == (root_code, root_out)
        assert NOTICE not in root_err
        assert f"[NEXT_COMMAND] /multi-llm:multi-llm --apply-suggestions {PLAN}" in out

    def test_the_real_graph_check_reads_the_root_s_task_list(self, repo, capsys):
        code, out, err = _run(["util", "check-tasks", "plans/x/tasks/tasks.md"], capsys)
        assert code == 0, err
        assert "[TASKS_GRAPH_OK]" in out


class TestNothingMovesWhenItShouldNot:
    def _assert_stayed(self, seen, err, here):
        assert [call["cwd"] for call in seen] == [str(here)]
        assert NOTICE not in err

    def test_a_plan_that_names_a_file_from_the_cwd(self, repo, monkeypatch, capsys):
        _root, here = repo
        (here / "plans").mkdir()
        (here / PLAN).write_text("# Another plan\n", encoding="utf-8")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan"), capsys)
        self._assert_stayed(seen, err, here)

    def test_a_plan_missing_from_the_root_too(self, repo, monkeypatch, capsys):
        _root, here = repo
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan", "plans/nope.md"), capsys)
        self._assert_stayed(seen, err, here)

    def test_the_real_error_still_names_the_cwd_spelling(self, repo, capsys):
        _root, here = repo
        code, _out, err = _run(
            ["util", "next-command", "plans/nope.md", "--host", "claude-code"], capsys,
        )
        assert code == 3
        assert f"plan file not found: {here / 'plans' / 'nope.md'}" in err
        assert NOTICE not in err

    def test_a_dotdot_path_that_leaves_the_repository_from_the_root(
        self, repo, monkeypatch, capsys,
    ):
        root, here = repo
        (root.parent / "outside.md").write_text("# outside\n", encoding="utf-8")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan", "../outside.md"), capsys)
        self._assert_stayed(seen, err, here)

    def test_a_link_in_the_repository_to_a_file_outside_it(self, repo, monkeypatch, capsys):
        root, here = repo
        outside = root.parent / "outside.md"
        outside.write_text("# outside\n", encoding="utf-8")
        (root / "plans" / "link.md").symlink_to(outside)
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan", "plans/link.md"), capsys)
        self._assert_stayed(seen, err, here)

    def test_outside_any_repository(self, tmp_path, monkeypatch, capsys):
        loose = Path(os.path.realpath(tmp_path)) / "loose"
        (loose / "sub").mkdir(parents=True)
        (loose / "plans").mkdir()
        (loose / PLAN).write_text("# Plan\n", encoding="utf-8")
        monkeypatch.chdir(loose / "sub")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan"), capsys)
        self._assert_stayed(seen, err, loose / "sub")

    def test_while_git_dir_redirects_git(self, repo, monkeypatch, capsys):
        root, here = repo
        monkeypatch.setenv("GIT_DIR", str(root / ".git"))
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan"), capsys)
        self._assert_stayed(seen, err, here)

    def test_a_mixed_spelling_is_not_guessed_at(self, repo, monkeypatch, capsys):
        """One path names a file from the cwd, the other only from the root:
        moving would break the first, so the command runs as typed."""
        _root, here = repo
        (here / "result.json").write_text("{}\n", encoding="utf-8")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(
            ["util", "update-tasks", "--plan-file", PLAN, "--tasks-file", "result.json"],
            capsys,
        )
        self._assert_stayed(seen, err, here)

    def test_an_absolute_plan(self, repo, monkeypatch, capsys):
        root, here = repo
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan", str(root / PLAN)), capsys)
        self._assert_stayed(seen, err, here)

    def test_help(self, repo, monkeypatch, capsys):
        _root, here = repo
        seen = _stub_targets(monkeypatch)
        _run(["util", "next-command", PLAN, "--help"], capsys)
        _code, _out, err = _run(["review-plan", PLAN, "--help"], capsys)
        assert [call["cwd"] for call in seen] == [str(here), str(here)]
        assert NOTICE not in err


# ---------------------------------------------------------------------------
# Run 11, end to end: the generate-tasks continuation typed in its phase folder
# ---------------------------------------------------------------------------

TYPED = "plans/auth.md"


def _dispatch(argv, cwd, env):
    result = subprocess.run(
        [sys.executable, str(SKILL_ROOT / "multi_llm.py"), *argv],
        cwd=str(cwd), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    payload = parse_result_line(result.stdout)
    assert payload is not None, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result, payload


@pytest.fixture
def auth_repo(tmp_path, skill_runner, mock_provider, fixture_manager):
    mock_provider.set_scenario("happy_path")
    skill_runner.extra_env.update(mock_provider.get_env())
    (tmp_path / "plans").mkdir()
    (tmp_path / TYPED).write_text(PLAN_BODY, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path, skill_runner._build_env()


def test_run_11_the_reaggregate_typed_in_the_phase_folder_finishes_the_phase(auth_repo):
    repo_dir, env = auth_repo
    _out, first = _dispatch(
        ["generate-tasks", TYPED, "--host", "claude-code", "--yes"], repo_dir, env,
    )
    phase = repo_dir / "plans" / "auth" / "generate-tasks"
    SkillRunner.simulate_subagents(
        phase / "task_generation_tasks.json", lambda _task: _valid_document()
    )
    tokens = shlex.split(first["next"])
    continuation = tokens[[os.path.basename(t) for t in tokens].index("multi_llm.py") + 1:]
    assert TYPED in continuation and "--reaggregate" in continuation

    # The host looked around in the phase folder and ran it from there.
    finished, result = _dispatch(continuation, phase, env)

    assert finished.returncode == 0, finished.stderr
    assert "Plan file not found" not in finished.stderr
    assert f"{NOTICE} {os.path.realpath(repo_dir)}, where {TYPED} is" in finished.stderr
    assert (repo_dir / "plans" / "auth" / "tasks" / "tasks.md").is_file()
    assert result["user_next_step"]["command"] == f"/multi-llm:multi-llm --implement {TYPED}"


# ---------------------------------------------------------------------------
# Fix round 27 (the Fable review of round 26, claude #4): an output path
# ---------------------------------------------------------------------------

class TestAnOutputPathIsReadToo:
    """review-plan's `--out-dir`, implement's `--output` and `util
    write-prompt`'s `--output` name a path the command creates, so it need
    not exist yet; but one that names something from the cwd (the path, or
    the first folder it names) is a mixed spelling, and a `..` out of the
    root or a link out of the repository is never followed there."""

    def _assert_stayed(self, seen, err, here):
        assert [call["cwd"] for call in seen] == [str(here)]
        assert NOTICE not in err

    @pytest.mark.parametrize("spelling", [
        ["--out-dir", "out"], ["--out-dir=out"], ["--out-dir", "out/new"],
        ["--out-dir", "./out"],
    ])
    def test_an_out_dir_that_names_a_folder_from_the_cwd(
        self, repo, monkeypatch, capsys, spelling,
    ):
        """The review's repro: `sub/out` exists, the plan only at the root."""
        _root, here = repo
        (here / "out").mkdir()
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan") + spelling, capsys)
        self._assert_stayed(seen, err, here)

    def test_an_out_dir_that_climbs_out_of_the_cwd(self, repo, monkeypatch, capsys):
        """`..` is spelled from here: from the root it leaves the repository."""
        _root, here = repo
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan") + ["--out-dir", "../out"], capsys)
        self._assert_stayed(seen, err, here)

    def test_an_out_dir_under_a_link_out_of_the_repository(
        self, repo, monkeypatch, capsys,
    ):
        root, here = repo
        outside = root.parent / "elsewhere"
        outside.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(
            _mode_argv("review-plan") + ["--out-dir", "link/new"], capsys)
        self._assert_stayed(seen, err, here)

    def test_a_new_out_dir_spelled_from_the_root_moves_with_the_plan(
        self, repo, monkeypatch, capsys,
    ):
        root, _here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(
            _mode_argv("review-plan") + ["--out-dir", "plans/x/reviews"], capsys)
        assert code == 0, err
        assert [call["cwd"] for call in seen] == [str(root)]
        assert f"{NOTICE} {root}, where {PLAN} is" in err

    def test_an_output_alone_never_moves_a_command(self, repo, monkeypatch, capsys):
        """Nothing it reads says the command is spelled from the root."""
        root, here = repo
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(_mode_argv("review-plan", str(root / PLAN)) + [
            "--out-dir", "plans/x/reviews"], capsys)
        self._assert_stayed(seen, err, here)

    def test_implements_tasks_output_from_the_cwd(self, repo, monkeypatch, capsys):
        _root, here = repo
        (here / "tasks.json").write_text("{}\n", encoding="utf-8")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(
            _mode_argv("implement") + ["--output", "tasks.json"], capsys)
        self._assert_stayed(seen, err, here)

    def test_write_prompts_output_from_the_cwd(self, repo, monkeypatch, capsys):
        _root, here = repo
        (here / "prompt.txt").write_text("x\n", encoding="utf-8")
        seen = _stub_targets(monkeypatch)
        _code, _out, err = _run(
            ["util", "write-prompt", *UTIL_CALLS["write-prompt"], "--output",
             "prompt.txt"], capsys)
        self._assert_stayed(seen, err, here)

    def test_write_prompts_new_output_moves_with_its_phase_dir(
        self, repo, monkeypatch, capsys,
    ):
        root, _here = repo
        seen = _stub_targets(monkeypatch)
        code, _out, err = _run(
            ["util", "write-prompt", *UTIL_CALLS["write-prompt"], "--output",
             "plans/x/implement/T001.txt"], capsys)
        assert code == 0, err
        assert [call["cwd"] for call in seen] == [str(root)]


#: Every option of a mode's or a `util` command's parser whose value can be a
#: path, and how the move reads it (fix round 27: review-plan's `--out-dir`
#: was missing, so a mixed spelling moved anyway). A new path option fails
#: `test_every_path_option_is_classified` until it is put here.
NOT_READ_BY_THE_MOVE = {
    # Repository-relative by definition: the same file from any cwd.
    "--files": "review-code's files, repository-relative",
    "--touched": "finalize-tracking's files, repository-relative",
    # Relative to something other than the cwd.
    "--summary-file-relative": "relative to the plan's folder",
    # init's target directory, which defaults to the root itself.
    "--dir": "init's target, as typed",
    # A Claude Code hook's input and transcript, outside the repository.
    "--hook-input": "a hook's input file",
    "--transcript": "a Claude Code transcript",
}
PATHISH = re.compile(r"file|dir|output|root|from|input|transcript|touched|^--files$")
NOT_A_PATH = {"--output-format", "--only-touched"}


def _parser_options(module_name: str):
    """The long options a module's parsers declare (``add_argument`` calls)."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    found.add(arg.value)
    return found


def test_every_path_option_is_classified():
    modules = {mode.target_module for mode in modes.MODES if mode.target_module}
    modules |= {module for module, _attr, _summary in multi_llm.UTIL_COMMANDS.values()
                if module}
    modules |= {"utils.cli_common"}
    options = set()
    for module in sorted(modules):
        options |= _parser_options(module)
    pathish = {opt for opt in options if PATHISH.search(opt)} - NOT_A_PATH
    classified = (set(multi_llm._MODE_INPUT_PATH_OPTIONS)
                  | set(multi_llm._MODE_OUTPUT_PATH_OPTIONS)
                  | set(multi_llm._UTIL_INPUT_PATH_OPTIONS)
                  | set(multi_llm._UTIL_OUTPUT_PATH_OPTIONS)
                  | set(NOT_READ_BY_THE_MOVE))
    assert pathish <= classified, sorted(pathish - classified)
    assert "--out-dir" in multi_llm._MODE_OUTPUT_PATH_OPTIONS
    # Every path a detached worker is handed absolute is one the move reads.
    assert set(detach._PATH_FLAGS) <= (set(multi_llm._MODE_INPUT_PATH_OPTIONS)
                                       | set(multi_llm._MODE_OUTPUT_PATH_OPTIONS))
