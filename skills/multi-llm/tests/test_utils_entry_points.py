"""Running a ``utils`` module as a command (docs/development/dispatcher.md#import-safety).

Every module with an entry point runs as ``python -m utils.<name>`` from the skill
directory. Nine also run by path: a guard at their top puts the skill root on
``sys.path``, found from the file's real location as Python finds ``sys.path[0]``,
and calls the package module's entry point with ``sys.argv`` untouched. So a run
by path, through a symlink to the file, or through a symlink to the skill
directory prints what the module prints as a plain script under the name typed:
the same help and errors, under the same program name, with the same exit code.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
UTILS = SKILL_ROOT / "utils"

GUARD = 'if __name__ == "__main__" and not __package__:'

#: The modules that run by path as well as with ``-m``, each with two calls that
#: print without touching anything: its help and a refused call. ``backup`` has
#: no parser, so both of its calls are usage errors.
BY_PATH = {
    "backup": ([], ["a", "b"]),
    "check_reply": (["--help"], ["--bogus-u20"]),
    "env_vars": (["--help"], ["--bogus-u20"]),
    "host_install": (["--help"], ["--bogus-u20"]),
    "html_report_generator": (["--help"], ["--bogus-u20"]),
    "metrics": (["--help"], ["--bogus-u20"]),
    "prior_changes": (["--help"], ["--bogus-u20"]),
    "report_check": (["--help"], ["--bogus-u20"]),
    "wakeup_check": (["--help"], ["--bogus-u20"]),
}

#: Of those, the ones whose parser takes its program name from ``sys.argv[0]``,
#: so the name changes with the spelling; the others name their program.
NAMED_BY_ARGV0 = ("env_vars", "html_report_generator", "prior_changes")

#: The by-path modules that are also ``multi_llm.py util`` commands and name
#: their own program, so all three spellings print the same help.
UTIL_COMMANDS = {
    "check_reply": "check-reply",
    "metrics": "metrics",
    "report_check": "report-check",
    "wakeup_check": "wakeup-check",
}

#: Modules that run with ``-m`` only: a refused call, its exit code and the
#: diagnostic it prints on stderr. ``<missing>`` is a path that does not exist.
MODULE_ONLY = {
    "batch_done": ([], 2, "error: the following arguments are required: --phase-dir"),
    "restore_batch": ([], 2, "error: the following arguments are required: --phase-dir"),
    "run_status": ([], 2, "error: the following arguments are required: --plan-file"),
    "tasks_graph_check": (["<missing>"], 2, "ERROR: cannot read "),
    "write_prompt": ([], 2, "error: the following arguments are required: --phase-dir"),
}

SPELLINGS = ("path", "file symlink", "renamed file symlink", "skill directory symlink")

#: The module's own code run as ``__main__``, as a plain script runs, with
#: ``sys.argv[0]`` the path typed: what a run by path must print.
_PLAIN_SCRIPT = (
    "import runpy, sys\n"
    "script, root, name = sys.argv[1:4]\n"
    "sys.argv[:] = [script] + sys.argv[4:]\n"
    "sys.path.insert(0, root)\n"
    "runpy.run_module('utils.' + name, run_name='__main__')\n"
)

Result = Tuple[int, bytes, bytes]


def _run(argv: Sequence[str], cwd: Path) -> Result:
    """``python <argv>`` from ``cwd`` with no PYTHONPATH, no colour and a fixed width."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONSAFEPATH", "FORCE_COLOR", "PYTHON_COLORS")}
    env.update(NO_COLOR="1", COLUMNS="100")
    done = subprocess.run(
        [sys.executable, *argv], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=120,
    )
    return done.returncode, done.stdout, done.stderr


def _past_program_name(result: Result) -> Result:
    """The exit code, the help after its usage block and the error after its
    ``<prog>: error:``: what does not depend on the program name."""
    code, out, err = result
    out, err = out.replace(b"\r\n", b"\n"), err.replace(b"\r\n", b"\n")
    return code, out.split(b"\n\n", 1)[-1], err.split(b": error: ", 1)[-1]


@pytest.fixture(scope="module")
def link_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("links")
    try:
        (root / "skill").symlink_to(SKILL_ROOT, target_is_directory=True)
        for name in BY_PATH:
            (root / f"{name}.py").symlink_to(UTILS / f"{name}.py")
            (root / f"run-{name}.py").symlink_to(UTILS / f"{name}.py")
    except (OSError, NotImplementedError) as exc:  # Windows without the symlink privilege
        pytest.skip(f"cannot create symlinks here: {exc}")
    return root


def _script(name: str, spelling: str, request) -> Path:
    if spelling == "path":
        return UTILS / f"{name}.py"
    links = request.getfixturevalue("link_dir")
    return {
        "file symlink": links / f"{name}.py",
        "renamed file symlink": links / f"run-{name}.py",
        "skill directory symlink": links / "skill" / "utils" / f"{name}.py",
    }[spelling]


def _calls(name: str) -> List[List[str]]:
    return [list(args) for args in BY_PATH[name]]


def test_the_by_path_modules_are_the_ones_with_the_guard():
    guarded = sorted(
        path.stem for path in UTILS.glob("*.py")
        if GUARD in path.read_text(encoding="utf-8")
    )
    assert guarded == sorted(BY_PATH)


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("name", sorted(BY_PATH))
def test_a_run_by_path_prints_what_the_plain_script_prints(name, spelling, request, tmp_path):
    script = _script(name, spelling, request)
    for args in _calls(name):
        got = _run([str(script), *args], tmp_path)
        want = _run(["-c", _PLAIN_SCRIPT, str(script), str(SKILL_ROOT), name, *args], tmp_path)
        assert got == want, f"{spelling} {script} {args}"
    if name in NAMED_BY_ARGV0:
        code, out, err = _run([str(script), "--help"], tmp_path)
        assert (code, err) == (0, b"")
        assert out.startswith(f"usage: {script.name} ".encode()), out[:120]


@pytest.mark.parametrize("name", sorted(BY_PATH))
def test_dash_m_prints_what_a_run_by_path_prints(name, tmp_path):
    for args in _calls(name):
        by_m = _run(["-m", f"utils.{name}", *args], SKILL_ROOT)
        by_path = _run([str(UTILS / f"{name}.py"), *args], tmp_path)
        if name in NAMED_BY_ARGV0:
            by_m, by_path = _past_program_name(by_m), _past_program_name(by_path)
        assert by_m == by_path, args


@pytest.mark.parametrize("name", sorted(UTIL_COMMANDS))
def test_util_help_is_the_module_help(name, tmp_path):
    code, out, err = _run(
        [str(SKILL_ROOT / "multi_llm.py"), "util", UTIL_COMMANDS[name], "--help"], tmp_path)
    m_code, m_out, m_err = _run(["-m", f"utils.{name}", "--help"], SKILL_ROOT)
    assert (code, err) == (m_code, m_err) == (0, b"")
    assert m_out.startswith(b"usage: ")
    # The dispatcher adds its MULTI_LLM_RESULT line after the module's help.
    assert out.startswith(m_out)
    assert out[len(m_out):].startswith(b"MULTI_LLM_RESULT: ")


@pytest.mark.parametrize("name", sorted(MODULE_ONLY))
def test_a_module_only_command_runs_with_dash_m(name, tmp_path):
    args, code, diagnostic = MODULE_ONLY[name]
    args = [str(tmp_path / "missing" / "tasks.md") if a == "<missing>" else a for a in args]
    help_code, help_out, help_err = _run(["-m", f"utils.{name}", "--help"], SKILL_ROOT)
    assert (help_code, help_err) == (0, b"")
    assert help_out.startswith(b"usage: ")
    got_code, got_out, got_err = _run(["-m", f"utils.{name}", *args], SKILL_ROOT)
    assert (got_code, got_out) == (code, b""), got_err[-400:]
    assert diagnostic.encode() in got_err
