"""Mock provider CLIs on a replaced PATH, runnable on POSIX and on Windows.

The executor suites (``test_write_executor.py``, ``test_subagent_executor.py``
and the orchestrator files built on them) spawn real subprocesses against
``mocks/mock_llm.py``, so a launcher must be something the product's own
resolution (``utils.llm_client._resolve_executable``) finds and starts.

On POSIX that is a symlink, or an executable script, named exactly like the CLI.
On Windows neither works: ``shutil.which`` only finds a name carrying a PATHEXT
extension, ``CreateProcess`` cannot start a shebang script, and a bare ``.cmd``
that calls python resolves as a batch shim, which the product refuses to launch
with any prompt carrying a newline or a quote (the cmd.exe reparse guard). So a
Windows launcher is shaped the way npm installs a real CLI: ``<name>.cmd``
dispatching to ``node mocks/<name>.js``. The product parses that shim and starts
node directly, cmd.exe never involved, and the node trampoline starts the
script's interpreter with argv passed through verbatim. ``harness/skill_runner.py``
installs its e2e launchers through :func:`install_mock_llm`, so the pair is
rendered here and nowhere else.

POSIX bytes are what the fixtures wrote before this module existed.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
SCENARIO_MATCHER = Path(__file__).parent / "scenario_matcher.py"

_IS_WINDOWS = os.name == "nt"

#: Windows only: the subdirectory of bin/ holding each launcher's node trampoline
#: and script body. Never bin/ itself, so a PATHEXT carrying ``.JS`` or ``.PY``
#: cannot resolve ``codex`` to one of them. ``SkillRunner.MOCK_JS_SUBDIR`` is
#: this constant.
SCRIPT_SUBDIR = "mocks"


def _windows_tool_dirs() -> List[str]:
    """The directories of the two tools a replaced Windows PATH must keep.

    ``git``: every fixture builds a real repository and the product shells out
    to it. ``node``: the product resolves the shim's ``node`` through PATH when
    no ``node.exe`` sits beside the shim. POSIX gets both from ``/usr/bin``;
    Windows installs each in its own directory. Resolved once, at import, while
    PATH is still the session's.
    """
    dirs: List[str] = []
    for tool in ("git", "node"):
        found = shutil.which(tool)
        if found and os.path.dirname(found) not in dirs:
            dirs.append(os.path.dirname(found))
    return dirs


_WINDOWS_TOOL_DIRS = _windows_tool_dirs() if _IS_WINDOWS else []
_NODE = shutil.which("node") if _IS_WINDOWS else None


def _registered_commands() -> List[str]:
    """Every provider CLI the registry knows, by the name PATH resolves.

    ``gate_binary_name()`` where the adapter declares one, and the registry key
    as well: agy, muse and kilocode declare none, and their keys are their
    binaries (kilocode also installs as ``kilo``).
    """
    from utils import provider_registry as registry
    from utils.providers.kilocode import BINARY_NAMES as KILOCODE_NAMES

    names = set(KILOCODE_NAMES)
    for key, provider in registry._PROVIDERS.items():
        names.add(key)
        declared = provider.gate_binary_name()
        if declared:
            names.add(os.path.basename(declared))
    return sorted(names)


def mock_path(bin_dir: Path, names: Iterable[str] = ()) -> str:
    """The PATH a fixture REPLACES the session's with: the mock bin dir first.

    Replaced, never prepended: these suites spawn real subprocesses, some with
    edit access, and a test that removes a mock launcher (the spawn-failed
    rungs) must not fall through to a developer's real CLI. On Windows the
    extra entries are git's and node's install directories, and node's is where
    npm puts its global CLIs under nvm-windows (scoop and choco put every shim
    in one directory too), so ``names`` and every registered provider CLI are
    checked to resolve nowhere but bin/ before the PATH is handed back — a
    fixture that mocks only ``codex`` can still be driven to a ``claude:``
    runner.
    """
    interpreter_dir = str(Path(sys.executable).parent)
    if not _IS_WINDOWS:
        return os.pathsep.join([str(bin_dir), interpreter_dir, "/usr/bin", "/bin"])
    rest = [interpreter_dir] + [d for d in _WINDOWS_TOOL_DIRS if d != interpreter_dir]
    for name in sorted(set(names) | set(_registered_commands())):
        leaked = shutil.which(name, path=os.pathsep.join(rest))
        assert leaked is None, (
            f"{name} resolves to {leaked!r} on the mock PATH with the mock removed; "
            f"this suite must never be able to reach a real provider CLI"
        )
    return os.pathsep.join([str(bin_dir), *rest])


def launcher_file(bin_dir: Path, name: str) -> Path:
    """The file PATH resolves ``name`` to: ``bin/<name>``, or ``bin/<name>.cmd``."""
    return bin_dir / (f"{name}.cmd" if _IS_WINDOWS else name)


def remove_launcher(bin_dir: Path, name: str) -> None:
    """Take ``name`` off the mock PATH; anything under ``mocks/`` is off it already."""
    launcher_file(bin_dir, name).unlink()


def install_mock_llm(bin_dir: Path, name: str, mock_llm: Path = MOCK_LLM) -> Path:
    """Put ``mock_llm`` (the real one unless a harness points elsewhere) on the
    mock PATH as the CLI ``name``.

    A *copied* POSIX launcher cannot find ``tests/scenario_matcher.py`` through
    its own ``__file__`` the way the real file and a symlink to it can, so the
    one definition of what a scenario means is copied beside it. The Windows
    trampoline runs the real file, and hands the identity over in
    ``MOCK_LLM_PROVIDER`` because python sets ``sys.argv[0]`` to the script. The
    identity is the adapter's id, which is the launcher's name for every CLI but
    one: the claude-code adapter's binary is ``claude`` (``mock_llm`` maps the
    two the same way when it reads ``sys.argv[0]``; an unmapped ``claude`` here
    would answer in cursor-agent's wire format, which is its fallback).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    if _IS_WINDOWS:
        identity = "claude-code" if name == "claude" else name
        return _install_windows(
            bin_dir, name, [sys.executable, str(mock_llm)],
            {"MOCK_LLM_PROVIDER": identity},
        )
    target = bin_dir / name
    try:
        target.symlink_to(mock_llm)
    except (OSError, NotImplementedError):  # pragma: no cover
        shutil.copyfile(mock_llm, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        matcher = mock_llm.parent.parent / SCENARIO_MATCHER.name
        shutil.copyfile(matcher, bin_dir / matcher.name)
    return target


def write_launcher(bin_dir: Path, name: str, source: str) -> Path:
    """Replace the CLI ``name`` with a script whose text is ``source``.

    ``source`` starts with its ``#!`` line, as a POSIX script does. On Windows
    that line picks the interpreter: python runs a ``#!...python...`` script,
    Git for Windows' bash a ``#!/bin/sh`` one (never ``System32\\bash.exe``,
    which is WSL's). Returns the file holding ``source``, which is what a test
    that edits the script afterwards must pass back in here.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    if not _IS_WINDOWS:
        target = bin_dir / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.write_text(source, encoding="utf-8")
        target.chmod(0o755)
        return target
    shebang = source.split("\n", 1)[0]
    if "python" in shebang:
        interpreter, suffix = sys.executable, ".py"
    elif shebang.endswith(("/sh", "/bash", " sh", " bash")):
        from utils.host_install import git_bash

        interpreter, suffix = git_bash(), ".sh"
        if interpreter is None:
            raise RuntimeError(f"{name}: a {shebang!r} launcher needs Git for Windows' bash")
    else:
        raise ValueError(f"{name}: no Windows interpreter for {shebang!r}")
    script = bin_dir / SCRIPT_SUBDIR / f"{name}{suffix}"
    script.parent.mkdir(parents=True, exist_ok=True)
    # LF only: bash reads a CR as part of the word before it.
    script.write_text(source, encoding="utf-8", newline="\n")
    _install_windows(bin_dir, name, [interpreter, str(script)], {})
    return script


def write_sh_command(path: Path, source: str) -> Path:
    """Write the ``#!/bin/sh`` script ``source`` at ``path``; return the command.

    For a shell-string setting such as ``driver.verify_command``: on POSIX the
    command is ``path`` itself. On Windows cmd.exe runs that string, so the
    command is a ``.cmd`` beside the script that runs it under Git for Windows'
    bash and hands its exit code back. A path rather than a quoted command
    line, because fixtures write the value into YAML as a plain scalar.
    """
    path.write_text(source, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    if not _IS_WINDOWS:
        return path
    from utils.host_install import git_bash

    command = path.with_suffix(".cmd")
    with open(command, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(f'@"{git_bash()}" "{path}" %*\n@EXIT /b %ERRORLEVEL%\n')
    return command


def _install_windows(
    bin_dir: Path, name: str, argv: List[str], env: Dict[str, str]
) -> Path:
    """Write ``<name>.cmd`` and the ``mocks/<name>.js`` it dispatches to."""
    if _NODE is None:
        raise RuntimeError(
            "The Windows mock launchers need Node.js on PATH: a provider that "
            "resolves to a bare .cmd shim is refused any prompt carrying cmd.exe "
            "metacharacters, so each mock is an npm-style shim the product "
            "launches as `node <cli.js>`. Install Node.js to run this suite."
        )
    trampoline = bin_dir / SCRIPT_SUBDIR / f"{name}.js"
    trampoline.parent.mkdir(parents=True, exist_ok=True)
    with open(trampoline, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_node_trampoline_source(argv, env))
    shim = launcher_file(bin_dir, name)
    with open(shim, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(_cmd_shim_source(name))
    return shim


def _cmd_shim_source(name: str) -> str:
    """An npm-shaped cmd shim: the quoted ``%~dp0``-relative ``.js`` target is
    what ``llm_client._NODE_SHIM_TARGET_RE`` parses to bypass cmd.exe. Run by
    cmd.exe instead (a test that starts the ``.cmd`` itself), ``%*`` forwards
    argv and the exit code is propagated."""
    return (
        "@ECHO off\n"
        "SETLOCAL\n"
        'SET "NODE_EXE=%~dp0node.exe"\n'
        'IF NOT EXIST "%NODE_EXE%" SET "NODE_EXE=node"\n'
        f'"%NODE_EXE%" "%~dp0\\{SCRIPT_SUBDIR}\\{name}.js" %*\n'
        "ENDLOCAL & EXIT /b %ERRORLEVEL%\n"
    )


def _node_trampoline_source(argv: List[str], env: Dict[str, str]) -> str:
    """``spawnSync`` with an argv list: no shell, so the prompt reaches the
    script byte for byte, quotes, newlines and percent signs included. Paths are
    JSON literals so backslashes and spaces survive."""
    return (
        '"use strict";\n'
        "// Windows mock CLI trampoline: the product resolves the sibling .cmd\n"
        "// shim to `node <this file>`, so cmd.exe never sees the prompt.\n"
        'const { spawnSync } = require("child_process");\n'
        f"const ARGV = {json.dumps(argv)};\n"
        f"const ENV = {json.dumps(env)};\n"
        "const res = spawnSync(ARGV[0], ARGV.slice(1).concat(process.argv.slice(2)), {\n"
        '  stdio: "inherit",\n'
        "  env: Object.assign({}, process.env, ENV),\n"
        "});\n"
        "if (res.error) {\n"
        '  process.stderr.write("mock trampoline failed to launch " + ARGV[0]'
        ' + ": " + res.error.message + "\\n");\n'
        "  process.exit(127);\n"
        "}\n"
        "process.exit(res.status === null ? 1 : res.status);\n"
    )
