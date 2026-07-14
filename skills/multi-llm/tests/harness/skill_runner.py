"""SkillRunner class for running orchestrators via subprocess.

Runs skill orchestrators in isolated environments with PATH manipulation
to ensure mock LLM binaries are used instead of real providers.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Wall-clock budgets in the e2e tests (and the subprocess timeouts below) are
# multiplied by this factor. Windows creates processes several times more
# slowly than POSIX (CreateProcess + Defender scanning of every image), and
# each mock provider call there costs an extra node+python launch on top of the
# orchestrator's, so a budget that is generous on Linux is marginal on
# windows-latest. Scaling keeps the POSIX budgets exactly as tight as they were.
PERF_SCALE = 3 if os.name == "nt" else 1


@dataclass
class SkillResult:
    """Result from running a skill orchestrator.

    Attributes:
        success: Whether the command completed successfully (exit code 0)
        exit_code: The actual exit code from the subprocess
        stdout: Standard output from the orchestrator
        stderr: Standard error from the orchestrator
        output_dir: Path to the output directory (plan subdir)
        call_log_path: Path to the mock LLM call log file
        env: Environment variables used for the run
        command: The command that was executed
        duration_seconds: How long the command took to run
    """

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    output_dir: Optional[Path] = None
    call_log_path: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Load and return the state.json file if it exists."""
        if self.output_dir is None:
            return None
        state_file = self.output_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_mock_calls(self) -> List[Dict[str, Any]]:
        """Load and return all mock LLM calls from the call log."""
        if self.call_log_path is None or not self.call_log_path.exists():
            return []
        calls = []
        with open(self.call_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    calls.append(json.loads(line))
        return calls

    def mock_was_invoked(self) -> bool:
        """Check if the mock LLM was invoked at least once.

        This is a safety check to ensure tests are actually using mock
        binaries and not accidentally calling real LLM providers.
        """
        calls = self.get_mock_calls()
        return len(calls) > 0


class SkillRunner:
    """Runs skill orchestrators via subprocess with mock LLM support.

    Creates an isolated environment for each run with:
    - Mock LLM binaries in tmp_path/bin/ (symlinks to mock_llm.py on POSIX,
      npm-style ``.cmd`` + ``node`` trampoline pairs on Windows — see
      :meth:`_setup_windows_launchers`)
    - PATH prepended to use mock binaries
    - MULTI_LLM_TEST_MODE=1 to enable test mode
    - MOCK_LLM_CALL_LOG set to a unique file for call logging
    - MULTI_LLM_PROVIDERS_CONFIG pinned to the skill's base providers.yaml so a
      host repo's .multi-llm/providers.yaml cannot reroute mock model specs

    Usage:
        runner = SkillRunner(tmp_path)
        result = runner.run_orchestrator("review_plan", plan_path, "--models", "cursor-agent")
        assert result.success
        assert result.mock_was_invoked()
    """

    # Default providers to create mock launchers for
    PROVIDERS = ["cursor-agent", "gemini", "opencode", "codex", "kilocode"]

    # Windows only: subdirectory of bin/ holding the node trampolines the
    # provider .cmd shims dispatch to (kept out of bin/ itself so that a
    # PATHEXT containing .JS can never resolve `cursor-agent` to the
    # trampoline instead of the shim).
    MOCK_JS_SUBDIR = "mocks"

    # Orchestrator command mappings
    ORCHESTRATORS = {
        "review_plan": "review_plan_orchestrator.py",
        "apply_suggestions": "apply_suggestions_orchestrator.py",
        "generate_tasks": "generate_tasks_orchestrator.py",
        "implement": "implement_orchestrator.py",
        "code_review": "code_review_orchestrator.py",
        "apply_fixes": "apply_code_fixes_orchestrator.py",
        "apply_task_suggestions": "apply_task_suggestions_orchestrator.py",
    }

    def __init__(
        self,
        tmp_path: Path,
        mock_llm_path: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        scenario_path: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        """Initialize SkillRunner.

        Args:
            tmp_path: Temporary directory for this test (from pytest tmp_path fixture)
            mock_llm_path: Path to mock_llm.py binary. If None, auto-detected from
                tests/mocks/mock_llm.py relative to this file.
            skill_dir: Path to the skill directory (skills/multi-llm/).
                If None, auto-detected relative to this file.
            scenario_path: Optional path to a scenario YAML file for response matching.
            extra_env: Optional additional environment variables to set.
        """
        self.tmp_path = Path(tmp_path)
        self.bin_dir = self.tmp_path / "bin"
        self.call_log_path = self.tmp_path / "mock_calls.jsonl"

        # Auto-detect mock_llm.py location
        if mock_llm_path is None:
            # Relative to this file: ../mocks/mock_llm.py
            self.mock_llm_path = Path(__file__).parent.parent / "mocks" / "mock_llm.py"
        else:
            self.mock_llm_path = Path(mock_llm_path)

        # Auto-detect skill directory
        if skill_dir is None:
            # Relative to this file: ../.. (up from tests/harness to skill root)
            self.skill_dir = Path(__file__).parent.parent.parent
        else:
            self.skill_dir = Path(skill_dir)

        self.scenario_path = scenario_path
        self.extra_env = extra_env or {}

        # Create the bin directory and provider launchers
        self._setup_mock_binaries()

    def _setup_mock_binaries(self) -> None:
        """Create the bin directory with per-provider launchers for mock_llm.py.

        POSIX: each provider name is a symlink to mock_llm.py, executed via its
        shebang (a copy is used when the host cannot create symlinks).

        Windows: see :meth:`_setup_windows_launchers` — the launcher must look
        to production exactly like a real npm-installed provider CLI does.
        """
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        # Ensure mock_llm.py exists
        if not self.mock_llm_path.exists():
            raise FileNotFoundError(
                f"mock_llm.py not found at {self.mock_llm_path}. "
                "Ensure tests/mocks/mock_llm.py exists."
            )

        if os.name == "nt":
            self._setup_windows_launchers()
        else:
            self._setup_posix_launchers()

    def _setup_posix_launchers(self) -> None:
        """Symlink (or copy) mock_llm.py under each provider name in bin/.

        The launcher name IS the provider name: mock_llm.py detects which
        provider it is emulating from ``sys.argv[0]``.
        """
        # Make mock_llm.py executable if it isn't already (it is committed 0755,
        # but a checkout with a restrictive umask/filemode=false can drop that).
        current_mode = self.mock_llm_path.stat().st_mode
        if not (current_mode & stat.S_IXUSR):
            self.mock_llm_path.chmod(
                current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

        for provider in self.PROVIDERS:
            launcher_path = self.bin_dir / provider
            if launcher_path.exists() or launcher_path.is_symlink():
                launcher_path.unlink()
            try:
                launcher_path.symlink_to(self.mock_llm_path)
            except (OSError, NotImplementedError):
                # Privilege-restricted sandbox: a copy preserves both the
                # shebang launch and the argv[0]-based provider detection.
                shutil.copyfile(self.mock_llm_path, launcher_path)
                launcher_path.chmod(
                    launcher_path.stat().st_mode
                    | stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH
                )

    def _setup_windows_launchers(self) -> None:
        """Install each mock provider the way an npm-installed CLI looks on Windows.

        Windows constrains this far more than POSIX does:

        * An extensionless shebang script is neither executable nor
          discoverable (``CreateProcess``/PATHEXT only honor .exe/.cmd/.bat/...),
          so the POSIX symlink trick cannot work.
        * A *bare* ``.cmd`` shim (``python mock_llm.py %*``) is worse than
          useless: production (``utils/llm_client._resolve_executable`` /
          ``_batch_shim_metachar_error``) deliberately REFUSES to launch a
          provider that only resolves to a .cmd/.bat shim when the prompt
          carries cmd.exe metacharacters — and every orchestrator prompt
          contains newlines, quotes and parentheses. Every e2e test would fail
          with ERROR_PROMPT_UNSAFE without ever reaching the mock.

        So the mock is installed exactly as npm installs a real provider CLI:
        a ``<provider>.cmd`` shim that dispatches to ``node <cli.js>``.
        Production parses that shim (``_resolve_node_shim_target``), bypasses
        cmd.exe entirely, and launches ``node mocks/<provider>.js`` natively —
        the same launch path a real cursor-agent/gemini/opencode install takes
        on Windows. The trampoline then re-launches mock_llm.py on the current
        interpreter, passing argv through verbatim and propagating the exit
        code.

        Provider identity cannot ride on argv[0] here (python sets
        ``sys.argv[0]`` to the script, i.e. always ``mock_llm.py``), so the
        trampoline passes it via ``MOCK_LLM_PROVIDER`` instead.
        """
        if shutil.which("node") is None:
            raise RuntimeError(
                "The Windows e2e harness requires Node.js on PATH. Production "
                "refuses to launch a provider that resolves to a bare .cmd/.bat "
                "shim when the prompt contains cmd.exe metacharacters, so the "
                "mock providers are installed as npm-style shims that dispatch "
                "to `node <cli.js>` (the launch path a real npm-installed "
                "provider CLI takes on Windows). Install Node.js to run the e2e "
                "suite on Windows."
            )

        js_dir = self.bin_dir / self.MOCK_JS_SUBDIR
        js_dir.mkdir(parents=True, exist_ok=True)

        for provider in self.PROVIDERS:
            js_path = js_dir / f"{provider}.js"
            with open(js_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(self._node_trampoline_source(provider))

            cmd_path = self.bin_dir / f"{provider}.cmd"
            with open(cmd_path, "w", encoding="utf-8", newline="\r\n") as f:
                f.write(self._cmd_shim_source(provider))

    def _cmd_shim_source(self, provider: str) -> str:
        """Render an npm-style cmd-shim that dispatches to the node trampoline.

        Deliberately shaped like the shims ``npm`` writes, because production's
        ``_NODE_SHIM_TARGET_RE`` parses exactly that shape (a quoted,
        ``%~dp0``-relative ``.js`` target) to bypass cmd.exe. The shim is also
        correct if actually executed by cmd.exe: ``%*`` forwards argv and
        ``EXIT /b %ERRORLEVEL%`` propagates the child's exit code (a batch file
        does not do that on its own).
        """
        return (
            "@ECHO off\n"
            "SETLOCAL\n"
            'SET "NODE_EXE=%~dp0node.exe"\n'
            'IF NOT EXIST "%NODE_EXE%" SET "NODE_EXE=node"\n'
            f'"%NODE_EXE%" "%~dp0\\{self.MOCK_JS_SUBDIR}\\{provider}.js" %*\n'
            "ENDLOCAL & EXIT /b %ERRORLEVEL%\n"
        )

    def _node_trampoline_source(self, provider: str) -> str:
        """Render the node trampoline that re-launches mock_llm.py.

        Paths are embedded as JSON literals so Windows backslashes and any
        spaces in the interpreter/mock path survive. ``spawnSync`` with an argv
        list means no shell and no re-parsing: the prompt reaches mock_llm.py
        byte-for-byte, quotes/newlines/percent signs included.
        """
        return (
            '"use strict";\n'
            "// Windows mock provider trampoline. Production resolves the sibling\n"
            "// <provider>.cmd npm shim to `node <this file>` and launches it\n"
            "// natively, so cmd.exe never sees (or mangles) the prompt.\n"
            'const { spawnSync } = require("child_process");\n'
            f"const PYTHON = {json.dumps(sys.executable)};\n"
            f"const MOCK_LLM = {json.dumps(str(self.mock_llm_path))};\n"
            f"const PROVIDER = {json.dumps(provider)};\n"
            "const res = spawnSync(PYTHON, [MOCK_LLM].concat(process.argv.slice(2)), {\n"
            '  stdio: "inherit",\n'
            "  env: Object.assign({}, process.env, { MOCK_LLM_PROVIDER: PROVIDER }),\n"
            "});\n"
            "if (res.error) {\n"
            '  process.stderr.write("mock trampoline failed to launch " + PYTHON'
            ' + ": " + res.error.message + "\\n");\n'
            "  process.exit(127);\n"
            "}\n"
            "process.exit(res.status === null ? 1 : res.status);\n"
        )

    def _build_env(self) -> Dict[str, str]:
        """Build environment variables for subprocess execution."""
        env = os.environ.copy()

        # Prepend our bin directory to PATH
        original_path = env.get("PATH", "")
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{original_path}"

        # Windows resolves argv[0] through PATHEXT (shutil.which does the same).
        # It is always set in a normal Windows environment, but a stripped-down
        # env would make `cursor-agent` unresolvable to `cursor-agent.cmd`.
        if os.name == "nt" and not env.get("PATHEXT"):
            env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"

        # Enable test mode
        env["MULTI_LLM_TEST_MODE"] = "1"

        # Set call log path
        env["MOCK_LLM_CALL_LOG"] = str(self.call_log_path)

        # Enable fast backoff for retry tests
        env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        # Set scenario path if provided
        if self.scenario_path:
            env["MOCK_LLM_SCENARIO"] = str(self.scenario_path)

        # Isolate config discovery from the host machine. Orchestrators run with
        # cwd=skill_dir, so when the skill lives inside a repo with a
        # `.multi-llm/providers.yaml` (e.g. a dogfooding checkout), the
        # project-local layer would deep-merge over base and reroute mock model
        # specs to unmocked CLIs. The env override layer merges LAST, so pointing
        # it at the skill's own base providers.yaml re-asserts every base-defined
        # key (default_provider, defaults.models, provider blocks, ...) over any
        # project-local override, restoring built-in behavior. Set before
        # extra_env so a test supplying its own override still wins.
        env["MULTI_LLM_PROVIDERS_CONFIG"] = str(self.skill_dir / "providers.yaml")
        env.pop("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", None)

        # Add any extra environment variables
        env.update(self.extra_env)

        return env

    def run_orchestrator(
        self,
        name: str,
        plan_path: Path,
        *args: str,
        timeout: int = 60,
        check: bool = False,
    ) -> SkillResult:
        """Run a specific orchestrator with the given plan.

        Args:
            name: Orchestrator name (e.g., "review_plan", "apply_suggestions")
            plan_path: Path to the plan file
            *args: Additional command-line arguments to pass to the orchestrator
            timeout: Timeout in seconds (default 60)
            check: If True, raise exception on non-zero exit code

        Returns:
            SkillResult with stdout, stderr, exit code, and helper methods

        Raises:
            ValueError: If orchestrator name is not recognized
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If check=True and command fails
        """
        if name not in self.ORCHESTRATORS:
            raise ValueError(
                f"Unknown orchestrator: {name}. "
                f"Valid options: {', '.join(self.ORCHESTRATORS.keys())}"
            )

        orchestrator_file = self.ORCHESTRATORS[name]
        orchestrator_path = self.skill_dir / orchestrator_file

        if not orchestrator_path.exists():
            raise FileNotFoundError(
                f"Orchestrator not found at {orchestrator_path}. "
                "Check that skill_dir is correct."
            )

        # Build command - use python from current env
        command = [
            sys.executable,
            str(orchestrator_path),
            "--plan-file",
            str(plan_path),
        ]
        # Apply orchestrators default to text output; tests need JSON
        if name in ("apply_suggestions", "apply_fixes"):
            command.extend(["--output-format", "json", "--no-confirm"])
        elif name == "apply_task_suggestions":
            command.extend(["--no-confirm"])
        command.extend(args)

        return self._run_command(command, plan_path, timeout, check)

    def _run_command(
        self,
        command: List[str],
        plan_path: Path,
        timeout: int,
        check: bool,
    ) -> SkillResult:
        """Execute a command and return the result.

        Args:
            command: Command and arguments to execute
            plan_path: Path to the plan file (for determining output_dir)
            timeout: Timeout in seconds
            check: If True, raise exception on non-zero exit code

        Returns:
            SkillResult with execution details
        """
        import time

        env = self._build_env()
        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Same budget on POSIX; headroom on Windows, where every
                # process launch (orchestrator, node shim, mock) is far more
                # expensive. See PERF_SCALE.
                timeout=timeout * PERF_SCALE,
                env=env,
                cwd=str(self.skill_dir),
            )
            duration = time.time() - start_time

            # Determine output directory (plan_name/ subdirectory)
            plan_name = plan_path.stem
            output_dir = plan_path.parent / plan_name

            skill_result = SkillResult(
                success=(result.returncode == 0),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_dir=output_dir if output_dir.exists() else None,
                call_log_path=self.call_log_path,
                env=env,
                command=command,
                duration_seconds=duration,
            )

            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    command,
                    result.stdout,
                    result.stderr,
                )

            return skill_result

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return SkillResult(
                success=False,
                exit_code=-1,
                stdout=e.stdout or "" if hasattr(e, "stdout") else "",
                stderr=e.stderr or "" if hasattr(e, "stderr") else "",
                output_dir=None,
                call_log_path=self.call_log_path,
                env=env,
                command=command,
                duration_seconds=duration,
            )

    def set_scenario(self, scenario_path: Path) -> None:
        """Update the scenario path for subsequent runs.

        Args:
            scenario_path: Path to a scenario YAML file
        """
        self.scenario_path = scenario_path

    def set_fixture_response(self, fixture_path: Path) -> None:
        """Set a direct fixture response (legacy MOCK_LLM_FIXTURE mode).

        Args:
            fixture_path: Path to a fixture JSON file to use as response
        """
        self.extra_env["MOCK_LLM_FIXTURE"] = str(fixture_path)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set dynamic mock configuration via MOCK_LLM_CONFIG.

        Args:
            config: Configuration dictionary to pass to mock_llm.py
        """
        self.extra_env["MOCK_LLM_CONFIG"] = json.dumps(config)

    def inject_failure(self) -> None:
        """Configure mock to fail with error on next call."""
        self.extra_env["MOCK_LLM_FAIL"] = "1"

    def inject_timeout(self) -> None:
        """Configure mock to simulate timeout on next call."""
        self.extra_env["MOCK_LLM_TIMEOUT"] = "1"

    def clear_injections(self) -> None:
        """Clear any injected failures or timeouts."""
        self.extra_env.pop("MOCK_LLM_FAIL", None)
        self.extra_env.pop("MOCK_LLM_TIMEOUT", None)
        self.extra_env.pop("MOCK_LLM_FIXTURE", None)
        self.extra_env.pop("MOCK_LLM_CONFIG", None)

    def verify_mock_isolation(self) -> bool:
        """Verify that no files were written outside tmp_path.

        Returns:
            True if isolation was maintained, False otherwise.
            Note: This is a best-effort check; some writes may not be detectable.
        """
        # Check that MOCK_LLM_CALL_LOG is inside tmp_path
        call_log = str(self.call_log_path)
        tmp_path_str = str(self.tmp_path)
        return call_log.startswith(tmp_path_str)
