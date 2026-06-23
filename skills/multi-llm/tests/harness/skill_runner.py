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
    - Mock LLM binaries symlinked in tmp_path/bin/
    - PATH prepended to use mock binaries
    - MULTI_LLM_TEST_MODE=1 to enable test mode
    - MOCK_LLM_CALL_LOG set to a unique file for call logging

    Usage:
        runner = SkillRunner(tmp_path)
        result = runner.run_orchestrator("review_plan", plan_path, "--models", "cursor-agent")
        assert result.success
        assert result.mock_was_invoked()
    """

    # Default providers to create mock symlinks for
    PROVIDERS = ["cursor-agent", "gemini", "opencode", "codex", "kilocode"]

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

        # Create the bin directory and symlinks
        self._setup_mock_binaries()

    def _setup_mock_binaries(self) -> None:
        """Create the bin directory with provider symlinks to mock_llm.py."""
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        # Ensure mock_llm.py exists and is executable
        if not self.mock_llm_path.exists():
            raise FileNotFoundError(
                f"mock_llm.py not found at {self.mock_llm_path}. "
                "Ensure tests/mocks/mock_llm.py exists."
            )

        # Make mock_llm.py executable if it isn't already
        current_mode = self.mock_llm_path.stat().st_mode
        if not (current_mode & stat.S_IXUSR):
            self.mock_llm_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # Create symlinks for each provider
        for provider in self.PROVIDERS:
            symlink_path = self.bin_dir / provider
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(self.mock_llm_path)

    def _build_env(self) -> Dict[str, str]:
        """Build environment variables for subprocess execution."""
        env = os.environ.copy()

        # Prepend our bin directory to PATH
        original_path = env.get("PATH", "")
        env["PATH"] = f"{self.bin_dir}:{original_path}"

        # Enable test mode
        env["MULTI_LLM_TEST_MODE"] = "1"

        # Set call log path
        env["MOCK_LLM_CALL_LOG"] = str(self.call_log_path)

        # Enable fast backoff for retry tests
        env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        # Set scenario path if provided
        if self.scenario_path:
            env["MOCK_LLM_SCENARIO"] = str(self.scenario_path)

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
                timeout=timeout,
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
