"""AssertionHelpers class for custom test assertions.

Provides domain-specific assertion methods for verifying multi-llm skill
outputs, state, and mock LLM calls.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .skill_runner import SkillResult
from .mock_provider import MockLLMCall


class AssertionHelpers:
    """Custom assertion methods for multi-llm skill tests.

    Provides clear, descriptive assertions for common test scenarios
    including state verification, output structure checks, and mock
    LLM call verification.

    Usage:
        helpers = AssertionHelpers()
        helpers.assert_state_phase_completed(result, "review-plan")
        helpers.assert_output_directory_structure(result.output_dir, ["grouped.json"])
        helpers.assert_llm_called_with(result, "review.*plan")
    """

    def assert_state_phase_completed(
        self,
        result: SkillResult,
        phase: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that a workflow phase is marked as completed in state.json.

        Args:
            result: SkillResult from an orchestrator run
            phase: Phase name (e.g., "review-plan", "apply-suggestions")
            message: Optional custom failure message

        Raises:
            AssertionError: If state doesn't exist or phase is not completed
        """
        state = result.get_state()
        if state is None:
            raise AssertionError(
                message or f"state.json not found in {result.output_dir}"
            )

        phases_completed = state.get("phases_completed", {})
        if phase not in phases_completed:
            completed_list = list(phases_completed.keys()) or ["(none)"]
            raise AssertionError(
                message
                or f"Phase '{phase}' not marked as completed. "
                f"Completed phases: {', '.join(completed_list)}"
            )

    def assert_state_phase_not_completed(
        self,
        result: SkillResult,
        phase: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that a workflow phase is NOT marked as completed.

        Args:
            result: SkillResult from an orchestrator run
            phase: Phase name to check
            message: Optional custom failure message

        Raises:
            AssertionError: If phase is marked as completed
        """
        state = result.get_state()
        if state is None:
            # No state file means no phases completed
            return

        phases_completed = state.get("phases_completed", {})
        if phase in phases_completed:
            raise AssertionError(
                message or f"Phase '{phase}' should not be completed but was"
            )

    def assert_state_phase_skipped(
        self,
        result: SkillResult,
        phase: str,
        expected_reason: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Assert that a workflow phase is marked as skipped.

        Args:
            result: SkillResult from an orchestrator run
            phase: Phase name to check
            expected_reason: Optional expected reason string (partial match)
            message: Optional custom failure message

        Raises:
            AssertionError: If phase is not skipped or reason doesn't match
        """
        state = result.get_state()
        if state is None:
            raise AssertionError(
                message or f"state.json not found in {result.output_dir}"
            )

        phases_skipped = state.get("phases_skipped", {})
        if phase not in phases_skipped:
            raise AssertionError(
                message or f"Phase '{phase}' should be skipped but was not"
            )

        if expected_reason:
            actual_reason = phases_skipped[phase].get("reason", "")
            if expected_reason.lower() not in actual_reason.lower():
                raise AssertionError(
                    message
                    or f"Phase '{phase}' skip reason mismatch. "
                    f"Expected '{expected_reason}' in '{actual_reason}'"
                )

    def assert_output_directory_structure(
        self,
        output_dir: Path,
        expected_files: List[str],
        expected_dirs: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        """Assert that the output directory contains expected files/directories.

        Args:
            output_dir: Path to the output directory
            expected_files: List of expected file names (can include paths like "review-plan/grouped.json")
            expected_dirs: Optional list of expected directory names
            message: Optional custom failure message

        Raises:
            AssertionError: If expected files or directories are missing
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise AssertionError(
                message or f"Output directory does not exist: {output_dir}"
            )

        # Check expected files
        missing_files = []
        for file_name in expected_files:
            file_path = output_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            existing = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
            raise AssertionError(
                message
                or f"Missing expected files: {missing_files}. "
                f"Existing files: {existing}"
            )

        # Check expected directories
        if expected_dirs:
            missing_dirs = []
            for dir_name in expected_dirs:
                dir_path = output_dir / dir_name
                if not dir_path.is_dir():
                    missing_dirs.append(dir_name)

            if missing_dirs:
                existing_dirs = [str(p.relative_to(output_dir)) for p in output_dir.iterdir() if p.is_dir()]
                raise AssertionError(
                    message
                    or f"Missing expected directories: {missing_dirs}. "
                    f"Existing directories: {existing_dirs}"
                )

    def assert_phase_directory_structure(
        self,
        result: SkillResult,
        phase: str,
        expected_files: List[str],
        message: Optional[str] = None,
    ) -> None:
        """Assert that a phase directory contains expected files.

        Args:
            result: SkillResult from an orchestrator run
            phase: Phase name (e.g., "review-plan")
            expected_files: List of expected file names within the phase directory
            message: Optional custom failure message

        Raises:
            AssertionError: If phase directory is missing or doesn't contain expected files
        """
        if result.output_dir is None:
            raise AssertionError(
                message or "No output directory in result"
            )

        phase_dir = result.output_dir / phase
        if not phase_dir.exists():
            raise AssertionError(
                message or f"Phase directory does not exist: {phase_dir}"
            )

        self.assert_output_directory_structure(
            phase_dir,
            expected_files,
            message=message,
        )

    def assert_llm_called_with(
        self,
        result: SkillResult,
        prompt_pattern: str,
        provider: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Assert that an LLM was called with a prompt matching a pattern.

        Args:
            result: SkillResult from an orchestrator run
            prompt_pattern: Regex pattern to match against prompts
            provider: Optional provider name to filter by
            message: Optional custom failure message

        Raises:
            AssertionError: If no matching call was found
        """
        calls = result.get_mock_calls()

        if not calls:
            raise AssertionError(
                message
                or f"No mock LLM calls recorded. "
                "Ensure mock binaries are being used (check PATH)."
            )

        # Filter by provider if specified
        if provider:
            calls = [c for c in calls if c.get("provider") == provider]
            if not calls:
                raise AssertionError(
                    message
                    or f"No mock calls for provider '{provider}'. "
                    f"Available providers: {set(c.get('provider') for c in result.get_mock_calls())}"
                )

        # Check for pattern match
        for call in calls:
            prompt = call.get("prompt", "")
            if re.search(prompt_pattern, prompt, re.IGNORECASE):
                return

        # No match found
        prompts_preview = [c.get("prompt", "")[:100] + "..." for c in calls[:3]]
        raise AssertionError(
            message
            or f"No LLM call matched pattern '{prompt_pattern}'. "
            f"Call prompts (first 3): {prompts_preview}"
        )

    def assert_llm_not_called(
        self,
        result: SkillResult,
        message: Optional[str] = None,
    ) -> None:
        """Assert that no LLM calls were made.

        Args:
            result: SkillResult from an orchestrator run
            message: Optional custom failure message

        Raises:
            AssertionError: If any LLM calls were recorded
        """
        calls = result.get_mock_calls()
        if calls:
            providers = set(c.get("provider") for c in calls)
            raise AssertionError(
                message
                or f"Expected no LLM calls but found {len(calls)} calls "
                f"to providers: {providers}"
            )

    def assert_llm_call_count(
        self,
        result: SkillResult,
        expected_count: int,
        provider: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Assert the number of LLM calls.

        Args:
            result: SkillResult from an orchestrator run
            expected_count: Expected number of calls
            provider: Optional provider name to filter by
            message: Optional custom failure message

        Raises:
            AssertionError: If call count doesn't match
        """
        calls = result.get_mock_calls()

        if provider:
            calls = [c for c in calls if c.get("provider") == provider]

        actual_count = len(calls)
        if actual_count != expected_count:
            raise AssertionError(
                message
                or f"Expected {expected_count} LLM calls "
                f"{'for ' + provider if provider else ''} "
                f"but found {actual_count}"
            )

    def assert_mock_was_invoked(
        self,
        result: SkillResult,
        message: Optional[str] = None,
    ) -> None:
        """Assert that the mock LLM was invoked (safety check).

        This verifies that tests are actually using mock binaries and not
        accidentally calling real LLM providers.

        Args:
            result: SkillResult from an orchestrator run
            message: Optional custom failure message

        Raises:
            AssertionError: If mock was not invoked
        """
        if not result.mock_was_invoked():
            raise AssertionError(
                message
                or "Mock LLM was not invoked. This may indicate that real LLM "
                "binaries were called instead of mocks. Check PATH configuration."
            )

    def assert_json_file_valid(
        self,
        file_path: Path,
        expected_type: Optional[type] = None,
        expected_keys: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assert that a JSON file is valid and optionally check its structure.

        Args:
            file_path: Path to the JSON file
            expected_type: Expected top-level type (dict, list)
            expected_keys: Expected keys if top-level is a dict
            message: Optional custom failure message

        Returns:
            The parsed JSON data

        Raises:
            AssertionError: If file doesn't exist, isn't valid JSON, or doesn't match expected structure
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise AssertionError(
                message or f"JSON file does not exist: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(
                message or f"Invalid JSON in {file_path}: {e}"
            )

        if expected_type is not None:
            if not isinstance(data, expected_type):
                raise AssertionError(
                    message
                    or f"Expected {expected_type.__name__} but got {type(data).__name__} "
                    f"in {file_path}"
                )

        if expected_keys is not None:
            if not isinstance(data, dict):
                raise AssertionError(
                    message
                    or f"Cannot check keys: data is {type(data).__name__}, not dict"
                )
            missing_keys = set(expected_keys) - set(data.keys())
            if missing_keys:
                raise AssertionError(
                    message
                    or f"Missing expected keys in {file_path}: {missing_keys}. "
                    f"Present keys: {list(data.keys())}"
                )

        return data

    def assert_state_has_fields(
        self,
        result: SkillResult,
        fields: List[str],
        message: Optional[str] = None,
    ) -> None:
        """Assert that state.json contains specific fields.

        Args:
            result: SkillResult from an orchestrator run
            fields: List of field names to check
            message: Optional custom failure message

        Raises:
            AssertionError: If state doesn't exist or is missing fields
        """
        state = result.get_state()
        if state is None:
            raise AssertionError(
                message or f"state.json not found in {result.output_dir}"
            )

        missing = [f for f in fields if f not in state]
        if missing:
            raise AssertionError(
                message
                or f"State missing fields: {missing}. "
                f"Present fields: {list(state.keys())}"
            )

    def assert_salvage_file_created(
        self,
        result: SkillResult,
        phase: str,
        model: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Path:
        """Assert that a salvage file was created for a phase.

        Args:
            result: SkillResult from an orchestrator run
            phase: Phase name (e.g., "review-plan")
            model: Optional model name to check for specific salvage file
            message: Optional custom failure message

        Returns:
            Path to the salvage file

        Raises:
            AssertionError: If no salvage file was created
        """
        if result.output_dir is None:
            raise AssertionError(
                message or "No output directory in result"
            )

        phase_dir = result.output_dir / phase

        if model:
            # Check for specific model salvage file
            salvage_pattern = f"salvage_{model.replace(':', '_')}*.json"
        else:
            salvage_pattern = "salvage_*.json"

        salvage_files = list(phase_dir.glob(salvage_pattern))

        if not salvage_files:
            existing = list(phase_dir.glob("*.json")) if phase_dir.exists() else []
            raise AssertionError(
                message
                or f"No salvage file matching '{salvage_pattern}' found in {phase_dir}. "
                f"Existing JSON files: {[f.name for f in existing]}"
            )

        return salvage_files[0]

    def assert_salvage_file_contents(
        self,
        salvage_path: Path,
        expected_fields: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assert salvage file has expected structure.

        Args:
            salvage_path: Path to the salvage file
            expected_fields: Expected field names (defaults to standard salvage fields)
            message: Optional custom failure message

        Returns:
            The parsed salvage data

        Raises:
            AssertionError: If salvage file doesn't have expected structure
        """
        if expected_fields is None:
            expected_fields = ["model", "raw_output", "output_path", "timestamp"]

        return self.assert_json_file_valid(
            salvage_path,
            expected_type=dict,
            expected_keys=expected_fields,
            message=message,
        )

    def assert_stdout_contains(
        self,
        result: SkillResult,
        pattern: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that stdout contains a pattern.

        Args:
            result: SkillResult from an orchestrator run
            pattern: String or regex pattern to search for
            message: Optional custom failure message

        Raises:
            AssertionError: If pattern not found in stdout
        """
        if pattern in result.stdout:
            return

        if re.search(pattern, result.stdout, re.IGNORECASE):
            return

        preview = result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
        raise AssertionError(
            message
            or f"Pattern '{pattern}' not found in stdout. "
            f"Stdout preview: {preview}"
        )

    def assert_stderr_contains(
        self,
        result: SkillResult,
        pattern: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that stderr contains a pattern.

        Args:
            result: SkillResult from an orchestrator run
            pattern: String or regex pattern to search for
            message: Optional custom failure message

        Raises:
            AssertionError: If pattern not found in stderr
        """
        if pattern in result.stderr:
            return

        if re.search(pattern, result.stderr, re.IGNORECASE):
            return

        preview = result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr
        raise AssertionError(
            message
            or f"Pattern '{pattern}' not found in stderr. "
            f"Stderr preview: {preview}"
        )

    def assert_exit_code(
        self,
        result: SkillResult,
        expected_code: int,
        message: Optional[str] = None,
    ) -> None:
        """Assert the exit code of the command.

        Args:
            result: SkillResult from an orchestrator run
            expected_code: Expected exit code
            message: Optional custom failure message

        Raises:
            AssertionError: If exit code doesn't match
        """
        if result.exit_code != expected_code:
            raise AssertionError(
                message
                or f"Expected exit code {expected_code} but got {result.exit_code}. "
                f"stderr: {result.stderr[:200] if result.stderr else '(empty)'}"
            )

    def assert_plan_unchanged(
        self,
        original_path: Path,
        original_content: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that a plan file was not modified.

        Args:
            original_path: Path to the plan file
            original_content: Original content of the plan
            message: Optional custom failure message

        Raises:
            AssertionError: If plan content has changed
        """
        if not original_path.exists():
            raise AssertionError(
                message or f"Plan file no longer exists: {original_path}"
            )

        current_content = original_path.read_text(encoding="utf-8")
        if current_content != original_content:
            raise AssertionError(
                message
                or f"Plan file was modified: {original_path}. "
                f"Original length: {len(original_content)}, "
                f"Current length: {len(current_content)}"
            )

    def assert_plan_modified(
        self,
        original_path: Path,
        original_content: str,
        message: Optional[str] = None,
    ) -> None:
        """Assert that a plan file was modified.

        Args:
            original_path: Path to the plan file
            original_content: Original content of the plan
            message: Optional custom failure message

        Raises:
            AssertionError: If plan content is unchanged
        """
        if not original_path.exists():
            raise AssertionError(
                message or f"Plan file no longer exists: {original_path}"
            )

        current_content = original_path.read_text(encoding="utf-8")
        if current_content == original_content:
            raise AssertionError(
                message or f"Plan file was not modified: {original_path}"
            )
