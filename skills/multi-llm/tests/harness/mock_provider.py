"""MockProvider class for configuring mock LLM behavior.

Provides a programmatic interface for configuring mock_llm.py behavior,
including setting scenarios, responses, and failure injection.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class MockLLMCall:
    """Represents a single mock LLM invocation.

    Attributes:
        timestamp: ISO timestamp of the call
        provider: The provider name detected from sys.argv[0]
        argv: Full command-line arguments
        args: Parsed arguments (excluding program name)
        prompt: The prompt text sent to the mock
        env: Relevant environment variables at call time
    """

    timestamp: str
    provider: str
    argv: List[str]
    args: List[str]
    prompt: str
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockLLMCall":
        """Create a MockLLMCall from a dictionary (JSONL entry)."""
        return cls(
            timestamp=data.get("timestamp", ""),
            provider=data.get("provider", "unknown"),
            argv=data.get("argv", []),
            args=data.get("args", []),
            prompt=data.get("prompt", ""),
            env=data.get("env", {}),
        )

    def prompt_contains(self, pattern: str) -> bool:
        """Check if the prompt contains a pattern (case-insensitive)."""
        return pattern.lower() in self.prompt.lower()

    def prompt_matches(self, pattern: str) -> bool:
        """Check if the prompt matches a regex pattern."""
        import re

        return bool(re.search(pattern, self.prompt, re.IGNORECASE))


class MockProvider:
    """Configures mock LLM behavior for tests.

    Provides a high-level interface for setting up mock responses,
    scenarios, and failure injection. Works in conjunction with
    SkillRunner to configure the mock_llm.py binary.

    Usage:
        mock = MockProvider(tmp_path)
        mock.set_scenario("happy_path")  # Use scenario YAML
        # or
        mock.set_response({"key": "value"})  # Direct response
        # or
        mock.fail_on_call(2)  # Fail on the second call

        # Get configuration for SkillRunner
        env = mock.get_env()
    """

    def __init__(
        self,
        tmp_path: Path,
        scenarios_dir: Optional[Path] = None,
        responses_dir: Optional[Path] = None,
    ):
        """Initialize MockProvider.

        Args:
            tmp_path: Temporary directory for this test
            scenarios_dir: Directory containing scenario YAML files.
                If None, auto-detected from tests/fixtures/e2e/scenarios/
            responses_dir: Directory containing response JSON files.
                If None, auto-detected from tests/fixtures/e2e/responses/
        """
        self.tmp_path = Path(tmp_path)
        self.call_log_path = self.tmp_path / "mock_calls.jsonl"

        # Auto-detect directories
        if scenarios_dir is None:
            self.scenarios_dir = (
                Path(__file__).parent.parent / "fixtures" / "e2e" / "scenarios"
            )
        else:
            self.scenarios_dir = Path(scenarios_dir)

        if responses_dir is None:
            self.responses_dir = (
                Path(__file__).parent.parent / "fixtures" / "e2e" / "responses"
            )
        else:
            self.responses_dir = Path(responses_dir)

        # Configuration state
        self._scenario_path: Optional[Path] = None
        self._fixture_path: Optional[Path] = None
        self._config: Optional[Dict[str, Any]] = None
        self._fail: bool = False
        self._timeout: bool = False
        self._output_path: Optional[Path] = None

        # Track which call number to fail on (0-indexed)
        self._fail_on_calls: List[int] = []
        self._call_count: int = 0

    def set_scenario(self, name: str) -> "MockProvider":
        """Set the scenario to use for response matching.

        Args:
            name: Scenario name (without .yaml extension)

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If the scenario file doesn't exist
        """
        scenario_path = self.scenarios_dir / f"{name}.yaml"
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario not found: {scenario_path}. "
                f"Available: {self._list_scenarios()}"
            )
        self._scenario_path = scenario_path
        # Clear fixture path when setting scenario
        self._fixture_path = None
        self._config = None
        return self

    def set_scenario_path(self, path: Path) -> "MockProvider":
        """Set the scenario to use by direct path.

        Args:
            path: Full path to scenario YAML file

        Returns:
            self for method chaining
        """
        self._scenario_path = Path(path)
        self._fixture_path = None
        self._config = None
        return self

    def _list_scenarios(self) -> List[str]:
        """List available scenarios."""
        if not self.scenarios_dir.exists():
            return []
        return [p.stem for p in self.scenarios_dir.glob("*.yaml")]

    def set_fixture(self, phase: str, name: str) -> "MockProvider":
        """Set a direct fixture file to use as response (legacy mode).

        Args:
            phase: Phase directory (e.g., "review_plan", "validation")
            name: Fixture name (without .json extension)

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If the fixture file doesn't exist
        """
        fixture_path = self.responses_dir / phase / f"{name}.json"
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"Fixture not found: {fixture_path}"
            )
        self._fixture_path = fixture_path
        # Clear scenario when setting fixture
        self._scenario_path = None
        self._config = None
        return self

    def set_fixture_path(self, path: Path) -> "MockProvider":
        """Set a direct fixture file by path (legacy mode).

        Args:
            path: Full path to fixture JSON file

        Returns:
            self for method chaining
        """
        self._fixture_path = Path(path)
        self._scenario_path = None
        self._config = None
        return self

    def set_response(self, data: Union[Dict[str, Any], List[Any], str]) -> "MockProvider":
        """Set a direct response to return (via MOCK_LLM_CONFIG).

        Args:
            data: Response data (will be JSON-serialized if not a string)

        Returns:
            self for method chaining
        """
        if isinstance(data, str):
            self._config = {"response": data}
        else:
            self._config = {"response": data}
        # Clear other response modes
        self._scenario_path = None
        self._fixture_path = None
        return self

    def set_output_path(self, path: Path) -> "MockProvider":
        """Set the output path for mock to write response to.

        Args:
            path: Path where mock should write response

        Returns:
            self for method chaining
        """
        self._output_path = Path(path)
        return self

    def inject_failure(self) -> "MockProvider":
        """Configure mock to fail with error.

        Returns:
            self for method chaining
        """
        self._fail = True
        self._timeout = False
        return self

    def inject_timeout(self) -> "MockProvider":
        """Configure mock to simulate timeout.

        Returns:
            self for method chaining
        """
        self._timeout = True
        self._fail = False
        return self

    def fail_on_call(self, call_number: int) -> "MockProvider":
        """Configure mock to fail on a specific call number.

        Note: This requires test code to track call numbers and set
        MOCK_LLM_FAIL=1 before the specific call. This method just
        records the intention for reference.

        Args:
            call_number: 1-indexed call number to fail on

        Returns:
            self for method chaining
        """
        # Convert to 0-indexed internally
        self._fail_on_calls.append(call_number - 1)
        return self

    def clear(self) -> "MockProvider":
        """Clear all configuration.

        Returns:
            self for method chaining
        """
        self._scenario_path = None
        self._fixture_path = None
        self._config = None
        self._fail = False
        self._timeout = False
        self._output_path = None
        self._fail_on_calls = []
        return self

    def get_env(self) -> Dict[str, str]:
        """Get environment variables for the current configuration.

        Returns:
            Dictionary of environment variables to set
        """
        env = {}

        # Always set call log path
        env["MOCK_LLM_CALL_LOG"] = str(self.call_log_path)

        # Set scenario if configured
        if self._scenario_path:
            env["MOCK_LLM_SCENARIO"] = str(self._scenario_path)

        # Set fixture if configured (legacy mode)
        if self._fixture_path:
            env["MOCK_LLM_FIXTURE"] = str(self._fixture_path)

        # Set config if configured
        if self._config:
            env["MOCK_LLM_CONFIG"] = json.dumps(self._config)

        # Set failure injection
        if self._fail:
            env["MOCK_LLM_FAIL"] = "1"

        # Set timeout injection
        if self._timeout:
            env["MOCK_LLM_TIMEOUT"] = "1"

        # Set output path if configured
        if self._output_path:
            env["MOCK_OUTPUT_PATH"] = str(self._output_path)

        return env

    def get_calls(self) -> List[MockLLMCall]:
        """Load and return all mock LLM calls from the call log.

        Returns:
            List of MockLLMCall objects
        """
        if not self.call_log_path.exists():
            return []

        calls = []
        with open(self.call_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    calls.append(MockLLMCall.from_dict(data))

        return calls

    def get_call_count(self) -> int:
        """Get the total number of mock LLM calls.

        Returns:
            Number of calls recorded in the log
        """
        return len(self.get_calls())

    def get_calls_by_provider(self, provider: str) -> List[MockLLMCall]:
        """Get all calls for a specific provider.

        Args:
            provider: Provider name (e.g., "cursor-agent", "gemini")

        Returns:
            List of MockLLMCall objects for the specified provider
        """
        return [c for c in self.get_calls() if c.provider == provider]

    def get_calls_matching(self, pattern: str) -> List[MockLLMCall]:
        """Get all calls where the prompt matches a pattern.

        Args:
            pattern: Regex pattern to match against prompts

        Returns:
            List of MockLLMCall objects with matching prompts
        """
        return [c for c in self.get_calls() if c.prompt_matches(pattern)]

    def was_invoked(self) -> bool:
        """Check if the mock was invoked at least once.

        This is a safety check to ensure tests are actually using mock
        binaries and not accidentally calling real LLM providers.

        Returns:
            True if at least one call was recorded
        """
        return self.call_log_path.exists() and self.get_call_count() > 0

    def assert_invoked(self, message: Optional[str] = None) -> None:
        """Assert that the mock was invoked at least once.

        Args:
            message: Optional custom failure message

        Raises:
            AssertionError: If mock was not invoked
        """
        if not self.was_invoked():
            default_msg = (
                "Mock LLM was not invoked. This may indicate that real LLM "
                "binaries were called instead of mocks. Check PATH configuration."
            )
            raise AssertionError(message or default_msg)

    def clear_call_log(self) -> None:
        """Clear the call log file."""
        if self.call_log_path.exists():
            self.call_log_path.unlink()

    def create_custom_scenario(
        self,
        prompts: List[Dict[str, str]],
        name: str = "custom",
    ) -> Path:
        """Create a custom scenario YAML file.

        Args:
            prompts: List of dicts with "pattern" and "fixture" keys
            name: Name for the scenario file

        Returns:
            Path to the created scenario file
        """
        import yaml

        scenario = {
            "name": name,
            "description": f"Custom scenario created for test",
            "prompts": prompts,
        }

        scenario_path = self.tmp_path / f"{name}.yaml"
        with open(scenario_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(scenario, f)

        return scenario_path

    def create_response_fixture(
        self,
        data: Any,
        name: str = "custom_response",
    ) -> Path:
        """Create a custom response fixture file.

        Args:
            data: Data to write to the fixture
            name: Name for the fixture file

        Returns:
            Path to the created fixture file
        """
        fixture_path = self.tmp_path / f"{name}.json"
        with open(fixture_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return fixture_path
