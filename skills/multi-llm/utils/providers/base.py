"""Base protocol for LLM CLI providers."""

from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional, Tuple


def split_reasoning_effort(model: str, efforts: Collection[str]) -> Tuple[str, Optional[str]]:
    """Split an optional trailing ``:effort`` suffix off a model string.

    Returns ``(base_model, effort)`` when the substring after the LAST colon
    is a member of ``efforts`` and the part before it is non-empty; otherwise
    ``(model, None)`` and the model string passes through verbatim. Splitting
    on the last colon keeps model ids that legitimately contain colons (e.g.
    openrouter ``:free`` variants) intact unless the suffix is whitelisted.
    """
    base, _, suffix = model.rpartition(":")
    if base and suffix in efforts:
        return base, suffix
    return model, None


class LLMProvider(ABC):
    """Abstract base class for LLM CLI providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods and properties. This ensures a consistent
    interface for invoking different LLM CLI tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'cursor-agent', 'gemini').

        Returns:
            A unique string identifier for this provider.
        """
        pass

    @property
    @abstractmethod
    def default_timeout(self) -> int:
        """Default timeout in seconds for this provider.

        Returns:
            The default number of seconds to wait before timing out.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the CLI tool is available in PATH.

        Returns:
            True if the CLI tool can be found and executed, False otherwise.
        """
        pass

    @abstractmethod
    def build_command(self, prompt: str, model: str) -> List[str]:
        """Build the command line arguments for invocation.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use for generation.

        Returns:
            A list of command line arguments suitable for subprocess.run().
        """
        pass

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse output and return structured result.

        Args:
            stdout: The standard output from the CLI invocation.
            stderr: The standard error from the CLI invocation.

        Returns:
            A dictionary with at least 'success' (bool) and 'data' keys.
            On success, 'data' contains the parsed response.
            On failure, 'data' may contain error information.
        """
        pass

    def get_env(self, model: str) -> Dict[str, str]:
        """Return additional environment variables for subprocess.

        Override this method if the provider requires environment variables
        for configuration (e.g., model selection).

        Args:
            model: The model identifier to use for generation.

        Returns:
            A dictionary of environment variables to set for the subprocess.
            Empty dict by default.
        """
        return {}

    def get_remove_env(self) -> List[str]:
        """Return environment variable names to remove for subprocess.

        Override this method if the provider needs certain env vars stripped
        from the inherited environment (e.g., to avoid nested-session guards).

        Returns:
            A list of environment variable names to remove. Empty by default.
        """
        return []
