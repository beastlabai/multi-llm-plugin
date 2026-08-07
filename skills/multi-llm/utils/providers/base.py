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

    # How build_command() delivers the prompt to the CLI. Every current
    # provider CLI takes the prompt as an argv element (none has a documented
    # stdin or prompt-file mode used anywhere in this codebase), so "argv" is
    # the default. On Windows, argv transport is subject to hard command-line
    # length caps (cmd.exe: 8,191 chars for .cmd/.bat shims; CreateProcess:
    # 32,767), so llm_client enforces a prompt-length check for "argv"
    # providers there. An adapter that grows a stdin ("stdin") or prompt-file
    # ("file") mode should override this to opt out of that check.
    prompt_transport: str = "argv"

    # Which transports this adapter can actually deliver a prompt over (D2d).
    # ``prompt_transport`` above is the one the fan-out path uses by default;
    # this tuple is what a *caller* may ask for, and it is what the materialized
    # -prompt budget is derived from: a materialized prompt is far larger than
    # the metadata JSON it replaces (validation alone permits ~30,000 characters
    # of context before the suggestion list), and argv cannot carry that on
    # Windows, where the whole rendered command line is capped at 8,191 UTF-16
    # code units for a .cmd/.bat shim and 32,767 natively. An adapter declaring
    # "stdin" or "file" makes the budget effectively unbounded; an argv-only
    # adapter keeps the cap, and the producer splits the batch rather than
    # truncating.
    supported_prompt_transports: Tuple[str, ...] = ("argv",)

    def supports_prompt_transport(self, transport: str) -> bool:
        """Whether this adapter can deliver a prompt over ``transport``."""
        return transport in self.supported_prompt_transports

    def build_command_for_transport(
        self,
        prompt: str,
        model: str,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Build argv plus the text to write to the child's stdin.

        Args:
            prompt: The prompt text.
            model: The model identifier.
            transport: ``"argv"`` (prompt is an argv element), ``"stdin"``
                (``build_command`` omits it and the caller writes it to the
                child's stdin) or ``"file"`` (prompt already written to
                ``prompt_file``, whose path travels on argv).
            prompt_file: Path the prompt was written to, for ``"file"``.

        Returns:
            ``(argv, stdin_text)``; ``stdin_text`` is ``None`` unless the
            transport is ``"stdin"``.

        Raises:
            ValueError: when this adapter does not support ``transport``.
        """
        if transport == "argv":
            return self.build_command(prompt, model), None
        raise ValueError(
            f"{self.name} does not support the {transport!r} prompt transport "
            f"(supported: {', '.join(self.supported_prompt_transports)})"
        )

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

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """Extract a human-readable reason from a non-zero-exit invocation.

        ``parse_output`` is only called when the CLI exits 0, so a provider
        that reports its errors on *stdout* rather than stderr would otherwise
        have them discarded — the caller sees a bare exit code with empty
        stderr and nothing to diagnose. Override this to recover the message.

        Args:
            stdout: The standard output from the failed CLI invocation.
            stderr: The standard error from the failed CLI invocation.

        Returns:
            A short description of the failure, or None when the provider has
            nothing to add beyond the exit code and stderr. None by default.
        """
        return None

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
