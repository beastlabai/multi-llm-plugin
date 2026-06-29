"""Cursor-agent CLI provider implementation."""

import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider


class CursorAgentProvider(LLMProvider):
    """Provider for cursor-agent CLI tool.

    This provider wraps the cursor-agent CLI, which returns responses in a JSON
    wrapper format: {"type":"result","result":"..."}. The provider handles
    extraction of the actual response content and JSON parsing with fallback
    strategies for code blocks and raw JSON.
    """

    @property
    def name(self) -> str:
        """Return the provider identifier."""
        return "cursor-agent"

    @property
    def default_timeout(self) -> int:
        """Return the default timeout in seconds."""
        return 600

    def is_available(self) -> bool:
        """Check if cursor-agent CLI is available in PATH."""
        return shutil.which("cursor-agent") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        """Build the cursor-agent command with JSON output format.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use.

        Returns:
            Command arguments for cursor-agent CLI invocation.
        """
        # cursor-agent --print -f --output-format json --model <model> <prompt>
        return [
            "cursor-agent",
            "--print",
            "-f",
            "--output-format",
            "json",
            "--model",
            model,
            prompt,
        ]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse cursor-agent output and extract structured data.

        The cursor-agent with --output-format json returns responses wrapped in:
        {"type":"result","result":"..."}

        This method unwraps that format and attempts to parse the inner result
        as JSON, with fallback strategies for various response formats.

        Args:
            stdout: The standard output from cursor-agent.
            stderr: The standard error (unused but required by interface).

        Returns:
            A dictionary with 'success' and 'data' (or 'error') keys.
        """
        stdout = stdout.strip()

        # cursor-agent --output-format json returns: {"type":"result","result":"..."}
        if stdout.startswith("{"):
            try:
                wrapper = json.loads(stdout)
                if isinstance(wrapper, dict) and "result" in wrapper:
                    inner_result = wrapper["result"]
                    if isinstance(inner_result, str):
                        inner_result = inner_result.strip()
                        if inner_result.startswith(("[", "{")):
                            try:
                                return {"success": True, "data": json.loads(inner_result)}
                            except json.JSONDecodeError:
                                pass
                        # Try to extract JSON from text
                        return extract_json_from_text(inner_result, prefer_arrays=True)
                    return (
                        {"success": True, "data": inner_result}
                        if inner_result
                        else {"success": False, "error": "Empty result", "data": None}
                    )
                return {"success": True, "data": wrapper}
            except json.JSONDecodeError:
                pass

        # Try direct JSON parsing for arrays
        if stdout.startswith("["):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(stdout, prefer_arrays=True)
