"""Claude Code CLI provider implementation."""

import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider


class ClaudeCodeProvider(LLMProvider):
    """Provider for Claude Code CLI tool.

    This provider wraps the Claude Code CLI (claude), which returns responses in a JSON
    wrapper format: {"type":"result","result":"..."}. The provider handles
    extraction of the actual response content and JSON parsing with fallback
    strategies for code blocks and raw JSON.

    When invoked from within a Claude Code session, the CLAUDECODE env var is
    stripped to avoid the nested-session guard.
    """

    @property
    def name(self) -> str:
        """Return the provider identifier."""
        return "claude-code"

    @property
    def default_timeout(self) -> int:
        """Return the default timeout in seconds."""
        return 600

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available in PATH."""
        return shutil.which("claude") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        """Build the claude command with JSON output format.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use (e.g., 'sonnet', 'opus', 'haiku').

        Returns:
            Command arguments for Claude Code CLI invocation.
        """
        # claude -p --output-format json --model <model> <prompt>
        return [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            prompt,
        ]

    def get_remove_env(self) -> List[str]:
        """Strip CLAUDECODE env var to bypass nested-session guard.

        When this provider is invoked from within a Claude Code session,
        the parent process sets CLAUDECODE=1. The child `claude -p` process
        detects this and refuses to start. We strip it so the headless
        subprocess can run without hitting that guard.
        """
        return ["CLAUDECODE"]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse Claude Code output and extract structured data.

        Claude Code with --output-format json returns responses wrapped in:
        {"type":"result","result":"...","session_id":"...","total_cost_usd":...}

        This method unwraps that format and attempts to parse the inner result
        as JSON, with fallback strategies for various response formats.

        Args:
            stdout: The standard output from Claude Code CLI.
            stderr: The standard error (unused but required by interface).

        Returns:
            A dictionary with 'success' and 'data' (or 'error') keys.
        """
        stdout = stdout.strip()

        # Claude Code --output-format json returns: {"type":"result","result":"..."}
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
