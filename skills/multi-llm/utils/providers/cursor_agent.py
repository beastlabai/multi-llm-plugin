"""Cursor-agent CLI provider implementation."""

import json
import re
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import (
    LLMProvider,
    ModelListing,
    build_models_listing,
    is_valid_bare_id,
    strip_ansi,
    try_parse_json_ids,
)

# A `cursor-agent models` line is "<id> - <Description>" (e.g. "gpt-5.2-high -
# GPT-5.2 High"). The id is the first whitespace-free token, separated from the
# description by " - ". The "Available models" header (no " - ") and the trailing
# "Tip: ..." footer (no " - " after "Tip:", and it carries a ":") don't match.
_MODEL_LINE_RE = re.compile(r"^(\S+)\s+-\s+\S")


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

    can_list_models = True

    def is_available(self) -> bool:
        """Check if cursor-agent CLI is available in PATH."""
        return shutil.which("cursor-agent") is not None

    @staticmethod
    def _parse_models(raw: str) -> List[str]:
        """Parse `cursor-agent models` stdout → de-duplicated bare model ids.

        Accepts both the plain-text "<id> - <desc>" listing and a JSON shape
        (list of strings or {id/name} objects). Strips ANSI, skips the header /
        blank / "Tip:" footer lines, and drops any id with whitespace or ``:``.
        """
        raw = strip_ansi(raw)
        json_ids = try_parse_json_ids(raw)
        if json_ids is not None:
            candidates = json_ids
        else:
            candidates = [
                m.group(1)
                for line in raw.splitlines()
                if (m := _MODEL_LINE_RE.match(line.strip()))
            ]
        seen: set = set()
        out: List[str] = []
        for c in candidates:
            if is_valid_bare_id(c) and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def list_models(self, curated: List[str], *, timeout: int = 10) -> ModelListing:
        """List cursor-agent models via `cursor-agent models` (curated fallback)."""
        return build_models_listing(
            ["cursor-agent", "models"], self._parse_models, curated, timeout=timeout
        )

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
