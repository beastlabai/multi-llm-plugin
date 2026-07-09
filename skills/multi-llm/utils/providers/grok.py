"""Grok Build CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider


class GrokProvider(LLMProvider):
    """Provider for xAI's Grok Build CLI tool.

    Grok Build with --output-format json returns a single JSON object:
    {"text": "...", "stopReason": "...", "sessionId": "...", ...}

    The provider unwraps the "text" field and attempts to parse it as JSON,
    with fallback extraction for code blocks and JSON embedded in prose.
    """

    @property
    def name(self) -> str:
        return "grok"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("grok") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # grok --no-auto-update --always-approve -p <prompt> --output-format json -m <model>
        # --no-auto-update skips background update checks in automated runs;
        # --always-approve auto-approves tool executions (headless runs would
        # otherwise stall waiting for interactive approval).
        return [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "-p",
            prompt,
            "--output-format",
            "json",
            "-m",
            model,
        ]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse JSON wrapper output from Grok Build CLI."""
        stdout = stdout.strip()
        try:
            wrapper = json.loads(stdout)
            if isinstance(wrapper, dict) and "text" in wrapper:
                text = wrapper["text"]
                if isinstance(text, str):
                    text = text.strip()
                    if not text:
                        return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}
                    if text.startswith(("[", "{")):
                        try:
                            return {"success": True, "data": json.loads(text)}
                        except json.JSONDecodeError:
                            pass
                    return extract_json_from_text(text, prefer_arrays=True)
                return {"success": True, "data": text}
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}
