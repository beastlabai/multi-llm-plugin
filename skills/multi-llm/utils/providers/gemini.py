"""Gemini CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Provider for Gemini CLI tool."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_timeout(self) -> int:
        return 900  # Gemini can be slower

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # gemini --output-format json --model <model> "<prompt>"
        return ["gemini", "--output-format", "json", "--model", model, prompt]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        try:
            wrapper = json.loads(stdout)
            # Gemini returns: {"session_id": "...", "response": "...", "stats": {...}}
            if isinstance(wrapper, dict) and "response" in wrapper:
                response = wrapper["response"]
                # Try to parse the response as JSON if it looks like JSON
                if isinstance(response, str):
                    response = response.strip()
                    if response.startswith(('[', '{')):
                        try:
                            return {"success": True, "data": json.loads(response)}
                        except json.JSONDecodeError:
                            pass
                    # Apply JSON extraction fallback for non-JSON starting responses
                    return extract_json_from_text(response, prefer_arrays=True)
                return {"success": True, "data": response}
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}
