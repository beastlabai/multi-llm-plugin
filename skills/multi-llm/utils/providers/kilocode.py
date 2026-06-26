"""Kilo Code CLI provider implementation."""
import json
import re
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, ModelListing, build_models_listing, parse_line_ids


class KiloCodeProvider(LLMProvider):
    """Provider for Kilo Code CLI tool.

    Kilo Code uses `run --auto` for non-interactive mode with plain text output.
    The --json flag is avoided because it streams every reasoning token as a
    full-content-so-far JSON line, producing massive output (~2.6MB for ~50KB
    of actual content). Model selection is via `-m provider/model` flag.
    """

    @property
    def name(self) -> str:
        return "kilocode"

    @property
    def default_timeout(self) -> int:
        return 600

    can_list_models = True

    def is_available(self) -> bool:
        return shutil.which("kilocode") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        return ["kilocode", "run", "--auto", "-m", model, prompt]

    def list_models(self, curated: List[str], *, timeout: int = 10) -> ModelListing:
        """List kilocode models via `kilocode models` (curated fallback).

        Output is one id per line; ids are long namespaced paths
        (``kilo/anthropic/claude-opus-4.8``, ``openrouter/moonshotai/kimi-k2``)
        kept verbatim. Lines with a ``:`` suffix (``…:free`` / ``…:discounted``)
        are dropped per the bare-id contract. See ``parse_line_ids``.
        """
        return build_models_listing(
            ["kilocode", "models"], parse_line_ids, curated, timeout=timeout
        )

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse output from Kilo Code CLI.

        Strips ANSI escape codes from plain text output, then attempts
        JSON extraction from code blocks or embedded JSON in text.
        """
        # Strip ANSI escape codes from plain text output
        stdout = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        stdout = stdout.strip()

        if not stdout:
            return {"success": False, "error": "Empty output", "raw": stdout, "data": None}

        # Try direct JSON parsing
        if stdout.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        # Fall back to extraction from text/code blocks
        return extract_json_from_text(stdout, prefer_arrays=True)
