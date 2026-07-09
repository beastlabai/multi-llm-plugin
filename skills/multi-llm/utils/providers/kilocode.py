"""Kilo Code CLI provider implementation."""
import json
import re
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Reasoning variant names accepted by `kilocode run --variant` (verified on
# kilocode 7.4.1). Variants are model-specific: effort words for
# OpenAI-family/glm models, "thinking"/"instant" for kimi/minimax; unknown
# variants are silently ignored by the CLI.
REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant"}
)


class KiloCodeProvider(LLMProvider):
    """Provider for Kilo Code CLI tool.

    Kilo Code uses `run --auto` for non-interactive mode with plain text output.
    The --json flag is avoided because it streams every reasoning token as a
    full-content-so-far JSON line, producing massive output (~2.6MB for ~50KB
    of actual content). Model selection is via `-m provider/model` flag.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openai-native/gpt-5.5:high``), translated to ``--variant <effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name (keeping ``:free``-style ids intact).
    """

    @property
    def name(self) -> str:
        return "kilocode"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("kilocode") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return ["kilocode", "run", "--auto", "-m", base_model, "--variant", effort, prompt]
        return ["kilocode", "run", "--auto", "-m", model, prompt]

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
