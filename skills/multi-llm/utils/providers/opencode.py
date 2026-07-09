"""OpenCode CLI provider implementation with NDJSON event stream parsing."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Reasoning effort values accepted by `opencode run --variant` — the values are
# provider-specific, so this is the practical union across providers
# (verified live against opencode 1.17.15).
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class OpenCodeProvider(LLMProvider):
    """Provider for OpenCode CLI tool.

    OpenCode outputs NDJSON (newline-delimited JSON) events when using --format json.
    Event types include: step_start, text, tool_use, step_finish
    The actual LLM response is in the "text" events' part.text field.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openai/gpt-5.5:high``), translated to ``--variant <effort>``. Valid
    efforts are listed in REASONING_EFFORTS; anything else passes through
    verbatim as the model name.
    """

    @property
    def name(self) -> str:
        return "opencode"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # opencode run --format json --model <model> [--variant <effort>] "<prompt>"
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = ["opencode", "run", "--format", "json", "--model", base_model]
        if effort is not None:
            cmd += ["--variant", effort]
        cmd.append(prompt)
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse NDJSON event stream output from OpenCode.

        1. Parse each line as JSON
        2. Aggregate text from "text" type events
        3. Extract JSON from the aggregated text content
        """
        text_parts: List[str] = []

        # Parse NDJSON - one JSON object per line
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                # Extract text from "text" type events
                if event.get("type") == "text":
                    part = event.get("part", {})
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

        if not text_parts:
            return {"success": False, "error": "No text events found in output", "raw": stdout, "data": None}

        # Concatenate all text parts to get the full response
        full_text = "".join(text_parts).strip()

        if not full_text:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        # Try to parse as JSON directly
        if full_text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(full_text)}
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from the text (may be in code blocks)
        return extract_json_from_text(full_text, prefer_arrays=True)
