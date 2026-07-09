"""OpenAI Codex CLI provider implementation with NDJSON event stream parsing."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Reasoning effort values accepted by the OpenAI API for reasoning.effort
# (verified live against codex-cli 0.144.0).
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class CodexProvider(LLMProvider):
    """Provider for OpenAI Codex CLI tool.

    Codex outputs NDJSON (newline-delimited JSON) events when using --json flag.
    Uses 'codex exec --full-auto --json' for non-interactive execution.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``gpt-5.5:high``), translated to ``-c model_reasoning_effort=<effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name.
    """

    @property
    def name(self) -> str:
        return "codex"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return [
                "codex", "exec", "--full-auto", "--json",
                "--model", base_model,
                "-c", f"model_reasoning_effort={effort}",
                prompt,
            ]
        return ["codex", "exec", "--full-auto", "--json", "--model", model, prompt]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse NDJSON event stream output from Codex CLI."""
        text_parts: List[str] = []

        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                # Handle various event types
                if event_type == "text":
                    if "text" in event:
                        text_parts.append(event["text"])
                    part = event.get("part", {})
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                elif event_type == "message":
                    if "content" in event:
                        text_parts.append(event["content"])
                elif event_type == "content":
                    if "text" in event:
                        text_parts.append(event["text"])
            except json.JSONDecodeError:
                continue

        if not text_parts:
            return {"success": False, "error": "No text events found in output", "raw": stdout, "data": None}

        full_text = "".join(text_parts).strip()

        if not full_text:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        if full_text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(full_text)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(full_text, prefer_arrays=True)
