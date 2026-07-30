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

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Parse ``text`` as JSON, falling back to embedded-JSON extraction."""
        if text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(text, prefer_arrays=True)

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse the NDJSON event stream emitted by ``codex exec --json``.

        The assistant's reply arrives as ``item.completed`` events whose nested
        item has ``type == "agent_message"``. Three details matter, all verified
        against live codex-cli 0.144.0 output:

        - ``reasoning`` items are also ``item.completed`` and also carry a
          ``text`` field, so the item type must be checked or chain-of-thought
          leaks into the payload.
        - ``item.started``/``item.updated`` carry streaming deltas; ignoring
          them keeps text from being counted twice.
        - A turn may contain several agent messages (prose, then the answer),
          so the last is preferred over concatenating standalone JSON documents.
        """
        messages: List[str] = []
        legacy_parts: List[str] = []
        fatal_errors: List[str] = []
        item_errors: List[str] = []

        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            event_type = event.get("type", "")

            if event_type == "item.completed":
                item = event.get("item")
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str):
                        messages.append(text)
                elif item_type == "error":
                    # Not fatal on its own: codex emits these for benign
                    # conditions such as unrecognised model metadata, on runs
                    # that go on to succeed.
                    message = item.get("message")
                    if isinstance(message, str) and message:
                        item_errors.append(message)
            elif event_type == "error":
                message = event.get("message")
                if isinstance(message, str) and message:
                    fatal_errors.append(message)
            elif event_type == "turn.failed":
                error = event.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message:
                        fatal_errors.append(message)
            # Legacy event shapes, kept as a fallback for Codex builds that
            # predate the dotted-namespace stream. No 0.14x release emits these.
            elif event_type == "text":
                if isinstance(event.get("text"), str):
                    legacy_parts.append(event["text"])
                part = event.get("part")
                if isinstance(part, dict) and part.get("type") == "text" \
                        and isinstance(part.get("text"), str):
                    legacy_parts.append(part["text"])
            elif event_type == "message":
                if isinstance(event.get("content"), str):
                    legacy_parts.append(event["content"])
            elif event_type == "content":
                if isinstance(event.get("text"), str):
                    legacy_parts.append(event["text"])

        candidates: List[str] = []
        if messages:
            candidates.append(messages[-1])
            if len(messages) > 1:
                candidates.append("\n".join(messages))
        elif legacy_parts:
            candidates.append("".join(legacy_parts))

        if not candidates:
            error = "No text events found in output"
            reported = fatal_errors or item_errors
            if reported:
                error = "Codex reported an error: " + "; ".join(reported)
            return {"success": False, "error": error, "raw": stdout, "data": None}

        if not any(candidate.strip() for candidate in candidates):
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        result: Dict[str, Any] = {}
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            result = self._extract_json(candidate)
            if result.get("success"):
                return result

        if fatal_errors:
            result = dict(result)
            result["error"] = "{} (codex reported: {})".format(
                result.get("error", "Failed to parse response"), "; ".join(fatal_errors)
            )
        return result
