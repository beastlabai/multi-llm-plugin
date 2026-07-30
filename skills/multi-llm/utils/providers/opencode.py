"""OpenCode CLI provider implementation with NDJSON event stream parsing."""
import json
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Reasoning effort values accepted by `opencode run --variant` — the values are
# provider-specific, so this is the practical union across providers
# (verified live against opencode 1.17.15).
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class OpenCodeProvider(LLMProvider):
    """Provider for OpenCode CLI tool.

    OpenCode outputs NDJSON (newline-delimited JSON) events when using
    --format json. The top-level event types are ``step_start``,
    ``step_finish``, ``tool_use``, ``text``, ``reasoning`` (only with
    ``--thinking``, which this adapter never passes) and ``error``; each
    carries a ``part`` object except ``error``, which carries ``error``.
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

    @staticmethod
    def _iter_events(stdout: str):
        """Yield the well-formed JSON objects in an NDJSON stream."""
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
            if isinstance(event, dict):
                yield event

    @staticmethod
    def _direct_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse ``text`` as one complete JSON document, or return None."""
        if text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return None

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """Recover the reason for a non-zero exit from the event stream.

        OpenCode writes ``error`` events to *stdout* and leaves stderr empty,
        so without this the caller sees only "exited with code 1". Verified
        against a live auth failure, which produced a 304-byte error event on
        stdout and zero bytes on stderr.
        """
        messages: List[str] = []
        for event in self._iter_events(stdout):
            if event.get("type") != "error":
                continue
            error = event.get("error")
            if not isinstance(error, dict):
                continue
            name = error.get("name")
            data = error.get("data")
            detail = data.get("message") if isinstance(data, dict) else None
            parts = [p for p in (name, detail) if isinstance(p, str) and p]
            if parts:
                messages.append(": ".join(parts))
        return "; ".join(messages) if messages else None

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse NDJSON event stream output from OpenCode.

        OpenCode emits one assistant message per step, so a run that uses
        tools produces many ``text`` events — real sessions reach 47. The
        final answer is the last one; earlier ones are narration that may
        itself contain a JSON code block, so extracting from the concatenation
        would return a draft rather than the answer.

        Candidates are therefore ordered by confidence rather than position:
        an exact parse of the whole stream or of the final message is
        unambiguous and is tried first, and only then the loose extraction
        that can match a fragment or an early draft.
        """
        text_parts: List[str] = []
        for event in self._iter_events(stdout):
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                text = part["text"]
                if isinstance(text, str):
                    text_parts.append(text)

        if not text_parts:
            error = "No text events found in output"
            reason = self.describe_failure(stdout, stderr)
            if reason:
                error = f"OpenCode reported an error: {reason}"
            return {"success": False, "error": error, "raw": stdout, "data": None}

        last = text_parts[-1].strip()
        concatenated = "".join(text_parts).strip()
        # Newline-joined for the loose pass: separate steps are separate
        # messages, and gluing them directly runs the end of one into the
        # start of the next.
        joined = "\n".join(text_parts).strip()

        if not concatenated:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        # Exact parses first — they are unambiguous. The concatenation covers
        # a single document that arrived split across parts (a split can fall
        # mid-token, so only a separator-less join reassembles it); `last`
        # covers the ordinary case where the closing message is the answer.
        for candidate in (concatenated, last):
            if candidate:
                direct = self._direct_json(candidate)
                if direct is not None:
                    return direct

        # Then the loose pass, final message first so a late answer wins over
        # a draft quoted earlier in the run.
        result: Dict[str, Any] = {}
        for candidate in (last, joined):
            if not candidate:
                continue
            # Try to extract JSON from the text (may be in code blocks)
            result = extract_json_from_text(candidate, prefer_arrays=True)
            if result.get("success"):
                return result

        return result
