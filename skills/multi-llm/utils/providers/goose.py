"""Block goose CLI provider implementation with JSON envelope parsing."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Thinking effort values accepted by goose's ThinkingEffort parser, aliases
# included (verified on goose 1.41.0). goose has no CLI flag for this — the
# value goes via the GOOSE_THINKING_EFFORT env var / config key; invalid
# values are silently ignored by goose.
REASONING_EFFORTS = frozenset({"off", "none", "low", "medium", "high", "max", "xhigh"})


class GooseProvider(LLMProvider):
    """Provider for Block's goose CLI tool.

    goose run with --output-format json emits a single JSON envelope on
    stdout: {"messages": [...], "metadata": {...}}. Each message is
    {"role": "user"|"assistant", "content": [{"type": "text"|"thinking"|
    "toolRequest"|"toolResponse", ...}, ...]}. The model's answer is the
    concatenated "text" parts (type == "text") of the LAST message with
    role == "assistant".

    Config model names use the format "<goose-provider>/<model-id>"
    (e.g. "openrouter/z-ai/glm-5.2"): the segment before the first "/"
    maps to --provider (the goose backend provider) and the remainder to
    --model. A name without "/" is passed straight to --model and uses
    the GOOSE_PROVIDER env / configured default provider.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openrouter/z-ai/glm-5.2:high``), stripped off the full string BEFORE
    the provider/model split. goose has no CLI flag for reasoning effort,
    so get_env() maps the suffix to the GOOSE_THINKING_EFFORT env var
    instead; without a suffix the var is left unset so goose falls back to
    the user's own config. Valid efforts are listed in REASONING_EFFORTS;
    anything else passes through verbatim as the model name (keeping
    ``:free``-style ids intact).

    Gotcha: goose exits 0 EVEN ON PROVIDER ERRORS (401, rate limit).
    Failure manifests as the final assistant text containing
    "Ran into this error:" — parse_output detects this and returns an
    explicit failure result.
    """

    @property
    def name(self) -> str:
        return "goose"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("goose") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # goose run --no-session -q --output-format json --no-profile
        #   --max-turns 25 --with-builtin developer
        #   --provider <goose-provider> --model <model-id> -t "<prompt>"
        # --no-session avoids session files; the developer builtin enables
        # headless file read/write (review prompts ask the model to WRITE
        # the output JSON file) with no approval prompts. Auth is env-only
        # (e.g. OPENROUTER_API_KEY), inherited by the subprocess.
        cmd = [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
        ]
        base_model, _effort = split_reasoning_effort(model, REASONING_EFFORTS)
        head, _sep, rest = base_model.partition("/")
        if rest:
            cmd += ["--provider", head, "--model", rest]
        else:
            cmd += ["--model", base_model]
        cmd += ["-t", prompt]
        return cmd

    def get_env(self, model: str) -> Dict[str, str]:
        # goose has no CLI flag for reasoning effort; a ``:effort`` model
        # suffix is delivered via GOOSE_THINKING_EFFORT instead. Without a
        # suffix the var is deliberately NOT set, so goose falls back to
        # the user's own config.
        _base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return {"GOOSE_THINKING_EFFORT": effort}
        return {}

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse the single JSON envelope emitted by goose --output-format json.

        1. Parse the envelope; find the LAST message with role == "assistant"
        2. Concatenate its content[].text parts where type == "text"
        3. "Ran into this error:" in the text -> explicit failure (goose
           exits 0 even on provider errors)
        4. Parse the text as JSON, with fallback extraction for code blocks
           and JSON embedded in prose
        """
        stdout = stdout.strip()
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}

        messages = envelope.get("messages") if isinstance(envelope, dict) else None
        if not isinstance(messages, list):
            messages = []

        text = None
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                parts = [
                    part.get("text")
                    for part in message.get("content", [])
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ]
                text = "\n".join(parts)

        if text is None:
            return {"success": False, "error": "No assistant message in output", "raw": stdout, "data": None}

        text = text.strip()

        if "Ran into this error:" in text:
            return {"success": False, "error": text, "raw": stdout, "data": None}

        if not text:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        if text.startswith(("[", "{")):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return extract_json_from_text(text, prefer_arrays=True)
