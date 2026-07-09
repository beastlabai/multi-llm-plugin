"""Google Antigravity CLI (agy) provider implementation with JSON envelope parsing."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider


class AgyProvider(LLMProvider):
    """Provider for Google's Antigravity CLI tool (agy).

    agy --print mode with --output-format json emits a SINGLE JSON object
    on one stdout line:
      {"conversation_id": "...", "status": "SUCCESS", "response": "<answer>",
       "duration_seconds": ..., "num_turns": N, "usage": {...}}
    On error (exit 1) the envelope carries "status": "ERROR", an empty
    "response" and an "error" message. The model's answer is the "response"
    field.

    Config model names are the exact DISPLAY NAMES from `agy models`,
    with spaces and parentheses (e.g. "Gemini 3.1 Pro (High)"), passed
    verbatim as a single argv element to --model (list-form subprocess,
    no shell — spaces are safe).

    Gotcha: an invalid --model value is SILENTLY ignored — agy falls back
    to its default model and still reports status SUCCESS — so configured
    names must match `agy models` output exactly.

    --new-project is REQUIRED: without it print mode reuses a previous
    project whose workspace may be bound to a DIFFERENT directory (it then
    silently reads the wrong repo). --new-project binds the workspace to
    the current working directory.

    Auth is Google OAuth only (one-time `agy` sign-in, persisted in
    ~/.gemini/antigravity-cli/); no API keys are involved. An auth lapse
    degrades to the machine-readable ERROR envelope.
    """

    @property
    def name(self) -> str:
        return "agy"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("agy") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # agy --new-project --dangerously-skip-permissions
        #   --output-format json --print-timeout 20m
        #   --model "<Model Display Name>" -p "<prompt>"
        # --new-project binds the workspace to cwd (otherwise a previous
        # project's — possibly different — directory is silently reused);
        # --dangerously-skip-permissions auto-approves tool use so headless
        # file reads/writes work; --print-timeout raises the internal
        # print-mode cutoff (default 5m0s is too low for review-sized runs;
        # 20m aligns with the yaml timeout of 1200s). The model display name
        # is passed verbatim as one argv element.
        return [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            "20m",
            "--model",
            model,
            "-p",
            prompt,
        ]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse the single JSON envelope emitted by agy --output-format json.

        1. Parse the envelope; check "status"
        2. Non-SUCCESS status -> explicit failure carrying the envelope's
           "error" message
        3. SUCCESS -> unwrap "response"; empty -> failure
        4. Parse the response as JSON, with fallback extraction for code
           blocks and JSON embedded in prose
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

        if isinstance(envelope, dict) and "status" in envelope:
            status = envelope.get("status")
            if status != "SUCCESS":
                error = envelope.get("error") or f"agy returned status {status}"
                return {"success": False, "error": error, "raw": stdout, "data": None}

            text = envelope.get("response")
            if not isinstance(text, str) or not text.strip():
                return {"success": False, "error": "Empty response in agy output", "raw": stdout, "data": None}

            text = text.strip()
            if text.startswith(("[", "{")):
                try:
                    return {"success": True, "data": json.loads(text)}
                except json.JSONDecodeError:
                    pass
            return extract_json_from_text(text, prefer_arrays=True)

        # Valid JSON without the envelope shape: treat it as the answer itself
        return {"success": True, "data": envelope}
