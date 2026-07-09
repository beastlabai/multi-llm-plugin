"""Cline CLI provider implementation with JSONL event stream parsing."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Thinking levels accepted by `cline --thinking` (verified on cline 3.0.39).
# Hard-validated by the CLI — invalid values exit 1. Unlike -m, --thinking
# does not mutate ~/.cline.
REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})


class ClineProvider(LLMProvider):
    """Provider for the Cline CLI tool.

    Cline with --json runs headless and emits JSONL (one JSON object per
    line): hook_event / agent_event lines followed by a final
    {"type": "run_result", "finishReason": ..., "text": ...} line whose
    "text" field holds the model's answer. On success finishReason is
    "completed"; on error finishReason is "error" and "text" carries the
    error message.

    Config model names use the format "<cline-provider>/<model-id>"
    (e.g. "openrouter/z-ai/glm-5.2"): the segment before the first "/"
    maps to -P (the cline backend provider) and the remainder to -m (the
    model id). A name without "/" is passed straight to -m and uses the
    saved default provider.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openrouter/z-ai/glm-5.2:high``), stripped off the full string BEFORE
    the provider/model split and translated to ``--thinking <effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name (keeping ``:free``-style ids intact).
    """

    @property
    def name(self) -> str:
        return "cline"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("cline") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # cline --json [--thinking <effort>] -P <cline-provider> -m <model-id> "<prompt>"
        # --json auto-activates headless mode; auto-approve defaults to on,
        # so file-read tools work unattended. The prompt goes last as a
        # positional argument.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = ["cline", "--json"]
        if effort is not None:
            cmd += ["--thinking", effort]
        head, _sep, rest = base_model.partition("/")
        if rest:
            cmd += ["-P", head, "-m", rest]
        else:
            cmd += ["-m", base_model]
        cmd.append(prompt)
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse JSONL event stream output from the Cline CLI.

        1. Scan lines for the LAST {"type": "run_result", ...} event
        2. finishReason == "error" -> explicit failure with the error text
        3. Unwrap "text" and parse it as JSON, with fallback extraction for
           code blocks and JSON embedded in prose
        """
        run_result = None
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "run_result":
                run_result = event

        if run_result is None:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": "No run_result event found in output", "raw": stdout, "data": None}

        text = run_result.get("text")
        text = text.strip() if isinstance(text, str) else ""

        if run_result.get("finishReason") == "error":
            return {"success": False, "error": text or "Cline run failed", "raw": stdout, "data": None}

        if not text:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        if text.startswith(("[", "{")):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return extract_json_from_text(text, prefer_arrays=True)
