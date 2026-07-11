"""Pi coding agent CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

# Thinking levels accepted by `pi --thinking` (per pi.dev docs). pi also
# parses the same levels as a native ":level" suffix on --model; the adapter
# strips the suffix itself so only --thinking carries it.
REASONING_EFFORTS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})


class PiProvider(LLMProvider):
    """Provider for the Pi coding agent CLI (https://pi.dev).

    pi -p (print mode) runs one-shot headless and writes the final assistant
    text to stdout as plain text; --no-session keeps runs ephemeral (no
    session files written). On error pi exits non-zero with the message on
    stderr — the runner turns that into a failure before parse_output runs,
    so parse_output only ever sees exit-0 plain text.

    Model names are pi model patterns passed VERBATIM to --model — pi
    resolves "provider/id" forms itself (e.g. "openrouter/z-ai/glm-5.2",
    "anthropic/claude-opus-4-8") — no partition/splitting. Auth is pi's own:
    provider API-key env vars (e.g. OPENROUTER_API_KEY, ANTHROPIC_API_KEY)
    inherited by the subprocess, or keys stored via pi's /login
    (~/.pi/agent/auth.json).

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openrouter/z-ai/glm-5.2:high``), translated to ``--thinking <effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name (keeping ``:free``-style ids — and
    bedrock ``...-v1:0`` version suffixes — intact). Stripping the suffix
    before --model also keeps pi's own native ":level" parsing out of the
    picture, so the whitelist here is the single source of truth.

    pi has no permission prompts: built-in tools (read, bash, edit, write,
    grep, find, ls) run with the pi process's own permissions, so the
    headless file reads/writes review prompts rely on work unattended with
    no extra flags.
    """

    @property
    def name(self) -> str:
        return "pi"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("pi") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # pi --no-session -p [--thinking <effort>] --model <model> "<prompt>"
        # -p prints the final answer and exits; the prompt goes last as a
        # positional argument.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = ["pi", "--no-session", "-p"]
        if effort is not None:
            cmd += ["--thinking", effort]
        cmd += ["--model", base_model, prompt]
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse plain-text output from pi print mode.

        Attempts direct JSON parsing first, then falls back to extraction
        from code blocks or JSON embedded in prose.
        """
        stdout = stdout.strip()

        if not stdout:
            return {"success": False, "error": "Empty output", "raw": stdout, "data": None}

        if stdout.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(stdout, prefer_arrays=True)
