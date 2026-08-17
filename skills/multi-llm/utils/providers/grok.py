"""Grok Build CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema, split_reasoning_effort

# Reasoning effort values accepted by `grok --reasoning-effort`
# (verified live against grok 0.2.93).
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class GrokProvider(LLMProvider):
    """Provider for xAI's Grok Build CLI tool.

    Grok Build with --output-format json returns a single JSON object:
    {"text": "...", "stopReason": "...", "sessionId": "...", ...}

    The provider unwraps the "text" field and attempts to parse it as JSON,
    with fallback extraction for code blocks and JSON embedded in prose.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``grok-4:high``), translated to ``--reasoning-effort <effort>``. Valid
    efforts are listed in REASONING_EFFORTS; anything else passes through
    verbatim as the model name.
    """

    #: §9.4: the prompt trails. It used to sit at index 4 — the one adapter that
    #: rendered a flag *after* it — which made `cmd[:-1]` and
    #: `build_agent_command_for_transport` silently wrong the moment grok gained
    #: a stdin transport or an agent profile. It has neither today, so
    #: normalizing the order changes no dispatched behaviour: `-p` takes exactly
    #: one value and the flags are order-independent. Note the pair: a future
    #: non-argv transport must drop `-p` **and** the prompt, not `cmd[:-1]`.
    prompt_argv_index = -1

    @property
    def name(self) -> str:
        return "grok"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("grok") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # grok --no-auto-update --always-approve --output-format json
        #   -m <model> [--reasoning-effort <effort>] -p <prompt>
        # --no-auto-update skips background update checks in automated runs;
        # --always-approve auto-approves tool executions (headless runs would
        # otherwise stall waiting for interactive approval).
        # The `-p <prompt>` pair trails so the prompt is argv[-1] (§9.4); see
        # `prompt_argv_index` above for why the pair, not just the prompt, is
        # what a non-argv transport would have to drop.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "--output-format",
            "json",
            "-m",
            base_model,
        ]
        if effort is not None:
            cmd += ["--reasoning-effort", effort]
        cmd += ["-p", prompt]
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse JSON wrapper output from Grok Build CLI."""
        stdout = stdout.strip()
        try:
            wrapper = json.loads(stdout)
            if isinstance(wrapper, dict) and "text" in wrapper:
                text = wrapper["text"]
                if isinstance(text, str):
                    text = text.strip()
                    if not text:
                        return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}
                    if text.startswith(("[", "{")):
                        try:
                            return {"success": True, "data": json.loads(text)}
                        except json.JSONDecodeError:
                            pass
                    return extract_json_from_text(text, prefer_arrays=True)
                return {"success": True, "data": text}
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}
