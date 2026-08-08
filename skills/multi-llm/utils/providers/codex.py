"""OpenAI Codex CLI provider implementation with NDJSON event stream parsing."""
import json
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..json_extractor import extract_json_from_text
from .base import (
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    USAGE_FULL,
    LLMProvider,
    split_reasoning_effort,
)

# Reasoning effort values accepted by the OpenAI API for reasoning.effort
# (verified live against codex-cli 0.144.0).
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class CodexProvider(LLMProvider):
    """Provider for OpenAI Codex CLI tool.

    Codex outputs NDJSON (newline-delimited JSON) events when using --json flag.
    Uses 'codex exec --sandbox workspace-write --json' for non-interactive
    execution. The sandbox mode must be passed explicitly: codex's no-flag
    default is trust-dependent (read-only in a repo the user has not marked
    trusted), and this skill runs codex in arbitrary repositories.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``gpt-5.5:high``), translated to ``-c model_reasoning_effort=<effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name.
    """

    # `codex exec` reads the prompt from stdin when no prompt argument is given
    # (`codex exec < prompt.txt`), so this adapter has a real non-argv transport
    # (D2d). The fan-out default stays "argv"; materialized prompts use stdin.
    supported_prompt_transports = ("argv", "stdin")

    # Subagent-runner declarations (D3e/D3j/D3m). `-s read-only` is one of the
    # three documented sandbox policies (`codex exec --help`, verified against
    # codex-cli 0.146.0), so the read-only profile is the CLI's own sandbox
    # rather than anything this repo enforces afterwards.
    agent_profiles = (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE)
    supports_file_edits = True
    supports_workspace_reads = True
    usage_support = USAGE_FULL
    version_cmd = ("codex", "--version")
    min_version = "0.144.0"
    verified_version = "0.146.0"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        # --sandbox workspace-write, not the older --full-auto: 0.146.0 keeps
        # --full-auto only as a hidden compatibility trap (internally
        # `removed_full_auto`) and warns on every run. The two produce an
        # identical effective policy — approval_policy is forced to "never"
        # in exec mode regardless — so this is a rename, not a permission
        # change.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return [
                "codex", "exec", "--sandbox", "workspace-write", "--json",
                "--model", base_model,
                "-c", f"model_reasoning_effort={effort}",
                prompt,
            ]
        return [
            "codex", "exec", "--sandbox", "workspace-write", "--json",
            "--model", model, prompt,
        ]

    def build_command_for_transport(
        self,
        prompt: str,
        model: str,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Deliver the prompt on argv or on stdin (prompt omitted from argv)."""
        if transport != "stdin":
            return super().build_command_for_transport(
                prompt, model, transport, prompt_file
            )
        cmd = self.build_command(prompt, model)
        return cmd[:-1], prompt

    def build_agent_command(
        self, prompt: str, model: str, profile: str = PROFILE_AGENT_READONLY
    ) -> List[str]:
        """Build ``codex exec`` under an explicit sandbox policy (D3e).

        ``agent-readonly`` swaps ``--sandbox workspace-write`` (what the
        ``generate`` path hardcodes) for ``--sandbox read-only``: codex refuses
        model-generated writes itself, so a prompt-injected validator cannot edit
        the repository while pretending to score suggestions.

        ``agent-write`` keeps ``--sandbox workspace-write`` — codex's own
        documented write policy, rooted at the process ``cwd``, which the executor
        pins to the resolved repo root (D3h). It is deliberately **not**
        ``--dangerously-bypass-approvals-and-sandbox`` (nor the older
        ``--yolo``): a runner with no honored write sandbox is refused for
        ``agent-write``, so fabricating a bypass here would delete the very
        boundary that makes the profile eligible. Network access is left at
        codex's own workspace-write default rather than being widened here.
        """
        if profile == PROFILE_AGENT_WRITE:
            # Identical to `build_command`, which already names the sandbox
            # explicitly: codex's no-flag default is trust-dependent, so the flag
            # is the policy statement, and restating it would be the only change.
            return self.build_command(prompt, model)
        if profile != PROFILE_AGENT_READONLY:
            return super().build_agent_command(prompt, model, profile)
        cmd = self.build_command(prompt, model)
        return ["read-only" if arg == "workspace-write" else arg for arg in cmd]

    def parse_usage(self, stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
        """Read usage off the terminal turn event of ``codex exec --json`` (D3j).

        Selected by event identity (``turn.completed`` carrying a ``usage``
        object), never by position, and the **last** such event wins — the item
        stream the answer is recovered from carries no usage at all, so the two
        never interfere.
        """
        usage: Optional[Dict[str, Any]] = None
        for line in (stdout or "").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") not in ("turn.completed", "token_count"):
                continue
            candidate = event.get("usage")
            if isinstance(candidate, dict):
                usage = candidate

        if not usage:
            return None

        def _num(key: str) -> Optional[Any]:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            return None

        out: Dict[str, Any] = {}
        input_tokens = _num("input_tokens")
        cached = _num("cached_input_tokens")
        output_tokens = _num("output_tokens")
        reasoning = _num("reasoning_output_tokens")
        if input_tokens is not None:
            out["input_tokens"] = input_tokens
        if cached is not None:
            out["cached_input_tokens"] = cached
        if output_tokens is not None:
            # Reasoning output is billed output; codex reports it separately.
            out["output_tokens"] = output_tokens + (reasoning or 0)
        if "input_tokens" in out and "output_tokens" in out:
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        return out or None

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
