"""Claude Code CLI provider implementation."""

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

# Reasoning effort values accepted by `claude --effort`
# (verified live against claude 2.1.205).
REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

# The `agent-readonly` tool policy (D3e). Both flags are documented in
# `claude --help` (verified against 2.1.224) and take a comma- or
# space-separated list; a single comma-joined argv element is used so no part of
# the policy can be re-split by a shell.
#
# The allow-list is the positive half — only these tools may run — and the
# deny-list is belt-and-braces for the edit-capable ones plus Bash, which would
# otherwise be an unrestricted write primitive. Both are passed because an
# allow-list alone is a *permission* statement, while the deny-list survives a
# future default that widens the allowed set.
READONLY_ALLOWED_TOOLS = ("Read", "Grep", "Glob", "WebFetch", "WebSearch", "TodoWrite")
READONLY_DENIED_TOOLS = (
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "Bash",
    "Task",
)

# The `agent-write` tool policy (D3g/D3l). `--permission-mode acceptEdits` is
# Claude Code's own pre-approval for edits (documented in `claude --help`,
# verified against 2.1.224); it is NOT a sandbox, which is exactly why the
# executor layers the `driver.write_paths` policy and the pre-batch snapshot on
# top of it (D3l: the runner's mechanism is the preventive control, the policy is
# the detective second layer).
#
# `Task` stays denied on this profile too: a nested subagent would run outside
# every ceiling the executor computed (D3c `nested_exec`) and its writes would
# not appear in this dispatch's audit record. `Bash` IS allowed — an implement
# batch legitimately runs a formatter or a test — and that is precisely the hole
# the post-run policy recomputation exists to catch.
WRITE_ALLOWED_TOOLS = (
    "Read",
    "Grep",
    "Glob",
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "Bash",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)
WRITE_DENIED_TOOLS = ("Task",)


class ClaudeCodeProvider(LLMProvider):
    """Provider for Claude Code CLI tool.

    This provider wraps the Claude Code CLI (claude), which returns responses in a JSON
    wrapper format: {"type":"result","result":"..."}. The provider handles
    extraction of the actual response content and JSON parsing with fallback
    strategies for code blocks and raw JSON.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``opus:high``), translated to ``--effort <effort>``. Valid efforts are
    listed in REASONING_EFFORTS; anything else passes through verbatim as
    the model name.

    When invoked from within a Claude Code session, the CLAUDECODE env var is
    stripped to avoid the nested-session guard.
    """

    # `claude -p` reads the prompt from stdin when no prompt argument is given,
    # so this adapter has a real non-argv transport (D2d). Materialized prompts
    # are dispatched over stdin, which removes the Windows command-line cap from
    # the prompt budget entirely; the fan-out path's default stays "argv" so
    # existing invocations are byte-identical.
    supported_prompt_transports = ("argv", "stdin")

    # Subagent-runner declarations (D3e/D3j/D3m). `agent-write` is the
    # `--permission-mode acceptEdits` invocation below; `supports_file_edits`
    # tracks it so `tests/test_providers.py` can cross-check the declaration
    # against the implemented branch in both directions.
    agent_profiles = (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE)
    supports_file_edits = True
    supports_workspace_reads = True
    usage_support = USAGE_FULL
    version_cmd = ("claude", "--version")
    min_version = "2.0.0"
    verified_version = "2.1.224"

    @property
    def name(self) -> str:
        """Return the provider identifier."""
        return "claude-code"

    @property
    def default_timeout(self) -> int:
        """Return the default timeout in seconds."""
        return 600

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available in PATH."""
        return shutil.which("claude") is not None

    def build_command(self, prompt: str, model: str) -> List[str]:
        """Build the claude command with JSON output format.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use (e.g., 'sonnet', 'opus', 'haiku').

        Returns:
            Command arguments for Claude Code CLI invocation.
        """
        # claude -p --output-format json --model <model> [--effort <effort>] <prompt>
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            base_model,
        ]
        if effort is not None:
            cmd += ["--effort", effort]
        cmd.append(prompt)
        return cmd

    def build_command_for_transport(
        self,
        prompt: str,
        model: str,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Deliver the prompt on argv or on stdin.

        Under ``stdin`` the prompt is omitted from argv entirely and returned as
        the text to write to the child, so no part of it is measured against a
        command-line cap.
        """
        if transport != "stdin":
            return super().build_command_for_transport(
                prompt, model, transport, prompt_file
            )
        cmd = self.build_command(prompt, model)
        return cmd[:-1], prompt

    def build_agent_command(
        self, prompt: str, model: str, profile: str = PROFILE_AGENT_READONLY
    ) -> List[str]:
        """Build an agentic ``claude -p`` invocation for ``profile`` (D3e).

        ``agent-readonly`` is ``build_command`` plus an explicit tool policy:
        an allow-list of read-only tools and a deny-list of every edit-capable
        one. Writes are refused by Claude Code's own permission mechanism, not by
        anything this repo enforces after the fact, which is what makes it safe
        to hand an untrusted (plan/diff/review) prompt to.

        ``agent-write`` swaps that policy for the edit-capable one and adds
        ``--permission-mode acceptEdits``, which is what makes a headless apply /
        implement batch possible at all. It is deliberately **not** a
        ``--dangerously-skip-permissions``-class bypass: the executor pins ``cwd``
        to the resolved repo root (D3h), enforces ``driver.write_paths`` on both
        sides of the run and snapshots every file the batch may touch (D3l).
        """
        if profile == PROFILE_AGENT_WRITE:
            cmd = self.build_command(prompt, model)
            return [
                *cmd[:-1],
                "--permission-mode",
                "acceptEdits",
                "--allowed-tools",
                ",".join(WRITE_ALLOWED_TOOLS),
                "--disallowed-tools",
                ",".join(WRITE_DENIED_TOOLS),
                cmd[-1],
            ]
        if profile != PROFILE_AGENT_READONLY:
            return super().build_agent_command(prompt, model, profile)
        cmd = self.build_command(prompt, model)
        # build_command appends the prompt last; splice the policy in before it.
        return [
            *cmd[:-1],
            "--allowed-tools",
            ",".join(READONLY_ALLOWED_TOOLS),
            "--disallowed-tools",
            ",".join(READONLY_DENIED_TOOLS),
            cmd[-1],
        ]

    def parse_usage(self, stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
        """Recover usage from the ``--output-format json`` result envelope (D3j).

        ``parse_output`` unwraps ``result`` and **discards the envelope**, which
        is where ``usage`` and ``total_cost_usd`` live, so this captures it before
        it is dropped. Fields are selected by name and omitted when absent — a
        zero would read as a free run.
        """
        text = (stdout or "").strip()
        if not text.startswith("{"):
            return None
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            return None

        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        out: Dict[str, Any] = {}

        def _put(key: str, value: Any) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = value

        _put("input_tokens", usage.get("input_tokens"))
        _put("output_tokens", usage.get("output_tokens"))
        # Claude Code reports cache reads and cache writes separately; only the
        # read half is "cached input".
        _put("cached_input_tokens", usage.get("cache_read_input_tokens"))
        _put("cost_usd", envelope.get("total_cost_usd"))
        if "input_tokens" in out and "output_tokens" in out:
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        return out or None

    def get_remove_env(self) -> List[str]:
        """Strip CLAUDECODE env var to bypass nested-session guard.

        When this provider is invoked from within a Claude Code session,
        the parent process sets CLAUDECODE=1. The child `claude -p` process
        detects this and refuses to start. We strip it so the headless
        subprocess can run without hitting that guard.
        """
        return ["CLAUDECODE"]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse Claude Code output and extract structured data.

        Claude Code with --output-format json returns responses wrapped in:
        {"type":"result","result":"...","session_id":"...","total_cost_usd":...}

        This method unwraps that format and attempts to parse the inner result
        as JSON, with fallback strategies for various response formats.

        Args:
            stdout: The standard output from Claude Code CLI.
            stderr: The standard error (unused but required by interface).

        Returns:
            A dictionary with 'success' and 'data' (or 'error') keys.
        """
        stdout = stdout.strip()

        # Claude Code --output-format json returns: {"type":"result","result":"..."}
        if stdout.startswith("{"):
            try:
                wrapper = json.loads(stdout)
                if isinstance(wrapper, dict) and "result" in wrapper:
                    inner_result = wrapper["result"]
                    if isinstance(inner_result, str):
                        inner_result = inner_result.strip()
                        if inner_result.startswith(("[", "{")):
                            try:
                                return {"success": True, "data": json.loads(inner_result)}
                            except json.JSONDecodeError:
                                pass
                        # Try to extract JSON from text
                        return extract_json_from_text(inner_result, prefer_arrays=True)
                    return (
                        {"success": True, "data": inner_result}
                        if inner_result
                        else {"success": False, "error": "Empty result", "data": None}
                    )
                return {"success": True, "data": wrapper}
            except json.JSONDecodeError:
                pass

        # Try direct JSON parsing for arrays
        if stdout.startswith("["):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(stdout, prefer_arrays=True)
