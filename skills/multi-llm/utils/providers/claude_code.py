"""Claude Code CLI provider implementation."""

import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .base import (
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    USAGE_FULL,
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_RETRY_ENV,
    InternalRetryEnvelope,
    LLMProvider,
    StructuredSchema,
    split_reasoning_effort,
)
from .result_envelope import parse_result_envelope

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
#
# The web tools are denied on this profile, not merely left out of the allow-list.
# Every read/analyze prompt this profile ever runs is `context_mode: inline` or
# `workspace-read` — its inputs travel inside the prompt or come off the pinned
# repo root — so nothing here needs the network, while the text it is fed
# (plans, diffs, review comments) is exactly the untrusted material a prompt
# injection arrives in. With egress available, a `WebFetch` to an
# attacker-named URL carrying repository content in the query string is an
# exfiltration channel that needs no write tool at all. Denying it also puts
# this runner on the same egress footing as the codex read-only profile, whose
# `--sandbox read-only` denies network by default: one task class, one policy,
# whichever runner `driver.subagent_runner` happens to name.
READONLY_ALLOWED_TOOLS = ("Read", "Grep", "Glob", "TodoWrite")
READONLY_DENIED_TOOLS = (
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "Bash",
    "Task",
    "WebFetch",
    "WebSearch",
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

# §9.8. `--allowed-tools` / `--disallowed-tools` take `<tools...>`, so every
# token after one of them — up to a separator — is part of its VALUE. The agent
# profiles used to emit the prompt immediately after `--disallowed-tools`, and
# the variadic swallowed it: the adapter's own rendered argv, run verbatim at
# 2.1.233, exits 1 with `Error: Input must be provided either through stdin or
# as a prompt argument when using --print` and makes **zero requests**. Latent
# only because `preferred_prompt_transport('claude-code') == 'stdin'` drops the
# prompt from argv, but `build_agent_command` is a public, unit-tested surface.
#
# `--` before the prompt is one of the two fixes verified working (1 request,
# exit 0); it is the one chosen here because it keeps the prompt at `argv[-1]`,
# which every `cmd[:-1]` override, `build_agent_command_for_transport` and the
# cross-adapter `agent-write` invariant in `tests/test_providers.py` rely on.
PROMPT_SEPARATOR = "--"

# ─────────────────────────────────────────────────────────────────────────────
# §7.3 — the retry budget is delegated to the vendor CLI, in ITS OWN env names.
#
# The CLI ships a default of **10 retries**. Measured 2026-08-16 against a stub
# that 429s forever: 11 requests and **182 s** in a single rate-limited slot
# before the harness saw anything at all — a meaningful fraction of a 600 s
# provider slot spent inside one subprocess nobody could observe. Setting these
# is closer to a bug fix than a feature.
#
# The two names both donor plans proposed are FICTIONAL and provably inert: they
# occur zero times in the shipped binary (control: the API-key variable occurs
# 185 times), and a request-timeout value expressed in seconds is ignored while
# the millisecond one aborts an 8 s request at 2.37 s. `get_env` must therefore
# never emit them; `tests/test_providers.py` asserts their absence in both
# directions (not in the env, not as a key anywhere in this module).
# ─────────────────────────────────────────────────────────────────────────────

#: §3.6's floor exactly — not one more. Below 3, `--fallback-model`'s overload
#: path is silently dead (it needs three consecutive 529s), which is why L6 and
#: L7 are atomic group A3. Above 3 buys no additional fallback capability and
#: multiplies the worst-case wall by one more API_TIMEOUT_MS each. Values above
#: 15 are silently clamped by the CLI, with a warning.
CLAUDE_CODE_MAX_RETRIES = 3

#: §9.6.c's margin rule applied one level in: the CLI must reach its own cutoff
#: and exit *before* the harness's outer deadline, or the parent kills a live
#: child and reports a bare timeout instead of the CLI's own reason. Named, never
#: a literal inside get_env.
INNER_REQUEST_MARGIN_MS = 30_000

#: The hang ceiling, and the one judgement in §7.3 — flagged as one. It is above
#: every measured single request and below the point at which a request that has
#: produced nothing is anything but hung. FALSIFIER: an attempt aborting at
#: 300 s ± jitter *with a partial answer* is a legitimate generation being
#: truncated (API_TIMEOUT_MS bounds the whole request, not time-to-first-byte),
#: and the ceiling is raised — that is not evidence of a hang.
API_TIMEOUT_CEILING_MS = 300_000


def derive_api_timeout_ms(resolved_timeout_s: int) -> Optional[int]:
    """The CLI's own per-request timeout, in **milliseconds** (§7.3).

    Derived rather than constant, because "sized to sit under the outer
    per-provider slot timeout" is a relation, not a number:

        min(300_000, (slot_s * 1000) // (MAX_RETRIES + 1) - 30_000)

    Both configured ends are safe by construction — 1800 s ⇒ 300 000 ms (worst
    case 4 × 300 s inside a 1800 s slot), 600 s ⇒ 120 000 ms (worst case 4 ×
    120 s inside a 600 s slot).

    Returns ``None`` when the slot is too short to carry a whole retry budget
    (below ~120 s the derivation goes non-positive). Emitting a zero or negative
    value would abort every request instantly, so the variable is omitted and the
    CLI's own request default applies; the retry cap still ships, since it is the
    half that is a bug fix on its own.
    """
    value = min(
        API_TIMEOUT_CEILING_MS,
        (int(resolved_timeout_s) * 1000) // (CLAUDE_CODE_MAX_RETRIES + 1)
        - INNER_REQUEST_MARGIN_MS,
    )
    return value if value > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# §3.6 — `--fallback-model`, and §5.4.1's per-flag capability gate for it.
# ─────────────────────────────────────────────────────────────────────────────

#: Verified functionally at claude **2.1.233** while this adapter's declared
#: floor is `min_version = "2.0.0"`. A measured-at version is an upper bound on
#: the true floor, never the floor itself, so raising `min_version` to it would
#: disable installs that work (§5.4.1(a) — rejected). Hence the gate below.
FALLBACK_MODEL_FLAG = "--fallback-model"
FALLBACK_MODEL_VERIFIED_AT = "2.1.233"

# ─────────────────────────────────────────────────────────────────────────────
# §4.2 — `--json-schema`, and the SAME §5.4.1 gate (M14 absorbs both, later).
# ─────────────────────────────────────────────────────────────────────────────

#: Verified at claude **2.1.227** (§4.2) and re-verified functionally here at
#: 2.1.233 against a localhost Anthropic stub, while this adapter's declared
#: floor is `min_version = "2.0.0"`. Same §5.4.1(a) rejection as
#: `--fallback-model`: a measured-at version is an upper bound on the true floor,
#: so raising the floor to it would disable installs that work. Hence the gate —
#: and it is the same one help read, not a second one.
JSON_SCHEMA_FLAG = "--json-schema"
JSON_SCHEMA_VERIFIED_AT = "2.1.227"

#: One help read per CLI per process, memoised on the RESOLVED binary path, so a
#: four-model × ten-phase run performs at most one help spawn (§5.4.1).
_HELP_TOKENS: Dict[str, frozenset] = {}
#: One WARN per declined capability per process, not one per dispatch.
_GATE_WARNED: Set[Tuple[str, str]] = set()

_FLAG_TOKEN_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")


def reset_flag_gate_cache() -> None:
    """Forget the memoised help read (tests; never called on a run path)."""
    _HELP_TOKENS.clear()
    _GATE_WARNED.clear()


def _read_help_tokens(binary: str, remove_env: Sequence[str] = ()) -> frozenset:
    """The set of long options this binary's own help documents (§5.4.1).

    Containment matches §2.1's probe: ``stdin=DEVNULL``, captured output, a 5 s
    timeout, zero network, and the same env composition a real dispatch uses
    (the adapter's ``get_remove_env`` is applied, so a probe run from inside a
    Claude Code session is not refused by the nested-session guard).

    ``TERM=dumb`` and a wide ``COLUMNS`` are load-bearing: §3.6's own history is
    a *truncated* help rendering that hid what the flag required, and a token
    sniff must not inherit that failure mode. Both streams are read and
    concatenated (§5.4) because some CLIs banner to stderr.

    Failure posture — decline the *flag*, keep the *run*. A timeout, a non-zero
    exit, empty help or an unreadable stream all resolve to an empty set, i.e.
    "unsupported", which drops an optional flag rather than emitting one the
    binary may reject.
    """
    cached = _HELP_TOKENS.get(binary)
    if cached is not None:
        return cached

    env = {key: value for key, value in os.environ.items() if key not in remove_env}
    env["TERM"] = "dumb"
    env["COLUMNS"] = "1000"
    env["NO_COLOR"] = "1"
    tokens: frozenset = frozenset()
    try:
        completed = subprocess.run(
            [binary, "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=env,
        )
        if completed.returncode == 0:
            text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            tokens = frozenset(_FLAG_TOKEN_RE.findall(text))
    except (OSError, subprocess.SubprocessError, ValueError):
        tokens = frozenset()
    _HELP_TOKENS[binary] = tokens
    return tokens


def _warn_once(key: Tuple[str, str], message: str) -> None:
    if key in _GATE_WARNED:
        return
    _GATE_WARNED.add(key)
    print(message, file=sys.stderr)


def _base_name(model: str) -> str:
    """A model id with any ``:effort`` suffix removed."""
    base, _effort = split_reasoning_effort(str(model), REASONING_EFFORTS)
    return base


def validate_fallback_models(provider_config: Dict[str, Any]) -> None:
    """Validate ``providers.claude-code.fallback_models`` (§3.6).

    Run over the **whole** block, not just the key about to be dispatched: the
    contract is "fails the run at load, naming the offending key", so a bad list
    for one primary must not wait for that primary to be selected.

    The CLI **accepts a primary that is also its own fallback silently**
    (measured), so this validation is ours or nobody's — otherwise the argv
    carries a fallback that is a guaranteed no-op and only surfaces as an
    unexplained failure under load.

    Raises:
        ConfigError: naming the offending key, its value and the reason.
    """
    from ..provider_registry import ConfigError  # local: avoids an import cycle

    def fail(message: str) -> None:
        raise ConfigError(f"providers.claude-code.{message}")

    declared = {
        _base_name(entry)
        for entry in (provider_config.get("models") or [])
        if isinstance(entry, str) and entry.strip()
    }

    def check_list(where: str, primary: Optional[str], value: Any) -> None:
        if not isinstance(value, list) or not value:
            fail(
                f"{where}: expected a non-empty list of model ids, got "
                f"{value!r}. Remove the key rather than leaving it empty — the "
                f"shipped value is an empty map and emits no flag at all."
            )
        seen: List[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                fail(f"{where}: {entry!r} is not a model id")
            name = _base_name(entry.strip())
            if primary is not None and name == primary:
                fail(
                    f"{where}: {entry!r} is its own primary. The CLI accepts "
                    f"this silently and the fallback is then a guaranteed no-op."
                )
            if name in seen:
                fail(f"{where}: {entry!r} is repeated")
            seen.append(name)
            if declared and name not in declared:
                fail(
                    f"{where}: {entry!r} is not one of the models this provider "
                    f"declares ({', '.join(sorted(declared))})"
                )

    mapping = provider_config.get("fallback_models")
    if mapping:
        if not isinstance(mapping, dict):
            fail(
                f"fallback_models: expected a map of "
                f"{{<primary-model>: [<fallback>, …]}}, got {mapping!r}"
            )
        for primary, value in mapping.items():
            if not isinstance(primary, str) or not primary.strip():
                fail(f"fallback_models: {primary!r} is not a model id")
            check_list(
                f"fallback_models.{primary}", _base_name(primary.strip()), value
            )

    default = provider_config.get("fallback_models_default")
    if default is not None:
        check_list("fallback_models_default", None, default)


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
    # §4.2.2: this adapter really emits `--json-schema` when handed a schema (see
    # `build_command`), so the caller may mark an accepted result `native`.
    structured_output_support = True
    # No `auth_check`: the CLI ships no non-interactive login-status subcommand
    # (`/status` is an in-session slash command, and `claude setup-token` starts a
    # login flow), and inventing one out of a throwaway generating dispatch would
    # cost a model call per run. The auth state is therefore recorded `unknown`
    # (D3c) and a login notice is still caught reactively by auth_error_patterns.
    login_cmd = "claude login"
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

    # ── §3.6: the fallback list is configuration, and it ships empty ─────────

    def _provider_config(self) -> Dict[str, Any]:
        """This provider's own block from the layered config (§3.6).

        Read through the normal layering (base → auto-discovered
        ``<git-root>/.multi-llm/providers.yaml`` → ``MULTI_LLM_PROVIDERS_CONFIG``),
        so a fallback list is configured exactly where every other provider
        setting is.

        ``build_command`` has no anchor to thread (the signature is the base
        class's, shared by every adapter and every caller), so this is §9.3.b's
        backstop used deliberately and explicitly: the run's pinned plan path,
        or CWD when no pre-flight set one.
        """
        from ..provider_registry import get_run_anchor, load_config  # import cycle

        config = load_config(anchor=get_run_anchor())
        block = (config.get("providers") or {}).get(self.name)
        return block if isinstance(block, dict) else {}

    def fallback_models(self, model: str) -> List[str]:
        """The configured fallback list for ``model``, or ``[]`` (§3.6).

        **Default off, explicit per model, never silently substituted.** The
        probe proves the *flag* works; it says nothing about which model should
        stand in for which. There is no built-in haiku stand-in and no
        `sonnet → haiku` degrade ladder: the configured list *is* the whole
        source, joined verbatim in the user's own order, and an unconfigured
        primary emits no flag at all.

        The whole block is validated here rather than just the selected key, so a
        malformed entry fails the run before any child is spawned.
        """
        block = self._provider_config()
        validate_fallback_models(block)
        mapping = block.get("fallback_models") or {}
        primary = _base_name(model)
        configured = None
        if isinstance(mapping, dict):
            configured = mapping.get(primary) or mapping.get(model)
        if configured is None:
            configured = block.get("fallback_models_default")
        if not configured:
            return []
        return [str(entry).strip() for entry in configured]

    def supports_flag(self, flag: str) -> bool:
        """§5.4.1's per-flag capability gate, in its *optional-flag* form.

        Shipped inside this task rather than taken from §11.2's shared version
        machinery, because §3.6 is in the severable live-defect set and may not
        depend on it (M14 later generalizes the one-offs and deletes this).

        Unsupported *or undecidable* ⇒ ``False``: on an older CLI the flag is
        ``unknown option`` and a dead slot, so the gate declines toward the
        status quo. The token is matched as a whole word — ``--fallback-model``
        is a prefix of ``--fallback-model-chain``, and a substring probe would
        answer yes for a flag that does not exist.
        """
        binary = shutil.which("claude")
        if not binary:
            return False
        return flag in _read_help_tokens(binary, self.get_remove_env())

    def _fallback_flag_supported(self) -> bool:
        supported = self.supports_flag(FALLBACK_MODEL_FLAG)
        if not supported:
            _warn_once(
                ("claude", FALLBACK_MODEL_FLAG),
                f"[WARN] claude-code: {FALLBACK_MODEL_FLAG} is not documented by "
                f"the installed CLI's own help (verified at "
                f"{FALLBACK_MODEL_VERIFIED_AT}); the configured fallback models "
                f"are not passed and this run has no overload/404 fallback. "
                f"Everything else is unchanged.",
            )
        return supported

    def _json_schema_flag_supported(self) -> bool:
        """§5.4.1's gate for ``--json-schema``, answered from the SAME help read.

        One help read per CLI per process, memoised on the resolved binary — not
        one per flag — so adding this capability adds no spawn. Optional-flag
        posture: unsupported or undecidable ⇒ the flag is omitted with one WARN
        and the phase falls back to prompt-level JSON plus §4.2.1's recovery
        branch. The degraded run is a documented degraded mode, and it is marked
        ``recovered`` (or fails as ``schema_violation``), **never** ``native``.
        """
        supported = self.supports_flag(JSON_SCHEMA_FLAG)
        if not supported:
            _warn_once(
                ("claude", JSON_SCHEMA_FLAG),
                f"[WARN] claude-code: {JSON_SCHEMA_FLAG} is not documented by "
                f"the installed CLI's own help (verified at "
                f"{JSON_SCHEMA_VERIFIED_AT}); the schema is not passed and this "
                f"phase falls back to prompt-level JSON with schema-validated "
                f"recovery. The result is marked 'recovered', never 'native'.",
            )
        return supported

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build the claude command with JSON output format.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use (e.g., 'sonnet', 'opus', 'haiku').
            timeout_s: Accepted and ignored — claude has no internal print-mode
                cutoff to keep under the parent's deadline (§9.6.a).
            schema: §4.2.2's optional structured-output schema. claude's
                transport is **inline** (`--json-schema <json>`), so only
                ``schema.document`` is read and ``schema.path`` is irrelevant
                here — the asymmetry with codex's file is encoded in the two
                adapters, not in the caller.

        Returns:
            Command arguments for Claude Code CLI invocation.
        """
        # claude -p --output-format json [--json-schema <json>]
        #        --model <model> [--effort <effort>]
        #        [--fallback-model <m1,m2>] <prompt>
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
        ]
        # §4.2.2's normative splice point, at a FIXED INDEX immediately after
        # `--output-format json` — never an append before `cmd[-1]`. That one
        # position makes all six rows right at once (generate on argv and on
        # stdin, both agent profiles on both transports):
        #
        # * it is inside the policy region, ahead of `--model`, so the prompt is
        #   never the token after the schema and the schema is never the token
        #   after the prompt;
        # * it precedes the variadic `--allowed-tools` / `--disallowed-tools`
        #   the agent profiles splice in later, so §9.8's swallowing trap cannot
        #   reach it;
        # * `cmd[-1] is the prompt` still holds, which every `cmd[:-1]` override
        #   and `build_agent_command_for_transport` relies on.
        #
        # The value is rendered with sorted keys and compact separators so one
        # schema file always produces one argv byte string — a golden fixture
        # cannot churn on dict ordering.
        if schema is not None and self._json_schema_flag_supported():
            cmd += [
                JSON_SCHEMA_FLAG,
                json.dumps(
                    dict(schema.document), sort_keys=True, separators=(",", ":")
                ),
            ]
        cmd += [
            "--model",
            base_model,
        ]
        if effort is not None:
            cmd += ["--effort", effort]
        # §3.6's splice point: in the policy region beside --model/--effort,
        # INSIDE build_command and ahead of the prompt append. That one position
        # makes all four rows right at once — generate on argv and on stdin, and
        # both agent profiles, each carrying the flag *before* the variadic
        # --allowed-tools / --disallowed-tools and never adjacent to the prompt.
        # Appending it after the prompt would break `cmd[-1] is the prompt` in
        # two directions: build_command_for_transport's stdin branch returns
        # cmd[:-1] and would ship a valueless --fallback-model, and the agent
        # profiles splice their tool policy in after cmd[-1].
        fallbacks = self.fallback_models(model)
        if fallbacks and self._fallback_flag_supported():
            cmd += [FALLBACK_MODEL_FLAG, ",".join(fallbacks)]
        cmd.append(prompt)
        return cmd

    def build_command_for_transport(
        self,
        prompt: str,
        model: str,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Deliver the prompt on argv or on stdin.

        Under ``stdin`` the prompt is omitted from argv entirely and returned as
        the text to write to the child, so no part of it is measured against a
        command-line cap. The fixed-index schema splice is what makes the two
        transports differ by **exactly** the prompt element and nothing else.
        """
        if transport != "stdin":
            return super().build_command_for_transport(
                prompt, model, transport, prompt_file, timeout_s=timeout_s,
                schema=schema,
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        return cmd[:-1], prompt

    def build_agent_command(
        self,
        prompt: str,
        model: str,
        profile: str = PROFILE_AGENT_READONLY,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
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
            cmd = self.build_command(
                prompt, model, timeout_s=timeout_s, schema=schema
            )
            return [
                *cmd[:-1],
                "--permission-mode",
                "acceptEdits",
                "--allowed-tools",
                ",".join(WRITE_ALLOWED_TOOLS),
                "--disallowed-tools",
                ",".join(WRITE_DENIED_TOOLS),
                PROMPT_SEPARATOR,
                cmd[-1],
            ]
        if profile != PROFILE_AGENT_READONLY:
            return super().build_agent_command(
                prompt, model, profile, timeout_s=timeout_s, schema=schema
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        # build_command appends the prompt last; splice the policy in before it.
        return [
            *cmd[:-1],
            "--allowed-tools",
            ",".join(READONLY_ALLOWED_TOOLS),
            "--disallowed-tools",
            ",".join(READONLY_DENIED_TOOLS),
            PROMPT_SEPARATOR,
            cmd[-1],
        ]

    def build_agent_command_for_transport(
        self,
        prompt: str,
        model: str,
        profile: str = PROFILE_AGENT_READONLY,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Drop the ``--`` separator along with the prompt it shields (§9.8).

        The separator exists for exactly one reason: to end option parsing so the
        variadic ``--disallowed-tools`` cannot swallow the trailing positional.
        Under the ``stdin`` transport there is no positional left, so a surviving
        bare ``--`` would be a token with nothing to separate — and it would move
        the two claude-code agent rows in ``tests/container/expected_argv.yaml``
        (both ``transport: stdin``) for no behavioural reason. Dropping it keeps
        the stdin argv byte-identical to what this adapter has always emitted.
        """
        argv, stdin_text = super().build_agent_command_for_transport(
            prompt, model, profile, transport, prompt_file, timeout_s=timeout_s,
            schema=schema,
        )
        if transport == "stdin" and argv and argv[-1] == PROMPT_SEPARATOR:
            return argv[:-1], stdin_text
        return argv, stdin_text

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

        # §3.6: cost follows the served model. This extractor already holds the
        # one envelope that carries `modelUsage`, so the attribution rides the
        # same channel as the tokens it belongs to rather than being re-parsed.
        # It is a string among numbers on purpose — the numeric accumulator
        # ignores it, and nothing about "absent means not reported" changes.
        served = self._served_model_from_envelope(envelope)
        if served:
            out["served_model"] = served
        return out or None

    @staticmethod
    def _served_model_from_envelope(envelope: Dict[str, Any]) -> Optional[str]:
        """The model that actually served the turn, from ``modelUsage``.

        Keyed by model id. With more than one key (a turn that changed models
        mid-flight), the one that produced the most output is the one the answer
        came from; ties keep insertion order.
        """
        usage = envelope.get("modelUsage")
        if not isinstance(usage, dict) or not usage:
            return None
        names = [name for name in usage if isinstance(name, str) and name]
        if not names:
            return None

        def produced(name: str) -> float:
            entry = usage.get(name)
            if not isinstance(entry, dict):
                return 0.0
            value = entry.get("outputTokens", entry.get("output_tokens", 0))
            return float(value) if isinstance(value, (int, float)) else 0.0

        return max(names, key=produced)

    def parse_served_model(self, stdout: str, stderr: str = "") -> Optional[str]:
        """``served_model`` from the ``--output-format json`` envelope (§3.6).

        claude's envelope carries ``modelUsage`` keyed by the model that actually
        served the turn — measured: a 529 run's fourth request and its
        ``modelUsage`` both name ``claude-haiku-4-5-20251001``. Returns ``None``
        for anything unparseable; this is attribution, never a failure path.
        """
        text = (stdout or "").strip()
        if not text.startswith("{"):
            return None
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(envelope, dict):
            return None
        return self._served_model_from_envelope(envelope)

    def resolve_served_model(
        self,
        stdout: str,
        requested_model: str,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """``(served_model, is_fallback, warning)`` for one attempt (§3.6).

        Attribution is mandatory whenever a fallback is configured: without it a
        report attributes a haiku answer to the sonnet the user asked for, and
        the quality and cost delta is invisible at exactly the moment it matters.
        When the two differ the caller emits one WARN naming phase / requested /
        served, and every report that names a model labels that row with the
        **served** model and marks it ``fallback``.

        **Read, never depended on.** If ``modelUsage`` is absent, or names a
        model outside {primary} ∪ configured fallbacks, the *requested* model is
        recorded and a warning is returned — an attribution field never fails an
        otherwise-successful attempt.

        Alias matching is deliberately loose in one direction only: config names
        aliases (``sonnet``, ``haiku``) while the envelope names resolved ids
        (``claude-haiku-4-5-20251001``), so an alias matches an id that contains
        it. A miss costs a WARN and the requested model, never the run.
        """
        requested = _base_name(requested_model)
        served = self.parse_served_model(stdout, "")
        if not served:
            return (
                requested_model,
                False,
                f"claude-code: the result envelope carried no `modelUsage`, so "
                f"the answer is attributed to the requested model "
                f"({requested_model}) — it may have been served by a fallback.",
            )

        def matches(alias: str) -> bool:
            alias = _base_name(str(alias)).strip().lower()
            low = served.lower()
            return bool(alias) and (alias == low or alias in low)

        if matches(requested):
            return served, False, None
        for candidate in fallback_models or ():
            if matches(candidate):
                return (
                    served,
                    True,
                    f"claude-code: `--fallback-model` fired — requested "
                    f"{requested_model}, served {served}. The result is "
                    f"attributed to the served model.",
                )
        return (
            requested_model,
            False,
            f"claude-code: `modelUsage` names {served}, which is neither the "
            f"requested model ({requested_model}) nor a configured fallback "
            f"({', '.join(fallback_models or ()) or 'none'}); recording the "
            f"requested model.",
        )

    def resolved_timeout_s(self) -> int:
        """The outer per-provider slot timeout `get_env` derives from (§7.3).

        The same value §7.5 rule 1 takes its deadline from: ``providers.yaml``'s
        ``default_timeout`` (**1800** for claude-code) or the configured
        override, falling back to this adapter's own ``default_timeout`` (600)
        when the provider is absent from config or the read fails.

        ``get_env`` has no anchor to thread either, so it reads the same §9.3.b
        run anchor ``_provider_config`` does — the alternative is a slot timeout
        resolved against CWD while the fan-out resolves its own against the
        plan, i.e. an inner budget sized for the wrong outer deadline.
        """
        try:
            from ..provider_registry import (  # import cycle
                get_provider_timeout,
                get_run_anchor,
            )

            value = int(get_provider_timeout(self.name, anchor=get_run_anchor()))
        except Exception:  # noqa: BLE001 - env shaping must never fail a run
            return self.default_timeout
        return value if value > 0 else self.default_timeout

    @property
    def internal_retries(self) -> InternalRetryEnvelope:
        """What `claude -p` does inside one awaited subprocess (§7.6(b)).

        Measured 2026-08-16 against a stub that 429s forever: **11 requests and
        182 s** at the CLI's own default of 10 retries. The number declared here
        is **4**, the post-L6 one, because L6 has landed and every dispatch this
        harness makes carries ``CLAUDE_CODE_MAX_RETRIES=3``; declaring 11 would
        budget for a configuration nothing here runs.

        A **property**, not a constant, because ``max_wall_s`` is 4 ×
        ``API_TIMEOUT_MS`` and ``API_TIMEOUT_MS`` is itself *derived from the
        resolved slot timeout* (§7.3) — 1200 s at a 1800 s slot, 480 s at a 600 s
        one. Both halves therefore come from the same
        :func:`derive_api_timeout_ms` call ``get_env`` makes, so the bound §7.5
        rule 4 subtracts can never disagree with the env the child was handed.
        A slot too short to carry a whole retry budget emits no ``API_TIMEOUT_MS``
        at all, and the wall is then honestly ``None`` rather than a guess.

        ``retries_rate_limit`` is ``True`` — the 182 s measurement *is* a
        rate-limit chain — which is what makes §7.5 rule 6(b) forbid an identical
        rate-limit re-dispatch to this adapter: ≤ 3 harness backoffs would turn 4
        requests into 20 against the same upstream for the same verdict.
        """
        requests = CLAUDE_CODE_MAX_RETRIES + 1
        api_timeout_ms = derive_api_timeout_ms(self.resolved_timeout_s())
        return InternalRetryEnvelope(
            max_requests=requests,
            max_wall_s=(
                None if api_timeout_ms is None else requests * api_timeout_ms / 1000.0
            ),
            retries_rate_limit=True,
            configurable_by=("CLAUDE_CODE_MAX_RETRIES", "API_TIMEOUT_MS"),
        )

    def get_env(self, model: str) -> Dict[str, str]:
        """Delegate retry to the vendor CLI's own engine (§7.3, task L6).

        Absorbs transient 429s inside the one subprocess already awaited, instead
        of wrapping it in a harness-side retry loop. Two variables, both in the
        CLI's real names (see the module comment for the two fictional ones this
        must never emit), and both chosen together with §3.6's
        ``--fallback-model``: below three retries its overload path is silently
        dead, which is why they are atomic group A3.
        """
        env = {"CLAUDE_CODE_MAX_RETRIES": str(CLAUDE_CODE_MAX_RETRIES)}
        api_timeout_ms = derive_api_timeout_ms(self.resolved_timeout_s())
        if api_timeout_ms is not None:
            env["API_TIMEOUT_MS"] = str(api_timeout_ms)
        # §4.2.1's pinned structured-output budget. Emitted UNCONDITIONALLY, not
        # only when a schema is attached: the variable is inert without
        # `--json-schema`, and making it conditional would mean a dispatch that
        # acquired a schema by some other route (a future caller, a config
        # default) silently ran at the CLI's default of FIVE paid model calls.
        # Setting it here also keeps it in the one env builder both lanes share,
        # so Lane A and Lane B can never disagree about the budget.
        env[STRUCTURED_OUTPUT_RETRY_ENV] = str(MAX_STRUCTURED_OUTPUT_RETRIES)
        return env

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
        # One shared implementation (§9.5): cursor-agent emits the same envelope
        # and used to carry a byte-identical copy of the unwrapping.
        return parse_result_envelope(stdout)
