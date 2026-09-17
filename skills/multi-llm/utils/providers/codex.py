"""OpenAI Codex CLI provider implementation with NDJSON event stream parsing."""
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..json_extractor import extract_json_from_text
from .base import (
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    SANDBOX_FALLBACK_AUTO,
    SANDBOX_FALLBACK_OFF,
    SANDBOX_FALLBACK_POLICIES,
    SANDBOX_FALLBACK_PROFILES,
    USAGE_FULL,
    REASONING_LADDER,
    InternalRetryEnvelope,
    LLMProvider,
    ReasoningLadder,
    StructuredOutputUnavailable,
    StructuredSchema,
    _provider_block,
    _resolve_policy,
    split_reasoning_effort,
)

# Reasoning effort values this adapter EMITS as `model_reasoning_effort`
# (verified live against codex-cli 0.144.0/0.146.0).
#
# `max` was missing until L11, and its absence was not inert: `gpt-5.6-sol:max`
# is a combination codex's own catalog advertises (`codex debug models` →
# `supported_reasoning_levels` lists `max` for gpt-5.6-sol/-terra/-luna), so a
# whitelist stopping at `xhigh` made `split_reasoning_effort` find no known
# suffix and pass `gpt-5.6-sol:max` through as the MODEL NAME — a model id that
# does not exist, with no effort set at all.
#
# `none`/`minimal` stay in the set because the CLI takes them on the wire, but
# no catalog entry advertises them: they may be passed through when a user asks
# for one, never chosen by a default or a rung-down.
#
# codex is the one adapter that takes the ladder whole — all seven rungs, under
# their canonical names — plus one inbound-only alias below.
EFFORT_LADDER = ReasoningLadder(
    provider="codex",
    rungs=REASONING_LADDER,
    # Inbound-only alias: accepted on a model spec, rewritten before argv is
    # built, and never emitted. codex takes `ultra` as an input alias and
    # normalizes it down to `max` client-side; the harness does that rewrite
    # itself so the emitted value is always a name this adapter states, and
    # codex's own normalization becomes an unused backstop rather than a
    # dependency. (Muse is the mirror case: it has no `max` and renders the same
    # pair in the opposite direction.)
    inbound_aliases=(("ultra", "max"),),
)
REASONING_EFFORTS = EFFORT_LADDER.accepted
EFFORT_ALIASES = EFFORT_LADDER.aliases

#: What a model spec may CARRY, as opposed to what argv may show.
_ACCEPTED_EFFORTS = EFFORT_LADDER.accepted_inbound

#: The codex sandbox policy each invocation profile runs under — the one place
#: in this adapter a `--sandbox` value is spelled (M5).
#:
#: Before this table the same three policies were stated three unconnected ways:
#: a literal inside `_BASE_ARGV` for `generate`, an *implicit* re-use of that
#: literal by `build_agent_command`'s `agent-write` leg (which simply returned
#: `build_command`), and a second literal in its `agent-readonly` leg's
#: positional rewrite. Nothing tied them together, so editing the `generate`
#: default silently edited BOTH agent profiles: agent-write inherits it and the
#: readonly rewrite degenerates into a no-op, which makes agent-write and
#: agent-readonly byte-identical while `supports_file_edits` stays `True` —
#: nothing refuses the dispatch, so every write batch would run, move zero bytes
#: and be recorded as a D3g "clean failure". Each profile now names its own
#: policy here, so a change to one row cannot travel to another.
#:
#: The values are codex's own documented sandbox policies (`codex exec --help`,
#: verified against codex-cli 0.146.0) rather than anything this repo enforces
#: after the fact, which is what makes `agent-readonly` a real denial mechanism
#: and `agent-write` a real (cwd-rooted) boundary rather than a bypass.
_SANDBOX_BY_PROFILE = {
    PROFILE_GENERATE: "workspace-write",
    PROFILE_AGENT_WRITE: "workspace-write",
    PROFILE_AGENT_READONLY: "read-only",
}

#: The one `--sandbox` value this adapter renders that is NOT a row of the table
#: above, and the only way it is ever rendered is `_apply_sandbox(...,
#: fallback=True)` — a dispatch-time override no default argv carries (D6,
#: 2026-09-17). `danger-full-access` is codex's own third documented policy
#: (`codex exec --help`: `-s, --sandbox {read-only, workspace-write,
#: danger-full-access}`), chosen over `--dangerously-bypass-approvals-and-sandbox`
#: because approvals are already `never` in exec mode (see `build_command`) and
#: because it lands in the slot the positional rewrite already owns, so the
#: fallback is a value change and not a new flag. Measured on codex-cli 0.154.0
#: with the adapter's own probe shape: `codex sandbox -c 'sandbox_mode=
#: "read-only"' -- sh -c 'exit 0'` dies on `bwrap: loopback: Failed
#: RTM_NEWADDR: Operation not permitted` (rc 1); under `danger-full-access` the
#: same command runs (rc 0) with no bwrap started at all. `verified_version`
#: stays 0.146.0: 0.154.0 is where this one value was measured, and one
#: flag's measurement is not a re-verification of the whole adapter.
#:
#: What it costs, stated once here and again in every artifact of a slot that
#: ran under it: no OS boundary of any kind between the model's shell and this
#: machine — reads and writes anywhere the user can, and network from model
#: commands (workspace-write unshares it). `agent-readonly`'s "real denial
#: mechanism" does not exist under it; a prompt-injected validator could write.
#: That is exactly the footing cursor-agent, pi, claude-code and muse (always
#: `--disable-sandbox`) run the same read-only phases on, which is why the
#: value is rendered by default (`sandbox_fallback: auto`) rather than being
#: held back until a machine installs the AppArmor repair — codex is the one
#: reviewer that insists on its own sandbox, and a fan-out that dropped it for
#: that reason alone was measured (2026-09-17) to miss 2 unique validated-high
#: issues. Still: only after the probe said `unavailable` (never `unknown`),
#: only for the two read-only profiles in `SANDBOX_FALLBACK_PROFILES`, and
#: `off` puts the refusal back.
SANDBOX_FALLBACK_VALUE = "danger-full-access"


def sandbox_fallback_policy(anchor: Optional[str] = None) -> str:
    """Resolve ``providers.codex.sandbox_fallback`` (default ``auto``).

    Same shape as ``opencode.hermetic_policy``: read through the normal layering
    (base → ``<git-root>/.multi-llm/providers.yaml`` → ``MULTI_LLM_PROVIDERS_CONFIG``)
    via the shared ``_provider_block``, an unrecognised value warns once naming
    the key and falls back to the default. YAML 1.1 resolves a bare ``off`` /
    ``on`` (``no``/``yes``, ``false``/``true``) to a boolean, so a user who
    wrote either word gets what the word says — ``False`` is ``off``, ``True``
    is ``auto`` — rather than a warning about their YAML loader. Only ``off``
    narrows anything, so no spelling of "on" needs guarding: the key's whole
    job is to let a machine turn the default fallback *off*.
    """
    value = _provider_block("codex", anchor).get("sandbox_fallback")
    if isinstance(value, bool):
        return SANDBOX_FALLBACK_AUTO if value else SANDBOX_FALLBACK_OFF
    return _resolve_policy(
        value,
        SANDBOX_FALLBACK_POLICIES,
        SANDBOX_FALLBACK_AUTO,
        "providers.codex.sandbox_fallback",
    )


def _sandbox_for(profile: str) -> str:
    """The `--sandbox` value codex runs ``profile`` under (M5).

    Unknown profiles do not reach here: :meth:`CodexProvider.build_agent_command`
    hands anything outside :attr:`~CodexProvider.agent_profiles` to the base
    class, which raises ``ProfileUnsupported`` naming the adapter and the
    declared set. The raise below is a structural backstop for a *future* call
    site rather than a reachable path — what it rules out is this function ever
    answering an unrecognised profile with a default sandbox.
    """
    try:
        return _SANDBOX_BY_PROFILE[profile]
    except (KeyError, TypeError):
        raise ValueError(
            f"codex has no sandbox policy for profile {profile!r} "
            f"(declared: {', '.join(_SANDBOX_BY_PROFILE)})"
        ) from None


#: The invariant head of every `codex exec` this adapter builds.
#:
#: `--ephemeral` and `--skip-git-repo-check` are emitted **unconditionally** —
#: no version gate, no floor raise (M2). Both were re-verified functionally, not
#: merely in help text, at the declared `min_version` 0.144.0 as well as at
#: 0.146.0, so there is no release in the supported range that would need gating
#: and a runtime probe would only add a failure mode. That claim is no longer
#: only prose: M14 turned it into the declaration
#: `CodexProvider.flags_verified_at_floor`, which the shared gate reads, so
#: "exempt" is a *checked* statement about these flags rather than a paragraph
#: beside them.
#:
#: * `--ephemeral` suppresses the rollout file every call otherwise writes into
#:   `$CODEX_HOME/sessions`. multi-llm never resumes one, and a four-model,
#:   ten-phase run leaves forty of them behind. Measured: three stub-backed runs
#:   in one git dir take the session count 1 → 2 (without) → 2 (with), with
#:   identical `agent_message` output — persistence is suppressed, the work is
#:   not.
#: * `--skip-git-repo-check` is misnamed by its own error text: the gate it
#:   opens announces itself as "Not inside a trusted directory and
#:   --skip-git-repo-check was not specified", i.e. it is codex's *trust* gate.
#:   Without it, codex outside a git repo exits 1 having made ZERO requests —
#:   while `--ask` is perfectly usable there. That contrast is the measured
#:   discriminator for this flag.
#:
#: Both are boolean, so neither may be the last element before the prompt: a
#: valueless flag in final position would take the prompt as its value, and
#: `build_command_for_transport`'s stdin leg (which drops `cmd[-1]`) would then
#: drop the wrong element. Keeping them inside this head makes that structural.
#:
#: The sandbox value is `_sandbox_for(PROFILE_GENERATE)` rather than a literal:
#: this head is what `build_command` — the `generate` path — renders, so this is
#: that profile's call site into `_SANDBOX_BY_PROFILE` (M5). `--sandbox` keeps
#: its fixed offset either way, which is what the agent legs' positional rewrite
#: relies on.
_BASE_ARGV = (
    "codex", "exec", "--sandbox", _sandbox_for(PROFILE_GENERATE), "--json",
    "--ephemeral", "--skip-git-repo-check",
)

#: §4.2's contract asymmetry, encoded from day one: claude takes the schema
#: **inline**, codex takes a **file path**. No §5.4.1 probe — this is the
#: exemption, not an omission: `--output-schema <FILE>` is documented by
#: `codex exec --help` at the declared floor 0.144.0 and was re-verified
#: functionally, which is exactly §5.4.1's "verified at the adapter's declared
#: min_version" clause (the same reason `--ephemeral` carries none). It is
#: declared as such in `flags_verified_at_floor` below.
OUTPUT_SCHEMA_FLAG = "--output-schema"

# ─────────────────────────────────────────────────────────────────────────────
# §7.6(b) — the retry envelope, measured 2026-08-16.
#
# codex's two internal retry knobs are **config keys, not env vars**, and they
# compose multiplicatively: requests = (stream_max_retries + 1) ×
# (request_max_retries + 1) = 5 × 6 = 30. Both are **unreachable for the
# provider multi-llm uses** (§7.3): `-c request_max_retries=…` is an `unknown
# configuration field`, and `-c model_providers.openai.…` is refused with
# "model_providers contains reserved built-in provider IDs: `openai`. Built-in
# providers cannot be overridden." The env names both donor plans proposed
# (`OPENAI_MAX_RETRIES`, `OPENAI_TIMEOUT`) are absent from the binary and
# measurably inert — 30 requests in every configuration, with or without them.
#
# So this row is a declaration the harness cannot change, which is exactly why
# it has to be *declared*: codex stays on the harness-side ladder, and §7.5's
# arithmetic needs the number it cannot configure away.
# ─────────────────────────────────────────────────────────────────────────────

#: Defaults, verified against codex-cli 0.146.0. Named rather than inlined so
#: the 30 below is visibly the product and not a magic constant.
CODEX_STREAM_MAX_RETRIES = 4
CODEX_REQUEST_MAX_RETRIES = 5


class CodexProvider(LLMProvider):
    """Provider for OpenAI Codex CLI tool.

    Codex outputs NDJSON (newline-delimited JSON) events when using --json flag.
    Uses 'codex exec --sandbox workspace-write --json' for non-interactive
    execution. The sandbox mode must be passed explicitly: codex's no-flag
    default is trust-dependent (read-only in a repo the user has not marked
    trusted), and this skill runs codex in arbitrary repositories.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``gpt-5.5:high``), translated to ``-c model_reasoning_effort=<effort>``.
    Valid efforts are listed in REASONING_EFFORTS, plus the inbound-only
    aliases in EFFORT_ALIASES; anything else passes through verbatim as the
    model name.

    Every invocation also carries ``--ephemeral`` and ``--skip-git-repo-check``
    unconditionally — see ``_BASE_ARGV`` for what each one buys and why neither
    is version-gated.
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
    # D6 — the review fan-out asks codex for the array on stdout, never for a
    # file. `codex exec` writes through its own bubblewrap sandbox (`--sandbox
    # workspace-write`), and on a stock Ubuntu 24.04 with
    # `kernel.apparmor_restrict_unprivileged_userns=1` bwrap cannot start at
    # all (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`).
    # Measured twice on `codex:gpt-6-astra:xhigh`, 2026-09-10: seven minutes of
    # review work each time, three `Failed to write file` attempts through the
    # file tool, and a final message carrying no JSON — with `--ephemeral`
    # keeping no rollout to salvage it from. The review only ever needed the
    # array as the final `agent_message`, which `parse_output` already returns
    # and `invoke_with_file_output` already reads (`source: stdout_fallback`);
    # `save_model_result` writes the same per-model file from it either way.
    # `supports_file_edits` stays True — the agent-write profile really edits,
    # and D6 step 3's sandbox probe is what gates it — and
    # `structured_output_transport` stays "file": that is the `--output-schema`
    # sidecar the CLI reads, not a file the model writes.
    #
    # None of this makes the fan-out sandbox-free. The child still READS the
    # plan through the same bwrap, so a sandbox that cannot start ends the slot
    # before the model has an input — which is why `sandbox_probe_command`
    # answers for the `generate` profile too.
    supports_file_output = False
    supports_workspace_reads = True
    usage_support = USAGE_FULL
    # §4.2.2: this adapter really emits `--output-schema` when handed a
    # materialized schema (see `build_command`), so the caller may mark an
    # accepted result `native`.
    structured_output_support = True
    # …and it takes a PATH, not inline JSON — §4.2's contract asymmetry, stated
    # as data so the caller materializes the sidecar without knowing the name
    # "codex".
    structured_output_transport = "file"
    # `codex login status` prints the account it is logged in as, or "Not logged
    # in"; it generates nothing and contacts no model, which is what makes it
    # cheap enough for the once-per-run preflight probe (D3c). The credential
    # store it reads is ~/.codex/auth.json (or CODEX_API_KEY / OPENAI_API_KEY).
    auth_check = ("codex", "login", "status")
    login_cmd = "codex login"
    version_cmd = ("codex", "--version")
    # D6 — `codex exec --sandbox workspace-write` writes through bubblewrap, and
    # bubblewrap needs a user namespace a stock Ubuntu 24.04 denies it. Read by
    # `multi_llm.py doctor` to pick the bwrap self-test; the dispatch-time gate
    # is `sandbox_probe_command` below, which asks codex's own runner instead.
    sandbox_mechanism = "bubblewrap"
    # M14/§5.4 — the floor names the FEATURE, not a bare number: 0.144.0 is the
    # oldest build measured to emit the `item.completed`/`agent_message` JSONL
    # shape `parse_output` reads AND to accept the three flags below at the same
    # time. Both halves matter — a build that answers prompts but streams the
    # older event names produces a clean run this adapter cannot parse.
    min_version = "0.144.0"
    # No ceiling: no codex release has been measured to break this parser, and a
    # guessed `max_version_exclusive` would refuse runs nobody showed to be
    # broken. `verified_version` carries the "newer than the capture" *warning*,
    # which is the honest posture for an unmeasured future build.
    verified_version = "0.146.0"

    # M14/§5.4.1(a) — the exemption, as data. Each of these was re-verified
    # FUNCTIONALLY (not merely found in help text) at the declared floor above
    # as well as at 0.146.0, which is the one clause that lets a flag ship with
    # no runtime gate: there is no release in the supported range that would
    # need one, and a probe would only add a failure mode. See `_BASE_ARGV` for
    # what each one buys and how it was measured.
    flags_verified_at_floor = (
        "--ephemeral",
        "--skip-git-repo-check",
        OUTPUT_SCHEMA_FLAG,
    )

    # §7.2 — the per-adapter pattern overrides. Before this, *zero* adapters
    # extended the base English lists, including the four with structured error
    # events; codex's own failure text is one of the ones the base list misses.
    #
    # Both tuples EXTEND the base set rather than replacing it: the base entries
    # are audited-clean and shared, and dropping them here would quietly make
    # this adapter blinder than the default it started from.
    #
    # Measured against codex-cli 0.146.0 on 2026-08-16 with an invalid
    # OPENAI_API_KEY: the run ends on a `turn.failed` record whose
    # `error.message` reads `unexpected status 401 Unauthorized: Missing …
    # authentication in header, url: https://api.openai.com/v1/responses`.
    # "unauthorized" alone would have matched, but only because that same
    # sentence happens to carry "status" inside the context window — these two
    # phrases pin the shape itself. (The message's middle names the two HTTP
    # auth schemes; it is quoted only in prose here, because spelling it in
    # source trips `write_safety`'s own credential-shaped-token diagnostic.)
    auth_error_patterns = LLMProvider.auth_error_patterns + (
        "unexpected status 401",
        "authentication in header",
    )

    # §7.2's measured note: codex never retries a 429 internally — one request,
    # then this exact terminal. It carries no word the base list recognises
    # ("429" is there, but this is the sentence a user sees and the one §7.5
    # rule 6(b) reasons about), so the adapter declares it.
    rate_limit_patterns = LLMProvider.rate_limit_patterns + (
        "exceeded retry limit, last status: 429",
    )

    # §7.6(b). A constant, not a property: nothing about this envelope moves with
    # the resolved slot timeout — codex takes no timeout flag and this adapter
    # sets no env at all, so `configurable_by` is empty *from the harness's side*
    # (see the module comment for the two config keys and why they are
    # unreachable). `retries_rate_limit=False` is the measured negative: one
    # request, then `exceeded retry limit, last status: 429 Too Many Requests` —
    # so the harness's own backoff is the only one a 429 gets here, and §7.5 rule
    # 6(b) must NOT suppress it for this adapter. The ~25 s wall is what 30
    # requests really cost on 5xx/stream failures inside the one awaited
    # subprocess.
    internal_retries = InternalRetryEnvelope(
        max_requests=(CODEX_STREAM_MAX_RETRIES + 1) * (CODEX_REQUEST_MAX_RETRIES + 1),
        max_wall_s=25.0,
        retries_rate_limit=False,
        configurable_by=(),
    )

    @property
    def name(self) -> str:
        return "codex"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    @staticmethod
    def _schema_argv(schema: Optional[StructuredSchema]) -> List[str]:
        """``[--output-schema, <path>]`` for a materialized schema, or ``[]``.

        **The file is the caller's, and its absence fails before spawn** (§4.2.1,
        sixth acceptance case). Adapters are module-level singletons shared
        across a concurrent fan-out, so this one cannot write a per-run temp file
        without becoming a data race — ``utils.structured_output.schema_sidecar``
        is the writer, and it removes the directory in a ``finally``.

        Raising here rather than dropping the flag is deliberate and is the
        opposite posture from §5.4.1's *optional-flag* gate, for the reason that
        gate states: a gate declines toward the status quo when the **CLI** may
        not understand a flag, whereas this is our own caller failing to produce
        a file it said it would. Silently dropping it would run the phase
        unvalidated while the caller recorded `native`.
        """
        if schema is None:
            return []
        if not schema.path:
            raise StructuredOutputUnavailable(
                f"codex takes {OUTPUT_SCHEMA_FLAG} as a FILE, but {schema.name} "
                f"was handed over with no materialized path. Wrap the dispatch "
                f"in utils.structured_output.schema_sidecar(). Nothing was "
                f"dispatched.",
                provider="codex",
                schema=schema.name,
            )
        if not os.path.isfile(schema.path):
            raise StructuredOutputUnavailable(
                f"codex's {OUTPUT_SCHEMA_FLAG} file for {schema.name} is not "
                f"readable at {schema.path}. Nothing was dispatched.",
                provider="codex",
                schema=schema.name,
            )
        return [OUTPUT_SCHEMA_FLAG, schema.path]

    def sandbox_probe_command(self, profile: str) -> Optional[List[str]]:
        """codex's own runner, asked to start where a dispatch would (D6).

        ``codex sandbox -c 'sandbox_mode="<policy>"' -- <cmd>`` runs ``<cmd>``
        under exactly the bubblewrap policy ``codex exec --sandbox <policy>``
        would use, with no model and no prompt — ~30 ms measured, and on a
        stock Ubuntu 24.04 it dies on ``bwrap: loopback: Failed RTM_NEWADDR:
        Operation not permitted`` before the shell starts.

        **Every** profile this adapter declares answers, because every one of
        them is bubblewrap. The two ``workspace-write`` profiles —
        ``agent-write`` and ``generate``, the review fan-out's — write and
        delete one byte under ``.multi-llm/tmp`` of the current directory (the
        executor and the fan-out both run this with ``cwd`` at the repo root),
        and the shell leaves nothing behind it created. ``generate`` used to
        return ``None`` on the premise that "the review fan-out stopped needing
        the sandbox when ``supports_file_output`` went ``False``". That premise
        is wrong: the fan-out stopped needing the sandbox to *write*, and a
        bwrap that cannot start denies the child every file operation —
        including the READ of the plan file, which is the whole input to the
        phase. Measured 2026-09-16, `codex:gpt-5.6-terra` on a review-plan
        fan-out: 27.9 s, 61k input tokens, and a one-item "review" whose only
        suggestion was that the plan file could not be read — scored a success,
        because nothing had asked the sandbox whether it could start.

        ``agent-readonly`` runs the same probe with **no** write in it: an
        empty command under ``sandbox_mode="read-only"``, which is enough,
        because what fails on this machine is bwrap *starting* and nothing
        inside the sandbox gets to run at all. It used to answer ``None`` on
        the premise that "this probe writes a byte, and a read-only dispatch is
        never asked to" — true of the byte, irrelevant to the failure: every
        read kind (validation, revalidation, consolidation, salvage, task
        generation, the human-decision judge) dispatched codex unprobed, and
        the bwrap failure then surfaces inside the model's own tool results
        rather than on the stream, so the D3k classifier never sees ``bwrap:``
        and the JSON the model still returns is scored a valid batch — the same
        "the file could not be read" shape the ``generate`` row above fixed.

        The policy comes from ``_SANDBOX_BY_PROFILE`` so the probe cannot answer
        for a policy the dispatch does not use (M5).
        """
        try:
            policy = _sandbox_for(profile)
        except ValueError:
            return None
        if profile == PROFILE_AGENT_READONLY:
            # Nothing to clean up, so nothing to clean up *as*: the question is
            # only whether the sandbox comes up.
            script = "exit 0"
        else:
            script = (
                'd=.multi-llm/tmp; f="$d/sandbox-probe-$$"; '
                'mkdir -p "$d" && : > "$f" && rm -f "$f"; rc=$?; '
                'rmdir "$d" .multi-llm 2>/dev/null; exit $rc'
            )
        return [
            "codex", "sandbox", "-c", f'sandbox_mode="{policy}"',
            "--", "sh", "-c", script,
        ]

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # --sandbox workspace-write, not the older --full-auto: 0.146.0 keeps
        # --full-auto only as a hidden compatibility trap (internally
        # `removed_full_auto`) and warns on every run. The two produce an
        # identical effective policy — approval_policy is forced to "never"
        # in exec mode regardless — so this is a rename, not a permission
        # change.
        #
        # §4.2.2's splice point: the schema pair goes in the same policy region
        # as claude's, immediately after `_BASE_ARGV` and ahead of `--model`, so
        # it is never the last pair before the prompt and
        # `build_command_for_transport`'s stdin leg (which drops `cmd[-1]`) can
        # never drop half of it. `--sandbox` keeps its fixed offset inside
        # `_BASE_ARGV`, which is what `build_agent_command`'s positional rewrite
        # relies on.
        schema_argv = self._schema_argv(schema)
        base_model, effort = split_reasoning_effort(model, _ACCEPTED_EFFORTS)
        if effort is not None:
            return [
                *_BASE_ARGV,
                *schema_argv,
                "--model", base_model,
                "-c", f"model_reasoning_effort={EFFORT_ALIASES.get(effort, effort)}",
                prompt,
            ]
        return [*_BASE_ARGV, *schema_argv, "--model", model, prompt]

    def build_command_for_transport(
        self,
        prompt: str,
        model: str,
        transport: str = "argv",
        prompt_file: Optional[str] = None,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
        sandbox_fallback: bool = False,
    ) -> Tuple[List[str], Optional[str]]:
        """Deliver the prompt on argv or on stdin (prompt omitted from argv).

        ``sandbox_fallback`` is applied on the stdin leg here and on the argv leg
        by the base class — both through :meth:`apply_sandbox_fallback`, and
        both **before** the prompt is dropped, so the positional rewrite sees
        the argv exactly as :meth:`build_command` laid it out.
        """
        if transport != "stdin":
            return super().build_command_for_transport(
                prompt, model, transport, prompt_file, timeout_s=timeout_s,
                schema=schema, sandbox_fallback=sandbox_fallback,
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        if sandbox_fallback:
            cmd = self.apply_sandbox_fallback(cmd, PROFILE_GENERATE)
        return cmd[:-1], prompt

    def sandbox_fallback_policy(
        self, profile: str, anchor: Optional[str] = None
    ) -> Optional[str]:
        """``providers.codex.sandbox_fallback`` for a read-only profile; else ``None``.

        The profile gate is answered *here*, in the adapter, and not only at
        the call sites: ``agent-write`` comes back ``None`` whatever the config
        says, so no gate that consults this hook can be talked into widening a
        write dispatch by a config line. The two read-only profiles return the
        resolved policy (``auto`` by default), which the gates then require to
        be ``auto`` — beside the probe verdict ``unavailable`` — and ``off`` is
        the one word that puts the refusal back.
        """
        if profile not in SANDBOX_FALLBACK_PROFILES:
            return None
        return sandbox_fallback_policy(anchor)

    def apply_sandbox_fallback(self, argv: List[str], profile: str) -> List[str]:
        """Point ``argv``'s ``--sandbox`` slot at ``danger-full-access`` (D6 override).

        Structural, not policy: a profile outside ``SANDBOX_FALLBACK_PROFILES``
        raises rather than rendering, so the write profile cannot reach this
        value even from a caller that got every other gate wrong.
        """
        if profile not in SANDBOX_FALLBACK_PROFILES:
            raise ValueError(
                f"codex never runs the {profile!r} profile unsandboxed (the "
                f"fallback covers: {', '.join(SANDBOX_FALLBACK_PROFILES)})"
            )
        return self._apply_sandbox(argv, profile, fallback=True)

    def build_agent_command(
        self,
        prompt: str,
        model: str,
        profile: str = PROFILE_AGENT_READONLY,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build ``codex exec`` under an explicit sandbox policy (D3e).

        Both profiles start from :meth:`build_command` and then point the
        ``--sandbox`` slot at their **own** row of ``_SANDBOX_BY_PROFILE`` (M5).
        Neither leg inherits the other's policy by accident, and neither restates
        a policy string of its own — which is what makes the two profiles argv-
        distinguishable by construction rather than by the coincidence that they
        happen to have been written with different literals.

        ``agent-readonly`` renders ``--sandbox read-only``: codex refuses
        model-generated writes itself, so a prompt-injected validator cannot edit
        the repository while pretending to score suggestions.

        ``agent-write`` renders ``--sandbox workspace-write`` — codex's own
        documented write policy, rooted at the process ``cwd``, which the executor
        pins to the resolved repo root (D3h). It is deliberately **not**
        ``--dangerously-bypass-approvals-and-sandbox`` (nor the older
        ``--yolo``): a runner with no honored write sandbox is refused for
        ``agent-write``, so fabricating a bypass here would delete the very
        boundary that makes the profile eligible. Network access is left at
        codex's own workspace-write default rather than being widened here.

        **The one override** (D6, 2026-09-17): nothing this method renders is
        ever a bypass, and the default argv of every profile stays that way —
        but a *dispatch-time* rewrite exists, :meth:`apply_sandbox_fallback`,
        which points the same ``--sandbox`` slot at ``danger-full-access`` for
        ``generate`` and ``agent-readonly`` only. It is reachable through
        exactly one gate, applied by ``sandbox_refusal`` (fan-out) and
        ``_detect_sandbox`` (executor): the D6 probe verdict is ``unavailable``
        (bubblewrap cannot start here; never ``unknown``), the profile is one
        of the two read-only ones, and ``providers.codex.sandbox_fallback`` is
        ``auto`` — which it is unless a config layer wrote ``off``. So on a
        stock Ubuntu 24.04 this is what a read-only codex dispatch runs as by
        default: the read-only profile's own denial mechanism is gone — the
        model's shell has the operator's rights, as it does for every reviewer
        with no OS sandbox of its own — and every artifact of such a slot says
        so in so many words. ``agent-write`` is refused under a broken sandbox
        exactly as before, whatever the key says; the override raises on it
        rather than rendering.
        """
        if profile not in (PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
            return super().build_agent_command(
                prompt, model, profile, timeout_s=timeout_s, schema=schema
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        return self._apply_sandbox(cmd, profile)

    @staticmethod
    def _apply_sandbox(
        cmd: List[str], profile: str, *, fallback: bool = False
    ) -> List[str]:
        """Point ``cmd``'s ``--sandbox`` value at ``profile``'s policy (M5).

        The rewrite is *positional* — the element right after the first
        ``--sandbox`` — never a sweep of argv for the outgoing value. The prompt
        is the final element and a model string is arbitrary text, so a value
        sweep would silently rewrite either one if it happened to equal
        ``workspace-write``, leaving a stray ``read-only`` positional that codex
        would take as the prompt (and that
        ``base.build_agent_command_for_transport`` could no longer recognise at
        ``argv[-1]``). ``cmd`` is mutated in place and returned; every caller
        hands over a list :meth:`build_command` just built, so nothing shared is
        touched.

        ``fallback=True`` writes :data:`SANDBOX_FALLBACK_VALUE` into the slot
        instead of the profile's row (D6 override, 2026-09-17) — keyword-only,
        default ``False``, so every existing caller renders the row it always
        did. ``_sandbox_for`` still runs first: an unknown profile raises the
        same way on both paths, and a profile outside
        ``SANDBOX_FALLBACK_PROFILES`` is refused here too, one level below
        :meth:`apply_sandbox_fallback`, because two callers rendering the
        value is one too many places for a write profile to slip through.
        """
        sandbox_at = cmd.index("--sandbox")
        policy = _sandbox_for(profile)
        if fallback:
            if profile not in SANDBOX_FALLBACK_PROFILES:
                raise ValueError(
                    f"codex never runs the {profile!r} profile unsandboxed"
                )
            policy = SANDBOX_FALLBACK_VALUE
        cmd[sandbox_at + 1] = policy
        return cmd

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
