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

    Everything below was **measured** by the §6 step-5 probe on 2026-08-19
    against ``grok 0.2.93`` driven offline at a localhost stub
    (``--network none``, no credential); the raw captures are
    ``tests/fixtures/grok/`` and its README carries the full record.

    **Every failed run parses as a SUCCESS.** grok reports a failure as a
    well-formed JSON object on stdout, ``{"type": "error", "message": …}``. It
    carries no ``"text"`` key, so :meth:`parse_output` reaches its final
    ``return {"success": True, "data": wrapper}`` and hands the error envelope
    back as the model's answer — measured on ten distinct failures (401, 429,
    404, 400 context-length, 5xx-forever, a cut stream, a zero-byte response, a
    200 carrying an error object, an empty assistant message, and grok's own
    pre-flight model rejection). The **nonzero exit code is the only thing** that
    distinguishes a failed grok run from an answered one today, which is what
    makes the measured "every failure exits 1" below load-bearing rather than
    incidental. The fix is one branch — an object with ``message`` and no
    ``text`` is a failure — and it is recorded here rather than applied, because
    it flips every failure row in the committed manifest.

    **Three requests per exec, not one.** ``GET /v1/models`` (the catalog grok
    validates ``-m`` against, before dispatching anything), a concurrent
    ``POST /v1/responses`` session-title call with the **hardcoded** model
    ``grok-build`` — which also carries tools, so only the model id separates it
    — and then the real ``POST /v1/chat/completions``.

    **The internal-retry envelope is two-tiered, and the second tier is a silent
    hang.** 401, 404 and 400 are terminal at one chat request; a 429 gets exactly
    one retry (2 requests, 2.2 s); every transport-shaped failure gets **15 chat
    requests over 335 s** with a ~30 s backoff cap, writing **zero bytes to both
    channels** for the whole of it. :attr:`default_timeout` (600 s) is above that
    window, but any shorter slot turns a CLI that would have reported into the
    ``opencode-run-silent-hangs`` shape. No :attr:`internal_retries` envelope is
    declared yet; the measurement for one is in the fixture directory.

    **An upstream 404 classifies as ``server_error``, not ``invalid_model``**:
    grok wraps every HTTP failure in ``API error (status <n> <reason>)`` and the
    server table is consulted before the invalid-model table. Its *own* pre-flight
    rejection (``unknown model id``) classifies correctly, because that message
    carries no such wrapper — both shapes are committed as separate fixture rows.
    """

    #: §9.4: the prompt trails. It used to sit at index 4 — the one adapter that
    #: rendered a flag *after* it — which made `cmd[:-1]` and
    #: `build_agent_command_for_transport` silently wrong the moment grok gained
    #: a stdin transport or an agent profile. It has neither today, so
    #: normalizing the order changes no dispatched behaviour: `-p` takes exactly
    #: one value and the flags are order-independent. Note the pair: a future
    #: non-argv transport must drop `-p` **and** the prompt, not `cmd[:-1]`.
    prompt_argv_index = -1

    version_cmd = ("grok", "--version")
    #: 0.2.93 is where every fixture in ``tests/fixtures/grok/`` was captured —
    #: the §6 step-5 probe, ``--network none``, the stub and the failserver both
    #: running inside the container, no credential — and the only release this
    #: parser has been run against. A pin to re-confirm rather than a range to
    #: trust: nothing older has been measured, and the two shapes the parser
    #: depends on (``{"text": …}`` for an answer, ``{"type": "error", …}`` for a
    #: failure) are both vendor-internal envelopes.
    #: ``tests/container/cli-versions.lock`` pins the same string.
    #:
    #: ``grok --version`` prints ``grok 0.2.93 (<commit>)`` on stdout, whose
    #: first dotted token the base :meth:`parse_version` reads without an
    #: override.
    #: NO FLOOR, deliberately. `min_version` was set to 0.2.93 on 2026-08-19 — the
    #: version the §16 fixtures were captured against — and removed the same day
    #: for the reason §5.4.1(a) of harness-invocation-improvements.md states: a
    #: MEASURED-AT version is an upper bound on the true floor, never the floor.
    #: This is a PRE-EXISTING adapter, so a floor pinned to the release that
    #: happened to be in the image newly refuses installs that worked yesterday,
    #: with exit 3 and no evidence that 0.2.93 is where anything actually broke.
    #: (copilot/kimi/qwen-code/vibe keep theirs because they landed *together
    #: with* their probes; muse, cursor-agent and agy leave it empty for this
    #: same reason.) `verified_version` below is warning-only and carries the
    #: measurement instead.
    verified_version = "0.2.93"

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
            # GROK'S ERROR ENVELOPE IS A DICT WITHOUT `text`, and the catch-all
            # below used to hand it back as the ANSWER. Measured 2026-08-19 on the
            # fixture capture: ten distinct injected failures — 401, 404, 400
            # context-length, 500, a cut stream, a zero-byte body, an HTTP-200
            # error document — every one of them arrives as
            # `{"type":"error","message":"Internal error: …"}`, and every one of
            # them returned `success: True` with that envelope as `data`. Only the
            # nonzero exit code stood between a fan-out slot and an error string
            # recorded as a model's review. `type == "error"` is grok's own
            # discriminator; `message` carries the upstream text the classifier
            # then reads.
            if isinstance(wrapper, dict) and wrapper.get("type") == "error":
                message = wrapper.get("message")
                if not isinstance(message, str) or not message.strip():
                    message = json.dumps(wrapper)
                return {"success": False, "error": message.strip(),
                        "raw": stdout, "data": None}
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}
