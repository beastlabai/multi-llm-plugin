"""Grok Build CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import (
    REASONING_LADDER,
    LLMProvider,
    ReasoningLadder,
    StructuredSchema,
    split_reasoning_effort,
)

# Reasoning effort values accepted by `grok --reasoning-effort`
# (verified live against grok 0.2.93) — exactly the seven canonical rungs, which
# is why this declaration names the ladder itself rather than restating it.
EFFORT_LADDER = ReasoningLadder(
    provider="grok",
    rungs=REASONING_LADDER,
)
REASONING_EFFORTS = EFFORT_LADDER.accepted


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
    (``--network none``, no credential), and re-measured by the same rig on
    2026-10-08 against ``grok 1.0.46`` (what moved is listed at
    :attr:`verified_version`); the raw captures are
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

    **Three requests per exec at 0.2.93, five per answered exec at 1.0.46.**
    ``GET /v1/models`` (the catalog grok validates ``-m`` against, before
    dispatching anything), ``GET /`` (a base-URL reachability probe, new in
    1.0.x), a concurrent ``POST /v1/responses`` session-title call with the
    **hardcoded** model ``grok-4.6`` (``grok-build`` at 0.2.93) — which also
    carries tools, so only the model id separates it — the real ``POST
    /v1/chat/completions``, and since 1.0.x a tool-less second ``POST
    /v1/chat/completions`` on the ``-m`` model asking for a one-line summary of
    the reply (a failed turn makes none).

    **The internal-retry envelope is two-tiered, and the second tier is a silent
    hang.** 401, 404 and 400 are terminal at one chat request; a 429 gets exactly
    one retry (2 requests, 2.2 s at 0.2.93, 5.2 s at 1.0.46); at 0.2.93 every
    transport-shaped failure gets **15 chat requests over 335 s** with a ~30 s
    backoff cap, writing **zero bytes to both channels** for the whole of it. At
    1.0.46 a cut stream, an error object on a 200 and an empty answer still do
    (326 s), but a 5xx and a zero-byte response run **four such ladders back to
    back: 60 chat requests over ~1,340 s**, silent throughout. So
    :attr:`default_timeout` (600 s) covered the 0.2.93 window and does **not**
    cover the 1.0.46 5xx one: a fail-forever 5xx is now killed by the harness at
    600 s with zero bytes on both channels — the ``opencode-run-silent-hangs``
    shape, which any shorter slot already produced. No :attr:`internal_retries`
    envelope is declared yet; the measurement for one is in the fixture
    directory.

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
    #: D6 step 6 — an open question, recorded as such. `grok inspect` reports
    #: `Project trusted: no` for an untrusted repo and `build_command` passes
    #: `--always-approve` with no trust flag; whether that gate blocks a
    #: headless write is unmeasured. `multi_llm.py doctor` prints this verbatim
    #: until a measurement either way lands under Phase 5.
    trust_status = "untested"
    #: 0.2.93 is where every fixture in ``tests/fixtures/grok/`` was first
    #: captured — the §6 step-5 probe, ``--network none``, the stub and the
    #: failserver both running inside the container, no credential — and, until
    #: the 1.0.46 re-capture below, the only release this parser had been run
    #: against. A pin to re-confirm rather than a range to
    #: trust: nothing older has been measured, and the two shapes the parser
    #: depends on (``{"text": …}`` for an answer, ``{"type": "error", …}`` for a
    #: failure) are both vendor-internal envelopes.
    #: ``tests/container/cli-versions.lock`` pins the same string.
    #:
    #: ``grok --version`` prints ``grok 1.0.46 (<commit>)`` on stdout, whose
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
    #:
    #: 1.0.46 since 2026-10-08 (run 10, #10): every row re-captured on it by the
    #: same rig in the pinned image (cli-versions.lock moved to 1.0.46 the same
    #: day). Every parser verdict and classification is unchanged and the parser
    #: needed no change; the three ``no_visible_content`` rows and the pre-flight
    #: row are byte-identical. What moved: five requests per answered exec (a
    #: root probe and a reply-summary call on the ``-m`` model), the title model
    #: ``grok-4.6``, ``stopReason`` ``end_turn``, a 429 passing the vendor's own
    #: words (with the wait), no ``Request URL`` / response-header lines, and a
    #: 5xx or zero-byte response retried for ~1,340 s instead of 335 s.
    verified_version = "1.0.46"

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

    def salvage_text(self, stdout: str) -> Optional[str]:
        """The wrapper's ``text``, never the wrapper or grok's error body (D2.5).

        The text :meth:`parse_output` unwraps first: ``wrapper["text"]``,
        stripped, when stdout is one JSON object whose ``type`` is not
        ``"error"``. So the ask lane's answer text and the review lanes'
        salvage read what the parser treats as the answer
        (plans/ask-answer-recovery-and-false-success.md D2.5), and grok's
        ``{"type":"error","message":…}`` body — every one of the ten measured
        failures, and the 2026-09-22 body the ask lane once saved as an answer
        — gives ``""``, a final verdict. So does anything that is not one whole
        wrapper: ``generate`` always renders ``--output-format json``.

        ``text`` joins every turn of the run, so an answer carries any earlier
        narration with it; nothing in the output separates the two (§14).
        """
        try:
            if not isinstance(stdout, str):
                return ""
            try:
                wrapper = json.loads(stdout.strip())
            except ValueError:
                return ""
            if not isinstance(wrapper, dict) or wrapper.get("type") == "error":
                return ""
            text = wrapper.get("text")
            return text.strip() if isinstance(text, str) else ""
        except Exception:  # noqa: BLE001 - the hook must never raise
            return ""
