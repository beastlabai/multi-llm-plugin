"""Google Antigravity CLI (agy) provider implementation with JSON envelope parsing."""
import json
import shutil
import sys
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema, split_reasoning_effort

# Reasoning effort values accepted by `agy --effort` (read from `agy --help` on
# agy 1.1.13, offline and unauthenticated):
#
#   --effort   Reasoning effort for the current CLI session (low|medium|high)
#
# The adapter and providers.yaml both used to state that agy had no `:effort`
# suffix support, which was simply false — so every effort a user configured was
# refused by *our* whitelist for a flag agy really has (L11 #3, §13.4 #3).
#
# This is a narrower ladder than most: no `none`, no `minimal`, no `xhigh`, no
# `max`. An off-list suffix is deliberately NOT coerced onto the nearest rung —
# it stays part of the model name, where agy answers with an explicit ERROR
# envelope naming the unknown model. A silent downgrade would be the worse of
# the two failures, because agy's own effort request is invisible in its output.
REASONING_EFFORTS = frozenset({"low", "medium", "high"})

#: Seconds by which agy's own print-mode cutoff is held UNDER the parent's
#: `subprocess.run(timeout=…)` deadline (§9.6.c). The margin is the point and it
#: is deliberate: agy must reach its own cutoff, write its machine-readable
#: `status: ERROR` envelope and exit *before* the parent fires, so the harness
#: reports agy's own failure reason instead of killing a live child and
#: reporting a bare TIMEOUT. 30 s covers the envelope write plus process exit at
#: the measured envelope sizes; anything under ~5 s puts the two deadlines back
#: in a tie — which is exactly what the old hardcoded `20m` did against the
#: configured 1200 s.
INNER_SHUTDOWN_MARGIN_S = 30

#: Floor for the derived cutoff, and a misconfiguration guard: a timeout at or
#: below the margin would otherwise render a non-positive duration. A sub-90 s
#: agy run is a mistake rather than an intent, so the clamp is announced with
#: exactly one WARNING naming both numbers rather than applied silently.
MIN_INNER_TIMEOUT_S = 60

#: What `--print-timeout` rendered before the resolved timeout was threaded in.
#: It is not a fourth number — `20m` IS 1200 s, the same value `providers.yaml`
#: configures and :attr:`AgyProvider.default_timeout` falls back to — but it is
#: the spelling that made the two deadlines an exact tie. It survives for one
#: reason only: `timeout_s=None` must render this adapter's pre-L5 argv
#: byte-for-byte, so dropping the single `timeout_s=` call site in
#: `llm_client.invoke_with_provider` is a complete rollback. Nothing on a live
#: path reaches it — Lane A always resolves a timeout before it builds argv.
UNTHREADED_PRINT_TIMEOUT = "20m"


def print_timeout_value(timeout_s: Optional[int]) -> str:
    """Render `--print-timeout` for an outer deadline of ``timeout_s`` (§9.6.c).

    Whole seconds, never minutes: `--print-timeout` takes a Go duration string
    and ``time.ParseDuration`` accepts the bare-seconds spelling (``1170s``),
    which removes the minute-rounding rule instead of specifying one. Rounding
    to minutes is what let 1200 s and ``20m`` become an exact tie.

    The bare-seconds form is **verified against the real binary** (agy 1.1.13,
    offline, no auth, no quota): `agy --print-timeout 570s|1170s|2370s|60s help`
    all parse, while `1170` is refused with Go's own ``missing unit in duration``
    — so the `f"{m}m{s}s"` fallback §9.6.c held in reserve is not needed.
    """
    if timeout_s is None:
        return UNTHREADED_PRINT_TIMEOUT
    outer = int(timeout_s)
    inner = max(MIN_INNER_TIMEOUT_S, outer - INNER_SHUTDOWN_MARGIN_S)
    if inner + INNER_SHUTDOWN_MARGIN_S > outer:
        print(
            f"WARNING: agy timeout of {outer}s leaves nothing above the "
            f"{INNER_SHUTDOWN_MARGIN_S}s shutdown margin; --print-timeout is "
            f"clamped to {MIN_INNER_TIMEOUT_S}s, so agy will still be running "
            f"when the {outer}s deadline fires. Raise the configured timeout "
            f"to at least {MIN_INNER_TIMEOUT_S + INNER_SHUTDOWN_MARGIN_S}s.",
            file=sys.stderr,
        )
    return f"{inner}s"


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
    no shell — spaces are safe). They may carry an optional ``:effort``
    suffix (e.g. "Gemini 3.1 Pro (High):high"), translated to
    ``--effort <level>``; valid levels are listed in REASONING_EFFORTS.

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
        # 1200, matching `providers.yaml`'s `providers.agy.default_timeout`
        # (§9.6.d). This is the fallback twin of the configured number, not a
        # second number: the provider ships in base config, so
        # `get_provider_timeout` returns the yaml value and never reaches here.
        # It read 600 while the yaml read 1200 and the argv read `20m` — three
        # numbers for one fact, which is the disagreement L5 removes.
        return 1200

    def is_available(self) -> bool:
        return shutil.which("agy") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # agy --new-project --dangerously-skip-permissions
        #   --output-format json --print-timeout <derived>
        #   --model "<Model Display Name>" -p "<prompt>"
        # --new-project binds the workspace to cwd (otherwise a previous
        # project's — possibly different — directory is silently reused);
        # --dangerously-skip-permissions auto-approves tool use so headless
        # file reads/writes work; --print-timeout raises the internal
        # print-mode cutoff (agy's own default, 5m0s, is too low for
        # review-sized runs) and is DERIVED from the caller's already-resolved
        # timeout minus INNER_SHUTDOWN_MARGIN_S — see `print_timeout_value`.
        # It used to be the literal `20m`, which made a configured timeout
        # above 20 minutes a silent no-op (agy self-terminated at 1200 s and
        # returned its own ERROR envelope, so the user diagnosed agy rather
        # than their config) and one below it a race the parent's kill won
        # against a live agy. The model display name is passed verbatim as one
        # argv element.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            print_timeout_value(timeout_s),
            "--model",
            base_model,
        ]
        # `model[:effort]` -> `--effort <level>` (L11 #3). Splitting on the LAST
        # colon is safe for agy specifically because its display names carry no
        # colon of their own — "Gemini 3.1 Pro (High)" survives intact as one
        # argv element, and a suffix that is not a level stays glued to the name
        # so agy rejects it out loud rather than running at the wrong effort.
        if effort is not None:
            cmd += ["--effort", effort]
        cmd += ["-p", prompt]
        return cmd

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
