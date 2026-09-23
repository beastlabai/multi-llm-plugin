"""Block goose CLI provider implementation with JSON envelope parsing."""
import json
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..json_extractor import extract_json_from_text
from .base import (
    InternalRetryEnvelope,
    LLMProvider,
    ReasoningLadder,
    StructuredSchema,
    split_reasoning_effort,
)

# Thinking effort values accepted by goose's ThinkingEffort parser, aliases
# included (verified on goose 1.41.0). goose has no CLI flag for this — the
# value goes via the GOOSE_THINKING_EFFORT env var / config key; invalid
# values are silently ignored by goose.
#
# `off` is goose's own spelling of the `none` rung and it accepts BOTH, so it is
# declared as a local spelling rather than as a rung: the ladder still emits
# `none` for goose, and `off` is honoured because a user (or an older config)
# may have typed it.
EFFORT_LADDER = ReasoningLadder(
    provider="goose",
    rungs=("none", "low", "medium", "high", "xhigh", "max"),
    spellings=(("off", "none"),),
)
REASONING_EFFORTS = EFFORT_LADDER.accepted


class GooseProvider(LLMProvider):
    """Provider for Block's goose CLI tool.

    goose run with --output-format json emits a single JSON envelope on
    stdout: {"messages": [...], "metadata": {...}}. Each message is
    {"role": "user"|"assistant", "content": [{"type": "text"|"thinking"|
    "toolRequest"|"toolResponse", ...}, ...]}. The model's answer is the
    concatenated "text" parts (type == "text") of the LAST message with
    role == "assistant".

    Config model names use the format "<goose-provider>/<model-id>"
    (e.g. "openrouter/z-ai/glm-5.2"): the segment before the first "/"
    maps to --provider (the goose backend provider) and the remainder to
    --model. A name without "/" is passed straight to --model and uses
    the GOOSE_PROVIDER env / configured default provider.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openrouter/z-ai/glm-5.2:high``), stripped off the full string BEFORE
    the provider/model split. goose has no CLI flag for reasoning effort,
    so get_env() maps the suffix to the GOOSE_THINKING_EFFORT env var
    instead; without a suffix the var is left unset so goose falls back to
    the user's own config. Valid efforts are listed in REASONING_EFFORTS;
    anything else passes through verbatim as the model name (keeping
    ``:free``-style ids intact).

    Gotcha: goose exits 0 EVEN ON PROVIDER ERRORS (401, rate limit).
    Failure manifests as the final assistant text containing
    "Ran into this error:" — parse_output detects this and returns an
    explicit failure result.

    Two measured limits on that marker, both from the 1.41.0 fixture
    capture (tests/fixtures/goose/manifest.yaml). (1) It does not cover
    every failure vocabulary: a context overflow reads "Ran into this error
    TRYING TO COMPACT:" and both stream failures read "Network error:", so
    on those three the branch below never fires and the run is reported as
    an extraction failure instead. (2) A failed run can carry no assistant
    message at all — an empty completion and a 200 whose body is a provider
    error object both produce a well-formed envelope with status
    "completed" and only the echoed user message in it.

    Gotcha 2, and it is the parser's own: the envelope ECHOES the user
    prompt, so anything reading raw stdout is reading our prompt too.
    prompts/plan_review.txt contains the word "timeout", which alone makes
    base.classify_streams answer `timeout` on a completely successful run
    (the `prompt-echo` fixture row pins it).
    """

    version_cmd = ("goose", "--version")
    #: 1.41.0 is where every fixture in ``tests/fixtures/goose/`` was captured —
    #: the §6 step-5 probe, ``--network none``, the stub and the failserver both
    #: running inside the container, no credential — the release
    #: ``REASONING_EFFORTS`` and the exit-0 measurement below were both verified
    #: on, and the one ``tests/container/cli-versions.lock`` pins. It is a pin to
    #: re-confirm rather than a range to trust: the parser depends on three
    #: goose-internal shapes upstream is free to restyle — the
    #: ``{"messages": [...]}`` envelope, the ``content[].type == "text"`` part
    #: names, and the ``Ran into this error:`` prefix that is the ONLY signal a
    #: run failed (goose exits 0 on all nine measured failure modes).
    #:
    #: That prefix is already known not to cover everything at 1.41.0: goose
    #: writes ``Ran into this error trying to compact:`` on a context overflow and
    #: ``Network error:`` on both stream failures, so three of the committed rows
    #: fall through to an extraction failure instead
    #: (``tests/fixtures/goose/manifest.yaml``).
    #:
    #: **There is deliberately no ``min_version``.** A draft declared
    #: ``min_version = "1.41.0"`` because that is where the capture ran, which is
    #: `harness-invocation-improvements.md` §5.4.1(a)'s error: a measured-at
    #: version is an upper bound on the true floor, never the floor itself.
    #: goose is a **pre-existing** adapter and nothing measured says 1.40 fails —
    #: and note the direction of the risk here, which a floor gets backwards.
    #: The known gaps in the failure marker are gaps *at the pinned version*, so
    #: refusing older installs buys nothing; the release that would actually hurt
    #: is a newer one that restyles the prefix, and that is precisely the case
    #: ``verified_version`` warns on. ``min_version`` is a hard exit-3 refusal in
    #: ``subagent_executor._detect_version``; ``verified_version`` is a one-line
    #: warning, and only the warning is earned.
    #:
    #: ``version_cmd`` stays, because it is the only thing that can ever make
    #: ``verified_version`` speak: with no probe the detected version is
    #: ``unknown`` and the warning is skipped. Confirmed against the pinned
    #: binary 2026-08-20 (``docker run --rm --network none --entrypoint
    #: /opt/mllm/real/goose mllm-e2e:local --version``): ``goose --version``
    #: prints the dotted version **behind a leading space** (`` 1.41.0``), which
    #: the base :meth:`parse_version` still reads — its ``\b``-anchored search is
    #: not line-anchored, unlike ``cli-versions.lock``'s own verify pattern,
    #: which has to spell the ``\s*`` out.
    verified_version = "1.41.0"

    #: §7.2 — the exit-0 sniff, folded out of ``parse_output`` and into the
    #: classifier so the D3k ladder sees it too. goose exits **0** on a provider
    #: error (measured again on 1.41.0, 2026-08-16, with an invalid
    #: OPENROUTER_API_KEY), and the only signal is this phrase in the final
    #: assistant text. ``parse_output`` still returns an explicit failure — the
    #: two readings are the same marker, not two rules.
    exit_zero_failure_markers = (re.compile(r"ran into this error:"),)

    # §7.6(b), declared 2026-08-20 (C12). Measured against the committed
    # fail-forever captures in `tests/fixtures/goose/`, never inferred:
    #
    # * `max_requests = 8` — `GET /v1/models`, then SEVEN completions: FOUR
    #   attempts at the real turn and THREE at the session-title completion
    #   goose fires even under `--no-session`. Two independent rows measure the
    #   same eight — `exit-zero-failure` (500-forever, `requests: 8`,
    #   `wall_s: 6.67`, backoff ~0.1/1.1/2.8/6.6 s) and `rate-limit`
    #   (429-forever, `requests: 8`, `wall_s: 7.22`) — and the set's own header
    #   states it as the envelope for this CLI (`README.md:24-28`, `:54`;
    #   `manifest.yaml`'s notes on both rows).
    #
    # * `max_wall_s = None`, and this is a JUDGEMENT recorded rather than a gap.
    #   The ~6.7–7.2 s those rows measure is ROUND-TRIP TIME against a localhost
    #   failserver — transport, not a ceiling goose imposes on itself — so it
    #   says nothing about what the same eight requests cost against a real
    #   provider, and a bound taken from it would be a bound on the stub. The
    #   dataclass documents `None` as the honest posture for exactly this case:
    #   rule 4 then performs no admission test, because inventing a bound "would
    #   refuse attempts that had budget for themselves" (`base.py`'s
    #   `max_wall_s`). `codex` declares a stub-measured 25.0 s and is not the
    #   precedent here: codex's number is its own retry policy playing out, and
    #   goose's is the wire.
    #
    # * `retries_rate_limit = True` — the `rate-limit` row is the 5xx row's
    #   envelope: goose retries a 429 itself, four attempts at the real turn,
    #   8 requests over 7.2 s. **The budget consequence runs the way that
    #   surprises people**: declaring this REMOVES the harness's own rate-limit
    #   backoff for goose (§7.5 rule 6(b), `STOP_VENDOR_ALREADY_RETRIED`). A
    #   rate-limited goose slot used to get up to `MAX_RATE_LIMIT_BACKOFFS`
    #   further identical dispatches — 8 requests turned into 32 — and now gets
    #   none. It is a smaller retry budget, not a larger one.
    #
    # `configurable_by` is empty: `get_env` below sets only
    # `GOOSE_THINKING_EFFORT`, which changes how the model thinks and not how
    # many times the CLI retries, and `build_command` passes no retry knob.
    internal_retries = InternalRetryEnvelope(
        max_requests=8,
        max_wall_s=None,
        retries_rate_limit=True,
        configurable_by=(),
    )

    @property
    def name(self) -> str:
        return "goose"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("goose") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # goose run --no-session -q --output-format json --no-profile
        #   --max-turns 25 --with-builtin developer
        #   --provider <goose-provider> --model <model-id> -t "<prompt>"
        # --no-session avoids session files; the developer builtin enables
        # headless file read/write (review prompts ask the model to WRITE
        # the output JSON file) with no approval prompts. Auth is env-only
        # (e.g. OPENROUTER_API_KEY), inherited by the subprocess.
        cmd = [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
        ]
        base_model, _effort = split_reasoning_effort(model, REASONING_EFFORTS)
        head, _sep, rest = base_model.partition("/")
        if rest:
            cmd += ["--provider", head, "--model", rest]
        else:
            cmd += ["--model", base_model]
        cmd += ["-t", prompt]
        return cmd

    def get_env(self, model: str) -> Dict[str, str]:
        # goose has no CLI flag for reasoning effort; a ``:effort`` model
        # suffix is delivered via GOOSE_THINKING_EFFORT instead. Without a
        # suffix the var is deliberately NOT set, so goose falls back to
        # the user's own config.
        _base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return {"GOOSE_THINKING_EFFORT": effort}
        return {}

    def vendor_text(self, stdout: str, stderr: str) -> Tuple[str, str]:
        """The envelope minus the turns goose echoed back at us.

        goose's ``{"messages": [...], "metadata": {...}}`` envelope opens with
        the user prompt copied verbatim, so ``classify_failure`` was reading
        ``prompts/plan_review.txt`` alongside goose's answer — and that prompt's
        example block contains the word "timeout", which alone made a completely
        successful run classify as a transport fault (the ``prompt-echo``
        fixture row measured exactly that).

        The rule is the narrowest one that fixes it: drop ``messages[]`` entries
        whose ``role`` is ``"user"``, and nothing else. Everything goose's own
        diagnosis needs survives, because it lives in the assistant message —
        ``Ran into this error:``, ``Network error:``, the quoted
        ``500 Internal Server Error`` status line, the ``401 Unauthorized``
        gloss and the ``Available models for this provider`` list are all on
        that side of the filter, and ``metadata`` is untouched.

        Degrades to its input on anything unexpected: a non-JSON stdout (goose
        prints one on no measured path, but a killed process leaves a partial
        envelope), a non-list ``messages``, or an envelope with no user turn in
        it at all. stderr is returned unchanged — goose writes 0 bytes there on
        all eleven captures, and a reduction of a stream that carries no echo
        could only lose a diagnosis.
        """
        try:
            envelope = json.loads(stdout)
            messages = envelope["messages"]
            if not isinstance(messages, list):
                return stdout, stderr
            kept = [
                message for message in messages
                if not (isinstance(message, dict) and message.get("role") == "user")
            ]
            if len(kept) == len(messages):
                return stdout, stderr
            envelope["messages"] = kept
            return json.dumps(envelope, indent=2, ensure_ascii=False), stderr
        except (AttributeError, KeyError, TypeError, ValueError):
            return stdout, stderr

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse the single JSON envelope emitted by goose --output-format json.

        1. Parse the envelope; find the LAST message with role == "assistant"
        2. Concatenate its content[].text parts where type == "text"
        3. "Ran into this error:" in the text -> explicit failure (goose
           exits 0 even on provider errors)
        4. Parse the text as JSON, with fallback extraction for code blocks
           and JSON embedded in prose
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

        messages = envelope.get("messages") if isinstance(envelope, dict) else None
        if not isinstance(messages, list):
            messages = []

        text = None
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                parts = [
                    part.get("text")
                    for part in message.get("content", [])
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ]
                text = "\n".join(parts)

        if text is None:
            return {"success": False, "error": "No assistant message in output", "raw": stdout, "data": None}

        text = text.strip()

        if "Ran into this error:" in text:
            return {"success": False, "error": text, "raw": stdout, "data": None}

        if not text:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        if text.startswith(("[", "{")):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return extract_json_from_text(text, prefer_arrays=True)
