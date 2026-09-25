"""Pi coding agent CLI provider implementation."""
import json
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..json_extractor import extract_json_from_text
from .base import (
    LLMProvider,
    ReasoningLadder,
    StructuredSchema,
    split_reasoning_effort,
)

# Thinking levels accepted by `pi --thinking` (per pi.dev docs). pi also
# parses the same levels as a native ":level" suffix on --model; the adapter
# strips the suffix itself so only --thinking carries it.
#
# Seven levels, and they are the seven rungs with `off` in `none`'s place: pi
# spells the bottom rung `off` and takes no `none` at all, so `off` is declared
# as a spelling and the ladder renders canonical `none` as `off` here.
EFFORT_LADDER = ReasoningLadder(
    provider="pi",
    rungs=("minimal", "low", "medium", "high", "xhigh", "max"),
    spellings=(("off", "none"),),
)
REASONING_EFFORTS = EFFORT_LADDER.accepted

#: The bare HTTP-status prefix pi writes ahead of the vendor's JSON error object
#: on stderr (``401: {"message": …}``). It is the **only** place the status
#: survives — the JSON payload carries `type` and `code` but never the number —
#: so the prefix is stripped to reach the JSON and then put back in the message.
_STATUS_PREFIX = re.compile(r"^(\d{3})\s*:\s*")

#: ``describe_failure`` is a SUMMARY, not a second copy of the stream. The
#: caller keeps the full text — Lane A in ``details["stderr"]`` and in the log
#: file — so a long stderr is capped here rather than promoted into an error
#: string. pi's longest committed stderr is 218 bytes, so nothing measured is
#: actually cut; the cap exists so a future release cannot make it one.
_MAX_DIAGNOSIS_CHARS = 300


def _one_line(text: str) -> str:
    """Collapse a captured fragment to a single line, whitespace normalized."""
    return " ".join((text or "").split())


def _cap(text: str) -> str:
    """One short line (:data:`_MAX_DIAGNOSIS_CHARS`), with an explicit ellipsis."""
    if len(text) <= _MAX_DIAGNOSIS_CHARS:
        return text
    return text[:_MAX_DIAGNOSIS_CHARS - 1].rstrip() + "…"


def parse_answer_text(text: str) -> Dict[str, Any]:
    """The answer-channel reader: a bare model answer in, a result dict out.

    Direct ``json.loads`` when the text starts with ``[`` or ``{``, else the
    shared extractor over code blocks and JSON embedded in prose, arrays
    preferred.

    Module-level because it has **two** callers now. :meth:`PiProvider._parse_stdout`
    is one — pi's stdout *is* the answer — and :class:`~utils.providers.omp.OmpProvider`
    is the other, which reaches the same text by a different road: omp renders
    ``--mode json`` and the answer is the concatenated ``text`` content of the
    last assistant ``message_end``, so what arrives here is an answer either way.
    Lifting it is the whole of what those two adapters share; everything else
    about them — the channel model, the failure taxonomy, the tool policy, the
    transports, usage — diverges, which is why omp does not subclass pi.

    Behaviour is unchanged from when this was a method body, and pi's own tests
    are untouched.
    """
    text = (text or "").strip()

    if not text:
        return {"success": False, "error": "Empty output", "raw": text, "data": None}

    if text.startswith(('[', '{')):
        try:
            return {"success": True, "data": json.loads(text)}
        except json.JSONDecodeError:
            pass

    return extract_json_from_text(text, prefer_arrays=True)


class PiProvider(LLMProvider):
    """Provider for the Pi coding agent CLI (https://pi.dev).

    pi -p (print mode) runs one-shot headless and writes the final assistant
    text to stdout as plain text; --no-session keeps runs ephemeral (no
    session files written). On error pi exits non-zero with the message on
    stderr — the runner turns that into a failure before parse_output runs,
    so parse_output only ever sees exit-0 plain text.

    That last sentence is now MEASURED rather than claimed, across nine
    failure modes at 0.80.6 (tests/fixtures/pi/manifest.yaml), and it holds
    with one exception that matters. stdout is **zero bytes** on every
    failure and the diagnosis is on stderr behind a bare HTTP-status prefix
    (`401: {...}`), carrying the vendor's JSON error object verbatim — but a
    **400 context-length** is swallowed whole: nothing on either channel and
    exit **0**. That run is byte-identical to a model answering with the
    empty string, so on the one condition §7.2 declares terminal there is
    no signal at all for parse_output or classify_failure to read.

    **C4 reads that stderr** (2026-08-20). It used to be accepted by
    :meth:`parse_output` and never mentioned again, so all nine failure rows
    reported the same contentless ``Empty output`` while the vendor's JSON error
    object sat unused. :meth:`describe_failure` now summarizes it for both lanes
    and :meth:`parse_output` names it in its error branch; the precedence, and
    the 4 / 3 split it produces over the seven stderr-bearing captures, are in
    :meth:`_diagnose` and pinned by ``tests/test_stderr_diagnosis.py``. The three
    rows with nothing on either channel keep ``Empty output`` unembellished —
    nothing is invented from an exit code — and no classification moved.

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

    version_cmd = ("pi", "--version")
    #: 0.80.6 is where every fixture in ``tests/fixtures/pi/`` was captured — the
    #: §6 step-5 probe, ``--network none``, the stub and the failserver both
    #: running inside the container, no credential — and the release
    #: ``tests/container/cli-versions.lock`` pins. It is a pin to re-confirm
    #: rather than a range to trust, and what the parser depends on is not a
    #: string it can grep for but the ABSENCE of one: pi print mode writes the
    #: answer to stdout bare, and on a failure writes **zero bytes** there and
    #: puts the diagnosis on stderr behind an HTTP-status prefix; the class
    #: docstring's claim to that effect is measured across nine failure modes in
    #: ``tests/fixtures/pi/manifest.yaml``. A release that started printing a
    #: banner, a spinner or a `Tokens:` footer would break :meth:`parse_output`
    #: with no marker to warn on.
    #:
    #: **There is deliberately no ``min_version``.** A draft declared
    #: ``min_version = "0.80.6"`` — the version the capture happened to run on —
    #: which is `harness-invocation-improvements.md` §5.4.1(a)'s error exactly: a
    #: measured-at version is an upper bound on the true floor, never the floor.
    #: pi is a **pre-existing** adapter, nothing measured says 0.79 fails, and
    #: the hazard above is a *future* release growing a banner, which a floor
    #: cannot catch and a ``verified_version`` warning names. ``min_version`` is
    #: a hard exit-3 refusal in ``subagent_executor._detect_version`` and
    #: ``verified_version`` is a one-line warning; only the warning is earned.
    #:
    #: ``version_cmd`` stays, because it is the only thing that can ever make
    #: ``verified_version`` speak: with no probe the detected version is
    #: ``unknown`` and the warning is skipped. Confirmed against the pinned
    #: binary 2026-08-20 (``docker run --rm --network none --entrypoint
    #: /opt/mllm/real/pi mllm-e2e:local --version``): a bare ``0.80.6`` on
    #: stdout, which the base :meth:`parse_version` reads without an override.
    verified_version = "0.80.6"

    # Usage (§6.3 probe, pi 0.87.1, 2026-09-25): `--mode json` carries omp's usage shape
    # (tests/fixtures/pi/json-mode/); moving pi onto it + USAGE_FULL is a follow-up plan.

    @property
    def name(self) -> str:
        return "pi"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("pi") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # pi --no-session -p [--thinking <effort>] --model <model> "<prompt>"
        # -p prints the final answer and exits; the prompt goes last as a
        # positional argument.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = ["pi", "--no-session", "-p"]
        if effort is not None:
            cmd += ["--thinking", effort]
        cmd += ["--model", base_model, prompt]
        return cmd

    #: The two tiers :meth:`_diagnose` can answer from, in precedence order.
    #: Named so a test can grade WHICH rung produced an answer rather than only
    #: the text, and so the split itself (4 / 3 over the seven stderr-bearing
    #: captures) is an assertion instead of a claim.
    TIER_VENDOR_JSON = "vendor-json"
    TIER_VERBATIM = "verbatim"

    def _diagnose(self, stdout: str, stderr: str) -> Optional[Tuple[str, str]]:
        """``(tier, one-line message)`` for a failed pi run, or ``None``.

        The stderr precedence, in the fixed order every caller inherits:

        1. **Channel.** stderr is the diagnosis, stdout is the answer. ``stdout``
           is accepted and deliberately unread: pi writes **0 bytes to stdout on
           all ten** of its committed failure captures, and on the runs where it
           writes something that something is the answer, which belongs on
           ``raw`` and never in an error string.
        2. **Within stderr.** Strip the bare ``NNN: `` status prefix, parse the
           remainder as the vendor's JSON error object, and report its
           ``message`` with its ``code`` and the stripped status — ``auth-error``
           becomes *401: Incorrect API key provided: sk-stub***… (invalid_api_key)*.
           The status is kept because the prefix is the only place it survives.
           If the remainder does not parse, the stderr text is reported
           verbatim: ``truncated``'s *Stream ended without finish_reason* and
           ``zero-body``'s *Connection error.* are already the whole diagnosis.
        5. **Both channels empty is ``None``.** Three captures are 0 bytes on
           both — ``context-overflow`` (the silent 400), ``empty`` and
           ``exit-zero-failure``, byte-identical to each other — and they keep
           today's ``Empty output`` unembellished. Nothing is invented from an
           exit code: a status is text pi wrote or it is not reported.

        (Rungs 3 and 4 of the shared precedence are gemini's: pi has no
        three-layer envelope and writes no Node stack or temp-file path.)
        """
        text = _one_line(stderr)
        if not text:
            return None

        match = _STATUS_PREFIX.match(text)
        if match:
            payload = None
            try:
                payload = json.loads(text[match.end():])
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    line = f"{match.group(1)}: {_one_line(message)}"
                    code = payload.get("code")
                    if isinstance(code, str) and code.strip():
                        line = f"{line} ({code.strip()})"
                    return (self.TIER_VENDOR_JSON, _cap(line))

        return (self.TIER_VERBATIM, _cap(text))

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """pi's half of C4: the vendor's own words instead of a bare exit code.

        Lane A calls this on every non-zero exit and appends the answer to the
        slot's error; Lane B calls it when ``parse_output`` failed and recovered
        no text. pi exits non-zero on all but the silent 400, so this is the
        path that reaches a user. See :meth:`_diagnose` for the precedence.

        **Diagnostics, never classification.** Nothing here moves any run's
        ``classify_failure`` verdict — that reads the raw streams through the
        declared pattern tables and never consults this method.
        """
        diagnosis = self._diagnose(stdout, stderr)
        return None if diagnosis is None else diagnosis[1]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse plain-text output from pi print mode, and NAME the stderr.

        Attempts direct JSON parsing first, then falls back to extraction
        from code blocks or JSON embedded in prose.

        The rig calls this on failure rows regardless of exit code and Lane B
        calls it before anything else, so naming the stderr text in the error
        branch is what makes pi's nine failure rows distinguishable from each
        other at all — before C4 every one of them collapsed to the same
        contentless ``Empty output``. Only the ``error`` string grows: the
        parsed document, ``raw`` and ``success`` are untouched, so no caller's
        view of a *successful* run moves and stderr never enters ``data``.
        """
        result = self._parse_stdout(stdout)
        if result.get("success"):
            return result

        detail = self.describe_failure(stdout, stderr)
        if not detail:
            return result
        result = dict(result)
        result["error"] = f"{result.get('error') or 'parse failed'} (stderr: {detail})"
        return result

    def _parse_stdout(self, stdout: str) -> Dict[str, Any]:
        """Today's parser, unchanged — the answer channel, read on its own.

        The body now lives in the module-level :func:`parse_answer_text`, which
        omp calls too; this method is the name pi's own call sites and tests use.
        """
        return parse_answer_text(stdout)
