"""Pi coding agent CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema, split_reasoning_effort

# Thinking levels accepted by `pi --thinking` (per pi.dev docs). pi also
# parses the same levels as a native ":level" suffix on --model; the adapter
# strips the suffix itself so only --thinking carries it.
REASONING_EFFORTS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})


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

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse plain-text output from pi print mode.

        Attempts direct JSON parsing first, then falls back to extraction
        from code blocks or JSON embedded in prose.
        """
        stdout = stdout.strip()

        if not stdout:
            return {"success": False, "error": "Empty output", "raw": stdout, "data": None}

        if stdout.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        return extract_json_from_text(stdout, prefer_arrays=True)
