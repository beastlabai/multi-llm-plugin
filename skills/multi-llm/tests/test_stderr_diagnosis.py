"""C4 and C3's gemini row: the stderr that `pi` and `gemini` used to discard.

**What C4 fixed.** Both adapters have taken ``parse_output(self, stdout,
stderr)`` since the base class was written, and both accepted ``stderr`` and
then never mentioned it again — so 100% of the diagnosis was thrown away.
``pi`` writes **zero bytes to stdout on all ten** of its committed failure
captures and ``gemini`` on **eight of its nine**; every one of those failures
reported the same contentless string (``Empty output``, ``JSON parse error:
Expecting value: line 1 column 1 (char 0)``) while the vendor's full error
object sat unread on the other channel. The repair is two-part, because the two
lanes reach different hooks: ``describe_failure`` for the non-zero-exit path
(Lane A on every non-zero exit, Lane B when ``parse_output`` failed and
recovered no text) and ``parse_output``'s own error branch for the exit-0 case
and the §6.2 fixture rig.

**The five-rung stderr precedence, which this module grades rung by rung.**

1. **Channel** — stderr is the diagnosis, stdout is the answer. For seventeen of
   the nineteen failure captures the rung decides nothing (one channel is
   empty); gemini's ``exit-zero-failure`` is the nineteenth and is why it is
   written down at all.
2. **Within pi's stderr** — strip the bare ``NNN: `` status prefix, report the
   vendor JSON's ``message`` with its ``code`` and the stripped status; a
   non-JSON remainder is reported verbatim.
3. **Within gemini's stderr, three tiers** — the last ``ApiError`` blob, then
   the prose after the ``Full report`` path, then the first line verbatim.
4. **Two things are dropped first** — the Node ``    at …`` stack and the
   ``/tmp/mllm-inv/…`` report path, both per-invocation noise — and what is left
   is capped to one short line.
5. **Both channels empty keeps today's string, unembellished** — pi's three
   silent rows, with ``describe_failure`` returning ``None``. Nothing is
   invented from an exit code.

**The rule that binds all of it: ``describe_failure`` is diagnostics, never
classification.** Both callers wrap it in ``try``/``except`` and neither lets it
change a category. :class:`TestNoClassifyCellMoved` is that rule as an
assertion, and it is why gemini's ``exit-zero-failure`` markers (``Error
executing tool``, ``not found in registry``) are surfaced here as *prose* and
deliberately not declared as ``exit_zero_failure_markers``.

**C3's gemini row** is the one classification change in this file's scope, and
it is a pattern-table change rather than a prose one: gemini's measured 401
wording matched none of the ten base auth patterns, so a logged-out gemini read
as transport-shaped and burned the whole bounded-retry budget.
:class:`TestGeminiAuthPatterns` pins the fix and — the half that matters after
C1 — pins that **no** wording C3 adds matches a success capture anywhere in the
corpus, including the four adapters that echo our own prompt back into their
envelopes. That sweep is derived from every C3 adapter's own
``auth_error_patterns`` delta (:data:`C3_ADAPTERS`), not from gemini's two, so
muse's ``api key from`` is graded as directly as gemini's ``unauthenticated``.

**``kilocode`` joined on 2026-09-10**, from the other end: not a probe run but
one real ``review-code`` slot against a dead account, where the CLI painted
``Error: User not found.`` onto stderr, exited 1, and the operator-visible line
read ``kilocode exited with code 1`` — costing the reader a diagnosis step,
because a bare exit code sends them looking for a parse problem when the fix is
provider auth. It is graded on the same five rungs as pi and gemini, and it adds
the one thing neither of them could exercise: **paint**. This CLI colours its own
diagnostics, so the escapes come off at rung 4 alongside the run banner, and the
painted capture and its unpainted twin have to produce the identical sentence —
a diagnosis that changes with the terminal's colour support is not one.
``plans/reaggregate-identity-and-salvage-gating.md`` residual item 4 is the
entry that asked for these rows.

**pi moved onto ``--mode json`` on 2026-09-25** (plans/opencode-validator-false-
rate-limit-and-denied-turn.md §7, T7.6), and its half of this module was
re-derived from the recaptured 0.87.1 set rather than edited cell by cell. The
vendor's words now arrive on **stdout**, as the terminal assistant message's
``errorMessage`` — the same ``NNN: {json}`` string 0.80.6 wrote to stderr — with
stderr empty and exit 0, so pi's precedence is "stream verdict, else stderr"
(``STDOUT_ELSE_STDERR``, omp's). ``TIER_VENDOR_JSON``/``TIER_VERBATIM`` stay and
now apply to ``errorMessage``; ``TIER_STREAM`` names a structural verdict (a
``length`` cut); stderr answers only for a header-only pre-flight. The 0.80.6
text-mode set is kept in ``tests/fixtures/pi/text-mode-0.80.6/``, read here by
explicit path twice: :data:`PI_CLASSIFY_0_80_6` is the floor no recaptured row
may fall below (:class:`TestNoPiClassifyCellWorsened`), and its ``auth-error``
stderr is still the vendor-JSON sample omp's table borrows.

C4's and C3's tests live here rather than in ``test_providers.py`` on purpose:
they are one subject read end to end, and the hand-written per-adapter classes
in that module grade parsing. The channel each override declares is pinned
*there*, in ``_DESCRIBE_FAILURE_CHANNELS``, whose roster is derived from the
classes — so an override cannot land without declaring the stream it reads, and
kilocode's row is ``stderr``.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import invoke_with_provider  # noqa: E402
from utils.providers.base import LLMProvider, matches_any  # noqa: E402
from utils.providers.claude_code import ClaudeCodeProvider  # noqa: E402
from utils.providers.cline import ClineProvider  # noqa: E402
from utils.providers.gemini import GeminiProvider  # noqa: E402
from utils.providers.muse import MuseProvider  # noqa: E402
from utils.providers.gemini import _MAX_DIAGNOSIS_CHARS as GEMINI_CAP  # noqa: E402
from utils.providers.kilocode import KiloCodeProvider  # noqa: E402
from utils.providers.kilocode import _MAX_DIAGNOSIS_CHARS as KILOCODE_CAP  # noqa: E402
from utils.providers.omp import OmpProvider  # noqa: E402
from utils.providers.omp import stderr_diagnosis as omp_stderr_diagnosis  # noqa: E402
from utils.providers.pi import PiProvider  # noqa: E402
from utils.providers.pi import _MAX_DIAGNOSIS_CHARS as PI_CAP  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

PI = PiProvider()
GEMINI = GeminiProvider()
KILOCODE = KiloCodeProvider()


def _capture_rows(provider):
    """Every committed capture row for ``provider``, by name."""
    directory = FIXTURES / provider
    return sorted({path.name.split(".")[0] for path in directory.glob("*.txt")})


def _streams(provider, row):
    """``(stdout, stderr)`` for one capture row; a missing file is 0 bytes."""
    out = []
    for key in ("stdout", "stderr"):
        path = FIXTURES / provider / f"{row}.{key}.txt"
        out.append(path.read_text(encoding="utf-8") if path.is_file() else "")
    return out[0], out[1]


def _manifest(provider):
    return yaml.safe_load(
        (FIXTURES / provider / "manifest.yaml").read_text(encoding="utf-8")
    )


#: pi's historical 0.80.6 text-mode set, moved aside by T7.2. Never globbed: it
#: is read by explicit path only, as the floor and as omp's borrowed sample.
PI_TEXT_MODE = FIXTURES / "pi" / "text-mode-0.80.6"


def _pi_text_mode_streams(row):
    """``(stdout, stderr)`` of one 0.80.6 text-mode capture, by explicit path."""
    out = []
    for key in ("stdout", "stderr"):
        path = PI_TEXT_MODE / f"{row}.{key}.txt"
        out.append(path.read_text(encoding="utf-8") if path.is_file() else "")
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# The tier tables. Every committed capture is named, with the rung that must
# answer it — so the precedence is GRADED rather than the text alone, and a new
# capture cannot be added without deciding which rung owns it.
# --------------------------------------------------------------------------- #

#: pi's twelve failure rows under ``--mode json`` split **6 / 4 / 2**: vendor
#: JSON behind an ``NNN: `` prefix in the terminal ``errorMessage``, a bare
#: sentence (three ``errorMessage``s and the pre-flight's stderr), and two
#: structural ``length`` verdicts. ``None`` is a success, or ``empty`` — a
#: complete stream whose model answered nothing, which names no failure.
PI_TIERS = {
    "auth-error": PiProvider.TIER_VENDOR_JSON,
    "invalid-model": PiProvider.TIER_VENDOR_JSON,
    "rate-limit": PiProvider.TIER_VENDOR_JSON,
    "server-error": PiProvider.TIER_VENDOR_JSON,
    "exit-zero-failure": PiProvider.TIER_VENDOR_JSON,
    "context-overflow": PiProvider.TIER_VENDOR_JSON,
    "error-envelope": PiProvider.TIER_VERBATIM,
    "truncated": PiProvider.TIER_VERBATIM,
    "zero-body": PiProvider.TIER_VERBATIM,
    "preflight-no-key": PiProvider.TIER_VERBATIM,
    "length": PiProvider.TIER_STREAM,
    "length-recovery": PiProvider.TIER_STREAM,
    "empty": None,
    "success": None,
    "multi-assistant": None,
    "markers-in-tool-result": None,
    "context-overflow-compacted": None,
}

#: gemini's nine failure rows split **4 / 4 / 1**: the last ``ApiError`` blob,
#: the prose after the ``Full report`` path, the first line verbatim. The single
#: tier-(iii) row is ``exit-zero-failure``, which has no blob, no path and no
#: trailing envelope — and is also the only capture with bytes on both channels.
GEMINI_TIERS = {
    "auth-error": GeminiProvider.TIER_API_ERROR,
    "context-overflow": GeminiProvider.TIER_API_ERROR,
    "rate-limit": GeminiProvider.TIER_API_ERROR,
    "server-error": GeminiProvider.TIER_API_ERROR,
    "error-envelope": GeminiProvider.TIER_POST_PATH_PROSE,
    "invalid-model": GeminiProvider.TIER_POST_PATH_PROSE,
    "truncated": GeminiProvider.TIER_POST_PATH_PROSE,
    "zero-byte-response": GeminiProvider.TIER_POST_PATH_PROSE,
    "exit-zero-failure": GeminiProvider.TIER_FIRST_LINE,
    "ephemeral-only": None,
    "multi-assistant": None,
    "success": None,
}

#: The vendor's own words, per row. These are what make pi's nine failure rows
#: and gemini's eight distinguishable from each other at all — before C4 each
#: set collapsed to one contentless string.
PI_WORDS = {
    "auth-error": ("401: Incorrect API key provided", "(invalid_api_key)"),
    "invalid-model": ("404: The model `stub-model` does not exist", "(model_not_found)"),
    "rate-limit": ("429: Rate limit reached for stub-model", "(rate_limit_exceeded)"),
    # The vendor JSON's `code` is null on a 500, so the status and message
    # stand alone — the tier reports what is there, never a placeholder.
    "server-error": ("500: The server had an error while processing your request",),
    "exit-zero-failure": ("500: The server had an error while processing your request",),
    "context-overflow": ("400: This model's maximum context length is 8192 tokens",
                         "(context_length_exceeded)"),
    "error-envelope": ("Stream ended without finish_reason",),
    "truncated": ("Stream ended without finish_reason",),
    "zero-body": ("Connection error.",),
    "preflight-no-key": ("No API key found for stub.",),
    "length": ("Truncated output (stopReason length",),
    "length-recovery": ("Truncated output (stopReason length",),
}

GEMINI_WORDS = {
    "auth-error": ("401: Incorrect API key provided",),
    "context-overflow": ("400: This model's maximum context length is 8192 tokens",),
    "rate-limit": ("429: Rate limit reached for stub-model",),
    "server-error": ("500: The server had an error while processing your request",),
    "error-envelope": ("Error: Incomplete JSON segment at the end", "(code 1)"),
    "truncated": ("Error: Incomplete JSON segment at the end", "(code 1)"),
    "invalid-model": ("ModelNotFoundError: The model `gemini-2.5-flash` does not exist",),
    "zero-byte-response": ("Error: exception TypeError: fetch failed sending request",),
    "exit-zero-failure": ('Tool "write_file" not found in registry',),
}

#: Rung 5 under ``--mode json``: the one failure row that names no failure on
#: either channel. At 0.80.6 there were three byte-identical 0-byte rows
#: (``context-overflow``, ``empty``, ``exit-zero-failure``); at 0.87.1 only a
#: model that really answered the empty string is left — the overflow and the
#: 5xx carry the vendor's words (:class:`TestRungFiveKeepsTodaysString`).
PI_SILENT_ROWS = ("empty",)

#: Every graded json-mode row's ``classify_failure`` (0.87.1), as a literal so a
#: future edit to the adapter's *prose* that moved a *category* fails here by
#: name, which is the one thing C4 was not allowed to do.
PI_CLASSIFY = {
    "success": None,
    "multi-assistant": None,
    "markers-in-tool-result": None,
    "context-overflow-compacted": None,
    "truncated": None,
    "error-envelope": None,
    "exit-zero-failure": "server_error",
    "empty": None,
    "auth-error": "auth_failed",
    "rate-limit": "rate_limited",
    "invalid-model": "invalid_model",
    "context-overflow": "context_length_exceeded",
    "zero-body": "connection_error",
    "server-error": "server_error",
    "preflight-no-key": "auth_failed",
    "length": None,
    "length-recovery": None,
}

#: The 0.80.6 text-mode cells, verbatim from before the move (T7.6's floor):
#: :class:`TestNoPiClassifyCellWorsened` asserts no recaptured row classifies
#: worse. ``None`` is the least specific verdict, so a row may only stay put or
#: move off ``None``.
PI_CLASSIFY_0_80_6 = {
    "success": None,
    "truncated": None,
    "error-envelope": None,
    "exit-zero-failure": None,
    "empty": None,
    "auth-error": "auth_failed",
    "rate-limit": "rate_limited",
    "invalid-model": "invalid_model",
    "context-overflow": None,
    "zero-body": "connection_error",
    "server-error": None,
}

GEMINI_CLASSIFY = {
    "success": None,
    "truncated": None,
    "ephemeral-only": None,
    "multi-assistant": None,
    "error-envelope": None,
    # NOT moved by C4, and deliberately: the measured markers `Error executing
    # tool` / `not found in registry` sit on stderr and would make a plausible
    # `exit_zero_failure_markers` declaration. Declaring them moves this cell,
    # so it is out of C4's scope by C4's own rule.
    "exit-zero-failure": None,
    # C3, and the ONE cell in either manifest that moved. It read None.
    "auth-error": "auth_failed",
    "rate-limit": "rate_limited",
    "invalid-model": "invalid_model",
    "context-overflow": "context_length_exceeded",
    "server-error": None,
    "zero-byte-response": None,
}


# --------------------------------------------------------------------------- #
# kilocode's captures — literals rather than files, and why
# --------------------------------------------------------------------------- #
#
# pi and gemini were probed row by row against a localhost stub inside the E2E
# container, so their nineteen captures are committed as
# `<row>.{stdout,stderr}.txt` pairs with a `manifest.yaml` the §6.2 rig grades,
# and `_streams` reads them off disk. kilocode has had no such probe run. What it
# has is ONE real slot — a `--review-code --quick` fan-out on 2026-08-24 whose
# `kilocode` model was configured against an account the key no longer belonged
# to — and those bytes are already committed verbatim as a literal twice:
# `tests/test_json_extractor.py::KILOCODE_AUTH_FAILURE` (the salvage gate: is
# there anything here to repair?) and its twin in `tests/test_e2e_failures.py`
# (the path a slot with nothing to salvage takes). This is the third statement of
# the same measurement, for the third question it answers — what the operator
# reads — and it is written the same way rather than imported across test modules,
# which would couple three suites' collection to one file's identifier.

#: The measured bytes. Colour, the CLI's run banner, and a four-word failure.
KILOCODE_AUTH_STDERR = (
    "\x1b[0m\n> code · moonshotai/kimi-k3\n\x1b[0m\n"
    "\x1b[91m\x1b[1mError: \x1b[0mUser not found.\n"
)

#: The stdout the same slot produced: **nothing**. It is spelled out rather than
#: left implicit because it is what puts 100% of the diagnosis on stderr, which
#: is the whole reason this adapter needed the hook.
KILOCODE_AUTH_STDOUT = ""

#: Every row this adapter is graded on. The first is the capture; the next two
#: are **derivations of it, marked as such** — neither is a second measurement,
#: and each exists because it is a shape the same failure really arrives in:
#:
#: * ``auth-error-no-color`` — the same run with the paint off, which is what a
#:   ``NO_COLOR``/non-tty build prints (``NO_COLOR`` and ``FORCE_COLOR`` are both
#:   in ``base.ENV_ALLOWLIST_BASE_NAMES``, so the harness passes the user's choice
#:   straight through to the child and either rendering can reach this hook).
#: * ``auth-error-from-a-log`` — the same run after a round-trip through a log
#:   that dropped its ESC bytes, leaving ``[91m[1m`` as bare text flush against
#:   the label. ``tests/test_json_extractor.py`` pins that shape for the salvage
#:   gate for the same reason, and it is the row that calibrated
#:   ``kilocode._ERROR_LABEL`` (see that constant: anchored, this line loses to
#:   the content-free ``[0m`` residue above it).
#:
#: ``banner-only`` and ``clean-run`` are the two negatives, and they are the
#: point of the table rather than padding: one says the run header is never
#: promoted to a diagnosis, the other that a successful slot has none.
KILOCODE_CAPTURES = {
    "auth-error": (KILOCODE_AUTH_STDOUT, KILOCODE_AUTH_STDERR),
    "auth-error-no-color": (
        KILOCODE_AUTH_STDOUT,
        "\n> code · moonshotai/kimi-k3\n\nError: User not found.\n",
    ),
    "auth-error-from-a-log": (
        KILOCODE_AUTH_STDOUT,
        KILOCODE_AUTH_STDERR.replace("\x1b", ""),
    ),
    "banner-only": ("", "\x1b[0m\n> code · moonshotai/kimi-k3\n\x1b[0m\n"),
    "clean-run": ('[{"title": "x", "desc": "y", "importance": "low"}]', ""),
}

#: kilocode's five rows split **3 / 0 / 2**: every failing one carries the CLI's
#: own ``Error:`` label, so the fallback tier answers none of them today. That is
#: recorded as a measurement, not as dead code — the fallback is what speaks for
#: a failure whose wording nobody here has seen, and the day one lands it changes
#: this column rather than the adapter.
KILOCODE_TIERS = {
    "auth-error": KiloCodeProvider.TIER_ERROR_LINE,
    "auth-error-no-color": KiloCodeProvider.TIER_ERROR_LINE,
    "auth-error-from-a-log": KiloCodeProvider.TIER_ERROR_LINE,
    "banner-only": None,
    "clean-run": None,
}

#: The vendor's own words, per row — the four that name the failure, and the
#: whole sentence where the escapes did not survive to be stripped.
KILOCODE_WORDS = {
    "auth-error": ("Error: User not found.",),
    "auth-error-no-color": ("Error: User not found.",),
    "auth-error-from-a-log": ("User not found.",),
}

#: The acceptance snapshot, and the cell this change was **not** allowed to move.
#: ``User not found.`` is a wording ``providers/cline`` declares in its
#: ``auth_error_patterns`` because cline measured it; kilocode declares nothing of
#: the sort, so this capture classifies as ``None`` — no rung, no ``[needs-auth]``
#: tag appended to the operator's line. Declaring it here would be a
#: classification change with its own measurement, and ``describe_failure`` is
#: diagnostics, never classification.
KILOCODE_CLASSIFY = {row: None for row in KILOCODE_CAPTURES}


def _kilocode_streams(row):
    """``(stdout, stderr)`` for one kilocode row, mirroring :func:`_streams`."""
    return KILOCODE_CAPTURES[row]


# --------------------------------------------------------------------------- #
# omp's pre-flight captures — literals, for kilocode's reason
# --------------------------------------------------------------------------- #
#
# `tests/fixtures/omp/` is the §6.2 rig's business and holds the JSONL STREAMS
# the parser is graded on. These are the other channel: the pre-flight faults,
# which are the one shape omp puts on stderr at all. In `--mode json` a
# run that reached the backend writes a JSONL stream to stdout and ZERO bytes to
# stderr — on success and on an API failure alike — so everything below is a
# fault that happened before the first request, at zero spend, with an empty
# stdout. They are written here as literals because that is where the rule they
# grade lives: `stderr_diagnosis` is a module-level function in `omp.py` with
# three reductions and no tier machinery, and the bytes are short enough to read
# in place.
#
# Captured 2026-09-17 against omp 18.2.2 by running the real binary with
# `stdin` at /dev/null (an inherited non-EOF pipe blocks omp forever at startup
# phase `readPipedInput`, which is its own §1.6 measurement). One edit, and it is
# marked here rather than left to be noticed: the capturing user's home path is
# rewritten to `/home/e2e`. Nothing else was touched.

#: A bad `--model`: exit 1, 0 B stdout, four lines of plain text. No stack trace
#: at all — omp only dumps one when the failure came out of its own code.
OMP_MODEL_NOT_FOUND = (
    'Model "openrouter/nope/does-not-exist" not found\n'
    "\n"
    "Set an API key environment variable:\n"
    "  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.\n"
    "\n"
    "Or create /home/e2e/.omp/agent/models.yml\n"
)

#: No credential on the `$env` chain: the diagnosis, then the Bun frames. Both
#: indentations `omp._BUN_NOISE` knows about appear here on purpose — the binary
#: emitted the six-space form, and the four-space one is the shape a frame with
#: no source position takes — so the strip is graded on both rather than on
#: whichever the capture happened to carry.
OMP_NO_API_KEY = (
    "error: No API key found for openrouter.\n"
    "\n"
    "Use /login, set an API key environment variable, or create "
    "/home/e2e/.omp/agent/agent.db\n"
    "      at #xo (/$bunfs/root/omp-linux-x64:466354:15)\n"
    "      at async #in (/$bunfs/root/omp-linux-x64:466139:25)\n"
    "    at write (unknown)\n"
    "    at <anonymous> (/$bunfs/root/omp-linux-x64:587578:27)\n"
)

#: A rejected `--tools` name. Validated BEFORE model resolution, so this costs
#: nothing — and the "Valid tools:" list in it is incomplete and changes between
#: runs, which is why nothing in this repo parses it.
OMP_UNKNOWN_TOOL = (
    "CliUsageError: Unknown tool in --tools: python. Valid tools: read, write, "
    "goal, init_experiment, run_experiment, log_experiment, update_notes.\n"
    "      at s8e (/$bunfs/root/omp-linux-x64:19535:9)\n"
    "      at VU (/$bunfs/root/omp-linux-x64:603636:12)\n"
)

#: The SAME failure as :data:`OMP_NO_API_KEY`, as the binary really writes it:
#: Bun prints a **source-context prelude** — numbered lines of the minified
#: single-file binary's own source, then a caret column — *ahead* of the
#: `error:` line, and only then the frames. Taken verbatim from
#: `tests/fixtures/omp/auth-error-preflight.stderr.txt`, trimmed to two source
#: lines because the shape is what matters and six is just more of it.
#:
#: This row is why the strip is three patterns and not one. The frames follow
#: the diagnosis and cost nothing but noise; the prelude PRECEDES it, and pi's
#: 300-character cap — which `stderr_diagnosis` reuses — was being spent
#: entirely on minified source, ellipsing the one sentence a user acts on.
OMP_SOURCE_CONTEXT_PRELUDE = (
    "466353 |         const n = await this.#ge.getApiKey(this.model, this.sessionId);\n"
    "466354 |           throw new Error(`No API key found for ${this.model.provider}.\n"
    "                       ^\n"
    "error: No API key found for openrouter.\n"
    "\n"
    "Use /login, set an API key environment variable, or create "
    "/home/e2e/.omp/agent/agent.db\n"
    "      at #xo (/$bunfs/root/omp-linux-x64:466354:15)\n"
)

#: Text mode's spinner, which precedes the diagnosis there
#: (`Working...\n401 User not found.`). This adapter never renders text mode —
#: it always passes `--mode json` — so the row is a guard against the day a
#: degraded argv does, not a shape any current dispatch produces.
OMP_TEXT_MODE_SPINNER = "Working...\n401 User not found.\n"


def _omp_stderr(row):
    """One omp capture's stderr; its stdout is 0 bytes on every pre-flight row.

    ``vendor-json`` is the exception and is not omp's own output at all: it is
    ``pi``'s 0.80.6 ``auth-error`` stderr (the text-mode set, by explicit path —
    at 0.87.1 the same line is on stdout), borrowed to state the negative (see
    :data:`OMP_TIERS`).
    """
    if row == "vendor-json":
        return _pi_text_mode_streams("auth-error")[1]
    return {
        "model-not-found": OMP_MODEL_NOT_FOUND,
        "no-api-key": OMP_NO_API_KEY,
        "source-context-prelude": OMP_SOURCE_CONTEXT_PRELUDE,
        "unknown-tool": OMP_UNKNOWN_TOOL,
        "text-mode-spinner": OMP_TEXT_MODE_SPINNER,
        "json-mode-silence": "",
    }[row]


#: omp's rows are **all one tier**, and that is the measurement rather than an
#: omission: `stderr_diagnosis` is pi's verbatim rung with two omp-specific
#: strips in front of it, and pi's second rung has nothing to parse here because
#: omp writes no `NNN: {json}` prefix.
#:
#: ``vendor-json`` is what makes that a claim and not a shrug. It is pi's own
#: `auth-error` bytes — the exact shape pi's vendor-JSON tier exists for — and
#: omp reports them VERBATIM where pi recomposes them into
#: `401: Incorrect API key provided: … (invalid_api_key)`. A future edit that
#: taught omp to parse a vendor body would move this cell, which is the point.
#: ``json-mode-silence`` is rung 5: in json mode stderr is 0 bytes on success and
#: on an API failure alike, and nothing is invented from that.
OMP_TIERS = {
    "model-not-found": PiProvider.TIER_VERBATIM,
    "no-api-key": PiProvider.TIER_VERBATIM,
    "source-context-prelude": PiProvider.TIER_VERBATIM,
    "unknown-tool": PiProvider.TIER_VERBATIM,
    "text-mode-spinner": PiProvider.TIER_VERBATIM,
    "vendor-json": PiProvider.TIER_VERBATIM,
    "json-mode-silence": None,
}

#: The vendor's own words, per row — what a reader has to be left holding.
OMP_WORDS = {
    "model-not-found": ('Model "openrouter/nope/does-not-exist" not found',),
    "no-api-key": ("No API key found for openrouter.",),
    "source-context-prelude": ("No API key found for openrouter.",),
    "unknown-tool": ("Unknown tool in --tools: python",),
    "text-mode-spinner": ("401 User not found.",),
    "vendor-json": ("Incorrect API key provided", "invalid_api_key"),
}

#: What must NOT survive, per row. Each is a reduction with a reason:
#: `Working...` is text mode's spinner and says nothing about the failure, and a
#: Bun frame names an offset inside a 251 MB single-file binary, which is noise
#: to everyone who is not upstream.
OMP_DROPPED = {
    "no-api-key": ("at #xo", "at async #in", "at write (unknown)", "$bunfs"),
    # The prelude's own three shapes, plus the proof that the sentence behind it
    # survives: before the strip reached these lines the diagnosis was
    # `466353 | const n = await this.#ge.getApiKey(…` and the cap ate the rest.
    "source-context-prelude": ("466353 |", "466354 |", "^", "at #xo", "$bunfs"),
    "unknown-tool": ("at s8e", "at VU", "$bunfs"),
    "text-mode-spinner": ("Working...",),
}


def _omp_tier(stderr):
    """Which rung answered — and for omp, whether "verbatim" is really true.

    There is no tier constant to read off `stderr_diagnosis`: it returns a
    string. So the tier is *derived* from the answer, by asking the question the
    word `verbatim` makes: is every word of the diagnosis one of the stderr's
    own words, in the stderr's own order? pi's vendor-JSON rung fails that test
    by construction — it lifts `message` and `code` out of a JSON body and
    re-composes them with parentheses that appear nowhere in the bytes — so a
    row that started being parsed here would come back `parsed` and move its
    cell, rather than passing because the assertion was only "some string".
    """
    diagnosis = omp_stderr_diagnosis(stderr)
    if not diagnosis:
        return None
    words = diagnosis.split()
    if diagnosis.endswith("…"):
        # The cap cuts mid-token; that last fragment is not a word to match.
        words = words[:-1]
    haystack = stderr.split()
    index = 0
    for word in words:
        while index < len(haystack) and haystack[index] != word:
            index += 1
        if index == len(haystack):
            return "parsed"
        index += 1
    return PiProvider.TIER_VERBATIM


# --------------------------------------------------------------------------- #
# Rungs 2 and 3 — which tier answered
# --------------------------------------------------------------------------- #

class TestEveryCaptureLandsOnItsDeclaredTier:
    """The precedence, graded per capture rather than asserted in prose."""

    @pytest.mark.parametrize("row", sorted(PI_TIERS))
    def test_pi(self, row):
        """"Stream verdict, else stderr" (``_diagnose_failure``), graded per row."""
        stdout, stderr = _streams("pi", row)
        diagnosis = PI._diagnose_failure(stdout, stderr)
        tier = None if diagnosis is None else diagnosis[0]

        assert tier == PI_TIERS[row]

    @pytest.mark.parametrize("row", sorted(GEMINI_TIERS))
    def test_gemini(self, row):
        stdout, stderr = _streams("gemini", row)
        diagnosis = GEMINI._diagnose(stdout, stderr)
        tier = None if diagnosis is None else diagnosis[0]

        assert tier == GEMINI_TIERS[row]

    @pytest.mark.parametrize("row", sorted(OMP_TIERS))
    def test_omp(self, row):
        """omp's rung, derived rather than read off a constant (`_omp_tier`).

        omp has no `_diagnose` and no tier machinery: `stderr_diagnosis` is one
        function with three reductions, and every row lands on what pi calls its
        verbatim rung. The derivation is what keeps that from being vacuous — a
        row omp started PARSING would come back `parsed` and fail here.
        """
        assert _omp_tier(_omp_stderr(row)) == OMP_TIERS[row]

    @pytest.mark.parametrize("provider,table", [("pi", PI_TIERS), ("gemini", GEMINI_TIERS)])
    def test_the_table_names_every_committed_capture(self, provider, table):
        """A new capture forces a tier decision instead of going ungraded."""
        assert sorted(table) == _capture_rows(provider), (
            f"{provider}'s tier table and its committed captures disagree — a "
            f"capture must be given the rung that answers it"
        )

    def test_pi_splits_six_four_two(self):
        """Measured at 0.87.1: vendor JSON / bare sentence / structural verdict.

        0.80.6 split 4 / 3 over seven stderr lines; json mode adds the 5xx
        ``exit-zero-failure`` twin and the no-longer-silent 400 to the first
        tier, the pre-flight's stderr to the second, and the two ``length``
        rows as a third.
        """
        tiers = [tier for tier in PI_TIERS.values() if tier is not None]

        assert tiers.count(PiProvider.TIER_VENDOR_JSON) == 6
        assert tiers.count(PiProvider.TIER_VERBATIM) == 4
        assert tiers.count(PiProvider.TIER_STREAM) == 2
        assert len(tiers) == 12, "pi has twelve json-mode failure captures that name a reason"

    def test_gemini_splits_four_four_one(self):
        """The plan's prediction, measured: blob / post-path prose / first line."""
        tiers = [tier for tier in GEMINI_TIERS.values() if tier is not None]

        assert tiers.count(GeminiProvider.TIER_API_ERROR) == 4
        assert tiers.count(GeminiProvider.TIER_POST_PATH_PROSE) == 4
        assert tiers.count(GeminiProvider.TIER_FIRST_LINE) == 1
        assert len(tiers) == 9, "gemini has nine failure captures, all with stderr"

    def test_gemini_collapses_ten_identical_retry_blobs_to_one_line(self):
        """Why tier (i) says *the LAST* blob rather than *a* blob.

        ``rate-limit.stderr.txt`` is 15,013 bytes carrying nine identical
        ``ApiError`` blobs, one per backoff attempt; ``server-error.stderr.txt``
        is 14,191 with ten. Both must come out as one sentence.
        """
        for row in ("rate-limit", "server-error"):
            _, stderr = _streams("gemini", row)
            reason = GEMINI.describe_failure("", stderr)

            assert len(stderr) > 14000
            assert stderr.count("ApiError:") >= 9
            assert reason is not None and len(reason) <= GEMINI_CAP


# --------------------------------------------------------------------------- #
# The message itself
# --------------------------------------------------------------------------- #

class TestTheMessageCarriesTheVendorsWords:

    @pytest.mark.parametrize("row", sorted(PI_WORDS))
    def test_pi(self, row):
        stdout, stderr = _streams("pi", row)
        reason = PI.describe_failure(stdout, stderr)

        assert reason is not None
        for needle in PI_WORDS[row]:
            assert needle in reason

    @pytest.mark.parametrize("row", sorted(GEMINI_WORDS))
    def test_gemini(self, row):
        stdout, stderr = _streams("gemini", row)
        reason = GEMINI.describe_failure(stdout, stderr)

        assert reason is not None
        for needle in GEMINI_WORDS[row]:
            assert needle in reason

    @pytest.mark.parametrize("row", sorted(OMP_WORDS))
    def test_omp(self, row):
        stderr = _omp_stderr(row)
        reason = OmpProvider().describe_failure("", stderr)

        assert reason is not None
        for needle in OMP_WORDS[row]:
            assert needle in reason

    def test_every_pi_failure_row_gets_a_distinct_message(self):
        """The point of C4 for pi: rows that used to be one string.

        Three pairs stay identical, each for a measured reason and none a
        parser could fix: ``truncated``/``error-envelope`` (pi never writes the
        200-with-error-object's text anywhere), ``server-error``/
        ``exit-zero-failure`` (the same failserver ``500``, two containers) and
        ``length``/``length-recovery`` (a recoverable length with nothing to
        compact ends exactly like a full one).
        """
        messages = {
            row: PI.describe_failure(*_streams("pi", row))
            for row in PI_TIERS
            if PI_TIERS[row] is not None
        }

        assert len(set(messages.values())) == 9
        assert messages["truncated"] == messages["error-envelope"]
        assert messages["server-error"] == messages["exit-zero-failure"]
        assert messages["length"] == messages["length-recovery"]

    def test_every_gemini_failure_row_gets_a_distinct_message(self):
        """Same for gemini, with the same one honest collision."""
        messages = {
            row: GEMINI.describe_failure(*_streams("gemini", row))
            for row in GEMINI_TIERS
            if GEMINI_TIERS[row] is not None
        }

        assert len(set(messages.values())) == 8
        assert messages["truncated"] == messages["error-envelope"]

    @pytest.mark.parametrize("row", sorted(_capture_rows("gemini")))
    def test_gemini_never_reports_object_object(self, row):
        """``error.message`` is the literal string ``[object Object]`` at 0.25.2.

        It is the only structured message field gemini offers and upstream
        stringification destroyed it, so the envelope contributes its ``code``
        and nothing else. Reporting it would be worse than reporting nothing:
        it looks like a diagnosis and is not one.
        """
        stdout, stderr = _streams("gemini", row)
        reason = GEMINI.describe_failure(stdout, stderr)

        assert "[object Object]" not in (reason or "")
        assert "object object" not in (reason or "").lower()


class TestRungFourDropsTheNoiseAndCaps:
    """A 15 KB stderr may not become the error string, and nor may a temp path."""

    @pytest.mark.parametrize(
        "provider,adapter,cap",
        [("pi", PI, PI_CAP), ("gemini", GEMINI, GEMINI_CAP)],
    )
    def test_no_message_is_long_multiline_or_noisy(self, provider, adapter, cap):
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            reason = adapter.describe_failure(stdout, stderr)
            if reason is None:
                continue

            assert "\n" not in reason, f"{provider}/{row}: not one line"
            assert len(reason) <= cap, f"{provider}/{row}: over the cap"
            assert "    at " not in reason, f"{provider}/{row}: a Node stack frame"
            assert "/tmp/mllm-inv/" not in reason, f"{provider}/{row}: a temp path"
            assert "node_modules" not in reason, f"{provider}/{row}: a vendor path"

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_the_same_guard_holds_for_parse_outputs_error(self, provider, adapter):
        """The message reaches a user through ``parse_output`` too (Lane B's
        rung 5 and the §6.2 rig), so the cap has to survive that path as well.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            error = str(adapter.parse_output(stdout, stderr).get("error") or "")

            assert "    at " not in error
            assert "/tmp/mllm-inv/" not in error
            assert "\n" not in error

    def test_a_15kb_stderr_is_capped_rather_than_copied(self):
        """The caller keeps the full text; this hook is the summary."""
        _, stderr = _streams("gemini", "rate-limit")
        reason = GEMINI.describe_failure("", stderr)

        assert len(stderr) > 100 * len(reason)


# --------------------------------------------------------------------------- #
# Rung 1 — the channel, in the one place it decides anything
# --------------------------------------------------------------------------- #

class TestRungOneChannel:
    """stdout is the answer and stays on ``raw``; stderr is the diagnosis."""

    def test_geminis_exit_zero_failure_splits_the_two_channels(self):
        """The single capture with bytes on BOTH channels.

        1,174 bytes of stdout whose ``response`` is the model repeating the
        prompt's own confirmation sentence *"JSON written to …"* for a file that
        does not exist, and 193 bytes of stderr carrying the real reason. Rung 1
        is what stops the lie being reported as the diagnosis.
        """
        stdout, stderr = _streams("gemini", "exit-zero-failure")
        assert len(stdout) > 1000 and len(stderr) > 100

        result = GEMINI.parse_output(stdout, stderr)

        assert result["success"] is False
        # The answer channel: still on `raw`, untouched.
        assert "JSON written to" in result["raw"]
        # The diagnosis channel: reaches `error`, and only `error`.
        assert 'Tool "write_file" not found in registry' in result["error"]
        assert "JSON written to" not in result["error"]
        assert result["data"] is None

    @pytest.mark.parametrize(
        "provider,adapter",
        [("gemini", GEMINI)],
    )
    def test_stdout_never_becomes_the_reason(self, provider, adapter):
        """``describe_failure`` reads stderr and nothing else, on every capture."""
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)

            assert adapter.describe_failure(stdout, stderr) == \
                adapter.describe_failure("", stderr)
            assert adapter.describe_failure(stdout, "") is None

    def test_pi_reads_the_stream_first_and_stderr_only_when_it_is_silent(self):
        """pi under ``--mode json`` is ``STDOUT_ELSE_STDERR``, graded per capture.

        Where the stream answers, stderr may not change the text; where it is
        silent (the header-only pre-flight, a clean ``stop``), stderr answers
        alone. The one capture with bytes on stderr is ``preflight-no-key``.
        """
        for row in _capture_rows("pi"):
            stdout, stderr = _streams("pi", row)
            from_stream = PI.describe_failure(stdout, "")
            if from_stream is not None:
                assert PI.describe_failure(stdout, stderr) == from_stream, row
            else:
                assert PI.describe_failure(stdout, stderr) == \
                    PI.describe_failure("", stderr), row

        stdout, stderr = _streams("pi", "preflight-no-key")
        assert PI.describe_failure(stdout, "") is None
        assert PI.describe_failure(stdout, stderr) == "No API key found for stub."

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_a_successful_run_is_byte_for_byte_what_it_was(self, provider, adapter):
        """Only the ``error`` string grows. A success has none, so nothing moves.

        stderr is empty on every success capture in both sets, but the assertion
        is made with a loud stderr anyway: a parser that folded the diagnosis
        into a *successful* result would be a far worse bug than the one C4
        fixes.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            if not adapter.parse_output(stdout, stderr).get("success"):
                continue

            assert adapter.parse_output(stdout, "ApiError: {\"error\":{\"code\":401," \
                "\"message\":\"boom\"}}") == adapter.parse_output(stdout, "")


# --------------------------------------------------------------------------- #
# Rung 5 — both channels empty
# --------------------------------------------------------------------------- #

class TestRungFiveKeepsTodaysString:

    @pytest.mark.parametrize("row", PI_SILENT_ROWS)
    def test_pis_silent_row_is_unembellished(self, row):
        """A whole stream whose model answered nothing: no reason is invented."""
        stdout, stderr = _streams("pi", row)
        assert stderr == "", "this row is 0 bytes on stderr"
        assert '"stopReason":"stop"' in stdout and '"errorMessage"' not in stdout

        assert PI.describe_failure(stdout, stderr) is None
        assert PI.parse_output(stdout, stderr)["error"] == "Empty output"

    def test_the_0806_three_are_no_longer_one_event(self):
        """At 0.80.6 ``context-overflow``, ``empty`` and ``exit-zero-failure``
        were byte-identical (0 bytes, exit 0) — a swallowed overflow, a real
        empty answer and an exit-0 failure were the same event. The 0.80.6
        captures still are; the 0.87.1 ones differ, and so do their verdicts.
        """
        rows = ("context-overflow", "empty", "exit-zero-failure")
        old = {row: _pi_text_mode_streams(row) for row in rows}
        assert set(old.values()) == {("", "")}

        new = {row: _streams("pi", row) for row in rows}
        assert len(set(new.values())) == 3
        assert [PI.classify_failure(*new[row]) for row in rows] == [
            "context_length_exceeded", None, "server_error",
        ]

    @pytest.mark.parametrize("row", PI_SILENT_ROWS)
    def test_the_manifest_records_that_as_the_measurement(self, row):
        """"Not a gap" has to be written where the next reader will look."""
        notes = _manifest("pi")["rows"][row]["notes"]

        assert "rung 5" in notes.lower()
        assert "measurement" in notes.lower()

    def test_gemini_has_one_such_row_and_it_behaves_the_same(self):
        """``ephemeral-only`` is gemini's only 0-byte-stderr failure capture."""
        stdout, stderr = _streams("gemini", "ephemeral-only")
        assert stderr == ""

        assert GEMINI.describe_failure(stdout, stderr) is None
        assert GEMINI.parse_output(stdout, stderr)["error"] == "No valid JSON found in text"


# --------------------------------------------------------------------------- #
# parse_output names the stderr, and the manifests grade it
# --------------------------------------------------------------------------- #

class TestParseOutputNamesTheStderr:
    """The half that covers the exit-0 case and the §6.2 fixture rig."""

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_every_stderr_bearing_failure_row_carries_its_reason(self, provider, adapter):
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            reason = adapter.describe_failure(stdout, stderr)
            result = adapter.parse_output(stdout, stderr)
            if reason is None or result.get("success"):
                continue

            assert reason in result["error"], (
                f"{provider}/{row}: parse_output dropped the stderr diagnosis"
            )

    def test_pis_text_mode_leg_keeps_todays_string_as_the_prefix(self):
        """The 0.80.6 rule, on the back-compat leg that still serves it.

        A pi that ignores ``--mode`` writes text-mode bytes; ``parse_output``
        reads them as 0.80.6's parser did — the stderr is *added*, never
        *substituted*. Graded over the moved-aside 0.80.6 captures, by path.
        """
        rows = sorted({p.name.split(".")[0] for p in PI_TEXT_MODE.glob("*.txt")})
        assert len(rows) == 11
        for row in rows:
            stdout, stderr = _pi_text_mode_streams(row)
            result = PI.parse_output(stdout, stderr)
            if result.get("success"):
                continue
            bare = PI._parse_stdout(stdout)

            assert result["error"].startswith(str(bare["error"])), row
            assert result["raw"] == bare["raw"]
            assert result["data"] == bare["data"]

    @pytest.mark.parametrize(
        "provider,adapter",
        [("gemini", GEMINI)],
    )
    def test_todays_string_is_kept_as_the_prefix(self, provider, adapter):
        """The stderr is *added*, never *substituted*.

        Every existing ``error_contains`` in both manifests keeps holding, and a
        caller matching on the old wording is not broken by the improvement.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            result = adapter.parse_output(stdout, stderr)
            if result.get("success"):
                continue

            bare = adapter._parse_stdout(stdout)

            assert result["error"].startswith(str(bare["error"]))
            assert result["raw"] == bare["raw"]
            assert result["data"] == bare["data"]

    @pytest.mark.parametrize("provider", ["pi", "gemini"])
    def test_the_manifest_names_the_vendors_words_on_every_such_row(self, provider):
        """The improvement is graded by the rig, not asserted here.

        Every failure row with non-empty stderr must declare an
        ``error_contains`` beyond the generic parser string — otherwise the row
        would pass just as happily with the diagnosis thrown away again.
        """
        generic = {"Empty output", "JSON parse error", "No valid JSON found in text"}
        rows = _manifest(provider)["rows"]

        for name, row in rows.items():
            if "not_applicable" in row:
                continue
            _, stderr = _streams(provider, name)
            expect = row["expect"]
            if expect["success"] or not stderr.strip():
                continue

            specific = [n for n in (expect.get("error_contains") or ()) if n not in generic]

            assert specific, (
                f"{provider}/{name} has {len(stderr)} bytes of stderr and no "
                f"error_contains naming the vendor's own words"
            )

    def test_geminis_exit_zero_failure_row_pins_both_channels(self):
        """The Done-when cell: stderr in ``error_contains``, stdout in ``raw``."""
        row = _manifest("gemini")["rows"]["exit-zero-failure"]

        assert 'Tool "write_file" not found in registry' in row["expect"]["error_contains"]
        assert any("JSON written to" in n for n in row["expect"]["raw_contains"])


# --------------------------------------------------------------------------- #
# The rule C4 is bound by
# --------------------------------------------------------------------------- #

class TestNoClassifyCellMoved:
    """``describe_failure`` is diagnostics, never classification."""

    @pytest.mark.parametrize(
        "provider,adapter,expected",
        [("pi", PI, PI_CLASSIFY), ("gemini", GEMINI, GEMINI_CLASSIFY)],
    )
    def test_the_declared_cells_are_the_expected_ones(self, provider, adapter, expected):
        rows = _manifest(provider)["rows"]
        declared = {
            name: row["expect"].get("classify_failure")
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        assert declared == expected

    @pytest.mark.parametrize(
        "provider,adapter,expected",
        [("pi", PI, PI_CLASSIFY), ("gemini", GEMINI, GEMINI_CLASSIFY)],
    )
    def test_the_adapter_agrees_with_them(self, provider, adapter, expected):
        rows = _manifest(provider)["rows"]

        for name, verdict in expected.items():
            stdout, stderr = _streams(provider, name)
            exit_code = rows[name]["exit_code"]

            assert adapter.classify_failure(stdout, stderr, exit_code) == verdict

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_classification_does_not_consult_the_diagnosis_hook(
        self, provider, adapter, monkeypatch
    ):
        """Structural, not incidental: break the hook, keep every verdict.

        ``classify_failure`` reads the raw streams through the declared pattern
        tables. If it ever grew a dependency on the prose built here, a change
        to a *message* could silently move a *category* — which is the one thing
        C4 may not do.
        """
        rows = _manifest(provider)["rows"]
        before = {
            name: adapter.classify_failure(*_streams(provider, name), row["exit_code"])
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        def boom(*_args, **_kwargs):
            raise AssertionError("classify_failure must not call describe_failure")

        monkeypatch.setattr(type(adapter), "describe_failure", boom)
        monkeypatch.setattr(type(adapter), "parse_output", boom)

        after = {
            name: adapter.classify_failure(*_streams(provider, name), row["exit_code"])
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        assert after == before

    def test_gemini_still_declares_no_exit_zero_failure_markers(self):
        """Out of C4's scope, by C4's own rule.

        ``Error executing tool`` and ``not found in registry`` sit on stderr and
        would make a plausible declaration — and declaring them would move
        ``exit-zero-failure``'s ``classify_failure`` cell from ``null`` to
        ``runner_error``. C4 makes the sentence visible; classifying on it is a
        separate change with its own row.
        """
        assert GeminiProvider.exit_zero_failure_markers == ()

        stdout, stderr = _streams("gemini", "exit-zero-failure")

        assert 'not found in registry' in GEMINI.describe_failure(stdout, stderr)
        assert GEMINI.classify_failure(stdout, stderr, 0) is None


class TestNoPiClassifyCellWorsened:
    """T7.6: after the json-mode recapture no row classifies worse than 0.80.6.

    "Grade every capture before and after" (§13.1) applied to pi. Specificity
    is one-directional: a verdict may stay what it was or move off ``None``;
    it may never change from one class to another or fall back to ``None``.
    """

    def test_the_floor_is_the_moved_aside_0806_manifest(self):
        """The literal is the text-mode manifest's own cells, not a memory."""
        rows = yaml.safe_load(
            (PI_TEXT_MODE / "manifest.yaml").read_text(encoding="utf-8")
        )["rows"]
        declared = {
            name: row["expect"].get("classify_failure")
            for name, row in rows.items()
            if "not_applicable" not in row
        }
        assert declared == PI_CLASSIFY_0_80_6

    @pytest.mark.parametrize("row", sorted(PI_CLASSIFY_0_80_6))
    def test_no_recaptured_row_is_less_specific(self, row):
        before = PI_CLASSIFY_0_80_6[row]
        stdout, stderr = _streams("pi", row)
        exit_code = _manifest("pi")["rows"][row]["exit_code"]
        after = PI.classify_failure(stdout, stderr, exit_code)

        if before is not None:
            assert after == before, f"{row}: {before} -> {after}"

    def test_context_overflow_is_no_longer_none(self):
        """The row that was silent at 0.80.6 (C11's recorded non-fix)."""
        stdout, stderr = _streams("pi", "context-overflow")

        assert PI.classify_failure(stdout, stderr, 0) == "context_length_exceeded"

    def test_exactly_the_expected_cells_improved(self):
        improved = {
            row for row, before in PI_CLASSIFY_0_80_6.items()
            if before is None and PI_CLASSIFY[row] is not None
        }
        assert improved == {"context-overflow", "server-error", "exit-zero-failure"}


# --------------------------------------------------------------------------- #
# C3 — gemini's row
# --------------------------------------------------------------------------- #

#: The four adapters C3 widens, by the provider id their fixture dir carries.
#: The repo-wide sweep below derives its patterns from these classes rather than
#: spelling gemini's two out, so muse's ``api key from``, cline's ``user not
#: found`` and claude-code's ``failed to authenticate`` are swept by the same
#: assertion — and so a *fifth* wording added to any of them is swept on the day
#: it lands rather than the day someone remembers to retype it here.
C3_ADAPTERS = {
    "claude-code": ClaudeCodeProvider,
    "cline": ClineProvider,
    "gemini": GeminiProvider,
    "muse": MuseProvider,
}


def _added_auth_patterns(adapter):
    """The auth wordings ``adapter`` declares on top of the base ten."""
    return tuple(
        p for p in adapter.auth_error_patterns
        if p not in LLMProvider.auth_error_patterns
    )


class TestGeminiAuthPatterns:
    """C3: gemini's measured 401 wording, classified as the credential it is."""

    def test_the_measured_401_now_classifies_as_auth(self):
        """Driven by the existing fixture row, not by new bytes."""
        stdout, stderr = _streams("gemini", "auth-error")

        assert GEMINI.classify_failure(stdout, stderr, 145) == "auth_failed"

    def test_the_base_set_is_widened_rather_than_replaced(self):
        """Two patterns, one per measured wording, on top of the ten base ones."""
        added = _added_auth_patterns(GeminiProvider)

        assert GeminiProvider.auth_error_patterns[:len(LLMProvider.auth_error_patterns)] \
            == LLMProvider.auth_error_patterns
        assert added == ("unauthenticated", "incorrect api key")

    @pytest.mark.parametrize("pattern", ["unauthenticated", "incorrect api key"])
    def test_each_added_pattern_is_needed_and_sufficient_on_its_own(self, pattern):
        """Neither is dead weight, and neither is the base set in disguise.

        The 401 carries BOTH wordings — Google's status word and the upstream
        vendor's sentence — so each is checked alone against the capture.
        """
        _, stderr = _streams("gemini", "auth-error")
        haystack = f"\n{stderr}".lower()

        assert matches_any((pattern,), haystack)
        assert not matches_any(LLMProvider.auth_error_patterns, haystack)

    #: Capture-name prefixes the byte sweep below skips, each because carrying
    #: these wordings is the row's WHOLE JOB rather than an accident.
    #:
    #: ``auth-error`` (and ``auth-error-preflight``) is the original: a 401 row
    #: is the one place an auth wording is supposed to appear.
    #:
    #: ``markers-in-tool-result`` joined with omp's fixtures (2026-09-17) and is
    #: the other shape entirely — a hazard this file had no example of before.
    #: The markers sit inside a ``read`` ``tool_execution_end``: our own data
    #: handed back to us, a reviewer that read a plan quoting ``401 User not
    #: found.``, while the run really ended on a 429. Under ``classify_streams``'
    #: fixed order (auth before rate limit) that quoted 401 would outrank the
    #: real 429 and fail the whole batch ``auth_failed`` with no retry, and what
    #: stops it is ``OmpProvider.vendor_text`` reducing the stream to the
    #: terminal message's own sentence before any table is consulted. The
    #: manifest grades exactly that on this row — ``classify_failure:
    #: rate_limited`` over bytes whose ``raw_contains`` names the quoted 401 — so
    #: the claim is asserted, and harder than here. A byte grep is asking a
    #: different question of the same file, and the only way to answer it would
    #: be to delete the evidence.
    _CAPTURES_THAT_MAY_CARRY_THE_WORDS = ("auth-error", "markers-in-tool-result")

    def test_no_success_capture_anywhere_matches_the_new_patterns(self):
        """C1's hazard, checked the way C3 asks: against the ECHOING adapters.

        ``goose``, ``copilot``, ``vibe`` and ``muse`` echo our own prompt back
        inside their JSON envelopes, so a widened auth table is measured against
        their *success* captures as well as their 401s — an echoed
        "authentication required" would otherwise match more wordings on the one
        class that fails a whole batch. The sweep is over every committed
        capture in the repo rather than a roster of four, so a fifth echoing
        adapter is covered on the day its fixtures land.

        Both axes are derived, not typed: the patterns come from each C3
        adapter's own :attr:`auth_error_patterns` delta, so muse's ``api key
        from`` is swept as literally as gemini's ``unauthenticated``. Spelling
        gemini's two out here would have left the other three guarded only
        indirectly, through manifest cells graded after ``vendor_text`` — which
        is the same check one hop removed, and one hop is where a future
        echoing capture gets through.
        """
        added = {
            provider: _added_auth_patterns(adapter)
            for provider, adapter in sorted(C3_ADAPTERS.items())
        }

        assert all(added.values()), (
            f"a C3 adapter declares no added auth wording to sweep: {added}"
        )

        offenders = []

        for path in sorted(FIXTURES.glob("*/*")):
            if path.suffix not in (".txt", ".jsonl", ".json") or path.name == "manifest.yaml":
                continue
            if path.name.startswith(self._CAPTURES_THAT_MAY_CARRY_THE_WORDS):
                continue
            haystack = "\n" + path.read_text(encoding="utf-8", errors="replace").lower()
            for provider, patterns in added.items():
                for pattern in patterns:
                    if matches_any((pattern,), haystack):
                        offenders.append(
                            f"{path.relative_to(FIXTURES)} ~ {provider}:{pattern!r}"
                        )

        assert offenders == [], (
            f"C3's added auth patterns match non-auth captures: {offenders}"
        )

    @pytest.mark.parametrize(
        "row", ["success", "multi-assistant", "ephemeral-only"]
    )
    def test_geminis_own_non_failure_rows_are_unmoved(self, row):
        stdout, stderr = _streams("gemini", row)
        exit_code = _manifest("gemini")["rows"][row]["exit_code"]

        assert GEMINI.classify_failure(stdout, stderr, exit_code) is None

    def test_the_auth_probe_haystack_is_untouched(self):
        """``classify_auth_probe`` shares the pattern set, and should.

        An auth probe never carries our prompt, and a widened set makes the
        credential verdict *more* correct there — the one verdict allowed to
        refuse a run. Pinned so the sharing is deliberate rather than noticed
        later.
        """
        _, stderr = _streams("gemini", "auth-error")

        assert GEMINI.classify_auth_probe("", stderr, 145) is False


# --------------------------------------------------------------------------- #
# C12 — gemini's row
# --------------------------------------------------------------------------- #

class TestGeminiInternalRetryEnvelope:
    """C12: the chain gemini's own docstring measured and stopped short of."""

    def test_the_numbers_are_the_measured_ones(self):
        envelope = GeminiProvider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 10
        assert envelope.max_wall_s == 243.0
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == ()

    def test_the_capture_that_produced_them_is_committed(self):
        """10 requests / 243 s is ``rate-limit``, and the manifest says so."""
        row = _manifest("gemini")["rows"]["rate-limit"]

        assert row["requests"] == GeminiProvider.internal_retries.max_requests
        assert "10 requests over 243 s" in row["notes"]

    def test_declaring_it_removes_the_harnesss_own_rate_limit_backoff(self):
        """The budget consequence, written where a budget reader will find it.

        ``retries_rate_limit: True`` means rule 6(b) gives gemini **no**
        identical rate-limit re-dispatch: a smaller retry budget, not a larger
        one. A slot that used to end after several dispatches — each re-running
        a four-minute chain the CLI had already run — now ends after one.
        """
        from utils.retry_ladder import STOP_VENDOR_ALREADY_RETRIED

        source = Path(__file__).parent.parent / "utils" / "providers" / "gemini.py"
        text = source.read_text(encoding="utf-8")

        assert "REMOVES a retry" in text
        assert "6(b)" in text
        assert STOP_VENDOR_ALREADY_RETRIED


# --------------------------------------------------------------------------- #
# Residual item 4 — kilocode's operator-visible line
# --------------------------------------------------------------------------- #

#: The spec of the slot that produced the capture, kept verbatim so the Lane A
#: assertion below runs through the same ``provider:model`` parse the failing run
#: did — including the ``:thinking`` suffix, which ``split_reasoning_effort``
#: takes off as a ``--variant`` rather than as part of the model id.
KILOCODE_SPEC = "kilocode:openrouter/moonshotai/kimi-k3:thinking"


class TestKilocodeNamesTheAccountFailureInsteadOfItsExitCode:
    """Residual item 4: the hook that turns an exit code back into a sentence.

    The incident, in one line: a `review-code` fan-out slot died on an account
    the configured key no longer belonged to, the CLI said so on stderr in
    colour, and the operator read ``kilocode exited with code 1``. Nothing was
    wrong with the harness — ``base.LLMProvider.describe_failure``'s contract is
    that neither caller prints raw stderr of its own, so with no override there
    was nothing to print. The adapter is the place that knows how to read this
    CLI's stderr, and this class is the mapping pinned the way pi's and gemini's
    are: capture, tier, words, and the two cells that may not move with them.
    """

    @pytest.mark.parametrize("row", sorted(KILOCODE_TIERS))
    def test_every_capture_lands_on_its_declared_tier(self, row):
        """The precedence, graded per row rather than asserted in prose."""
        diagnosis = KILOCODE._diagnose(*_kilocode_streams(row))
        tier = None if diagnosis is None else diagnosis[0]

        assert tier == KILOCODE_TIERS[row]

    def test_the_tier_table_names_every_committed_capture(self):
        """A new row forces a tier decision instead of going ungraded.

        The same rule ``test_the_table_names_every_committed_capture`` applies to
        pi and gemini, read against the literals above rather than against a
        fixture directory, because that is where kilocode's bytes live.
        """
        assert sorted(KILOCODE_TIERS) == sorted(KILOCODE_CAPTURES)

    @pytest.mark.parametrize("row", sorted(KILOCODE_WORDS))
    def test_the_message_carries_the_vendors_own_words(self, row):
        reason = KILOCODE.describe_failure(*_kilocode_streams(row))

        assert reason is not None
        for needle in KILOCODE_WORDS[row]:
            assert needle in reason

    def test_the_operator_visible_line_is_the_one_the_plan_asked_for(self):
        """The close condition, end to end through the lane that failed.

        Not a reconstruction of ``llm_client``'s format string: the real Lane A
        non-zero-exit branch runs, over the captured bytes, and the assertion is
        on the ``error`` a fan-out slot would print. The ``before`` half is
        asserted in the same test with the override swapped back out for the
        base implementation, because "instead of ``kilocode exited with code
        1``" is half of what residual item 4 asks for and a test that only
        checks the new string cannot tell an improvement from a coincidence.
        """
        stdout, stderr = _kilocode_streams("auth-error")

        def _run():
            failed = MagicMock()
            failed.returncode = 1
            failed.stdout = stdout
            failed.stderr = stderr
            with patch("utils.llm_client.subprocess.run", return_value=failed), \
                    patch("shutil.which", return_value="/usr/bin/kilocode"):
                return invoke_with_provider(
                    prompt="Review this diff.", model_spec=KILOCODE_SPEC
                )

        after = _run()
        with patch.object(
            KiloCodeProvider, "describe_failure", LLMProvider.describe_failure
        ):
            before = _run()

        assert before["success"] is False
        assert before["error"] == "kilocode exited with code 1"
        assert after["success"] is False
        assert after["error"] == "kilocode exited with code 1: Error: User not found."
        # The summary is added, never substituted: the full stream stays where
        # the contract says it stays, so nothing that reads `details["stderr"]`
        # or the log file loses a byte to the improvement.
        assert after["details"]["stderr"] == stderr

    def test_colour_never_changes_the_diagnosis(self):
        """A message that depends on the terminal's colour support is not one.

        The painted capture and its unpainted twin are the same failure, so they
        must produce the same sentence — which is what makes the ANSI strip part
        of the diagnosis rather than a cosmetic pass. This is the row pi and
        gemini could not contribute: neither CLI paints its stderr.
        """
        painted = KILOCODE.describe_failure(*_kilocode_streams("auth-error"))
        plain = KILOCODE.describe_failure(*_kilocode_streams("auth-error-no-color"))

        assert painted == plain == "Error: User not found."

    def test_the_run_banner_is_never_promoted_to_a_diagnosis(self):
        """Rung 5, and the row that makes it a rule for this adapter.

        ``> code · moonshotai/kimi-k3`` is printed by a run that succeeds as
        readily as by one that dies. Reported as a reason it would read like a
        diagnosis and be none — the failure mode gemini's dropped stack and
        ``/tmp/mllm-inv/…`` path exist to prevent — so a stderr holding only the
        banner gets ``None``, and no row's message quotes it.
        """
        assert KILOCODE.describe_failure(*_kilocode_streams("banner-only")) is None

        for row in KILOCODE_CAPTURES:
            reason = KILOCODE.describe_failure(*_kilocode_streams(row)) or ""

            assert "moonshotai/kimi-k3" not in reason, f"{row}: the banner leaked"
            assert not reason.startswith(">"), f"{row}: the banner leaked"

    @pytest.mark.parametrize("row", sorted(KILOCODE_CAPTURES))
    def test_the_answer_channel_never_becomes_the_reason(self, row):
        """Rung 1, asserted with a loud stdout every row is given for free.

        kilocode's captured stdout is empty, so the rung decides nothing on the
        bytes alone — and that is exactly when it needs asserting, because a
        future edit reading stdout here would pass every other test in this
        class. The stdout used is fabricated and says so: it is a model's own
        answer, the thing ``parse_output`` exists to read and the last thing a
        failure reason may quote.
        """
        stdout, stderr = _kilocode_streams(row)
        answer = stdout or '[{"title": "A finding", "desc": "…", "importance": "low"}]'

        assert KILOCODE.describe_failure(answer, stderr) == \
            KILOCODE.describe_failure("", stderr)
        assert KILOCODE.describe_failure(answer, "") is None

    def test_no_message_is_long_multiline_or_still_painted(self):
        """Rung 4: one short line, and the escapes really are off it.

        The caller keeps the full stream (``details["stderr"]`` and the log
        file), so this hook is the summary. The ESC-byte assertion is the
        kilocode-specific half: a message that still carried ``\\x1b[91m`` would
        repaint whatever line the operator's own terminal put it on.
        """
        for row in KILOCODE_CAPTURES:
            reason = KILOCODE.describe_failure(*_kilocode_streams(row))
            if reason is None:
                continue

            assert "\n" not in reason, f"{row}: not one line"
            assert len(reason) <= KILOCODE_CAP, f"{row}: over the cap"
            assert "\x1b" not in reason, f"{row}: an escape byte survived"

    def test_a_long_stderr_is_capped_rather_than_copied(self):
        """No committed capture is long; the cap is for the release that is.

        Fabricated on purpose — kilocode's real stderr is 62 bytes, so the only
        honest way to assert the bound is to hand it something that exceeds it
        and say so.
        """
        stderr = "\x1b[91mError: \x1b[0m" + ("retrying the request; " * 800)

        reason = KILOCODE.describe_failure("", stderr)

        assert reason is not None
        assert len(reason) <= KILOCODE_CAP
        assert reason.endswith("…")
        assert len(stderr) > 50 * len(reason)


class TestOmpStripsTheSpinnerAndTheBunTrace:
    """omp's pre-flight channel: three reductions, and what each one is for.

    This is the half of omp's failure reporting that does NOT come off the
    stream. In `--mode json` every failure that reached the backend rides
    in-stream — `stopReason: "error"` with an `errorStatus` and an
    `errorMessage` — and stderr is 0 bytes; `describe_failure` recovers those
    through the shared terminal-verdict reader, and the §6.2 rig grades them
    against committed captures. What is left for stderr is the pre-flight set:
    a bad model, a missing key, a rejected `--tools` name, a missing overlay.
    Those exit 1 with an empty stdout, and `stderr_diagnosis` is the only thing
    standing between them and an operator reading `omp exited with code 1`.
    """

    def test_the_tier_table_names_every_capture(self):
        """A new row forces a rung decision instead of going ungraded — the same
        rule `test_the_table_names_every_committed_capture` applies to pi and
        gemini, read against the literals above because that is where omp's
        pre-flight bytes live."""
        assert sorted(OMP_TIERS) == sorted(set(OMP_WORDS) | {"json-mode-silence"})

    def test_there_is_no_vendor_json_tier_and_that_is_measured(self):
        """The same bytes, read by the two adapters that share `parse_answer_text`.

        pi's second rung parses the `NNN: {json}` body it was written for; omp
        has no such rung, because omp never writes that prefix — so handed pi's
        own `auth-error` capture it reports the line as it stands. Asserting it
        over pi's real bytes is what makes "verbatim tier only" a measurement
        rather than a description of code nobody exercised.
        """
        stderr = _omp_stderr("vendor-json")

        assert PI._diagnose("", stderr)[0] == PiProvider.TIER_VENDOR_JSON
        assert _omp_tier(stderr) == PiProvider.TIER_VERBATIM
        assert omp_stderr_diagnosis(stderr) == " ".join(stderr.split())

    @pytest.mark.parametrize("row", sorted(OMP_DROPPED))
    def test_the_noise_is_dropped_by_name(self, row):
        """Rung 4, omp's two members of it.

        `Working...` is text mode's spinner and precedes the diagnosis there, so
        keeping it would put a progress indicator at the front of every text-mode
        failure. A Bun frame names a byte offset inside a 251 MB single-file
        binary — `at <anonymous> (/$bunfs/root/omp-linux-x64:587578:27)` — which
        is noise to everyone who is not upstream, and on the no-key row there are
        four of them behind one sentence.
        """
        diagnosis = omp_stderr_diagnosis(_omp_stderr(row))

        for needle in OMP_DROPPED[row]:
            assert needle not in diagnosis
        for needle in OMP_WORDS[row]:
            assert needle in diagnosis

    def test_a_frame_is_dropped_for_its_indentation_and_not_for_the_word_at(self):
        """Both frame indentations `_BUN_NOISE` matches, and the negative.

        The rule for the frame arm is the **leading whitespace**, which is what
        keeps a real sentence containing the word "at" from being mistaken for
        one — a diagnosis that lost `… at 18.2.2` would be the mirror-image bug,
        and a sentence that BEGINS `at …` would be lost outright if the arm
        matched at column zero. The prelude arms are different and deliberately
        so: a numbered source line and a caret column are unambiguous wherever
        they sit.
        """
        stderr = (
            "error: something went wrong at startup\n"
            "      at #xo (/$bunfs/root/omp-linux-x64:466354:15)\n"
            "    at write (unknown)\n"
        )

        assert omp_stderr_diagnosis(stderr) == "error: something went wrong at startup"
        # The mirror image: a diagnosis whose first word is "at" is a sentence,
        # not a frame, and the arm's required indentation is what tells them
        # apart at column zero.
        assert omp_stderr_diagnosis("at the third attempt the model gave up\n") == (
            "at the third attempt the model gave up"
        )

    def test_a_silent_channel_is_reported_as_silence(self):
        """Rung 5. In json mode a 0-byte stderr is the NORMAL case — success and
        API failure alike — so nothing may be invented from it, and the stream is
        what speaks instead.
        """
        assert omp_stderr_diagnosis("") == ""
        assert omp_stderr_diagnosis("\n \n\t\n") == ""
        assert OmpProvider().describe_failure("", "") is None

    @pytest.mark.parametrize("row", sorted(OMP_WORDS))
    def test_no_message_is_long_multiline_or_noisy(self, row):
        """Rung 4's bound. The caller keeps the full stream — `details["stderr"]`
        and the log file — so this hook is the SUMMARY, and it reuses pi's own
        `_cap` rather than declaring a second number.
        """
        diagnosis = omp_stderr_diagnosis(_omp_stderr(row))

        assert diagnosis
        assert "\n" not in diagnosis
        assert len(diagnosis) <= PI_CAP
        assert "    at " not in diagnosis

    def test_a_long_stderr_is_capped_rather_than_copied(self):
        """No captured pre-flight stderr is long; the cap is for the release that
        is. Fabricated on purpose, and said so — the only honest way to assert a
        bound none of the real bytes reach.
        """
        stderr = "error: " + ("the overlay could not be read; " * 200)

        diagnosis = omp_stderr_diagnosis(stderr)

        assert len(diagnosis) <= PI_CAP
        assert diagnosis.endswith("…")
        assert len(stderr) > 20 * len(diagnosis)

    @pytest.mark.parametrize("row", sorted(OMP_WORDS))
    def test_the_answer_channel_never_becomes_the_reason(self, row):
        """Rung 1, asserted with a loud stdout every row is given for free.

        omp's pre-flight stdout is empty, so the rung decides nothing on the
        bytes alone — and that is exactly when it needs asserting. The stdout
        used is a `stop` stream carrying a findings array, which is the thing
        `parse_output` exists to read and the last thing a failure reason may
        quote; `describe_failure` must answer with the stderr diagnosis, because
        a `stop` stream carries no verdict of its own.
        """
        stderr = _omp_stderr(row)
        answer = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": '[{"title": "A finding", "desc": "…", '
                                    '"importance": "low"}]',
                        }
                    ],
                    "stopReason": "stop",
                },
            }
        )

        assert omp_stderr_diagnosis(stderr) == omp_stderr_diagnosis(stderr)
        assert OmpProvider().describe_failure(answer, stderr) == \
            omp_stderr_diagnosis(stderr)
        assert OmpProvider().describe_failure(answer, "") is None


class TestNoKilocodeClassifyCellMoved:
    """``describe_failure`` is diagnostics, never classification — here too."""

    @pytest.mark.parametrize("row", sorted(KILOCODE_CLASSIFY))
    def test_the_declared_cells_are_the_expected_ones(self, row):
        """Every row classifies as ``None``, and that is a decision.

        ``User not found.`` is the wording ``providers/cline`` declares in its
        ``auth_error_patterns``, because cline measured it coming out of its own
        CLI. kilocode declares nothing of the sort and this change did not add
        it: the capture is an auth failure that the classifier does not call one,
        so no rung moves, no retry decision moves, and no ``[needs-auth]`` tag is
        appended to the line the previous class pins. Widening the table is a
        separate change with its own measurement.
        """
        stdout, stderr = _kilocode_streams(row)

        assert KILOCODE.classify_failure(stdout, stderr, 1) == KILOCODE_CLASSIFY[row]

    def test_classification_does_not_consult_the_diagnosis_hook(self, monkeypatch):
        """Structural, not incidental: break the hook, keep every verdict.

        The same assertion pi and gemini get. If ``classify_failure`` ever grew a
        dependency on the prose built by ``describe_failure``, a change to a
        *message* could silently move a *category*.
        """
        before = {
            row: KILOCODE.classify_failure(*_kilocode_streams(row), 1)
            for row in KILOCODE_CAPTURES
        }

        def boom(*_args, **_kwargs):
            raise AssertionError("classify_failure must not call describe_failure")

        monkeypatch.setattr(KiloCodeProvider, "describe_failure", boom)
        monkeypatch.setattr(KiloCodeProvider, "_diagnose", boom)

        after = {
            row: KILOCODE.classify_failure(*_kilocode_streams(row), 1)
            for row in KILOCODE_CAPTURES
        }

        assert after == before

    def test_a_successful_run_is_byte_for_byte_what_it_was(self):
        """Only a failure grew a sentence. ``parse_output`` did not move at all.

        kilocode's parse path was left alone on purpose — both lanes reach the
        new hook, and pi's and gemini's matching ``parse_output`` clauses exist
        to satisfy a §6.2 fixture rig kilocode has no manifest for. Asserted with
        a loud stderr rather than the captured empty one, because a parser that
        folded the diagnosis into a *successful* result would be a far worse bug
        than the one being fixed.
        """
        stdout, _ = _kilocode_streams("clean-run")

        assert KILOCODE.parse_output(stdout, KILOCODE_AUTH_STDERR) == \
            KILOCODE.parse_output(stdout, "")
        assert KILOCODE.parse_output(stdout, KILOCODE_AUTH_STDERR)["success"] is True
