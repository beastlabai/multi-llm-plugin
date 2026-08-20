"""Base protocol for LLM CLI providers."""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Pattern, Tuple

# The status vocabulary is the retry ladder's, never a second copy of it (§7.2).
# The dependency runs one way only: this module may import the ladder, and the
# ladder must stay free of provider imports — it reads an adapter's retry
# envelope *structurally*, through a Protocol, for exactly that reason.
try:  # pragma: no cover - import shim mirrors the rest of utils/
    from ..retry_ladder import (
        STATUS_AUTH_FAILED,
        STATUS_CONTEXT_LENGTH_EXCEEDED,
        STATUS_RATE_LIMITED,
        STATUS_RUNNER_ERROR,
        STATUS_SPAWN_FAILED,
        STATUS_TIMEOUT,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils.retry_ladder import (
        STATUS_AUTH_FAILED,
        STATUS_CONTEXT_LENGTH_EXCEEDED,
        STATUS_RATE_LIMITED,
        STATUS_RUNNER_ERROR,
        STATUS_SPAWN_FAILED,
        STATUS_TIMEOUT,
    )

#: The three invocation profiles (D3e). ``generate`` is today's callee path (no
#: tools, no workspace); ``agent-readonly`` is an agentic invocation with writes
#: denied by the runner's own sandbox/permission mechanism and is what every
#: read/analyze subagent runs under; ``agent-write`` is the implement/apply path.
PROFILE_GENERATE = "generate"
PROFILE_AGENT_READONLY = "agent-readonly"
PROFILE_AGENT_WRITE = "agent-write"

# ---------------------------------------------------------------------------
# Non-interactive git env (§2.2)
# ---------------------------------------------------------------------------

#: ``git_env.credential_helper`` — the closed policy set. ``auto`` is the shipped
#: scoping (:data:`HELPER_WIPE_PROFILES`); ``wipe`` and ``keep`` are the two
#: escape hatches, settable globally, per provider, or through
#: ``MULTI_LLM_GIT_CREDENTIAL_HELPER`` for one run.
GIT_HELPER_AUTO = "auto"
GIT_HELPER_WIPE = "wipe"
GIT_HELPER_KEEP = "keep"
GIT_HELPER_POLICIES: Tuple[str, ...] = (
    GIT_HELPER_AUTO,
    GIT_HELPER_WIPE,
    GIT_HELPER_KEEP,
)

#: The profiles that get ``credential.helper=""`` under the ``auto`` policy: the
#: ``generate`` callee path (which is the Lane A fan-out) and ``agent-readonly``.
#: ``agent-write`` is deliberately **absent** — see :func:`git_env`.
HELPER_WIPE_PROFILES = frozenset({PROFILE_GENERATE, PROFILE_AGENT_READONLY})

#: The measured, load-bearing half, applied to **every** provider and **both**
#: lanes. ``GIT_TERMINAL_PROMPT=0`` alone turns the 75.1 s tty-prompt hang into a
#: 0.253 s failure; ``GIT_ASKPASS=""`` is read by git as *no askpass* (never
#: exec'd) and uniquely short-circuits the whole ``GIT_ASKPASS → core.askPass →
#: SSH_ASKPASS`` chain, which ``GIT_TERMINAL_PROMPT`` does not. Neither suppresses
#: a credential helper that *answers*, so authentication that works today keeps
#: working everywhere.
#:
#: The last three were measured **inert on Linux** and are **untested on Windows
#: and macOS**: carry them as cheap hedges, never as a claim about those
#: platforms (verification is owned by ``plans/windows-macos-portability.md``).
#: ``credential.interactive`` / ``credential.guiPrompt`` are deliberately absent —
#: they are not git config keys at all (absent from ``git-config(1)`` and from the
#: binary's strings; they are Git Credential Manager settings), and so is
#: ``LC_ALL=C``: every git call in this repo uses ``--porcelain``/``rev-parse``/
#: ``diff``/``show`` and none passes ``env=``, so this block never reaches them.
GIT_ENV_GLOBAL: Dict[str, str] = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "",
    "SSH_ASKPASS": "",
    "SSH_ASKPASS_REQUIRE": "never",
    "GCM_INTERACTIVE": "never",
}


def wipes_credential_helper(
    profile: str, credential_helper: str = GIT_HELPER_AUTO
) -> bool:
    """Whether this dispatch gets ``credential.helper=""`` (§2.2).

    ``auto`` is the scoping decision, not a guess: the wipe is the only measured
    cure for a *hanging* helper (90.1 s and an orphan → 0.27 s and no child), but
    it cannot tell a helper that hangs from one that works, so it equally
    suppresses ``store``, an OS keychain and ``gh auth git-credential``. It is
    therefore applied only where a private-remote credential has no business
    being needed at all.
    """
    if credential_helper == GIT_HELPER_KEEP:
        return False
    if credential_helper == GIT_HELPER_WIPE:
        return True
    return profile in HELPER_WIPE_PROFILES


def git_env(
    profile: str = PROFILE_GENERATE,
    *,
    credential_helper: str = GIT_HELPER_AUTO,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The non-interactive git env delta for one dispatch (§2.2).

    Every provider shells out to git under an agent profile, and a credential
    prompt behind ``capture_output=True`` blocks forever: measured at **75.1 s
    with two empty streams** for a tty prompt and **90.1 s plus an orphaned
    helper** for a GUI-style credential helper, both against a private https
    remote under a scratch ``HOME``. ``stdin=DEVNULL`` does not help — git prompts
    on ``/dev/tty``.

    Two layers, and the layering is the whole design:

    * :data:`GIT_ENV_GLOBAL` — unconditional, every provider, both lanes.
    * ``GIT_CONFIG_COUNT`` + ``credential.helper=""`` — **scoped** to
      :data:`HELPER_WIPE_PROFILES` (``generate``, i.e. the Lane A fan-out, and
      ``agent-readonly``).

    **The wipe must never be globalised onto ``agent-write``.** That profile
    really does run ``git commit``/``status``/``clone``/``fetch`` against the
    user's own remotes, and an empty helper list would fail those closed — the
    one profile that legitimately needs a private-remote credential. Its
    protection against a hanging helper is the process-group kill plus the
    per-task budget: a *bound*, not a fast failure. A later "unify the git env"
    refactor that moves these three keys out of the ``profile`` branch and into
    the unconditional block is a behaviour regression, not a cleanup.

    **Accepted trade-off, in the scoped lanes only.** A private-HTTPS
    ``clone``/``fetch`` that works today returns in ~0.27 s with ``fatal: could
    not read Username for 'https://…': terminal prompts disabled``; the helper is
    never invoked and leaves no orphan. That is a fast, legible failure rather
    than a hang or a silent wrong answer, and those lanes are Q&A and analysis
    over a workspace that is already checked out.
    ``git_env.credential_helper: keep`` restores today's behaviour for them.

    ``base_env`` (default ``os.environ``, which is what both spawn sites merge
    this delta over) is read only to find an existing ``GIT_CONFIG_COUNT``: a
    caller's own injected keys are preserved and ours is **appended last**,
    because for a multi-valued key an empty value *resets the list*, so it only
    suppresses helpers declared before it.
    """
    env = dict(GIT_ENV_GLOBAL)
    if not wipes_credential_helper(profile, credential_helper):
        return env
    source = os.environ if base_env is None else base_env
    try:
        index = int(str(source.get("GIT_CONFIG_COUNT", "") or "0"))
    except (TypeError, ValueError):
        index = 0
    index = max(index, 0)
    env["GIT_CONFIG_COUNT"] = str(index + 1)
    env[f"GIT_CONFIG_KEY_{index}"] = "credential.helper"
    env[f"GIT_CONFIG_VALUE_{index}"] = ""
    return env

# ─────────────────────────────────────────────────────────────────────────────
# §4.2.1 — the structured-output retry budget, pinned in the child env.
#
# It lives HERE rather than in ``utils/structured_output`` because an adapter's
# ``get_env`` is what emits it and an adapter may not import that module (it
# imports this one, and the reverse edge would be a cycle through
# ``utils.providers.__init__``). ``utils.structured_output`` re-exports both
# names, so there is one definition and no second copy of the number.
# ─────────────────────────────────────────────────────────────────────────────

#: The CLI's default is **5**, i.e. five *paid* model calls per invocation,
#: multiplied by the fan-out width and again by our own retry ladder. Set
#: explicitly so a vendor default change cannot silently re-inflate the bill.
#:
#: Measured 2026-08-16 (claude 2.1.233 at a localhost stub, zero paid tokens):
#: the variable caps **total** structured-output attempts, not repair calls on
#: top of the first one — ``=1`` produced exactly 1 request against a stub whose
#: tool input never validated, where the default produced 5.
MAX_STRUCTURED_OUTPUT_RETRIES = 1
STRUCTURED_OUTPUT_RETRY_ENV = "MAX_STRUCTURED_OUTPUT_RETRIES"

#: Declared usage-extraction fidelity (D3j), mirrored by ``runners[*].usage`` in
#: the D5b capability data. ``none`` is the honest default: an adapter with no
#: fixture-verified extractor is duration-only, never a promise of token counts.
USAGE_NONE = "none"
USAGE_PARTIAL = "partial"
USAGE_FULL = "full"


@dataclass(frozen=True)
class InternalRetryEnvelope:
    """What one vendor CLI does *inside* one awaited subprocess (§7.6(b)).

    §7.5's arithmetic needs one number per adapter, and before this declaration
    that number existed only in a plan's measurements. The retry ladder subtracts
    it in two places:

    * **rule 4** — an attempt is admitted only when what is left of the slot's
      deadline can hold one worst-case :attr:`max_wall_s`; otherwise the slot
      ends ``budget_exhausted`` with **no** child spawned, rather than a child
      being killed part-way through a retry chain the harness cannot see.
    * **rule 6(b)** — an adapter whose CLI already retries 429s internally
      (:attr:`retries_rate_limit`) gets **no** identical rate-limit re-dispatch:
      that would repeat a chain that already ran :attr:`max_requests` times
      against the same upstream, at the same price, for the same verdict.

    Every field is **measured, never inferred**. An adapter nobody measured
    declares ``None`` (:attr:`LLMProvider.internal_retries`), and §7.5 rule 6(c)
    reads that as *assume one request* — never *assume the worst*, which would
    fail slots that would have succeeded.

    The ladder reads :attr:`max_requests`, :attr:`max_wall_s` and
    :attr:`retries_rate_limit` **structurally** (``retry_ladder``'s
    ``InternalRetryEnvelopeLike`` Protocol), so this module is not importable
    from there and must not become so: the policy object stays pure and
    provider-free, and the structural read is the seam.
    """

    #: Requests the CLI may make to the upstream API inside one dispatch, at the
    #: configuration this harness actually runs (i.e. *including* whatever
    #: ``get_env`` pins). Never the CLI's out-of-the-box number when the adapter
    #: overrides it.
    max_requests: int

    #: The worst-case wall clock one dispatch may burn *inside* the child, in
    #: seconds. ``None`` means "no bound this adapter can state": rule 4 then
    #: performs no admission test, which is the honest posture — inventing a
    #: bound would refuse attempts that had budget for themselves.
    max_wall_s: Optional[float] = None

    #: Whether the CLI retries an HTTP 429 itself. ``False`` means the harness's
    #: own backoff is the only one a rate limit gets.
    retries_rate_limit: bool = False

    #: The env vars or config keys that change the numbers above — documentation
    #: for the human reading a budget, not a control surface. Empty means nothing
    #: reachable from here changes them.
    configurable_by: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredSchema:
    """One schema on its way to a CLI — the object, and where it was written.

    §4.2.2 asks for **one optional parameter and two transports**, and the two
    transports need different things: claude takes the schema *inline*
    (``--json-schema <json>``) while codex takes a *path*
    (``--output-schema <FILE>``). Handing the adapter a bare ``dict`` would serve
    only the first; handing it a bare path would serve only the second and would
    make the argv fixtures unreadable. So the caller hands over both halves in
    one value and each adapter picks the half it can carry.

    :attr:`path` is filled in by the **caller**, never by the adapter: adapters
    are module-level singletons in ``_PROVIDERS`` shared across a concurrent
    fan-out and cannot hold per-run state (the constraint that sank §4.1's T9).
    ``utils.structured_output.schema_sidecar`` is the one writer, and it removes
    the directory in a ``finally``.

    :attr:`name` is the schema **file** name. It is carried so every refusal and
    every WARN can name the file a human has to go and fix, rather than printing
    a schema body.
    """

    document: Dict[str, Any]
    path: Optional[str] = None
    name: str = "<inline>"

    def with_path(self, path: str) -> "StructuredSchema":
        """The same schema, now materialized at ``path``."""
        return StructuredSchema(document=self.document, path=path, name=self.name)


class StructuredOutputUnavailable(Exception):
    """A schema was handed to an adapter that cannot transport it (§4.2.1).

    Raised **before spawn**, so no partial argv reaches the CLI — §4.2.1's sixth
    acceptance case ("missing sidecar / schema file (codex)"). It is a
    configuration error, never a retryable failure: nothing about re-running the
    same dispatch would produce the missing file.
    """

    exit_code = 3

    def __init__(self, message: str, provider: str = "", schema: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.schema = schema


class ProfileUnsupported(Exception):
    """This adapter cannot honor the requested invocation profile (D3e).

    Raised by :meth:`LLMProvider.build_agent_command`'s default implementation,
    so an adapter with no genuinely read-only (or edit-capable) mode is refused
    at dispatch with the adapter and profile named rather than being handed a
    fabricated command. ``utils/subagent_executor`` re-exports this name and
    classifies it as D3k rung 7 (``rejected``, no child started).
    """

    exit_code = 3

    def __init__(self, message: str, provider: str = "", profile: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.profile = profile


def split_reasoning_effort(model: str, efforts: Collection[str]) -> Tuple[str, Optional[str]]:
    """Split an optional trailing ``:effort`` suffix off a model string.

    Returns ``(base_model, effort)`` when the substring after the LAST colon
    is a member of ``efforts`` and the part before it is non-empty; otherwise
    ``(model, None)`` and the model string passes through verbatim. Splitting
    on the last colon keeps model ids that legitimately contain colons (e.g.
    openrouter ``:free`` variants) intact unless the suffix is whitelisted.
    """
    base, _, suffix = model.rpartition(":")
    if base and suffix in efforts:
        return base, suffix
    return model, None


#: Terms that make a *bare* token a real failure signal rather than a
#: coincidence. A one-word pattern ("429", "unauthorized") is required to occur
#: near one of these — outside its own span — because on its own it appears in
#: places that say nothing about the run: opencode prints per-step costs
#: (``"cost":0.0429``), every runner prints token counts (``14290``), and a
#: review answer quotes the plan and diff under review ("the unauthorized branch
#: returns early").
_ERROR_CONTEXT = re.compile(
    r"http|status|code|error|err\b|response|fail|denied|forbidden|reject|"
    r"invalid|expired|exceeded|missing|quota|retry|throttl|too\s*many|rate|"
    r"limit|credential|login|log\s*in|sign\s*in|api[\s_-]*key|401|403|429|529"
)

#: How far either side of a bare-token match :data:`_ERROR_CONTEXT` is looked for.
_CONTEXT_WINDOW = 48

#: Compiled-pattern cache: the classifier runs on every failed dispatch.
_STREAM_PATTERNS: Dict[str, "re.Pattern[str]"] = {}


def stream_pattern(pattern: str) -> "re.Pattern[str]":
    """Compile one classifier substring into a boundary-aware regex.

    A textual pattern keeps a left word boundary while staying a prefix match on
    the right — "rate limit" must still recognize "rate limits" and
    "rate-limited" — and any run of separators inside the pattern matches any run
    of separators in the stream, so one entry covers ``rate limit``,
    ``rate-limit`` and ``ratelimit`` alike. A numeric pattern (an HTTP status
    code) must additionally stand alone as a token, so ``429`` is not found
    inside ``0.0429`` or ``14290``.

    Patterns are matched against a lowercased haystack, so the compiled regex is
    lowercase and carries no ``IGNORECASE`` flag.
    """
    token = (pattern or "").strip().lower()
    cached = _STREAM_PATTERNS.get(token)
    if cached is not None:
        return cached
    if token.isdigit():
        regex = re.compile(rf"(?<![\w.]){token}(?![\w.])")
    else:
        body = re.sub(r"(?:\\?[ _\-])+", r"[\\s_\\-]*", re.escape(token))
        regex = re.compile(rf"(?<![a-z0-9]){body}")
    _STREAM_PATTERNS[token] = regex
    return regex


def _needs_context(token: str) -> bool:
    """Whether ``token`` is too bare to classify a run on its own.

    A multi-word phrase ("not logged in", "too many requests", "quota_exceeded")
    is specific enough to stand alone; a single token ("429", "unauthorized",
    "overloaded") is not, because it occurs inside numbers and inside the prose a
    runner prints about somebody else's code.
    """
    return re.search(r"[\s_\-]", token) is None


def matches_any(patterns: Collection[str], haystack: str) -> bool:
    """Whether any declared classifier pattern really occurs in ``haystack``.

    This is the whole difference between the executor classifying a run and
    guessing at one: the D3k ladder consults
    :meth:`LLMProvider.classify_failure` for a run with no usable result, and a
    bare ``pattern in haystack`` test moves such a run onto a rung with different
    retry rules than the one it belongs on — ``rate_limited`` is retried *without*
    consuming the retry budget, ``auth_failed`` fails the entire batch.
    """
    for pattern in patterns:
        token = (pattern or "").strip().lower()
        if not token:
            continue
        regex = stream_pattern(token)
        bare = _needs_context(token)
        for match in regex.finditer(haystack):
            if not bare:
                return True
            window = "\n".join((
                haystack[max(0, match.start() - _CONTEXT_WINDOW):match.start()],
                haystack[match.end():match.end() + _CONTEXT_WINDOW],
            ))
            if _ERROR_CONTEXT.search(window):
                return True
    return False


# ---------------------------------------------------------------------------
# The one failure classifier (§7.2, §7.1.a)
#
# Everything below is *new* data for categories the adapter pattern sets never
# covered. The two declared sets — ``auth_error_patterns`` and
# ``rate_limit_patterns`` — and the matcher above are **not** touched: they were
# audited on 2026-08-16 and measured clean (``stream_pattern`` gives a numeric
# token its own boundaries, and a bare token additionally needs an
# ``_ERROR_CONTEXT`` word within 48 characters), with
# ``tests/test_providers.py::TestClassifyFailure`` pinning it. This section adds
# categories beside them, and then makes every other classifier in the tree call
# this one instead of growing its own.
#
# The new tables are **regexes**, not substrings, because §7.2 asks for regexes
# ("maximum context length is N tokens") and because a regex states its own
# boundaries: ``\bthrottl(?:ed|ing)\b`` needs no context window to be safe,
# whereas the bare token ``throttled`` handed to ``matches_any`` would.
# ---------------------------------------------------------------------------

#: Categories the ladder has no status for. They exist so the ``Reason:`` line a
#: user acts on and the retry decision can be read off the *same* verdict —
#: §7.1.a's seven-branch ``elif`` chain is a lookup on these.
CATEGORY_CONNECTION_ERROR = "connection_error"
CATEGORY_SERVER_ERROR = "server_error"
CATEGORY_INVALID_MODEL = "invalid_model"
CATEGORY_PERMISSION_DENIED = "permission_denied"

#: §7.2's retryable-code allowlist, shared across both harnesses. Auth is
#: deliberately absent (no retry fixes a logged-out CLI) and so is
#: :data:`STATUS_CONTEXT_LENGTH_EXCEEDED`: until §7.6(a) ships a prompt splitter,
#: re-dispatching an oversized prompt unchanged spends budget to reproduce the
#: same error. Open question 13 owns the splitter; this frozenset owns the fact
#: that nothing may retry it in the meantime.
RETRYABLE_FAILURE_CATEGORIES = frozenset({
    STATUS_RATE_LIMITED,
    STATUS_TIMEOUT,
    CATEGORY_CONNECTION_ERROR,
    CATEGORY_SERVER_ERROR,
})

#: A shell reports "could not execute what you asked for" as 127, on every
#: POSIX shell and through every wrapper script. It is the one launch verdict
#: that needs no text at all.
MISSING_BINARY_EXIT_CODE = 127


def _markers(*patterns: str) -> Tuple["Pattern[str]", ...]:
    """Compile one classifier table. Haystacks are lowercased before matching."""
    return tuple(re.compile(p) for p in patterns)


def _matches_marker(markers: Collection["Pattern[str]"], haystack: str) -> bool:
    return any(marker.search(haystack) for marker in markers)


#: Root-cause launch failures, ordered **before** the auth patterns (§7.2). A
#: sandbox that refused the exec, or a shim that could not find the binary,
#: prints text that reads like a permission or credential problem — and reading
#: it as ``auth_failed`` is expensive twice over, because rung 1 fails the whole
#: batch and because the actual fix (install the CLI, grant the exec) is never
#: mentioned. Each tuple is ``(category, markers)`` and the first match wins.
#:
#: The markers are deliberately specific phrases rather than the bare
#: ``'not found'`` Lane A used: that one called **every** provider's failure
#: "cursor-agent command not found", and it swallowed "model not found" too.
LAUNCH_FAILURE_MARKERS: Tuple[Tuple[str, Tuple["Pattern[str]", ...]], ...] = (
    (STATUS_SPAWN_FAILED, _markers(
        r"command not found",
        r"executable file not found",
        r"\bbinary not found\b",
        r"not found in \$?path\b",
        r"is not recognized as an internal or external command",
        r"cannot execute binary file",
        r"\bexec format error\b",
    )),
    (CATEGORY_PERMISSION_DENIED, _markers(
        r"permission denied",
        r"operation not permitted",
        r"\beacces\b",
        r"\beperm\b",
    )),
)

#: Per-vendor context-overflow shapes (§7.2). These are the *only* thing that
#: may produce :data:`STATUS_CONTEXT_LENGTH_EXCEEDED`, because that status is
#: terminal: a false positive fails a slot that a retry would have completed.
#: Lane A's ``'context' in str(error)`` matched ``context_mode``, "in this
#: context" and any prose quoting a plan's Context section; every regex here
#: carries the numbers or the verb that make it a real overflow report.
CONTEXT_OVERFLOW_MARKERS = _markers(
    # OpenAI: "This model's maximum context length is 128000 tokens. However,
    # your messages resulted in 214537 tokens."
    r"maximum context length",
    r"\bcontext[_ ]length[_ ]exceeded\b",
    r"reduce the length of the messages",
    # Anthropic: "prompt is too long: 248113 tokens > 200000 maximum"
    r"prompt is too long",
    # Gemini: "The input token count (1093122) exceeds the maximum number of
    # tokens allowed (1048576)."
    r"input token count\s*\(?\d+\)?\s*exceeds",
    # The shapes several gateways normalise the three above onto.
    r"exceeds the (?:model'?s )?(?:maximum )?context (?:window|length)",
    r"context window (?:of \d+ tokens )?exceeded",
    r"\btoo many tokens\b",
    # Mistral Vibe, measured at 2.24.1: "The conversation context exceeds the
    # model's maximum limit. Use /rewind ...". The row above does not reach it --
    # vibe puts the noun first and calls the bound a "limit" rather than a
    # "window" or a "length" -- and without this the run classifies as
    # transport-shaped and burns the executor's whole bounded-retry path on a
    # condition no retry can fix. "conversation context" is what keeps it off
    # prose: an answer discussing a plan's Context section does not say it.
    r"conversation context exceeds",
    # aider, measured at 0.86.2 on the 2026-08-19 fixture capture: it SWALLOWS
    # litellm's ContextWindowExceededError and prints its own help instead, so
    # not one marker above appears anywhere in the run --
    #     "Model openai/stub-model has hit a token limit!"
    #     "Input tokens: ~103 of 0 -- possibly exhausted context window!"
    # The upstream 400's wording never reaches stdout at all. Without these two
    # the row classified as None, i.e. transport-shaped, and the executor spent
    # its whole bounded-retry budget on the one condition §7.2 calls terminal.
    # Both phrasings are specific enough to stay off prose: an answer discussing
    # token budgets does not say "has hit a token limit" or "exhausted context
    # window", and `_markers` anchors on the error context regardless.
    r"hit a token limit",
    r"exhausted context window",
)

#: Rate-limit shapes the adapters' declared English set does not carry, kept
#: here rather than added to it: a bare ``throttled`` handed to
#: :func:`matches_any` would need an ``_ERROR_CONTEXT`` word within 48
#: characters, and the measured phrasing ("Request was throttled") has none. A
#: regex states its own boundary and needs no window — which is why this is a
#: separate table and not an edit to the audited-clean adapter set.
SHARED_RATE_LIMIT_MARKERS = _markers(
    r"\bthrottl(?:ed|ing)\b",
    r"\bquota[ _-]?limit\b",
    r"\busage limit (?:reached|exceeded)\b",
)

#: A wall-clock cutoff, wherever it was imposed. Checked before the connection
#: table on purpose: "connection timed out" is a timeout, and
#: ``validation.py``'s persisted vocabulary has always classified it that way.
TIMEOUT_MARKERS = _markers(
    r"\btimed out\b",
    r"\btime[ _-]?out\b",
    r"deadline exceeded",
    r"\betimedout\b",
)

CONNECTION_ERROR_MARKERS = _markers(
    r"connection (?:refused|reset|closed|aborted|error|failed)",
    r"could not connect",
    r"unable to connect",
    r"failed to connect",
    r"network is unreachable",
    r"no route to host",
    r"temporary failure in name resolution",
    r"name or service not known",
    r"\bgetaddrinfo\b",
    r"\becconnrefused\b",
    r"\benotfound\b",
    r"\bdns\b.*\bfail",
)

SERVER_ERROR_MARKERS = _markers(
    r"\b5\d{2}\b\s*(?:internal|server|bad gateway|service|gateway)",
    r"internal server error",
    r"service unavailable",
    r"bad gateway",
    r"gateway time",
    r"\bupstream error\b",
    r"\bapi error\b",
    r"\bserver error\b",
)

#: "The model you named does not exist / is not available to this account."
#: Its own category because the fix is a config edit, never a retry.
INVALID_MODEL_MARKERS = _markers(
    r"model not found",
    r"\bmodel_not_found\b",
    r"invalid model",
    r"unknown model",
    r"\bmodel .{0,40}(?:does not exist|is not available)",
    r"unsupported model",
)


def classify_streams(
    stdout: str,
    stderr: str,
    exit_code: int = 0,
    *,
    auth_patterns: Optional[Collection[str]] = None,
    rate_limit_patterns: Optional[Collection[str]] = None,
    exit_zero_markers: Collection["Pattern[str]"] = (),
) -> Optional[str]:
    """Classify one captured run. **The** implementation, for both lanes (§7.2).

    Returns a status from :mod:`utils.retry_ladder`'s vocabulary, one of the
    ``CATEGORY_*`` names above, or ``None`` for "nothing recognized" — which the
    caller must read as *transport-shaped*, never as "fine".

    The order is the whole design, and each step earns its place:

    1. ``exit_code == 127`` — the shell's own "I could not run that".
    2. :data:`LAUNCH_FAILURE_MARKERS` — root cause **before** auth (§7.2), since
       a refused exec prints text that reads like a credential problem.
    3. :data:`CONTEXT_OVERFLOW_MARKERS` — terminal and unambiguous, and it must
       not be laundered into a retryable category by a later table.
    4. auth, then rate limit — the existing order, pinned by
       ``TestClassifyFailure::test_auth_wins_over_a_rate_limit_in_the_same_stream``:
       misreading a logged-out CLI as a rate limit retries what cannot be fixed.
    5. timeout, connection, server, invalid model — the rest of the allowlist.
    6. the adapter's own exit-0 marker, last: it says only "this run failed",
       so anything more specific must have had its chance first.

    ``auth_patterns``/``rate_limit_patterns`` default to :class:`LLMProvider`'s
    declared sets, which is what lets a caller with no adapter in hand — Lane A's
    ``_format_failure_details``, ``validation.py`` — get the same verdict as one
    with an adapter. Both are matched through :func:`matches_any`, never as raw
    substrings.
    """
    haystack = f"{stdout or ''}\n{stderr or ''}".lower()

    if exit_code == MISSING_BINARY_EXIT_CODE:
        return STATUS_SPAWN_FAILED

    if not haystack.strip():
        return None

    for category, markers in LAUNCH_FAILURE_MARKERS:
        if _matches_marker(markers, haystack):
            return category

    if _matches_marker(CONTEXT_OVERFLOW_MARKERS, haystack):
        return STATUS_CONTEXT_LENGTH_EXCEEDED

    auth = (
        LLMProvider.auth_error_patterns if auth_patterns is None else auth_patterns
    )
    if matches_any(auth, haystack):
        return STATUS_AUTH_FAILED

    rate = (
        LLMProvider.rate_limit_patterns
        if rate_limit_patterns is None
        else rate_limit_patterns
    )
    if matches_any(rate, haystack) or _matches_marker(
        SHARED_RATE_LIMIT_MARKERS, haystack
    ):
        return STATUS_RATE_LIMITED

    if _matches_marker(TIMEOUT_MARKERS, haystack):
        return STATUS_TIMEOUT
    if _matches_marker(CONNECTION_ERROR_MARKERS, haystack):
        return CATEGORY_CONNECTION_ERROR
    if _matches_marker(SERVER_ERROR_MARKERS, haystack):
        return CATEGORY_SERVER_ERROR
    if _matches_marker(INVALID_MODEL_MARKERS, haystack):
        return CATEGORY_INVALID_MODEL

    if _matches_marker(exit_zero_markers, haystack):
        # The exit code carried no verdict (goose, aider, cursor-agent and cline
        # all exit 0 on a provider failure), and the adapter's own marker is the
        # only thing that says this run failed at all.
        return STATUS_RUNNER_ERROR

    return None


def context_length_message(model: str) -> str:
    """The §7.6(a) sentence a slot fails with, recorded verbatim in the report.

    Not a placeholder for the splitter: open question 13 owns that design, and
    until it is answered the shipping behaviour is to classify the condition,
    name it, and fail the slot with the two remedies a human can actually apply.
    """
    return (
        f"the prompt exceeded {model or 'this model'}'s context window; narrow "
        f"the plan/diff scope or configure a larger-context model"
    )


class LLMProvider(ABC):
    """Abstract base class for LLM CLI providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods and properties. This ensures a consistent
    interface for invoking different LLM CLI tools.
    """

    # How build_command() delivers the prompt to the CLI. Every current
    # provider CLI takes the prompt as an argv element (none has a documented
    # stdin or prompt-file mode used anywhere in this codebase), so "argv" is
    # the default. On Windows, argv transport is subject to hard command-line
    # length caps (cmd.exe: 8,191 chars for .cmd/.bat shims; CreateProcess:
    # 32,767), so llm_client enforces a prompt-length check for "argv"
    # providers there. An adapter that grows a stdin ("stdin") or prompt-file
    # ("file") mode should override this to opt out of that check.
    prompt_transport: str = "argv"

    # Which transports this adapter can actually deliver a prompt over (D2d).
    # ``prompt_transport`` above is the one the fan-out path uses by default;
    # this tuple is what a *caller* may ask for, and it is what the materialized
    # -prompt budget is derived from: a materialized prompt is far larger than
    # the metadata JSON it replaces (validation alone permits ~30,000 characters
    # of context before the suggestion list), and argv cannot carry that on
    # Windows, where the whole rendered command line is capped at 8,191 UTF-16
    # code units for a .cmd/.bat shim and 32,767 natively. An adapter declaring
    # "stdin" or "file" makes the budget effectively unbounded; an argv-only
    # adapter keeps the cap, and the producer splits the batch rather than
    # truncating.
    supported_prompt_transports: Tuple[str, ...] = ("argv",)

    # Where ``build_command`` puts the prompt in the argv it returns (§9.4).
    #
    # ``-1`` — the last element — is the assumption every ``cmd[:-1]`` override
    # and :meth:`build_agent_command_for_transport` already relied on, silently.
    # Measured 2026-08-16: **two** adapters violated it. grok put the prompt at
    # index 4 (normalized since — the ``-p <prompt>`` pair now trails), and aider
    # has no standalone prompt element at all because it ships ``"/ask " +
    # prompt`` after ``--message``. That second case is why a bare assert could
    # not be written and this is a declaration instead: ``None`` means "not a
    # standalone argv element", and an adapter declaring it must never be handed
    # a non-argv transport or an agent profile, both of which drop ``argv[-1]``.
    #
    # ``tests/test_provider_housekeeping.py`` checks the declaration against the
    # argv each adapter really renders, in both directions.
    prompt_argv_index: Optional[int] = -1

    # Whether this adapter's ``generate`` invocation can write the per-model
    # result file the Lane A review prompts name in their ``{output_location}``
    # block (``llm_client.render_output_location``).
    #
    # ``True`` is the default and the honest one for a CLI agent that carries
    # write tools — it is what ``invoke_with_file_output``'s whole file-first
    # contract is built on. ``False`` is a *measured* refusal, never a
    # preference, and it exists because two adapters deliberately run their CLI
    # in a read-only chat mode: cursor under ``--mode ask`` (the write-refusal
    # that replaced ``-f``/``--yolo`` — see
    # :data:`~utils.providers.cursor_agent.CURSOR_REQUIRED_FLAGS`) and aider
    # under ``/ask``. Handed a "write this file" instruction those two spend the
    # whole turn explaining that they cannot, no file appears, and the slot dies
    # on ``JSON parsing failed from both file and stdout`` — a refusal sentence
    # is not a parse failure, so no retry rung can recover it either.
    #
    # The declaration is read at prompt-render time, so an adapter that cannot
    # write is *asked* for stdout rather than being asked for a file and left to
    # the fallback it can only reach by disobeying its own mode. Nothing about
    # the parse side moves: ``invoke_with_file_output`` still reads the file
    # first, so an adapter that writes one anyway is still served from it.
    supports_file_output: bool = True

    # ------------------------------------------------------------------
    # Subagent-runner capability declarations (D3/D3e/D3j/D3m)
    #
    # All five are **defaulting** members, following the ``prompt_transport``
    # pattern above: none of the thirteen shipped adapters has to change to keep
    # importing, instantiating and serving the ``generate`` path, and the model
    # fan-out — which never asks for a profile — is unaffected entirely. An
    # adapter opts in by overriding them, and ``tests/test_providers.py``
    # cross-checks every declaration against the members actually implemented so
    # the data cannot drift from the code.
    # ------------------------------------------------------------------

    #: Invocation profiles this adapter really implements. ``generate`` is
    #: implicit for every adapter (that is ``build_command``); anything else must
    #: be backed by a real ``build_agent_command`` branch.
    agent_profiles: Tuple[str, ...] = (PROFILE_GENERATE,)

    #: True only when this adapter has an edit-capable invocation (``agent-write``).
    supports_file_edits: bool = False

    #: True when the adapter's agentic invocation can explore a repository
    #: (``context_mode: workspace-read``, D2c).
    supports_workspace_reads: bool = False

    #: Declared usage fidelity (D3j): ``none`` | ``partial`` | ``full``.
    usage_support: str = USAGE_NONE

    #: What the vendor CLI does *inside* one awaited subprocess before it returns
    #: (§7.6(b)). ``None`` means unmeasured: the ladder then assumes a single
    #: request and refuses to hand the adapter a retry-count env var (§7.3).
    #:
    #: The converse is a test, not a convention — an adapter whose ``get_env``
    #: sets a retry-shaped variable while this stays ``None`` is a budget bug,
    #: because §7.5 rule 4 would subtract one request's worth of wall clock from
    #: the deadline while the child quietly makes several.
    #:
    #: An adapter whose envelope depends on the *resolved* slot timeout (as
    #: claude-code's does — its wall is 4 × ``API_TIMEOUT_MS``, itself derived)
    #: overrides this with a property, so the declaration and the emitted env are
    #: one derivation rather than two copies of it.
    internal_retries: Optional[InternalRetryEnvelope] = None

    #: Whether this adapter really emits a schema flag when handed a ``schema=``
    #: (§4.2.2). ``False`` is the default and the honest one: the default
    #: builders below **ignore** ``schema`` entirely, so a caller that assumed
    #: otherwise would be reading prose as validated output.
    #:
    #: The caller reads this **before** dispatch: ``False`` ⇒ the phase runs its
    #: prompt-level JSON path and the result is marked ``recovered`` /
    #: ``unvalidated`` per §4.2.1, never ``native``. Only ``claude-code`` and
    #: ``codex`` set it ``True``, and an adapter that sets it ``True`` while
    #: emitting no flag is a bug ``tests/test_structured_output.py`` catches.
    structured_output_support: bool = False

    #: How this adapter carries a schema: ``"inline"`` (claude's
    #: ``--json-schema <json>``) or ``"file"`` (codex's
    #: ``--output-schema <FILE>``). §4.2's contract asymmetry, **declared** so
    #: the caller knows whether to materialize a sidecar without branching on an
    #: adapter name. Meaningless while ``structured_output_support`` is ``False``.
    structured_output_transport: str = "inline"

    #: argv that prints the CLI version, the oldest release whose streams this
    #: parser is known to handle, and the release the shipped parser fixtures
    #: were captured from (D3m). Empty means "not declared" — the executor
    #: records ``unknown`` and warns rather than blocking.
    version_cmd: Tuple[str, ...] = ()
    min_version: str = ""
    verified_version: str = ""

    #: A cheap, **non-generating** argv that reports whether this CLI is logged in
    #: (D3c) — a ``login status`` / ``whoami``-class subcommand. The executor
    #: probes it once per distinct runner per run, in the same preflight that runs
    #: ``version_cmd``, so a logged-out runner is refused *before* a child is
    #: spawned instead of being discovered one dispatch at a time (installed is not
    #: authenticated: every ``is_available()`` is a bare ``shutil.which``).
    #: Empty means "this CLI offers no such subcommand": the probe is skipped and
    #: the auth state is recorded ``unknown`` rather than assumed good.
    auth_check: Tuple[str, ...] = ()

    #: The command that fixes a failed ``auth_check``, named verbatim in the
    #: exit-3 refusal. Mirrors ``runners[*].login_cmd`` in ``hosts/capabilities.yaml``,
    #: which is the same fact from the host's side.
    login_cmd: str = ""

    #: Substrings that identify a login/credential notice on either stream, so a
    #: logged-out CLI lands on D3k rung 1 (``auth_failed``, no retry) instead of
    #: burning the retry budget as generic runner noise (D3c).
    auth_error_patterns: Tuple[str, ...] = (
        "not logged in",
        "please log in",
        "please login",
        "authentication required",
        "unauthorized",
        "invalid api key",
        "api key is missing",
        "no credentials",
        "credentials not found",
        "run `login`",
    )

    #: Substrings that identify a rate-limit / quota / overloaded response. A
    #: 429 is **not** a transport failure (D3i): it is retried with backoff and
    #: jitter and does not consume the retry budget, so it needs its own
    #: classification rather than being absorbed by "no parsed text". The bare
    #: status code is matched **with context** (see :func:`stream_pattern`) — a
    #: substring test would find it inside a cost or a token count and hand a
    #: never-retried failure an unlimited retry path.
    rate_limit_patterns: Tuple[str, ...] = (
        "429",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "quota exceeded",
        "quota_exceeded",
        "insufficient_quota",
        "overloaded",
    )

    #: Regexes that make an **exit-0** run a failure (§7.2). Four shipped CLIs
    #: report a provider failure with exit code 0 — goose, aider, cline and
    #: cursor-agent (measured in all six auth states, §5.3) — so for them the
    #: exit code carries no verdict at all and only a stream marker does. These
    #: are consulted **last** in :func:`classify_streams`: they say "this run
    #: failed", never *why*, so every specific category gets its chance first.
    #:
    #: Empty for an adapter whose exit code is authoritative. Regexes rather than
    #: substrings because the markers are structured-JSON fragments
    #: (``"finishReason":"error"``) whose boundaries the substring matcher's
    #: separator rules were never designed for.
    exit_zero_failure_markers: Tuple["Pattern[str]", ...] = ()

    def supports_prompt_transport(self, transport: str) -> bool:
        """Whether this adapter can deliver a prompt over ``transport``."""
        return transport in self.supported_prompt_transports

    def classify_failure(
        self, stdout: str, stderr: str, exit_code: int = 0
    ) -> Optional[str]:
        """Classify a run: a ladder status, a ``CATEGORY_*`` name, or ``None``.

        The adapter supplies the *data* — its declared pattern sets and its own
        exit-0 markers — and :func:`classify_streams` is the one implementation
        of the rule. That split is the point of §7.2: before it, Lane A carried
        five substring copies of this decision and ``validation.py`` a sixth, and
        they disagreed with each other and with this method about the same
        streams.

        Both streams are read, because several providers report errors on stdout
        and leave stderr empty. An unrecognized error stays ``None`` —
        "transport-shaped" — and falls back to the executor's bounded retry path
        rather than looping forever.

        Auth is still checked before rate limiting: a logged-out CLI is D3k rung
        1 (no retry, whole batch fails), and misreading it as a rate limit would
        retry a condition no retry can fix. What now precedes **both** is the
        launch-failure tuple, because a refused exec prints text that reads like
        a credential problem (§7.2's root-cause-before-auth rule).

        Every declared pattern is still matched through :func:`stream_pattern`,
        never as a bare substring: the sets contain a status code (``429``) and
        short words (``unauthorized``) that occur inside numbers and longer
        tokens a runner routinely prints.
        """
        return classify_streams(
            stdout,
            stderr,
            exit_code,
            auth_patterns=self.auth_error_patterns,
            rate_limit_patterns=self.rate_limit_patterns,
            exit_zero_markers=self.exit_zero_failure_markers,
        )

    def classify_auth_probe(
        self, stdout: str, stderr: str, exit_code: int = 0
    ) -> Optional[bool]:
        """Read an ``auth_check`` run as authenticated / not / unknown (D3c).

        ``False`` means a login/credential notice was recognized — that is the
        only verdict that refuses a run, so it is the only one drawn from a
        pattern match. A non-zero exit with nothing recognized is ``None``
        ("unknown"), because an older CLI without the subcommand, a broken pipe or
        an offline probe all exit non-zero without saying anything about
        credentials, and blocking a run on that would be worse than the dispatch
        it is meant to save. ``True`` is a clean exit with no notice.
        """
        haystack = f"{stdout or ''}\n{stderr or ''}".lower()
        if matches_any(self.auth_error_patterns, haystack):
            return False
        if exit_code != 0:
            return None
        return True

    def supports_profile(self, profile: str) -> bool:
        """Whether this adapter declares ``profile`` (D3e)."""
        return profile in self.agent_profiles

    def build_agent_command(
        self,
        prompt: str,
        model: str,
        profile: str = PROFILE_AGENT_READONLY,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build the argv for an *agentic* invocation under ``profile`` (D3e).

        The executor never picks a raw command: it picks a profile, and the
        adapter renders the invocation whose own sandbox/permission mechanism
        enforces it. There is deliberately no fabricated fallback — an adapter
        with no genuinely read-only mode raises rather than being dispatched with
        write access to an untrusted prompt.

        The profile also scopes the child's git environment (§2.2), and that
        scoping is asymmetric on purpose: ``generate`` and ``agent-readonly`` run
        with ``credential.helper=""`` so a hanging helper fails in ~0.27 s instead
        of burning the slot and orphaning a child, while ``agent-write`` keeps
        whatever helper the user configured because it is the profile that
        legitimately reaches private remotes. **Re-globalising that wipe is
        forbidden** — see :func:`git_env`.

        Args:
            prompt: The prompt text.
            model: The model identifier.
            profile: ``agent-readonly`` or ``agent-write``.
            timeout_s: The caller's already-resolved outer timeout, in whole
                seconds (§9.6.a). Keyword-only, optional, and ignored by every
                adapter with no internal cutoff of its own — see
                :meth:`build_command` for why it exists and what ``None`` means.
            schema: §4.2.2's optional structured-output schema. Keyword-only,
                optional, and **ignored by this default implementation** — see
                :attr:`structured_output_support`.

        Returns:
            argv suitable for ``subprocess.run``.

        Raises:
            ProfileUnsupported: always, unless the adapter overrides this.
        """
        raise ProfileUnsupported(
            f"{self.name} does not implement the {profile!r} invocation profile "
            f"(declared: {', '.join(self.agent_profiles) or 'none'})",
            provider=self.name,
            profile=profile,
        )

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
        """``build_agent_command`` plus the text to write to the child's stdin.

        Mirrors :meth:`build_command_for_transport` for the agentic profiles. The
        default implementation assumes the prompt is the **last** argv element
        (which every profile command this repo ships builds that way), so a
        ``stdin`` transport simply drops it and returns it as the child's input.
        """
        argv = self.build_agent_command(
            prompt, model, profile, timeout_s=timeout_s, schema=schema
        )
        if transport == "argv":
            return argv, None
        if transport == "stdin" and self.supports_prompt_transport("stdin"):
            if argv and argv[-1] == prompt:
                return argv[:-1], prompt
            return [arg for arg in argv if arg != prompt], prompt
        raise ValueError(
            f"{self.name} does not support the {transport!r} prompt transport "
            f"for profile {profile!r} (supported: "
            f"{', '.join(self.supported_prompt_transports)})"
        )

    def parse_usage(self, stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
        """Extract normalized usage from a captured stream, or ``None`` (D3j).

        One closed, all-optional structure: ``input_tokens``, ``output_tokens``,
        ``total_tokens``, ``cached_input_tokens``, ``tool_uses``, ``cost_usd``.
        **Absent means "not reported", never 0** — a zero would average into
        per-phase totals and read as a free run — and nothing is ever derived
        from character counts or estimated.

        Returns ``None`` by default: an adapter is duration-only until a captured
        fixture proves its extractor.
        """
        return None

    def parse_version(self, stdout: str, stderr: str) -> Optional[str]:
        """Recover the CLI version from ``version_cmd`` output (D3m).

        The default finds the first dotted numeric token on either stream, which
        covers ``codex-cli 0.146.0``, ``1.18.12`` and ``2.1.224 (Claude Code)``
        alike. Unparseable output is ``None`` (recorded ``unknown`` plus a
        warning, never a block: a CLI that dropped ``--version`` still answers
        prompts fine).
        """
        import re

        for text in (stdout or "", stderr or ""):
            match = re.search(r"\b(\d+\.\d+(?:\.\d+)*)\b", text)
            if match:
                return match.group(1)
        return None

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
        """Build argv plus the text to write to the child's stdin.

        Args:
            prompt: The prompt text.
            model: The model identifier.
            transport: ``"argv"`` (prompt is an argv element), ``"stdin"``
                (``build_command`` omits it and the caller writes it to the
                child's stdin) or ``"file"`` (prompt already written to
                ``prompt_file``, whose path travels on argv).
            prompt_file: Path the prompt was written to, for ``"file"``.
            timeout_s: The caller's already-resolved outer timeout, forwarded to
                :meth:`build_command` (§9.6.a/b). This is the Lane A leg:
                ``invoke_with_provider`` computes ``effective_timeout`` before it
                builds anything, so the adapter never re-derives it.

        Returns:
            ``(argv, stdin_text)``; ``stdin_text`` is ``None`` unless the
            transport is ``"stdin"``.

        Raises:
            ValueError: when this adapter does not support ``transport``.
        """
        if transport == "argv":
            return (
                self.build_command(
                    prompt, model, timeout_s=timeout_s, schema=schema
                ),
                None,
            )
        raise ValueError(
            f"{self.name} does not support the {transport!r} prompt transport "
            f"(supported: {', '.join(self.supported_prompt_transports)})"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'cursor-agent', 'gemini').

        Returns:
            A unique string identifier for this provider.
        """
        pass

    @property
    @abstractmethod
    def default_timeout(self) -> int:
        """Default timeout in seconds for this provider.

        Returns:
            The default number of seconds to wait before timing out.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the CLI tool is available in PATH.

        Returns:
            True if the CLI tool can be found and executed, False otherwise.
        """
        pass

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build the command line arguments for invocation.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use for generation.
            timeout_s: The caller's already-resolved outer timeout in whole
                seconds, for the adapters whose CLI has an internal cutoff of
                its own to keep *under* it (§9.6.a). Twelve of the thirteen
                adapters have no such flag and ignore it.

                It is **keyword-only and optional on purpose**. Adapter
                instances are module-level singletons shared across a concurrent
                fan-out, so stashing a per-call timeout on ``self`` would be a
                data race rather than an API; and ``timeout_s=None`` must render
                argv byte-for-byte identical to what the adapter emitted before
                this parameter existed, which is what makes dropping the one
                call site in ``llm_client`` a complete rollback.
            schema: §4.2.2's optional structured-output schema, on the same terms
                and for the same reasons as ``timeout_s``: keyword-only,
                optional, ignored by the eleven adapters that declare no
                :attr:`structured_output_support`, and ``schema=None`` — the
                default and the only value those eleven ever see — renders argv
                **byte-for-byte identical** to today's. That is what keeps every
                existing ``expected_argv.yaml`` row green and what makes the
                per-adapter flag the off switch (§14.4).

        Returns:
            A list of command line arguments suitable for subprocess.run().
        """
        pass

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse output and return structured result.

        Args:
            stdout: The standard output from the CLI invocation.
            stderr: The standard error from the CLI invocation.

        Returns:
            A dictionary with at least 'success' (bool) and 'data' keys.
            On success, 'data' contains the parsed response.
            On failure, 'data' may contain error information.
        """
        pass

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """Extract a human-readable reason from a non-zero-exit invocation.

        ``parse_output`` is only called when the CLI exits 0, so a provider
        that reports its errors on *stdout* rather than stderr would otherwise
        have them discarded — the caller sees a bare exit code with empty
        stderr and nothing to diagnose. Override this to recover the message.

        Args:
            stdout: The standard output from the failed CLI invocation.
            stderr: The standard error from the failed CLI invocation.

        Returns:
            A short description of the failure, or None when the provider has
            nothing to add beyond the exit code and stderr. None by default.
        """
        return None

    def get_env(self, model: str) -> Dict[str, str]:
        """Return **adapter-specific** environment variables for subprocess.

        Override this method if the provider requires environment variables
        for configuration (e.g., model selection).

        This is deliberately *not* where the non-interactive git block lives:
        that block is profile-scoped (§2.2) and this hook has no profile, and it
        must reach adapters that override this method too. It is composed in
        :meth:`resolve_env`, which is what both spawn sites call. **Never move
        ``git_env``'s ``GIT_CONFIG_COUNT`` / ``credential.helper=""`` keys in
        here** — they would then apply to ``agent-write``, whose git access to
        the user's own remotes the scoping exists to preserve.

        Args:
            model: The model identifier to use for generation.

        Returns:
            A dictionary of environment variables to set for the subprocess.
            Empty dict by default.
        """
        return {}

    def git_credential_helper_policy(self, anchor: Optional[str] = None) -> str:
        """This provider's resolved ``git_env.credential_helper`` (§2.2).

        ``MULTI_LLM_GIT_CREDENTIAL_HELPER`` → ``providers.<name>.git_env`` →
        top-level ``git_env`` → ``auto``. Read through the registry accessor
        rather than by indexing the config, exactly like the ``driver.*`` keys.
        """
        try:  # pragma: no cover - import shim mirrors the rest of utils/
            from ..provider_registry import get_git_credential_helper
        except ImportError:  # pragma: no cover - direct-script import shim
            from utils.provider_registry import get_git_credential_helper
        return get_git_credential_helper(self.name, anchor=anchor)

    def resolve_env(
        self,
        model: str,
        profile: str = PROFILE_GENERATE,
        *,
        credential_helper: Optional[str] = None,
        base_env: Optional[Mapping[str, str]] = None,
        anchor: Optional[str] = None,
    ) -> Dict[str, str]:
        """The full env delta a spawn applies: the §2.2 git block, then the adapter's.

        The one composition point, called by both spawn sites
        (``llm_client.call_llm`` for the Lane A fan-out under ``generate``, and
        ``subagent_executor._spawn`` with the dispatch's own profile), so the git
        block cannot be present on one lane and missing on the other. The
        adapter's own :meth:`get_env` is merged **last** and therefore wins on any
        key it names.

        ``profile`` is what scopes the credential-helper wipe, and
        ``agent-write`` must keep resolving to *no* wipe — see :func:`git_env`.
        """
        policy = (
            credential_helper
            if credential_helper is not None
            else self.git_credential_helper_policy(anchor=anchor)
        )
        env = git_env(profile, credential_helper=policy, base_env=base_env)
        env.update(self.get_env(model) or {})
        return env

    def get_remove_env(self) -> List[str]:
        """Return environment variable names to remove for subprocess.

        Override this method if the provider needs certain env vars stripped
        from the inherited environment (e.g., to avoid nested-session guards).

        Returns:
            A list of environment variable names to remove. Empty by default.
        """
        return []
