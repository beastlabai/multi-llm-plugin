"""Base protocol for LLM CLI providers."""

import re
from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional, Tuple

#: The three invocation profiles (D3e). ``generate`` is today's callee path (no
#: tools, no workspace); ``agent-readonly`` is an agentic invocation with writes
#: denied by the runner's own sandbox/permission mechanism and is what every
#: read/analyze subagent runs under; ``agent-write`` is the implement/apply path.
PROFILE_GENERATE = "generate"
PROFILE_AGENT_READONLY = "agent-readonly"
PROFILE_AGENT_WRITE = "agent-write"

#: Declared usage-extraction fidelity (D3j), mirrored by ``runners[*].usage`` in
#: the D5b capability data. ``none`` is the honest default: an adapter with no
#: fixture-verified extractor is duration-only, never a promise of token counts.
USAGE_NONE = "none"
USAGE_PARTIAL = "partial"
USAGE_FULL = "full"


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

    def supports_prompt_transport(self, transport: str) -> bool:
        """Whether this adapter can deliver a prompt over ``transport``."""
        return transport in self.supported_prompt_transports

    def classify_failure(
        self, stdout: str, stderr: str, exit_code: int = 0
    ) -> Optional[str]:
        """Classify a run as ``auth_failed`` / ``rate_limited``, or ``None``.

        Classification lives in the adapter because it is the only layer that
        knows the provider's error shape; the base implementation covers the two
        shapes every CLI shares, matching on **both** streams because several
        providers report errors on stdout and leave stderr empty. An unrecognized
        error stays ``None`` — "transport-shaped" — and falls back to the
        executor's bounded retry path rather than looping forever.

        Auth is checked first: a logged-out CLI is D3k rung 1 (no retry, whole
        batch fails), and misreading it as a rate limit would retry a condition
        no retry can fix.

        Every pattern is matched through :func:`stream_pattern`, never as a bare
        substring: the declared sets contain a status code (``429``) and short
        words (``unauthorized``) that occur inside numbers and longer tokens a
        runner routinely prints, and both misreadings move a run onto a rung with
        different retry rules than the one it belongs on.
        """
        haystack = f"{stdout or ''}\n{stderr or ''}".lower()
        if not haystack.strip():
            return None
        if matches_any(self.auth_error_patterns, haystack):
            return "auth_failed"
        if matches_any(self.rate_limit_patterns, haystack):
            return "rate_limited"
        return None

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
        self, prompt: str, model: str, profile: str = PROFILE_AGENT_READONLY
    ) -> List[str]:
        """Build the argv for an *agentic* invocation under ``profile`` (D3e).

        The executor never picks a raw command: it picks a profile, and the
        adapter renders the invocation whose own sandbox/permission mechanism
        enforces it. There is deliberately no fabricated fallback — an adapter
        with no genuinely read-only mode raises rather than being dispatched with
        write access to an untrusted prompt.

        Args:
            prompt: The prompt text.
            model: The model identifier.
            profile: ``agent-readonly`` or ``agent-write``.

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
    ) -> Tuple[List[str], Optional[str]]:
        """``build_agent_command`` plus the text to write to the child's stdin.

        Mirrors :meth:`build_command_for_transport` for the agentic profiles. The
        default implementation assumes the prompt is the **last** argv element
        (which every profile command this repo ships builds that way), so a
        ``stdin`` transport simply drops it and returns it as the child's input.
        """
        argv = self.build_agent_command(prompt, model, profile)
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

        Returns:
            ``(argv, stdin_text)``; ``stdin_text`` is ``None`` unless the
            transport is ``"stdin"``.

        Raises:
            ValueError: when this adapter does not support ``transport``.
        """
        if transport == "argv":
            return self.build_command(prompt, model), None
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
    def build_command(self, prompt: str, model: str) -> List[str]:
        """Build the command line arguments for invocation.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use for generation.

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
        """Return additional environment variables for subprocess.

        Override this method if the provider requires environment variables
        for configuration (e.g., model selection).

        Args:
            model: The model identifier to use for generation.

        Returns:
            A dictionary of environment variables to set for the subprocess.
            Empty dict by default.
        """
        return {}

    def get_remove_env(self) -> List[str]:
        """Return environment variable names to remove for subprocess.

        Override this method if the provider needs certain env vars stripped
        from the inherited environment (e.g., to avoid nested-session guards).

        Returns:
            A list of environment variable names to remove. Empty by default.
        """
        return []
