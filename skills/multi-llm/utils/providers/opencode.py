"""OpenCode CLI provider implementation with NDJSON event stream parsing."""
import atexit
import contextlib
import contextvars
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple

from ..json_extractor import extract_json_from_text
from .base import (
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    REASONING_LADDER,
    USAGE_FULL,
    LLMProvider,
    ModelVerdict,
    ReasoningLadder,
    StructuredSchema,
    split_reasoning_effort,
)

# OpenCode ships a built-in `plan` agent whose permission set denies every edit
# tool ("Plan mode. Disallows all edit tools." — verified against the agent
# table in the 1.18.x binary's embedded bundle, alongside `edit: {"*": "deny"}`).
# `--agent` is a documented `opencode run` flag, so the read-only profile is
# opencode's own permission mechanism rather than a convention this repo layers
# on top.
READONLY_AGENT = "plan"

# OpenCode's built-in edit-capable agent, the one `opencode run` uses when
# `--agent` is omitted ("build" — the primary agent in the same embedded agent
# table `plan` comes from). It is named EXPLICITLY on the `agent-write` profile
# rather than relying on the default, so a future change to opencode's default
# agent cannot silently move a write dispatch onto a different permission set.
WRITE_AGENT = "build"

# Reasoning effort values accepted by `opencode run --variant` — the values are
# provider-specific, so this is the practical union across providers
# (verified live against opencode 1.17.15): all seven canonical rungs, since
# every one of the 24 observed per-model `variants` sets is a subset of them.
#
# `model_caps` carries the one combination that is not merely unsupported but
# MEASURED TO HANG: `--variant` on `openrouter/deepseek/deepseek-v4-flash-0731`
# writes zero bytes to either stream, never reaches session creation, and burns
# the whole slot timeout, where the plain (suffix-free) model answers in ~3 s.
# It was a prose comment in `providers.yaml` before M12 and is a declaration
# here now, so a config that re-adds the suffix is graded against the adapter
# rather than against a paragraph nobody parses.
EFFORT_LADDER = ReasoningLadder(
    provider="opencode",
    rungs=REASONING_LADDER,
    model_caps=(("openrouter/deepseek/deepseek-v4-flash-0731", ()),),
)
REASONING_EFFORTS = EFFORT_LADDER.accepted

# ─────────────────────────────────────────────────────────────────────────────
# R3 — the structural model-spec refusal (opencode only).
#
# `opencode run --help` documents `-m, --model` as "model to use in the format
# of provider/model", and the resolver enforces it by *splitting* on the first
# `/`: a bare token becomes provider `<token>` with an empty model half, which
# fails as `ProviderModelNotFoundError: Model not found: <token>/.` before any
# HTTP call is attempted. Measured on opencode 1.18.18.
#
# That is the whole predicate, and the reason it is safe to refuse on: an empty
# model half is not a name any catalog, any release or any server can supply, so
# the dispatch is unresolvable *by construction*. It is deliberately NOT a
# catalog test — a well-formed slug that opencode's catalog does not list may
# still be live server-side or served by a provider declared in opencode's own
# config, which multi-llm cannot read, and refusing those would break working
# setups. Refuse only what the addressing scheme cannot express, never what the
# catalog merely does not list.
#
# If a later opencode release starts resolving bare tokens against a default
# provider, this check is withdrawn back to a warning rather than kept on for
# tidiness.
# ─────────────────────────────────────────────────────────────────────────────

#: The version the bare-token behaviour above was observed on.
STRUCTURAL_MODEL_VERSION = "1.18.18"

# ─────────────────────────────────────────────────────────────────────────────
# M4 / donor §2.3 — hermetic runs.
#
# The user's own opencode config can silently change the effective model, agent
# or permission set per machine and per repository. The containment is an empty,
# run-scoped `XDG_CONFIG_HOME` plus `--pure`; `XDG_DATA_HOME` is passed through
# untouched, which is what keeps `auth.json` (and therefore `auth list` and
# `models`) working — auth is never config-borne.
#
# `XDG_CACHE_HOME` is deliberately NOT redirected: it holds `opencode/models.json`,
# and a cold cache degrades *silently* (exit 0, empty stderr, a truncated model
# list). `XDG_STATE_HOME` and `XDG_RUNTIME_DIR` are likewise untouched.
#
# Measured on opencode 1.18.18, three deviations from the donor's text worth
# recording where the code is:
#   * `XDG_CONFIG_HOME` alone leaves the **workspace** `opencode.json` in force —
#     `--pure` does not close that layer either (measured: a workspace file
#     pinning `model` and an extra agent still shows up in `opencode debug
#     config` under both). `OPENCODE_DISABLE_PROJECT_CONFIG=1` is what yields the
#     empty-config shape, so it is set alongside the redirect; without it the
#     project-config isolation this section exists to provide does not exist.
#   * the directory does not stay empty: opencode seeds `opencode/.gitignore`
#     and an `opencode/opencode.jsonc` holding nothing but `$schema`. That is
#     fixed content with no per-slot state, so the one-shared-directory rule
#     stands; cleanup is `rmtree`, not `rmdir`. A future version that writes
#     slot-*varying* state there is what moves this to per-slot.
#   * `OPENCODE_CONFIG`, `OPENCODE_CONFIG_CONTENT` and `OPENCODE_CONFIG_DIR` are
#     stripped from the inherited environment while hermetic, because
#     `OPENCODE_CONFIG_CONTENT` is a *last-applied deep-merge layer* and an
#     inherited one would survive — and win over — the redirect.
# ─────────────────────────────────────────────────────────────────────────────

#: `opencode run --pure` — "run without external plugins". Optional, not
#: load-bearing: without it the redirect still removes the global config layer,
#: so an unsupported or undecidable `--help` sniff omits the flag, WARNs the
#: reduced containment and keeps the run.
PURE_FLAG = "--pure"

#: `providers.opencode.hermetic`. `off` restores today's inheritance; `on` (the
#: default) and `auto` both apply the redirect to the profiles in
#: :data:`HERMETIC_PROFILES`. `auto` is accepted and behaves as `on` today — it
#: exists so the documented vocabulary is a value the loader accepts rather than
#: a validation error, and it is the value that stays *scoped* if `on` is later
#: widened to write slots.
HERMETIC_POLICIES = ("auto", "on", "off")
HERMETIC_DEFAULT = "on"

#: Fan-out (`generate`, which is what Lane A and `ask` dispatch under) and
#: `agent-readonly` only. `agent-write` keeps today's inherited config: the
#: built-in `build` agent is *expected* to survive `--pure`, and "expected" is
#: not the standard this repo applies to permission-carrying dispatch. Write
#: slots join once the ship gates are green on a write slot.
HERMETIC_PROFILES = frozenset({PROFILE_GENERATE, PROFILE_AGENT_READONLY})

#: `tempfile.mkdtemp` prefix, so a leaked directory is sweepable by name.
HERMETIC_DIR_PREFIX = "multi-llm-opencode-"

#: Config-carrying variables an inherited environment may hold. Stripped while
#: hermetic — except the one this adapter emits itself, when it emits it.
CONFIG_ENV_NAMES = ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "OPENCODE_CONFIG_DIR")

#: Closes the workspace-`opencode.json` layer the redirect does not reach.
PROJECT_CONFIG_ENV = "OPENCODE_DISABLE_PROJECT_CONFIG"

_CONFIG_HOME_LOCK = threading.RLock()
_CONFIG_HOME: Optional[str] = None
#: How many dispatch contexts are currently inside the shared directory. One
#: directory per *process*, shared by every concurrent slot (no slot writes
#: per-slot state there), removed when the last slot leaves.
_CONFIG_HOME_USERS = 0
_ATEXIT_REGISTERED = False

#: The active dispatch's hermetic state, or ``None``. A ``ContextVar`` rather
#: than a plain global because it must be per-thread AND per-asyncio-task: Lane A
#: fans out concurrently, and a slot must never see another slot's env.
#:
#: Argv and env both read it, which is the point — ``--pure`` without the paired
#: `XDG_CONFIG_HOME` redirect is meaningless, so the two halves are rendered from
#: one piece of state and cannot drift apart. Outside a dispatch context the
#: adapter renders exactly the argv and env it rendered before M4.
_ACTIVE: contextvars.ContextVar[Optional[Dict[str, Optional[str]]]] = (
    contextvars.ContextVar("multi_llm_opencode_hermetic", default=None)
)

_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message, file=sys.stderr)


def _load_config(anchor: Optional[str] = None) -> Dict[str, Any]:
    """multi-llm's merged config, or ``{}`` if it cannot be read.

    A config problem may never turn into a refusal or a crash on a dispatch
    path, so every failure here resolves to "no configuration", which means the
    documented defaults.
    """
    try:  # pragma: no cover - import shim mirrors the rest of utils/
        from ..provider_registry import load_config
    except ImportError:  # pragma: no cover - direct-script import shim
        from utils.provider_registry import load_config  # type: ignore[no-redef]
    try:
        config = load_config(anchor=anchor)
    except Exception:  # noqa: BLE001 - a config read may not break a dispatch
        return {}
    return config if isinstance(config, dict) else {}


def _opencode_block(anchor: Optional[str] = None) -> Dict[str, Any]:
    providers = _load_config(anchor).get("providers")
    if not isinstance(providers, dict):
        return {}
    block = providers.get("opencode")
    return block if isinstance(block, dict) else {}


def hermetic_policy(anchor: Optional[str] = None) -> str:
    """Resolve ``providers.opencode.hermetic`` (default :data:`HERMETIC_DEFAULT`).

    An unrecognized value warns once naming the key and falls back to the
    default: this is read on every dispatch, far from any parser, and raising
    out of the middle of a fan-out over a typo would be worse than the typo.
    """
    value = _opencode_block(anchor).get("hermetic")
    if value is None:
        return HERMETIC_DEFAULT
    if isinstance(value, bool):
        # YAML 1.1 resolves a bare `on`/`off` to a boolean, and a user who wrote
        # the documented word should get the documented behaviour rather than a
        # warning about their own YAML loader.
        return "on" if value else "off"
    candidate = str(value).strip().lower()
    if candidate in HERMETIC_POLICIES:
        return candidate
    _warn_once(
        f"hermetic:{candidate}",
        f"multi-llm config warning: ignoring providers.opencode.hermetic="
        f"{value!r} (expected one of {', '.join(HERMETIC_POLICIES)}); using "
        f"{HERMETIC_DEFAULT}",
    )
    return HERMETIC_DEFAULT


def hermetic_enabled(profile: str, anchor: Optional[str] = None) -> bool:
    """Whether this profile runs hermetically under the resolved policy."""
    if hermetic_policy(anchor) == "off":
        return False
    return profile in HERMETIC_PROFILES


def _register_sweep() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(sweep_config_home)
    _ATEXIT_REGISTERED = True


def acquire_config_home() -> str:
    """The process-scoped 0700 config home, created on first use.

    One directory per orchestrator process, **shared** by every concurrent
    opencode slot: it exists only so opencode's config loader finds nothing of
    the user's, no slot keeps state in it, and a per-slot directory would
    multiply `mkdtemp` calls and cleanup races for zero isolation.
    """
    global _CONFIG_HOME, _CONFIG_HOME_USERS
    with _CONFIG_HOME_LOCK:
        if _CONFIG_HOME is None or not os.path.isdir(_CONFIG_HOME):
            _CONFIG_HOME = tempfile.mkdtemp(prefix=HERMETIC_DIR_PREFIX)
            os.chmod(_CONFIG_HOME, 0o700)
            _register_sweep()
        _CONFIG_HOME_USERS += 1
        return _CONFIG_HOME


def release_config_home() -> None:
    """Drop one user of the shared directory; remove it when the last leaves."""
    global _CONFIG_HOME_USERS
    with _CONFIG_HOME_LOCK:
        _CONFIG_HOME_USERS = max(0, _CONFIG_HOME_USERS - 1)
        if _CONFIG_HOME_USERS == 0:
            sweep_config_home()


def sweep_config_home() -> None:
    """Remove the directory this process created. Idempotent, never raises.

    **Never removes a directory this process did not create** — a user-supplied
    ``XDG_CONFIG_HOME`` is overridden for opencode slots and never deleted. It is
    `rmtree`, not `rmdir`, because opencode seeds a stub config file there
    (measured, 1.18.18); a leak costs one inode and is sweepable by prefix under
    ``TMPDIR``.
    """
    global _CONFIG_HOME, _CONFIG_HOME_USERS
    with _CONFIG_HOME_LOCK:
        path, _CONFIG_HOME, _CONFIG_HOME_USERS = _CONFIG_HOME, None, 0
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)


#: One bare ``provider/model`` slug, the whole line: no whitespace, at least
#: one ``/``, nothing that could be a log field or a decoration.
_MODEL_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[^\s]+$")

#: A ``--print-logs`` line, which is what a listing must NOT open with.
_LOG_LINE = re.compile(r"^\s*(?:timestamp|level|service)=", re.IGNORECASE)


def _list_models(
    argv: List[str], env: Dict[str, str]
) -> Optional["subprocess.CompletedProcess"]:
    """Run the listing subcommand with the executor's own probe pins (D3 step 4).

    Mirrors ``subagent_executor._run_probe`` exactly — ``timeout=30``,
    ``stdin=DEVNULL``, captured text streams, ``None`` on any ``OSError`` /
    ``SubprocessError`` — because that ceiling is the one that matters: a
    ``subprocess.TimeoutExpired`` **is** a ``SubprocessError``, so a listing
    that hangs is killed at 30 s and read as *unknown*, which blocks nothing.
    It takes ``env`` because the listing must be read under the dispatch's own
    hermetic delta, which the executor's probe has no channel for.
    """
    launcher: List[str] = [argv[0]]
    try:  # pragma: no cover - import shim mirrors the rest of utils/
        from ..llm_client import _resolve_executable
    except ImportError:  # pragma: no cover - direct-script import shim
        try:
            from utils.llm_client import _resolve_executable  # type: ignore
        except ImportError:
            _resolve_executable = None  # type: ignore[assignment]
    if _resolve_executable is not None:
        try:
            launcher, _ = _resolve_executable(argv[0])
            launcher = list(launcher)
        except Exception:  # noqa: BLE001 - fall back to the bare name
            launcher = [argv[0]]
    try:
        return subprocess.run(
            [*launcher, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_model_listing(text: str) -> Optional[FrozenSet[str]]:
    """The slugs of one ``opencode models`` stdout, or ``None`` if it is not one.

    ``None`` — "this is not a listing" — for an empty stream, a stream that
    opens with a log line, or one whose non-blank lines are mostly not
    ``provider/model`` shaped. A parsed listing is never empty by construction.
    """
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines or _LOG_LINE.match(lines[0]):
        return None
    slugs = [line for line in lines if _MODEL_SLUG.match(line)]
    if not slugs or len(slugs) * 2 < len(lines):
        return None
    return frozenset(slugs)


def _read_help_tokens(binary: str) -> frozenset:
    """The long options ``opencode run --help`` documents (§5.4.1's flag gate).

    **The last local copy.** This was a near-twin of the claude-code adapter's
    own sniff; M14 has since landed the shared gate
    (``base.read_help_tokens`` / ``LLMProvider.supports_flag``) and deleted the
    copies in ``claude_code.py`` and ``cursor_agent.py``, together with codex's
    prose exemption (now ``flags_verified_at_floor``). This one was outside that
    task's file set and so still stands. Folding it in is a two-line change —
    ``help_argv = ("run", "--help")`` on the adapter plus ``supports_flag``
    instead of the body below — and the only reason it is still here is
    ownership, not design.

    Failure posture — decline the *flag*, keep the *run*: a timeout, a non-zero
    exit, an unreadable stream or *anything else this probe can hit* resolves to
    the empty set, i.e. the flag is treated as unsupported and simply omitted.
    The `except` is deliberately total: a diagnostic read of a binary's own help
    text may not be the thing that ends a dispatch, and the worst outcome of
    declining is the reduced containment the caller is warned about.
    """
    cached = _HELP_TOKENS.get(binary)
    if cached is not None:
        return cached
    env = dict(os.environ)
    env["TERM"] = "dumb"
    env["COLUMNS"] = "1000"
    env["NO_COLOR"] = "1"
    tokens: frozenset = frozenset()
    try:
        completed = subprocess.run(
            [binary, "run", "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=env,
        )
        if completed.returncode == 0:
            text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            tokens = frozenset(_FLAG_TOKEN_RE.findall(text))
    except Exception:  # noqa: BLE001 - see the failure posture above
        tokens = frozenset()
    _HELP_TOKENS[binary] = tokens
    return tokens


_FLAG_TOKEN_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")
_HELP_TOKENS: Dict[str, frozenset] = {}


class _UnreconstructibleProvider(Exception):
    """A slot's provider exists only in the config the redirect replaces."""


def _defines_an_implementation(block: Dict[str, Any]) -> bool:
    """True when a `provider.<name>` block supplies the provider *itself*.

    Deliberately narrow, because this is the only predicate that can produce a
    refusal and an over-broad one would refuse a working setup. Two shapes
    qualify, and they are the two that cannot be reconstructed once the user's
    config is out of reach: an ``npm`` SDK package, or an ``options.baseURL``
    pointing the provider at a server multi-llm has never heard of — the shape
    §2.3 measured on a real machine (a custom base URL with its own ``models``
    map and a live ``apiKey``).

    Everything else is an *override* of a provider opencode already has, and an
    override still resolves after the redirect; it merely loses the override. In
    particular a user who adds a model to a built-in provider — ``provider:
    {opencode: {models: {...}}}`` — is not declaring a new provider, and
    refusing them would be exactly the over-refusal this design forbids.
    """
    if block.get("npm"):
        return True
    options = block.get("options")
    return isinstance(options, dict) and bool(options.get("baseURL"))


def _user_config_paths() -> List[str]:
    """The user's own opencode config files, read from the *inherited* env.

    This runs before the redirect is applied to a child, so ``XDG_CONFIG_HOME``
    here is still the user's. ``OPENCODE_CONFIG`` (an explicit file path) is
    honoured because opencode honours it.
    """
    paths: List[str] = []
    explicit = os.environ.get("OPENCODE_CONFIG")
    if explicit:
        paths.append(os.path.expanduser(explicit))
    home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    for name in ("opencode.jsonc", "opencode.json"):
        paths.append(os.path.join(home, "opencode", name))
    return paths


def _user_defines_provider(prefix: str) -> bool:
    """True when the user's own opencode config *defines* this provider.

    Deliberately narrow. An unreadable or unparseable config answers ``False``
    (dispatch, don't refuse) — a config problem may never become a refusal, the
    same rule the structural model check follows. So does a block that only
    overrides a provider opencode already has: see
    :func:`_defines_an_implementation`.
    """
    for path in _user_config_paths():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        providers = document.get("provider")
        if not isinstance(providers, dict):
            continue
        block = providers.get(prefix)
        if isinstance(block, dict) and _defines_an_implementation(block):
            return True
    return False


def reset_hermetic_state() -> None:
    """Forget the memoised help read and the warn-once set (tests only)."""
    _HELP_TOKENS.clear()
    _WARNED.clear()


class OpenCodeProvider(LLMProvider):
    """Provider for OpenCode CLI tool.

    OpenCode outputs NDJSON (newline-delimited JSON) events when using
    --format json. The top-level event types are ``step_start``,
    ``step_finish``, ``tool_use``, ``text``, ``reasoning`` (only with
    ``--thinking``, which this adapter never passes) and ``error``; each
    carries a ``part`` object except ``error``, which carries ``error``.
    The actual LLM response is in the "text" events' part.text field.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openai/gpt-5.5:high``), translated to ``--variant <effort>``. Valid
    efforts are listed in REASONING_EFFORTS; anything else passes through
    verbatim as the model name.
    """

    # `opencode run` reads the message from stdin when no message argument is
    # given, so this adapter has a real non-argv transport (D2d). The fan-out
    # default stays "argv"; materialized prompts use stdin.
    supported_prompt_transports = ("argv", "stdin")

    # Subagent-runner declarations (D3e/D3j/D3m).
    agent_profiles = (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE)
    supports_file_edits = True
    supports_workspace_reads = True
    usage_support = USAGE_FULL
    # `opencode auth list` prints the credentials it has stored (and nothing
    # else): no model call, no session, so it is the cheap non-generating probe
    # the once-per-run auth preflight needs (D3c).
    auth_check = ("opencode", "auth", "list")
    login_cmd = "opencode auth login"
    #: D3 step 2 — the model tokens opencode serves with NO credential present,
    #: declared rather than sniffed and measured before being written down.
    #: `opencode/big-pickle`: probed 2026-09-11 on opencode 1.18.30 from a
    #: scratch HOME/XDG_CONFIG_HOME/XDG_DATA_HOME/XDG_STATE_HOME pair whose
    #: `opencode auth list` printed `└  0 credentials`, with OPENAI_API_KEY,
    #: GEMINI_API_KEY, ANTHROPIC_API_KEY and OPENROUTER_API_KEY unset:
    #: `opencode run --model opencode/big-pickle --print-logs --log-level ERROR
    #: "Reply with the single word OK."` exited 0 with stdout `OK` in a few
    #: seconds. `openrouter/*`, `openai/*` and `google/*` stay OUT: each needs a
    #: stored or environment credential and the auth leg must keep refusing
    #: them. Zen's catalog is server-side and `opencode/sonnet` vanished from
    #: it once already, so re-probe this on each opencode bump and drop the
    #: entry the day the probe stops answering (see hosts/capabilities.yaml).
    credential_free_models = frozenset({"opencode/big-pickle"})
    version_cmd = ("opencode", "--version")
    min_version = "1.17.15"
    # NOT the newest release the parser has been graded on — the release the
    # DECLARED fixtures were captured from, and a floor `engines.opencode` is
    # asserted against (test_opencode_package.py). D3 (2026-09-11) added a real
    # 1.18.30 capture beside them, `tests/fixtures/opencode/run_review.1.18.30.jsonl`,
    # and grades the parser on both pins rather than moving this one: moving it
    # is a support-window decision with its own checklist (recaptured fixtures,
    # this constant, `engines.opencode`, the packaging README, the container
    # pin and lockfile, cli-versions.lock, host_matrix.yaml, re-recorded rows).
    verified_version = "1.18.12"

    # §7.2 — per-adapter pattern overrides, EXTENDING the base sets (never
    # replacing them: the base entries are audited-clean and shared).
    #
    # `ProviderAuthError` is the `error.name` of a live opencode auth failure,
    # captured in `tests/fixtures/opencode/provider_auth_error.jsonl` — 304 bytes
    # on **stdout** with stderr empty. The `data.message` that accompanies it is
    # the upstream SDK's, so it varies per provider ("… API key is missing …"
    # happens to match the base list for Google and would not for another);
    # the event *name* is opencode's own and does not.
    auth_error_patterns = LLMProvider.auth_error_patterns + (
        "providerautherror",
    )
    # No rate-limit override: opencode's measured rate-limit behaviour is to
    # retry internally while writing to **neither** stream until the timeout
    # fires, so there is no captured text to declare. Declaring a guessed one
    # would be worse than declaring none — this is the honest "unmeasured".
    rate_limit_patterns = LLMProvider.rate_limit_patterns

    #: `opencode auth list` exits 0 whether or not anything is stored (§5.3), so
    #: the count line is the only verdict. Measured 2026-08-16 against a scratch
    #: HOME/XDG pair: a logged-out run prints exactly "└  0 credentials" and no
    #: Environment section at all, while an env-key-only run prints the same
    #: count *plus* "N environment variable(s)" and really can authenticate.
    _EMPTY_CREDENTIALS = re.compile(r"\b0 credentials\b")
    _ENVIRONMENT_KEYS = re.compile(r"\b\d+ environment variables?\b")

    @property
    def name(self) -> str:
        return "opencode"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    # ── R3 — structural model-spec refusal ────────────────────────────────

    def describe_unresolvable_model(self, model: str) -> Optional[str]:
        """Why opencode's *addressing scheme* cannot express this model, or None.

        Structural and nothing else: after any ``:effort`` suffix is stripped, a
        model token with no ``/`` cannot be addressed at all, because opencode
        splits on the first ``/`` and a bare token yields an empty model half
        (``Model not found: <token>/.``, opencode 1.18.18) before a single
        catalog lookup or HTTP call happens.

        **This never consults a catalog.** A catalog that is empty, stale or
        unreadable can only ever downgrade the caller to "no warning"; it can
        never produce a refusal here, because this method does not read one. A
        well-formed but uncatalogued slug (``stub/stub-model``, any provider a
        user declared in opencode's own config) returns ``None`` and keeps
        today's warn-and-dispatch.

        The message states the required shape and echoes the offending token; it
        does not rewrite the spec, because the correct provider prefix for an
        unknown model cannot be inferred and guessing one would dispatch
        somewhere the caller never asked for.
        """
        base_model, _effort = split_reasoning_effort(model, REASONING_EFFORTS)
        token = base_model.strip()
        if "/" in token:
            return None
        return (
            f"opencode models resolve as 'provider/model' (`opencode run --help`: "
            f"\"model to use in the format of provider/model\"); {token!r} has no "
            f"provider prefix, so opencode splits it into provider {token!r} with "
            f"an empty model half and fails with 'ProviderModelNotFoundError: "
            f"Model not found: {token}/.' before any HTTP call (observed on "
            f"opencode {STRUCTURAL_MODEL_VERSION}) — pass e.g. 'opencode/{token}' "
            f"or the provider that serves it."
        )

    # ── D3 step 4 — the CLI's own catalog ─────────────────────────────────

    #: The listing subcommand: one bare ``provider/model`` per stdout line and
    #: nothing else (measured on 1.18.30 — no header, no decoration).
    models_cmd = ("opencode", "models")

    def describe_unlisted_model(
        self,
        model: str,
        *,
        anchor: Optional[str] = None,
        profile: str = PROFILE_AGENT_READONLY,
    ) -> ModelVerdict:
        """Is ``model`` in ``opencode models``' live listing? (D3 step 4)

        **Read in the dispatch's own configuration, or the verdict is wrong.**
        The listing is taken inside the same :meth:`hermetic_dispatch` context
        the dispatch it vets will enter — same ``XDG_CONFIG_HOME`` redirect, same
        ``OPENCODE_CONFIG_CONTENT`` rebuilt from multi-llm's own
        ``config_providers`` — so a provider defined only in the user's own
        ``opencode.json`` is absent here exactly as it would be at dispatch. When
        :func:`hermetic_enabled` is false for ``profile`` the probe runs plain,
        matching dispatch again. ``XDG_CACHE_HOME`` is never redirected (it holds
        ``opencode/models.json``), so this reads the same catalog cache a
        dispatch would rather than paying a cold fetch.

        *Normalization*: the ``:effort`` suffix is split off with the same
        :func:`split_reasoning_effort` call :meth:`build_command` makes before it
        emits ``--variant``, so the token compared is the token dispatched. Then
        whole lines, exact, case-sensitive.

        *Unknown means unknown.* A probe that could not run, a non-zero exit,
        zero parsed slugs, or output that is not a listing at all (a stream whose
        lines are mostly not ``provider/model`` shaped, or that opens with a
        ``timestamp=… level=…`` log line) is ``known=None`` — never ``False``.
        "Not in an empty list" is not a fact. ``known=False`` is reserved for a
        listing that parsed cleanly and does not contain the token, and it
        carries a diagnostic composed *here* from that listing — the ids sharing
        the token's ``provider/`` prefix, nearest first, capped at three, plus
        how many ids that prefix lists. ``opencode models`` prints no "Did you
        mean" line; quoting the CLI's own is D4's job on the dispatch path.
        """
        base_model, _effort = split_reasoning_effort(model or "", REASONING_EFFORTS)
        token = base_model.strip()
        if "/" not in token:
            # Structural — `describe_unresolvable_model` refuses it first.
            return ModelVerdict(known=None)
        try:
            with self.hermetic_dispatch(profile, token, anchor=anchor) as refusal:
                if refusal:
                    # The slot itself is refused before spawn; a listing taken
                    # here would describe a configuration no dispatch will use.
                    return ModelVerdict(known=None)
                env = dict(os.environ)
                for name in self.get_remove_env() or ():
                    env.pop(name, None)
                env.update(self.get_env(token) or {})
                completed = _list_models(list(self.models_cmd), env)
        except Exception:  # noqa: BLE001 - a catalog read may never end a preflight
            return ModelVerdict(known=None)
        if completed is None or completed.returncode != 0:
            return ModelVerdict(known=None)
        listing = _parse_model_listing(completed.stdout or "")
        if listing is None:
            return ModelVerdict(known=None)
        if token in listing:
            return ModelVerdict(known=True)

        prefix = token.split("/", 1)[0]
        siblings = [slug for slug in listing if slug.startswith(prefix + "/")]
        nearest = tuple(
            difflib.get_close_matches(token, siblings, n=3, cutoff=0.0)
        )
        if siblings:
            hint = (
                f"; nearest: {', '.join(nearest)}" if nearest else ""
            )
            detail = (
                f"`opencode models` lists {len(siblings)} id(s) under "
                f"'{prefix}/' and {token!r} is not one of them{hint}."
            )
        else:
            detail = (
                f"`opencode models` lists no ids under '{prefix}/' "
                f"({len(listing)} ids listed in all) — this opencode has no "
                f"provider {prefix!r} configured, or none it can authenticate "
                f"to — so {token!r} cannot resolve."
            )
        return ModelVerdict(known=False, detail=detail, suggestions=nearest)

    # ── M4 — hermetic dispatch ────────────────────────────────────────────

    @contextlib.contextmanager
    def hermetic_dispatch(
        self,
        profile: str,
        model: str = "",
        *,
        anchor: Optional[str] = None,
    ) -> Iterator[Optional[str]]:
        """Scope one dispatch to the run's empty config home (donor §2.3).

        Yields ``None`` normally, or a refusal string when the slot's model names
        a provider that the redirect would make unreconstructible — a silent
        model-resolution failure is the one outcome the redirect may not produce.

        The context is what both argv and env read, so ``--pure`` and the
        ``XDG_CONFIG_HOME`` redirect are emitted together or not at all. Its
        ``finally`` drops this slot's claim on the shared directory, which is
        removed when the last slot leaves — on normal exit, on exception, on a
        per-slot timeout and on ``KeyboardInterrupt`` alike, with the ``atexit``
        sweep as the idempotent backstop.
        """
        if not hermetic_enabled(profile, anchor):
            yield None
            return
        try:
            content = self._hermetic_config_content(model, anchor=anchor)
        except _UnreconstructibleProvider as exc:
            # Raised BEFORE the directory is claimed, so a refusal leaks nothing.
            yield str(exc)
            return
        home = acquire_config_home()
        token = _ACTIVE.set({"config_home": home, "config_content": content})
        try:
            yield None
        finally:
            _ACTIVE.reset(token)
            release_config_home()

    def _hermetic_argv(self) -> List[str]:
        """``[--pure]`` inside an active hermetic dispatch, else ``[]``."""
        if _ACTIVE.get() is None:
            return []
        binary = shutil.which("opencode")
        if binary and PURE_FLAG in _read_help_tokens(binary):
            return [PURE_FLAG]
        _warn_once(
            "pure-unsupported",
            f"multi-llm warning: `opencode run --help` does not list {PURE_FLAG} "
            f"— running without it. Containment is reduced (external plugins "
            f"still load); the config-home redirect still applies.",
        )
        return []

    def _hermetic_config_content(
        self, model: str, *, anchor: Optional[str] = None
    ) -> Optional[str]:
        """The minimal ``OPENCODE_CONFIG_CONTENT`` this slot needs, or None.

        Built from **multi-llm's own** resolved provider config
        (``providers.opencode.config_providers``) and from no other source —
        never copied out of the user's file, which is the config the redirect
        exists to neutralize. Credentials are never carried here; auth stays in
        ``auth.json`` under the retained ``XDG_DATA_HOME``.

        A slot whose model resolves to a built-in provider sets the variable not
        at all. A slot whose provider is defined **only** in the user's own
        opencode config raises :class:`_UnreconstructibleProvider`, because after
        the redirect there is nothing left for it to resolve against.
        """
        base_model, _effort = split_reasoning_effort(model or "", REASONING_EFFORTS)
        if "/" not in base_model:
            return None
        prefix = base_model.split("/", 1)[0].strip()
        if not prefix:
            return None
        declared = _opencode_block(anchor).get("config_providers")
        if isinstance(declared, dict) and isinstance(declared.get(prefix), dict):
            return json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "provider": {prefix: declared[prefix]},
                },
                sort_keys=True,
            )
        if _user_defines_provider(prefix):
            raise _UnreconstructibleProvider(
                f"opencode provider {prefix!r} (from model {base_model!r}) is "
                f"defined only in your own opencode config, which a hermetic run "
                f"replaces — it would not resolve, so this slot is refused rather "
                f"than dispatched into a silent model-resolution failure. Declare "
                f"it under providers.opencode.config_providers (credentials "
                f"excluded — they stay in auth.json), or set "
                f"providers.opencode.hermetic: off to keep today's inheritance."
            )
        return None

    # ── env / argv ────────────────────────────────────────────────────────

    def get_env(self, model: str) -> Dict[str, str]:
        """Adapter env delta; hermetic keys only inside a dispatch context."""
        env = dict(super().get_env(model) or {})
        slot = _ACTIVE.get()
        if slot is None:
            return env
        # Exactly one XDG variable changes. XDG_DATA_HOME (auth.json),
        # XDG_CACHE_HOME (models.json), XDG_STATE_HOME and XDG_RUNTIME_DIR are
        # passed through untouched.
        env["XDG_CONFIG_HOME"] = str(slot["config_home"])
        env[PROJECT_CONFIG_ENV] = "1"
        content = slot.get("config_content")
        if content:
            env["OPENCODE_CONFIG_CONTENT"] = content
        return env

    def get_remove_env(self) -> List[str]:
        """Inherited config carriers, stripped only while hermetic."""
        names = list(super().get_remove_env() or ())
        slot = _ACTIVE.get()
        if slot is None:
            return names
        emitted = {"OPENCODE_CONFIG_CONTENT"} if slot.get("config_content") else set()
        names.extend(name for name in CONFIG_ENV_NAMES if name not in emitted)
        return names

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # opencode run --format json --print-logs --log-level ERROR
        #     --model <model> [--variant <effort>] "<prompt>"
        #
        # --print-logs is the only way to learn why a run produced nothing.
        # A stream error (an exhausted key, say) makes opencode retry
        # internally while writing to neither stream, so the run burns the
        # full timeout and reports no cause. The logs go to stderr and are
        # pinned to ERROR, so stdout stays pure NDJSON and a healthy run
        # stays silent.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None and EFFORT_LADDER.refuses_for_model(base_model, effort):
            # The per-model cap, asked at the ONE place `--variant` is emitted, so
            # both routes to a suffix are covered by one check: a user-typed
            # `--models opencode:<model>:high`, and the host session's own picker
            # variant, which the OpenCode plugin propagates as the same suffix.
            # A capped model is one measured to HANG on `--variant` rather than
            # merely ignore it (see EFFORT_LADDER), so dropping the word is the
            # conservative move: the slot answers at the model's default effort
            # instead of burning its whole timeout writing nothing. Said out loud
            # once per model+effort, because a silently different effort is the
            # defect this propagation exists to fix.
            _warn_once(
                f"variant-capped:{base_model}:{effort}",
                f"multi-llm warning: dropping `--variant {effort}` for "
                f"{base_model} — that model is declared not to accept a "
                f"reasoning effort (measured to hang on one); dispatching at "
                f"its default effort instead",
            )
            effort = None
        cmd = [
            "opencode", "run", "--format", "json",
            "--print-logs", "--log-level", "ERROR",
            # `--pure` is rendered only inside an active hermetic dispatch
            # context, i.e. only when the paired `XDG_CONFIG_HOME` redirect is
            # also being applied. Outside one this is the byte-identical argv
            # this adapter built before M4.
            *self._hermetic_argv(),
            "--model", base_model,
        ]
        if effort is not None:
            cmd += ["--variant", effort]
        cmd.append(prompt)
        return cmd

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
        cwd: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Deliver the prompt on argv or on stdin (prompt omitted from argv)."""
        if transport != "stdin":
            return super().build_command_for_transport(
                prompt, model, transport, prompt_file, timeout_s=timeout_s,
                schema=schema,
                sandbox_fallback=sandbox_fallback,
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        return cmd[:-1], prompt

    def build_agent_command(
        self,
        prompt: str,
        model: str,
        profile: str = PROFILE_AGENT_READONLY,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build ``opencode run`` under the profile's own agent (D3e).

        ``agent-readonly`` runs the built-in ``plan`` agent (every edit tool
        denied); ``agent-write`` runs the built-in ``build`` agent, named
        explicitly rather than inherited from opencode's default. Either way the
        permission set is opencode's own, and ``cwd`` is pinned by the executor.
        """
        if profile == PROFILE_AGENT_WRITE:
            cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
            return [*cmd[:2], "--agent", WRITE_AGENT, *cmd[2:]]
        if profile != PROFILE_AGENT_READONLY:
            return super().build_agent_command(
                prompt, model, profile, timeout_s=timeout_s, schema=schema
            )
        cmd = self.build_command(prompt, model, timeout_s=timeout_s, schema=schema)
        # `--agent` belongs to the `run` subcommand, so it goes after
        # `["opencode", "run"]` — never before it.
        return [*cmd[:2], "--agent", READONLY_AGENT, *cmd[2:]]

    def parse_usage(self, stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
        """Sum per-step usage across the whole ``run --format json`` stream (D3j).

        OpenCode emits one assistant message per step — dozens for a long run —
        and the token counters live on each ``step_finish`` part
        (``tokens: {input, output, reasoning, cache: {read, write}}`` plus
        ``cost``, verified against the part schema in the 1.18.x bundle). Reading
        the final message alone under-reports a multi-step run, which is the
        specific bug this method exists to prevent, so every step is summed.
        ``tool_uses`` is the count of ``tool_use`` events — a real tool count, not
        an inference.
        """
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost_usd": 0.0,
        }
        seen = {key: False for key in totals}
        tool_uses = 0
        saw_tool_event = False

        def _num(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            return None

        for event in self._iter_events(stdout):
            event_type = event.get("type")
            if event_type == "tool_use":
                saw_tool_event = True
                tool_uses += 1
                continue
            if event_type != "step_finish":
                continue
            part = event.get("part")
            if not isinstance(part, dict) or part.get("type") != "step-finish":
                continue
            cost = _num(part.get("cost"))
            if cost is not None:
                totals["cost_usd"] += cost
                seen["cost_usd"] = True
            tokens = part.get("tokens")
            if not isinstance(tokens, dict):
                continue
            step_input = _num(tokens.get("input"))
            if step_input is not None:
                totals["input_tokens"] += step_input
                seen["input_tokens"] = True
            step_output = _num(tokens.get("output"))
            step_reasoning = _num(tokens.get("reasoning")) or 0
            if step_output is not None:
                # opencode's own roll-up folds reasoning into output; mirror it.
                totals["output_tokens"] += step_output + step_reasoning
                seen["output_tokens"] = True
            cache = tokens.get("cache")
            if isinstance(cache, dict):
                cache_read = _num(cache.get("read"))
                if cache_read is not None:
                    totals["cached_input_tokens"] += cache_read
                    seen["cached_input_tokens"] = True

        out: Dict[str, Any] = {
            key: value for key, value in totals.items() if seen[key]
        }
        if "input_tokens" in out and "output_tokens" in out:
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        if saw_tool_event:
            out["tool_uses"] = tool_uses
        return out or None

    @staticmethod
    def _iter_events(stdout: str):
        """Yield the well-formed JSON objects in an NDJSON stream."""
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
            if isinstance(event, dict):
                yield event

    @staticmethod
    def _direct_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse ``text`` as one complete JSON document, or return None."""
        if text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return None

    def classify_auth_probe(
        self, stdout: str, stderr: str, exit_code: int = 0
    ) -> Optional[bool]:
        """Read ``opencode auth list`` (§5.3/§7.2: exit 0 carries no verdict).

        The base implementation reports ``True`` on a clean exit with no login
        notice — and ``auth list`` *always* exits 0 and never prints one, so a
        logged-out opencode was being recorded READY and discovered one dispatch
        at a time. The count line is the verdict, and it is the only one:

        * ``0 credentials`` with no ``Environment`` section — nothing stored and
          nothing in the environment: ``False``, the one verdict that refuses a
          run, so it is drawn from the narrowest possible evidence.
        * ``0 credentials`` **plus** ``N environment variable(s)`` — opencode
          found a provider key in the environment and really can authenticate,
          so this must not refuse (it would fail closed on a working setup).
        * anything else falls through to the base rule.
        """
        haystack = f"{stdout or ''}\n{stderr or ''}".lower()
        if self._EMPTY_CREDENTIALS.search(haystack) and not self._ENVIRONMENT_KEYS.search(
            haystack
        ):
            return False
        return super().classify_auth_probe(stdout, stderr, exit_code)

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """Recover the reason for a non-zero exit from the event stream.

        OpenCode writes ``error`` events to *stdout* and leaves stderr empty,
        so without this the caller sees only "exited with code 1". Verified
        against a live auth failure, which produced a 304-byte error event on
        stdout and zero bytes on stderr.

        **stdout is the declared channel; stderr is enrichment (D5).** Since
        1.18.x the stdout event can be a wrapper with an ``err_…`` ref and no
        cause — ``UnknownError: Unexpected server error. Check server logs for
        details.`` — while the underlying exception goes to stderr under the
        ``--print-logs --log-level ERROR`` we pass, as a
        ``level=ERROR … error="<text>"`` line. When the stdout message has that
        wrapper shape, the first such stderr value is appended in parentheses,
        so the composed text extends the stdout-only one rather than replacing
        it (``tests/fixtures/opencode/invalid-model.*``). A stderr-only capture
        recovers nothing: the composition is gated on the wrapper being present.

        Diagnostics only, never classification: no marker table is consulted
        here and nothing in the composed string invites a caller to branch on
        it — ``classify_failure`` reads the same bytes for its own verdict.
        """
        messages: List[str] = []
        for event in self._iter_events(stdout):
            if event.get("type") != "error":
                continue
            error = event.get("error")
            if not isinstance(error, dict):
                continue
            name = error.get("name")
            data = error.get("data")
            detail = data.get("message") if isinstance(data, dict) else None
            parts = [p for p in (name, detail) if isinstance(p, str) and p]
            if parts:
                messages.append(": ".join(parts))
        if not messages:
            return None
        if any(self._SERVER_ERROR_WRAPPER in m for m in messages):
            cause = self._stderr_error_value(stderr)
            if cause:
                messages = [
                    f"{m} ({cause})" if self._SERVER_ERROR_WRAPPER in m else m
                    for m in messages
                ]
        return "; ".join(messages)

    #: The stdout wrapper whose cause lives on stderr (opencode ≥ 1.18.x).
    _SERVER_ERROR_WRAPPER = "UnknownError: Unexpected server error"

    #: One ``error="…"`` value on a log line. Anchored on a literal and free of
    #: nested quantifiers, so a 40 KB single-character line costs one linear
    #: scan and no backtracking (``test_every_override_survives_garbage_…``).
    _STDERR_ERROR_VALUE = re.compile(r'(?:^|\s)error="([^"\n]*)"')

    @classmethod
    def _stderr_error_value(cls, stderr: str) -> Optional[str]:
        """The first ``error="…"`` value on a ``level=ERROR`` stderr line."""
        for line in (stderr or "").splitlines():
            if "level=ERROR" not in line:
                continue
            match = cls._STDERR_ERROR_VALUE.search(line)
            if match and match.group(1).strip():
                return match.group(1).strip()
        return None

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse NDJSON event stream output from OpenCode.

        OpenCode emits one assistant message per step, so a run that uses
        tools produces many ``text`` events — real sessions reach 47. The
        final answer is the last one; earlier ones are narration that may
        itself contain a JSON code block, so extracting from the concatenation
        would return a draft rather than the answer.

        Candidates are therefore ordered by confidence rather than position:
        an exact parse of the whole stream or of the final message is
        unambiguous and is tried first, and only then the loose extraction
        that can match a fragment or an early draft.
        """
        text_parts: List[str] = []
        for event in self._iter_events(stdout):
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                text = part["text"]
                if isinstance(text, str):
                    text_parts.append(text)

        if not text_parts:
            error = "No text events found in output"
            reason = self.describe_failure(stdout, stderr)
            if reason:
                error = f"OpenCode reported an error: {reason}"
            return {"success": False, "error": error, "raw": stdout, "data": None}

        last = text_parts[-1].strip()
        concatenated = "".join(text_parts).strip()
        # Newline-joined for the loose pass: separate steps are separate
        # messages, and gluing them directly runs the end of one into the
        # start of the next.
        joined = "\n".join(text_parts).strip()

        if not concatenated:
            return {"success": False, "error": "Empty text response", "raw": stdout, "data": None}

        # Exact parses first — they are unambiguous. The concatenation covers
        # a single document that arrived split across parts (a split can fall
        # mid-token, so only a separator-less join reassembles it); `last`
        # covers the ordinary case where the closing message is the answer.
        for candidate in (concatenated, last):
            if candidate:
                direct = self._direct_json(candidate)
                if direct is not None:
                    return direct

        # Then the loose pass, final message first so a late answer wins over
        # a draft quoted earlier in the run.
        result: Dict[str, Any] = {}
        for candidate in (last, joined):
            if not candidate:
                continue
            # Try to extract JSON from the text (may be in code blocks)
            result = extract_json_from_text(candidate, prefer_arrays=True)
            if result.get("success"):
                return result

        return result
