"""Cursor-agent CLI provider implementation."""

import atexit
import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple

from .base import (
    PROFILE_GENERATE,
    USAGE_PARTIAL,
    LLMProvider,
    StructuredSchema,
    git_env,
)
from .result_envelope import parse_result_envelope

# ---------------------------------------------------------------------------
# §5.4.1 — the per-flag capability gate, in its **load-bearing** form
# ---------------------------------------------------------------------------

#: The two tokens that replace ``-f`` (M1/§3.1, atomic group **A5**). They are
#: inseparable and both are load-bearing:
#:
#: * without ``--trust`` the launch dies on a workspace-trust gate — measured at
#:   exit 1, 0.43 s, zero stdout, ``⚠ Workspace Trust Required … Pass --trust,
#:   --yolo, or -f``;
#: * ``--trust`` alone still **writes files**, because ``--print`` is documented
#:   as having "access to all tools, including write and shell". ``--mode ask``
#:   is the only one of the two that actually refuses ("I'm in Ask mode, so I
#:   can't create or edit files").
#:
#: So there is deliberately **no degraded argv**: dropping ``--trust`` fails the
#: launch, and the only other way to satisfy the trust gate is ``-f``/``--yolo``,
#: which is the write bypass this adapter exists to stop passing. When the gate
#: cannot confirm both tokens the slot is refused before spawn.
CURSOR_REQUIRED_FLAGS: Tuple[str, ...] = ("--trust", "--mode")

#: §5.1's state name for a CLI too old for a flag the dispatch needs. cursor
#: declares ``version_cmd = ()`` / ``min_version = ""`` (nothing is version-gated
#: at all), so the *help text* stands in for the version read — see
#: :func:`_help_flag_tokens`. M14 later absorbs this one-off into the shared
#: floor/ceiling machinery.
VERSION_TOO_LOW = "version-too-low"

#: How long the one help read may take. §2.1's probe containment.
_HELP_TIMEOUT_S = 5

#: A whole-word long-option token. ``--model`` must never be read as ``--mode``,
#: which a bare substring test would do — and confirming the wrong flag is
#: exactly the failure this gate exists to prevent.
_FLAG_TOKEN_RE = re.compile(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9-]*)(?![\w-])")

_FLAG_CACHE: Dict[str, FrozenSet[str]] = {}
_FLAG_CACHE_LOCK = threading.Lock()


class CursorFlagsUnconfirmed(RuntimeError):
    """The installed cursor-agent could not be shown to accept both tokens.

    Raised **before** any spawn, from :meth:`CursorAgentProvider.build_command`,
    which is called outside every caller's ``try`` on purpose: the fan-out
    records it as this model's failure (``run_single_model``'s per-model
    ``except``) and the rest of the fan-out is unaffected. That is the named
    refusal §5.4.1 requires — never a silent downgrade to a writable fan-out.
    """

    def __init__(self, missing: Tuple[str, ...], binary: str) -> None:
        self.missing = tuple(missing)
        self.binary = binary
        self.state = VERSION_TOO_LOW
        super().__init__(
            f"cursor-agent slot refused ({VERSION_TOO_LOW}): {binary} does not list "
            f"{', '.join(self.missing)} in its --help, and multi-llm will not run "
            f"cursor without {' + '.join(CURSOR_REQUIRED_FLAGS)} "
            "(--trust satisfies the workspace-trust gate, --mode ask is what refuses "
            "writes; the only other argv that launches is the -f/--yolo write bypass). "
            "Upgrade cursor-agent, or drop cursor from --models."
        )


def _probe_help_tokens(binary: str) -> FrozenSet[str]:
    """Every long-option token cursor-agent's own ``--help`` lists.

    Containment matches §2.1's probe: ``stdin=DEVNULL``, captured streams, a 5 s
    timeout and no network. The env comes from the same shared builder both
    spawn lanes use (:func:`base.git_env` under the ``generate`` profile) so Lane
    A and Lane B can never disagree about which flags a run emits, plus
    ``TERM=dumb``/``COLUMNS``/``NO_COLOR`` — a *truncated* help rendering hiding
    what a flag required is a failure mode this repo has already been bitten by
    once (§3.6), and a token sniff must not inherit it.

    Both streams are read **concatenated** (§5.4): some CLIs banner to stderr.

    Failure posture: a timeout, a non-zero exit, an unreadable stream or an empty
    help all resolve to "unsupported" — the empty set. For an *optional* flag
    that means omit-and-warn; for cursor's two it means the caller refuses the
    slot, because there is no safe argv to fall back to.
    """
    env = dict(os.environ)
    env.update(git_env(PROFILE_GENERATE))
    env.update({"TERM": "dumb", "COLUMNS": "1000", "NO_COLOR": "1"})
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HELP_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    help_text = f"{result.stdout or ''}\n{result.stderr or ''}"
    if not help_text.strip():
        return frozenset()
    return frozenset(_FLAG_TOKEN_RE.findall(help_text))


def _help_flag_tokens(binary: str) -> FrozenSet[str]:
    """:func:`_probe_help_tokens`, memoised on the resolved binary path.

    One help spawn per CLI per process, not one per flag: a four-model ×
    ten-phase run performs exactly one.
    """
    with _FLAG_CACHE_LOCK:
        cached = _FLAG_CACHE.get(binary)
    if cached is not None:
        return cached
    tokens = _probe_help_tokens(binary)
    with _FLAG_CACHE_LOCK:
        _FLAG_CACHE.setdefault(binary, tokens)
        return _FLAG_CACHE[binary]


def reset_flag_cache() -> None:
    """Forget the memoised help read (tests only)."""
    with _FLAG_CACHE_LOCK:
        _FLAG_CACHE.clear()


def _gateable_binary() -> Optional[str]:
    """The cursor-agent the gate would read, or ``None`` when there is none.

    ``shutil.which`` already tests for an executable, so the second half is a
    no-op in production and a guard against a resolver that points somewhere
    unrunnable: a path that cannot be spawned cannot reject a flag either, and
    turning "not installed" into "version-too-low" would only lose information.
    """
    binary = shutil.which("cursor-agent")
    if binary is None:
        return None
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        return None
    return binary


# ---------------------------------------------------------------------------
# Q11 — the residue ``--trust`` and ``--model`` leave in the user's profile
# ---------------------------------------------------------------------------
#
# Measured 2026-08-16. A single `--print --trust --mode ask --model <m>` run
# leaves **two** things behind in the user's own cursor profile, neither of
# which is multi-llm's to keep:
#
#   1. `<data>/projects/<slug>/.workspace-trusted` — 206 B of
#      `{"trustedAt":…,"workspacePath":…,"trustMethod":"cli-flag"}`. Markers
#      accumulate and are never garbage-collected (seven pre-existing ones on the
#      probe machine), and a marker also trusts that workspace for the user's own
#      later *interactive* cursor-agent sessions, which run with full tool
#      permissions.
#   2. `<config>/cli-config.json`'s `model` / `selectedModel` /
#      `modelSelectionHistory` (plus `modelParameters` and
#      `hasChangedDefaultModel`) — `--model` **rewrites the user's global default
#      model**. This is a side effect of `--model`, independent of `--trust`, and
#      a fan-out of several cursor rows would otherwise leave the user's
#      interactive CLI on whichever row finished last.
#
# **The primary mechanism is the per-run profile redirect below**, not this
# snapshot/restore guard. The guard is correct *within* one process (first arm
# wins, so two rows share the earliest clean snapshot) and **broken across two
# overlapping processes**: `_ARMED` is a module-level global, so a second
# orchestrator that arms after the first has already spawned a cursor row
# snapshots the *polluted* value as if it were the user's, and whichever process
# exits last writes that back — measured to move a user from `gpt-5.6-sol` to
# `composer-2.5` permanently. No in-process guard can see another process's
# baseline, so the fix is not to snapshot better but to never touch the user's
# files: see `isolate_profile_env`.
#
# What survives here is the **fallback**, active only when the redirect is off
# (`providers.cursor-agent.isolate_profile: off`) or refused. Its rules are
# deliberately narrow. Nothing outside the exact paths a snapshot recorded is
# ever touched; a marker that already existed is never removed (the user, or an
# earlier run, trusted that workspace on purpose); and a `cli-config.json` key is
# only put back when it actually moved.

_TRUST_MARKER_NAME = ".workspace-trusted"
_CLI_CONFIG_NAME = "cli-config.json"
_TRUST_METHOD_OURS = "cli-flag"

#: Every key a ``--model`` run rewrites. Everything else in ``cli-config.json``
#: is cursor's own bookkeeping (statsig, privacy and server-config caches) and is
#: left exactly as the run left it — reverting the whole file would also revert
#: those, and would clobber a concurrent interactive session's writes.
#:
#: ``modelParameters`` is keyed **per model id**, so a run adds its own model's
#: entry rather than replacing the user's: restoring it therefore means putting
#: the *whole mapping* back to its pre-run value, which is what the generic
#: per-key restore below already does (merging would leave the run's entry
#: behind). ``hasChangedDefaultModel`` is a bare flag cursor flips the first time
#: ``--model`` moves the default; a run that flips it tells the user's next
#: interactive session that *they* changed it. Both were leaking.
_MODEL_KEYS: Tuple[str, ...] = (
    "model",
    "selectedModel",
    "modelSelectionHistory",
    "modelParameters",
    "hasChangedDefaultModel",
)


def _cursor_config_dir() -> str:
    """Where ``cli-config.json`` lives, resolved exactly as cursor-agent does."""
    explicit = (os.environ.get("CURSOR_CONFIG_DIR") or "").strip()
    if explicit:
        return explicit
    xdg = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        return os.path.join(xdg, "cursor")
    return os.path.join(os.path.expanduser("~"), ".cursor")


def _cursor_data_dir() -> str:
    """Where ``projects/<slug>/`` lives, resolved exactly as cursor-agent does.

    Note this is a **different** env var from :func:`_cursor_config_dir`'s.
    """
    explicit = (os.environ.get("CURSOR_DATA_DIR") or "").strip()
    if explicit:
        return explicit
    return os.path.join(os.path.expanduser("~"), ".cursor")


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _detect_indent(payload: bytes) -> int:
    """The indent width the file already uses, so a rewrite keeps its shape."""
    for line in payload.decode("utf-8", errors="replace").splitlines():
        stripped = line.lstrip(" ")
        if stripped and stripped != line and not stripped.startswith("\t"):
            return len(line) - len(stripped)
    return 2


class CursorResidueSnapshot:
    """What the user's cursor profile looked like before this process spawned cursor.

    Recorded **before** the spawn (that is the whole point: pre-existence is not
    recoverable afterwards) and put back by :meth:`restore`, which is idempotent
    and never raises — a cleanup that can fail a run is worse than the residue.
    """

    def __init__(
        self,
        *,
        projects_dir: str,
        markers: FrozenSet[str],
        config_path: str,
        config_bytes: Optional[bytes],
        config_mode: Optional[int],
        workspaces: Tuple[str, ...],
    ) -> None:
        self.projects_dir = projects_dir
        self.markers = markers
        self.config_path = config_path
        self.config_bytes = config_bytes
        self.config_mode = config_mode
        self._workspaces: List[str] = list(workspaces)
        self._lock = threading.Lock()

    @property
    def workspaces(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._workspaces)

    def add_workspace(self, workspace: str) -> None:
        """Also clean markers for ``workspace`` (a second row, another cwd)."""
        resolved = os.path.realpath(workspace)
        with self._lock:
            if resolved not in self._workspaces:
                self._workspaces.append(resolved)

    # -- restore ---------------------------------------------------------

    def restore(self) -> List[str]:
        """Undo what this process's cursor runs changed. Never raises."""
        actions: List[str] = []
        for step in (self._restore_markers, self._restore_config):
            try:
                actions.extend(step())
            except Exception:  # noqa: BLE001 - cleanup must never fail a run
                continue
        return actions

    def _restore_markers(self) -> List[str]:
        removed: List[str] = []
        pattern = os.path.join(self.projects_dir, "*", _TRUST_MARKER_NAME)
        for path in glob.glob(pattern):
            if not self._marker_is_ours(path):
                continue
            try:
                os.remove(path)
            except OSError:
                continue
            removed.append(path)
        return removed

    def _marker_is_ours(self, path: str) -> bool:
        """Three conditions, all required, none of them inferred after the fact.

        The marker must have been **absent** when this process armed (a marker
        the user already had is theirs), must carry cursor's own
        ``trustMethod: cli-flag``, and must name a workspace this process
        actually ran in — its own cwd, or a directory containing it, which is the
        repo root a pinned dispatch uses.

        ``trustMethod`` is the sharpest of the three, and it was verified against
        real markers rather than assumed: a marker written by ``--trust`` carries
        ``"trustMethod": "cli-flag"``, while one the user acquired another way
        omits the field entirely (observed on a 2026-08-04 marker). So a build
        that stops writing the field makes this cleanup a no-op — residue stays,
        nothing of the user's is ever removed, which is the safe direction.
        """
        if path in self.markers:
            return False
        payload = _read_bytes(path)
        if payload is None:
            return False
        try:
            document = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return False
        if not isinstance(document, dict):
            return False
        if document.get("trustMethod") != _TRUST_METHOD_OURS:
            return False
        declared = document.get("workspacePath")
        if not isinstance(declared, str) or not declared:
            return False
        # realpath on BOTH sides of the comparison (D1c): asymmetry is the bug.
        marked = os.path.realpath(declared)
        for workspace in self.workspaces:
            if workspace == marked or workspace.startswith(marked + os.sep):
                return True
        return False

    def _restore_config(self) -> List[str]:
        current = _read_bytes(self.config_path)
        if current is None:
            return []
        if self.config_bytes is not None and current == self.config_bytes:
            return []
        try:
            current_document = json.loads(current.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            # Not ours to repair, and rewriting it would destroy whatever the
            # CLI is mid-write on.
            return []
        if not isinstance(current_document, dict):
            return []
        if self.config_bytes is None:
            original_document: Dict[str, Any] = {}
        else:
            try:
                loaded = json.loads(self.config_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                return []
            if not isinstance(loaded, dict):
                return []
            original_document = loaded

        restored = dict(current_document)
        moved: List[str] = []
        for key in _MODEL_KEYS:
            if key in original_document:
                if restored.get(key) != original_document[key]:
                    restored[key] = original_document[key]
                    moved.append(key)
            elif key in restored:
                del restored[key]
                moved.append(key)
        if not moved:
            return []

        if self.config_bytes is not None and restored == original_document:
            payload = self.config_bytes  # byte-for-byte, formatting included
        else:
            indent = _detect_indent(self.config_bytes or current)
            trailing = "\n" if (self.config_bytes or current).endswith(b"\n") else ""
            payload = (
                json.dumps(restored, indent=indent, ensure_ascii=False) + trailing
            ).encode("utf-8")
        self._write_atomically(payload)
        return [f"{self.config_path}:{key}" for key in moved]

    def _write_atomically(self, payload: bytes) -> None:
        directory = os.path.dirname(self.config_path) or "."
        handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=".cli-config.", suffix=".tmp", delete=False
        )
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if self.config_mode is not None:
                os.chmod(handle.name, self.config_mode)
            os.replace(handle.name, self.config_path)
        except OSError:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise


def snapshot_side_effects(workspace: Optional[str] = None) -> CursorResidueSnapshot:
    """Record the pre-spawn state of the two paths a cursor run rewrites."""
    projects_dir = os.path.join(_cursor_data_dir(), "projects")
    config_path = os.path.join(_cursor_config_dir(), _CLI_CONFIG_NAME)
    markers = frozenset(glob.glob(os.path.join(projects_dir, "*", _TRUST_MARKER_NAME)))
    config_bytes = _read_bytes(config_path)
    config_mode: Optional[int] = None
    try:
        config_mode = os.stat(config_path).st_mode & 0o777
    except OSError:
        config_mode = None
    return CursorResidueSnapshot(
        projects_dir=projects_dir,
        markers=markers,
        config_path=config_path,
        config_bytes=config_bytes,
        config_mode=config_mode,
        workspaces=(os.path.realpath(workspace or os.getcwd()),),
    )


_GUARD_LOCK = threading.Lock()
_ARMED: Optional[CursorResidueSnapshot] = None
_ATEXIT_REGISTERED = False


def arm_side_effect_guard(workspace: Optional[str] = None) -> CursorResidueSnapshot:
    """Snapshot once per process, at the first cursor spawn, and restore at exit.

    **First arm wins.** Under a concurrent fan-out a later row's reading is
    already polluted by an earlier row, so taking the earliest snapshot is what
    makes the restore converge on the user's real state rather than on another
    row's model. Later arms only widen the set of workspaces whose markers are
    ours to clean.

    ``atexit`` is the process-level ``finally``: it covers a normal exit and an
    unhandled exception, which is as far as an in-process guard reaches. A
    ``SIGKILL``'d run leaves the residue behind — the next run's restore does not
    pick it up either, because by then the marker is *pre-existing* and therefore
    out of bounds by the rule above. That is the safe direction to fail.
    """
    global _ARMED, _ATEXIT_REGISTERED
    with _GUARD_LOCK:
        if _ARMED is not None:
            armed = _ARMED
        else:
            armed = snapshot_side_effects(workspace=workspace)
            _ARMED = armed
            if not _ATEXIT_REGISTERED:
                atexit.register(restore_side_effects)
                _ATEXIT_REGISTERED = True
    armed.add_workspace(workspace or os.getcwd())
    return armed


def armed_snapshot() -> Optional[CursorResidueSnapshot]:
    """The snapshot this process is holding, if any."""
    with _GUARD_LOCK:
        return _ARMED


def restore_side_effects() -> List[str]:
    """Put the user's cursor profile back. Idempotent; safe to call at any time."""
    global _ARMED
    with _GUARD_LOCK:
        armed = _ARMED
        _ARMED = None
    return armed.restore() if armed is not None else []


def reset_side_effect_guard() -> None:
    """Drop the armed snapshot without restoring (tests only)."""
    global _ARMED
    with _GUARD_LOCK:
        _ARMED = None


# ---------------------------------------------------------------------------
# The per-run profile redirect — the primary mechanism (§2.3's shape)
# ---------------------------------------------------------------------------
#
# Give the run its own cursor profile and the residue problem stops existing:
# cursor never opens the user's `cli-config.json`, so there is nothing to
# snapshot, nothing for a second overlapping process to race on, and nothing to
# restore. The cross-process defect described above is not mitigated by this, it
# is **unreachable**.
#
# **Why this is safe, and why the earlier "logged-out CLI" worry was wrong.**
# cursor-agent 2026.08.11 resolves three things by three different rules:
#
#   * `cli-config.json` ← `CURSOR_CONFIG_DIR` → `$XDG_CONFIG_HOME/cursor` → `~/.cursor`
#   * `projects/<slug>/` ← `CURSOR_DATA_DIR`   → `~/.cursor`
#   * `auth.json`       ← a **third** rule that consults *neither* variable:
#     `$XDG_CONFIG_HOME/cursor/auth.json` on Linux (verified 0600 on the probe
#     machine), `~/.cursor` on macOS, `%APPDATA%\Cursor` on Windows.
#
# So redirecting the first two leaves credentials exactly where the CLI looks for
# them. `XDG_CONFIG_HOME` itself is therefore **never** redirected here — doing
# that *would* move `auth.json` on Linux and disable the provider outright. The
# only two variables this code writes are the two named above, and a test asserts
# that the env delta contains no third key.

#: ``providers.cursor-agent.isolate_profile``. Mirrors §2.3's
#: ``providers.opencode.hermetic``.
ISOLATE_AUTO = "auto"
ISOLATE_ON = "on"
ISOLATE_OFF = "off"
ISOLATE_VALUES: Tuple[str, ...] = (ISOLATE_AUTO, ISOLATE_ON, ISOLATE_OFF)
#: ``auto``, not ``on``, and the asymmetry of the two failure modes is why.
#: Only Linux's auth rule was *measured*; macOS and Windows are read off the
#: bundle. If that reading is wrong on either, ``on`` presents every cursor slot
#: as logged out — a disabled provider — whereas ``auto`` falls back to the
#: snapshot/restore guard, which is exactly the behaviour that shipped before
#: this change: correct within a run, broken across overlapping runs. Strictly
#: no worse, on the platforms nobody could test. This follows the same rule
#: §2.4 applies to the env allowlist ("opt-in per platform until that platform's
#: column is green") and §3.5 applies to unprobed rows. Set ``isolate_profile:
#: on`` to opt a non-Linux host in, or widen ``_AUTH_UNAFFECTED_PLATFORMS``
#: once that platform is measured.
ISOLATE_DEFAULT = ISOLATE_AUTO

#: Platforms where cursor's *auth* rule is **verified** not to follow either
#: redirected variable. ``auto`` redirects only here; ``on`` redirects
#: everywhere. Widening this set is a measurement, not an opinion.
_AUTH_UNAFFECTED_PLATFORMS: Tuple[str, ...] = ("linux",)

#: ------------------------------------------------------------------------
#: THE decision point. Flipping this one line is the whole per-run/per-spawn
#: switch; :func:`_profile_dir_for_spawn` is the only reader, and the tests in
#: ``tests/test_cursor_profile_isolation.py::TestScopeIsParameterized`` cover
#: both values. The parameterization stays; the **default is now measured**.
#:
#: * ``per-run`` (default, §2.3's shape) — one process-scoped directory shared by
#:   every cursor row.
#: * ``per-spawn`` — one directory per child.
#:
#: **The live probe ran, and the answer is that parameters do not bleed: argv
#: wins completely.** cursor's model list is fully pre-parameterized — effort,
#: thinking and fast are baked into one atomic slug (``gpt-5.5-extra-high``,
#: ``claude-opus-5-thinking-high-fast``) and the rest ride on argv
#: (``'claude-opus-4-8[context=1m,effort=high,fast=false]'``). A deliberately
#: hostile seeded config (``selectedModel = claude-opus-5-thinking-max``,
#: ``maxMode: true``, and ``modelParameters`` for the argv slug set to
#: ``context:1m, reasoning:xhigh, effort:max, thinking:true``) produced a run
#: **equivalent** to one from an empty config: same
#: ``providerOptions.cursor.modelName``, an empty reasoning block, and a scan
#: for ``opus``/``xhigh``/``effort``/``1m``/``thinking``/``maxMode`` found none
#: in either. The run even *reset* the seeded ``maxMode: true`` to ``false``.
#: **The config file is where the resolved argv lands — it is output, never
#: input**, so concurrent rows racing on one shared file cannot alter each
#: other's model, effort, thinking, fast or max-mode.
#:
#: Per-spawn is therefore not merely unnecessary, it is now actively **worse**,
#: for two measured reasons:
#:
#: 1. **Cost.** A cold profile dir costs ~3.35 s (8.85 s vs 5.50 s warm),
#:    mostly re-fetching the 703 KB statsig cache. Per-run pays that once;
#:    per-spawn charges every child, so a 4-row fan-out burns ~13 s
#:    re-downloading the same cache four times.
#: 2. **The leaked ``worker-server``.** cursor-agent spawns a detached
#:    ``worker-server`` daemon that outlives the CLI, reparents to init and
#:    **inherits the redirect env** (verified: its ``CURSOR_CONFIG_DIR``
#:    pointed into the scratch dir). A per-spawn directory would be torn down
#:    at *row* end — while that row's worker is still live and holding it.
#:    Per-run keeps removal at process exit, which is after every row; see
#:    :func:`cleanup_isolated_profiles` for the residual that still leaves.
#: ------------------------------------------------------------------------
SCOPE_PER_RUN = "per-run"
SCOPE_PER_SPAWN = "per-spawn"
PROFILE_DIR_SCOPE: str = SCOPE_PER_RUN

#: Prefix under ``TMPDIR``, so a leaked directory is sweepable by name.
_PROFILE_PREFIX = "multi-llm-cursor-"

_PROFILE_LOCK = threading.Lock()
#: Every directory **this process created**. Nothing outside this list is ever
#: removed — a user-supplied profile is respected, never deleted.
_PROFILE_DIRS: List[str] = []
_RUN_PROFILE_DIR: Optional[str] = None
_PROFILE_ATEXIT_REGISTERED = False

#: The two variables the redirect writes, and the only two it reads to decide
#: whether the user has already pinned a profile of their own.
_PROFILE_ENV_VARS: Tuple[str, ...] = ("CURSOR_CONFIG_DIR", "CURSOR_DATA_DIR")


def _provider_config(anchor: Optional[str] = None) -> Dict[str, Any]:
    """This adapter's own block from the layered config.

    Read through the normal layering (base → auto-discovered
    ``<git-root>/.multi-llm/providers.yaml`` → ``MULTI_LLM_PROVIDERS_CONFIG``),
    so the switch lives exactly where every other provider setting does.
    ``resolve_env`` forwards the run's anchor when it has one; otherwise this
    falls back to the pinned run anchor, §9.3.b's documented backstop.
    """
    from ..provider_registry import get_run_anchor, load_config  # import cycle

    try:
        config = load_config(anchor=anchor if anchor is not None else get_run_anchor())
    except Exception:  # noqa: BLE001 - a config failure must not decide this
        return {}
    block = (config.get("providers") or {}).get("cursor-agent")
    return block if isinstance(block, dict) else {}


def isolate_profile_setting(anchor: Optional[str] = None) -> str:
    """``auto`` | ``on`` | ``off``, defaulting to :data:`ISOLATE_DEFAULT`.

    YAML 1.1 turns a bare ``on``/``off`` into a bool, so both spellings —
    ``isolate_profile: off`` and ``isolate_profile: "off"`` — resolve to the same
    enum value. Anything unrecognised falls back to the default rather than
    failing the run: this switch decides *how well* a run is contained, and an
    unreadable value must never be the reason a fan-out does not start.
    """
    raw = _provider_config(anchor).get("isolate_profile", ISOLATE_DEFAULT)
    if isinstance(raw, bool):
        return ISOLATE_ON if raw else ISOLATE_OFF
    value = str(raw).strip().lower()
    return value if value in ISOLATE_VALUES else ISOLATE_DEFAULT


def user_pinned_profile_dirs() -> Tuple[str, ...]:
    """Which of the two variables the *user* already set, if any.

    A user who exports ``CURSOR_CONFIG_DIR`` or ``CURSOR_DATA_DIR`` is pinning
    their own cursor profile on purpose. That is respected verbatim: the run
    does **not** redirect (the fallback guard takes over instead), and the
    directory they named is never seeded, never written and never removed.
    """
    return tuple(name for name in _PROFILE_ENV_VARS if (os.environ.get(name) or "").strip())


def _seed_profile_dir(path: str) -> None:
    """Put whatever a fresh cursor profile *needs* into ``path``. One function.

    **A no-op, and the live probe is why — no seed is required.** Measured with
    both ``CURSOR_CONFIG_DIR`` and ``CURSOR_DATA_DIR`` pointed at one **empty**
    directory: a real cursor-agent run exited **0**, authenticated normally and
    wrote **0 bytes to stderr** — no login prompt, no first-run setup, no
    complaint about the missing ``cli-config.json``. ``auth.json`` was
    byte-identical before and after (it follows neither variable, as above) and
    the user's ``~/.cursor/projects`` trust markers were untouched. The CLI
    built what it needed on its own: ``cli-config.json``,
    ``statsig-cache.json`` (703 KB), ``chats/<hash>/<session>/{meta.json,
    store.db}`` and ``projects/<slug>/{.workspace-trusted, mcp-auth.json,
    agent-transcripts/…}``. So the empty 0700 directory *is* the seed, and this
    function stays empty. The one measured cost is time, not correctness: a
    cold directory ran in **8.85 s against 5.50 s warm**, a ~3.35 s penalty
    spent mostly re-fetching the statsig cache — which is an argument about
    *how many* directories a run creates (see :data:`PROFILE_DIR_SCOPE`), not
    about seeding one.

    It remains the single hook, so if a future cursor build does need something
    the minimum goes **here** and nowhere else. One rule binds whatever lands
    here: **never copy the user's
    ``cli-config.json`` in**. That would reintroduce the leak in reverse — the
    run would inherit (and then diverge from) the user's model, parameters and
    history, and a per-spawn scope would hand every row the same stale copy.
    Seed only literals this module owns.
    """
    return None


def _create_profile_dir() -> str:
    """A fresh 0700 profile directory, owned and tracked by this process."""
    global _PROFILE_ATEXIT_REGISTERED

    path = tempfile.mkdtemp(prefix=_PROFILE_PREFIX)
    try:
        os.chmod(path, 0o700)  # mkdtemp already does this; asserted, not assumed
        os.makedirs(os.path.join(path, "projects"), mode=0o700, exist_ok=True)
        _seed_profile_dir(path)
    except Exception:
        # A half-built profile is worse than none: unwind it and let the caller
        # fall back to the guard rather than pointing cursor at a broken dir.
        shutil.rmtree(path, ignore_errors=True)
        raise
    _PROFILE_DIRS.append(path)
    if not _PROFILE_ATEXIT_REGISTERED:
        atexit.register(cleanup_isolated_profiles)
        _PROFILE_ATEXIT_REGISTERED = True
    return path


def _profile_dir_for_spawn() -> str:
    """The directory this spawn gets — the **only** reader of the scope switch."""
    global _RUN_PROFILE_DIR

    if PROFILE_DIR_SCOPE == SCOPE_PER_SPAWN:
        return _create_profile_dir()
    # per-run: created lazily at the first cursor spawn and shared by every row.
    if _RUN_PROFILE_DIR is None:
        _RUN_PROFILE_DIR = _create_profile_dir()
    return _RUN_PROFILE_DIR


def isolate_profile_env(anchor: Optional[str] = None) -> Optional[Dict[str, str]]:
    """The env delta that gives this spawn its own cursor profile, or ``None``.

    ``None`` means "not redirected, use the fallback guard" and has exactly three
    causes, all deliberate:

    * ``isolate_profile: off`` — the user chose today's behaviour;
    * ``isolate_profile: auto`` on a platform where cursor's auth rule has not
      been verified against the redirect (see
      :data:`_AUTH_UNAFFECTED_PLATFORMS`);
    * the user pinned ``CURSOR_CONFIG_DIR``/``CURSOR_DATA_DIR`` themselves, or
      the directory could not be created.

    Never returns a third key. Redirecting ``XDG_CONFIG_HOME`` would move
    ``auth.json`` on Linux and present as a logged-out CLI, which is the one
    outcome this must not produce.
    """
    setting = isolate_profile_setting(anchor)
    if setting == ISOLATE_OFF:
        return None
    if setting == ISOLATE_AUTO and not sys.platform.startswith(_AUTH_UNAFFECTED_PLATFORMS):
        return None
    if user_pinned_profile_dirs():
        return None
    with _PROFILE_LOCK:
        try:
            path = _profile_dir_for_spawn()
        except Exception:  # noqa: BLE001 - containment must never fail a run
            # No temp dir, no redirect — and the guard is what the caller arms
            # instead. Refusing the slot over a cleanup mechanism would be a
            # worse trade than running with the mitigation that already shipped.
            return None
    return {"CURSOR_CONFIG_DIR": path, "CURSOR_DATA_DIR": path}


def isolated_profile_dirs() -> Tuple[str, ...]:
    """Every profile directory this process created and has not yet removed."""
    with _PROFILE_LOCK:
        return tuple(_PROFILE_DIRS)


#: Bounded, **unslept** re-walks of one profile tree. ``rmtree`` lists a
#: directory once and then unlinks what it listed, so a file the live
#: ``worker-server`` creates *after* that listing survives the pass and keeps its
#: parent non-empty; a second walk sees it and removes it. Three passes because
#: this must never become a wait: a worker writing continuously would otherwise
#: spin here, and blocking is the one outcome worse than the residue.
_REMOVE_PASSES = 3


def _remove_profile_tree(path: str) -> bool:
    """Best-effort removal of one profile tree. **Never raises, never waits.**

    Returns whether ``path`` is gone. The contract, in the order it matters:

    1. **Never raise and never block.** This runs from ``atexit`` and from a
       ``finally``; a cleanup that raised would turn a completed fan-out into a
       failed one, and a cleanup that waited on a daemon would delay every run
       for a directory nobody is reading. So: ``ignore_errors=True``, a
       ``try``/``except`` around the whole body rather than trust in
       ``rmtree``'s internals, no sleep, no retry loop with a clock.
    2. **Still remove it.** "Leak it and let ``TMPDIR`` sort it out" is not the
       cheap option here — one profile holds a 703 KB ``statsig-cache.json``
       plus a ``chats/<hash>/<session>/store.db`` *per run*. Hence the bounded
       re-walk above rather than a single pass.
    3. **Never kill anything.** cursor-agent leaks a detached ``worker-server``
       that reparents to init and inherits this directory through the redirect
       env. Reaping it is ``plans/timeout-orphans-and-reaggregate-determinism``'s
       job, not this module's; this code only has to stay correct next to it.

    **The residual, stated plainly:** a worker that outlives *this process*
    loses its config dir out from under it. That is the safe direction to fail.
    The CLI has already exited with the result the run consumed, so nothing a
    caller is waiting on depends on that directory; it holds only a telemetry
    cache and this run's own chat store; and — decisively — cursor's credentials
    are **not** in it (``auth.json`` follows neither redirected variable), so a
    worker losing it cannot log the user out or damage anything the user owns.
    The opposite direction — keeping the tree alive for an unbounded daemon —
    leaks per-run megabytes for a process nobody tracks.
    """
    try:
        for _ in range(_REMOVE_PASSES):
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                return True
        return False
    except Exception:  # noqa: BLE001 - see rule 1: cleanup may never raise
        return False


def remove_profile_dir(path: str) -> bool:
    """Remove ``path`` **iff this process created it**. Never raises.

    The containment rule §2.3 states outright: a directory this process did not
    create is not this process's to delete, whatever it is named and wherever it
    sits. The return value answers *ownership* — removal itself is the
    best-effort contract in :func:`_remove_profile_tree`, so ``True`` means "this
    was ours and it was removed as far as removal can be guaranteed".
    """
    with _PROFILE_LOCK:
        if path not in _PROFILE_DIRS:
            return False
        _PROFILE_DIRS.remove(path)
        global _RUN_PROFILE_DIR
        if _RUN_PROFILE_DIR == path:
            _RUN_PROFILE_DIR = None
    _remove_profile_tree(path)
    return True


def cleanup_isolated_profiles() -> List[str]:
    """Remove every profile directory this process created. Idempotent; never raises.

    Registered with ``atexit`` the moment the first directory is created — the
    process-level ``finally``, covering a normal exit and an unhandled exception
    — and called directly by :func:`isolated_profile_session`'s ``finally`` for
    callers that own a bounded scope. Under the default per-run scope that
    ``atexit`` pass is already after *every* row, which is why the leaked
    ``worker-server`` (see :data:`PROFILE_DIR_SCOPE`) is only ever racing the
    **last** row's daemon rather than each row's in turn.

    Returns the directories that are actually gone. A tree a live holder kept
    non-empty across :data:`_REMOVE_PASSES` walks is *not* reported as removed
    and is put back on the tracking list, so a bounded caller's ``finally`` that
    could not win the race hands it to the ``atexit`` pass instead of silently
    dropping it. Nothing here waits for, signals or kills that holder. A leak
    that survives both costs one directory under ``TMPDIR``, sweepable by the
    ``multi-llm-cursor-`` prefix.
    """
    with _PROFILE_LOCK:
        paths = list(_PROFILE_DIRS)
        _PROFILE_DIRS.clear()
        global _RUN_PROFILE_DIR
        _RUN_PROFILE_DIR = None
    removed: List[str] = []
    residue: List[str] = []
    for path in paths:
        (removed if _remove_profile_tree(path) else residue).append(path)
    if residue:
        with _PROFILE_LOCK:
            _PROFILE_DIRS.extend(p for p in residue if p not in _PROFILE_DIRS)
    return removed


@contextlib.contextmanager
def isolated_profile_session() -> Iterator[None]:
    """Scope the redirected profiles to a block, with the ``finally`` §2.3 asks for.

    ``atexit`` alone would hold the directories for the life of the interpreter;
    this is the explicit boundary a caller (or a test) uses when the run is
    shorter than the process.
    """
    try:
        yield
    finally:
        cleanup_isolated_profiles()


def reset_run_profile_dir() -> None:
    """Forget the per-run directory without removing it (tests only).

    A faithful stand-in for a *second process*: the new interpreter shares
    nothing, so it builds its own profile.
    """
    global _RUN_PROFILE_DIR
    with _PROFILE_LOCK:
        _RUN_PROFILE_DIR = None


def reset_profile_state() -> None:
    """Remove every created directory and drop the registration (tests only).

    The trailing ``clear()`` matters: :func:`cleanup_isolated_profiles`
    deliberately *re-tracks* a tree a live holder kept, and one test's simulated
    holder must not become the next test's inherited state.
    """
    global _PROFILE_ATEXIT_REGISTERED
    cleanup_isolated_profiles()
    with _PROFILE_LOCK:
        _PROFILE_DIRS.clear()
        _PROFILE_ATEXIT_REGISTERED = False


class CursorAgentProvider(LLMProvider):
    """Provider for cursor-agent CLI tool.

    This provider wraps the cursor-agent CLI, which returns responses in a JSON
    wrapper format: {"type":"result","result":"..."}. The provider handles
    extraction of the actual response content and JSON parsing with fallback
    strategies for code blocks and raw JSON.
    """

    # D3j: the `--output-format json` envelope really does carry a `usage`
    # block, so this adapter is not duration-only. `partial` rather than `full`
    # because the envelope has **no cost field at all** — the nine keys are
    # `type`, `subtype`, `is_error`, `duration_ms`, `duration_api_ms`, `result`,
    # `session_id`, `request_id`, `usage`, and `usage` itself is exactly
    # `{inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens}`. Nothing
    # in it prices a run, and `cost_usd` is never derived from token counts
    # (that would need a per-model price table this repo does not own), so the
    # honest declaration is `partial`.
    usage_support = USAGE_PARTIAL

    #: §7.2/§5.3 — cursor-agent exited **0 in all six auth states measured**, so
    #: for this CLI the exit code carries no verdict whatsoever and only a
    #: stream field does. Two shapes, both measured on 2026-08-16:
    #:
    #: * the ``--output-format json`` envelope's ``is_error`` flag (the nine-key
    #:   result object ``parse_output`` unwraps), and
    #: * a plain-text ``Error: Authentication required. Please run 'agent login'
    #:   first, …`` printed *instead of* the envelope when auth fails before the
    #:   session starts. That one already matches the base auth set
    #:   ("authentication required"); it is the ``is_error`` envelope that
    #:   nothing else would have caught.
    #:
    #: Reading the auth *state* off ``status --format json`` is M13's job; this
    #: is only about a dispatch that produced no usable result.
    exit_zero_failure_markers = (
        re.compile(r'"is_error"\s*:\s*true'),
    )

    @property
    def name(self) -> str:
        """Return the provider identifier."""
        return "cursor-agent"

    @property
    def default_timeout(self) -> int:
        """Return the default timeout in seconds."""
        return 600

    def is_available(self) -> bool:
        """Check if cursor-agent CLI is available in PATH."""
        return shutil.which("cursor-agent") is not None

    def resolve_env(
        self, model: str, profile: str = PROFILE_GENERATE, **options: Any
    ) -> Dict[str, str]:
        """The shared env delta, plus this run's own cursor profile (Q11).

        ``resolve_env`` is the composition point *both* spawn sites call
        (``llm_client`` for the Lane A fan-out and ``subagent_executor._spawn``
        for a dispatch), and it runs immediately before the child starts — which
        makes it the only hook an adapter owns that is guaranteed to see every
        real spawn and no argv-only caller. :meth:`build_command` deliberately
        does **not** redirect or arm: it is called by ``detach._spec_binary``
        purely to read a binary name, and by every argv fixture.

        **Exactly one of the two mitigations is active per spawn.** The redirect
        is primary: with ``CURSOR_CONFIG_DIR``/``CURSOR_DATA_DIR`` pointing at
        this run's own directory, cursor never opens the user's profile, so
        there is nothing to snapshot and the cross-process corruption the guard
        cannot see is unreachable. Only when the redirect declines — the config
        switch is ``off``, ``auto`` on an unverified platform, the user pinned
        their own profile, or the directory could not be created — is the
        snapshot/restore guard armed as the fallback.

        The keyword-only options the base declares are forwarded untouched
        rather than re-listed: this override adds a side effect and decides
        nothing about the env itself, so naming them here would only create a
        second place to keep in sync with ``LLMProvider.resolve_env``.
        """
        env = super().resolve_env(model, profile, **options)
        redirect = isolate_profile_env(anchor=options.get("anchor"))
        if redirect is None:
            arm_side_effect_guard()
            return env
        env.update(redirect)
        return env

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        """Build the cursor-agent command with JSON output format.

        The argv is ``--print --trust --mode ask --output-format json --model
        <model> <prompt>`` (M1/§3.1). ``-f`` — ``--force``, alias ``--yolo``,
        "Run Everything" — is gone: it was a write bypass on a read-only fan-out,
        and it measurably let the CLI create files. See
        :data:`CURSOR_REQUIRED_FLAGS` for why the two replacements are one
        change and not two.

        ``--mode ask`` applies to **every** cursor Lane A phase, not just the
        text-only ones. Q12 was answered at tool level on 2026-08-16: under this
        exact argv cursor's own agent transcript carries ``tool_use`` entries for
        ``Grep`` and ``Read``, and the run found a file by an unguessable token
        it was never given. Ask mode keeps its read tools, so ``review-code`` and
        ``review-tasks`` are not degraded by it and no ``--mode plan`` fallback is
        needed.

        Args:
            prompt: The prompt text to send to the LLM.
            model: The model identifier to use.

        Returns:
            Command arguments for cursor-agent CLI invocation.

        Raises:
            CursorFlagsUnconfirmed: when cursor-agent is installed but its help
                does not list both gated tokens — the slot is refused as
                ``version-too-low`` rather than degraded.
        """
        # The gate (§5.4.1). Its precondition is that there is a binary to gate:
        # a *missing* (or unexecutable) cursor-agent is not the gate's refusal to
        # make. `is_available()` already reports that, `llm_client` checks it
        # before building anything, `detach._spec_binary` asks this method for
        # the binary name precisely so it can say "cursor-agent is not
        # installed", and nothing that cannot be spawned can reject a flag.
        # Refusing here would replace a precise message with silence.
        binary = _gateable_binary()
        if binary is not None:
            confirmed = _help_flag_tokens(binary)
            missing = tuple(flag for flag in CURSOR_REQUIRED_FLAGS if flag not in confirmed)
            if missing:
                raise CursorFlagsUnconfirmed(missing, binary)

        return [
            "cursor-agent",
            "--print",
            "--trust",
            "--mode",
            "ask",
            "--output-format",
            "json",
            "--model",
            model,
            prompt,
        ]

    def parse_usage(self, stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
        """Recover usage from the ``--output-format json`` result envelope (D3j).

        ``parse_output`` unwraps ``result`` and **discards the envelope**, which
        is where ``usage`` lives, so this captures it before it is dropped. The
        block is camelCase (``inputTokens``/``outputTokens``/``cacheReadTokens``/
        ``cacheWriteTokens``), unlike claude-code's snake_case one, and fields
        are selected by name and omitted when absent — a zero would read as a
        free run. Only the cache **read** half is "cached input";
        ``cacheWriteTokens`` is deliberately dropped rather than folded into any
        total, because the normalized structure has no cache-write slot and
        claude-code's ``cache_creation_input_tokens`` is dropped for the same
        reason. No ``cost_usd``: the envelope carries no price (hence
        ``USAGE_PARTIAL``).
        """
        text = (stdout or "").strip()
        if not text.startswith("{"):
            return None
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            return None

        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        out: Dict[str, Any] = {}

        def _put(key: str, value: Any) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = value

        _put("input_tokens", usage.get("inputTokens"))
        _put("output_tokens", usage.get("outputTokens"))
        _put("cached_input_tokens", usage.get("cacheReadTokens"))
        if "input_tokens" in out and "output_tokens" in out:
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        return out or None

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse cursor-agent output and extract structured data.

        The cursor-agent with --output-format json returns responses wrapped in:
        {"type":"result","result":"..."}

        This method unwraps that format and attempts to parse the inner result
        as JSON, with fallback strategies for various response formats.

        Args:
            stdout: The standard output from cursor-agent.
            stderr: The standard error (unused but required by interface).

        Returns:
            A dictionary with 'success' and 'data' (or 'error') keys.
        """
        # One shared implementation (§9.5): claude-code emits the same envelope
        # and used to carry a byte-identical copy of the unwrapping.
        return parse_result_envelope(stdout)
