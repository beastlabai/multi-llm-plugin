"""Base protocol and utilities for LLM CLI providers."""

import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Matches CSI / SGR ANSI escape sequences (colour, cursor moves). Listing CLIs are
# invoked with NO_COLOR=1/TERM=dumb to suppress these at the source; this is the
# belt-and-suspenders strip for any CLI that colourises regardless.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


@dataclass
class ModelListing:
    """Result of asking a provider which models it can run.

    Attributes:
        models: Full catalogue as BARE model ids (the exact string the adapter's
            ``build_command`` passes to ``--model``/``-m``; may contain ``/`` but
            never ``:``). ``[]`` for non-listing providers. Curated ids are floated
            to the top so a "Show all…" pass shows them first.
        source: Where ``models``/``recommended`` came from — ``"cli"`` when a
            listing command succeeded, ``"curated"`` when it fell back to the curated
            input (failure/empty/garbled), ``"none"`` for a provider with no command.
        recommended: Short subset shown first in the picker (the curated ids).
        note: Optional one-line human notice (e.g. a fallback reason) for the picker.
    """

    models: List[str]
    source: str = "curated"          # "cli" | "curated" | "none"
    recommended: List[str] = field(default_factory=list)
    note: Optional[str] = None


def is_valid_bare_id(s: Any) -> bool:
    """True if ``s`` is usable as a bare model id (non-empty, no space, no ``:``).

    A bare id may contain ``/`` (namespaced opencode/kilocode ids) but never a
    ``:`` — that would make the ``provider:model`` round-trip through
    ``parse_model_spec`` (which splits on the first colon) ambiguous.
    """
    return isinstance(s, str) and bool(s) and not any(c.isspace() for c in s) and ":" not in s


def try_parse_json_ids(text: str) -> Optional[List[str]]:
    """If ``text`` is a JSON list of model ids, return them; else None.

    Tolerates two JSON shapes: a list of strings, or a list of objects each
    carrying an ``id`` (preferred) or ``name`` string. Returns None when the text
    is not JSON or not a recognised shape, so the caller can fall back to
    line-based parsing. Never raises.
    """
    import json

    text = text.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        # Some CLIs wrap the list, e.g. {"models": [...]} / {"data": [...]}.
        for key in ("models", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return None
    if not isinstance(data, list):
        return None
    ids: List[str] = []
    for entry in data:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            val = entry.get("id") or entry.get("name")
            if isinstance(val, str):
                ids.append(val)
    return ids


def parse_line_ids(raw: str) -> List[str]:
    """Parse a one-id-per-line listing (opencode / kilocode) → bare model ids.

    Strips ANSI, tries a JSON shape first, else treats each non-blank line as a
    candidate id; keeps only valid bare ids (drops whitespace/``:``-bearing lines,
    e.g. kilocode's ``…:free`` / ``…:discounted`` variants), de-duplicated.
    """
    raw = strip_ansi(raw)
    json_ids = try_parse_json_ids(raw)
    if json_ids is not None:
        candidates = json_ids
    else:
        candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    seen = set()
    out: List[str] = []
    for c in candidates:
        if is_valid_bare_id(c) and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def float_curated(full: List[str], curated: List[str]) -> List[str]:
    """Return ``full`` with the valid ``curated`` bare ids floated to the top.

    Curated ids are emitted first (in their given order), de-duplicated, then the
    remaining catalogue entries. Curated ids absent from the live catalogue are
    still included up front so the user can always re-select a configured pick.
    Any curated id containing ``:`` (an invalid bare id, see the id-format
    contract) is dropped.
    """
    seen = set()
    ordered: List[str] = []
    for c in curated:
        if isinstance(c, str) and c and ":" not in c and c not in seen:
            ordered.append(c)
            seen.add(c)
    for m in full:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def run_models_command(argv: List[str], *, timeout: int) -> Optional[str]:
    """Run a provider's ``models`` subcommand defensively; return stdout or None.

    Hardening so a picker can never hang or be polluted by a listing CLI:
      * ``stdin=DEVNULL`` — an auth-prompting CLI can't block on input.
      * neutralised env (``TERM=dumb``, ``NO_COLOR=1``, ``PAGER=cat``) — no pager
        wait, no colour escapes.
      * stdout/stderr captured separately; only stdout is returned.
      * hard ``timeout``; any non-zero exit, timeout, or OS error → None.

    Returns the raw stdout string on a clean (returncode 0) run, else None. Never
    raises.
    """
    env = dict(os.environ)
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env["PAGER"] = "cat"
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def build_models_listing(argv, parser, curated, *, timeout):
    """Run a listing command and wrap the result in a ``ModelListing``.

    Shared orchestration for listing-capable adapters: runs ``argv`` defensively
    (``run_models_command``), applies the adapter's ``parser`` (raw stdout → list
    of bare ids), and on any failure/empty/garbled result falls back to the
    ``curated`` input with a one-line ``note``. Never raises.

    ``parser`` is the only adapter-specific piece (each adapter owns its own output
    parsing); everything else — failure tolerance, curated fallback, floating
    curated to the top — is uniform.
    """
    cmd_str = " ".join(argv)
    raw = run_models_command(argv, timeout=timeout)
    if raw is None:
        return ModelListing(
            models=[], source="curated", recommended=list(curated),
            note=f"`{cmd_str}` failed, timed out, or needs auth; showing curated models",
        )
    try:
        ids = parser(raw)
    except Exception:
        # A misbehaving adapter parser (garbled output, future bug) must not abort
        # the picker / --init; degrade to the same curated fallback as empty output.
        ids = None
    if not ids:
        return ModelListing(
            models=[], source="curated", recommended=list(curated),
            note=f"`{cmd_str}` returned no parseable models; showing curated models",
        )
    return ModelListing(
        models=float_curated(ids, curated),
        source="cli",
        recommended=list(curated),
    )


class LLMProvider(ABC):
    """Abstract base class for LLM CLI providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods and properties. This ensures a consistent
    interface for invoking different LLM CLI tools.
    """

    # Capability flag — overridden ``True`` by adapters that implement a CLI
    # ``models`` listing. Drives whether the init picker offers a "Show all…"
    # sentinel for this provider (read STATICALLY, before any list_models() call).
    can_list_models: bool = False

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

    def list_models(self, curated: List[str], *, timeout: int = 10) -> ModelListing:
        """Return the models this provider can run, given its curated bare ids.

        ``curated`` is the provider's base-config ``models:`` list as BARE model
        ids (no ``provider:`` prefix). The init flow always supplies it, sourced
        from the base config, so even a listing-capable adapter can float its
        curated picks to the top of ``recommended``/``models``.

        The default (this method) is for **non-listing** providers: it returns the
        curated input as ``recommended`` with an empty ``models`` and
        ``source="curated"``. Adapters that shell out override this, MUST be
        timeout-bounded, and MUST fall back to ``curated`` on any failure — never
        raise.
        """
        return ModelListing(
            models=[],
            source="curated",
            recommended=list(curated),
        )
