#!/usr/bin/env python3
"""``multi_llm.py doctor [--json] [--warn-only]`` — machine health, per provider (D6 step 5).

One line per provider in the resolved config: **present** (``is_available``),
**version** against the adapter's ``min_version``, **auth** through the
adapter's own ``auth_check``, and **sandbox** — the adapter's model-free
``sandbox_probe_command`` where it declares one (codex, through its own
``codex sandbox`` runner) or, for any adapter that names bubblewrap, the 2 ms
``bwrap --ro-bind / / --dev /dev --unshare-net -- /bin/true`` self-test. No
model is ever called: every probe here is a status subcommand the executor's
preflight already runs, and the whole report is budgeted to about a second
(``--budget``) — a probe that outlives its share is reported ``unprobed``
rather than allowed to blow the budget.

Exit **0** when nothing fails, **3** when a present provider is below its
floor, logged out, or its live sandbox cannot start; ``--warn-only`` always
exits 0 (the Claude Code plugin's ``SessionStart`` hook, ``hooks/hooks.json``,
runs it that way — advisory, never blocking). When a sandbox row fails the
repair is printed verbatim: the AppArmor profile that grants ``/usr/bin/bwrap``
the ``userns`` right, with the ``sysctl`` alternative named and not recommended.
The plugin prints the profile; it never installs it.

A BROKEN runner row also carries the fallback ladder (landlock rung,
2026-09-22): a second probe, of the read-only ``generate`` profile, walks the
adapter's rungs inside what is left of the sandbox job's budget, and the row
names the ``sandbox_fallback`` key in force and the rung *that* probe selected
— ``→ landlock`` (with its ``retires_when``), the unsandboxed sentence, a
refusal, or ``rung unprobed`` when the budget ran out first. Rendered only:
``agent-write`` still drives the verdict and the exit code. Separately, a
``[features] use_legacy_landlock = true`` in codex's own ``config.toml`` is a
WARN (a plain file read; never ``codex features``).

This is the one mode allowed to spawn provider CLIs outside a phase, and only
their model-free probes. It takes no lock, writes nothing, and does not read
or touch any phase directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cli_common import EXIT_NEEDS_HUMAN, EXIT_OK, build_parent_parser, dispatcher_prefix
from .stream_bootstrap import bootstrap_streams

#: The whole report's wall-clock budget, in seconds. Sub-second is the
#: contract (the hook runs at every session start); ``--budget`` widens it for
#: a deliberate slow check.
DEFAULT_BUDGET_S = 1.0

#: The 2 ms bubblewrap self-test the plan measured on a stock Ubuntu 24.04:
#: ``bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted``, exit 1.
BWRAP_SELF_TEST: Tuple[str, ...] = (
    "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-net", "--", "/bin/true",
)

#: The repair, verbatim. One more profile of the shape Ubuntu already ships
#: for chrome/Discord/1password: ``flags=(unconfined)`` plus ``userns,`` and
#: nothing else, so it grants bwrap exactly the right the kernel setting took
#: away and loosens nothing else on the machine.
APPARMOR_PROFILE = """\
# /etc/apparmor.d/bwrap
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,

  # Site-specific additions and overrides. See local/README for details.
  include if exists <local/bwrap>
}
"""

APPARMOR_LOAD_COMMAND = "sudo apparmor_parser -r /etc/apparmor.d/bwrap"

SYSCTL_ALTERNATIVE = "sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"

SANDBOX_FIX_SENTENCE = (
    "the runner's own sandbox cannot start under this repo; on Ubuntu 24.04 this "
    "is the AppArmor user-namespace restriction, see `multi_llm.py doctor`"
)


def _repair_block() -> Dict[str, Any]:
    return {
        "apparmor_profile_path": "/etc/apparmor.d/bwrap",
        "apparmor_profile": APPARMOR_PROFILE,
        "load_command": APPARMOR_LOAD_COMMAND,
        "sysctl_alternative": SYSCTL_ALTERNATIVE,
        "sysctl_alternative_recommended": False,
        "sysctl_alternative_note": (
            "loosens user-namespace creation for every binary on the machine, "
            "where the profile grants it to /usr/bin/bwrap alone"
        ),
    }


def _render_repair() -> List[str]:
    lines = [
        "",
        "Repair (yours to install, with sudo — the plugin only prints it):",
        "  write this profile to /etc/apparmor.d/bwrap:",
        "",
    ]
    lines += [f"    {line}" if line else "" for line in APPARMOR_PROFILE.rstrip("\n").splitlines()]
    lines += [
        "",
        f"  then load it: {APPARMOR_LOAD_COMMAND}",
        f"  The alternative, `{SYSCTL_ALTERNATIVE}`, loosens every binary on the "
        "machine and is not recommended.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Probes. Each returns a small dict; none calls a model.
# ---------------------------------------------------------------------------

def _run(argv: Sequence[str], *, timeout: float, cwd: Optional[str] = None) -> Optional[Tuple[int, str, str]]:
    """``(returncode, stdout, stderr)`` or ``None`` when the probe could not run
    or did not finish inside ``timeout``."""
    if timeout <= 0:
        return None
    from .llm_client import _resolve_executable

    launcher, _ = _resolve_executable(argv[0])
    try:
        completed = subprocess.run(
            [*launcher, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(timeout, 0.01),
            stdin=subprocess.DEVNULL,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _bwrap_self_test(timeout: float) -> Tuple[str, str]:
    """``(state, detail)``: ``ok`` | ``unavailable`` | ``absent`` | ``unknown``."""
    if shutil.which(BWRAP_SELF_TEST[0]) is None:
        return "absent", "bwrap is not on PATH"
    result = _run(BWRAP_SELF_TEST, timeout=timeout)
    if result is None:
        return "unknown", "the bwrap self-test could not be read in time"
    code, out, err = result
    tail = (err or out).strip().splitlines()
    detail = tail[-1] if tail else ""
    return ("ok" if code == 0 else "unavailable"), detail


def _probe_version(provider: Any, name: str, timeout: float) -> Dict[str, Any]:
    from .providers.base import version_verdict

    argv = list(getattr(provider, "version_cmd", ()) or ())
    row: Dict[str, Any] = {
        "version": "unknown", "state": "unprobed", "detail": "", "ok": True,
        "min_version": (getattr(provider, "min_version", "") or "").strip(),
    }
    if not argv:
        row["detail"] = "no version_cmd declared"
        return row
    result = _run(argv, timeout=timeout)
    if result is None:
        row["detail"] = "slow: the probe did not answer within the budget"
        return row
    _, out, err = result
    try:
        parsed = provider.parse_version(out, err)
    except Exception:  # noqa: BLE001 - a diagnostic must not crash the report
        parsed = None
    if not parsed:
        row["state"] = "unread"
        row["detail"] = f"`{' '.join(argv)}` printed nothing this adapter reads as a version"
        return row
    row["version"] = parsed
    verdict = version_verdict(provider, parsed, name=name)
    row["state"] = verdict.state
    row["detail"] = verdict.detail
    row["ok"] = not verdict.refuses
    return row


def _probe_auth(provider: Any, timeout: float) -> Dict[str, Any]:
    argv = list(getattr(provider, "auth_check", ()) or ())
    row: Dict[str, Any] = {"state": "unprobed", "detail": "", "ok": True}
    if not argv:
        row["detail"] = "no auth_check declared"
        return row
    result = _run(argv, timeout=timeout)
    if result is None:
        row["detail"] = "slow: the probe did not answer within the budget"
        return row
    code, out, err = result
    try:
        verdict = provider.classify_auth_probe(out, err, code)
    except Exception:  # noqa: BLE001
        verdict = None
    if verdict is True:
        row["state"] = "ok"
    elif verdict is False:
        login = getattr(provider, "login_cmd", "") or f"{argv[0]} login"
        row.update(state="not-logged-in", ok=False, detail=f"run `{login}`")
    else:
        row["state"] = "unknown"
        row["detail"] = f"`{' '.join(argv)}` exited {code} without a recognised verdict"
    return row


def _sandbox_fallback_for(
    provider: Any, cwd: Optional[str], verdict: Any = None,
) -> Tuple[str, Optional[Any]]:
    """``(policy, rung)``: the adapter's resolved ``sandbox_fallback`` key for
    the read-only profiles and the fallback rung that key lets the next run
    take, given ``verdict`` — the read-only profile's own probe (D6,
    2026-09-17; landlock rung, 2026-09-22).

    ``policy`` is read the way the fan-out reads it — through the adapter's own
    hook, for ``generate``, anchored on the repo root — so ``doctor`` reports
    the key the next run will actually apply; ``""`` for an adapter that
    declares none. ``rung`` is the :class:`SandboxRung` the probe's walk
    selected **and** the key permits (``auto`` any, ``contained`` a contained
    one, ``off`` none — the gates' own ``sandbox_fallback_rung``), or ``None``.
    The row never names a rung the probe did not return: without a verdict
    there is no rung, whatever the key says.
    """
    from .providers.base import PROFILE_GENERATE
    from .subagent_executor import sandbox_fallback_rung

    hook = getattr(provider, "sandbox_fallback_policy", None)
    if not callable(hook):
        return "", None
    try:
        policy = hook(PROFILE_GENERATE, anchor=cwd)
    except Exception:  # noqa: BLE001 - a config read never breaks the report
        return "", None
    if not isinstance(policy, str):
        return "", None
    if verdict is None or getattr(verdict, "state", "") != "unavailable":
        return policy, None
    try:
        rung = sandbox_fallback_rung(provider, PROFILE_GENERATE, verdict.rung, policy)
    except Exception:  # noqa: BLE001
        rung = None
    return policy, rung


def _declared_rung(provider: Any, name: str) -> Optional[Any]:
    """The adapter's declared read-only rung called ``name``, or ``None``."""
    from .providers.base import PROFILE_GENERATE

    hook = getattr(provider, "sandbox_fallback_rungs", None)
    if not callable(hook):
        return None
    try:
        rungs = tuple(hook(PROFILE_GENERATE) or ())
    except Exception:  # noqa: BLE001
        return None
    return next((rung for rung in rungs if getattr(rung, "name", None) == name), None)


def _probe_fallback(
    provider: Any, row: Dict[str, Any], deadline: float, cwd: Optional[str],
) -> None:
    """The second probe: the read-only profile the fallback applies to (T6).

    ``agent-write`` reaches no rung, so the row's own probe can never report
    one; this probe of ``generate`` walks the adapter's ladder and the row
    renders the rung *it* selected. It gets only what is left of the sandbox
    job's ``deadline`` — the walk inside it shares that same remainder — so a
    slow ``agent-write`` probe truncates this one rather than lengthening the
    report, and a probe nobody had time to run is ``unprobed``, never
    ``unavailable``. Rendered only: it changes neither ``state``, ``ok`` nor
    the exit code.
    """
    from .providers.base import PROFILE_GENERATE
    from .subagent_executor import (
        RUNG_ABSENT, RUNG_REJECTED, RUNG_UNAVAILABLE, sandbox_verdict,
    )

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        row["fallback_probe"] = "unprobed"
        return
    verdict = sandbox_verdict(provider, PROFILE_GENERATE, cwd=cwd, timeout=remaining)
    state = verdict.state
    if state == "unknown" and time.monotonic() >= deadline:
        state = "unprobed"
    row["fallback_probe"] = state
    row["fallback_walk"] = [list(step) for step in verdict.walk]
    _, rung = _sandbox_fallback_for(provider, cwd, verdict)
    if rung is not None:
        row["fallback_rung"] = rung.name
        row["fallback_contained"] = bool(rung.contained)
        if rung.retires_when:
            row["fallback_retires_when"] = list(rung.retires_when)
    # A contained rung that expires upstream, measured here and found unusable:
    # the doctor-side half of Open question 2. Named with its `retires_when`,
    # because "the boundary you had is gone" is the one day the expiry matters.
    for name, outcome in verdict.walk:
        if outcome not in (RUNG_REJECTED, RUNG_UNAVAILABLE, RUNG_ABSENT):
            continue
        declared = _declared_rung(provider, name)
        if declared is not None and declared.contained and declared.retires_when:
            row["fallback_lost"] = {
                "rung": name, "outcome": outcome,
                "retires_when": list(declared.retires_when),
            }
            break


def _probe_sandbox(provider: Any, timeout: float, cwd: Optional[str]) -> Dict[str, Any]:
    """The adapter's own runner first; the bwrap self-test for a bubblewrap
    user without one; ``none`` for everything else.

    The row also carries ``fallback``: the adapter's resolved
    ``sandbox_fallback`` key (``""`` where none is declared), and — when the
    runner's ``agent-write`` probe says ``unavailable`` under a key that may
    take a rung — the second, read-only probe's findings (``fallback_probe``,
    ``fallback_walk``, ``fallback_rung``, …; see :func:`_probe_fallback`). They
    change neither the verdict nor the exit code — ``agent-write`` stays
    refused under a broken sandbox whatever the key says, and the repair is
    still the fix — they are rendered beside ``BROKEN`` so the operator sees
    what read-only phases will run under on this machine.

    ``timeout`` is the whole job's budget: every probe here gets only what is
    left of it, recomputed before each.
    """
    from .providers.base import PROFILE_AGENT_WRITE, SANDBOX_FALLBACK_OFF
    from .subagent_executor import sandbox_verdict

    deadline = time.monotonic() + max(timeout, 0.0)
    mechanism = (getattr(provider, "sandbox_mechanism", "") or "").strip()
    bypassed = (getattr(provider, "sandbox_disabled_by", "") or "").strip()
    policy, _ = _sandbox_fallback_for(provider, cwd)
    row: Dict[str, Any] = {
        "state": "none", "detail": "", "ok": True, "mechanism": mechanism,
        "bypassed_by": bypassed, "probe": "",
        "fallback": policy,
        "fallback_probe": "", "fallback_walk": [], "fallback_rung": None,
    }
    if timeout <= 0:
        row.update(state="unprobed", detail="slow: the report budget was exhausted")
        return row
    state, detail = "not-required", ""
    if not bypassed:
        probe = sandbox_verdict(provider, PROFILE_AGENT_WRITE, cwd=cwd, timeout=timeout)
        state, detail = probe.state, probe.tail
        if state != "not-required":
            row["probe"] = "runner"
    if state == "not-required":
        if "bubblewrap" not in mechanism.lower():
            return row
        state, detail = _bwrap_self_test(max(deadline - time.monotonic(), 0.0))
        row["probe"] = "bwrap"
    row["state"] = state
    row["detail"] = detail
    # A bypassed sandbox is reported, never failed: the adapter no longer puts
    # it in the dispatch path (muse, `--disable-sandbox`).
    row["ok"] = state != "unavailable" or bool(bypassed)
    # `off` takes no rung whatever the walk finds, so its row needs no probe.
    if (
        state == "unavailable" and row["probe"] == "runner"
        and policy and policy != SANDBOX_FALLBACK_OFF
    ):
        _probe_fallback(provider, row, deadline, cwd)
    return row


def _fallback_note(provider_name: str, sandbox: Dict[str, Any]) -> str:
    """The ``(fallback: …)`` clause of a BROKEN row, or ``""``.

    Names the key in force **and** what the read-only probe found, so the
    three key values and the three endings stay apart on screen: a rung the
    probe selected (``→ landlock …``); the uncontained rung (today's
    unsandboxed sentence); and a rung nobody measured because the report's
    budget ran out (``rung unprobed``), which must never read as either.
    """
    from .providers.base import SANDBOX_FALLBACK_CONTAINED, SANDBOX_FALLBACK_OFF

    policy = sandbox.get("fallback") or ""
    if not policy:
        return ""
    if policy == SANDBOX_FALLBACK_OFF:
        return (
            f" (fallback: off — {provider_name} is refused by config for "
            f"read-only phases too, not merely unavailable)"
        )
    probe_state = sandbox.get("fallback_probe") or ""
    walk = [tuple(step) for step in sandbox.get("fallback_walk") or ()]
    rung = sandbox.get("fallback_rung")
    lost = sandbox.get("fallback_lost")
    lost_clause = ""
    if lost and lost.get("rung") != rung:
        version, sentence = (list(lost.get("retires_when") or ()) + ["", ""])[:2]
        lost_clause = (
            f"; rung {lost['rung']} is not usable here ({lost['outcome']}) — "
            f"measured at {provider_name} {version}: {sentence}"
        )
    if probe_state == "unprobed" or any(outcome == "unprobed" for _, outcome in walk):
        return f" (fallback: {policy} — rung unprobed: the report budget was exhausted)"
    if rung and sandbox.get("fallback_contained"):
        note = (
            f" (fallback: {policy} → {rung} — {provider_name} runs read-only "
            f"phases under a real sandbox; agent-write still refused"
        )
        retires = sandbox.get("fallback_retires_when")
        if retires:
            version, sentence = (list(retires) + ["", ""])[:2]
            note += f"; measured at {provider_name} {version}: {sentence}"
        return note + ")"
    if rung:
        return (
            f" (fallback: {policy} — {provider_name} runs unsandboxed "
            f"for read-only phases{lost_clause})"
        )
    if probe_state == "ok":
        return (
            f" (fallback: {policy} — not needed: the read-only profile's own "
            f"sandbox starts)"
        )
    if probe_state in ("unknown", "not-required", ""):
        return (
            f" (fallback: {policy} — no rung selected: the read-only probe "
            f"gave no answer)"
        )
    if any(outcome == "unknown" for _, outcome in walk):
        return (
            f" (fallback: {policy} — no rung taken: a rung probe gave no "
            f"answer, so {provider_name} is refused for read-only phases too"
            f"{lost_clause})"
        )
    if policy == SANDBOX_FALLBACK_CONTAINED:
        return (
            f" (fallback: {policy} — bypass disabled: no contained rung is "
            f"usable here, so {provider_name} is refused for read-only phases "
            f"too{lost_clause})"
        )
    return (
        f" (fallback: {policy} — no rung usable here: {provider_name} is "
        f"refused for read-only phases too{lost_clause})"
    )


# ---------------------------------------------------------------------------
# Constraint 2: `use_legacy_landlock` enabled globally in codex's own config
# ---------------------------------------------------------------------------

def _codex_config_path() -> str:
    """``$CODEX_HOME/config.toml``, else ``~/.codex/config.toml``."""
    home = (os.environ.get("CODEX_HOME") or "").strip()
    if home:
        return os.path.join(os.path.expanduser(home), "config.toml")
    return os.path.join(os.path.expanduser("~"), ".codex", "config.toml")


def _toml_module() -> Any:
    """``tomllib`` (3.11+), else ``tomli`` if the user happens to have it,
    else ``None`` — the runtime dependency surface is ``pyyaml`` alone, so on
    Python 3.10 under ``uv run --no-dev`` there is usually no parser at all."""
    try:
        import tomllib as _toml  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Python 3.10
        try:
            import tomli as _toml  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None
    return _toml


def _load_toml(text: str) -> Optional[Dict[str, Any]]:
    _toml = _toml_module()
    if _toml is None:
        return None
    try:
        data = _toml.loads(text)
    except Exception:  # noqa: BLE001 - unparseable is "not detected"
        return None
    return data if isinstance(data, dict) else None


_TOML_KEY = r'(?:[A-Za-z0-9_-]+|"[^"\n]*"|\'[^\'\n]*\')'
_TOML_TABLE_RE = re.compile(r"^\[\s*(" + _TOML_KEY + r"(?:\s*\.\s*" + _TOML_KEY + r")*)\s*\]\s*(?:#.*)?$")
_TOML_ARRAY_TABLE_RE = re.compile(r"^\[\[")
_TOML_ASSIGN_RE = re.compile(
    r"^(" + _TOML_KEY + r"(?:\s*\.\s*" + _TOML_KEY + r")*)\s*=\s*(.*?)\s*(?:#.*)?$"
)
_TOML_INLINE_LANDLOCK_RE = re.compile(
    r"""^\{(?:.*,)?\s*["']?use_legacy_landlock["']?\s*=\s*true\s*[,}]"""
)


def _toml_key_parts(dotted: str) -> List[str]:
    return [
        part.strip().strip("\"'")
        for part in re.findall(_TOML_KEY, dotted)
    ]


def _scan_landlock_flag(text: str) -> bool:
    """Stdlib-only match of ``features.use_legacy_landlock = true`` for when no
    TOML parser is importable (Python 3.10 without ``tomli``).

    Not a TOML parser: it tracks the current ``[table]`` header line by line
    and recognises the three spellings a config would use — the key under
    ``[features]``, the dotted ``features.use_legacy_landlock`` at the top
    level, and an inline ``features = { use_legacy_landlock = true }``. Lines
    inside a multi-line string are not tracked, so a stray match there is
    possible; a doctor WARN is the only consequence, never a refusal.
    """
    table: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _TOML_ARRAY_TABLE_RE.match(line):
            table = ["<array>"]
            continue
        header = _TOML_TABLE_RE.match(line)
        if header:
            table = _toml_key_parts(header.group(1))
            continue
        assign = _TOML_ASSIGN_RE.match(line)
        if not assign:
            continue
        path = table + _toml_key_parts(assign.group(1))
        value = assign.group(2)
        if path == ["features", "use_legacy_landlock"] and value == "true":
            return True
        if path == ["features"] and _TOML_INLINE_LANDLOCK_RE.match(value):
            return True
    return False


def _landlock_flag_enabled(text: str) -> bool:
    if _toml_module() is None:
        return _scan_landlock_flag(text)
    data = _load_toml(text)
    if data is None:
        return False
    features = data.get("features")
    return isinstance(features, dict) and features.get("use_legacy_landlock") is True


def _global_landlock_config() -> Optional[str]:
    """The codex config file that enables ``[features] use_legacy_landlock``
    globally, or ``None`` (landlock rung, constraint 2).

    A plain file read — never ``codex features enable``/``list``, which would
    set or depend on the very thing being looked for. A missing, unreadable or
    unparseable file is "not detected", never an error.
    """
    path = _codex_config_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return None
    return path if _landlock_flag_enabled(text) else None


def _landlock_config_warning(path: str) -> str:
    return (
        f"codex: {path} enables `[features] use_legacy_landlock = true` "
        f"globally — every workspace-write codex dispatch on this machine, "
        f"agent-write included, will panic (exit 101) until that line is "
        f"removed; no per-dispatch `-c` of multi-llm's can clear it"
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def collect(
    *, budget_s: float = DEFAULT_BUDGET_S, host_id: Optional[str] = None,
    anchor: Optional[str] = None,
) -> Dict[str, Any]:
    """Every configured provider, probed concurrently inside one budget."""
    from . import host_capabilities as caps
    from .provider_registry import get_provider, load_config

    started = time.monotonic()
    try:
        host = caps.resolve_host_id(explicit=host_id)
    except Exception:  # noqa: BLE001 - an unknown host id does not stop a report
        host = host_id or "unknown"
    config = load_config(anchor=anchor)
    names = list((config.get("providers") or {}).keys())

    rows: List[Dict[str, Any]] = []
    present: List[Tuple[str, Any]] = []
    for name in names:
        provider = get_provider(name)
        row: Dict[str, Any] = {"provider": name, "present": False, "adapter": provider is not None}
        if provider is None:
            row["detail"] = "no built-in adapter (config name is ignored at runtime)"
        else:
            try:
                row["present"] = bool(provider.is_available())
            except Exception:  # noqa: BLE001
                row["present"] = False
            trust = (getattr(provider, "trust_status", "") or "").strip()
            if trust:
                row["trust"] = trust
        rows.append(row)
        if row["present"]:
            present.append((name, provider))

    # Every probe is a subprocess with its own timeout at the budget, and they
    # run concurrently, so the wall clock is the slowest probe rather than the
    # sum — what makes a seventeen-adapter roster fit in a second.
    deadline = started + max(budget_s, 0.05)
    cwd = _probe_cwd(anchor)
    by_name = {row["provider"]: row for row in rows}

    def probe(item: Tuple[str, Any, str]) -> Tuple[str, str, Dict[str, Any]]:
        name, provider, kind = item
        remaining = max(deadline - time.monotonic(), 0.0)
        if kind == "version":
            result = _probe_version(provider, name, remaining)
        elif kind == "auth":
            result = _probe_auth(provider, remaining)
        else:
            result = _probe_sandbox(provider, remaining, cwd)
        return name, kind, result

    if present:
        # Each probe gets a job, rather than serializing three full-budget
        # probes inside one provider job. Queued jobs use the same deadline.
        jobs = [(name, provider, kind) for name, provider in present
                for kind in ("version", "auth", "sandbox")]
        with ThreadPoolExecutor(max_workers=min(3 * len(present), 24)) as pool:
            for name, kind, result in pool.map(probe, jobs):
                by_name[name][kind] = result

    warnings: List[str] = []
    if "codex" in names:
        landlock_config = _global_landlock_config()
        if landlock_config:
            warnings.append(_landlock_config_warning(landlock_config))

    failures: List[str] = []
    sandbox_broken = False
    for row in rows:
        if not row["present"]:
            continue
        if not row["version"]["ok"]:
            failures.append(f"{row['provider']}: version — {row['version']['detail']}")
        if not row["auth"]["ok"]:
            failures.append(f"{row['provider']}: auth — not logged in ({row['auth']['detail']})")
        sandbox = row["sandbox"]
        if sandbox["state"] == "unavailable":
            sandbox_broken = True
            if not sandbox["ok"]:
                failures.append(f"{row['provider']}: sandbox — {SANDBOX_FIX_SENTENCE}")

    return {
        "host": host,
        "budget_s": budget_s,
        "elapsed_s": round(time.monotonic() - started, 3),
        "configured": len(names),
        "present": len(present),
        "providers": rows,
        "failures": failures,
        "warnings": warnings,
        "sandbox_broken": sandbox_broken,
        "repair": _repair_block() if sandbox_broken else None,
    }


def _probe_cwd(anchor: Optional[str]) -> Optional[str]:
    from .subagent_executor import _probe_cwd as executor_probe_cwd

    return executor_probe_cwd(anchor)


def render_text(report: Dict[str, Any], *, warn_only: bool) -> List[str]:
    lines = [
        f"multi-llm doctor — host {report['host']}, {report['configured']} provider(s) "
        f"configured, {report['present']} present, {report['elapsed_s']:.2f} s "
        f"(budget {report['budget_s']:.1f} s)"
    ]
    width = max((len(r["provider"]) for r in report["providers"]), default=8)
    for row in report["providers"]:
        name = row["provider"].ljust(width)
        if not row["present"]:
            note = row.get("detail") or "absent"
            lines.append(f"  {name}  {note}")
            continue
        version = row["version"]
        v = version["version"]
        if version["state"] == "unprobed":
            v = f"unprobed ({version['detail']})"
        elif version["min_version"]:
            v += f" (floor {version['min_version']})"
        if not version["ok"]:
            v += f" — BELOW FLOOR: {version['detail']}"
        auth = row["auth"]
        a = auth["state"]
        if a in ("unprobed", "unknown") and auth["detail"]:
            a += f" ({auth['detail']})"
        elif a == "not-logged-in":
            a += f" ({auth['detail']})"
        sandbox = row["sandbox"]
        s = sandbox["state"]
        if s == "unavailable":
            s = ("broken, bypassed by " + sandbox["bypassed_by"]) if sandbox["bypassed_by"] else "BROKEN"
            if sandbox["detail"]:
                s += f": {sandbox['detail']}"
            # D6 override (2026-09-17), landlock rung (2026-09-22): the row
            # stays BROKEN and the repair is still printed — the write profile
            # is refused regardless — but the operator is told what the key in
            # force does on this machine, from the read-only probe's own walk.
            if not sandbox["bypassed_by"]:
                s += _fallback_note(row["provider"], sandbox)
        elif s == "ok":
            s = f"ok ({sandbox['probe']})"
        elif s == "none":
            s = "none"
        elif sandbox["detail"]:
            s += f" ({sandbox['detail']})"
        line = f"  {name}  present  version {v}  auth {a}  sandbox {s}"
        if row.get("trust"):
            line += f"  trust: {row['trust']}"
        lines.append(line)
    for warning in report.get("warnings") or ():
        lines.append(f"WARN: {warning}")
    if report["failures"]:
        head = "WARN" if warn_only else "FAIL"
        for failure in report["failures"]:
            lines.append(f"{head}: {failure}")
    if report["sandbox_broken"]:
        lines += _render_repair()
    if not report["failures"]:
        lines.append("OK: every present provider passed its checks.")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="multi_llm.py doctor",
        parents=[build_parent_parser()],
        description=(
            "Check every configured provider CLI on this machine — present, "
            "version, auth, sandbox — with no model calls, in about a second."
        ),
        epilog=(
            f"Exit 0 when nothing fails, 3 otherwise; --warn-only always exits 0.\n"
            f"Example: {dispatcher_prefix()} doctor --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON object.")
    parser.add_argument(
        "--warn-only", action="store_true",
        help="Report failures but exit 0 (what the SessionStart hook runs).",
    )
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_S, metavar="SECONDS",
        help=f"Wall-clock budget for the whole report (default {DEFAULT_BUDGET_S:.1f}).",
    )
    parser.add_argument(
        "--plan-file", default=None, metavar="PATH",
        help="Optional discovery anchor for the project-local config layer.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    args = parse_args(argv)
    anchor = os.path.abspath(args.plan_file) if args.plan_file else None
    report = collect(
        budget_s=max(float(args.budget), 0.05),
        host_id=getattr(args, "host", None),
        anchor=anchor,
    )
    exit_code = EXIT_NEEDS_HUMAN if report["failures"] else EXIT_OK
    report["exit"] = EXIT_OK if args.warn_only else exit_code
    report["warn_only"] = bool(args.warn_only)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("\n".join(render_text(report, warn_only=bool(args.warn_only))))
    return report["exit"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
