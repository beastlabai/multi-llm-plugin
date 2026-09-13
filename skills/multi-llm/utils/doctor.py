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

This is the one mode allowed to spawn provider CLIs outside a phase, and only
their model-free probes. It takes no lock, writes nothing, and does not read
or touch any phase directory.
"""

from __future__ import annotations

import argparse
import json
import os
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
    "the runner's own sandbox cannot write under this repo; on Ubuntu 24.04 this "
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


def _probe_sandbox(provider: Any, timeout: float, cwd: Optional[str]) -> Dict[str, Any]:
    """The adapter's own runner first; the bwrap self-test for a bubblewrap
    user without one; ``none`` for everything else."""
    from .providers.base import PROFILE_AGENT_WRITE
    from .subagent_executor import sandbox_verdict

    mechanism = (getattr(provider, "sandbox_mechanism", "") or "").strip()
    bypassed = (getattr(provider, "sandbox_disabled_by", "") or "").strip()
    row: Dict[str, Any] = {
        "state": "none", "detail": "", "ok": True, "mechanism": mechanism,
        "bypassed_by": bypassed, "probe": "",
    }
    state, detail = "not-required", ""
    if not bypassed:
        state, detail = sandbox_verdict(provider, PROFILE_AGENT_WRITE, cwd=cwd, timeout=timeout)
        if state != "not-required":
            row["probe"] = "runner"
    if state == "not-required":
        if "bubblewrap" not in mechanism.lower():
            return row
        state, detail = _bwrap_self_test(timeout)
        row["probe"] = "bwrap"
    row["state"] = state
    row["detail"] = detail
    # A bypassed sandbox is reported, never failed: the adapter no longer puts
    # it in the dispatch path (muse, `--disable-sandbox`).
    row["ok"] = state != "unavailable" or bool(bypassed)
    return row


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
    remaining = max(budget_s - (time.monotonic() - started), 0.05)
    cwd = _probe_cwd(anchor)
    by_name = {row["provider"]: row for row in rows}

    def probe(item: Tuple[str, Any]) -> Tuple[str, Dict[str, Any]]:
        name, provider = item
        return name, {
            "version": _probe_version(provider, name, remaining),
            "auth": _probe_auth(provider, remaining),
            "sandbox": _probe_sandbox(provider, remaining, cwd),
        }

    if present:
        with ThreadPoolExecutor(max_workers=min(3 * len(present), 24)) as pool:
            for name, result in pool.map(probe, present):
                by_name[name].update(result)

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
