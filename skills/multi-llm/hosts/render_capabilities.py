#!/usr/bin/env python3
"""Render ``references/host-capabilities.md`` from ``hosts/capabilities.yaml`` (D5b).

The capability matrix is **data with a schema, and the markdown is a build
artifact**. Keeping a prose table by hand would leave the shims,
``install-host --check`` and the tests each re-encoding the same facts, and would
put a table an agent might read back on the read path.

The generated document has a **fixed shape, not an authored one**: one row per host
with the columns the design names, followed by one short per-host section carrying
only the free-form ``notes`` strings and that host's ``env_guards`` names — the one
place a reader can see which variables get stripped from a dispatched runner.
Nothing hand-written survives a render, which is what makes the byte-compare
meaningful rather than a formatting lottery.

Usage::

    python hosts/render_capabilities.py            # write references/host-capabilities.md
    python hosts/render_capabilities.py --check     # exit 1 if the checked-in copy is stale
    python hosts/render_capabilities.py --stdout    # print, write nothing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SKILL_ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import host_capabilities as caps  # noqa: E402
from utils.stream_bootstrap import bootstrap_streams  # noqa: E402

#: Where the rendered document lands. A ``references/*.md`` file, and therefore
#: exempt from the host-neutrality lint: naming host tools is its whole job.
OUTPUT = os.path.join("references", "host-capabilities.md")

_HEADER = """<!-- GENERATED FILE — DO NOT EDIT.
     Rendered from hosts/capabilities.yaml by hosts/render_capabilities.py.
     Edit the YAML and re-render; CI fails if this file differs from a fresh
     render, so a hand-edit cannot ship. -->

# Host capabilities

Human-readable view of `hosts/capabilities.yaml`. **Nothing reads this file at run
time** — not an orchestrator, not a shim, and above all not a driving agent. The
YAML is consumed by `multi_llm.py`, by `install-host` and by the tests; this
document exists for people reading the repo.

Every value carries a **provenance**: `verified` means a recorded live smoke run
observed it, `documented` means host docs only, `unknown` is the shipping default.
A `documented`-only value never selects the non-conservative path — which is why
hosts whose native subagent dispatch is merely documented ship as
`subagents: executor`, and why a documented command-timeout ceiling is worked
around with `--detach` rather than trusted.

The three **capability verbs** are the spine of each entry. The host-neutral
instruction core says *spawn subagents* / *ask the user* / *run detached*; the
columns below say what each resolves to, and a shim bakes that resolution in as a
sentence when it is installed.
"""

_MATRIX_HEAD = [
    "| Host | Shim (repo → installed) | `arguments` | *spawn subagents* | "
    "*ask the user* | *run detached* | Default runner | `nested_exec` "
    "(spawn/network/creds) | Max subagents | Timeout ceiling | Verified |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]


def _verb_cell(host: Dict[str, Any], verb: str) -> str:
    value = str(host[verb])
    tool = (host.get("native_tool") or {}).get(verb)
    provenance = host["provenance"].get(verb, "unknown")
    cell = f"`{value}`"
    if tool:
        cell += f" (`{tool}`)"
    return f"{cell} — {provenance}"


def _shim_cell(host: Dict[str, Any]) -> str:
    shim = host["shim"]
    template = shim.get("template")
    installed = shim["installed"]
    if template is None:
        return f"*not rendered* → `{installed}`"
    return f"`{template}` → `{installed}`"


def _timeout_cell(host: Dict[str, Any]) -> str:
    ceiling = host["timeout_ceiling_s"]
    if not ceiling:
        return "none documented"
    override = host["timeout_override"]
    channels = [
        f"{name}: `{value}`"
        for name, value in override.items()
        if value != "none"
    ]
    raised = ", ".join(channels) if channels else "not raisable"
    return f"{ceiling} s (raise via {raised})"


def _verified_cell(host: Dict[str, Any]) -> str:
    verified = sorted(
        field for field, value in host["provenance"].items() if value == "verified"
    )
    return ", ".join(f"`{f}`" for f in verified) if verified else "*none yet*"


def _matrix(hosts: Dict[str, Any]) -> List[str]:
    rows = list(_MATRIX_HEAD)
    for host_id in sorted(hosts):
        host = hosts[host_id]
        rows.append(
            "| `{id}` | {shim} | `{args}` | {sub} | {ask} | {det} | `{runner}` | "
            "{nested} | {maxc} | {timeout} | {verified} |".format(
                id=host_id,
                shim=_shim_cell(host),
                args=host["arguments"],
                sub=_verb_cell(host, "subagents"),
                ask=_verb_cell(host, "ask"),
                det=_verb_cell(host, "detach"),
                runner=host["default_subagent_runner"],
                nested="/".join(
                    str(host["nested_exec"][key])
                    for key in ("spawn", "network", "credentials")
                ),
                maxc=host["max_concurrent_subagents"],
                timeout=_timeout_cell(host),
                verified=_verified_cell(host),
            )
        )
    return rows


def _sections(hosts: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["", "## Per-host notes and stripped variables", ""]
    for host_id in sorted(hosts):
        host = hosts[host_id]
        lines.append(f"### `{host_id}`")
        lines.append("")
        lines.append(
            f"- Installable: `{str(host['installable']).lower()}`; "
            f"reload after install: `{host['reload']}`; "
            f"multi-turn session: `{str(host['multi_turn']).lower()}`"
        )
        tiers = host["subagent_tiers"]
        if tiers is None:
            lines.append(
                "- Subagent tiers: resolved from `providers.yaml` "
                "`driver.subagent_runner{,_cheap}` (this host does not dispatch "
                "subagents natively)"
            )
        else:
            cheap = tiers.get("cheap", tiers["full"])
            lines.append(
                f"- Subagent tiers: full `{tiers['full']}`, cheap `{cheap}`"
            )
        lines.append("- Verb sentences a rendered shim carries:")
        for sentence in caps.verb_sentences(host_id):
            lines.append(f"  - {sentence}")
        for scope in ("project", "global", "legacy"):
            paths = host["install_paths"][scope]
            rendered = ", ".join(f"`{p}`" for p in paths) if paths else "*none*"
            lines.append(f"- Install paths ({scope}): {rendered}")
        runners = host["runners"]
        lines.append("- Runners:")
        for name in sorted(runners):
            spec = runners[name]
            lines.append(
                f"  - `{name}`: bundled=`{str(spec['bundled']).lower()}`, "
                f"needs_auth=`{str(spec['needs_auth']).lower()}`"
                + (f" (`{spec['login_cmd']}`)" if spec["login_cmd"] else "")
                + f", profiles=[{', '.join(spec['profiles'])}]"
                + f", usage=`{spec['usage']}`"
            )
        guards = host["env_guards"]
        for key, label in (
            ("guards", "Refuse-to-start guards stripped from a dispatched runner"),
            ("session_scoped", "Session-scoping variables stripped"),
            ("passthrough", "Sanctioned exceptions that survive stripping"),
        ):
            names = guards[key]
            rendered = ", ".join(f"`{n}`" for n in names) if names else (
                "*none recorded* — "
                + (
                    "audited, found nothing"
                    if host["provenance"]["env_guards"] != "unknown"
                    else "the env diff has not run yet"
                )
            )
            lines.append(f"- {label}: {rendered}")
        notes = host["nested_exec"]["notes"].strip()
        if notes:
            lines.append(f"- Nested-exec notes: {' '.join(notes.split())}")
        lines.append("")
    return lines


def render(base: Optional[str] = None) -> str:
    """The whole document, deterministically ordered."""
    hosts = caps.load(base).get("hosts", {})
    body = [_HEADER, ""]
    body += _matrix(hosts)
    body += _sections(hosts)
    return "\n".join(body).rstrip("\n") + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit 1 if the checked-in document differs from a fresh render "
             "(the CI staleness guard).",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print the render; write nothing."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    text = render()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    target = SKILL_ROOT / OUTPUT
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current == text:
            print(f"{OUTPUT} is up to date.")
            return 0
        print(
            f"ERROR: {OUTPUT} is stale. It is generated from "
            "hosts/capabilities.yaml — re-run `python "
            "hosts/render_capabilities.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    target.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
