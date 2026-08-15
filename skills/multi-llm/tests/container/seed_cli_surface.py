#!/usr/bin/env python3
"""Seed and verify `cli-surface.yaml` — the ONE upstream-facing drift signal (C8).

Run inside the container, where the pinned CLIs actually live.

**Why this exists, and why the argv golden does not replace it.** Both sides of
the argv comparison originate in this repository, so it cannot move when upstream
does: if a CLI renames or removes a flag, the wrapper keeps logging the argv *we*
generate, byte-identical to the expected table, and the assertion stays green.
Read the argv table as coverage of our own ``build_command()``.

This probe is the other half. Per CLI it records the version string plus the exact
set of options and subcommands our adapter depends on, keyed to **where they
live** — ``codex exec --help``, ``goose run --help``, ``opencode run --help``, not
just the top-level help, because several of these CLIs list only subcommands at
the top level and would make a contract of the wrong text.

Two assertions, deliberately of different strengths:

* **presence is hard** — every option in the contract must still appear in that
  version's help. A missing one fails the row naming the CLI, the pinned and
  installed versions, and the option that disappeared: a pre-written bug report.
* **the digest is soft** — help text that changed while every option we need
  survived is *reported*, not failed. That is the early warning a deprecation
  notice is. `codex`'s ``--full-auto`` → ``--sandbox workspace-write`` was exactly
  this class: the argv golden stays green straight through it in both phases,
  while the probe sees the note appear and then the removal.

A CLI whose help is unparseable or absent records ``probe: none`` with its reason
and is exempted explicitly, so the gap is data rather than a silent pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REAL_DIR = Path(os.environ.get("MLLM_REAL_DIR", "/opt/mllm/real"))
SKILL_DIR = Path(os.environ.get("MLLM_SKILL_DIR", "/opt/mllm/skill"))
RIG_DIR = Path(os.environ.get("MLLM_RIG_DIR", "/opt/mllm/rig"))


def option_is_documented(option: str, help_text: str) -> bool:
    """Is ``option`` present in ``help_text`` as a whole token?

    Deliberately **not** ``option in help_text``. A one-character flag is a
    substring of almost any help text — ``-m`` occurs inside ``--model`` and
    inside every hyphenated word — so a substring probe is true whatever
    upstream does, and the "presence is hard" assertion it feeds could never
    fail for a short option: the vacuous half of the contract. Bounding both
    ends on "not another option character" keeps ``-m`` from matching inside
    ``--model`` or ``auto-merge`` and keeps ``--model`` from being satisfied by
    ``--models``, while still accepting every spelling help text really uses:
    ``-m, --model``, ``-p/--print``, ``--model=VALUE``, ``[--model <MODEL>]``.
    """
    return re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", help_text) is not None


def missing_options(options, help_text: str) -> list[str]:
    """The contract options this help text no longer documents.

    Imported by ``tests/test_container_cli_surface.py`` rather than reimplemented
    there: the gate and the seeder asserting presence by two different rules is
    how a re-seeded contract starts disagreeing with the test that reads it.
    """
    return [opt for opt in options if not option_is_documented(opt, help_text)]


def derivation_prompt() -> str:
    """The prompt every ``build_command()`` is derived against.

    **Not** a bare word, because one adapter's argv is prompt-conditional:
    ``AiderProvider.build_command`` scans the prompt for *existing absolute
    file paths* and appends a ``--read <path>`` pair per hit, which is how the
    real orchestrator prompts (they name the plan file) get their context in.
    Derived against ``"PROMPT"`` that scan finds nothing, so ``--read`` never
    enters the hard-presence list even though ``expected_argv.yaml`` asserts it
    on every aider row — an upstream release dropping the flag would leave this
    probe green while breaking those rows. So the seed names a path that really
    exists in the image, and a missing one is a hard stop rather than a
    silently thinner contract: the derivation would still *succeed*, just
    without the option it exists to cover.
    """
    seed = SKILL_DIR / "multi_llm.py"
    if not seed.is_file():
        raise SystemExit(
            f"cannot seed the derivation: {seed} does not exist, so a "
            f"prompt-conditional adapter (aider's --read) would silently "
            f"derive a thinner contract than we dispatch"
        )
    return f"PROMPT, concerning the plan at {seed}"


def adapter_contract() -> dict:
    """The options and subcommand each adapter actually emits, per profile.

    Derived from the product's own ``build_command`` / ``build_agent_command``
    rather than hand-listed, so the contract cannot drift from what we dispatch
    — plus, per row, the flags the **rig itself** injects (``redirect.yaml``'s
    ``argv_inject``), which no adapter emits and which nothing else would
    probe. muse is the whole of that set today: ``--base-url`` is its only
    redirect mechanism (no env var, no config key), so the flag every muse
    container row depends on lives outside shipping code and would otherwise
    be the one unprobed link in the row.
    """
    sys.path.insert(0, str(SKILL_DIR))

    import yaml

    from utils.provider_registry import get_provider, load_base_config
    from utils.providers.base import (
        PROFILE_AGENT_READONLY,
        PROFILE_AGENT_WRITE,
        PROFILE_GENERATE,
    )

    rows = {
        row["cli"]: row
        for row in (yaml.safe_load(
            (RIG_DIR / "redirect.yaml").read_text(encoding="utf-8")) or {})["clis"]
        if row.get("lane") == "provider"
    }
    config = load_base_config()
    prompt = derivation_prompt()
    out: dict[str, dict] = {}
    for name, row in sorted(rows.items()):
        provider = get_provider(name)
        model = row.get("model_arg") or config["providers"][name]["models"][0]
        options: set[str] = set()
        subcommands: set[str] = set()
        profiles: list[str] = []
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
            try:
                argv = (provider.build_command(prompt, str(model))
                        if profile == PROFILE_GENERATE
                        else provider.build_agent_command(prompt, str(model), profile))
            except Exception:  # noqa: BLE001 - an undeclared profile simply raises
                continue
            profiles.append(profile)
            options.update(token for token in argv if token.startswith("-")
                           and token != "--")
            # The leading non-flag token is the subcommand, where there is one.
            for token in argv[1:]:
                if token.startswith("-"):
                    break
                if token in ("exec", "run"):
                    subcommands.add(token)
                break
        # The rig's own leg of the argv, recorded separately so a reader can
        # tell a flag the product emits from one only the wrapper adds, and
        # merged into `options` because presence is the same question for
        # both: if upstream drops it, the row breaks either way.
        injected = sorted({
            token
            for token in ((row.get("argv_inject") or {}).get("args") or [])
            if token.startswith("-") and token != "--"
        })
        options.update(injected)
        out[name] = {
            "binary": row.get("binary", name),
            "profiles": profiles,
            "options": sorted(options),
            "rig_options": injected,
            "subcommands": sorted(subcommands),
            "help_invocation": row.get("help_invocation", f"{name} --help"),
        }
    return out


def run_help(binary: str, invocation: str) -> tuple[str, int]:
    """Run the recorded help invocation against the REAL binary, offline."""
    parts = invocation.split()
    argv = [str(REAL_DIR / binary), *parts[1:]]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<{exc}>", 127
    return (completed.stdout or "") + (completed.stderr or ""), completed.returncode


def version_of(binary: str) -> str:
    try:
        completed = subprocess.run([str(REAL_DIR / binary), "--version"],
                                   capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return text.splitlines()[0] if text else "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/work/out/cli-surface.json")
    args = parser.parse_args(argv)

    contract = adapter_contract()
    report = {}
    for cli, entry in contract.items():
        binary = entry["binary"]
        help_text, code = run_help(binary, entry["help_invocation"])
        if code != 0 and not help_text.strip():
            report[cli] = dict(entry, probe="none",
                               reason=f"help invocation exited {code} with no output",
                               version=version_of(binary))
            continue
        missing = missing_options(entry["options"], help_text)
        report[cli] = dict(
            entry,
            probe="ok",
            version=version_of(binary),
            help_sha256=hashlib.sha256(help_text.encode("utf-8", "replace")).hexdigest(),
            help_bytes=len(help_text),
            missing_options=missing,
        )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True),
                              encoding="utf-8")
    for cli, entry in sorted(report.items()):
        flag = "MISSING " + ",".join(entry["missing_options"]) \
            if entry.get("missing_options") else "ok"
        print(f"{cli:12} {entry.get('probe'):5} {entry.get('version','?')[:28]:28} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
